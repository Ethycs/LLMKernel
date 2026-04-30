"""``python -m llm_kernel serve`` entry point (PLAN-S5.0.3d).

TCP transport + bearer-token auth + handshake envelope, behind the same
envelope contract as PTY mode (RFC-006). The ``serve`` subcommand
exposes the kernel to external drivers (the headless ``llmnb`` CLI,
future Rust/Go orchestrators, remote operators) without compromising
the trusted-local-PTY default.

Security model
--------------

- Default bind: ``127.0.0.1`` only (loopback). Operator MUST explicitly
  set ``0.0.0.0`` to expose externally; ``llmnb serve --help`` documents
  this as a trusted-network model.
- Token comparison: constant-time (``hmac.compare_digest``).
- Token NEVER on argv (would leak via ``ps``). Pass via env-var name only.
- Token absent or mismatched on handshake -> close with ``auth_failed``
  envelope. The kernel does not retry; one chance per connection.

V1 design choices
-----------------

- **Threading, not asyncio.** The serve loop runs as a blocking accept
  on a dedicated thread; per-connection work runs on a second thread.
  V1 doesn't need async sophistication; the simpler model is auditable
  for the security-sensitive boundary it sits on.
- **One connection at a time.** A second simultaneous client receives a
  ``kernel_busy`` handshake response and the connection closes.
  Multi-client is V2+ (PLAN-S5.0.3 §10 risk #4).
- **Kernel survives disconnect.** When a client disconnects, the kernel
  keeps running and accepts a fresh connection. SIGINT triggers
  graceful shutdown.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import signal
import socket
import sys
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

from .wire import WIRE_MAJOR, WIRE_VERSION

logger: logging.Logger = logging.getLogger("llm_kernel.serve_mode")


#: Kernel package version emitted in handshake responses.
KERNEL_VERSION: str = "1.0.0"

#: Capabilities the V1 kernel advertises in handshake responses.
KERNEL_CAPABILITIES: List[str] = [
    "family_a", "family_b", "family_c", "family_f", "family_g",
]


# ---------------------------------------------------------------------------
# Argv parsing (no token on argv -- only token-env var name).
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm_kernel serve",
        description=(
            "Serve a kernel over TCP with bearer-token auth (PLAN-S5.0.3d). "
            "Token is read from the environment variable named via "
            "--auth-token-env (default LLMNB_AUTH_TOKEN); never accepted on "
            "argv (would leak via `ps`). Default bind is 127.0.0.1; setting "
            "0.0.0.0 requires explicit operator action and is intended for "
            "trusted networks only -- there is no mTLS in V1 (PLAN-S5.0.3 "
            "§10 risk #3)."
        ),
    )
    parser.add_argument(
        "--transport",
        choices=["tcp"],
        default="tcp",
        help="Transport mode (only 'tcp' in V1; 'unix' is queued for V2+).",
    )
    parser.add_argument(
        "--bind",
        default="127.0.0.1:7474",
        help=(
            "Bind address HOST:PORT (default 127.0.0.1:7474). "
            "Use 0.0.0.0 to expose externally; that is a trusted-network "
            "decision the operator MUST make explicitly."
        ),
    )
    parser.add_argument(
        "--auth-token-env",
        default="LLMNB_AUTH_TOKEN",
        help=(
            "Name of the environment variable holding the auth token "
            "(default LLMNB_AUTH_TOKEN). Token is read from os.environ; "
            "argv is NEVER an acceptable carrier (leaks via `ps`)."
        ),
    )
    parser.add_argument(
        "--proxy",
        choices=["litellm", "passthrough", "none"],
        default="none",
        help=(
            "Optional proxy to start alongside the kernel. "
            "'none' (default) skips the proxy and is appropriate for "
            "drivers that supply their own. 'litellm' starts the LiteLLM "
            "proxy (requires ANTHROPIC_API_KEY); 'passthrough' starts the "
            "transparent Anthropic passthrough."
        ),
    )
    return parser


def _parse_bind(raw: str) -> tuple[str, int]:
    """Parse HOST:PORT into (host, port). Raises ValueError on bad input."""
    if ":" not in raw:
        raise ValueError(f"--bind must be HOST:PORT (got {raw!r})")
    host, port_str = raw.rsplit(":", 1)
    if not host:
        raise ValueError(f"--bind host is empty (got {raw!r})")
    try:
        port = int(port_str)
    except ValueError as exc:
        raise ValueError(f"--bind port is not an integer (got {raw!r})") from exc
    if not (0 <= port <= 65535):
        raise ValueError(f"--bind port out of range (got {port})")
    return host, port


# ---------------------------------------------------------------------------
# Handshake helpers (constant-time auth + version-skew rules).
# ---------------------------------------------------------------------------


def _send_frame(sock: socket.socket, envelope: Dict[str, Any]) -> None:
    """Send a single JSON envelope as a newline-delimited frame."""
    data = (json.dumps(envelope, default=str) + "\n").encode("utf-8")
    try:
        sock.sendall(data)
    except OSError:
        # Best-effort: caller will close the socket; we don't double-log.
        pass


def _recv_frame(
    sock: socket.socket, buf: bytearray, *, timeout: float,
) -> Optional[Dict[str, Any]]:
    """Read a single newline-delimited JSON frame, blocking up to ``timeout``.

    Returns ``None`` on timeout, EOF, or parse failure.
    """
    sock.settimeout(timeout)
    while True:
        nl = buf.find(b"\n")
        if nl >= 0:
            line = bytes(buf[:nl])
            del buf[: nl + 1]
            if not line.strip():
                continue
            try:
                return json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
        try:
            chunk = sock.recv(4096)
        except (socket.timeout, TimeoutError):
            return None
        except OSError:
            return None
        if not chunk:
            return None
        buf.extend(chunk)


def _handshake_error(
    sock: socket.socket, error_code: str, *, extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Send a handshake response carrying an error code and close the socket."""
    payload: Dict[str, Any] = {
        "wire_version": WIRE_VERSION,
        "kernel_version": KERNEL_VERSION,
        "error": error_code,
    }
    if extra:
        payload.update(extra)
    _send_frame(sock, {"type": "kernel.handshake", "payload": payload})


def _handshake_response(
    session_id: str, *, warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a successful handshake response envelope."""
    payload: Dict[str, Any] = {
        "kernel_version": KERNEL_VERSION,
        "wire_version": WIRE_VERSION,
        "session_id": session_id,
        "accepted_capabilities": list(KERNEL_CAPABILITIES),
    }
    if warnings:
        payload["warnings"] = warnings
    return {"type": "kernel.handshake", "payload": payload}


def _validate_handshake(
    envelope: Dict[str, Any], *, expected_token: str,
) -> tuple[Optional[str], List[str]]:
    """Validate a handshake request. Return (error_code, warnings).

    On success, ``error_code is None``. On failure, ``error_code`` is one of
    ``wire-failure | version_mismatch_major | auth_failed``.
    """
    if not isinstance(envelope, dict):
        return ("wire-failure", [])
    if envelope.get("type") != "kernel.handshake":
        return ("wire-failure", [])
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        return ("wire-failure", [])

    # Wire-version check (major must match; minor mismatch is a warning).
    client_wire = payload.get("wire_version", "")
    if not isinstance(client_wire, str) or not client_wire:
        return ("wire-failure", [])
    try:
        client_major = int(client_wire.split(".", 1)[0])
    except (ValueError, IndexError):
        return ("wire-failure", [])
    if client_major != WIRE_MAJOR:
        return ("version_mismatch_major", [])
    warnings: List[str] = []
    if client_wire != WIRE_VERSION:
        warnings.append("minor_version_skew")

    # Auth check (TCP only -- this serve mode is TCP-exclusive).
    auth = payload.get("auth")
    if not isinstance(auth, dict):
        return ("auth_failed", warnings)
    if auth.get("scheme") != "bearer":
        return ("auth_failed", warnings)
    presented = auth.get("token")
    if not isinstance(presented, str) or not presented:
        return ("auth_failed", warnings)
    # Constant-time compare. Encode to bytes to avoid surrogate issues.
    if not hmac.compare_digest(
        presented.encode("utf-8"), expected_token.encode("utf-8"),
    ):
        return ("auth_failed", warnings)

    return (None, warnings)


# ---------------------------------------------------------------------------
# Kernel scaffolding for serve mode.
# ---------------------------------------------------------------------------


def _boot_kernel_for_serve(proxy_mode: str) -> Dict[str, Any]:
    """Boot the minimum kernel surface needed to dispatch envelopes.

    Returns a dict carrying the dispatcher, tracker, server, etc. The
    caller owns teardown. Mirrors the boot scaffold in
    ``__main__._run_agent_supervisor_smoke`` and ``llm_client.boot``;
    factored out so the serve loop stays readable.
    """
    from unittest.mock import MagicMock

    from .custom_messages import CustomMessageDispatcher
    from .run_tracker import RunTracker

    # MagicMock kernel shell -- the dispatcher needs ``.session``,
    # ``.iopub_socket``, ``.shell.comm_manager``, ``._parent_header``.
    # Same pattern as the smokes; isolated to this boot helper.
    kernel = MagicMock()

    class _Session:
        def send(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    class _CommMgr:
        def register_target(self, *_args: Any, **_kwargs: Any) -> None: ...
        def unregister_target(self, *_args: Any, **_kwargs: Any) -> None: ...

    kernel.session = _Session()
    kernel.iopub_socket = MagicMock()
    kernel.shell.comm_manager = _CommMgr()
    kernel._parent_header = {}

    session_id = str(uuid.uuid4())
    dispatcher = CustomMessageDispatcher(kernel)
    tracker = RunTracker(
        trace_id=session_id, sink=dispatcher,
        agent_id="serve", zone_id="serve",
    )

    # Optional proxy lifecycle. Most external-driver use cases supply
    # their own proxy out-of-band; default 'none' keeps the serve
    # surface minimal.
    server: Any = None
    if proxy_mode == "litellm":
        from . import litellm_proxy as _proxy_mod
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        server = _proxy_mod.LiteLLMProxyServer(
            api_key=api_key, host="127.0.0.1", port=0,
        )
        server.start()
    elif proxy_mode == "passthrough":
        from . import anthropic_passthrough as _pt_mod
        server = _pt_mod.AnthropicPassthroughServer(
            run_tracker=tracker, host="127.0.0.1", port=0,
        )
        server.start()

    return {
        "kernel": kernel,
        "dispatcher": dispatcher,
        "tracker": tracker,
        "server": server,
        "session_id": session_id,
    }


def _teardown_serve_kernel(state: Dict[str, Any]) -> None:
    """Stop subsystems started by :func:`_boot_kernel_for_serve`."""
    server = state.get("server")
    if server is not None:
        try:
            server.stop()
        except Exception:  # pragma: no cover - defensive
            logger.exception("serve: proxy stop raised")


# ---------------------------------------------------------------------------
# Per-connection routing.
#
# After handshake succeeds, the client sends Family A/B/C/F/G envelopes
# and the kernel routes them via the dispatcher. Outbound frames (Family
# F snapshots, Family A spans, lifecycle) flow back over the same socket.
# We register a single handler that captures every dispatcher emit and
# forwards it on the socket, mirroring the comm-write path in pty-mode.
# ---------------------------------------------------------------------------


def _serve_connection(
    sock: socket.socket,
    *,
    expected_token: str,
    boot_state: Dict[str, Any],
    stop_event: threading.Event,
) -> None:
    """Handle exactly one connection, including handshake and dispatch.

    Closes ``sock`` on return.
    """
    dispatcher = boot_state["dispatcher"]
    out_lock = threading.Lock()

    def _send_safely(envelope: Dict[str, Any]) -> None:
        with out_lock:
            _send_frame(sock, envelope)

    buf = bytearray()
    try:
        # Handshake request -- generous timeout; client may be slow to
        # connect after binding. 30s matches the driver-side default.
        request = _recv_frame(sock, buf, timeout=30.0)
        if request is None:
            _handshake_error(sock, "wire-failure")
            return

        error_code, warnings = _validate_handshake(
            request, expected_token=expected_token,
        )
        if error_code is not None:
            _handshake_error(sock, error_code)
            return

        # Allocate a session id per accepted connection (forward-compat
        # with V2+ multi-client; V1 always issues a fresh one).
        session_id = str(uuid.uuid4())
        _send_safely(_handshake_response(session_id, warnings=warnings or None))

        # Mirror the dispatcher's emits onto the socket. ``emit`` is the
        # kernel-side fan-out for both run-lifecycle (Family A) and
        # notebook-metadata (Family F); registering a global handler is
        # the simplest way to forward without re-implementing routing.
        forwarded: List[Callable[[], None]] = []

        def _forward(envelope: Dict[str, Any]) -> None:
            try:
                _send_safely(dict(envelope))
            except Exception:  # pragma: no cover - defensive
                logger.exception("serve: forward raised")

        # The dispatcher's register_handler API is for inbound handlers;
        # outbound is via dispatcher.emit. We monkey-patch emit to also
        # forward, restoring on disconnect. This is the simplest way to
        # tap the existing fan-out without reaching into private state.
        original_emit = dispatcher.emit

        def _emit_and_forward(envelope: Dict[str, Any]) -> None:
            try:
                original_emit(envelope)
            finally:
                _forward(envelope)

        dispatcher.emit = _emit_and_forward  # type: ignore[method-assign]
        forwarded.append(lambda: setattr(dispatcher, "emit", original_emit))

        # Read loop: for each inbound envelope, route via dispatcher.
        try:
            while not stop_event.is_set():
                envelope = _recv_frame(sock, buf, timeout=1.0)
                if envelope is None:
                    # Timeout (poll) or EOF. Check for stop_event first.
                    if stop_event.is_set():
                        break
                    # Disambiguate "timeout" from "EOF" via a non-blocking
                    # zero-length recv. On a still-connected socket this
                    # raises BlockingIOError; on a closed peer it either
                    # returns b'' (BSD/Linux/Windows) or raises.
                    sock.settimeout(0.0)
                    try:
                        peek = sock.recv(1, socket.MSG_PEEK)
                    except BlockingIOError:
                        continue   # no data, still connected
                    except OSError:
                        break      # closed
                    if not peek:
                        # Peer EOF; exit the dispatch loop.
                        break
                    continue

                # Dispatch via the same path as comm_msg in IPython mode:
                # build a fake comm_msg shape so _on_comm_msg validates +
                # fans out.  The dispatcher accepts the envelope shape
                # directly via the v2 thin form.
                try:
                    if hasattr(dispatcher, "_on_comm_msg"):
                        dispatcher._on_comm_msg({"content": {"data": envelope}})
                    else:
                        # Defensive: dispatcher API change.
                        logger.warning(
                            "serve: dispatcher missing _on_comm_msg; dropped",
                        )
                except Exception:  # pragma: no cover - defensive
                    logger.exception("serve: dispatch raised; continuing")
        finally:
            for restore in forwarded:
                try:
                    restore()
                except Exception:  # pragma: no cover - defensive
                    pass
    finally:
        # Half-close + drain, same discipline as _busy_close: avoid the
        # WSAECONNABORTED race where a hard close right after a send
        # discards in-flight bytes from the peer's read view.
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        try:
            sock.settimeout(1.0)
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Top-level serve loop.
# ---------------------------------------------------------------------------


def _busy_close(sock: socket.socket) -> None:
    """Reject a second simultaneous client with a kernel_busy handshake.

    Per PLAN-S5.0.3 §5.2 + the wire-handshake atom: one connection at a
    time in V1. The second client gets a single ``kernel_busy`` envelope
    and is closed immediately.

    Closing discipline: ``shutdown(SHUT_WR)`` to flush our write side and
    signal EOF on the client's read side, THEN drain any client-side
    bytes (which we ignore), THEN ``close()``. Skipping the half-close
    on Windows triggers WSAECONNABORTED (10053) on the peer's recv when
    the close races with the just-sent frame -- the kernel_busy bytes
    sit in the kernel's send buffer when ``close()`` issues an RST,
    losing the frame the driver was supposed to read.
    """
    try:
        _handshake_error(sock, "kernel_busy")
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        sock.shutdown(socket.SHUT_WR)
    except OSError:
        pass
    # Drain briefly so the client's buffered handshake request doesn't
    # cause an RST on close. We don't care about the bytes.
    try:
        sock.settimeout(1.0)
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


def main(argv: Optional[List[str]] = None) -> int:
    """Run the serve subcommand. Returns a process exit code.

    Exit codes:
        0  clean shutdown (SIGINT / SIGTERM).
        2  argv / token-env error (token missing, bad --bind).
        3  socket bind / listen error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Token MUST come from env; argv would leak via `ps`.
    token = os.environ.get(args.auth_token_env, "")
    if not token:
        sys.stderr.write(
            f"llm_kernel serve: env var {args.auth_token_env!r} is unset or "
            "empty. Generate a token with `llmnb auth init` (or set the var "
            "directly), then re-run.\n"
        )
        sys.stderr.flush()
        return 2

    try:
        host, port = _parse_bind(args.bind)
    except ValueError as exc:
        sys.stderr.write(f"llm_kernel serve: {exc}\n")
        sys.stderr.flush()
        return 2

    if host == "0.0.0.0":
        sys.stderr.write(
            "llm_kernel serve: WARNING binding to 0.0.0.0 exposes the "
            "kernel on every reachable interface. There is no mTLS in "
            "V1 -- this mode is for trusted networks only (PLAN-S5.0.3 "
            "§10 risk #3).\n"
        )
        sys.stderr.flush()

    # Bind + listen.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind((host, port))
        listener.listen(8)
    except OSError as exc:
        sys.stderr.write(
            f"llm_kernel serve: failed to bind {host}:{port}: {exc}\n"
        )
        sys.stderr.flush()
        try:
            listener.close()
        except OSError:
            pass
        return 3

    actual_host, actual_port = listener.getsockname()[:2]
    sys.stderr.write(
        f"llm_kernel serve v{KERNEL_VERSION}; "
        f"listening on {actual_host}:{actual_port}; "
        f"wire_version={WIRE_VERSION}\n"
    )
    sys.stderr.flush()

    # Boot the kernel internals once; reused across connections so the
    # kernel "keeps running" when a client disconnects.
    boot_state = _boot_kernel_for_serve(args.proxy)

    stop_event = threading.Event()
    busy_lock = threading.Lock()
    active_thread: Optional[threading.Thread] = None

    def _on_signal(_signum: int, _frame: Any) -> None:
        stop_event.set()
        try:
            listener.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    if hasattr(signal, "SIGINT"):
        try:
            signal.signal(signal.SIGINT, _on_signal)
        except ValueError:
            pass
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _on_signal)
        except ValueError:
            pass

    # Accept loop. Threading + a busy_lock semaphore is the simplest way
    # to enforce the V1 single-connection invariant. asyncio would buy us
    # nothing here (no fan-out, no high concurrency).
    try:
        listener.settimeout(1.0)
        while not stop_event.is_set():
            try:
                conn, _addr = listener.accept()
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                break

            # If a thread is already serving, reject the new client
            # immediately. We use a short Lock acquisition: try to take
            # it; if we can, install the new thread; else send busy.
            with busy_lock:
                if active_thread is not None and active_thread.is_alive():
                    _busy_close(conn)
                    continue

                def _runner(client_sock: socket.socket = conn) -> None:
                    try:
                        _serve_connection(
                            client_sock,
                            expected_token=token,
                            boot_state=boot_state,
                            stop_event=stop_event,
                        )
                    except Exception:  # pragma: no cover - defensive
                        logger.exception("serve: connection thread crashed")

                t = threading.Thread(
                    target=_runner, name="llmnb-serve-conn", daemon=True,
                )
                active_thread = t
                t.start()
    finally:
        stop_event.set()
        try:
            listener.close()
        except OSError:
            pass
        if active_thread is not None and active_thread.is_alive():
            active_thread.join(timeout=5.0)
        _teardown_serve_kernel(boot_state)

    return 0


__all__ = ["main", "_validate_handshake", "_parse_bind", "KERNEL_CAPABILITIES"]

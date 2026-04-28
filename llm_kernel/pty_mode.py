"""``python -m llm_kernel pty-mode`` entry point (RFC-008 §3 + §4).

The kernel-side half of the PTY+socket two-channel transport. The
extension allocates a UDS / pipe / TCP socket, listens, then spawns this
entry point with ``LLMKERNEL_IPC_SOCKET=<address>`` and (when running
under a real PTY) ``LLMKERNEL_PTY_MODE=1``. We:

1. Read the env vars; fail fast on missing socket address.
2. (POSIX, only when stdin is a TTY) put stdin/stdout into termios raw
   mode so the slave-side line discipline doesn't buffer or interpret
   bytes. On Windows or when stdin is a pipe, this is a no-op.
3. Install SIGINT (interrupt running operation) and SIGTERM (clean
   shutdown with final ``notebook.metadata`` snapshot) handlers.
4. Bring up the kernel subsystems via the existing 5-tuple
   :func:`attach_kernel_subsystems` against a synthetic kernel shell.
5. Construct a :class:`SocketWriter`, connect to the address.
6. Install the :class:`OtlpDataPlaneHandler` on the root logger plus
   every ``llm_kernel.*`` child; remove any pre-existing stdout-bound
   :class:`StreamHandler`.
7. Emit the ready handshake LogRecord (RFC-008 §4 "Ready handshake")
   with the required ``llmnb.kernel.*`` attributes.
8. Write the boot banner ``LLMKernel pty-mode v1.0.0; socket=<address>``
   to PTY stderr (RFC-008 §3).
9. Wire :class:`MetadataWriter` snapshots to flow through the socket
   writer (the in-process IOPub path doesn't apply here -- there's no
   IPython kernel session in pty-mode).
10. Read newline-delimited frames from the socket (extension -> kernel
    RFC-006 envelopes); dispatch each to the existing dispatcher.
11. Run until SIGTERM or socket EOF; emit a final
    ``notebook.metadata`` snapshot, close, exit 0.

The ``pty-mode-smoke`` arm in :mod:`llm_kernel.__main__` exercises this
end-to-end without node-pty: a server in the same process opens a UDS
pair, spawns the kernel as a subprocess, waits for the ready handshake,
sends a ``cell_execute`` envelope, and asserts a Family A run.start
arrives back.

This module is ALSO the only place :envvar:`LLMKERNEL_IPC_SOCKET` and
:envvar:`LLMKERNEL_PTY_MODE` are read (RFC-008 §3 boundary discipline).
"""

from __future__ import annotations

import logging
import os
import platform
import signal
import socket
import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ._otlp_log_handler import OtlpDataPlaneHandler
from .socket_writer import SocketWriter

logger: logging.Logger = logging.getLogger("llm_kernel.pty_mode")

#: Kernel package version emitted in the ready handshake. RFC-008 §4
#: requires this; we hardcode for V1 since reading ``pyproject.toml`` at
#: runtime adds a tomllib dependency for no benefit.
KERNEL_VERSION: str = "1.0.0"

#: RFC implementation versions emitted in the ready handshake's
#: ``llmnb.kernel.rfc_NNN_version`` attributes (RFC-008 §4 + §"Drift on
#: connect"). Track these alongside the RFC documents.
RFC_VERSIONS: Dict[str, str] = {
    "rfc_001_version": "1.0.0",
    "rfc_002_version": "1.0.0",
    "rfc_003_version": "1.0.0",
    "rfc_004_version": "1.0.0",
    "rfc_005_version": "1.0.0",
    "rfc_006_version": "2.0.0",
    "rfc_007_version": "1.0.0",
    "rfc_008_version": "1.0.0",
}

#: Env vars read ONLY in this module per RFC-008 §3 boundary discipline.
ENV_IPC_SOCKET: str = "LLMKERNEL_IPC_SOCKET"
ENV_PTY_MODE: str = "LLMKERNEL_PTY_MODE"
ENV_SESSION_ID: str = "LLMKERNEL_SESSION_ID"


# ---------------------------------------------------------------------------
# Synthetic kernel shell for subsystem attach
# ---------------------------------------------------------------------------


class _PtyKernelSession:
    """Stand-in for ``ipykernel.session.Session`` in pty-mode.

    The kernel's :class:`CustomMessageDispatcher` writes Family A spans
    via ``kernel.session.send`` to ``kernel.iopub_socket``. In pty-mode
    there is no IPython kernel, so we redirect those calls onto the
    data-plane :class:`SocketWriter` -- the run lifecycle becomes
    ordinary OTLP frames on the socket per RFC-008 §6 dispatch.

    Only the ``send`` shape the dispatcher uses is implemented; any other
    attribute access raises so we discover surface gaps loudly.
    """

    def __init__(self, writer: SocketWriter) -> None:
        self._writer = writer

    def send(self, _socket: Any, msg_type: str, **kwargs: Any) -> None:
        """Forward an IOPub send onto the data-plane socket.

        We unwrap the ``application/vnd.rts.run+json`` payload (RFC-006
        §1) and emit the OTLP span itself. The ``msg_type``
        (``display_data`` / ``update_display_data``) is irrelevant on
        the wire -- the receiver discriminates by ``traceId+spanId``
        per RFC-008 §6.
        """
        content = kwargs.get("content") or {}
        data = content.get("data") or {}
        # RFC-006 §1: only ``application/vnd.rts.run+json`` rides
        # display_data in v2; we ignore other MIMEs (defensive).
        for mime, payload in data.items():
            if mime == "application/vnd.rts.run+json" and isinstance(payload, dict):
                self._writer.write_frame(payload)


class _PtyCommManager:
    """Stand-in for ``ipykernel.comm.CommManager`` in pty-mode.

    The dispatcher registers a Comm target and, on attach, expects a
    Comm object with ``on_msg`` and ``send``. In pty-mode we synthesize
    a "self-attach": when :meth:`register_target` is invoked, we
    immediately fire the open callback with a :class:`_PtyComm` whose
    ``send`` rides the data-plane socket. The dispatcher's pre-attach
    buffer drains as a result, and outbound Family B-F envelopes flow
    through ``comm.send`` -> :class:`SocketWriter`.

    Inbound traffic (extension -> kernel) is delivered by
    :meth:`_PtyComm.deliver`, which the pty-mode read loop calls per
    parsed frame.
    """

    def __init__(self, writer: SocketWriter) -> None:
        self._writer = writer
        self._comms: Dict[str, "_PtyComm"] = {}
        self._lock = threading.Lock()

    def register_target(
        self, name: str, callback: Callable[..., None],
    ) -> None:
        comm = _PtyComm(name=name, writer=self._writer)
        with self._lock:
            self._comms[name] = comm
        # Fire the open callback synchronously; the dispatcher caches
        # ``comm`` and starts flushing buffered envelopes immediately.
        callback(comm, {"content": {"data": {}}})

    def unregister_target(
        self, name: str, _callback: Callable[..., None],
    ) -> None:
        with self._lock:
            self._comms.pop(name, None)

    def deliver(self, name: str, msg: Dict[str, Any]) -> None:
        """Forward an inbound frame to the registered Comm's ``on_msg``."""
        with self._lock:
            comm = self._comms.get(name)
        if comm is None:
            logger.warning(
                "pty_mode: inbound frame for unregistered target %r dropped",
                name,
            )
            return
        comm.deliver(msg)


class _PtyComm:
    """One Comm-shaped object the dispatcher uses to send/receive."""

    def __init__(self, name: str, writer: SocketWriter) -> None:
        self.name = name
        self._writer = writer
        self._on_msg: Optional[Callable[[Dict[str, Any]], None]] = None

    def on_msg(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._on_msg = callback

    def send(self, data: Dict[str, Any]) -> None:
        """Send a v2 thin envelope onto the data-plane socket.

        ``data`` is the RFC-006 §3 thin form ``{type, payload,
        correlation_id?}`` -- the receiver dispatches by
        ``type+payload`` per RFC-008 §6.
        """
        self._writer.write_frame(data)

    def close(self) -> None:
        # No teardown work in pty-mode; the dispatcher owns the lifecycle.
        self._on_msg = None

    def deliver(self, msg: Dict[str, Any]) -> None:
        """Synchronously invoke the registered ``on_msg`` callback."""
        cb = self._on_msg
        if cb is None:
            logger.warning("pty_mode: comm %r has no on_msg; frame dropped", self.name)
            return
        try:
            cb(msg)
        except Exception:  # pragma: no cover - defensive
            logger.exception("pty_mode: comm %r on_msg raised", self.name)


class _PtyKernel:
    """Minimal kernel-shaped object that satisfies the dispatcher.

    The dispatcher reads ``self.session``, ``self.iopub_socket``,
    ``self.shell.comm_manager``, and ``self._parent_header``. We set up
    each one to ride the data-plane socket per RFC-008 §6.
    """

    def __init__(self, writer: SocketWriter) -> None:
        self.session = _PtyKernelSession(writer)
        # ``iopub_socket`` is opaque to the dispatcher (it just hands the
        # ZMQ socket to ``session.send``); a sentinel string is enough.
        self.iopub_socket = "<pty-mode-iopub-placeholder>"
        self.shell = _PtyShell(writer)
        self._parent_header: Dict[str, Any] = {}


class _PtyShell:
    def __init__(self, writer: SocketWriter) -> None:
        self.comm_manager = _PtyCommManager(writer)


# ---------------------------------------------------------------------------
# Termios raw mode + signal handlers
# ---------------------------------------------------------------------------


def _set_raw_mode_if_tty() -> Optional[Callable[[], None]]:
    """Put stdin into termios raw mode when running under a real PTY.

    Returns a restorer callable (or ``None`` when no-op) so the main
    loop can hand control back on shutdown. On Windows or when stdin
    isn't a TTY (e.g., the pty-mode-smoke harness uses pipes), this is
    a no-op -- per RFC-008 §3 the termios setup is optional hardening.
    """
    if platform.system() == "Windows":
        # Windows lacks termios entirely; the named-pipe transport
        # (TODO(I-T-W)) handles its own line-discipline concerns.
        return None
    try:
        import termios
        import tty
    except ImportError:  # pragma: no cover - non-POSIX
        return None
    try:
        fd = sys.stdin.fileno()
    except (AttributeError, ValueError, OSError):
        return None
    try:
        if not os.isatty(fd):
            return None
    except OSError:
        return None
    try:
        original = termios.tcgetattr(fd)
        tty.setraw(fd)
    except (termios.error, OSError):  # pragma: no cover - defensive
        logger.exception("pty_mode: termios setup failed; continuing")
        return None

    def _restore() -> None:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, original)
        except (termios.error, OSError):  # pragma: no cover - defensive
            logger.exception("pty_mode: termios restore failed")

    return _restore


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Run the pty-mode kernel; return a process exit code.

    Reads ``LLMKERNEL_IPC_SOCKET`` from the env. Fails fast (returns 2
    after writing to PTY stderr) when missing -- the extension is
    expected to set this before spawning per RFC-008 §4 step 1.
    """
    from . import _diagnostics
    _diagnostics.mark("pty_mode_main_entry")
    address = os.environ.get(ENV_IPC_SOCKET, "")
    pty_flag = os.environ.get(ENV_PTY_MODE) == "1"
    session_id = os.environ.get(ENV_SESSION_ID) or str(uuid.uuid4())
    _diagnostics.mark("pty_mode_env_read", socket=address, session_id=session_id, pty_flag=pty_flag)
    if not address:
        sys.stderr.write(
            f"LLMKernel pty-mode: {ENV_IPC_SOCKET} env var is required\n"
        )
        sys.stderr.flush()
        _diagnostics.mark("pty_mode_exit_no_socket", rc=2)
        return 2

    # Termios raw mode -- only meaningful under a real PTY. The smoke
    # spawns under pipes, so this is typically a no-op.
    termios_restore = _set_raw_mode_if_tty() if pty_flag else None

    writer = SocketWriter()
    try:
        writer.connect(address)
    except OSError as exc:
        sys.stderr.write(
            f"LLMKernel pty-mode: failed to connect to socket {address!r}: {exc}\n"
        )
        sys.stderr.flush()
        if termios_restore is not None:
            termios_restore()
        _diagnostics.mark("pty_mode_exit_socket_connect_failed", error=str(exc), rc=3)
        return 3
    _diagnostics.mark("pty_mode_socket_connected")

    # Banner BEFORE installing the OTLP handler so it actually reaches
    # PTY stderr (RFC-008 §3 "Boot output"). Once the handler is up,
    # nothing else writes to stderr unless the OTLP pipeline crashes.
    sys.stderr.write(
        f"LLMKernel pty-mode v{KERNEL_VERSION}; socket={address}\n"
    )
    sys.stderr.flush()

    # Ready handshake FIRST -- RFC-008 §4 makes this the first frame on
    # the data plane. Subsystem initialization below logs through the
    # OTLP handler; logs MUST come after the ready record so the
    # extension can recognize the kernel before processing anything else.
    _emit_ready_handshake(writer, session_id)
    _diagnostics.mark("pty_mode_ready_emitted")

    # BSP-001: proxy lifecycle. Either trust an externally-provided URL
    # (operator already started a proxy and points us at it) or start
    # one ourselves and publish its URL via the env var that
    # ``attach_agent_supervisor`` reads downstream. Failures here are
    # K11/K12 and we exit BEFORE attaching subsystems.
    proxy_server = _start_owned_proxy_or_none()
    if proxy_server is _PROXY_START_FAILED:
        # K12 already marked + stderr already written by helper.
        if termios_restore is not None:
            termios_restore()
        return 12
    if proxy_server is _PROXY_CONFIG_REJECTED:
        # K11 already marked + stderr already written by helper.
        if termios_restore is not None:
            termios_restore()
        return 11

    # Install OTLP handler on the root + ``llm_kernel`` namespace.
    # Removing existing stdout-bound StreamHandlers per RFC-008 §7
    # ("once the handler is up, no Python text reaches the PTY").
    handler = OtlpDataPlaneHandler(
        socket_writer=writer,
        extra_attributes={"llmnb.kernel.session_id": session_id},
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler_restore = _install_handler(handler)

    # Bring up subsystems against the synthetic kernel shell.
    kernel = _PtyKernel(writer)
    # Inline import: ``_kernel_hooks`` pulls a chunk of dependencies the
    # tests don't always need at module import time.
    from ._kernel_hooks import attach_kernel_subsystems
    dispatcher, tracker, bridge, supervisor, metadata_writer = (
        attach_kernel_subsystems(kernel)
    )

    # ``MetadataWriter.snapshot`` already calls ``dispatcher.emit``,
    # which routes through our synthetic Comm to the SocketWriter --
    # no extra wiring needed. The hook below is for shutdown-time final
    # snapshot triggered explicitly (RFC-008 §4 step 6).
    shutdown_event = threading.Event()
    interrupt_event = threading.Event()

    # Wire K-CM dispatcher's collaborators (added in the V1 mega-round).
    # The dispatcher's hydrate / shutdown / B / C handlers consult these
    # attributes; missing wiring means the handlers fall back to logging
    # and EOF-driven shutdown rather than the spec'd paths. See RFC-006
    # §7.1, §"Family B", §"Family C", §8 v2.0.2 hydrate.
    if hasattr(dispatcher, "set_shutdown_event"):
        dispatcher.set_shutdown_event(shutdown_event)
    if hasattr(dispatcher, "set_metadata_writer"):
        dispatcher.set_metadata_writer(metadata_writer)
    if hasattr(dispatcher, "set_agent_supervisor"):
        dispatcher.set_agent_supervisor(supervisor)
    if hasattr(dispatcher, "set_drift_detector"):
        from .drift_detector import DriftDetector
        dispatcher.set_drift_detector(DriftDetector())
    # Register the dispatcher's Comm target with the synthetic kernel so
    # inbound RFC-006 v2 envelopes (operator.action, layout.edit,
    # agent_graph.query, notebook.metadata, kernel.shutdown_request)
    # actually reach their handlers. Without this start() call, the
    # `kernel.shell.comm_manager.deliver(...)` in `_dispatch_inbound_line`
    # finds no target and silently drops every inbound envelope.
    dispatcher.start()
    _diagnostics.mark("pty_mode_dispatcher_started")

    if hasattr(dispatcher, "set_current_volatile_provider"):
        # Provide a callable that returns the kernel's current volatile
        # state for DriftDetector.compare(persisted_volatile, current).
        def _current_volatile() -> Dict[str, Any]:
            return {
                "kernel": {
                    "rfc_001_version": "1.0.0",
                    "rfc_002_version": "1.0.1",
                    "rfc_003_version": "2.0.2",  # RFC-006 actually
                    "rfc_005_version": "1.0.0",
                    "rfc_006_version": "2.0.2",
                    "rfc_008_version": "1.0.0",
                    "kernel_version": KERNEL_VERSION,
                },
            }
        dispatcher.set_current_volatile_provider(_current_volatile)

    def _on_sigint(_signum: int, _frame: Any) -> None:
        # Keep the handler minimal -- ``logger.info`` here would acquire
        # the SocketWriter lock from the same thread that may already
        # hold it (Python signals run on the main thread between
        # bytecodes), risking a deadlock. The interrupt event is set;
        # the read loop notices and emits the LogRecord cleanly.
        interrupt_event.set()

    def _on_sigterm(_signum: int, _frame: Any) -> None:
        shutdown_event.set()

    # ``signal.signal`` only works on the main thread; tests that run
    # ``main`` in a worker thread skip handler installation gracefully.
    if hasattr(signal, "SIGINT"):
        try:
            signal.signal(signal.SIGINT, _on_sigint)
        except ValueError:
            logger.debug("pty_mode: SIGINT handler not installable from this thread")
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _on_sigterm)
        except ValueError:
            logger.debug("pty_mode: SIGTERM handler not installable from this thread")

    # The ready handshake was emitted earlier (right after socket
    # connect, before subsystem init logs) so it is genuinely the first
    # frame on the data plane per RFC-008 §4.

    # Read loop: parse newline-delimited JSON frames and dispatch.
    try:
        _run_read_loop(writer, kernel, shutdown_event, interrupt_event)
    except Exception:  # pragma: no cover - defensive
        logger.exception("pty_mode: read loop crashed; shutting down")
    finally:
        # Final notebook.metadata snapshot per RFC-008 §4 step 6 ("Kernel
        # emits final notebook.metadata snapshot").
        try:
            metadata_writer.snapshot(trigger="shutdown")
        except Exception:  # pragma: no cover - defensive
            logger.exception("pty_mode: final metadata snapshot raised")
        try:
            metadata_writer.stop(emit_final=False)
        except Exception:  # pragma: no cover - defensive
            logger.exception("pty_mode: metadata_writer.stop raised")
        try:
            dispatcher.stop()
        except Exception:  # pragma: no cover - defensive
            logger.exception("pty_mode: dispatcher.stop raised")
        # BSP-001 §3 step 7: stop kernel-owned proxy on exit. None means
        # external lifecycle (operator owns it; not ours to stop).
        if proxy_server is not None and proxy_server not in (
            _PROXY_START_FAILED, _PROXY_CONFIG_REJECTED,
        ):
            try:
                proxy_server.stop()
                _diagnostics.mark("pty_mode_proxy_stopped")
            except Exception:  # pragma: no cover - defensive
                logger.exception("pty_mode: proxy_server.stop raised")
        writer.close()
        if termios_restore is not None:
            termios_restore()
        if handler_restore is not None:
            handler_restore()
    return 0


# ---------------------------------------------------------------------------
# Helpers (factored for testability)
# ---------------------------------------------------------------------------

#: Sentinel returned by :func:`_start_owned_proxy_or_none` when the
#: ``(auth, proxy)`` combination is illegal per BSP-001 §2 (K11).
_PROXY_CONFIG_REJECTED: object = object()
#: Sentinel returned by :func:`_start_owned_proxy_or_none` when the
#: chosen proxy server failed to start (port unavailable, mitmdump
#: missing, etc.) — K12 per BSP-001 §5.
_PROXY_START_FAILED: object = object()


def _start_owned_proxy_or_none() -> Any:
    """BSP-001: resolve and start the proxy per the lifecycle contract.

    Returns:
      * ``None`` — external lifecycle (``LLMKERNEL_LITELLM_ENDPOINT_URL``
        was set; operator owns it). Caller continues boot.
      * a server object exposing ``start() / stop() / base_url()`` —
        kernel-owned, already started. Caller stops it on exit.
      * :data:`_PROXY_CONFIG_REJECTED` — K11. Caller exits 11.
      * :data:`_PROXY_START_FAILED` — K12. Caller exits 12.

    Reads env vars per BSP-001 §2 table; sets
    ``LLMKERNEL_LITELLM_ENDPOINT_URL`` to the bound URL when starting a
    kernel-owned server so :mod:`._kernel_hooks` reads it uniformly.
    """
    from . import _diagnostics

    external_url = os.environ.get("LLMKERNEL_LITELLM_ENDPOINT_URL", "").strip()
    if external_url:
        _diagnostics.mark("pty_mode_proxy_external", url=external_url)
        return None

    # Resolve the (auth, proxy) combination per BSP-001 §2.
    auth_mode = "api_key" if os.environ.get("LLMKERNEL_USE_BARE") == "1" else "oauth"
    proxy_mode = os.environ.get("LLMKERNEL_PROXY_MODE", "passthrough").strip().lower()
    if proxy_mode not in ("passthrough", "litellm"):
        sys.stderr.write(
            f"LLMKernel pty-mode: K11 unknown LLMKERNEL_PROXY_MODE={proxy_mode!r}; "
            f"valid: passthrough | litellm\n"
        )
        sys.stderr.flush()
        _diagnostics.mark(
            "pty_mode_proxy_config_rejected",
            auth_mode=auth_mode, proxy_mode=proxy_mode,
            reason="unknown_proxy_mode",
        )
        return _PROXY_CONFIG_REJECTED
    if auth_mode == "oauth" and proxy_mode == "litellm":
        sys.stderr.write(
            "LLMKernel pty-mode: K11 (auth=oauth, proxy=litellm) is illegal; "
            "LiteLLM proxy breaks OAuth model-resolution preflight. Set "
            "LLMKERNEL_PROXY_MODE=passthrough or LLMKERNEL_USE_BARE=1.\n"
        )
        sys.stderr.flush()
        _diagnostics.mark(
            "pty_mode_proxy_config_rejected",
            auth_mode=auth_mode, proxy_mode=proxy_mode,
            reason="oauth_litellm_incompatible",
        )
        return _PROXY_CONFIG_REJECTED

    # Construct + start.
    try:
        if proxy_mode == "passthrough":
            from . import anthropic_passthrough as _pt
            server = _pt.AnthropicPassthroughServer(host="127.0.0.1", port=0)
        else:  # proxy_mode == "litellm"
            from . import litellm_proxy as _proxy
            server = _proxy.LiteLLMProxyServer(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                host="127.0.0.1", port=0,
            )
        server.start()
    except Exception as exc:
        sys.stderr.write(
            f"LLMKernel pty-mode: K12 proxy ({proxy_mode}) start failed: {exc}\n"
        )
        sys.stderr.flush()
        _diagnostics.mark(
            "pty_mode_proxy_start_failed",
            proxy_mode=proxy_mode,
            error_type=type(exc).__name__, error=str(exc),
        )
        return _PROXY_START_FAILED

    base_url = server.base_url()
    os.environ["LLMKERNEL_LITELLM_ENDPOINT_URL"] = base_url
    _diagnostics.mark(
        "pty_mode_proxy_started",
        proxy_mode=proxy_mode, auth_mode=auth_mode, url=base_url,
    )
    return server


def _install_handler(handler: logging.Handler) -> Callable[[], None]:
    """Attach ``handler`` to root + ``llm_kernel`` loggers; strip stdout sinks.

    Returns a restorer that reverses the installation -- removes the
    handler and reattaches any stripped stdout sinks. Tests that run
    :func:`main` in-process call the restorer to keep subsequent tests'
    logger state clean.
    """
    root = logging.getLogger()
    saved_level = root.level
    llm_logger = logging.getLogger("llm_kernel")
    saved_llm_level = llm_logger.level
    # Strip any StreamHandler that targets stdout -- those would otherwise
    # interleave human text into our PTY (RFC-008 §3 disallows that for
    # non-control content).
    stripped: List[logging.Handler] = []
    for existing in list(root.handlers):
        if isinstance(existing, logging.StreamHandler):
            stream = getattr(existing, "stream", None)
            if stream in (sys.stdout,):
                root.removeHandler(existing)
                stripped.append(existing)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # ``llm_kernel`` subtree: ensure messages propagate up to root with
    # our handler attached. We don't need to attach the same handler
    # twice.
    llm_logger.setLevel(logging.INFO)

    def _restore() -> None:
        try:
            root.removeHandler(handler)
        except ValueError:
            pass
        for h in stripped:
            root.addHandler(h)
        root.setLevel(saved_level)
        llm_logger.setLevel(saved_llm_level)

    return _restore


def _emit_ready_handshake(writer: SocketWriter, session_id: str) -> None:
    """Emit the RFC-008 §4 ready handshake LogRecord on the data plane.

    Bypasses the Python ``logging`` machinery so attribute encoding is
    deterministic regardless of which formatter the operator wired.
    """
    from ._attrs import encode_attrs

    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    attrs: Dict[str, Any] = {
        "event.name": "kernel.ready",
        "llmnb.kernel.version": KERNEL_VERSION,
        "llmnb.kernel.session_id": session_id,
        "llmnb.kernel.python_version": py_version,
    }
    for key, value in RFC_VERSIONS.items():
        attrs[f"llmnb.kernel.{key}"] = value
    now_ns = int(_utc_epoch_ns())
    record: Dict[str, Any] = {
        "timeUnixNano": str(now_ns),
        "observedTimeUnixNano": str(now_ns),
        "severityNumber": 9,  # INFO
        "severityText": "INFO",
        "body": {"stringValue": "kernel.ready"},
        "attributes": encode_attrs(attrs),
    }
    writer.write_frame(record)


def _utc_epoch_ns() -> int:
    """Current UTC epoch in nanoseconds. Wrapper so tests can monkey-patch."""
    # ``time.time_ns`` is monotonic-ish in practice but the OTel field
    # spec wants wall-clock; ``datetime.now(tz=UTC)`` gives us that.
    return int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)


async def _async_serve_socket(state: Dict[str, Any]) -> None:
    """Drive the existing sync inbound-line dispatcher off uvicorn's loop.

    BSP-004 V2 (focused retry): runs :func:`_run_read_loop` on a DEDICATED
    :class:`threading.Thread`, not the default :class:`ThreadPoolExecutor`.

    Why dedicated thread, not executor:
      - Executor pool workers have ambiguous lifecycle — Python may reuse
        the same OS thread for unrelated callables, can recycle a worker
        out from under blocking I/O during shutdown, and on Windows
        there's a known interaction where ``subprocess.Popen`` invoked
        on a pool worker during a blocking ``select`` (which is what
        ``_run_read_loop`` is doing) regressed test #4 by EOF'ing the
        parent's socket recv ~1.4s after agent_spawn dispatch.
      - A dedicated thread mirrors the pre-BSP-004 main-thread behavior
        as closely as possible while keeping uvicorn on the asyncio loop.
        It owns the read loop's select/recv pair end-to-end with no pool
        scheduling between iterations.
      - Lifecycle is explicit: thread starts here, exits on
        shutdown_event or socket EOF, joined by lifespan shutdown.

    Halts when the socket EOFs or :attr:`state["async_done"]` is set
    (lifespan shutdown signals this; the inner _run_read_loop polls
    shutdown_event every 500ms via select timeout).
    """
    import asyncio
    from . import _diagnostics

    sock = state["writer"]._sock  # type: ignore[attr-defined]
    if sock is None:
        _diagnostics.mark("async_serve_socket_no_sock")
        return
    kernel = state["kernel"]
    shutdown_event = state["shutdown_event"]
    interrupt_event = state.get("interrupt_event")
    done = asyncio.Event()
    state["async_done"] = done

    _diagnostics.mark("async_serve_socket_started")

    # Dedicated thread, not executor. daemon=True so a stuck read loop
    # (paranoid — _run_read_loop has a select timeout and exits on
    # shutdown_event) doesn't block process exit.
    read_thread = threading.Thread(
        target=_run_read_loop,
        args=(state["writer"], kernel, shutdown_event, interrupt_event),
        name="llmkernel-read-loop",
        daemon=True,
    )
    state["read_thread"] = read_thread
    read_thread.start()

    # Bridge: when the lifespan shutdown signals state["async_done"],
    # set shutdown_event so the read thread's select timeout notices and
    # the loop exits cleanly within ~500ms.
    async def _watch_shutdown() -> None:
        await done.wait()
        shutdown_event.set()

    watch_task = asyncio.create_task(_watch_shutdown())

    # Park the lifespan-attached coroutine here until the read thread
    # exits (socket EOF or shutdown). Polling instead of join() because
    # join() is blocking and would freeze the asyncio loop; this poll
    # cycles at the shutdown_event's natural cadence.
    try:
        while read_thread.is_alive():
            await asyncio.sleep(0.5)
    finally:
        _diagnostics.mark("async_serve_socket_exited")
        watch_task.cancel()
        try:
            await watch_task
        except (asyncio.CancelledError, Exception):
            pass
        # Give the read thread a moment to finish its current iteration
        # if shutdown_event has been set; the daemon flag ensures we
        # don't hang here forever.
        read_thread.join(timeout=2.0)


def boot_kernel() -> Any:
    """BSP-004: synchronous boot. Returns a state dict on success or an int
    exit code on early failure (matches the legacy :func:`main` exit codes:
    2 = no socket env var, 3 = socket connect failed, 11 = K11 proxy config,
    12 = K12 proxy start). Used by :mod:`llm_kernel.app` lifespan.
    """
    from . import _diagnostics
    _diagnostics.mark("pty_mode_main_entry")
    address = os.environ.get(ENV_IPC_SOCKET, "")
    pty_flag = os.environ.get(ENV_PTY_MODE) == "1"
    session_id = os.environ.get(ENV_SESSION_ID) or str(uuid.uuid4())
    _diagnostics.mark("pty_mode_env_read", socket=address, session_id=session_id, pty_flag=pty_flag)
    if not address:
        sys.stderr.write(
            f"LLMKernel pty-mode: {ENV_IPC_SOCKET} env var is required\n"
        )
        sys.stderr.flush()
        _diagnostics.mark("pty_mode_exit_no_socket", rc=2)
        return 2

    termios_restore = _set_raw_mode_if_tty() if pty_flag else None

    writer = SocketWriter()
    try:
        writer.connect(address)
    except OSError as exc:
        sys.stderr.write(
            f"LLMKernel pty-mode: failed to connect to socket {address!r}: {exc}\n"
        )
        sys.stderr.flush()
        if termios_restore is not None:
            termios_restore()
        _diagnostics.mark("pty_mode_exit_socket_connect_failed", error=str(exc), rc=3)
        return 3
    _diagnostics.mark("pty_mode_socket_connected")

    sys.stderr.write(
        f"LLMKernel pty-mode v{KERNEL_VERSION}; socket={address}\n"
    )
    sys.stderr.flush()

    _emit_ready_handshake(writer, session_id)
    _diagnostics.mark("pty_mode_ready_emitted")

    proxy_server = _start_owned_proxy_or_none()
    if proxy_server is _PROXY_START_FAILED:
        if termios_restore is not None:
            termios_restore()
        return 12
    if proxy_server is _PROXY_CONFIG_REJECTED:
        if termios_restore is not None:
            termios_restore()
        return 11

    handler = OtlpDataPlaneHandler(
        socket_writer=writer,
        extra_attributes={"llmnb.kernel.session_id": session_id},
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler_restore = _install_handler(handler)

    kernel = _PtyKernel(writer)
    from ._kernel_hooks import attach_kernel_subsystems
    dispatcher, tracker, bridge, supervisor, metadata_writer = (
        attach_kernel_subsystems(kernel)
    )

    shutdown_event = threading.Event()
    interrupt_event = threading.Event()

    if hasattr(dispatcher, "set_shutdown_event"):
        dispatcher.set_shutdown_event(shutdown_event)
    if hasattr(dispatcher, "set_metadata_writer"):
        dispatcher.set_metadata_writer(metadata_writer)
    if hasattr(dispatcher, "set_agent_supervisor"):
        dispatcher.set_agent_supervisor(supervisor)
    if hasattr(dispatcher, "set_drift_detector"):
        from .drift_detector import DriftDetector
        dispatcher.set_drift_detector(DriftDetector())
    dispatcher.start()
    _diagnostics.mark("pty_mode_dispatcher_started")

    if hasattr(dispatcher, "set_current_volatile_provider"):
        def _current_volatile() -> Dict[str, Any]:
            return {
                "kernel": {
                    "rfc_001_version": "1.0.0",
                    "rfc_002_version": "1.0.1",
                    "rfc_003_version": "2.0.2",
                    "rfc_005_version": "1.0.0",
                    "rfc_006_version": "2.0.2",
                    "rfc_008_version": "1.0.0",
                    "kernel_version": KERNEL_VERSION,
                },
            }
        dispatcher.set_current_volatile_provider(_current_volatile)

    # Signal handlers — only if on the main thread (uvicorn calls lifespan
    # off the main thread, so signal.signal raises ValueError; uvicorn
    # installs its own signal handlers anyway and translates them to
    # lifespan shutdown).
    def _on_sigint(_signum: int, _frame: Any) -> None:
        interrupt_event.set()

    def _on_sigterm(_signum: int, _frame: Any) -> None:
        shutdown_event.set()

    if hasattr(signal, "SIGINT"):
        try:
            signal.signal(signal.SIGINT, _on_sigint)
        except ValueError:
            logger.debug("pty_mode: SIGINT handler not installable from this thread")
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _on_sigterm)
        except ValueError:
            logger.debug("pty_mode: SIGTERM handler not installable from this thread")

    return {
        "writer": writer,
        "kernel": kernel,
        "dispatcher": dispatcher,
        "tracker": tracker,
        "bridge": bridge,
        "supervisor": supervisor,
        "metadata_writer": metadata_writer,
        "shutdown_event": shutdown_event,
        "interrupt_event": interrupt_event,
        "termios_restore": termios_restore,
        "handler_restore": handler_restore,
        "proxy_server": proxy_server,
        "session_id": session_id,
    }


def shutdown_kernel(state: Dict[str, Any]) -> None:
    """BSP-004: cleanup matching :func:`boot_kernel`. Mirrors the legacy
    :func:`main`'s ``finally`` block.
    """
    from . import _diagnostics
    metadata_writer = state.get("metadata_writer")
    dispatcher = state.get("dispatcher")
    proxy_server = state.get("proxy_server")
    writer = state.get("writer")
    termios_restore = state.get("termios_restore")
    handler_restore = state.get("handler_restore")

    if metadata_writer is not None:
        try:
            metadata_writer.snapshot(trigger="shutdown")
        except Exception:
            logger.exception("shutdown_kernel: final metadata snapshot raised")
        try:
            metadata_writer.stop(emit_final=False)
        except Exception:
            logger.exception("shutdown_kernel: metadata_writer.stop raised")
    if dispatcher is not None:
        try:
            dispatcher.stop()
        except Exception:
            logger.exception("shutdown_kernel: dispatcher.stop raised")
    if proxy_server is not None and proxy_server not in (
        _PROXY_START_FAILED, _PROXY_CONFIG_REJECTED,
    ):
        try:
            proxy_server.stop()
            _diagnostics.mark("pty_mode_proxy_stopped")
        except Exception:
            logger.exception("shutdown_kernel: proxy_server.stop raised")
    if writer is not None:
        writer.close()
    if termios_restore is not None:
        termios_restore()
    if handler_restore is not None:
        handler_restore()


def _run_read_loop(
    writer: SocketWriter,
    kernel: _PtyKernel,
    shutdown_event: threading.Event,
    interrupt_event: Optional[threading.Event] = None,
) -> None:
    """Read newline-delimited JSON frames and dispatch them.

    Pulls bytes off the same underlying socket the writer holds. The
    socket is full-duplex so concurrent ``sendall`` (writer thread) and
    ``recv`` (this loop) are safe at the OS level. We use ``select`` so
    ``settimeout`` doesn't bleed into the writer thread's ``sendall``
    path -- the write path stays blocking, the read path polls.

    Halts on socket EOF or when ``shutdown_event`` is set. When
    ``interrupt_event`` is set (SIGINT delivered), drains the flag and
    emits the ``kernel.interrupt_handled`` LogRecord on the data plane
    (RFC-008 §4 step 5).
    """
    import select as _select
    from . import _diagnostics

    sock = writer._sock  # type: ignore[attr-defined]
    if sock is None:
        _diagnostics.mark("read_loop_no_socket")
        return
    _diagnostics.mark("read_loop_entered")
    buffer = bytearray()
    while not shutdown_event.is_set():
        # SIGINT bookkeeping: the handler set ``interrupt_event``; we
        # turn that into the data-plane LogRecord here, off the signal
        # handler's stack, so SocketWriter's lock isn't reentered.
        if interrupt_event is not None and interrupt_event.is_set():
            interrupt_event.clear()
            logger.info(
                "kernel.interrupt_handled",
                extra={"event.name": "kernel.interrupt_handled"},
            )
        try:
            ready, _, _ = _select.select([sock], [], [], 0.5)
        except (OSError, ValueError) as exc:
            _diagnostics.mark(
                "read_loop_select_error",
                error_type=type(exc).__name__, error=str(exc),
            )
            break
        if not ready:
            continue
        try:
            chunk = sock.recv(4096)
        except OSError as exc:
            _diagnostics.mark(
                "read_loop_recv_error",
                error_type=type(exc).__name__, error=str(exc),
            )
            break
        if not chunk:
            # Peer closed -- RFC-008 §4 step 6 normal shutdown path.
            _diagnostics.mark("read_loop_peer_eof")
            break
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            if not line.strip():
                continue
            _dispatch_inbound_line(line, kernel)
    if shutdown_event.is_set():
        _diagnostics.mark("read_loop_shutdown_event_set")


def _dispatch_inbound_line(line: bytes, kernel: _PtyKernel) -> None:
    """Parse one line and route it to the kernel's Comm dispatcher.

    Inbound frames from the extension are RFC-006 v2 thin envelopes per
    RFC-008 §6; we synthesize a Comm message shape so the existing
    :meth:`CustomMessageDispatcher._on_comm_msg` handles it.
    """
    import json
    try:
        frame = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        logger.warning("pty_mode: non-JSON frame dropped (%d bytes)", len(line))
        return
    if not isinstance(frame, dict):
        logger.warning("pty_mode: non-object frame dropped")
        return
    # The dispatcher's Comm target is RFC-006 §2's ``llmnb.rts.v2``.
    msg = {"content": {"data": frame}}
    kernel.shell.comm_manager.deliver("llmnb.rts.v2", msg)


__all__ = [
    "ENV_IPC_SOCKET",
    "ENV_PTY_MODE",
    "ENV_SESSION_ID",
    "KERNEL_VERSION",
    "OtlpDataPlaneHandler",
    "RFC_VERSIONS",
    "SocketWriter",
    "main",
]

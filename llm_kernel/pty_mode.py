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
    address = os.environ.get(ENV_IPC_SOCKET, "")
    pty_flag = os.environ.get(ENV_PTY_MODE) == "1"
    session_id = os.environ.get(ENV_SESSION_ID) or str(uuid.uuid4())
    if not address:
        sys.stderr.write(
            f"LLMKernel pty-mode: {ENV_IPC_SOCKET} env var is required\n"
        )
        sys.stderr.flush()
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
        return 3

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
        writer.close()
        if termios_restore is not None:
            termios_restore()
        if handler_restore is not None:
            handler_restore()
    return 0


# ---------------------------------------------------------------------------
# Helpers (factored for testability)
# ---------------------------------------------------------------------------


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

    sock = writer._sock  # type: ignore[attr-defined]
    if sock is None:
        return
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
        except (OSError, ValueError):
            break
        if not ready:
            continue
        try:
            chunk = sock.recv(4096)
        except OSError:
            break
        if not chunk:
            # Peer closed -- RFC-008 §4 step 6 normal shutdown path.
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

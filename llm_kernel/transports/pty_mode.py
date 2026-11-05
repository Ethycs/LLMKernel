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
8. Write the boot banner ``LLMKernel pty-mode v<KERNEL_VERSION>; socket=<address>``
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

import asyncio
import logging
import os
import platform
import signal
import socket
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from llm_kernel.capture._otlp_log_handler import OtlpDataPlaneHandler
from llm_kernel.capture.socket_writer import SocketWriter
from llm_kernel.wire import versions as _versions

logger: logging.Logger = logging.getLogger("llm_kernel.transports.pty_mode")

#: ``KERNEL_VERSION`` (RFC-008 §4 ``llmnb.kernel.version``) and
#: ``RFC_VERSIONS`` (the ``llmnb.kernel.rfc_NNN_version`` attributes) are
#: forwarded from the single source :mod:`llm_kernel.wire.versions` via
#: ``__getattr__`` below.  ``pty_mode.KERNEL_VERSION`` /
#: ``pty_mode.RFC_VERSIONS`` stay importable from here and always reflect
#: the live source; call sites in this module read ``_versions.*`` at
#: call time so a runtime override propagates to the ready frame, the
#: boot banner and the drift provider alike.
_FORWARDED_VERSION_NAMES = ("KERNEL_VERSION", "RFC_VERSIONS")


def __getattr__(name: str) -> Any:
    if name in _FORWARDED_VERSION_NAMES:
        return getattr(_versions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: Current-environment volatile config for ``DriftDetector.compare``.
#: This is the source module's own function (not a copy) so pty-mode,
#: serve, the app, the facade and the legacy kernel all register the
#: same provider object.
_current_volatile = _versions.current_volatile

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


# ---------------------------------------------------------------------------
# Facade transport adapter (PLAN-kernel-facade commit 6).
#
# Everything that is a wire or process concern lives here: the socket, the
# ready handshake, the kernel-owned proxy, the OTLP log handler, termios
# state, signal handlers and the inbound read loop. The :class:`Kernel`
# facade owns the subsystems and their start/shutdown ordering. One
# lifetime looks like::
#
#     transport = PtyTransport(address=..., session_id=...)
#     rc = transport.prepare()                 # RFC-008 §4 steps 1-3
#     async with Kernel(workdir, kernel_shell=transport.kernel_shell) as k:
#         await k.bind_transport(transport)    # read loop starts
#         await transport.wait_closed()        # EOF / shutdown request
#     transport.close()                        # after the final snapshot hit the wire
#
# The socket deliberately outlives ``Kernel.shutdown()``: the facade drains
# transports first and emits the final ``notebook.metadata`` snapshot during
# subsystem teardown, so closing the writer in ``stop()`` would drop the
# RFC-008 §4 step-6 frame. ``stop()`` only halts the read loop.
# ---------------------------------------------------------------------------


def _install_signal_handlers(
    shutdown_event: threading.Event, interrupt_event: threading.Event,
) -> None:
    """SIGINT -> interrupt_event, SIGTERM -> shutdown_event (main thread only).

    ``signal.signal`` only works on the main thread; callers that run on a
    worker thread (tests, uvicorn's lifespan) skip installation gracefully.
    The handlers stay minimal: ``logger.info`` here could re-enter the
    SocketWriter lock from the signal frame.
    """
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


class PtyTransport:
    """RFC-008 pty-mode transport for the :class:`~llm_kernel.kernel_facade.Kernel`.

    Satisfies the facade's thin transport hook (async ``bind(kernel)`` /
    ``stop()``) and adds the process-side steps around it. Exit codes from
    :meth:`prepare` match the legacy ``main``: 3 socket connect failed,
    11 proxy configuration rejected (K11), 12 proxy start failed (K12).
    """

    def __init__(
        self, *, address: str, session_id: str, pty_flag: bool = False,
    ) -> None:
        self.address: str = address
        self.session_id: str = session_id
        self.pty_flag: bool = pty_flag
        self.writer: SocketWriter = SocketWriter()
        #: The transport-supplied egress handed to ``Kernel(kernel_shell=...)``.
        self.kernel_shell: _PtyKernel = _PtyKernel(self.writer)
        self.shutdown_event: threading.Event = threading.Event()
        self.interrupt_event: threading.Event = threading.Event()
        self._termios_restore: Optional[Callable[[], None]] = None
        self._handler_restore: Optional[Callable[[], None]] = None
        self._proxy_server: Any = None
        self._read_thread: Optional[threading.Thread] = None
        self._prepared: bool = False
        self._closed: bool = False

    # -- RFC-008 §4 steps 1-3: connect, ready handshake, proxy, OTLP handler
    def prepare(self) -> int:
        """Bring the wire up. Returns 0, or the process exit code to use."""
        from llm_kernel._internal import _diagnostics

        # Termios raw mode -- only meaningful under a real PTY. The smoke
        # spawns under pipes, so this is typically a no-op.
        self._termios_restore = _set_raw_mode_if_tty() if self.pty_flag else None
        try:
            self.writer.connect(self.address)
        except OSError as exc:
            sys.stderr.write(
                f"LLMKernel pty-mode: failed to connect to socket "
                f"{self.address!r}: {exc}\n"
            )
            sys.stderr.flush()
            self._restore_terminal()
            _diagnostics.mark(
                "pty_mode_exit_socket_connect_failed", error=str(exc), rc=3,
            )
            return 3
        _diagnostics.mark("pty_mode_socket_connected")
        # Banner BEFORE installing the OTLP handler so it actually reaches
        # PTY stderr (RFC-008 §3 "Boot output").
        sys.stderr.write(
            f"LLMKernel pty-mode v{_versions.KERNEL_VERSION}; socket={self.address}\n"
        )
        sys.stderr.flush()
        # Ready handshake FIRST -- RFC-008 §4 makes this the first frame on
        # the data plane; subsystem logs must come after it.
        _emit_ready_handshake(self.writer, self.session_id)
        _diagnostics.mark("pty_mode_ready_emitted")
        # BSP-001 proxy lifecycle; K11/K12 exit before any subsystem exists.
        proxy_server = _start_owned_proxy_or_none()
        if proxy_server is _PROXY_START_FAILED:
            self._restore_terminal()
            return 12
        if proxy_server is _PROXY_CONFIG_REJECTED:
            self._restore_terminal()
            return 11
        self._proxy_server = proxy_server
        # OTLP handler on root + ``llm_kernel``; strips stdout sinks per
        # RFC-008 §7 so no Python text reaches the PTY after this point.
        handler = OtlpDataPlaneHandler(
            socket_writer=self.writer,
            extra_attributes={"llmnb.kernel.session_id": self.session_id},
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._handler_restore = _install_handler(handler)
        self._prepared = True
        return 0

    # -- facade transport hook ----------------------------------------
    async def bind(self, kernel: Any) -> None:
        """Attach the transport collaborators and start the read loop.

        Called by ``Kernel.bind_transport`` once the subsystems are wired.
        """
        from llm_kernel._internal import _diagnostics

        if not self._prepared:
            raise RuntimeError("PtyTransport.prepare() must succeed before bind()")
        dispatcher = kernel.dispatcher
        # K-CM collaborators that are transport concerns: the shutdown
        # event the ``kernel.shutdown_request`` handler sets, and the
        # volatile-state provider the drift detector compares against.
        if hasattr(dispatcher, "set_shutdown_event"):
            dispatcher.set_shutdown_event(self.shutdown_event)
        if hasattr(dispatcher, "set_current_volatile_provider"):
            dispatcher.set_current_volatile_provider(_current_volatile)
        _install_signal_handlers(self.shutdown_event, self.interrupt_event)
        self._read_thread = threading.Thread(
            target=_run_read_loop,
            args=(
                self.writer, self.kernel_shell,
                self.shutdown_event, self.interrupt_event,
            ),
            name="llmkernel-read-loop",
            daemon=True,
        )
        self._read_thread.start()
        _diagnostics.mark("pty_mode_read_loop_started")

    async def wait_closed(self) -> None:
        """Block until the read loop ends (peer EOF, SIGTERM or a
        ``kernel.shutdown_request``)."""
        thread = self._read_thread
        if thread is None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, thread.join)

    async def stop(self) -> None:
        """Halt the read loop. The socket stays open: ``Kernel.shutdown``
        still has to emit the final snapshot through it (RFC-008 §4 step 6).
        """
        self.shutdown_event.set()
        thread = self._read_thread
        if (
            thread is not None and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: thread.join(timeout=5.0))

    # -- process-side teardown, after the facade is down ---------------
    def close(self) -> None:
        """Stop the kernel-owned proxy, close the socket, restore the
        terminal and the logging handlers. Idempotent."""
        from llm_kernel._internal import _diagnostics

        if self._closed:
            return
        self._closed = True
        proxy_server = self._proxy_server
        if proxy_server is not None and proxy_server not in (
            _PROXY_START_FAILED, _PROXY_CONFIG_REJECTED,
        ):
            try:
                proxy_server.stop()
                _diagnostics.mark("pty_mode_proxy_stopped")
            except Exception:  # pragma: no cover - defensive
                logger.exception("pty_mode: proxy_server.stop raised")
        self.writer.close()
        self._restore_terminal()
        if self._handler_restore is not None:
            self._handler_restore()
            self._handler_restore = None

    def _restore_terminal(self) -> None:
        if self._termios_restore is not None:
            self._termios_restore()
            self._termios_restore = None


async def run_pty_kernel(
    transport: PtyTransport, *, workdir: Optional[Path] = None,
) -> int:
    """One pty-mode kernel lifetime on the facade. Returns the exit code.

    ``transport.prepare()`` must already have returned 0.
    """
    from llm_kernel._internal import _diagnostics
    from llm_kernel.kernel_facade import Kernel, KernelStartError

    try:
        try:
            async with Kernel(
                workdir=workdir or Path.cwd(),
                kernel_shell=transport.kernel_shell,
            ) as kernel:
                await kernel.bind_transport(transport)
                _diagnostics.mark("pty_mode_dispatcher_started")
                await transport.wait_closed()
        except KernelStartError:
            logger.exception("pty_mode: kernel start failed")
            return 1
    finally:
        # Only now, after the final notebook.metadata snapshot has been
        # emitted by Kernel.shutdown(), may the socket go away.
        transport.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Run the pty-mode kernel; return a process exit code.

    Reads ``LLMKERNEL_IPC_SOCKET`` from the env. Fails fast (returns 2
    after writing to PTY stderr) when missing -- the extension is
    expected to set this before spawning per RFC-008 §4 step 1.
    """
    from llm_kernel._internal import _diagnostics

    _diagnostics.mark("pty_mode_main_entry")
    address = os.environ.get(ENV_IPC_SOCKET, "")
    pty_flag = os.environ.get(ENV_PTY_MODE) == "1"
    session_id = os.environ.get(ENV_SESSION_ID) or str(uuid.uuid4())
    _diagnostics.mark(
        "pty_mode_env_read", socket=address, session_id=session_id, pty_flag=pty_flag,
    )
    if not address:
        sys.stderr.write(
            f"LLMKernel pty-mode: {ENV_IPC_SOCKET} env var is required\n"
        )
        sys.stderr.flush()
        _diagnostics.mark("pty_mode_exit_no_socket", rc=2)
        return 2
    transport = PtyTransport(
        address=address, session_id=session_id, pty_flag=pty_flag,
    )
    rc = transport.prepare()
    if rc != 0:
        return rc
    return asyncio.run(run_pty_kernel(transport))


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
    from llm_kernel._internal import _diagnostics

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
            from llm_kernel.providers import anthropic_passthrough as _pt
            server = _pt.AnthropicPassthroughServer(host="127.0.0.1", port=0)
        else:  # proxy_mode == "litellm"
            from llm_kernel.providers import litellm_proxy as _proxy
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
    from llm_kernel._internal._attrs import encode_attrs

    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    attrs: Dict[str, Any] = {
        "event.name": "kernel.ready",
        "llmnb.kernel.version": _versions.KERNEL_VERSION,
        "llmnb.kernel.session_id": session_id,
        "llmnb.kernel.python_version": py_version,
    }
    # One ``llmnb.kernel.rfc_NNN_version`` per RFC the single source defines.
    for key, value in _versions.RFC_VERSIONS.items():
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


# ---------------------------------------------------------------------------
# BSP-004 synchronous boot helpers, as facade consumers.
#
# ``boot_kernel`` / ``shutdown_kernel`` are the in-process pty-mode lifetime
# for a caller that does not own an event loop. They run exactly one
# :class:`~llm_kernel.kernel_facade.Kernel` on a private loop thread, the
# way serve mode's ``_boot_kernel_for_serve`` does, and reach the subsystems
# through the facade's read-only accessors. ``Kernel.start`` owns the wiring:
# the dispatcher, the metadata writer, the supervisor, the drift detector and
# the volatile-version provider, plus the PLAN-S5.0.1 defense (the
# contamination detector bound to the writer, the outbound sanitizer bound to
# the transport's socket writer). Nothing is attached by hand here.
# ---------------------------------------------------------------------------


def boot_kernel(*, workdir: Optional[Path] = None) -> Any:
    """BSP-004: synchronous boot. Returns a state dict on success or an int
    exit code on early failure (matches :func:`main`: 2 = no socket env var,
    3 = socket connect failed, 11 = K11 proxy config, 12 = K12 proxy start).
    """
    from llm_kernel._internal import _diagnostics
    from llm_kernel.kernel_facade import Kernel

    _diagnostics.mark("pty_mode_main_entry")
    address = os.environ.get(ENV_IPC_SOCKET, "")
    pty_flag = os.environ.get(ENV_PTY_MODE) == "1"
    session_id = os.environ.get(ENV_SESSION_ID) or str(uuid.uuid4())
    _diagnostics.mark(
        "pty_mode_env_read", socket=address, session_id=session_id, pty_flag=pty_flag,
    )
    if not address:
        sys.stderr.write(
            f"LLMKernel pty-mode: {ENV_IPC_SOCKET} env var is required\n"
        )
        sys.stderr.flush()
        _diagnostics.mark("pty_mode_exit_no_socket", rc=2)
        return 2
    transport = PtyTransport(
        address=address, session_id=session_id, pty_flag=pty_flag,
    )
    rc = transport.prepare()
    if rc != 0:
        return rc

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(
        target=loop.run_forever, name="llmkernel-pty-loop", daemon=True,
    )
    loop_thread.start()
    kernel = Kernel(
        workdir=workdir or Path.cwd(), kernel_shell=transport.kernel_shell, loop=loop,
    )

    async def _start() -> None:
        await kernel.start()
        await kernel.bind_transport(transport)
        _diagnostics.mark("pty_mode_dispatcher_started")

    try:
        asyncio.run_coroutine_threadsafe(_start(), loop).result(timeout=60.0)
    except BaseException:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5.0)
        transport.close()
        raise
    return {
        "kernel": transport.kernel_shell,
        "facade": kernel,
        "transport": transport,
        "writer": transport.writer,
        "dispatcher": kernel.dispatcher,
        "tracker": kernel.run_tracker,
        "supervisor": kernel.supervisor,
        "metadata_writer": kernel.metadata_writer,
        "shutdown_event": transport.shutdown_event,
        "interrupt_event": transport.interrupt_event,
        "session_id": session_id,
        "loop": loop,
        "loop_thread": loop_thread,
    }


def shutdown_kernel(state: Dict[str, Any]) -> None:
    """BSP-004: cleanup matching :func:`boot_kernel`. ``Kernel.shutdown``
    halts the read loop, emits the final ``notebook.metadata`` snapshot and
    stops the subsystems; only then does the socket go away."""
    kernel = state.get("facade")
    loop = state.get("loop")
    if kernel is not None and loop is not None:
        try:
            asyncio.run_coroutine_threadsafe(kernel.shutdown(), loop).result(timeout=30.0)
        except Exception:  # pragma: no cover - defensive
            logger.exception("shutdown_kernel: kernel shutdown raised")
        loop.call_soon_threadsafe(loop.stop)
        loop_thread = state.get("loop_thread")
        if loop_thread is not None:
            loop_thread.join(timeout=5.0)
        if not loop.is_running():
            loop.close()
    transport = state.get("transport")
    if transport is not None:
        transport.close()


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
    from llm_kernel._internal import _diagnostics

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
    kernel.shell.comm_manager.deliver(_versions.COMM_TARGET, msg)


__all__ = [
    "ENV_IPC_SOCKET",
    "ENV_PTY_MODE",
    "ENV_SESSION_ID",
    "KERNEL_VERSION",
    "OtlpDataPlaneHandler",
    "PtyTransport",
    "RFC_VERSIONS",
    "SocketWriter",
    "main",
    "run_pty_kernel",
]

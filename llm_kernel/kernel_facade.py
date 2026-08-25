"""Embeddable LLMKernel facade — ``from llm_kernel import Kernel``.

This is the implementation of PLAN-kernel-facade.md (substrate-trajectory
slice 2, facade half). It hoists the subsystem construction that today
lives in each entry point (``transports/pty_mode.py``,
``transports/serve_mode.py``, the dormant ``app.py`` lifespan, and the
IPython ``kernel.py``) into a single class that owns the subsystem
references, so the kernel can be driven as a Python object rather than
only spawned as a subprocess.

The construction order mirrors the canonical wiring in
``_kernel_hooks.attach_kernel_subsystems`` plus the dispatcher-collaborator
wiring the pty transport performs after boot; see
``docs/04 - Reference/kernel/_wiring-audit.md`` for the audited map this
was built against.

Scope of this commit (facade commit 2): the class skeleton —
``start`` / ``shutdown`` / async-context-manager lifecycle — and the
in-process kernel shell the dispatcher needs. Public delegators
(spawn/send/etc.) and event-log accessors land in later commits.

Deviations from the plan, per the wiring audit:
  * ``start()`` raises :class:`KernelStartError` on boot failure rather
    than ``os._exit`` (that lives in the dormant ``app.py``; hard-exiting
    would kill an embedding host).
  * No ``MCPManager`` is constructed — it is the outbound client, built
    only by the IPython kernel, and is not part of the capture wiring.
  * ``MetadataWriter`` is constructed with ``workspace_root=workdir``;
    there is no ``event_log_path`` / ``events.jsonl`` sink in the tree.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from llm_kernel._kernel_hooks import _DEFAULT_LITELLM_ENDPOINT
from llm_kernel.capture.custom_messages import CustomMessageDispatcher
from llm_kernel.capture.drift_detector import DriftDetector
from llm_kernel.capture.metadata_writer import MetadataWriter
from llm_kernel.capture.run_tracker import RunTracker
from llm_kernel.supervision.agent_supervisor import AgentSupervisor

logger: logging.Logger = logging.getLogger("llm_kernel.kernel_facade")

#: Default identity for the embedded kernel's run-tracker. Library callers
#: are one logical "operator"; per-zone/per-agent identity is carried on
#: the spawn calls, not the kernel-level tracker.
_DEFAULT_AGENT_ID: str = "kernel"
_DEFAULT_ZONE_ID: str = "default"


class KernelStartError(RuntimeError):
    """Raised when :meth:`Kernel.start` cannot wire the subsystems.

    Carries the return-code ``rc`` the subprocess entry points would have
    exited with (2/3/11/12 per RFC-008), so a subprocess wrapper can
    translate it back into an exit code while an embedded host stays
    alive. ``rc`` is ``None`` when the failure is not one of the known
    boot-failure codes.
    """

    def __init__(self, message: str, *, rc: Optional[int] = None) -> None:
        super().__init__(message)
        self.rc: Optional[int] = rc


# ---------------------------------------------------------------------------
# In-process kernel shell — the embedded analog of pty_mode._PtyKernel.
#
# CustomMessageDispatcher reads ``kernel.session``, ``kernel.iopub_socket``,
# ``kernel.shell.comm_manager`` and ``kernel._parent_header`` and routes
# Family-A spans through ``session.send`` and Family-B..G envelopes through
# ``comm.send``. The pty transport points those at a SocketWriter; the
# embedded facade points them at an in-memory sink so a Python caller can
# read the captured event stream without a wire.
# ---------------------------------------------------------------------------


class _EmbeddedSink:
    """Thread-safe collector of the envelopes the dispatcher emits.

    Both Family-A run spans (via ``session.send``) and Family-B..G thin
    envelopes (via ``comm.send``) land here in emission order. The event-log
    accessors (a later commit) read from this; for the lifecycle commit it
    simply exists so nothing is dropped on the floor.
    """

    def __init__(self) -> None:
        self._frames: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def write_frame(self, frame: Dict[str, Any]) -> None:
        with self._lock:
            self._frames.append(frame)

    def snapshot(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of the frames captured so far."""
        with self._lock:
            return list(self._frames)


class _EmbeddedKernelSession:
    """Stand-in for ``ipykernel.session.Session`` (Family-A span egress)."""

    def __init__(self, sink: _EmbeddedSink) -> None:
        self._sink = sink

    def send(self, _socket: Any, _msg_type: str, **kwargs: Any) -> None:
        content = kwargs.get("content") or {}
        data = content.get("data") or {}
        for mime, payload in data.items():
            if mime == "application/vnd.rts.run+json" and isinstance(payload, dict):
                self._sink.write_frame(payload)


class _EmbeddedComm:
    """One Comm-shaped object the dispatcher sends/receives through."""

    def __init__(self, name: str, sink: _EmbeddedSink) -> None:
        self.name = name
        self._sink = sink
        self._on_msg: Optional[Callable[[Dict[str, Any]], None]] = None

    def on_msg(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._on_msg = callback

    def send(self, data: Dict[str, Any]) -> None:
        self._sink.write_frame(data)

    def close(self) -> None:
        self._on_msg = None

    def deliver(self, msg: Dict[str, Any]) -> None:
        cb = self._on_msg
        if cb is None:
            logger.warning("embedded comm %r has no on_msg; frame dropped", self.name)
            return
        try:
            cb(msg)
        except Exception:  # pragma: no cover - defensive
            logger.exception("embedded comm %r on_msg raised", self.name)


class _EmbeddedCommManager:
    """Stand-in for ``ipykernel.comm.CommManager``.

    On ``register_target`` it synthesizes a self-attach (fires the open
    callback synchronously with an :class:`_EmbeddedComm`), draining the
    dispatcher's pre-attach buffer exactly as the pty comm manager does.
    """

    def __init__(self, sink: _EmbeddedSink) -> None:
        self._sink = sink
        self._comms: Dict[str, _EmbeddedComm] = {}
        self._lock = threading.Lock()

    def register_target(self, name: str, callback: Callable[..., None]) -> None:
        comm = _EmbeddedComm(name=name, sink=self._sink)
        with self._lock:
            self._comms[name] = comm
        callback(comm, {"content": {"data": {}}})

    def unregister_target(self, name: str, _callback: Callable[..., None]) -> None:
        with self._lock:
            self._comms.pop(name, None)

    def deliver(self, name: str, msg: Dict[str, Any]) -> None:
        with self._lock:
            comm = self._comms.get(name)
        if comm is None:
            logger.warning("embedded: inbound frame for unregistered target %r dropped", name)
            return
        comm.deliver(msg)


class _EmbeddedShell:
    def __init__(self, sink: _EmbeddedSink) -> None:
        self.comm_manager = _EmbeddedCommManager(sink)


class _EmbeddedKernel:
    """Minimal kernel-shaped object satisfying CustomMessageDispatcher."""

    def __init__(self, sink: _EmbeddedSink) -> None:
        self.session = _EmbeddedKernelSession(sink)
        self.iopub_socket = "<embedded-iopub-placeholder>"
        self.shell = _EmbeddedShell(sink)
        self._parent_header: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# The facade.
# ---------------------------------------------------------------------------


class Kernel:
    """Embeddable LLMKernel facade.

    Stability: **unstable.** The public API may change before the substrate
    trajectory's slice 4 lands; after that, semantic versioning applies.

    Threadsafety: public methods are threadsafe (they delegate to
    subsystems whose own locks cover the call). Event subscription is
    async-only.

    Event loop: :meth:`start` captures ``asyncio.get_running_loop()`` (or an
    explicit ``loop=``); the facade never calls ``asyncio.run`` for you.

    Usage::

        async with Kernel(workdir=Path("./scratch")) as k:
            ...   # delegators + accessors land in later facade commits
    """

    def __init__(
        self,
        *,
        workdir: Path,
        config: Optional[Dict[str, Any]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        shutdown_grace_seconds: float = 10.0,
    ) -> None:
        self._workdir: Path = Path(workdir)
        self._config: Dict[str, Any] = dict(config or {})
        self._loop: Optional[asyncio.AbstractEventLoop] = loop
        self._shutdown_grace: float = shutdown_grace_seconds

        # Subsystem references — private; never exposed as public attributes
        # (Con 1 in the plan: reaching inside is a signal to extend the
        # facade, not to poke the subsystems).
        self._embedded_kernel: Optional[_EmbeddedKernel] = None
        self._sink: _EmbeddedSink = _EmbeddedSink()
        self._dispatcher: Optional[CustomMessageDispatcher] = None
        self._run_tracker: Optional[RunTracker] = None
        self._metadata_writer: Optional[MetadataWriter] = None
        self._supervisor: Optional[AgentSupervisor] = None
        self._trace_id: Optional[str] = None
        self._started: bool = False
        self._lifecycle_lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        """Wire subsystems in the canonical order. Idempotent.

        Order (mirrors ``_kernel_hooks`` + the pty collaborator wiring):
        dispatcher → run-tracker → metadata-writer → supervisor →
        dispatcher collaborators. No transports are bound. Raises
        :class:`KernelStartError` on failure (never ``os._exit``).
        """
        with self._lifecycle_lock:
            if self._started:
                return
            if self._loop is None:
                self._loop = asyncio.get_running_loop()
            try:
                kernel = _EmbeddedKernel(self._sink)
                dispatcher = CustomMessageDispatcher(kernel)
                dispatcher.start()

                trace_id = uuid.uuid4().hex
                run_tracker = RunTracker(
                    trace_id=trace_id, sink=dispatcher,
                    agent_id=_DEFAULT_AGENT_ID, zone_id=_DEFAULT_ZONE_ID,
                )
                metadata_writer = MetadataWriter(
                    dispatcher=dispatcher, run_tracker=run_tracker,
                    workspace_root=self._workdir,
                )
                metadata_writer.start()

                url = (
                    self._config.get("litellm_endpoint_url")
                    or _DEFAULT_LITELLM_ENDPOINT
                )
                supervisor = AgentSupervisor(
                    run_tracker=run_tracker, dispatcher=dispatcher,
                    litellm_endpoint_url=url,
                )

                # Dispatcher collaborator wiring — mirrors pty_mode.main
                # (the setters attach_kernel_subsystems does not do).
                dispatcher.set_metadata_writer(metadata_writer)
                dispatcher.set_agent_supervisor(supervisor)
                dispatcher.set_drift_detector(DriftDetector())
            except Exception as exc:
                # Best-effort partial teardown so a failed start leaves no
                # autosave / heartbeat threads running.
                self._teardown_locked(
                    dispatcher=locals().get("dispatcher"),
                    metadata_writer=locals().get("metadata_writer"),
                    supervisor=None,
                )
                raise KernelStartError(f"Kernel.start() failed: {exc}") from exc

            self._embedded_kernel = kernel
            self._dispatcher = dispatcher
            self._run_tracker = run_tracker
            self._metadata_writer = metadata_writer
            self._supervisor = supervisor
            self._trace_id = trace_id
            self._started = True

    async def shutdown(self) -> None:
        """Reverse construction: terminate agents → snapshot + stop the
        metadata writer → stop the dispatcher. Idempotent."""
        with self._lifecycle_lock:
            if not self._started:
                return
            self._teardown_locked(
                dispatcher=self._dispatcher,
                metadata_writer=self._metadata_writer,
                supervisor=self._supervisor,
            )
            self._embedded_kernel = None
            self._dispatcher = None
            self._run_tracker = None
            self._metadata_writer = None
            self._supervisor = None
            self._trace_id = None
            self._started = False

    def _teardown_locked(
        self, *,
        dispatcher: Optional[CustomMessageDispatcher],
        metadata_writer: Optional[MetadataWriter],
        supervisor: Optional[AgentSupervisor],
    ) -> None:
        """Best-effort teardown of whatever is wired. Caller holds the
        lifecycle lock. Each step is independently guarded so one failure
        does not strand later cleanup.

        Agents are stopped before the final snapshot so they stop emitting
        into the log first; this is the first teardown path to call
        ``terminate_all`` (no live path does today — see the wiring audit).
        """
        if supervisor is not None:
            try:
                supervisor.terminate_all(grace_seconds=self._shutdown_grace)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Kernel.shutdown: supervisor.terminate_all raised")
        if metadata_writer is not None:
            try:
                metadata_writer.snapshot(trigger="shutdown")
            except Exception:  # pragma: no cover - defensive
                logger.exception("Kernel.shutdown: metadata snapshot raised")
            try:
                metadata_writer.stop(emit_final=False)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Kernel.shutdown: metadata stop raised")
        if dispatcher is not None:
            try:
                dispatcher.stop()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Kernel.shutdown: dispatcher stop raised")

    async def __aenter__(self) -> "Kernel":
        await self.start()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.shutdown()


__all__ = ["Kernel", "KernelStartError"]

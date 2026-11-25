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
import os
import threading
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

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
        self._cond = threading.Condition(threading.Lock())
        self._closed = False

    def write_frame(self, frame: Dict[str, Any]) -> None:
        # Called from subsystem threads (dispatcher emit). Appends and
        # wakes any async subscriber blocked in wait_for_after.
        with self._cond:
            if self._closed:
                return
            self._frames.append(frame)
            self._cond.notify_all()

    def snapshot(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of every frame captured so far."""
        with self._cond:
            return list(self._frames)

    def get_since(self, index: int) -> List[Dict[str, Any]]:
        """Return frames at ``index`` onward (empty if none newer)."""
        with self._cond:
            return list(self._frames[index:])

    def close(self) -> None:
        """Mark the sink closed and wake all waiters (subscribers end)."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def wait_for_after(self, index: int, timeout: float) -> bool:
        """Block until a frame past ``index`` exists, the sink closes, or
        ``timeout`` elapses. Returns ``closed``. Runs in an executor
        thread so the async subscriber can await it. This is the thread→
        loop bridge: subsystem threads notify; the event loop waits here.
        """
        with self._cond:
            if len(self._frames) <= index and not self._closed:
                self._cond.wait(timeout=timeout)
            return self._closed


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
        kernel_shell: Optional[Any] = None,
    ) -> None:
        self._workdir: Path = Path(workdir)
        self._config: Dict[str, Any] = dict(config or {})
        self._loop: Optional[asyncio.AbstractEventLoop] = loop
        self._shutdown_grace: float = shutdown_grace_seconds
        # Pluggable egress. When None (the default), start() builds the
        # in-process _EmbeddedKernel and the dispatcher emits into the
        # in-memory sink that read_event_log/subscribe_events read. A
        # transport can instead supply a kernel-shaped shell that routes
        # the dispatcher's output to a wire (e.g. the PTY SocketWriter);
        # in that mode the embedded sink stays empty by design.
        self._kernel_shell: Optional[Any] = kernel_shell

        # Subsystem references — private; never exposed as public attributes
        # (Con 1 in the plan: reaching inside is a signal to extend the
        # facade, not to poke the subsystems).
        self._embedded_kernel: Optional[_EmbeddedKernel] = None
        self._sink: _EmbeddedSink = _EmbeddedSink()
        self._dispatcher: Optional[CustomMessageDispatcher] = None
        self._run_tracker: Optional[RunTracker] = None
        self._metadata_writer: Optional[MetadataWriter] = None
        self._supervisor: Optional[AgentSupervisor] = None
        self._transports: List[Any] = []
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
                if self._kernel_shell is not None:
                    # Transport-supplied egress: the dispatcher emits to
                    # the wire, not the embedded sink. read_event_log and
                    # subscribe_events see nothing in this mode.
                    kernel = self._kernel_shell
                else:
                    # Fresh sink per start so a restart-after-shutdown collects
                    # into a live (non-closed) buffer, and any subscribers of a
                    # prior run have already ended on that run's sink.close().
                    self._sink = _EmbeddedSink()
                    kernel = _EmbeddedKernel(self._sink)
                dispatcher = CustomMessageDispatcher(kernel)
                dispatcher.start()

                trace_id = (
                    self._config.get("trace_id")
                    or os.environ.get("LLMKERNEL_RUN_TRACE_ID")
                    or uuid.uuid4().hex
                )
                run_tracker = RunTracker(
                    trace_id=trace_id, sink=dispatcher,
                    agent_id=os.environ.get("LLMKERNEL_AGENT_ID", _DEFAULT_AGENT_ID),
                    zone_id=os.environ.get("LLMKERNEL_ZONE_ID", _DEFAULT_ZONE_ID),
                )
                metadata_writer = MetadataWriter(
                    dispatcher=dispatcher, run_tracker=run_tracker,
                    workspace_root=self._workdir,
                )
                metadata_writer.start()

                url = (
                    self._config.get("litellm_endpoint_url")
                    or os.environ.get("LLMKERNEL_LITELLM_ENDPOINT_URL")
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
        """Reverse construction: drain transports → terminate agents →
        snapshot + stop the metadata writer → stop the dispatcher → close
        the sink. Idempotent.
        """
        # Flip state + capture references under the lock (no await inside,
        # so we never block the loop holding a threading lock). A concurrent
        # shutdown then sees ``_started`` False and is a no-op, and the sync
        # delegators start refusing immediately.
        with self._lifecycle_lock:
            if not self._started:
                return
            self._started = False
            transports = list(self._transports)
            self._transports = []
            dispatcher = self._dispatcher
            metadata_writer = self._metadata_writer
            supervisor = self._supervisor
            self._embedded_kernel = None
            self._dispatcher = None
            self._run_tracker = None
            self._metadata_writer = None
            self._supervisor = None
            self._trace_id = None
        # Outside the lock: drain transports first (their stop() is async),
        # then the sync subsystem teardown, then close the sink.
        for transport in transports:
            try:
                await transport.stop()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Kernel.shutdown: transport stop raised")
        self._teardown_locked(
            dispatcher=dispatcher,
            metadata_writer=metadata_writer,
            supervisor=supervisor,
        )
        self._sink.close()  # end any live event subscribers

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

    # -- transport-adapter surface (facade commits 6-8) -----------------
    #
    # The process entry points (pty-mode, serve, the uvicorn app) are
    # consumers of the facade: they create one Kernel, hand it their
    # transport-supplied egress and bind their transport. Their adapters
    # need the wired subsystems to attach transport-specific collaborators
    # (shutdown events, volatile-state providers, proxy run-trackers).
    # These accessors are read-only and raise until start() has run; they
    # are not a license to re-wire the subsystems from outside.

    @property
    def started(self) -> bool:
        """True between a successful :meth:`start` and :meth:`shutdown`."""
        return self._started

    @property
    def workdir(self) -> Path:
        return self._workdir

    @property
    def trace_id(self) -> Optional[str]:
        """The run-tracker trace id of the current run (None when stopped)."""
        return self._trace_id

    @property
    def dispatcher(self) -> CustomMessageDispatcher:
        self._require_started()
        assert self._dispatcher is not None
        return self._dispatcher

    @property
    def run_tracker(self) -> RunTracker:
        self._require_started()
        assert self._run_tracker is not None
        return self._run_tracker

    @property
    def metadata_writer(self) -> MetadataWriter:
        self._require_started()
        assert self._metadata_writer is not None
        return self._metadata_writer

    @property
    def supervisor(self) -> AgentSupervisor:
        return self._require_started()

    # -- public delegators (facade commit 3) ----------------------------

    def _require_started(self) -> AgentSupervisor:
        if not self._started or self._supervisor is None:
            raise RuntimeError(
                "Kernel is not started; call `await start()` (or use "
                "`async with Kernel(...)`) before driving agents"
            )
        return self._supervisor

    def spawn_agent(
        self, *, zone_id: str, agent_id: str, task: str, work_dir: Path,
        **kwargs: Any,
    ) -> str:
        """Spawn an agent and return its ``agent_id``.

        Thin wrapper over :meth:`AgentSupervisor.spawn`; extra keyword
        args (``api_key``, ``model``, ``use_bare``, ``set_base_url``,
        ``resume_claude_session_id``) pass straight through.
        """
        handle = self._require_started().spawn(
            zone_id=zone_id, agent_id=agent_id, task=task,
            work_dir=work_dir, **kwargs,
        )
        return handle.agent_id

    def send_user_turn(
        self, agent_id: str, text: str, *,
        cell_id: Optional[str] = None,
        inject_notebook_prefix: bool = True,
    ) -> Dict[str, Any]:
        """Continue an agent with one operator turn.

        See :meth:`AgentSupervisor.send_user_turn`; ``inject_notebook_prefix
        =False`` (SE-03) is the non-notebook-caller path.
        """
        return self._require_started().send_user_turn(
            agent_id, text, cell_id=cell_id,
            inject_notebook_prefix=inject_notebook_prefix,
        )

    def revert(self, agent_id: str, target_turn_id: str) -> None:
        """Revert an agent's conversation to ``target_turn_id``."""
        self._require_started().revert(agent_id, target_turn_id)

    def stop(self, agent_id: str) -> None:
        """Stop an agent's current run."""
        self._require_started().stop(agent_id)

    def list_agents(self) -> List[Dict[str, Any]]:
        """Return a summary of every live (spawned) agent."""
        return self._require_started().list_agents()

    def list_zones(self) -> List[Dict[str, Any]]:
        """Return the zones that currently have live agents.

        Derived from :meth:`list_agents` — one dict per distinct
        ``zone_id`` with its live ``agent_ids``. This is the *live* view;
        the full persisted zone set lives in the notebook metadata and is
        a separate (deferred) read, not a supervisor concern.
        """
        zones: Dict[str, Dict[str, Any]] = {}
        for a in self._require_started().list_agents():
            z = zones.setdefault(
                a["zone_id"], {"zone_id": a["zone_id"], "agent_ids": []},
            )
            z["agent_ids"].append(a["agent_id"])
        return list(zones.values())

    # -- event log access (facade commit 4) -----------------------------

    def read_event_log(
        self, *, since: int = 0, until: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return captured envelopes in emission order, sliced ``[since:until]``.

        Native, in-memory read of the event stream the kernel emitted —
        Family-A run spans plus Family-B..G control envelopes, exactly as a
        wire consumer would have seen them. ``since`` / ``until`` are
        indices into that stream. Works after shutdown too (returns what
        was captured before teardown).
        """
        frames = self._sink.snapshot()
        return frames[max(0, since):until]

    async def subscribe_events(
        self, *, since: int = 0,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield captured envelopes from ``since`` onward, then live ones as
        they arrive, until the kernel shuts down.

        Async-only. Frames are appended by subsystem threads; this bridges
        them to the caller's loop by awaiting the sink's blocking wait in an
        executor. Ends cleanly when :meth:`shutdown` closes the sink.
        """
        if not self._started:
            raise RuntimeError(
                "Kernel is not started; call `await start()` before "
                "subscribing to events"
            )
        loop = self._loop or asyncio.get_running_loop()
        sink = self._sink  # follow this run's sink across a later restart
        idx = max(0, since)
        while True:
            for frame in sink.get_since(idx):
                yield frame
                idx += 1
            closed = await loop.run_in_executor(
                None, sink.wait_for_after, idx, 1.0,
            )
            if closed:
                for frame in sink.get_since(idx):
                    yield frame
                    idx += 1
                return

    # -- transport binding + run loop (facade commit 5) -----------------

    async def bind_transport(self, transport: Any) -> None:
        """Bind a transport to the running kernel.

        Thin hook, not the formal Transport protocol (that is a later
        follow-up): ``transport`` need only expose async ``bind(kernel)``
        and ``stop()``. ``bind`` is awaited now; ``stop`` is awaited first
        during :meth:`shutdown`. Bound transports are drained before agents
        are terminated so the wire is quiet before the subsystems go down.
        """
        self._require_started()
        await transport.bind(self)
        self._transports.append(transport)

    async def run_forever(self) -> None:
        """Block until cancelled — convenience for a subprocess wrapper.

        ``async with Kernel(...) as k: await k.run_forever()`` keeps the
        process alive until the surrounding task is cancelled (e.g. a
        SIGTERM handler the wrapper installs). Embedded/library callers
        typically drive the kernel directly and never call this.
        """
        self._require_started()
        await asyncio.Event().wait()  # until the task is cancelled


__all__ = ["Kernel", "KernelStartError"]

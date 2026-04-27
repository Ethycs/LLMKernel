"""LLMKernel custom-message dispatcher (RFC-006 v2 wire format).

Production :class:`Sink` Track B2's :class:`RunTracker` expects, plus
the inbound router for envelopes arriving from the extension. Connects
two Jupyter primitives to the RFC-006 v2 catalog:

* **Family A** (``run.start`` / ``run.event`` / ``run.complete``) rides
  on ``display_data`` / ``update_display_data`` IOPub messages with
  ``transient.display_id == spanId``.  Per RFC-006 §1 the cell-output
  carries ONE MIME-typed payload of type
  ``application/vnd.rts.run+json`` whose value is the OTLP span itself
  -- there is no envelope at this layer; the OTLP span is
  self-describing.  The legacy ``application/vnd.rts.envelope+json``
  MIME emitted alongside is dropped (deprecated at v2.0, removed
  before v2.1 per RFC-006 §1 "Conformance during transition").
* **Families B-F** ride on a Jupyter ``Comm`` at target
  ``llmnb.rts.v2``.  The v2 Comm envelope is the thin shape per
  RFC-006 §3 -- ``{type, payload, correlation_id?}`` only.  The
  removed v1 fields (``direction``, ``timestamp``, ``rfc_version``)
  are reconstructable from the carrier (Comm direction is implicit;
  the target name encodes the major version) so re-emitting them
  burns bytes without information.
* **Family F** (``notebook.metadata``) is the persistence channel
  RFC-005 §"Persistence strategy" requires.  ``MetadataWriter``
  emits snapshots through this dispatcher's :meth:`emit` path; the
  extension-side ``metadata-applier`` consumes them via
  ``vscode.NotebookEdit.updateNotebookMetadata``.

Thread-safe -- the run-tracker emits from the MCP-server thread, the
LiteLLM proxy from uvicorn's thread, the metadata writer from its
30-second timer thread, and Comm callbacks from the kernel main
thread.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional

from .run_envelope import (
    DIRECTION_KERNEL_TO_EXTENSION, RFC003_MESSAGE_TYPES, make_envelope,
    validate_envelope,
)

if TYPE_CHECKING:  # pragma: no cover
    from ipykernel.ipkernel import IPythonKernel

logger: logging.Logger = logging.getLogger("llm_kernel.custom_messages")

#: Registered Comm target.  Extension-side counterparts (the TS
#: ``MessageRouter``) MUST attach a Comm with this exact ``target_name``
#: per RFC-006 §2; mismatched majors fail at Comm-open time, which is
#: the version-rejection mechanism the v2 supersession promises.
DEFAULT_COMM_TARGET: str = "llmnb.rts.v2"
#: MIME for the OTLP run payload (RFC-006 §1).
MIME_RUN: str = "application/vnd.rts.run+json"
#: Legacy v1 envelope MIME.  RFC-006 §1 "Conformance during transition"
#: marks dual emission deprecated at v2.0 and bans it by v2.1; the v2
#: kernel emits ONLY :data:`MIME_RUN` on Family A.  This constant is
#: retained for documentation and for tests asserting deprecation.
MIME_ENVELOPE: str = "application/vnd.rts.envelope+json"
#: Pre-attach buffer cap; oldest is dropped with a warning beyond this.
DEFAULT_BUFFER_SIZE: int = 128
#: Family E heartbeat cadence (seconds) per RFC-006 §7 v2.0.2 amendment.
#: The kernel MUST emit ``heartbeat.kernel`` every 5 seconds in V1.
DEFAULT_HEARTBEAT_INTERVAL_SEC: float = 5.0

_RUN_LIFECYCLE_TYPES = frozenset({"run.start", "run.event", "run.complete"})

#: Inbound handler signature: receives the v2 thin envelope as a dict.
InboundHandler = Callable[[Dict[str, Any]], None]


def _to_thin_v2(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Project an internal v1-shaped envelope onto the RFC-006 v2 thin form.

    The kernel's internal contract between the run-tracker and the
    dispatcher is still ``{message_type, direction, correlation_id,
    timestamp, rfc_version, payload}`` (the ``run_envelope`` module
    validates that shape).  RFC-006 §3 specifies the over-the-wire
    Comm envelope is the thin form ``{type, payload, correlation_id?}``;
    this helper produces that form.

    ``correlation_id`` is preserved iff the inbound envelope's
    ``message_type`` belongs to a request/response pair (currently
    only ``agent_graph.query`` / ``agent_graph.response`` per RFC-006
    §5).  For all other types it is omitted because RFC-006 §3 makes
    it optional and receivers MUST NOT depend on it.
    """
    msg_type = envelope["message_type"]
    out: Dict[str, Any] = {
        "type": msg_type,
        "payload": envelope["payload"],
    }
    # RFC-006 §5 makes correlation_id required for request/response
    # pairs.  Other families MAY include one for tracing but receivers
    # MUST NOT depend on it; we only forward when present AND when the
    # type is in the request/response set, to keep wire diffs small.
    if msg_type in {"agent_graph.query", "agent_graph.response"}:
        out["correlation_id"] = envelope["correlation_id"]
    return out


def _from_thin_v2(thin: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Reconstruct an internal v1 envelope from a v2 thin form.

    Inbound v2 envelopes carry ``{type, payload, correlation_id?}``.
    The dispatcher's internal handler dispatch keys on
    ``message_type``; we synthesize the missing fields with safe
    defaults so :func:`validate_envelope` accepts the result and the
    handler chain receives a stable shape.  Returns ``None`` on
    structurally-malformed input (the caller logs and drops).
    """
    if not isinstance(thin, dict):
        return None
    if "type" not in thin or "payload" not in thin:
        return None
    from .run_envelope import (
        DIRECTION_EXTENSION_TO_KERNEL, RFC003_VERSION, _utc_now_iso,
    )
    correlation_id = thin.get("correlation_id")
    if not isinstance(correlation_id, str) or not correlation_id:
        # Synthesize a UUID-shaped placeholder; validate_envelope only
        # requires the field be a non-empty string of one of the known
        # shapes, so a UUID4 is the safest synthetic value.
        import uuid as _uuid
        correlation_id = str(_uuid.uuid4())
    return {
        "message_type": thin["type"],
        "direction": DIRECTION_EXTENSION_TO_KERNEL,
        "correlation_id": correlation_id,
        "timestamp": _utc_now_iso(),
        "rfc_version": RFC003_VERSION,
        "payload": thin["payload"],
    }


class CustomMessageDispatcher:
    """Routes RFC-006 v2 envelopes between the kernel and the extension.

    Outbound (``emit``) forwards a validated internal envelope as an
    IOPub ``display_data`` / ``update_display_data`` (Family A) or a
    Comm ``comm_msg`` (Families B-F).  Inbound (``_on_comm_msg``)
    accepts the RFC-006 §3 thin form, reconstructs the internal
    envelope shape, validates, and dispatches to per-message-type
    handlers registered via :meth:`register_handler`.  Holds a
    regular reference to the kernel; ``LLMKernel.do_shutdown`` MUST
    call :meth:`stop`.
    """

    def __init__(
        self,
        kernel: "IPythonKernel",
        comm_target: str = DEFAULT_COMM_TARGET,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        heartbeat_interval_sec: float = DEFAULT_HEARTBEAT_INTERVAL_SEC,
    ) -> None:
        """Bind to ``kernel``; ``comm_target`` defaults to ``llmnb.rts.v2``.

        ``heartbeat_interval_sec`` controls the Family E ``heartbeat.kernel``
        cadence per RFC-006 §7 v2.0.2 amendment.  Tests override to a
        small value (e.g., 0.1s) so the heartbeat loop can be exercised
        in <1s without sleeping the spec's 5s.
        """
        self._kernel = kernel
        self._comm_target: str = comm_target
        self._buffer_size: int = buffer_size
        self._lock: threading.RLock = threading.RLock()
        self._active_comm: Any = None
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=buffer_size)
        self._handlers: Dict[str, List[InboundHandler]] = {}
        self._started: bool = False
        # Family E heartbeat state.  ``_start_time`` is monotonic so
        # restarting the kernel mid-clock-skew does not produce
        # negative ``uptime_seconds`` values.
        self._heartbeat_interval_sec: float = heartbeat_interval_sec
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop_event: threading.Event = threading.Event()
        self._start_time: float = time.monotonic()
        self._last_run_complete_iso: Optional[str] = None
        # RFC-008 §4 step 6 graceful-shutdown coordination: when set,
        # the inbound ``kernel.shutdown_request`` handler signals this
        # event so the read loop in ``pty_mode`` (or whichever process
        # owns the kernel lifecycle) can exit its loop and drop into
        # the final-snapshot finally block.  May be ``None`` for
        # in-process / IPython kernels where SIGTERM owns shutdown.
        self._shutdown_event: Optional[threading.Event] = None
        # RFC-006 §9 "Hydrate exclusivity": at most one
        # ``notebook.metadata`` ``mode:"hydrate"`` envelope per session.
        self._hydrate_processed: bool = False
        # Optional collaborator references resolved lazily from the
        # kernel attribute namespace (per RFC-006 §"Consumers" the
        # dispatcher routes inbound traffic; the metadata writer and
        # agent supervisor own the per-family business logic).  The
        # accessors below resolve these on demand so hot-restart of
        # collaborators (e.g., a respawn) is observed.
        self._metadata_writer_override: Any = None
        self._agent_supervisor_override: Any = None
        self._drift_detector_override: Any = None
        # Source for ``current_volatile`` on hydrate.  The kernel's
        # actual environment supplies this in production (see
        # :meth:`_collect_current_volatile`); tests inject directly.
        self._current_volatile_provider: Optional[Callable[[], Dict[str, Any]]] = None

    # -- Collaborator wiring ----------------------------------------------
    #
    # The dispatcher does not own the metadata writer, agent supervisor,
    # or drift detector; it *consumes* them per RFC-006 §"Consumers".
    # The integration owners (``_kernel_hooks.attach_kernel_subsystems``
    # and ``pty_mode``) wire collaborators onto the kernel's attribute
    # namespace; we resolve them lazily so a respawn or late-binding
    # in tests is honored without dispatcher restart.

    def set_shutdown_event(self, event: threading.Event) -> None:
        """Bind the dispatcher's graceful-shutdown signal.

        The host (``pty_mode.main`` or the kernel lifecycle owner) calls
        this to give the dispatcher the same ``threading.Event`` its
        read loop watches.  When ``kernel.shutdown_request`` arrives
        (RFC-006 §7.1 / RFC-008 §4 step 6), the inbound handler sets
        the event so the host's finally block runs the final
        ``notebook.metadata`` snapshot before clean exit.  Idempotent.
        """
        self._shutdown_event = event

    def set_metadata_writer(self, writer: Any) -> None:
        """Bind a metadata writer collaborator (overrides kernel-attr lookup).

        Tests use this to inject a mock without standing up the full
        ``attach_kernel_subsystems`` chain.  Production uses the
        kernel-attr surface (``_llmnb_metadata_writer``).
        """
        self._metadata_writer_override = writer

    def set_agent_supervisor(self, supervisor: Any) -> None:
        """Bind an agent supervisor collaborator (overrides kernel-attr lookup)."""
        self._agent_supervisor_override = supervisor

    def set_drift_detector(self, detector: Any) -> None:
        """Bind a drift detector collaborator (overrides kernel-attr lookup)."""
        self._drift_detector_override = detector

    def set_current_volatile_provider(
        self, provider: Callable[[], Dict[str, Any]],
    ) -> None:
        """Bind a callable returning the kernel's current volatile config.

        Used by the hydrate handler to drive ``DriftDetector.compare``
        with the live environment values (kernel version, RFC versions,
        etc.).  Tests inject a static dict; production wires the live
        runtime collector.
        """
        self._current_volatile_provider = provider

    def _resolve_metadata_writer(self) -> Any:
        if self._metadata_writer_override is not None:
            return self._metadata_writer_override
        return getattr(self._kernel, "_llmnb_metadata_writer", None)

    def _resolve_agent_supervisor(self) -> Any:
        if self._agent_supervisor_override is not None:
            return self._agent_supervisor_override
        return getattr(self._kernel, "_llmnb_agent_supervisor", None)

    def _resolve_drift_detector(self) -> Any:
        if self._drift_detector_override is not None:
            return self._drift_detector_override
        return getattr(self._kernel, "_llmnb_drift_detector", None)

    def _collect_current_volatile(self) -> Dict[str, Any]:
        """Return the current-environment volatile-config dict.

        Production uses the provider injected via
        :meth:`set_current_volatile_provider` (the pty-mode bootstrap
        wires it to the live kernel state).  When no provider is set we
        return an empty dict; the drift detector treats missing fields
        as "skip this category" rather than producing spurious drift
        events.
        """
        provider = self._current_volatile_provider
        if provider is None:
            return {}
        try:
            return dict(provider())
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "current_volatile provider raised; returning empty dict",
            )
            return {}

    def start(self) -> None:
        """Register the Comm target and start the heartbeat thread.

        Idempotent.  Registers the built-in inbound handlers (Family
        B/C/F/§7.1) so the dispatcher is wire-ready before any
        envelope arrives -- handlers may be added later but the four
        built-ins are part of the dispatcher's contract per RFC-006
        §§4, 5, 7.1, 8.
        """
        with self._lock:
            if self._started:
                logger.debug("dispatcher already started; ignoring")
                return
            self._kernel.shell.comm_manager.register_target(
                self._comm_target, self._on_comm_open
            )
            self._started = True
            self._register_builtin_handlers()
            self._start_heartbeat_locked()
            logger.info("dispatcher started; comm_target=%s", self._comm_target)

    def stop(self) -> None:
        """Close the active Comm, stop the heartbeat thread, and unregister."""
        with self._lock:
            if not self._started:
                return
            comm = self._active_comm
            self._active_comm = None
            self._heartbeat_stop_event.set()
            heartbeat_thread = self._heartbeat_thread
            self._heartbeat_thread = None
            try:
                self._kernel.shell.comm_manager.unregister_target(
                    self._comm_target, self._on_comm_open
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("failed to unregister %s", self._comm_target)
            self._started = False
        # Join the heartbeat thread OUTSIDE the lock -- the loop calls
        # ``self.emit`` (Engineering Guide §11.7: never join inside a
        # lock the joined thread may need to acquire).
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2.0)
        if comm is not None:
            try:
                comm.close()
            except Exception:  # pragma: no cover - defensive
                logger.exception("failed to close active comm")
        logger.info("dispatcher stopped")

    def emit(self, envelope: Dict[str, Any]) -> None:
        """Validate and route an outbound envelope.

        Family A rides IOPub ``display_data`` / ``update_display_data``
        keyed by ``transient.display_id == spanId``; only the OTLP span
        itself is emitted (RFC-006 §1 -- no envelope MIME).  Other
        families are flattened to the RFC-006 §3 thin v2 envelope and
        ride the Comm.  When no Comm is attached, the v2-projected
        envelope is buffered (oldest dropped beyond ``buffer_size``).
        Raises ``ValueError`` if validation fails (no IOPub emission).
        """
        validate_envelope(envelope)  # internal contract; see module docs.
        message_type = envelope["message_type"]
        if message_type in _RUN_LIFECYCLE_TYPES:
            # Stamp the most-recent ``run.complete`` time so the next
            # heartbeat can advertise ``last_run_timestamp`` per RFC-006
            # §7 Family E payload.  We use envelope["timestamp"] (already
            # ISO 8601 from ``make_envelope``) so heartbeat clients see
            # the same wall-clock time the receiver did.
            if message_type == "run.complete":
                ts = envelope.get("timestamp")
                if isinstance(ts, str) and ts:
                    self._last_run_complete_iso = ts
            self._emit_run_lifecycle(envelope)
            return
        thin = _to_thin_v2(envelope)
        with self._lock:
            comm = self._active_comm
            if comm is None:
                if len(self._buffer) >= self._buffer_size:
                    dropped = self._buffer.popleft()
                    logger.warning(
                        "dispatcher buffer overflow; dropped oldest "
                        "(type=%s)",
                        dropped.get("type"),
                    )
                self._buffer.append(thin)
                return
        self._send_via_comm(comm, thin)

    def register_handler(
        self, message_type: str, handler: InboundHandler
    ) -> Callable[[], None]:
        """Register an inbound handler for ``message_type``.

        Multiple handlers per type are supported; each receives a copy
        of the reconstructed internal envelope.  Returns a dispose
        function that removes this specific handler.  Raises
        ``ValueError`` if ``message_type`` is outside the catalog.
        """
        if message_type not in RFC003_MESSAGE_TYPES:
            raise ValueError(
                f"message_type {message_type!r} not in RFC-006 v2 catalog"
            )
        with self._lock:
            self._handlers.setdefault(message_type, []).append(handler)

        def _dispose() -> None:
            with self._lock:
                handlers = self._handlers.get(message_type, [])
                try:
                    handlers.remove(handler)
                except ValueError:
                    pass

        return _dispose

    def _on_comm_open(self, comm: Any, msg: Dict[str, Any]) -> None:
        """Handle the extension attaching a Comm to our target.

        Stashes the comm, registers ``on_msg`` for inbound traffic, and
        flushes any v2-thin envelopes buffered before the attach.
        """
        with self._lock:
            self._active_comm = comm
            buffered = list(self._buffer)
            self._buffer.clear()
        try:
            comm.on_msg(self._on_comm_msg)
        except Exception:  # pragma: no cover - defensive
            logger.exception("failed to attach on_msg handler")
        logger.info(
            "comm attached on %s; flushing %d buffered envelope(s)",
            self._comm_target, len(buffered),
        )
        for thin in buffered:
            self._send_via_comm(comm, thin)

    def _on_comm_msg(self, msg: Dict[str, Any]) -> None:
        """Handle an inbound ``comm_msg`` from the extension.

        Per RFC-006 §3 the inbound shape is the thin v2 envelope.  We
        accept either the v2 thin form (``{type, payload,
        correlation_id?}``) or the legacy v1 form (full envelope) for
        transition tolerance; the legacy form is recognized by the
        presence of ``message_type``.  Validates, then fans out to
        every registered handler for its type.  Unknown types drop
        with a warning per RFC-006 W4 (V1 fail-closed).
        """
        try:
            data = msg["content"]["data"]
        except (KeyError, TypeError):
            logger.warning("inbound comm_msg missing content.data; dropped")
            return
        if isinstance(data, dict) and "message_type" in data:
            # Legacy v1 envelope -- accept on the receiver side; senders
            # MUST flip to v2 thin form per RFC-006 W10.
            envelope = data
        else:
            envelope = _from_thin_v2(data) if isinstance(data, dict) else None
            if envelope is None:
                logger.warning(
                    "inbound comm_msg failed v2 thin-envelope parse; dropped"
                )
                return
        try:
            validate_envelope(envelope)
        except ValueError as exc:
            logger.warning("inbound envelope failed validation: %s; dropped", exc)
            return
        message_type = envelope["message_type"]
        with self._lock:
            handlers = list(self._handlers.get(message_type, []))
        if not handlers:
            logger.warning(
                "inbound type=%s has no registered handlers; dropped "
                "(RFC-006 W4 fail-closed)", message_type,
            )
            return
        for handler in handlers:
            try:
                handler(dict(envelope))
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "inbound handler for %s raised; continuing", message_type,
                )

    def _emit_run_lifecycle(self, envelope: Dict[str, Any]) -> None:
        """Emit a Family A envelope as an IOPub display message.

        Per RFC-006 §1 the cell-output carries ONLY the OTLP span at
        ``application/vnd.rts.run+json`` -- the legacy
        ``application/vnd.rts.envelope+json`` MIME is dropped.  The
        display_id equals the span's spanId so
        ``update_display_data`` lands on the originating cell.
        """
        payload: Dict[str, Any] = envelope["payload"]
        message_type: str = envelope["message_type"]
        # display_id is the OTLP spanId; we fall back to the envelope's
        # correlation_id only if the payload was minted outside the
        # kernel's run-tracker (defensive).
        display_id: str = payload.get("spanId") or envelope["correlation_id"]
        msg_type = (
            "display_data" if message_type == "run.start"
            else "update_display_data"
        )
        content: Dict[str, Any] = {
            "data": {MIME_RUN: payload},
            "metadata": {},
            "transient": {"display_id": display_id},
        }
        self._iopub_send(msg_type, content)

    def _iopub_send(self, msg_type: str, content: Dict[str, Any]) -> None:
        """Serialize ``content`` onto the kernel's IOPub socket.

        Holds the lock around ``session.send`` so concurrent producers
        do not interleave ZMQ frames on the same socket.
        """
        with self._lock:
            session = self._kernel.session
            iopub_socket = self._kernel.iopub_socket
            parent = getattr(self._kernel, "_parent_header", None) or {}
            try:
                session.send(
                    iopub_socket, msg_type, content=content, parent=parent, ident=None,
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "session.send failed for msg_type=%s; envelope dropped", msg_type,
                )

    def _send_via_comm(self, comm: Any, thin: Dict[str, Any]) -> None:
        """Send a v2 thin envelope through the active Comm."""
        try:
            comm.send(thin)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "comm.send raised for type=%s; envelope dropped",
                thin.get("type"),
            )

    # ------------------------------------------------------------------
    # Family E: heartbeat.kernel emitter (RFC-006 §7 v2.0.2 amendment)
    # ------------------------------------------------------------------

    def _start_heartbeat_locked(self) -> None:
        """Start the heartbeat daemon thread.  Caller MUST hold ``self._lock``.

        Called from :meth:`start`; idempotent against an already-running
        thread.  The thread is daemon=True so an unclean exit doesn't
        keep the kernel process alive (Engineering Guide §9 / §11.7).
        """
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="llmnb.dispatcher.heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        """Emit a ``heartbeat.kernel`` envelope every interval until stop.

        Cadence is :data:`DEFAULT_HEARTBEAT_INTERVAL_SEC` (5s per RFC-006
        §7 v2.0.2) by default; tests pass a smaller value via the
        constructor for fast assertions.  Each iteration consults
        :meth:`_kernel_state` for the operator-facing badge, and stamps
        ``last_run_timestamp`` from the most recent ``run.complete`` we
        observed (or ``None`` if no run has closed yet).
        """
        while not self._heartbeat_stop_event.is_set():
            # Wait first so we don't burst-emit at startup; an immediate
            # heartbeat would race with the ready handshake.
            if self._heartbeat_stop_event.wait(self._heartbeat_interval_sec):
                return
            try:
                payload = {
                    "kernel_state": self._kernel_state(),
                    "uptime_seconds": time.monotonic() - self._start_time,
                    "last_run_timestamp": self._last_run_complete_iso,
                }
                envelope = make_envelope(
                    "heartbeat.kernel", payload,
                    correlation_id=str(uuid.uuid4()),
                    direction=DIRECTION_KERNEL_TO_EXTENSION,
                )
                self.emit(envelope)
            except Exception:  # pragma: no cover - defensive
                # Never let a heartbeat error tear down the kernel; log
                # and continue.  Per Engineering Guide §11.7, the log
                # call here may re-enter the dispatcher's path through
                # OTLP -- ``_lock`` is an RLock so this is safe.
                logger.exception("heartbeat loop iteration raised; continuing")

    def _kernel_state(self) -> str:
        """Return the operator-facing kernel-state badge for the heartbeat.

        Maps dispatcher state onto RFC-006 §7's enum:
        ``starting | ok | degraded | shutting_down``.  V1 reports
        ``shutting_down`` as soon as the shutdown event is set
        (regardless of whether the actual shutdown sequence has
        finished); ``ok`` is the normal post-start state.
        """
        if self._shutdown_event is not None and self._shutdown_event.is_set():
            return "shutting_down"
        if not self._started:
            return "starting"
        return "ok"

    # ------------------------------------------------------------------
    # Built-in inbound handlers (Families B/C/F + §7.1 shutdown)
    # ------------------------------------------------------------------

    def _register_builtin_handlers(self) -> None:
        """Register the dispatcher-owned inbound handlers.

        Per RFC-006 §"Consumers" the dispatcher routes inbound traffic
        on these types to the metadata writer (B, F) and agent
        supervisor (F-hydrate).  Registration happens in :meth:`start`
        so the dispatcher is wire-ready before any extension envelope
        arrives; users may add additional handlers later.
        """
        # RFC-006 §7.1 + RFC-008 §4 step 6: graceful shutdown.
        self.register_handler(
            "kernel.shutdown_request", self._handle_kernel_shutdown_request,
        )
        # RFC-006 §4: Family B layout edits.
        self.register_handler("layout.edit", self._handle_layout_edit)
        # RFC-006 §5: Family C agent-graph queries.
        self.register_handler("agent_graph.query", self._handle_agent_graph_query)
        # RFC-006 §8: Family F notebook.metadata (now bidirectional in
        # v2.0.2 -- the kernel handles inbound mode:"hydrate" envelopes).
        self.register_handler("notebook.metadata", self._handle_notebook_metadata)

    def _handle_kernel_shutdown_request(self, envelope: Dict[str, Any]) -> None:
        """Handle ``kernel.shutdown_request`` per RFC-006 §7.1.

        Sets the host's shutdown event (the same ``threading.Event``
        the read loop in ``pty_mode`` watches) so the host's finally
        block runs the final ``notebook.metadata`` snapshot before
        clean exit.  EOF remains the V1.0.0 fallback shutdown trigger
        per RFC-006 §7.1; both paths converge on the host's finally
        block.

        ``reason`` is informational; we log it as an OTLP-style log
        record (``event.name = kernel.shutdown_requested`` / severity
        INFO / attribute ``llmnb.shutdown_reason``) so tape capture
        retains it.
        """
        payload = envelope.get("payload") or {}
        reason = payload.get("reason") if isinstance(payload, dict) else None
        # Single-call structured log; ``logger.info`` rides the OTLP
        # data-plane handler in pty-mode.  RLock on ``self._lock``
        # makes the re-entrant log-on-emit path safe (Engineering
        # Guide §11.7).
        logger.info(
            "kernel.shutdown_requested",
            extra={
                "event.name": "kernel.shutdown_requested",
                "llmnb.shutdown_reason": reason if isinstance(reason, str) else "",
            },
        )
        event = self._shutdown_event
        if event is None:
            # Defensive: no host-bound shutdown event.  We log and
            # return; the EOF fallback (RFC-006 §7.1 last paragraph)
            # will eventually trigger when the extension closes the
            # socket.
            logger.warning(
                "kernel.shutdown_request: no shutdown_event bound; "
                "relying on EOF fallback (RFC-006 §7.1)"
            )
            return
        event.set()

    def _handle_layout_edit(self, envelope: Dict[str, Any]) -> None:
        """Handle ``layout.edit`` per RFC-006 §4.

        Calls ``MetadataWriter.apply_layout_edit(operation, parameters)``
        to mutate the layout tree, then emits a ``layout.update`` echo
        per RFC-006 §4 ("Kernel applies, then echoes the new state via
        ``layout.update``").  When the writer is missing, log and drop
        -- the dispatcher cannot synthesize layout state on its own.
        """
        writer = self._resolve_metadata_writer()
        if writer is None:
            logger.warning(
                "layout.edit: no MetadataWriter bound; dropped (RFC-006 W4)",
            )
            return
        payload = envelope.get("payload") or {}
        operation = payload.get("operation")
        parameters = payload.get("parameters") or {}
        try:
            writer.apply_layout_edit(
                operation=operation, parameters=parameters,
            )
            update_payload = writer.emit_layout_update()
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "layout.edit: writer.apply_layout_edit/emit_layout_update raised",
            )
            return
        if not isinstance(update_payload, dict):
            logger.warning(
                "layout.edit: writer.emit_layout_update returned non-dict; dropped",
            )
            return
        out = make_envelope(
            "layout.update", update_payload,
            correlation_id=str(uuid.uuid4()),
            direction=DIRECTION_KERNEL_TO_EXTENSION,
        )
        self.emit(out)

    def _handle_agent_graph_query(self, envelope: Dict[str, Any]) -> None:
        """Handle ``agent_graph.query`` per RFC-006 §5.

        Calls ``MetadataWriter.apply_agent_graph_command(command,
        parameters)`` and emits the matching ``agent_graph.response``
        with the originating ``correlation_id`` (RFC-006 §5 requires
        the response's correlation_id equal the query's).
        """
        writer = self._resolve_metadata_writer()
        if writer is None:
            logger.warning(
                "agent_graph.query: no MetadataWriter bound; dropped",
            )
            return
        payload = envelope.get("payload") or {}
        query_type = payload.get("query_type")
        try:
            response_payload = writer.apply_agent_graph_command(
                command=query_type, parameters=payload,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "agent_graph.query: writer.apply_agent_graph_command raised",
            )
            return
        if not isinstance(response_payload, dict):
            logger.warning(
                "agent_graph.query: response payload was not a dict; dropped",
            )
            return
        correlation_id = envelope.get("correlation_id") or str(uuid.uuid4())
        out = make_envelope(
            "agent_graph.response", response_payload,
            correlation_id=correlation_id,
            direction=DIRECTION_KERNEL_TO_EXTENSION,
        )
        self.emit(out)

    def _handle_notebook_metadata(self, envelope: Dict[str, Any]) -> None:
        """Handle inbound ``notebook.metadata`` envelopes per RFC-006 §8.

        Bidirectional in v2.0.2.  Behavior by ``mode``:

        * ``"hydrate"`` -- the extension's file-open path. Validate
          single-shot per RFC-006 §9 "Hydrate exclusivity"; call
          ``MetadataWriter.hydrate``; drive ``DriftDetector.compare``
          against the live volatile config; respawn agents from
          ``config.recoverable.agents[]`` via
          ``AgentSupervisor.respawn_from_config``; emit a confirmation
          ``mode:"snapshot"`` envelope with ``trigger:"hydrate_complete"``.
        * ``"snapshot"`` -- only valid as a hydrate-confirmation echo
          which the kernel itself never sends inbound; log and ignore.
        * ``"patch"`` -- V1 rejects (RFC-006 §8); emit a wire-failure
          log record.
        """
        payload = envelope.get("payload") or {}
        mode = payload.get("mode")
        if mode == "hydrate":
            self._handle_hydrate(payload)
            return
        if mode == "snapshot":
            logger.info(
                "notebook.metadata mode=snapshot received inbound; "
                "logged-and-ignored (kernel never receives this)"
            )
            return
        if mode == "patch":
            logger.warning(
                "wire-failure: notebook.metadata mode=patch is V1.5+; "
                "rejected (RFC-006 §8)",
                extra={"event.name": "wire-failure"},
            )
            return
        logger.warning(
            "wire-failure: notebook.metadata mode=%r is unknown; rejected",
            mode, extra={"event.name": "wire-failure"},
        )

    def _handle_hydrate(self, payload: Dict[str, Any]) -> None:
        """Process one ``mode:"hydrate"`` envelope (RFC-006 §8 / §9).

        Per RFC-006 §9 "Hydrate exclusivity" we accept at most one
        hydrate per session; subsequent hydrates emit a wire-failure
        LogRecord and are otherwise dropped.
        """
        with self._lock:
            already = self._hydrate_processed
            if not already:
                self._hydrate_processed = True
        if already:
            logger.warning(
                "wire-failure: notebook.metadata mode=hydrate received twice; "
                "rejected (RFC-006 §9 hydrate exclusivity)",
                extra={"event.name": "wire-failure"},
            )
            return
        snapshot = payload.get("snapshot")
        if not isinstance(snapshot, dict):
            logger.warning(
                "wire-failure: notebook.metadata mode=hydrate missing snapshot; "
                "rejected",
                extra={"event.name": "wire-failure"},
            )
            return

        writer = self._resolve_metadata_writer()
        supervisor = self._resolve_agent_supervisor()
        detector = self._resolve_drift_detector()
        if writer is None or supervisor is None:
            logger.warning(
                "notebook.metadata mode=hydrate: missing collaborator "
                "(writer=%s supervisor=%s); rejected",
                writer is not None, supervisor is not None,
            )
            return

        # Step 1: hydrate the writer's in-memory state.
        try:
            writer.hydrate(snapshot)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "notebook.metadata mode=hydrate: writer.hydrate raised; aborting",
            )
            return

        # Step 2: drift comparison.  K-MW's hydrate already absorbs the
        # persisted ``drift_log``; we append fresh comparison events
        # by writing into the writer's drift log via
        # ``append_drift_event`` (already exposed) so the next
        # snapshot includes them.
        if detector is not None:
            persisted_volatile = (
                snapshot.get("config", {}).get("volatile", {}).get("kernel", {})
                if isinstance(snapshot.get("config"), dict) else {}
            )
            current_volatile = self._collect_current_volatile()
            try:
                # ``DriftDetector.compare`` accepts the full persisted
                # snapshot plus current-environment kwargs; we drive the
                # kernel-volatile comparison here and let the agent /
                # MCP comparisons run when their inputs are available.
                drift_events = detector.compare(
                    snapshot, current_kernel=current_volatile,
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "notebook.metadata mode=hydrate: drift_detector.compare raised",
                )
                drift_events = []
            for event in drift_events or []:
                if not isinstance(event, dict):
                    continue
                # Prefer a writer-exposed method when available; fall
                # back to direct mutation of the in-memory list (the
                # writer's hydrate already created/replaced it).
                appender = getattr(writer, "append_drift_event", None)
                if callable(appender):
                    try:
                        appender(
                            field_path=event.get("field_path", ""),
                            previous_value=event.get("previous_value"),
                            current_value=event.get("current_value"),
                            severity=event.get("severity", "info"),
                            detected_at=event.get("detected_at"),
                        )
                    except Exception:  # pragma: no cover - defensive
                        logger.exception(
                            "append_drift_event raised; continuing",
                        )
                else:  # pragma: no cover - K-MW provides append_drift_event
                    in_memory = getattr(writer, "_drift_log", None)
                    if isinstance(in_memory, list):
                        in_memory.append(event)

        # Step 3: respawn agents from the recoverable config.
        recoverable_agents = []
        cfg = snapshot.get("config")
        if isinstance(cfg, dict):
            recoverable = cfg.get("recoverable")
            if isinstance(recoverable, dict):
                ag = recoverable.get("agents")
                if isinstance(ag, list):
                    recoverable_agents = ag
        try:
            supervisor.respawn_from_config(recoverable_agents)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "notebook.metadata mode=hydrate: supervisor.respawn_from_config raised",
            )

        # Step 4: emit a confirmation ``mode:"snapshot"`` envelope.
        try:
            post_snapshot = writer.snapshot()
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "notebook.metadata mode=hydrate: writer.snapshot raised; "
                "no confirmation emitted",
            )
            return
        if not isinstance(post_snapshot, dict):
            logger.warning(
                "notebook.metadata mode=hydrate: writer.snapshot returned "
                "non-dict; no confirmation emitted",
            )
            return
        confirmation_payload = {
            "mode": "snapshot",
            "snapshot_version": post_snapshot.get("snapshot_version", 0),
            "snapshot": post_snapshot,
            "trigger": "hydrate_complete",
        }
        confirmation = make_envelope(
            "notebook.metadata", confirmation_payload,
            correlation_id=str(uuid.uuid4()),
            direction=DIRECTION_KERNEL_TO_EXTENSION,
        )
        self.emit(confirmation)


__all__ = [
    "CustomMessageDispatcher",
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_COMM_TARGET",
    "DEFAULT_HEARTBEAT_INTERVAL_SEC",
    "InboundHandler",
    "MIME_ENVELOPE",
    "MIME_RUN",
]

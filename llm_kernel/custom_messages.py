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
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional

from .run_envelope import RFC003_MESSAGE_TYPES, validate_envelope

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
    ) -> None:
        """Bind to ``kernel``; ``comm_target`` defaults to ``llmnb.rts.v2``."""
        self._kernel = kernel
        self._comm_target: str = comm_target
        self._buffer_size: int = buffer_size
        self._lock: threading.RLock = threading.RLock()
        self._active_comm: Any = None
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=buffer_size)
        self._handlers: Dict[str, List[InboundHandler]] = {}
        self._started: bool = False

    def start(self) -> None:
        """Register the Comm target on the kernel.  Idempotent."""
        with self._lock:
            if self._started:
                logger.debug("dispatcher already started; ignoring")
                return
            self._kernel.shell.comm_manager.register_target(
                self._comm_target, self._on_comm_open
            )
            self._started = True
            logger.info("dispatcher started; comm_target=%s", self._comm_target)

    def stop(self) -> None:
        """Close the active Comm and unregister the target.  Idempotent."""
        with self._lock:
            if not self._started:
                return
            comm = self._active_comm
            self._active_comm = None
            try:
                self._kernel.shell.comm_manager.unregister_target(
                    self._comm_target, self._on_comm_open
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("failed to unregister %s", self._comm_target)
            self._started = False
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


__all__ = [
    "CustomMessageDispatcher",
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_COMM_TARGET",
    "InboundHandler",
    "MIME_ENVELOPE",
    "MIME_RUN",
]

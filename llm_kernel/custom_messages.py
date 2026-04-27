"""LLMKernel custom-message dispatcher (Stage 2 Track B3).

Production :class:`Sink` Track B2's :class:`RunTracker` expects, plus the
inbound router for RFC-003 envelopes arriving from the extension. Connects
two Jupyter primitives to the RFC-003 catalog (``docs/rfcs/RFC-003``):

* Family A (``run.start`` / ``run.event`` / ``run.complete``) rides on
  ``display_data`` / ``update_display_data`` IOPub messages with
  ``transient.display_id == run_id``. The cell-output renderer keys on
  ``application/vnd.rts.run+json`` (DR-0009 / chapter 06).
* Families B-E ride on a Jupyter ``Comm`` at target ``llmnb.rts.v1``. The
  kernel registers the target at startup; the extension attaches a Comm
  and the dispatcher buffers any messages produced before the attach.

Thread-safe — the run-tracker emits from the MCP-server thread, the
LiteLLM proxy from uvicorn's thread, and Comm callbacks from the kernel
main thread.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List

from .run_envelope import RFC003_MESSAGE_TYPES, validate_envelope

if TYPE_CHECKING:  # pragma: no cover
    from ipykernel.ipkernel import IPythonKernel

logger: logging.Logger = logging.getLogger("llm_kernel.custom_messages")

#: Registered Comm target. Extension-side counterparts (the TS
#: ``MessageRouter``) MUST attach a Comm with this exact ``target_name``.
DEFAULT_COMM_TARGET: str = "llmnb.rts.v1"
#: MIME for the run payload (chapter 06 §"In-place display updates").
MIME_RUN: str = "application/vnd.rts.run+json"
#: MIME carrying the full RFC-003 envelope alongside the run payload.
MIME_ENVELOPE: str = "application/vnd.rts.envelope+json"
#: Pre-attach buffer cap; oldest is dropped with a warning beyond this.
DEFAULT_BUFFER_SIZE: int = 128

_RUN_LIFECYCLE_TYPES = frozenset({"run.start", "run.event", "run.complete"})

#: Inbound handler signature: receives the RFC-003 envelope as a dict.
InboundHandler = Callable[[Dict[str, Any]], None]


class CustomMessageDispatcher:
    """Routes RFC-003 envelopes between the kernel and the extension.

    Outbound (``emit``) forwards a validated envelope as an IOPub
    ``display_data`` / ``update_display_data`` (Family A) or a Comm
    ``comm_msg`` (Families B-E). Inbound (``_on_comm_msg``) validates
    inbound envelopes and dispatches to per-message-type handlers
    registered via :meth:`register_handler`. Holds a regular reference
    to the kernel; ``LLMKernel.do_shutdown`` MUST call :meth:`stop`.
    """

    def __init__(
        self,
        kernel: "IPythonKernel",
        comm_target: str = DEFAULT_COMM_TARGET,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        """Bind to ``kernel``; ``comm_target`` defaults to ``llmnb.rts.v1``."""
        self._kernel = kernel
        self._comm_target: str = comm_target
        self._buffer_size: int = buffer_size
        self._lock: threading.RLock = threading.RLock()
        self._active_comm: Any = None
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=buffer_size)
        self._handlers: Dict[str, List[InboundHandler]] = {}
        self._started: bool = False

    def start(self) -> None:
        """Register the Comm target on the kernel. Idempotent."""
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
        """Close the active Comm and unregister the target. Idempotent."""
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
        """Validate and route an outbound RFC-003 envelope.

        Family A rides IOPub ``display_data`` / ``update_display_data``
        keyed by ``transient.display_id == correlation_id``. The other
        message types ride the Comm; if no Comm is attached, the
        envelope is buffered (oldest dropped beyond ``buffer_size``).
        Raises ``ValueError`` if validation fails (no IOPub emission).
        """
        validate_envelope(envelope)  # RFC-003 F1.
        message_type = envelope["message_type"]
        if message_type in _RUN_LIFECYCLE_TYPES:
            self._emit_run_lifecycle(envelope)
            return
        with self._lock:
            comm = self._active_comm
            if comm is None:
                if len(self._buffer) >= self._buffer_size:
                    dropped = self._buffer.popleft()
                    logger.warning(
                        "dispatcher buffer overflow; dropped oldest "
                        "(message_type=%s, correlation_id=%s)",
                        dropped.get("message_type"), dropped.get("correlation_id"),
                    )
                self._buffer.append(envelope)
                return
        self._send_via_comm(comm, envelope)

    def register_handler(
        self, message_type: str, handler: InboundHandler
    ) -> Callable[[], None]:
        """Register an inbound handler for ``message_type``.

        Multiple handlers per type are supported; each receives a copy
        of the envelope. Returns a dispose function that removes this
        specific handler. Raises ``ValueError`` if ``message_type`` is
        outside the RFC-003 catalog.
        """
        if message_type not in RFC003_MESSAGE_TYPES:
            raise ValueError(
                f"message_type {message_type!r} not in RFC-003 v1.0.0 catalog"
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
        flushes any envelopes buffered before the attach.
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
        for envelope in buffered:
            self._send_via_comm(comm, envelope)

    def _on_comm_msg(self, msg: Dict[str, Any]) -> None:
        """Handle an inbound ``comm_msg`` from the extension.

        Validates, then fans out to every registered handler for its
        ``message_type``. Unknown types drop with a warning per RFC-003
        F2 (V1 fail-closed).
        """
        try:
            envelope = msg["content"]["data"]
        except (KeyError, TypeError):
            logger.warning("inbound comm_msg missing content.data; dropped")
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
                "inbound message_type=%s has no registered handlers; dropped "
                "(RFC-003 F2 fail-closed)", message_type,
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
        """Emit a Family A envelope as an IOPub display message."""
        payload: Dict[str, Any] = envelope["payload"]
        message_type: str = envelope["message_type"]
        # display_id == run_id (== payload.id for run.start, payload.run_id for the rest).
        display_id: str = (
            payload.get("id") if message_type == "run.start" else payload.get("run_id")
        ) or envelope["correlation_id"]
        msg_type = "display_data" if message_type == "run.start" else "update_display_data"
        content: Dict[str, Any] = {
            "data": {MIME_RUN: payload, MIME_ENVELOPE: envelope},
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

    def _send_via_comm(self, comm: Any, envelope: Dict[str, Any]) -> None:
        """Send an envelope through the active Comm."""
        try:
            comm.send(envelope)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "comm.send raised for message_type=%s; envelope dropped",
                envelope.get("message_type"),
            )


__all__ = [
    "CustomMessageDispatcher",
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_COMM_TARGET",
    "InboundHandler",
    "MIME_ENVELOPE",
    "MIME_RUN",
]

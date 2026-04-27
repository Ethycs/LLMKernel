"""Python ``logging.Handler`` that emits OTLP/JSON LogRecords (RFC-008 §7).

Installed by :mod:`llm_kernel.pty_mode` on the root logger and on every
``llm_kernel.*`` child. Once installed it MUST be the only sink that
reaches stdout/the data plane: any pre-existing :class:`StreamHandler`
that writes to ``sys.stdout`` is removed (kernel boot output already
went to PTY ``stderr`` directly per RFC-008 §3 "Boot output").

Severity mapping (RFC-008 §7):

==========  ==================
Python      OTel severityNumber
----------  ------------------
DEBUG       5  (TRACE)
INFO        9
WARNING     13 (WARN)
ERROR       17
CRITICAL    21 (FATAL)
==========  ==================

The handler is lock-free above the underlying :class:`SocketWriter`; the
writer's own producer-side lock serializes wire bytes. ``emit`` swallows
its own exceptions per :meth:`logging.Handler.handleError`'s contract so
a bad LogRecord never crashes the kernel main loop.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from ._attrs import encode_attrs

if TYPE_CHECKING:  # pragma: no cover
    from .socket_writer import SocketWriter

#: Map ``logging`` levels to OTel ``severityNumber`` (RFC-008 §7).
SEVERITY_MAP: Dict[int, int] = {
    logging.DEBUG: 5,
    logging.INFO: 9,
    logging.WARNING: 13,
    logging.ERROR: 17,
    logging.CRITICAL: 21,
}


class OtlpDataPlaneHandler(logging.Handler):
    """Format Python LogRecords as OTLP/JSON LogRecords on the data plane.

    Constructed with a :class:`SocketWriter` that does the actual byte
    write. Optional ``extra_attributes`` are merged into every emitted
    record's ``attributes`` -- used by :mod:`llm_kernel.pty_mode` to
    stamp the session_id on every log line (RFC-008 §4 ready-handshake
    attribute persists).
    """

    def __init__(
        self,
        socket_writer: "SocketWriter",
        extra_attributes: Optional[Dict[str, Any]] = None,
        level: int = logging.NOTSET,
    ) -> None:
        super().__init__(level=level)
        self._writer = socket_writer
        self._extra_attributes: Dict[str, Any] = dict(extra_attributes or {})

    # The handler's own emit MUST NOT call back into the logging
    # subsystem on failure (the kernel logger configuration would
    # recurse). Use :meth:`logging.Handler.handleError` semantics.
    def emit(self, record: logging.LogRecord) -> None:
        try:
            otlp = self._build_record(record)
            self._writer.write_frame(otlp)
        except Exception:  # pragma: no cover - defensive
            self.handleError(record)

    # Public for tests; the `pty_mode` smoke validates the wire shape.
    def _build_record(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Shape one Python LogRecord as OTLP/JSON.

        ``timeUnixNano`` is the ``record.created`` epoch in ns;
        ``observedTimeUnixNano`` is wall-clock at emission. Per OTLP/JSON
        the int64-shaped fields are JSON strings.
        """
        attrs: Dict[str, Any] = {
            "logger.name": record.name,
            "code.function": record.funcName,
            "code.lineno": record.lineno,
        }
        # Stamp extra attributes (e.g., session_id) AFTER the per-record
        # ones so callers can override `code.*` in tests if they want.
        attrs.update(self._extra_attributes)
        return {
            "timeUnixNano": str(int(record.created * 1e9)),
            "observedTimeUnixNano": str(time.time_ns()),
            "severityNumber": SEVERITY_MAP.get(record.levelno, 9),
            "severityText": record.levelname,
            "body": {"stringValue": self.format(record)},
            "attributes": encode_attrs(attrs),
        }


__all__ = ["OtlpDataPlaneHandler", "SEVERITY_MAP"]

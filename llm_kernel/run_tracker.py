"""LLMKernel run-tracker (Stage 2 Track B2).

Captures every kernel-mediated interaction (MCP tool call, LiteLLM model
call) as a LangSmith-shaped run record and emits RFC-003 run-lifecycle
envelopes via a pluggable :class:`Sink`. The production sink is the
Track B3 :class:`llm_kernel.custom_messages.CustomMessageDispatcher`,
which serializes envelopes onto Jupyter messaging (``display_data`` /
``update_display_data`` keyed on ``display_id``); tests use a list-sink.
Sources: chapter 07 §Chat flow JSON, chapter 08 (DR-0015), RFC-003
Family A. Thread-safe via :class:`threading.RLock`.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, Iterator, List, Literal, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .run_envelope import RFC003_VERSION, make_envelope, validate_envelope

logger: logging.Logger = logging.getLogger("llm_kernel.run_tracker")

#: LangSmith ``run_type`` enum (chapter 07 §Chat flow JSON).
RunType = Literal["llm", "tool", "chain", "retriever", "agent", "embedding"]
#: RFC-003 ``run.event`` ``event_type`` enum.
EventType = Literal["token", "tool_call", "tool_result", "log", "error"]
#: RFC-003 ``run.complete`` ``status`` enum.
RunStatus = Literal["success", "error", "timeout"]


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class RunEvent(BaseModel):
    """One incremental update against an open run record (RFC-003 ``run.event``)."""

    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class RunRecord(BaseModel):
    """LangSmith-shaped run record persisted into ``metadata.rts.event_log``."""

    model_config = ConfigDict(extra="forbid")

    id: str
    trace_id: str
    parent_run_id: Optional[str] = None
    name: str
    run_type: RunType
    start_time: str
    end_time: Optional[str] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Optional[Dict[str, Any]] = None
    events: List[RunEvent] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None


class Sink(Protocol):
    """Callback the run-tracker uses to publish envelopes downstream.

    The production implementation is
    :class:`llm_kernel.custom_messages.CustomMessageDispatcher`, which
    serializes envelopes onto Jupyter messaging; tests use a
    list-appending shim.
    """

    def emit(self, envelope: Dict[str, Any]) -> None:  # pragma: no cover
        """Publish an RFC-003 envelope downstream."""
        ...


class RunTracker:
    """Thread-safe LangSmith-shaped recorder + RFC-003 envelope emitter."""

    def __init__(
        self, trace_id: str, sink: Sink,
        agent_id: Optional[str] = None, zone_id: Optional[str] = None,
    ) -> None:
        """Create a tracker bound to one trace; ``sink`` receives envelopes.

        ``sink`` is typically a
        :class:`llm_kernel.custom_messages.CustomMessageDispatcher` in
        production and a list-sink in tests. ``agent_id`` / ``zone_id``
        are folded into every run's ``metadata``.
        """
        self.trace_id: str = trace_id
        self.sink: Sink = sink
        self.agent_id: Optional[str] = agent_id
        self.zone_id: Optional[str] = zone_id
        self._lock: threading.RLock = threading.RLock()
        self._runs: Dict[str, RunRecord] = {}
        self._order: List[str] = []

    def start_run(
        self,
        name: str,
        run_type: str,
        inputs: Dict[str, Any],
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Open a run record and emit a ``run.start`` envelope.

        Returns the new UUIDv4 run id. Callers MUST pass it back to
        :meth:`event`, :meth:`complete_run`, or :meth:`fail_run`.
        # TODO(B5): the LiteLLM proxy will call this with run_type="llm"
        # for every model call.
        """
        run_id = str(uuid.uuid4())
        merged: Dict[str, Any] = dict(metadata or {})
        if self.agent_id is not None:
            merged.setdefault("agent_id", self.agent_id)
        if self.zone_id is not None:
            merged.setdefault("zone_id", self.zone_id)

        record = RunRecord(
            id=run_id, trace_id=self.trace_id, parent_run_id=parent_run_id,
            name=name, run_type=run_type,  # type: ignore[arg-type]
            start_time=_utc_now_iso(), inputs=dict(inputs or {}),
            tags=list(tags or []), metadata=merged,
        )
        with self._lock:
            self._runs[run_id] = record
            self._order.append(run_id)
        payload: Dict[str, Any] = {
            "id": record.id, "trace_id": record.trace_id,
            "parent_run_id": record.parent_run_id, "name": record.name,
            "run_type": record.run_type, "start_time": record.start_time,
            "inputs": record.inputs, "tags": record.tags,
            "metadata": record.metadata,
        }
        self._emit(make_envelope("run.start", payload, correlation_id=run_id))
        return run_id

    def event(self, run_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Append a :class:`RunEvent` to ``run_id`` and emit ``run.event``.

        Raises ``KeyError`` if ``run_id`` does not name an open run.
        """
        timestamp = _utc_now_iso()
        ev = RunEvent(event_type=event_type, data=dict(data or {}), timestamp=timestamp)  # type: ignore[arg-type]
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise KeyError(f"unknown run_id {run_id!r}; did you call start_run?")
            record.events.append(ev)
        payload: Dict[str, Any] = {
            "run_id": run_id, "event_type": event_type,
            "data": ev.data, "timestamp": timestamp,
        }
        self._emit(make_envelope("run.event", payload, correlation_id=run_id))

    def complete_run(
        self, run_id: str, outputs: Dict[str, Any], status: str = "success"
    ) -> None:
        """Mark ``run_id`` complete; emits ``run.complete`` with ``status``.

        Raises ``KeyError`` if ``run_id`` does not name an open run.
        """
        end_time = _utc_now_iso()
        outs = dict(outputs or {})
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise KeyError(f"unknown run_id {run_id!r}; did you call start_run?")
            record.end_time = end_time
            record.outputs = outs
        payload: Dict[str, Any] = {"run_id": run_id, "end_time": end_time,
            "outputs": outs, "error": None, "status": status}
        self._emit(make_envelope("run.complete", payload, correlation_id=run_id))

    def fail_run(
        self, run_id: str, error: Dict[str, Any], status: str = "error"
    ) -> None:
        """Close ``run_id`` with an error payload; emits ``run.complete``.

        ``error`` should carry ``kind`` / ``message`` / ``traceback`` per
        RFC-003 §run.complete. Raises ``KeyError`` if ``run_id`` is unknown.
        """
        end_time = _utc_now_iso()
        err = dict(error or {})
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise KeyError(f"unknown run_id {run_id!r}; did you call start_run?")
            record.end_time = end_time
            record.error = err
            if record.outputs is None:
                record.outputs = {}
            outs = dict(record.outputs)
        payload: Dict[str, Any] = {"run_id": run_id, "end_time": end_time,
            "outputs": outs, "error": err, "status": status}
        self._emit(make_envelope("run.complete", payload, correlation_id=run_id))

    def get_run(self, run_id: str) -> RunRecord:
        """Return the :class:`RunRecord` for ``run_id``; raises KeyError if unknown."""
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise KeyError(f"unknown run_id {run_id!r}")
            return record

    def iter_runs(self) -> Iterator[RunRecord]:
        """Yield run records in insertion order (open and closed alike)."""
        with self._lock:
            records = [self._runs[rid] for rid in list(self._order)]
        for record in records:
            yield record

    def event_log_jsonl(self) -> str:
        """Return the append-only log as JSON-lines (per chapter 07 §Chat flow JSON).

        One JSON object per line, insertion order; the resulting string is
        what gets persisted under ``metadata.rts.event_log`` in ``.llmnb``.
        """
        buf = StringIO()
        for record in self.iter_runs():
            buf.write(record.model_dump_json())
            buf.write("\n")
        return buf.getvalue()

    def _emit(self, envelope: Dict[str, Any]) -> None:
        """Validate ``envelope`` against RFC-003 then forward to the sink.

        The production sink (Track B3 ``CustomMessageDispatcher``)
        writes this envelope onto Jupyter messaging with
        ``display_id == correlation_id``.
        """
        validate_envelope(envelope)  # RFC-003 §Failure modes F1.
        try:
            self.sink.emit(envelope)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "run-tracker sink raised; envelope dropped",
                extra={"message_type": envelope.get("message_type"),
                    "correlation_id": envelope.get("correlation_id"),
                    "rfc_version": RFC003_VERSION},
            )


__all__ = ["EventType", "RunEvent", "RunRecord", "RunStatus", "RunTracker", "RunType", "Sink"]

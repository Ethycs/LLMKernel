"""LLMKernel run-tracker (Stage 2 Track B2; refactor R1-K to OTLP/JSON).

Captures every kernel-mediated interaction (MCP tool call, LiteLLM model
call) as an OTLP/JSON span and emits RFC-003 run-lifecycle envelopes via
a pluggable :class:`Sink`. The production sink is the Track B3
:class:`llm_kernel.custom_messages.CustomMessageDispatcher`, which
serializes envelopes onto Jupyter messaging (``display_data`` /
``update_display_data`` keyed on ``display_id``); tests use a list-sink.

Span shape follows the OpenTelemetry OTLP/JSON encoding:

* ids are lowercase hex (``spanId`` 16 chars / ``traceId`` 32 chars)
* timestamps are JSON-string decimal nanoseconds (``startTimeUnixNano``)
* attributes are an OTLP attribute list (see :mod:`._attrs`)
* status is ``{code: "STATUS_CODE_OK|ERROR|UNSET", message: ""}``
* in-progress spans persist with ``endTimeUnixNano: null`` and
  ``status.code: "STATUS_CODE_UNSET"``

Sources: chapter 07 §Chat flow JSON, chapter 08 (DR-0015), RFC-003
Family A. Thread-safe via :class:`threading.RLock`. The merged-span
form persisted into ``metadata.rts.event_log`` is downstream work
(RFC-005); this module emits the live envelopes.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
import uuid
from io import StringIO
from typing import Any, Dict, Iterator, List, Literal, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ._attrs import decode_attrs, encode_attrs
from .run_envelope import RFC003_VERSION, make_envelope, validate_envelope

logger: logging.Logger = logging.getLogger("llm_kernel.run_tracker")

#: Run-type enum stored in ``attributes["llmnb.run_type"]`` per the
#: OTLP refactor.  Adds ``agent_emit`` per RFC-002 §"Process lifecycle"
#: 3 / RFC-005 §"`agent_emit` runs": every byte of agent output that
#: bypassed structured channels (prose, reasoning, system messages,
#: stderr, malformed tool-use) lands as an ``agent_emit`` span.
RunType = Literal[
    "llm", "tool", "chain", "retriever", "agent", "embedding", "agent_emit",
]
#: RFC-003 ``run.event`` ``event_type`` enum (used as the OTLP event
#: ``name`` field).
EventType = Literal["token", "tool_call", "tool_result", "log", "error"]
#: RFC-003 ``run.complete`` ``status`` enum (legacy LangSmith form);
#: maps to OTel ``status.code`` via :func:`_to_otel_status`.
RunStatus = Literal["success", "error", "timeout"]

#: OTel canonical span-status codes per the OTLP/JSON spec.
_STATUS_OK: str = "STATUS_CODE_OK"
_STATUS_ERROR: str = "STATUS_CODE_ERROR"
_STATUS_UNSET: str = "STATUS_CODE_UNSET"

#: Default span kind for V1.  Internal because every span emitted from
#: the kernel models a kernel-mediated operation, not an external RPC
#: server hop.
_SPAN_KIND_INTERNAL: str = "SPAN_KIND_INTERNAL"


def _now_unix_nano_str() -> str:
    """Return the current wall-clock time as a JSON-string of decimal nanoseconds.

    OTLP/JSON encodes 64-bit timestamps as strings to preserve precision
    across JSON's 53-bit number range.  Resolution is the OS clock's
    nanosecond resolution (``time.time_ns``).
    """
    return str(time.time_ns())


def _new_span_id() -> str:
    """Return a fresh 16-lowercase-hex OTLP spanId (8 random bytes)."""
    return secrets.token_hex(8)


def _new_trace_id() -> str:
    """Return a fresh 32-lowercase-hex OTLP traceId (16 random bytes)."""
    return secrets.token_hex(16)


def _coerce_trace_id(value: Optional[str]) -> str:
    """Coerce ``value`` (UUID or hex) to a 32-lowercase-hex traceId.

    Accepts ``None`` (allocates fresh), a UUID4 string (strips dashes),
    or an existing 32-hex string (passed through lowercased).  Anything
    else falls back to a fresh traceId so the kernel never emits a
    malformed OTLP id.
    """
    if not value:
        return _new_trace_id()
    s = value.strip().lower()
    # UUID with dashes -> hex without dashes (32 chars).
    try:
        return uuid.UUID(s).hex
    except (ValueError, AttributeError):
        pass
    if len(s) == 32 and all(c in "0123456789abcdef" for c in s):
        return s
    logger.warning("malformed trace_id %r; allocating fresh OTLP traceId", value)
    return _new_trace_id()


def _to_otel_status(status: str) -> Dict[str, str]:
    """Map a legacy ``RunStatus`` (or OTel code) to an OTel status object."""
    if status in (_STATUS_OK, _STATUS_ERROR, _STATUS_UNSET):
        return {"code": status, "message": ""}
    if status == "success":
        return {"code": _STATUS_OK, "message": ""}
    if status in ("error", "timeout"):
        return {"code": _STATUS_ERROR, "message": ""}
    return {"code": _STATUS_UNSET, "message": ""}


class SpanEvent(BaseModel):
    """One incremental update against an open span (OTLP ``Span.Event``)."""

    model_config = ConfigDict(extra="forbid")

    timeUnixNano: str
    name: str
    attributes: List[Dict[str, Any]] = Field(default_factory=list)


class Span(BaseModel):
    """OTLP/JSON-shaped span persisted into the run log.

    Mirrors the on-wire OTLP ``Span`` message: 16-byte ``traceId`` /
    8-byte ``spanId`` as lowercase hex, decimal-nanosecond timestamps
    as JSON strings, attributes as OTLP key/value pairs, status as the
    OTel canonical object.  ``endTimeUnixNano`` is ``None`` while the
    span is in progress; ``status.code`` is ``STATUS_CODE_UNSET`` until
    :meth:`RunTracker.complete_run` or :meth:`RunTracker.fail_run` is
    called.
    """

    model_config = ConfigDict(extra="forbid")

    traceId: str
    spanId: str
    parentSpanId: Optional[str] = None
    name: str
    kind: str = _SPAN_KIND_INTERNAL
    startTimeUnixNano: str
    endTimeUnixNano: Optional[str] = None
    attributes: List[Dict[str, Any]] = Field(default_factory=list)
    events: List[SpanEvent] = Field(default_factory=list)
    links: List[Dict[str, Any]] = Field(default_factory=list)
    status: Dict[str, str] = Field(
        default_factory=lambda: {"code": _STATUS_UNSET, "message": ""}
    )


# Backwards-compatible alias: callers and tests import ``RunRecord`` from
# this module.  Post-refactor ``RunRecord`` is an OTLP ``Span``; the
# alias keeps the import path stable while the field shape changes.
RunRecord = Span
RunEvent = SpanEvent


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
    """Thread-safe OTLP-shaped recorder + RFC-003 envelope emitter."""

    def __init__(
        self, trace_id: str, sink: Sink,
        agent_id: Optional[str] = None, zone_id: Optional[str] = None,
    ) -> None:
        """Create a tracker bound to one trace; ``sink`` receives envelopes.

        ``trace_id`` is coerced to a 32-lowercase-hex OTLP traceId
        (UUID input is accepted and stripped of dashes).  ``sink`` is
        typically a :class:`llm_kernel.custom_messages.CustomMessageDispatcher`
        in production and a list-sink in tests.  ``agent_id`` /
        ``zone_id`` are folded into every span's attributes.
        """
        self.trace_id: str = _coerce_trace_id(trace_id)
        self.sink: Sink = sink
        self.agent_id: Optional[str] = agent_id
        self.zone_id: Optional[str] = zone_id
        self._lock: threading.RLock = threading.RLock()
        self._spans: Dict[str, Span] = {}
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
        """Open a span and emit a ``run.start`` envelope.

        Returns the new ``spanId`` (16 lowercase hex chars).  Callers
        MUST pass it back to :meth:`event`, :meth:`complete_run`, or
        :meth:`fail_run`.

        The legacy LangSmith fields (``run_type``, ``inputs``, ``tags``,
        ``metadata``, ``parent_run_id``) are mapped onto OTLP attributes
        and structural fields per the R1-K migration table:

        * ``run_type`` -> ``attributes["llmnb.run_type"]``
        * ``inputs`` (dict) -> ``input.value`` (JSON) + ``input.mime_type``
        * ``tags`` (list) -> ``attributes["llmnb.tags"]`` (arrayValue)
        * ``metadata.{agent,zone,cell}_id`` -> ``llmnb.{agent,zone,cell}_id``
        * ``parent_run_id`` -> ``parentSpanId``
        """
        span_id = _new_span_id()
        start_ns = _now_unix_nano_str()

        # Build the attribute dict in plain Python; encode_attrs at the
        # end produces the OTLP wire shape.
        attrs: Dict[str, Any] = {"llmnb.run_type": run_type}

        if inputs:
            try:
                import json
                attrs["input.value"] = json.dumps(inputs, default=str)
            except (TypeError, ValueError):
                attrs["input.value"] = str(inputs)
            attrs["input.mime_type"] = "application/json"

        merged_meta: Dict[str, Any] = dict(metadata or {})
        if self.agent_id is not None:
            merged_meta.setdefault("agent_id", self.agent_id)
        if self.zone_id is not None:
            merged_meta.setdefault("zone_id", self.zone_id)
        for key in ("agent_id", "zone_id", "cell_id"):
            if key in merged_meta:
                attrs[f"llmnb.{key}"] = merged_meta.pop(key)
        # Specific RFC-005 attributes get hoisted directly under the
        # ``llmnb.*`` namespace (not wrapped in ``llmnb.metadata.*``)
        # so the operator surface and the renderer can dispatch on
        # them without de-nesting.  Currently: the agent_emit family
        # per RFC-005 §"`agent_emit` runs".
        for key in ("emit_kind", "emit_content", "parser_diagnostic"):
            if key in merged_meta:
                attrs[f"llmnb.{key}"] = merged_meta.pop(key)
        # Any remaining metadata is preserved under llmnb.metadata.<key>
        # so callers don't lose context (touched_files / log_signature /
        # jsonrpc_id / etc are surfaced this way).
        for key, value in merged_meta.items():
            attrs[f"llmnb.metadata.{key}"] = value

        if tags:
            attrs["llmnb.tags"] = list(tags)

        span = Span(
            traceId=self.trace_id,
            spanId=span_id,
            parentSpanId=parent_run_id,
            name=name,
            kind=_SPAN_KIND_INTERNAL,
            startTimeUnixNano=start_ns,
            endTimeUnixNano=None,
            attributes=encode_attrs(attrs),
            events=[],
            links=[],
            status={"code": _STATUS_UNSET, "message": ""},
        )
        with self._lock:
            self._spans[span_id] = span
            self._order.append(span_id)

        # The envelope payload IS the span shell at this point: an
        # in-progress OTLP span with null endTimeUnixNano.
        payload: Dict[str, Any] = span.model_dump()
        self._emit(make_envelope("run.start", payload, correlation_id=span_id))
        return span_id

    def event(self, run_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Append a :class:`SpanEvent` to ``run_id`` and emit ``run.event``.

        ``event_type`` is the OTel event ``name``; ``data`` is encoded
        as the event's ``attributes``.  Raises ``KeyError`` if
        ``run_id`` does not name an open span.
        """
        ts = _now_unix_nano_str()
        ev = SpanEvent(
            timeUnixNano=ts,
            name=event_type,
            attributes=encode_attrs(dict(data or {})),
        )
        with self._lock:
            span = self._spans.get(run_id)
            if span is None:
                raise KeyError(
                    f"unknown spanId {run_id!r}; did you call start_run?"
                )
            span.events.append(ev)
        payload: Dict[str, Any] = {
            "spanId": run_id,
            "event": ev.model_dump(),
        }
        self._emit(make_envelope("run.event", payload, correlation_id=run_id))

    def complete_run(
        self, run_id: str, outputs: Dict[str, Any],
        status: str = _STATUS_OK,
    ) -> None:
        """Mark ``run_id`` complete; emits ``run.complete`` with OTel status.

        ``status`` accepts both legacy LangSmith strings (``"success"`` /
        ``"error"`` / ``"timeout"``) and OTel codes
        (``STATUS_CODE_OK`` / ``STATUS_CODE_ERROR`` /
        ``STATUS_CODE_UNSET``); they are normalized via
        :func:`_to_otel_status`.  ``outputs`` (a dict) is folded onto
        the span's attributes as ``output.value`` (JSON) +
        ``output.mime_type``.  Raises ``KeyError`` if ``run_id`` is
        unknown.
        """
        end_ns = _now_unix_nano_str()
        otel_status = _to_otel_status(status)
        with self._lock:
            span = self._spans.get(run_id)
            if span is None:
                raise KeyError(
                    f"unknown spanId {run_id!r}; did you call start_run?"
                )
            span.endTimeUnixNano = end_ns
            span.status = otel_status
            attrs = decode_attrs(list(span.attributes))
            if outputs:
                try:
                    import json
                    attrs["output.value"] = json.dumps(outputs, default=str)
                except (TypeError, ValueError):
                    attrs["output.value"] = str(outputs)
                attrs["output.mime_type"] = "application/json"
            span.attributes = encode_attrs(attrs)

        payload: Dict[str, Any] = {
            "spanId": run_id,
            "endTimeUnixNano": end_ns,
            "status": otel_status,
            "attributes": list(span.attributes),
        }
        self._emit(make_envelope("run.complete", payload, correlation_id=run_id))

    def fail_run(
        self, run_id: str, error: Dict[str, Any],
        status: str = _STATUS_ERROR,
    ) -> None:
        """Close ``run_id`` with an error payload; emits ``run.complete``.

        ``error`` may use the legacy LangSmith schema (``kind`` /
        ``message`` / ``traceback``) or the OTel exception semconv
        (``exception.type`` / ``exception.message`` /
        ``exception.stacktrace``); either is normalized onto the span's
        attributes per the OTel exception conventions.  Raises
        ``KeyError`` if ``run_id`` is unknown.
        """
        end_ns = _now_unix_nano_str()
        otel_status = _to_otel_status(status)

        # Map legacy LangSmith error keys onto OTel exception semconv.
        err = dict(error or {})
        exc_type = err.pop("exception.type", None) or err.pop("kind", None) or ""
        exc_msg = err.pop("exception.message", None) or err.pop("message", None) or ""
        exc_trace = (
            err.pop("exception.stacktrace", None)
            or err.pop("traceback", None)
            or err.pop("stacktrace", None)
            or ""
        )

        with self._lock:
            span = self._spans.get(run_id)
            if span is None:
                raise KeyError(
                    f"unknown spanId {run_id!r}; did you call start_run?"
                )
            span.endTimeUnixNano = end_ns
            otel_status = {"code": _STATUS_ERROR, "message": str(exc_msg) or ""}
            span.status = otel_status
            attrs = decode_attrs(list(span.attributes))
            if exc_type:
                attrs["exception.type"] = exc_type
            if exc_msg:
                attrs["exception.message"] = exc_msg
            if exc_trace:
                attrs["exception.stacktrace"] = exc_trace
            for key, value in err.items():
                # Anything left after stripping the exception keys lands
                # under llmnb.error.* so it's not silently dropped.
                attrs[f"llmnb.error.{key}"] = value
            span.attributes = encode_attrs(attrs)

        payload: Dict[str, Any] = {
            "spanId": run_id,
            "endTimeUnixNano": end_ns,
            "status": otel_status,
            "attributes": list(span.attributes),
        }
        self._emit(make_envelope("run.complete", payload, correlation_id=run_id))

    def get_run(self, run_id: str) -> Span:
        """Return the :class:`Span` for ``run_id``; raises KeyError if unknown."""
        with self._lock:
            span = self._spans.get(run_id)
            if span is None:
                raise KeyError(f"unknown spanId {run_id!r}")
            return span

    def iter_runs(self) -> Iterator[Span]:
        """Yield spans in insertion order (open and closed alike)."""
        with self._lock:
            spans = [self._spans[rid] for rid in list(self._order)]
        for span in spans:
            yield span

    def event_log_jsonl(self) -> str:
        """Return the append-only span log as JSON-lines.

        One OTLP span per line, insertion order.  This is the
        persistent form (RFC-005 ``.llmnb`` event log uses the same
        merged-span shape).
        """
        buf = StringIO()
        for span in self.iter_runs():
            buf.write(span.model_dump_json())
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


__all__ = [
    "EventType", "RunEvent", "RunRecord", "RunStatus", "RunTracker",
    "RunType", "Sink", "Span", "SpanEvent",
]

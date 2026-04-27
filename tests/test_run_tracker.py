"""Stage 2 Track B2 contract tests for the run-tracker (OTLP/JSON form).

These pytest tests exercise :class:`llm_kernel.run_tracker.RunTracker`
through an in-memory list-sink, asserting:

* RFC-003 envelope shapes for ``run.start`` / ``run.event`` /
  ``run.complete`` after the R1-K refactor (payloads now ride OTLP
  ``Span`` / ``SpanEvent`` shape).
* OTLP span updates (events list, ``endTimeUnixNano``, status codes,
  exception attributes).
* Thread safety of concurrent ``start_run`` calls.
* Round-trippability of the JSONL event log (one OTLP span per line).

The run-tracker is sync; we use :mod:`threading` for the concurrency case
and avoid pulling in pytest-asyncio for this module.

Run with::

    pixi run -e kernel pytest vendor/LLMKernel/tests/test_run_tracker.py -v
"""

from __future__ import annotations

import json
import re
import secrets
import threading
import uuid
from typing import Any, Dict, List

import pytest

from llm_kernel._attrs import decode_attrs
from llm_kernel.run_envelope import (
    DIRECTION_KERNEL_TO_EXTENSION,
    RFC003_VERSION,
    validate_envelope,
)
from llm_kernel.run_tracker import RunRecord, RunTracker, Span

#: 16-lowercase-hex regex (OTLP spanId).
_SPAN_ID_RE = re.compile(r"^[0-9a-f]{16}$")
#: 32-lowercase-hex regex (OTLP traceId).
_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class ListSink:
    """Trivial in-memory :class:`Sink` for assertions."""

    def __init__(self) -> None:
        self.envelopes: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def emit(self, envelope: Dict[str, Any]) -> None:
        with self._lock:
            self.envelopes.append(envelope)


def _new_tracker() -> tuple[RunTracker, ListSink]:
    sink = ListSink()
    tracker = RunTracker(
        # Tracker accepts UUID input and coerces to 32-hex traceId.
        trace_id=str(uuid.uuid4()),
        sink=sink,
        agent_id="alpha",
        zone_id="refactor",
    )
    return tracker, sink


def test_start_emits_run_start_envelope() -> None:
    """A ``run.start`` envelope MUST validate against RFC-003 §Envelope."""
    tracker, sink = _new_tracker()
    run_id = tracker.start_run(
        name="notify",
        run_type="tool",
        inputs={"observation": "ok", "importance": "info"},
        tags=["agent:alpha"],
    )

    # start_run returns the OTLP spanId (16 lowercase hex chars).
    assert _SPAN_ID_RE.match(run_id), f"expected 16-hex spanId, got {run_id!r}"

    assert len(sink.envelopes) == 1
    envelope = sink.envelopes[0]
    validate_envelope(envelope)

    assert envelope["message_type"] == "run.start"
    assert envelope["direction"] == DIRECTION_KERNEL_TO_EXTENSION
    assert envelope["correlation_id"] == run_id
    assert envelope["rfc_version"] == RFC003_VERSION

    payload = envelope["payload"]
    # OTLP span shape:
    assert payload["spanId"] == run_id
    assert _TRACE_ID_RE.match(payload["traceId"])
    assert payload["name"] == "notify"
    assert payload["parentSpanId"] is None
    assert payload["kind"] == "SPAN_KIND_INTERNAL"
    # In-progress spans carry null endTimeUnixNano + UNSET status.
    assert payload["endTimeUnixNano"] is None
    assert payload["status"]["code"] == "STATUS_CODE_UNSET"
    # startTimeUnixNano MUST be a JSON-string of decimal nanos
    # (preserves 64-bit precision per OTLP/JSON spec).
    assert isinstance(payload["startTimeUnixNano"], str)
    assert payload["startTimeUnixNano"].isdigit()

    # Decode attributes back to a plain dict to assert semconv keys.
    attrs = decode_attrs(payload["attributes"])
    assert attrs["llmnb.run_type"] == "tool"
    assert attrs["llmnb.agent_id"] == "alpha"
    assert attrs["llmnb.zone_id"] == "refactor"
    # ``inputs`` rides as input.value (JSON string) + input.mime_type.
    assert attrs["input.mime_type"] == "application/json"
    assert json.loads(attrs["input.value"]) == {
        "observation": "ok", "importance": "info",
    }
    assert attrs["llmnb.tags"] == ["agent:alpha"]


def test_event_appends_to_run_record_and_emits() -> None:
    """Two ``run.event`` envelopes MUST land and the span MUST grow."""
    tracker, sink = _new_tracker()
    run_id = tracker.start_run(
        name="notify", run_type="tool", inputs={"observation": "x", "importance": "info"}
    )

    tracker.event(run_id, "tool_call", {"tool": "notify", "arguments": {"x": 1}})
    tracker.event(run_id, "tool_result", {"tool": "notify", "result": {"acknowledged": True}})

    # 1 start + 2 events == 3 envelopes
    assert len(sink.envelopes) == 3
    for env in sink.envelopes:
        validate_envelope(env)

    event_envs = [e for e in sink.envelopes if e["message_type"] == "run.event"]
    assert len(event_envs) == 2
    assert all(e["correlation_id"] == run_id for e in event_envs)
    # Each run.event payload carries one OTLP SpanEvent under ``event``.
    assert event_envs[0]["payload"]["event"]["name"] == "tool_call"
    assert event_envs[1]["payload"]["event"]["name"] == "tool_result"

    span = tracker.get_run(run_id)
    assert isinstance(span, Span)
    assert len(span.events) == 2
    # timeUnixNano is a numeric string; monotonic ordering compares as int.
    assert int(span.events[0].timeUnixNano) <= int(span.events[1].timeUnixNano)


def test_complete_run_sets_end_time_and_outputs() -> None:
    """``complete_run`` MUST stamp endTimeUnixNano + status and emit run.complete."""
    tracker, sink = _new_tracker()
    run_id = tracker.start_run(name="notify", run_type="tool", inputs={})
    tracker.complete_run(run_id, outputs={"acknowledged": True})

    final = sink.envelopes[-1]
    validate_envelope(final)
    assert final["message_type"] == "run.complete"
    assert final["correlation_id"] == run_id
    payload = final["payload"]
    # OTel canonical status code (the legacy ``"success"`` is mapped).
    assert payload["status"]["code"] == "STATUS_CODE_OK"
    assert payload["spanId"] == run_id
    assert isinstance(payload["endTimeUnixNano"], str)
    assert payload["endTimeUnixNano"].isdigit()
    # ``outputs`` rides as output.value / output.mime_type attributes.
    attrs = decode_attrs(payload["attributes"])
    assert attrs["output.mime_type"] == "application/json"
    assert json.loads(attrs["output.value"]) == {"acknowledged": True}

    span = tracker.get_run(run_id)
    assert span.endTimeUnixNano is not None
    assert span.status["code"] == "STATUS_CODE_OK"


def test_fail_run_emits_complete_with_error_status() -> None:
    """``fail_run`` MUST flag status=ERROR and surface OTel exception attrs."""
    tracker, sink = _new_tracker()
    run_id = tracker.start_run(name="ask", run_type="tool", inputs={"question": "?"})
    err = {
        "exception.type": "TimeoutError",
        "exception.message": "operator did not respond",
        "exception.stacktrace": "",
    }
    tracker.fail_run(run_id, error=err)

    final = sink.envelopes[-1]
    validate_envelope(final)
    assert final["message_type"] == "run.complete"
    payload = final["payload"]
    assert payload["status"]["code"] == "STATUS_CODE_ERROR"
    # status.message echoes the exception message per OTel semconv.
    assert "operator did not respond" in payload["status"]["message"]

    span = tracker.get_run(run_id)
    attrs = decode_attrs(span.attributes)
    assert attrs["exception.type"] == "TimeoutError"
    assert attrs["exception.message"] == "operator did not respond"
    assert span.endTimeUnixNano is not None


def test_fail_run_accepts_legacy_kind_message_keys() -> None:
    """fail_run MUST also accept the legacy ``kind`` / ``message`` schema.

    Production callers such as the agent supervisor still pass the
    LangSmith-shaped error envelope; the tracker normalizes onto OTel
    ``exception.*`` attribute keys per the R1-K migration table.
    """
    tracker, sink = _new_tracker()
    run_id = tracker.start_run(name="ask", run_type="tool", inputs={})
    tracker.fail_run(
        run_id,
        error={"kind": "Cancelled", "message": "interrupted", "traceback": "frames"},
    )
    span = tracker.get_run(run_id)
    attrs = decode_attrs(span.attributes)
    assert attrs["exception.type"] == "Cancelled"
    assert attrs["exception.message"] == "interrupted"
    assert attrs["exception.stacktrace"] == "frames"
    assert span.status["code"] == "STATUS_CODE_ERROR"


def test_concurrent_starts_get_unique_ids() -> None:
    """Threaded ``start_run`` MUST allocate distinct OTLP spanIds."""
    tracker, sink = _new_tracker()
    barrier = threading.Barrier(8)
    ids: List[str] = []
    ids_lock = threading.Lock()

    def worker(idx: int) -> None:
        barrier.wait()
        rid = tracker.start_run(
            name=f"worker-{idx}",
            run_type="chain",
            inputs={"i": idx},
        )
        with ids_lock:
            ids.append(rid)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(ids) == 8
    assert len(set(ids)) == 8
    # One ``run.start`` envelope per starter, all RFC-003-valid.
    assert len(sink.envelopes) == 8
    for env in sink.envelopes:
        validate_envelope(env)
        assert env["message_type"] == "run.start"
        # correlation_id is the OTLP spanId (16 lowercase hex chars).
        assert _SPAN_ID_RE.match(env["correlation_id"])


def test_event_log_jsonl_is_round_trippable() -> None:
    """Each line of ``event_log_jsonl`` MUST parse as a Span-shaped JSON."""
    tracker, _sink = _new_tracker()
    rid_a = tracker.start_run(name="notify", run_type="tool", inputs={"x": 1})
    tracker.event(rid_a, "log", {"message": "midway"})
    tracker.complete_run(rid_a, outputs={"acknowledged": True})

    rid_b = tracker.start_run(name="ask", run_type="tool", inputs={"q": "?"})
    tracker.fail_run(
        rid_b,
        error={"exception.type": "Cancelled",
               "exception.message": "interrupted",
               "exception.stacktrace": ""},
    )

    log = tracker.event_log_jsonl()
    lines = [ln for ln in log.splitlines() if ln.strip()]
    assert len(lines) == 2

    parsed = [json.loads(ln) for ln in lines]
    # Round-trip via the pydantic model to assert the on-disk shape.
    spans = [RunRecord.model_validate(obj) for obj in parsed]
    assert {s.spanId for s in spans} == {rid_a, rid_b}

    finished = next(s for s in spans if s.spanId == rid_a)
    finished_attrs = decode_attrs(finished.attributes)
    assert finished_attrs["output.mime_type"] == "application/json"
    assert json.loads(finished_attrs["output.value"]) == {"acknowledged": True}
    assert finished.endTimeUnixNano is not None
    assert len(finished.events) == 1
    assert finished.events[0].name == "log"

    failed = next(s for s in spans if s.spanId == rid_b)
    failed_attrs = decode_attrs(failed.attributes)
    assert failed_attrs["exception.type"] == "Cancelled"
    assert failed.status["code"] == "STATUS_CODE_ERROR"


def test_event_against_unknown_run_raises() -> None:
    """``event`` against an unknown id MUST raise KeyError."""
    tracker, _ = _new_tracker()
    with pytest.raises(KeyError):
        tracker.event(secrets.token_hex(8), "log", {})


def test_run_tracker_with_dispatcher_sink() -> None:
    """A start/event/complete cycle MUST emit three IOPub messages
    when the run-tracker's sink is a real :class:`CustomMessageDispatcher`.

    Track B3 contract proof: B2's run-tracker, B3's dispatcher, and the
    kernel's IOPub seam are wired end-to-end.
    """
    from unittest.mock import MagicMock

    from llm_kernel.custom_messages import CustomMessageDispatcher

    sent: List[tuple] = []

    class _Session:
        def send(self, sock: Any, msg_type: str, **kwargs: Any) -> None:
            sent.append((msg_type, kwargs))

    class _CommMgr:
        def register_target(self, n: str, cb: Any) -> None: ...
        def unregister_target(self, n: str, cb: Any) -> None: ...

    kernel = MagicMock()
    kernel.session = _Session()
    kernel.iopub_socket = MagicMock()
    kernel.shell.comm_manager = _CommMgr()
    kernel._parent_header = {}

    dispatcher = CustomMessageDispatcher(kernel)
    tracker = RunTracker(
        trace_id=str(uuid.uuid4()), sink=dispatcher,
        agent_id="alpha", zone_id="refactor",
    )
    rid = tracker.start_run(name="notify", run_type="tool", inputs={"x": 1})
    tracker.event(rid, "tool_call", {"tool": "notify"})
    tracker.complete_run(rid, outputs={"acknowledged": True})

    assert [m for m, _ in sent] == [
        "display_data", "update_display_data", "update_display_data",
    ]
    for _msg, kwargs in sent:
        # display_id is the OTLP spanId, consistent across the lifecycle.
        assert kwargs["content"]["transient"] == {"display_id": rid}

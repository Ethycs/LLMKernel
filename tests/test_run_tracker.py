"""Stage 2 Track B2 contract tests for the run-tracker.

These pytest tests exercise :class:`llm_kernel.run_tracker.RunTracker`
through an in-memory list-sink, asserting:

* RFC-003 envelope shapes for ``run.start`` / ``run.event`` / ``run.complete``
* LangSmith run-record updates (events list, end_time, outputs, error)
* Thread safety of concurrent ``start_run`` calls
* Round-trippability of the JSONL event log

The run-tracker is sync; we use :mod:`threading` for the concurrency case
and avoid pulling in pytest-asyncio for this module.

Run with::

    pixi run -e kernel pytest vendor/LLMKernel/tests/test_run_tracker.py -v
"""

from __future__ import annotations

import json
import threading
import uuid
from typing import Any, Dict, List

import pytest

from llm_kernel.run_envelope import (
    DIRECTION_KERNEL_TO_EXTENSION,
    RFC003_VERSION,
    validate_envelope,
)
from llm_kernel.run_tracker import RunRecord, RunTracker


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

    assert len(sink.envelopes) == 1
    envelope = sink.envelopes[0]
    validate_envelope(envelope)

    assert envelope["message_type"] == "run.start"
    assert envelope["direction"] == DIRECTION_KERNEL_TO_EXTENSION
    assert envelope["correlation_id"] == run_id
    assert envelope["rfc_version"] == RFC003_VERSION

    payload = envelope["payload"]
    assert payload["id"] == run_id
    assert payload["name"] == "notify"
    assert payload["run_type"] == "tool"
    assert payload["inputs"] == {"observation": "ok", "importance": "info"}
    assert payload["parent_run_id"] is None
    assert payload["tags"] == ["agent:alpha"]
    # agent_id / zone_id from the tracker MUST appear in metadata.
    assert payload["metadata"]["agent_id"] == "alpha"
    assert payload["metadata"]["zone_id"] == "refactor"
    # Run-start payload MUST NOT carry end_time / outputs.
    assert "end_time" not in payload
    assert "outputs" not in payload


def test_event_appends_to_run_record_and_emits() -> None:
    """Two ``run.event`` envelopes MUST land and the record MUST grow."""
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
    assert event_envs[0]["payload"]["event_type"] == "tool_call"
    assert event_envs[1]["payload"]["event_type"] == "tool_result"

    record = tracker.get_run(run_id)
    assert isinstance(record, RunRecord)
    assert len(record.events) == 2
    # Timestamps MUST be monotonic (string-ordered ISO 8601).
    assert record.events[0].timestamp <= record.events[1].timestamp


def test_complete_run_sets_end_time_and_outputs() -> None:
    """``complete_run`` MUST stamp end_time/outputs and emit run.complete."""
    tracker, sink = _new_tracker()
    run_id = tracker.start_run(name="notify", run_type="tool", inputs={})
    tracker.complete_run(run_id, outputs={"acknowledged": True})

    final = sink.envelopes[-1]
    validate_envelope(final)
    assert final["message_type"] == "run.complete"
    assert final["correlation_id"] == run_id
    payload = final["payload"]
    assert payload["status"] == "success"
    assert payload["outputs"] == {"acknowledged": True}
    assert payload["end_time"] is not None
    assert payload["error"] is None

    record = tracker.get_run(run_id)
    assert record.end_time is not None
    assert record.outputs == {"acknowledged": True}
    assert record.error is None


def test_fail_run_emits_complete_with_error_status() -> None:
    """``fail_run`` MUST flag status=error and carry the error dict."""
    tracker, sink = _new_tracker()
    run_id = tracker.start_run(name="ask", run_type="tool", inputs={"question": "?"})
    err = {"kind": "TimeoutError", "message": "operator did not respond", "traceback": ""}
    tracker.fail_run(run_id, error=err)

    final = sink.envelopes[-1]
    validate_envelope(final)
    assert final["message_type"] == "run.complete"
    payload = final["payload"]
    assert payload["status"] == "error"
    assert payload["error"] == err

    record = tracker.get_run(run_id)
    assert record.error == err
    assert record.end_time is not None


def test_concurrent_starts_get_unique_ids() -> None:
    """Threaded ``start_run`` MUST allocate distinct UUIDv4 ids."""
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
        uuid.UUID(env["correlation_id"])


def test_event_log_jsonl_is_round_trippable() -> None:
    """Each line of ``event_log_jsonl`` MUST parse as a RunRecord-shaped JSON."""
    tracker, _sink = _new_tracker()
    rid_a = tracker.start_run(name="notify", run_type="tool", inputs={"x": 1})
    tracker.event(rid_a, "log", {"message": "midway"})
    tracker.complete_run(rid_a, outputs={"acknowledged": True})

    rid_b = tracker.start_run(name="ask", run_type="tool", inputs={"q": "?"})
    tracker.fail_run(rid_b, error={"kind": "Cancelled", "message": "interrupted", "traceback": ""})

    log = tracker.event_log_jsonl()
    lines = [ln for ln in log.splitlines() if ln.strip()]
    assert len(lines) == 2

    parsed = [json.loads(ln) for ln in lines]
    # Round-trip via the pydantic model to assert the on-disk shape.
    records = [RunRecord.model_validate(obj) for obj in parsed]
    assert {r.id for r in records} == {rid_a, rid_b}

    finished = next(r for r in records if r.id == rid_a)
    assert finished.outputs == {"acknowledged": True}
    assert finished.end_time is not None
    assert len(finished.events) == 1
    assert finished.events[0].event_type == "log"

    failed = next(r for r in records if r.id == rid_b)
    assert failed.error is not None
    assert failed.error["kind"] == "Cancelled"


def test_event_against_unknown_run_raises() -> None:
    """``event`` against an unknown id MUST raise KeyError."""
    tracker, _ = _new_tracker()
    with pytest.raises(KeyError):
        tracker.event(str(uuid.uuid4()), "log", {})


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
        assert kwargs["content"]["transient"] == {"display_id": rid}

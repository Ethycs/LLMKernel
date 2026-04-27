"""K-MW queue-overflow disk fallback (RFC-005 §F13).

Verifies the writer drops a checkpoint marker + direct-write of the
snapshot when the in-memory event-log queue exceeds
``event_log_queue_cap``, drains the queue, increments the checkpoint
counter, and emits a ``metadata.queue_overflow`` warning AFTER the
lock has been released (per Engineering Guide §11.7).
"""

from __future__ import annotations

import json
import logging as _logging
import secrets
from typing import Any, Dict
from pathlib import Path

import pytest

from llm_kernel.metadata_writer import MetadataWriter


def _make_span(name: str) -> Dict[str, Any]:
    return {
        "traceId": secrets.token_hex(16),
        "spanId": secrets.token_hex(8),
        "parentSpanId": None,
        "name": name,
        "kind": "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": "0",
        "endTimeUnixNano": "1",
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "attributes": [], "events": [], "links": [],
    }


def test_overflow_writes_marker_and_snapshot(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Overflow drops marker + snapshot files in workspace_root."""
    cap = 4
    writer = MetadataWriter(
        autosave_interval_sec=999.0,
        event_log_queue_cap=cap,
        workspace_root=tmp_path,
        session_id="s-overflow",
    )
    with caplog.at_level(_logging.WARNING, logger="llm_kernel.metadata_writer"):
        for i in range(cap + 3):
            writer.record_run(_make_span(f"run-{i}"))

    marker_path = tmp_path / ".llmnb-overflow-marker.json"
    snapshot_path = tmp_path / ".llmnb-overflow-snapshot.json"
    assert marker_path.exists(), "checkpoint marker file not written"
    assert snapshot_path.exists(), "overflow snapshot file not written"

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["kernel_session_id"] == "s-overflow"
    assert marker["queue_size_at_overflow"] >= cap + 1
    assert isinstance(marker["snapshot_version"], int)
    assert isinstance(marker["overflow_at"], str)

    snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snap["snapshot_version"] == marker["snapshot_version"]
    assert "event_log" in snap
    assert "schema_version" in snap


def test_overflow_drains_queue_so_subsequent_record_runs_fit(
    tmp_path: Path,
) -> None:
    """After overflow the in-memory queue is drained back to zero."""
    cap = 3
    writer = MetadataWriter(
        autosave_interval_sec=999.0,
        event_log_queue_cap=cap,
        workspace_root=tmp_path,
    )
    for i in range(cap + 1):
        writer.record_run(_make_span(f"run-{i}"))
    # Internal: queue should now be empty (overflow drained it).
    # Use the public-ish path: build a snapshot and confirm runs <= cap.
    snap = writer.snapshot()
    assert len(snap["event_log"]["runs"]) <= cap


def test_overflow_increments_checkpoint_counter_each_time(
    tmp_path: Path,
) -> None:
    """Each overflow round increments ``_overflow_checkpoint_count``."""
    cap = 2
    writer = MetadataWriter(
        autosave_interval_sec=999.0,
        event_log_queue_cap=cap,
        workspace_root=tmp_path,
    )
    # First overflow: cap+1 = 3 records.
    for i in range(cap + 1):
        writer.record_run(_make_span(f"a-{i}"))
    first = writer._overflow_checkpoint_count  # noqa: SLF001
    assert first == 1
    # Second overflow round.
    for i in range(cap + 1):
        writer.record_run(_make_span(f"b-{i}"))
    assert writer._overflow_checkpoint_count == first + 1  # noqa: SLF001


def test_overflow_emits_metadata_queue_overflow_log_record(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The overflow log record carries event.name + checkpoint count."""
    cap = 2
    writer = MetadataWriter(
        autosave_interval_sec=999.0,
        event_log_queue_cap=cap,
        workspace_root=tmp_path,
    )
    with caplog.at_level(_logging.WARNING, logger="llm_kernel.metadata_writer"):
        for i in range(cap + 1):
            writer.record_run(_make_span(f"run-{i}"))

    overflow_records = [
        r for r in caplog.records
        if getattr(r, "event.name", None) == "metadata.queue_overflow"
    ]
    assert overflow_records, "no metadata.queue_overflow log record emitted"
    rec = overflow_records[0]
    assert rec.levelno == _logging.WARNING
    assert getattr(rec, "llmnb.checkpoint_count", None) == 1
    assert getattr(rec, "llmnb.queue_size_at_overflow", 0) >= cap + 1


def test_overflow_log_emitted_after_lock_release(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Engineering Guide §11.7: log handler MUST be able to re-enter the writer.

    A handler that calls back into ``writer.record_run`` while the
    overflow log record is being processed MUST NOT deadlock.
    """
    cap = 2
    writer = MetadataWriter(
        autosave_interval_sec=999.0,
        event_log_queue_cap=cap,
        workspace_root=tmp_path,
    )

    seen: list = []

    class _ReentrantHandler(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            # Simulate a handler that touches the writer's data plane
            # while the log record is being processed.  If the writer
            # were holding a non-RLock at this point, this call would
            # deadlock.
            seen.append(record.getMessage())
            try:
                writer.append_drift_event(
                    field_path="diagnostic.handler_reentry",
                    previous_value=None, current_value=True, severity="info",
                )
            except Exception:  # pragma: no cover - defensive
                pass

    handler = _ReentrantHandler()
    handler.setLevel(_logging.WARNING)
    target_logger = _logging.getLogger("llm_kernel.metadata_writer")
    target_logger.addHandler(handler)
    try:
        for i in range(cap + 1):
            writer.record_run(_make_span(f"run-{i}"))
    finally:
        target_logger.removeHandler(handler)

    assert any("queue overflow" in m for m in seen), seen

"""PLAN-S6.0 §5 round-trip tests for the in-tree event log."""

from __future__ import annotations

from typing import Any, Dict, List

from llm_kernel.event_log import EventLogReplayer
from llm_kernel.metadata_writer import MetadataWriter
from llm_kernel.run_envelope import (
    DIRECTION_EXTENSION_TO_KERNEL, DIRECTION_KERNEL_TO_EXTENSION, make_envelope,
)


def _envelope(message_type: str, payload: Dict[str, Any], **kw: Any) -> Dict[str, Any]:
    """Helper: build a v1 envelope with a stable correlation_id."""
    return make_envelope(
        message_type, payload,
        correlation_id=kw.get("correlation_id", "00000000-0000-0000-0000-000000000001"),
        direction=kw.get("direction", DIRECTION_EXTENSION_TO_KERNEL),
        timestamp=kw.get("timestamp", "2026-04-30T00:00:00.000Z"),
    )


def _snapshot_envelope(seq: int) -> Dict[str, Any]:
    """Helper: build a Family F mode=snapshot envelope."""
    return _envelope(
        "notebook.metadata",
        {
            "mode": "snapshot",
            "snapshot_version": seq,
            "snapshot": {
                "schema_version": "1.0.0",
                "session_id": "fixture",
                "snapshot_version": seq,
                "marker": f"checkpoint-{seq}",
            },
            "trigger": "save",
        },
        direction=DIRECTION_KERNEL_TO_EXTENSION,
    )


def test_append_and_read() -> None:
    """A known sequence appended via the writer reads back identically.

    Drives ``MetadataWriter.capture_envelope`` directly (the dispatcher
    tee delegates here), then constructs a replayer and asserts the
    envelopes-after-snapshot iterator yields the appended sequence.
    """
    writer = MetadataWriter(autosave_interval_sec=999.0)

    # Open with a snapshot so the replayer has a checkpoint to anchor on.
    snap = _snapshot_envelope(1)
    op_a = _envelope("operator.action", {"action_type": "zone_mutate", "n": 1})
    op_b = _envelope("operator.action", {"action_type": "zone_mutate", "n": 2})
    layout = _envelope(
        "layout.update", {"tree": {"id": "root"}},
        direction=DIRECTION_KERNEL_TO_EXTENSION,
    )

    for env in (snap, op_a, op_b, layout):
        writer.capture_envelope(env)

    persisted = writer.snapshot()
    log: List[Dict[str, Any]] = persisted["zone"]["event_log"]

    # Round-trip via the replayer.
    replayer = EventLogReplayer(log)
    after = list(replayer.envelopes_after_snapshot())
    assert [e["message_type"] for e in after] == [
        "operator.action", "operator.action", "layout.update",
    ]
    # Identity preserved (deep-copy on append, but content equal).
    assert after[0]["payload"] == op_a["payload"]
    assert after[2]["payload"] == layout["payload"]


def test_latest_snapshot() -> None:
    """A log containing 3 snapshots returns the third on latest_snapshot()."""
    writer = MetadataWriter(autosave_interval_sec=999.0)
    s1 = _snapshot_envelope(1)
    s2 = _snapshot_envelope(2)
    s3 = _snapshot_envelope(3)
    op = _envelope("operator.action", {"between": True})
    for env in (s1, op, s2, op, s3):
        writer.capture_envelope(env)

    persisted = writer.snapshot()
    log = persisted["zone"]["event_log"]
    replayer = EventLogReplayer(log)
    latest = replayer.latest_snapshot()
    assert latest is not None
    assert latest["payload"]["snapshot_version"] == 3
    assert latest["payload"]["snapshot"]["marker"] == "checkpoint-3"

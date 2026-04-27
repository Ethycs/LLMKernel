"""K-MW :meth:`MetadataWriter.acknowledge_drift` (RFC-005 / RFC-006 §6).

The MCP ``drift_acknowledged`` operator action and the Family D
``operator.action.drift_acknowledged`` envelope both call this method
to flip ``operator_acknowledged`` on a specific drift entry.
"""

from __future__ import annotations

from llm_kernel.metadata_writer import MetadataWriter


def test_acknowledge_drift_returns_true_on_match() -> None:
    """Matching field_path + detected_at returns True and flips the flag."""
    writer = MetadataWriter(autosave_interval_sec=999.0)
    event = writer.append_drift_event(
        field_path="config.volatile.kernel.model_default",
        previous_value="claude-sonnet-4-5",
        current_value="claude-sonnet-4-6",
        severity="warn",
    )
    detected_at = event["detected_at"]

    ok = writer.acknowledge_drift(
        field_path="config.volatile.kernel.model_default",
        detected_at=detected_at,
    )
    assert ok is True
    snap = writer.snapshot()
    assert snap["drift_log"][0]["operator_acknowledged"] is True


def test_acknowledge_drift_is_idempotent() -> None:
    """Calling twice returns True both times; state stays acknowledged."""
    writer = MetadataWriter(autosave_interval_sec=999.0)
    event = writer.append_drift_event(
        field_path="config.volatile.kernel.model_default",
        previous_value="x", current_value="y", severity="info",
    )
    a = writer.acknowledge_drift(
        field_path=event["field_path"], detected_at=event["detected_at"],
    )
    b = writer.acknowledge_drift(
        field_path=event["field_path"], detected_at=event["detected_at"],
    )
    assert a is True
    assert b is True
    snap = writer.snapshot()
    assert snap["drift_log"][0]["operator_acknowledged"] is True


def test_acknowledge_drift_returns_false_on_no_match() -> None:
    """Unknown field_path/detected_at returns False without raising."""
    writer = MetadataWriter(autosave_interval_sec=999.0)
    writer.append_drift_event(
        field_path="config.volatile.kernel.model_default",
        previous_value="x", current_value="y", severity="info",
    )
    ok = writer.acknowledge_drift(
        field_path="config.volatile.kernel.model_default",
        detected_at="9999-01-01T00:00:00.000Z",  # wrong timestamp
    )
    assert ok is False
    ok = writer.acknowledge_drift(
        field_path="not.a.real.path",
        detected_at="2026-04-26T13:22:45.000Z",
    )
    assert ok is False


def test_acknowledge_drift_only_flips_matching_entry() -> None:
    """When several drift events share a field_path, only the matching
    one (by detected_at) is acknowledged."""
    writer = MetadataWriter(autosave_interval_sec=999.0)
    e1 = writer.append_drift_event(
        field_path="config.volatile.kernel.model_default",
        previous_value="a", current_value="b", severity="warn",
        detected_at="2026-04-26T13:22:45.000Z",
    )
    e2 = writer.append_drift_event(
        field_path="config.volatile.kernel.model_default",
        previous_value="b", current_value="c", severity="warn",
        detected_at="2026-04-26T14:00:00.000Z",
    )
    ok = writer.acknowledge_drift(
        field_path=e2["field_path"], detected_at=e2["detected_at"],
    )
    assert ok is True
    snap = writer.snapshot()
    flags = [(d["detected_at"], d["operator_acknowledged"])
             for d in snap["drift_log"]]
    assert (e1["detected_at"], False) in flags
    assert (e2["detected_at"], True) in flags

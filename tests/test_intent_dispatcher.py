"""K-MW BSP-003 §10 intent dispatcher.

Covers :meth:`MetadataWriter.submit_intent` -- the public mutation
entrypoint per BSP-003 §6:

* FIFO serialization of concurrent submissions.
* Idempotency on duplicate ``intent_id`` (BSP-003 §6 step 2).
* CAS rejection K41 on ``expected_snapshot_version`` mismatch
  (BSP-003 §6 step 3 + §8 K41).
* K40 on unknown ``intent_kind`` (BSP-003 §8 K40).
* K42 on validator rejection (BSP-003 §8 K42).
* ``intent_applied`` event-log entries are emitted per applied
  intent (BSP-003 §6 step 7).
* The post-apply ``notebook.metadata`` snapshot envelope is emitted
  (BSP-003 §6 step 9).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

from llm_kernel.metadata_writer import MetadataWriter


def _new_writer() -> MetadataWriter:
    return MetadataWriter(autosave_interval_sec=999.0)


def _envelope(
    intent_kind: str,
    parameters: Dict[str, Any],
    intent_id: str,
    expected_snapshot_version: int | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "action_type": "zone_mutate",
        "intent_kind": intent_kind,
        "parameters": parameters,
        "intent_id": intent_id,
    }
    if expected_snapshot_version is not None:
        payload["expected_snapshot_version"] = expected_snapshot_version
    return {"type": "operator.action", "payload": payload}


# ---------------------------------------------------------------------------
# Happy path: dispatch a layout edit through the registry.
# ---------------------------------------------------------------------------


def test_submit_intent_dispatches_layout_edit_and_bumps_version() -> None:
    """A registered ``apply_layout_edit`` intent applies and bumps version."""
    writer = _new_writer()
    initial = writer.emit_layout_update()["snapshot_version"]
    result = writer.submit_intent(_envelope(
        intent_kind="apply_layout_edit",
        parameters={
            "operation": "add_zone",
            "parameters": {
                "node_spec": {"id": "zone-a", "type": "zone"},
            },
        },
        intent_id="intent-001",
    ))
    assert result["applied"] is True
    assert result["already_applied"] is False
    assert result["error_code"] is None
    assert result["snapshot_version"] > initial
    # The layout state was actually mutated.
    tree = writer.emit_layout_update()["tree"]
    assert tree["children"][0]["id"] == "zone-a"


def test_submit_intent_dispatches_agent_graph_command() -> None:
    """A ``apply_agent_graph_command`` intent applies and returns response."""
    writer = _new_writer()
    result = writer.submit_intent(_envelope(
        intent_kind="apply_agent_graph_command",
        parameters={
            "command": "upsert_node",
            "parameters": {
                "node": {
                    "id": "agent:alpha", "type": "agent", "properties": {},
                },
            },
        },
        intent_id="intent-graph-001",
    ))
    assert result["applied"] is True
    assert isinstance(result["response"], dict)
    assert result["response"]["ok"] is True


def test_submit_intent_records_intent_applied_log_entry() -> None:
    """Each applied intent appends to the in-memory intent log (§6 step 7)."""
    writer = _new_writer()
    writer.submit_intent(_envelope(
        intent_kind="apply_layout_edit",
        parameters={
            "operation": "add_zone",
            "parameters": {"node_spec": {"id": "z1", "type": "zone"}},
        },
        intent_id="intent-record-001",
    ))
    log = writer.iter_intent_log()
    assert len(log) == 1
    entry = log[0]
    assert entry["type"] == "intent_applied"
    assert entry["intent_id"] == "intent-record-001"
    assert entry["intent_kind"] == "apply_layout_edit"
    assert isinstance(entry["snapshot_version"], int)
    assert isinstance(entry["recorded_at"], str)


def test_submit_intent_emits_post_apply_snapshot_envelope() -> None:
    """BSP-003 §6 step 9: post-apply ``notebook.metadata`` envelope is emitted."""
    writer = _new_writer()
    writer.submit_intent(_envelope(
        intent_kind="apply_layout_edit",
        parameters={
            "operation": "add_zone",
            "parameters": {"node_spec": {"id": "z1", "type": "zone"}},
        },
        intent_id="intent-emit-001",
    ))
    env = writer.take_last_envelope()
    assert env is not None
    assert env["message_type"] == "notebook.metadata"
    payload = env["payload"]
    assert payload["mode"] == "snapshot"
    assert payload["trigger"] == "intent_applied"
    assert isinstance(payload["snapshot_version"], int)


# ---------------------------------------------------------------------------
# Idempotency: duplicate intent_id is a no-op.
# ---------------------------------------------------------------------------


def test_submit_intent_idempotent_on_duplicate_intent_id() -> None:
    """Re-submitting the same ``intent_id`` returns ``already_applied=True``."""
    writer = _new_writer()
    first = writer.submit_intent(_envelope(
        intent_kind="apply_layout_edit",
        parameters={
            "operation": "add_zone",
            "parameters": {"node_spec": {"id": "z1", "type": "zone"}},
        },
        intent_id="intent-dup",
    ))
    assert first["applied"] is True
    second = writer.submit_intent(_envelope(
        intent_kind="apply_layout_edit",
        parameters={
            "operation": "add_zone",
            "parameters": {"node_spec": {"id": "z2", "type": "zone"}},
        },
        intent_id="intent-dup",
    ))
    assert second["applied"] is False
    assert second["already_applied"] is True
    assert second["snapshot_version"] == first["snapshot_version"]
    # State was NOT mutated by the duplicate (z2 should be absent).
    tree = writer.emit_layout_update()["tree"]
    ids = [c["id"] for c in tree["children"]]
    assert ids == ["z1"], ids


# ---------------------------------------------------------------------------
# CAS path: K41 on expected_snapshot_version mismatch.
# ---------------------------------------------------------------------------


def test_submit_intent_cas_accepts_matching_expected_version() -> None:
    """CAS path applies when ``expected_snapshot_version`` matches current."""
    writer = _new_writer()
    # Apply once to bump version.
    writer.submit_intent(_envelope(
        intent_kind="apply_layout_edit",
        parameters={
            "operation": "add_zone",
            "parameters": {"node_spec": {"id": "z1", "type": "zone"}},
        },
        intent_id="intent-cas-warmup",
    ))
    cur = writer.emit_layout_update()["snapshot_version"]
    result = writer.submit_intent(_envelope(
        intent_kind="apply_layout_edit",
        parameters={
            "operation": "add_zone",
            "parameters": {"node_spec": {"id": "z2", "type": "zone"}},
        },
        intent_id="intent-cas-ok",
        expected_snapshot_version=cur,
    ))
    assert result["applied"] is True


def test_submit_intent_cas_rejects_stale_expected_version_with_k41() -> None:
    """Stale ``expected_snapshot_version`` returns K41 with no state change."""
    writer = _new_writer()
    writer.submit_intent(_envelope(
        intent_kind="apply_layout_edit",
        parameters={
            "operation": "add_zone",
            "parameters": {"node_spec": {"id": "z1", "type": "zone"}},
        },
        intent_id="intent-cas-warmup",
    ))
    before = writer.emit_layout_update()
    stale = before["snapshot_version"] - 5  # arbitrary stale version
    result = writer.submit_intent(_envelope(
        intent_kind="apply_layout_edit",
        parameters={
            "operation": "add_zone",
            "parameters": {"node_spec": {"id": "z-rejected", "type": "zone"}},
        },
        intent_id="intent-cas-stale",
        expected_snapshot_version=stale,
    ))
    assert result["applied"] is False
    assert result["error_code"] == "K41"
    assert "CAS" in result["error_reason"]
    # State unchanged.
    after = writer.emit_layout_update()
    after_ids = [c["id"] for c in after["tree"]["children"]]
    assert "z-rejected" not in after_ids


# ---------------------------------------------------------------------------
# Unknown intent_kind: K40.
# ---------------------------------------------------------------------------


def test_submit_intent_unknown_kind_returns_k40() -> None:
    """An unregistered ``intent_kind`` triggers K40 (BSP-003 §8)."""
    writer = _new_writer()
    result = writer.submit_intent(_envelope(
        intent_kind="not_a_real_kind",
        parameters={},
        intent_id="intent-k40",
    ))
    assert result["applied"] is False
    assert result["error_code"] == "K40"
    assert "not_a_real_kind" in result["error_reason"]


# ---------------------------------------------------------------------------
# Validation: K42 on bad parameters.
# ---------------------------------------------------------------------------


def test_submit_intent_returns_k42_on_apply_function_rejection() -> None:
    """An apply that returns false (no-op) bubbles up as K42."""
    writer = _new_writer()
    result = writer.submit_intent(_envelope(
        intent_kind="apply_layout_edit",
        parameters={
            "operation": "remove_node",
            "parameters": {"node_id": "does-not-exist"},
        },
        intent_id="intent-k42",
    ))
    assert result["applied"] is False
    assert result["error_code"] == "K42"


def test_submit_intent_returns_k42_on_missing_required_parameter() -> None:
    """Missing ``operation`` parameter raises validator K42."""
    writer = _new_writer()
    result = writer.submit_intent(_envelope(
        intent_kind="apply_layout_edit",
        parameters={"parameters": {}},
        intent_id="intent-k42-missing",
    ))
    assert result["applied"] is False
    assert result["error_code"] == "K42"


def test_submit_intent_returns_k42_on_malformed_envelope() -> None:
    """Envelope with no ``payload`` returns K42."""
    writer = _new_writer()
    result = writer.submit_intent({"type": "operator.action"})
    assert result["applied"] is False
    assert result["error_code"] == "K42"


def test_submit_intent_returns_k42_on_missing_intent_id() -> None:
    """Envelope with missing/empty ``intent_id`` returns K42."""
    writer = _new_writer()
    result = writer.submit_intent({
        "type": "operator.action",
        "payload": {
            "intent_kind": "apply_layout_edit",
            "parameters": {},
        },
    })
    assert result["applied"] is False
    assert result["error_code"] == "K42"


# ---------------------------------------------------------------------------
# FIFO serialization under concurrent submissions.
# ---------------------------------------------------------------------------


def test_submit_intent_serializes_concurrent_submissions() -> None:
    """Concurrent submissions interleave deterministically; no version skips."""
    writer = _new_writer()
    threads: List[threading.Thread] = []
    results: List[Dict[str, Any]] = []
    results_lock = threading.Lock()

    def _submit(i: int) -> None:
        out = writer.submit_intent(_envelope(
            intent_kind="apply_layout_edit",
            parameters={
                "operation": "add_zone",
                "parameters": {
                    "node_spec": {"id": f"zone-{i}", "type": "zone"},
                },
            },
            intent_id=f"intent-fifo-{i}",
        ))
        with results_lock:
            results.append(out)

    for i in range(8):
        t = threading.Thread(target=_submit, args=(i,), daemon=True)
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    # All 8 applied; all 8 unique snapshot_versions.
    applied = [r for r in results if r["applied"]]
    assert len(applied) == 8
    versions = [r["snapshot_version"] for r in applied]
    assert len(set(versions)) == 8, versions
    # The layout has 8 zones.
    tree = writer.emit_layout_update()["tree"]
    assert len(tree["children"]) == 8


# ---------------------------------------------------------------------------
# record_event / acknowledge_drift bridges through the registry.
# ---------------------------------------------------------------------------


def test_submit_intent_record_event_appends_drift_log() -> None:
    """``record_event`` intent writes through to the drift log."""
    writer = _new_writer()
    result = writer.submit_intent(_envelope(
        intent_kind="record_event",
        parameters={
            "field_path": "config.volatile.kernel.model_default",
            "previous_value": "claude-sonnet-4-5",
            "current_value": "claude-sonnet-4-6",
            "severity": "warn",
        },
        intent_id="intent-record-evt",
    ))
    assert result["applied"] is True
    snap = writer.snapshot()
    assert any(
        entry["field_path"] == "config.volatile.kernel.model_default"
        for entry in snap["drift_log"]
    )


def test_submit_intent_acknowledge_drift_through_registry() -> None:
    """The acknowledge_drift intent flips the ``operator_acknowledged`` flag."""
    writer = _new_writer()
    event = writer.append_drift_event(
        field_path="config.volatile.kernel.model_default",
        previous_value="x", current_value="y", severity="info",
    )
    result = writer.submit_intent(_envelope(
        intent_kind="acknowledge_drift",
        parameters={
            "field_path": event["field_path"],
            "detected_at": event["detected_at"],
        },
        intent_id="intent-ack",
    ))
    assert result["applied"] is True
    snap = writer.snapshot()
    assert snap["drift_log"][0]["operator_acknowledged"] is True

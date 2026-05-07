"""K-CTXR BSP-008 §7 / §8 -- ``record_run_frame`` writer handler tests.

Per ``docs/notebook/BSP-008-contextpacker-runframes.md`` and
``docs/atoms/concepts/run-frame.md``: the BSP-003 intent registry
accepts ``record_run_frame`` envelopes that persist a RunFrame dict
under ``metadata.rts.zone.run_frames.<run_id>``.

Coverage:

1. Round-trip: start frame (status=running) followed by a terminal
   frame (status=complete) updates the persisted record in place.
2. Idempotency: the same ``run_id`` from the same ``cell_id`` accepts
   re-submission without K42.
3. K102 cell-id mismatch: the same ``run_id`` from a DIFFERENT
   ``cell_id`` is rejected with K42 + the K102 marker text.
4. Validation: missing ``run_id`` / ``cell_id`` / ``executor_id`` /
   ``context_manifest_id`` / ``started_at`` / ``status`` produces K42.
5. Persistence path: after a successful submit, the writer's snapshot
   surfaces the record at ``zone.run_frames[<run_id>]`` with all of
   the submitted fields preserved.

Engineering Guide §11.7 parallel-test-safety: every test instantiates
its own writer via :func:`_new_writer` (no shared state, no
``tmp_path`` needed because the writer is in-memory only).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from llm_kernel.metadata_writer import MetadataWriter


def _new_writer() -> MetadataWriter:
    return MetadataWriter(autosave_interval_sec=999.0)


def _envelope(
    intent_kind: str,
    parameters: Dict[str, Any],
    intent_id: str,
) -> Dict[str, Any]:
    return {
        "type": "operator.action",
        "payload": {
            "action_type": "zone_mutate",
            "intent_kind": intent_kind,
            "parameters": parameters,
            "intent_id": intent_id,
        },
    }


def _run_frame(
    *,
    run_id: str,
    cell_id: str = "c1",
    executor_id: str = "agent-alpha",
    context_manifest_id: str = "manifest-001",
    status: str = "running",
    turn_head_before: Optional[str] = None,
    turn_head_after: Optional[str] = None,
    started_at: str = "2026-05-06T12:00:00.000Z",
    ended_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a minimal RunFrame dict matching BSP-008 §7."""
    return {
        "run_id": run_id,
        "cell_id": cell_id,
        "executor_id": executor_id,
        "turn_head_before": turn_head_before,
        "turn_head_after": turn_head_after,
        "context_manifest_id": context_manifest_id,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
    }


# ---------------------------------------------------------------------------
# 1. Start + terminal round-trip
# ---------------------------------------------------------------------------


def test_record_run_frame_round_trip() -> None:
    """A start frame + terminal frame update the same record in place."""
    writer = _new_writer()
    run_id = uuid.uuid4().hex
    # Start frame.
    start_result = writer.submit_intent(_envelope(
        intent_kind="record_run_frame",
        parameters={"run_frame": _run_frame(
            run_id=run_id, status="running",
            started_at="2026-05-06T12:00:00.000Z",
            ended_at=None,
        )},
        intent_id=f"i-rrf-start-{run_id}",
    ))
    assert start_result["applied"] is True, start_result
    assert start_result["error_code"] is None
    # Terminal frame.
    term_result = writer.submit_intent(_envelope(
        intent_kind="record_run_frame",
        parameters={"run_frame": _run_frame(
            run_id=run_id, status="complete",
            turn_head_after="t-42",
            started_at="2026-05-06T12:00:00.000Z",
            ended_at="2026-05-06T12:00:01.500Z",
        )},
        intent_id=f"i-rrf-end-{run_id}",
    ))
    assert term_result["applied"] is True, term_result
    snap = writer.snapshot()
    persisted = snap["zone"]["run_frames"][run_id]
    assert persisted["status"] == "complete"
    assert persisted["ended_at"] == "2026-05-06T12:00:01.500Z"
    assert persisted["turn_head_after"] == "t-42"


# ---------------------------------------------------------------------------
# 2. Idempotency on run_id (same cell_id)
# ---------------------------------------------------------------------------


def test_record_run_frame_idempotent_on_run_id() -> None:
    """Same run_id + same cell_id is accepted on re-submission (no K42)."""
    writer = _new_writer()
    run_id = uuid.uuid4().hex
    frame = _run_frame(run_id=run_id, status="running")
    # First submit.
    r1 = writer.submit_intent(_envelope(
        intent_kind="record_run_frame",
        parameters={"run_frame": frame},
        intent_id=f"i-rrf-1-{run_id}",
    ))
    assert r1["applied"] is True
    # Second submit (different intent_id so the §6 step 2 idempotency
    # check does NOT short-circuit; the record-level idempotency is what
    # we are exercising).
    r2 = writer.submit_intent(_envelope(
        intent_kind="record_run_frame",
        parameters={"run_frame": frame},
        intent_id=f"i-rrf-2-{run_id}",
    ))
    assert r2["applied"] is True, r2
    assert r2["error_code"] is None


# ---------------------------------------------------------------------------
# 3. K102 — duplicate run_id with a different cell_id
# ---------------------------------------------------------------------------


def test_record_run_frame_k102_on_cell_id_mismatch() -> None:
    """Same run_id + different cell_id is rejected with K42 + K102 marker."""
    writer = _new_writer()
    run_id = uuid.uuid4().hex
    # First submit binds run_id to cell_id="c1".
    r1 = writer.submit_intent(_envelope(
        intent_kind="record_run_frame",
        parameters={"run_frame": _run_frame(
            run_id=run_id, cell_id="c1", status="running",
        )},
        intent_id=f"i-rrf-1-{run_id}",
    ))
    assert r1["applied"] is True
    # Second submit attempts to rebind to cell_id="c2".
    r2 = writer.submit_intent(_envelope(
        intent_kind="record_run_frame",
        parameters={"run_frame": _run_frame(
            run_id=run_id, cell_id="c2", status="running",
        )},
        intent_id=f"i-rrf-conflict-{run_id}",
    ))
    assert r2["applied"] is False
    assert r2["error_code"] == "K42"
    assert "K102" in (r2["error_reason"] or "")


# ---------------------------------------------------------------------------
# 4. Validation — required fields
# ---------------------------------------------------------------------------


def test_record_run_frame_validates_required_fields() -> None:
    """Each required field — run_id, cell_id, executor_id, context_manifest_id,
    status, started_at — produces K42 when missing."""
    writer = _new_writer()
    base = _run_frame(run_id="r1")
    required = (
        "run_id",
        "cell_id",
        "executor_id",
        "context_manifest_id",
        "status",
        "started_at",
    )
    for idx, field in enumerate(required):
        params = {"run_frame": dict(base)}
        params["run_frame"].pop(field, None)
        result = writer.submit_intent(_envelope(
            intent_kind="record_run_frame",
            parameters=params,
            intent_id=f"i-rrf-missing-{field}-{idx}",
        ))
        assert result["applied"] is False, (
            f"missing {field!r} should be rejected: {result}"
        )
        assert result["error_code"] == "K42", (
            f"missing {field!r} should map to K42: {result}"
        )
        assert field in (result["error_reason"] or ""), (
            f"K42 reason for missing {field!r} should mention the field: "
            f"{result['error_reason']!r}"
        )


def test_record_run_frame_validates_status_enum() -> None:
    """``status`` outside the BSP-008 §7 enum (+ ``running``) is rejected."""
    writer = _new_writer()
    result = writer.submit_intent(_envelope(
        intent_kind="record_run_frame",
        parameters={"run_frame": _run_frame(
            run_id="r-bad-status", status="cancelled",
        )},
        intent_id="i-rrf-bad-status",
    ))
    assert result["applied"] is False
    assert result["error_code"] == "K42"
    assert "status" in (result["error_reason"] or "")


# ---------------------------------------------------------------------------
# 5. Persistence path — snapshot()['zone']['run_frames'][run_id]
# ---------------------------------------------------------------------------


def test_record_run_frame_persistence_path() -> None:
    """After submit, the snapshot exposes the record at the documented path."""
    writer = _new_writer()
    run_id = uuid.uuid4().hex
    frame = _run_frame(
        run_id=run_id,
        cell_id="c-target",
        executor_id="agent-omega",
        context_manifest_id="manifest-XYZ",
        status="running",
        turn_head_before="t-prev",
        started_at="2026-05-06T13:00:00.000Z",
    )
    result = writer.submit_intent(_envelope(
        intent_kind="record_run_frame",
        parameters={"run_frame": frame},
        intent_id=f"i-rrf-persist-{run_id}",
    ))
    assert result["applied"] is True, result
    snap = writer.snapshot()
    zone = snap.get("zone", {})
    run_frames = zone.get("run_frames", {})
    assert run_id in run_frames
    persisted = run_frames[run_id]
    # All submitted fields round-trip verbatim.
    for key, expected in frame.items():
        assert persisted[key] == expected, (
            f"field {key!r}: expected {expected!r}, got {persisted.get(key)!r}"
        )


# ---------------------------------------------------------------------------
# Bonus: direct-shape acceptance (params={...} sans the "run_frame" wrapper)
# ---------------------------------------------------------------------------


def test_record_run_frame_accepts_unwrapped_params() -> None:
    """Per the handler's accept-both contract: params={...} works too."""
    writer = _new_writer()
    run_id = uuid.uuid4().hex
    result = writer.submit_intent(_envelope(
        intent_kind="record_run_frame",
        parameters=_run_frame(run_id=run_id, status="running"),
        intent_id=f"i-rrf-unwrapped-{run_id}",
    ))
    assert result["applied"] is True, result
    snap = writer.snapshot()
    assert run_id in snap["zone"]["run_frames"]


def test_record_run_frame_terminal_without_ended_at_is_accepted() -> None:
    """Per the handler doc: terminal status without ended_at is accepted + logged."""
    writer = _new_writer()
    run_id = uuid.uuid4().hex
    frame = _run_frame(
        run_id=run_id, status="complete",
        turn_head_after="t-99", ended_at=None,
    )
    # ended_at omitted entirely (not just None).
    frame.pop("ended_at", None)
    result = writer.submit_intent(_envelope(
        intent_kind="record_run_frame",
        parameters={"run_frame": frame},
        intent_id=f"i-rrf-term-no-ended-{run_id}",
    ))
    assert result["applied"] is True, result

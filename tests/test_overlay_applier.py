"""K-OVERLAY BSP-007 §9 -- overlay_applier test surface.

Covers the 15 tests called out in BSP-007 §9 plus a handful of
applier-level invariants the spec leaves implicit. Each test drives
the writer through ``submit_intent`` so the BSP-003 envelope shape and
the applier's lock / version-bump path are exercised end-to-end.

Mock surface: K95 ``overlay_blocked_by_execution`` is exercised via the
:meth:`MetadataWriter.set_cell_execution_state` test seam (the kernel's
real run-tracker isn't attached in this harness).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm_kernel.metadata_writer import MetadataWriter


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _envelope(
    intent_kind: str,
    parameters: Dict[str, Any],
    intent_id: str,
    expected_snapshot_version: Optional[int] = None,
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


def _new_writer() -> MetadataWriter:
    return MetadataWriter(autosave_interval_sec=999.0)


def _seed_cell(
    writer: MetadataWriter,
    cell_id: str,
    *,
    kind: str = "agent",
    bound_agent_id: Optional[str] = "alpha",
    section_id: Optional[str] = None,
    intent_id: Optional[str] = None,
) -> None:
    """Convenience: install a cell record via set_cell_metadata."""
    iid = intent_id or f"seed-{cell_id}"
    params: Dict[str, Any] = {"cell_id": cell_id, "kind": kind}
    if bound_agent_id is not None or kind in ("scratch", "checkpoint", "agent"):
        params["bound_agent_id"] = bound_agent_id
    if section_id is not None:
        params["section_id"] = section_id
    res = writer.submit_intent(_envelope("set_cell_metadata", params, iid))
    assert res["applied"], (cell_id, res)


def _apply_commit(
    writer: MetadataWriter,
    operations: List[Dict[str, Any]],
    *,
    intent_id: str,
    message: str = "",
) -> Dict[str, Any]:
    return writer.submit_intent(_envelope(
        "apply_overlay_commit",
        {"operations": operations, "message": message},
        intent_id,
    ))


# ---------------------------------------------------------------------
# Primitives: apply_commit, revert, named refs, diff.
# ---------------------------------------------------------------------


def test_apply_commit_advances_head_and_records() -> None:
    """§9 — single valid op; HEAD advances; commits[] grows by 1."""
    writer = _new_writer()
    _seed_cell(writer, "c_1")
    result = _apply_commit(writer, [
        {"kind": "set_pin", "cell_id": "c_1", "value": True},
    ], intent_id="i-1", message="pin c_1")
    assert result["applied"] is True, result
    # The handler returned a dict via the dispatcher's response path
    # carrying the applier-minted commit_id.
    assert result["response"] and result["response"]["commit_id"]
    snap = writer.snapshot()
    overlay = snap["zone"]["overlay"]
    assert isinstance(overlay["commits"], list)
    assert len(overlay["commits"]) == 1
    assert overlay["refs"]["HEAD"] == overlay["commits"][0]["commit_id"]
    # The pin landed on the cell.
    assert snap["cells"]["c_1"]["pinned"] is True
    # The intent_log carries the apply event.
    intent_log = snap["event_log"].get("intent_log", [])
    assert any(e["intent_kind"] == "apply_overlay_commit" for e in intent_log)


def test_apply_commit_atomic_rejects_partial() -> None:
    """§9 — submit a commit with two ops where the second fails;
    assert the first is rolled back and HEAD is unchanged.
    """
    writer = _new_writer()
    _seed_cell(writer, "c_1")
    # Pre-state: no overlay commits.
    snap_pre = writer.snapshot()
    assert "overlay" not in snap_pre.get("zone", {}) or (
        snap_pre["zone"]["overlay"]["commits"] == []
    )
    result = _apply_commit(writer, [
        {"kind": "set_pin", "cell_id": "c_1", "value": True},
        {"kind": "set_pin", "cell_id": "c_does_not_exist", "value": True},
    ], intent_id="i-atomic")
    assert result["applied"] is False
    assert result["error_code"] == "K90"
    assert result["response"]["marker"] == "overlay_commit_invalid"
    assert result["response"]["details"]["failed_operation_index"] == 1
    # Atomicity: the pin from op[0] was rolled back.
    snap_post = writer.snapshot()
    assert not snap_post["cells"]["c_1"].get("pinned")
    overlay = snap_post.get("zone", {}).get("overlay", {})
    # No commits appended.
    assert overlay.get("commits", []) == []


def test_revert_preserves_history() -> None:
    """§9 — apply 5 commits; revert to commit 2; assert commits[] still
    contains all 5; HEAD points at commit 2; commits 3-5 still inspectable.
    """
    writer = _new_writer()
    for i in range(1, 6):
        _seed_cell(writer, f"c_{i}")
    commit_ids: List[str] = []
    for i in range(1, 6):
        result = _apply_commit(writer, [
            {"kind": "set_pin", "cell_id": f"c_{i}", "value": True},
        ], intent_id=f"i-c{i}")
        assert result["applied"] is True, (i, result)
        snap = writer.snapshot()
        commit_ids.append(snap["zone"]["overlay"]["commits"][-1]["commit_id"])
    assert len(commit_ids) == 5
    # Revert to commit 2 (index 1 in the list).
    rev_result = writer.submit_intent(_envelope(
        "revert_overlay_to_commit",
        {"commit_id": commit_ids[1]},
        "i-rev",
    ))
    assert rev_result["applied"] is True, rev_result
    snap = writer.snapshot()
    overlay = snap["zone"]["overlay"]
    assert len(overlay["commits"]) == 5  # All commits preserved.
    assert overlay["refs"]["HEAD"] == commit_ids[1]
    # Commits 3-5 still inspectable.
    seen_ids = [c["commit_id"] for c in overlay["commits"]]
    for cid in commit_ids:
        assert cid in seen_ids


def test_revert_unreachable_commit_rejected() -> None:
    """§9 — revert with a random ULID; expect K91."""
    writer = _new_writer()
    _seed_cell(writer, "c_1")
    _apply_commit(writer, [
        {"kind": "set_pin", "cell_id": "c_1"},
    ], intent_id="i-1")
    result = writer.submit_intent(_envelope(
        "revert_overlay_to_commit",
        {"commit_id": "ovc_does_not_exist_at_all"},
        "i-bad-rev",
    ))
    assert result["applied"] is False
    assert result["error_code"] == "K91"
    assert result["response"]["marker"] == "overlay_commit_unreachable"


def test_named_ref_immutable_after_creation_in_v1() -> None:
    """§9 — create tag at commit 3; attempt to re-create at commit 4 → K92."""
    writer = _new_writer()
    for i in range(1, 5):
        _seed_cell(writer, f"c_{i}")
    commit_ids: List[str] = []
    for i in range(1, 5):
        _apply_commit(writer, [
            {"kind": "set_pin", "cell_id": f"c_{i}"},
        ], intent_id=f"i-c{i}")
        snap = writer.snapshot()
        commit_ids.append(snap["zone"]["overlay"]["commits"][-1]["commit_id"])
    # Create tag at commit 3.
    res_a = writer.submit_intent(_envelope(
        "create_overlay_ref",
        {"name": "v1-ship", "commit_id": commit_ids[2]},
        "i-ref-a",
    ))
    assert res_a["applied"] is True, res_a
    # Attempt to re-create at commit 4 -> K92.
    res_b = writer.submit_intent(_envelope(
        "create_overlay_ref",
        {"name": "v1-ship", "commit_id": commit_ids[3]},
        "i-ref-b",
    ))
    assert res_b["applied"] is False
    assert res_b["error_code"] == "K92"
    assert res_b["response"]["marker"] == "overlay_ref_conflict"
    assert res_b["response"]["details"]["existing_commit_id"] == commit_ids[2]


def test_diff_linear_history() -> None:
    """§9 — diff(commit_2, commit_5) returns ops from commits 3, 4, 5."""
    writer = _new_writer()
    for i in range(1, 6):
        _seed_cell(writer, f"c_{i}")
    commit_ids: List[str] = []
    for i in range(1, 6):
        _apply_commit(writer, [
            {"kind": "set_pin", "cell_id": f"c_{i}"},
        ], intent_id=f"i-c{i}")
        snap = writer.snapshot()
        commit_ids.append(snap["zone"]["overlay"]["commits"][-1]["commit_id"])
    diff_ops = writer.diff_overlay_commits(commit_ids[1], commit_ids[4])
    # Should be the operations from commits 3, 4, 5 (3 commits).
    assert len(diff_ops) == 3
    # Each op pinned a cell.
    pinned_cells = [op["cell_id"] for op in diff_ops]
    assert pinned_cells == ["c_3", "c_4", "c_5"]


# ---------------------------------------------------------------------
# Cell-merge invariants (§6).
# ---------------------------------------------------------------------


def test_merge_rejects_kind_mismatch() -> None:
    """§9 — merge agent_cell + checkpoint_cell → K93 different_primary_kind."""
    writer = _new_writer()
    _seed_cell(writer, "c_a", kind="agent")
    _seed_cell(writer, "c_b", kind="checkpoint", bound_agent_id=None)
    result = _apply_commit(writer, [
        {"kind": "merge_cells", "cell_a": "c_a", "cell_b": "c_b"},
    ], intent_id="i-bad-merge")
    assert result["applied"] is False
    assert result["error_code"] == "K93"
    assert result["response"]["marker"] == "overlay_merge_rejected"
    assert result["response"]["details"]["reason"] == "different_primary_kind"


def test_merge_rejects_pin_boundary() -> None:
    """§9 — pin between two otherwise-mergeable cells → K93 pin_boundary."""
    writer = _new_writer()
    _seed_cell(writer, "c_a", kind="agent", bound_agent_id="alpha")
    _seed_cell(writer, "c_b", kind="agent", bound_agent_id="alpha")
    # Pin c_b first.
    pin_result = _apply_commit(writer, [
        {"kind": "set_pin", "cell_id": "c_b", "value": True},
    ], intent_id="i-pin")
    assert pin_result["applied"] is True
    # Now try to merge.
    merge_result = _apply_commit(writer, [
        {"kind": "merge_cells", "cell_a": "c_a", "cell_b": "c_b"},
    ], intent_id="i-merge")
    assert merge_result["applied"] is False
    assert merge_result["error_code"] == "K93"
    assert merge_result["response"]["details"]["reason"] == "pin_boundary"


def test_merge_produces_sub_turns() -> None:
    """§9 — merge two single-turn cells; resulting cell has sub-turn addressing."""
    writer = _new_writer()
    _seed_cell(writer, "c_a", kind="agent", bound_agent_id="alpha")
    _seed_cell(writer, "c_b", kind="agent", bound_agent_id="alpha")
    result = _apply_commit(writer, [
        {"kind": "merge_cells", "cell_a": "c_a", "cell_b": "c_b"},
    ], intent_id="i-merge")
    assert result["applied"] is True, result
    snap = writer.snapshot()
    # c_a survives, c_b removed.
    assert "c_a" in snap["cells"]
    assert "c_b" not in snap["cells"]
    # Sub-turn handle stamped on c_a.
    assert snap["cells"]["c_a"]["sub_turn_addressing"] is True
    assert "c_b" in snap["cells"]["c_a"]["merged_from"]


# ---------------------------------------------------------------------
# Split invariants (§6.2).
# ---------------------------------------------------------------------


def test_split_at_invalid_boundary_rejected() -> None:
    """§9 — split mid-turn → K94."""
    writer = _new_writer()
    _seed_cell(writer, "c_1", kind="agent", bound_agent_id="alpha")
    result = _apply_commit(writer, [
        {
            "kind": "split_cell",
            "cell_id": "c_1",
            "at_turn_id": "t_mid",
            "mid_turn": True,
        },
    ], intent_id="i-bad-split")
    assert result["applied"] is False
    assert result["error_code"] == "K94"
    assert result["response"]["marker"] == "overlay_split_rejected"
    assert result["response"]["details"]["reason"] == "mid_turn"


def test_split_separates_tool_calls_from_parent_rejected() -> None:
    """§9 — split that would orphan a tool call → K94 would_orphan_tool_calls."""
    writer = _new_writer()
    _seed_cell(writer, "c_1", kind="agent", bound_agent_id="alpha")
    result = _apply_commit(writer, [
        {
            "kind": "split_cell",
            "cell_id": "c_1",
            "at_turn_id": "t_tool_parent",
            "would_orphan_tool_calls": True,
        },
    ], intent_id="i-bad-split-tool")
    assert result["applied"] is False
    assert result["error_code"] == "K94"
    assert (
        result["response"]["details"]["reason"]
        == "would_orphan_tool_calls"
    )


# ---------------------------------------------------------------------
# K95 — execution-in-flight.
# ---------------------------------------------------------------------


def test_overlay_blocked_during_execution() -> None:
    """§9 — start a run on c_5; submit merge_cells(c_4, c_5); expect K95.

    We use the writer's ``set_cell_execution_state`` test seam
    because the real run tracker is not attached in unit tests.
    """
    writer = _new_writer()
    _seed_cell(writer, "c_4", kind="agent", bound_agent_id="alpha")
    _seed_cell(writer, "c_5", kind="agent", bound_agent_id="alpha")
    writer.set_cell_execution_state("c_5", True)
    result = _apply_commit(writer, [
        {"kind": "merge_cells", "cell_a": "c_4", "cell_b": "c_5"},
    ], intent_id="i-blocked")
    assert result["applied"] is False
    assert result["error_code"] == "K95"
    assert result["response"]["marker"] == "overlay_blocked_by_execution"


# ---------------------------------------------------------------------
# Section ops.
# ---------------------------------------------------------------------


def test_section_delete_requires_empty() -> None:
    """§9 — delete a non-empty section without bulk-move → K90 section_not_empty."""
    writer = _new_writer()
    # Create a section and a cell in it; then attempt delete.
    _seed_cell(writer, "c_1", kind="agent", section_id="sec_arch")
    _apply_commit(writer, [
        {"kind": "create_section", "section_id": "sec_arch", "title": "Arch"},
        {"kind": "move_cell", "cell_id": "c_1",
         "target_section_id": "sec_arch", "position": 0},
    ], intent_id="i-create")
    # Now try to delete.
    result = _apply_commit(writer, [
        {"kind": "delete_section", "section_id": "sec_arch"},
    ], intent_id="i-delete")
    assert result["applied"] is False
    assert result["error_code"] == "K90"
    assert result["response"]["details"]["reason"] == "section_not_empty"


# PLAN-S5.5 §5 — comprehensive coverage for the four section overlay ops.
# Verifies the shipped BSP-007 (3a430cb) validators against the design.


def test_create_section_happy_path() -> None:
    """§5 — create a section; snapshot reflects id/title/empty cell_range/
    status defaulted to ``open`` and collapsed defaulted to ``false``."""
    writer = _new_writer()
    result = _apply_commit(writer, [
        {"kind": "create_section",
         "section_id": "sec_arch", "title": "Architecture"},
    ], intent_id="i-create-sec")
    assert result["applied"] is True, result
    snap = writer.snapshot()
    sec = snap["zone"]["sections"]["sec_arch"]
    assert sec["id"] == "sec_arch"
    assert sec["title"] == "Architecture"
    assert sec["parent_section_id"] is None
    assert sec["cell_range"] == []
    assert sec["status"] == "open"
    assert sec["collapsed"] is False


def test_create_section_rejects_parent_section_id() -> None:
    """§5 — non-null ``parent_section_id`` → K90 (decision D3: flat-only)."""
    writer = _new_writer()
    result = _apply_commit(writer, [
        {"kind": "create_section",
         "section_id": "sec_nested",
         "title": "Nested",
         "parent_section_id": "sec_outer"},
    ], intent_id="i-nested")
    assert result["applied"] is False
    assert result["error_code"] == "K90"
    assert result["response"]["details"]["reason"] == "nested_sections_forbidden"


def test_create_section_rejects_duplicate_id() -> None:
    """§5 — creating an already-existing section_id → K90."""
    writer = _new_writer()
    _apply_commit(writer, [
        {"kind": "create_section", "section_id": "sec_dup", "title": "First"},
    ], intent_id="i-first")
    result = _apply_commit(writer, [
        {"kind": "create_section", "section_id": "sec_dup", "title": "Second"},
    ], intent_id="i-second")
    assert result["applied"] is False
    assert result["error_code"] == "K90"
    assert result["response"]["details"]["reason"] == "duplicate_section_id"


def test_create_section_preserves_explicit_status_and_collapsed() -> None:
    """Operator-supplied status / collapsed flags persist verbatim.
    (Status state machine is Phase 1b; persistence is Phase 1a.)"""
    writer = _new_writer()
    result = _apply_commit(writer, [
        {"kind": "create_section",
         "section_id": "sec_done",
         "title": "Wrapped Up",
         "status": "complete",
         "collapsed": True},
    ], intent_id="i-done")
    assert result["applied"] is True, result
    sec = writer.snapshot()["zone"]["sections"]["sec_done"]
    assert sec["status"] == "complete"
    assert sec["collapsed"] is True


def test_rename_section_changes_title_keeps_id() -> None:
    """§5 — rename mutates title only; id remains stable per atom."""
    writer = _new_writer()
    _apply_commit(writer, [
        {"kind": "create_section", "section_id": "sec_r", "title": "Old"},
    ], intent_id="i-create")
    result = _apply_commit(writer, [
        {"kind": "rename_section", "section_id": "sec_r", "title": "New"},
    ], intent_id="i-rename")
    assert result["applied"] is True, result
    sec = writer.snapshot()["zone"]["sections"]["sec_r"]
    assert sec["id"] == "sec_r"
    assert sec["title"] == "New"


def test_rename_section_unknown_id_rejected() -> None:
    """§5 — rename of a non-existent section → K90 unknown_section."""
    writer = _new_writer()
    result = _apply_commit(writer, [
        {"kind": "rename_section",
         "section_id": "sec_ghost", "title": "Phantom"},
    ], intent_id="i-rename-ghost")
    assert result["applied"] is False
    assert result["error_code"] == "K90"
    assert result["response"]["details"]["reason"] == "unknown_section"


def test_delete_section_happy_path_empty() -> None:
    """§5 — delete an empty section; snapshot.sections no longer contains it."""
    writer = _new_writer()
    _apply_commit(writer, [
        {"kind": "create_section", "section_id": "sec_temp", "title": "Temp"},
    ], intent_id="i-create")
    result = _apply_commit(writer, [
        {"kind": "delete_section", "section_id": "sec_temp"},
    ], intent_id="i-delete")
    assert result["applied"] is True, result
    snap_sections = writer.snapshot().get("zone", {}).get("sections", {})
    assert "sec_temp" not in snap_sections


def test_delete_section_unknown_id_rejected() -> None:
    """§5 — delete of an unknown section → K90 unknown_section."""
    writer = _new_writer()
    result = _apply_commit(writer, [
        {"kind": "delete_section", "section_id": "sec_void"},
    ], intent_id="i-delete-void")
    assert result["applied"] is False
    assert result["error_code"] == "K90"
    assert result["response"]["details"]["reason"] == "unknown_section"


def test_move_cells_into_section_appends_to_cell_range() -> None:
    """§5 — move_cells_into_section bulk-appends; dual-rep cells.section_id
    and sections.cell_range[] agree."""
    writer = _new_writer()
    _seed_cell(writer, "c_a", kind="agent")
    _seed_cell(writer, "c_b", kind="agent")
    _apply_commit(writer, [
        {"kind": "create_section", "section_id": "sec_x", "title": "X"},
    ], intent_id="i-create-x")
    result = _apply_commit(writer, [
        {"kind": "move_cells_into_section",
         "target_section_id": "sec_x",
         "cell_ids": ["c_a", "c_b"],
         "position": 0},
    ], intent_id="i-move-batch")
    assert result["applied"] is True, result
    snap = writer.snapshot()
    assert snap["zone"]["sections"]["sec_x"]["cell_range"] == ["c_a", "c_b"]
    assert snap["cells"]["c_a"]["section_id"] == "sec_x"
    assert snap["cells"]["c_b"]["section_id"] == "sec_x"


def test_move_cells_into_section_empty_list_rejected() -> None:
    """§5 — empty ``cell_ids`` list → K90 cell_ids_required."""
    writer = _new_writer()
    _apply_commit(writer, [
        {"kind": "create_section", "section_id": "sec_y", "title": "Y"},
    ], intent_id="i-create-y")
    result = _apply_commit(writer, [
        {"kind": "move_cells_into_section",
         "target_section_id": "sec_y", "cell_ids": [], "position": 0},
    ], intent_id="i-move-empty")
    assert result["applied"] is False
    assert result["error_code"] == "K90"
    assert result["response"]["details"]["reason"] == "cell_ids_required"


def test_set_section_status_not_yet_shipped() -> None:
    """PLAN-S5.5 Phase 1b: ``set_section_status`` is queued; the
    operation kind is not yet in OVERLAY_OPERATION_KINDS, so submitting
    it surfaces an unknown-operation rejection. This test pins the
    current state so a Phase 1b implementer knows where to wire."""
    writer = _new_writer()
    _apply_commit(writer, [
        {"kind": "create_section", "section_id": "sec_s", "title": "S"},
    ], intent_id="i-create-s")
    result = _apply_commit(writer, [
        {"kind": "set_section_status",
         "section_id": "sec_s", "new_status": "in_progress"},
    ], intent_id="i-status-not-yet")
    # Phase 1b: this assertion flips to ``True`` when the op lands.
    assert result["applied"] is False
    assert result["error_code"] == "K90"


# ---------------------------------------------------------------------
# Promote span.
# ---------------------------------------------------------------------


def test_promote_span_creates_new_cell() -> None:
    """§9 — promote a span; new cell exists with the span as a binding;
    turn list is empty.
    """
    writer = _new_writer()
    result = _apply_commit(writer, [
        {
            "kind": "promote_span",
            "span_id": "sp_42",
            "cell_kind": "artifact",
            "new_cell_id": "prom_42",
        },
    ], intent_id="i-prom")
    assert result["applied"] is True, result
    snap = writer.snapshot()
    rec = snap["cells"]["prom_42"]
    assert rec["kind"] == "artifact"
    assert rec["bound_span_id"] == "sp_42"
    assert rec["turns"] == []
    assert rec["bound_agent_id"] is None


# ---------------------------------------------------------------------
# Wire envelope round-trip.
# ---------------------------------------------------------------------


def test_apply_overlay_commit_intent_envelope() -> None:
    """§9 — round-trip the BSP-003 envelope; assert dispatch to the
    overlay applier (not a per-op handler).
    """
    writer = _new_writer()
    _seed_cell(writer, "c_1")
    envelope = _envelope(
        "apply_overlay_commit",
        {
            "message": "split In[12] before span 4",
            "operations": [
                {"kind": "set_pin", "cell_id": "c_1", "value": True},
            ],
        },
        "i-envelope",
    )
    result = writer.submit_intent(envelope)
    assert result["applied"] is True, result
    # The intent_kind that landed in the intent_log is the wrapper,
    # NOT one of the inner op kinds.
    snap = writer.snapshot()
    intent_log = snap["event_log"].get("intent_log", [])
    kinds = [e["intent_kind"] for e in intent_log]
    assert "apply_overlay_commit" in kinds
    assert "set_pin" not in kinds  # never dispatched as a top-level intent
    # The applier minted a commit and applied the inner op.
    assert snap["cells"]["c_1"]["pinned"] is True


# ---------------------------------------------------------------------
# Bonus: persistence shape contract (locked-interface guarantee).
# ---------------------------------------------------------------------


def test_overlay_state_shape_locked_interface() -> None:
    """The locked interface: overlay dict ALWAYS has commits + refs.

    Per the brief: ``MetadataWriter.snapshot()['zone']['overlay']``
    MUST always be a dict with ``commits`` (list) and ``refs`` (dict).
    HEAD-less state is ``commits == []`` and ``refs == {}``. After the
    first apply, ``refs.HEAD`` exists.
    """
    writer = _new_writer()
    # Pre-apply: empty zone (overlay sub may or may not exist; if it
    # exists it must have the locked shape).
    _seed_cell(writer, "c_1")
    _apply_commit(writer, [
        {"kind": "set_pin", "cell_id": "c_1"},
    ], intent_id="i-shape")
    overlay = writer.snapshot()["zone"]["overlay"]
    assert isinstance(overlay, dict)
    assert isinstance(overlay["commits"], list)
    assert isinstance(overlay["refs"], dict)
    assert "HEAD" in overlay["refs"]

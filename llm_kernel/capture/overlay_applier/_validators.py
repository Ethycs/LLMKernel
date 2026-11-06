from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

from llm_kernel.capture.overlay_applier._helpers import *  # noqa: F401,F403

logger = logging.getLogger(__name__)

def _validate_split_cell(state: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Validate + apply ``split_cell`` per BSP-007 §6.2.

    V1 split is overlay-only -- the underlying turn DAG is unchanged
    and the cell record at ``cell_id`` is left intact. We synthesize a
    sibling cell record so subsequent ops in the same commit can
    address the split.
    """
    cell_id = _require_str(params, "cell_id", "split_cell")
    at = params.get("at")
    cells = state.get("cells")
    if not isinstance(cells, dict) or cell_id not in cells:
        raise OverlayRejected(
            "K94", f"split_cell: unknown cell_id {cell_id!r}",
            details={"reason": "unknown_cell", "cell_id": cell_id},
        )
    _check_not_executing(state, cell_id, "split_cell", k_code="K95")

    # Per atom: the split point must be a turn boundary or a span
    # boundary, never inside a tool_use / tool_result / system /
    # result span. V1 enforces the structural shape of ``at``; the
    # detailed span-kind check is deferred to S5.5 (the cell-text
    # surface owns the span kind enumeration).
    if isinstance(at, dict):
        kind = at.get("kind")
        if kind == "char_offset":
            sp = at.get("span_index")
            co = at.get("char_offset")
            if not isinstance(sp, int) or sp < 0 or not isinstance(co, int) or co < 0:
                raise OverlayRejected(
                    "K94",
                    "split_cell: char_offset requires span_index>=0 and char_offset>=0",
                    details={"reason": "split_at_invalid",
                             "cell_id": cell_id, "at_turn_id": None},
                )
        elif kind == "span_boundary":
            idx = at.get("before_span_index")
            if not isinstance(idx, int) or idx < 0:
                raise OverlayRejected(
                    "K94",
                    "split_cell: span_boundary requires before_span_index>=0",
                    details={"reason": "split_at_invalid",
                             "cell_id": cell_id, "at_turn_id": None},
                )
        elif kind == "turn_boundary":
            tid = at.get("at_turn_id") or at.get("turn_id")
            if not isinstance(tid, str) or not tid:
                raise OverlayRejected(
                    "K94",
                    "split_cell: turn_boundary requires at_turn_id (non-empty string)",
                    details={"reason": "split_at_invalid",
                             "cell_id": cell_id, "at_turn_id": tid},
                )
        else:
            raise OverlayRejected(
                "K94",
                f"split_cell: unsupported at.kind={kind!r}",
                details={"reason": "split_at_invalid_kind",
                         "cell_id": cell_id, "at_turn_id": None},
            )
    elif "at_turn_id" in params:
        # Legacy / spec-§9 shape: { "kind": "split_cell", "cell_id":..., "at_turn_id":... }
        tid = params.get("at_turn_id")
        if not isinstance(tid, str) or not tid:
            raise OverlayRejected(
                "K94", "split_cell: at_turn_id must be a non-empty string",
                details={"reason": "split_at_invalid",
                         "cell_id": cell_id, "at_turn_id": tid},
            )
        # Detect the spec'd "would orphan a tool call" rejection: if
        # the caller annotates the turn as orphan-creating, surface K94.
        if params.get("would_orphan_tool_calls"):
            raise OverlayRejected(
                "K94",
                "split_cell: at_turn_id would separate a tool call from its parent turn",
                details={"reason": "would_orphan_tool_calls",
                         "cell_id": cell_id, "at_turn_id": tid},
            )
        # Detect the mid-turn rejection.
        if params.get("mid_turn"):
            raise OverlayRejected(
                "K94", "split_cell: at_turn_id is mid-turn (not a turn boundary)",
                details={"reason": "mid_turn",
                         "cell_id": cell_id, "at_turn_id": tid},
            )
    else:
        raise OverlayRejected(
            "K94", "split_cell: missing 'at' descriptor or 'at_turn_id'",
            details={"reason": "split_at_missing", "cell_id": cell_id},
        )

    # Apply: synthesize the new sibling cell record. Per atom
    # decision S4 (flag inheritance) the new cell inherits kind /
    # section_id / pinned / excluded / scratch from the source.
    new_cell_id = params.get("new_cell_id")
    if not isinstance(new_cell_id, str) or not new_cell_id:
        new_cell_id = f"{cell_id}__split_{uuid.uuid4().hex[:8]}"
    src = cells[cell_id]
    if isinstance(src, dict):
        sibling = {
            "kind": src.get("kind"),
            "bound_agent_id": src.get("bound_agent_id"),
            "section_id": src.get("section_id"),
            "capabilities": list(src.get("capabilities") or []),
        }
        for flag in ("pinned", "excluded", "scratch", "checkpoint"):
            if flag in src:
                sibling[flag] = src[flag]
        cells[new_cell_id] = sibling


def _validate_merge_cells(state: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Validate + apply ``merge_cells`` per BSP-007 §6.1 / §6.3.

    On success: ``cell_b`` is removed; ``cell_a`` carries a sub-turn
    addressing handle (we record ``merged_from`` on ``cell_a`` so
    Inspect mode can render the §6.4 sub-turn shape).
    """
    cell_a = _require_str(params, "cell_a", "merge_cells")
    cell_b = _require_str(params, "cell_b", "merge_cells")
    if cell_a == cell_b:
        raise OverlayRejected(
            "K93", "merge_cells: cell_a and cell_b must differ",
            details={"reason": "self_merge",
                     "cell_a": cell_a, "cell_b": cell_b},
        )
    cells = state.get("cells")
    if not isinstance(cells, dict):
        cells = {}
    if cell_a not in cells:
        raise OverlayRejected(
            "K93", f"merge_cells: cell_a {cell_a!r} not found",
            details={"reason": "unknown_cell",
                     "cell_a": cell_a, "cell_b": cell_b},
        )
    if cell_b not in cells:
        raise OverlayRejected(
            "K93", f"merge_cells: cell_b {cell_b!r} not found",
            details={"reason": "unknown_cell",
                     "cell_a": cell_a, "cell_b": cell_b},
        )

    rec_a = cells[cell_a]
    rec_b = cells[cell_b]

    # Same primary cell kind (BSP-007 §6.1, decision D5).
    kind_a = rec_a.get("kind") if isinstance(rec_a, dict) else None
    kind_b = rec_b.get("kind") if isinstance(rec_b, dict) else None
    if kind_a != kind_b:
        raise OverlayRejected(
            "K93",
            f"merge_cells: different_primary_kind ({kind_a!r} vs {kind_b!r})",
            details={"reason": "different_primary_kind",
                     "cell_a": cell_a, "cell_b": cell_b,
                     "kind_a": kind_a, "kind_b": kind_b},
        )

    # Reserved kinds error (atom note: tool|artifact|control|native
    # cannot be merged in V1).
    _RESERVED_MERGE_KINDS = {"tool", "artifact", "control", "native"}
    if kind_a in _RESERVED_MERGE_KINDS:
        raise OverlayRejected(
            "K93",
            f"merge_cells: reserved cell kind {kind_a!r} cannot be merged in V1",
            details={"reason": "reserved_kind_unmergeable",
                     "cell_a": cell_a, "cell_b": cell_b, "kind": kind_a},
        )

    # Same agent provenance for agent cells.
    if kind_a == "agent":
        bag_a = rec_a.get("bound_agent_id") if isinstance(rec_a, dict) else None
        bag_b = rec_b.get("bound_agent_id") if isinstance(rec_b, dict) else None
        if bag_a != bag_b:
            raise OverlayRejected(
                "K93",
                "merge_cells: different bound_agent_id",
                details={"reason": "different_agent_provenance",
                         "cell_a": cell_a, "cell_b": cell_b,
                         "bound_agent_id_a": bag_a,
                         "bound_agent_id_b": bag_b},
            )

    # Same section.
    sec_a = rec_a.get("section_id") if isinstance(rec_a, dict) else None
    sec_b = rec_b.get("section_id") if isinstance(rec_b, dict) else None
    if sec_a != sec_b:
        raise OverlayRejected(
            "K93", "merge_cells: cells live in different sections",
            details={"reason": "different_section",
                     "cell_a": cell_a, "cell_b": cell_b,
                     "section_a": sec_a, "section_b": sec_b},
        )

    # No pin / exclude / checkpoint boundary between them.
    for flag, marker_reason in (
        ("pinned", "pin_boundary"),
        ("excluded", "exclude_boundary"),
        ("checkpoint", "checkpoint_boundary"),
    ):
        if (
            (isinstance(rec_a, dict) and rec_a.get(flag))
            or (isinstance(rec_b, dict) and rec_b.get(flag))
        ):
            raise OverlayRejected(
                "K93",
                f"merge_cells: {marker_reason} between cells",
                details={"reason": marker_reason,
                         "cell_a": cell_a, "cell_b": cell_b},
            )

    # Neither cell may be currently executing (§6.1 / §6.3 / KB §22.7).
    _check_not_executing(state, cell_a, "merge_cells", k_code="K95")
    _check_not_executing(state, cell_b, "merge_cells", k_code="K95")

    # Re-merge of an already-merged cell forbidden (decision D6, K94 per atom).
    if isinstance(rec_a, dict) and rec_a.get("merged_from"):
        raise OverlayRejected(
            "K94",
            "merge_cells: cell_a already a merge result; split first to re-arrange",
            details={"reason": "already_merged", "cell_a": cell_a},
        )

    # Apply: stamp a sub-turn handle on cell_a and drop cell_b.
    if isinstance(rec_a, dict):
        merged_from = list(rec_a.get("merged_from") or [])
        merged_from.append(cell_b)
        rec_a["merged_from"] = merged_from
        # BSP-007 §6.4 -- sub-turn addressing flag (the renderer uses
        # this to expose ``cell:c_a.k`` resolution).
        rec_a["sub_turn_addressing"] = True
    cells.pop(cell_b, None)


def _validate_move_cell(state: Dict[str, Any], params: Dict[str, Any]) -> None:
    cell_id = _require_str(params, "cell_id", "move_cell")
    target_section_id = _require_str(
        params, "target_section_id", "move_cell",
    )
    position = params.get("position", params.get("position_index"))
    if not isinstance(position, int) or position < 0:
        raise OverlayRejected(
            "K90", "move_cell: position (non-negative int) is required",
            details={"reason": "position_required", "cell_id": cell_id},
        )
    cells = state.get("cells")
    if not isinstance(cells, dict) or cell_id not in cells:
        raise OverlayRejected(
            "K90", f"move_cell: unknown cell_id {cell_id!r}",
            details={"reason": "unknown_cell", "cell_id": cell_id},
        )
    sections = state.setdefault("sections", {})
    if not isinstance(sections, dict):
        sections = {}
        state["sections"] = sections
    if target_section_id not in sections:
        raise OverlayRejected(
            "K90", f"move_cell: unknown target_section_id {target_section_id!r}",
            details={"reason": "unknown_section",
                     "section_id": target_section_id, "cell_id": cell_id},
        )
    _check_not_executing(state, cell_id, "move_cell", k_code="K95")
    record = cells[cell_id]
    if isinstance(record, dict):
        # Cross-checkpoint forbidden (decision M2). If the cell carries
        # a checkpoint flag it cannot move.
        if record.get("checkpoint"):
            raise OverlayRejected(
                "K93",
                "move_cell: cross-checkpoint relocation forbidden (decision M2)",
                details={"reason": "checkpoint_boundary",
                         "cell_id": cell_id},
            )
        old_section_id = record.get("section_id")
        record["section_id"] = target_section_id
        # Maintain the dual-representation invariant (D8).
        if isinstance(old_section_id, str) and old_section_id in sections:
            old_sec = sections[old_section_id]
            if isinstance(old_sec, dict):
                cr = old_sec.get("cell_range")
                if isinstance(cr, list) and cell_id in cr:
                    cr.remove(cell_id)
        target_sec = sections[target_section_id]
        if isinstance(target_sec, dict):
            cr = target_sec.setdefault("cell_range", [])
            if isinstance(cr, list):
                # Clamp position to len(cr).
                if position > len(cr):
                    position = len(cr)
                cr.insert(position, cell_id)


def _validate_create_section(state: Dict[str, Any], params: Dict[str, Any]) -> None:
    section_id = _require_str(params, "section_id", "create_section")
    title = _require_str(params, "title", "create_section")
    parent = params.get("parent_section_id")
    if parent is not None:
        # Decision D3 -- flat in V1.
        raise OverlayRejected(
            "K90",
            "create_section: parent_section_id MUST be null in V1 (decision D3)",
            details={"reason": "nested_sections_forbidden",
                     "section_id": section_id, "parent_section_id": parent},
        )
    sections = state.setdefault("sections", {})
    if not isinstance(sections, dict):
        sections = {}
        state["sections"] = sections
    if section_id in sections:
        raise OverlayRejected(
            "K90",
            f"create_section: section_id {section_id!r} already exists",
            details={"reason": "duplicate_section_id",
                     "section_id": section_id},
        )
    sections[section_id] = {
        "id": section_id,
        "title": title,
        "parent_section_id": None,
        "cell_range": [],
        "summary": params.get("summary"),
        "status": params.get("status", "open"),
        "collapsed": bool(params.get("collapsed", False)),
        "flow_policy": None,
    }


def _validate_delete_section(state: Dict[str, Any], params: Dict[str, Any]) -> None:
    section_id = _require_str(params, "section_id", "delete_section")
    sections = state.get("sections") or {}
    if not isinstance(sections, dict) or section_id not in sections:
        raise OverlayRejected(
            "K90", f"delete_section: unknown_section {section_id!r}",
            details={"reason": "unknown_section",
                     "section_id": section_id},
        )
    sec = sections[section_id]
    if isinstance(sec, dict):
        cr = sec.get("cell_range") or []
        if isinstance(cr, list) and len(cr) > 0:
            raise OverlayRejected(
                "K90",
                "delete_section: section_not_empty (decision SD1)",
                details={"reason": "section_not_empty",
                         "section_id": section_id,
                         "cell_count": len(cr)},
            )
    sections.pop(section_id, None)


def _validate_rename_section(state: Dict[str, Any], params: Dict[str, Any]) -> None:
    section_id = _require_str(params, "section_id", "rename_section")
    title = _require_str(params, "title", "rename_section")
    sections = state.get("sections") or {}
    if not isinstance(sections, dict) or section_id not in sections:
        raise OverlayRejected(
            "K90", f"rename_section: unknown_section {section_id!r}",
            details={"reason": "unknown_section",
                     "section_id": section_id},
        )
    sec = sections[section_id]
    if isinstance(sec, dict):
        sec["title"] = title


def _validate_move_cells_into_section(
    state: Dict[str, Any], params: Dict[str, Any],
) -> None:
    target_section_id = _require_str(
        params, "target_section_id", "move_cells_into_section",
    )
    cell_ids = params.get("cell_ids")
    if not isinstance(cell_ids, list) or not cell_ids:
        raise OverlayRejected(
            "K90",
            "move_cells_into_section: cell_ids must be a non-empty list",
            details={"reason": "cell_ids_required",
                     "section_id": target_section_id},
        )
    position = params.get("position", params.get("position_index", 0))
    if not isinstance(position, int) or position < 0:
        raise OverlayRejected(
            "K90", "move_cells_into_section: position (non-negative int) required",
            details={"reason": "position_required",
                     "section_id": target_section_id},
        )
    # Apply each as an individual move_cell op.
    for offset, cid in enumerate(cell_ids):
        if not isinstance(cid, str) or not cid:
            raise OverlayRejected(
                "K90",
                "move_cells_into_section: cell_ids entries must be non-empty strings",
                details={"reason": "cell_id_invalid",
                         "section_id": target_section_id,
                         "cell_id": cid},
            )
        _validate_move_cell(state, {
            "cell_id": cid,
            "target_section_id": target_section_id,
            "position": position + offset,
        })


#: Span names that are one half of a tool call (discipline/tool-calls-atomic).
_TOOL_CALL_HALF_SPAN_NAMES: FrozenSet[str] = frozenset({"tool_use", "tool_result"})
#: Terminal-status span names promoted as ``checkpoint`` (decision D7).
_CHECKPOINT_SPAN_NAMES: FrozenSet[str] = frozenset({"report_completion", "result"})
#: Section statuses a promote may target (v1-section-status-interruptibility).
_PROMOTE_ALLOWED_SECTION_STATUSES: Tuple[str, ...] = ("open", "complete")


def _infer_promote_kind(span: Dict[str, Any]) -> Optional[str]:
    """Decision D7: infer the promoted cell kind from the span.

    Returns ``None`` for half a tool call (``tool_use`` / ``tool_result``
    on its own), which the caller rejects.
    """
    name = span.get("name")
    attrs = span.get("attributes")
    if not isinstance(attrs, dict):
        attrs = {}
    if name in _TOOL_CALL_HALF_SPAN_NAMES:
        return None
    if (name in _CHECKPOINT_SPAN_NAMES
            or attrs.get("llmnb.tool_name") == "report_completion"):
        return "checkpoint"
    # propose_edit, agent_emit / text prose and whole tool-call spans.
    return "artifact"


def _validate_promote_span(state: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Validate + apply ``promote_span`` per the promote-span atom.

    Preconditions: the span exists on a turn in the zone; it is not
    half of a tool call; its source cell is not executing (K95); the
    target section (default: the source cell's section) has status
    ``open`` or ``complete`` (K95; ``complete`` needs
    ``operator_confirmed``). The new cell lands at the end of the
    target section's ``cell_range`` (decision SD3).
    """
    span_id = _require_str(params, "span_id", "promote_span")
    resolve = state.get("resolve_span")
    found = resolve(span_id) if callable(resolve) else None
    if not isinstance(found, dict) or not isinstance(found.get("span"), dict):
        raise OverlayRejected(
            "K90", f"promote_span: unknown span_id {span_id!r}",
            details={"reason": "unknown_span", "span_id": span_id},
        )
    span = found["span"]
    source_cell_id = found.get("cell_id")
    inferred_kind = _infer_promote_kind(span)
    if inferred_kind is None:
        raise OverlayRejected(
            "K90",
            f"promote_span: span {span_id!r} is half of a tool call "
            f"({span.get('name')!r}); promote the tool call as a unit",
            details={"reason": "tool_call_half", "span_id": span_id,
                     "span_name": span.get("name")},
        )
    cell_kind = params.get("cell_kind")
    if cell_kind is None:
        cell_kind = inferred_kind
    if cell_kind not in ("artifact", "checkpoint"):
        raise OverlayRejected(
            "K90",
            f"promote_span: cell_kind must be 'artifact' or 'checkpoint' "
            f"(got {cell_kind!r})",
            details={"reason": "cell_kind_invalid", "span_id": span_id,
                     "cell_kind": cell_kind},
        )
    if cell_kind != inferred_kind:
        # V1 honors the explicit kind but logs the D7 conflict.
        logger.warning(
            "promote_span: explicit cell_kind %r conflicts with inferred %r "
            "(span_id=%s)", cell_kind, inferred_kind, span_id,
            extra={"event.name": "wire-failure",
                   "llmnb.span_id": span_id},
        )
    if isinstance(source_cell_id, str) and source_cell_id:
        _check_not_executing(state, source_cell_id, "promote_span",
                             k_code="K95")
    cells = state.setdefault("cells", {})
    if not isinstance(cells, dict):
        cells = {}
        state["cells"] = cells
    section_id = params.get("section_id")
    if section_id is None:
        src = cells.get(source_cell_id) if isinstance(source_cell_id, str) else None
        if isinstance(src, dict):
            section_id = src.get("section_id")
    sections = state.get("sections") or {}
    if section_id is not None:
        if not isinstance(sections, dict) or section_id not in sections:
            raise OverlayRejected(
                "K90", f"promote_span: unknown_section {section_id!r}",
                details={"reason": "unknown_section", "span_id": span_id,
                         "section_id": section_id},
            )
        sec = sections[section_id]
        current_status = sec.get("status", "open") if isinstance(sec, dict) else "open"
        confirmed = bool(params.get("operator_confirmed")
                         or state.get("operator_confirmed"))
        if (current_status not in _PROMOTE_ALLOWED_SECTION_STATUSES
                or (current_status == "complete" and not confirmed)):
            raise OverlayRejected(
                "K95",
                f"promote_span: section {section_id!r} status "
                f"{current_status!r} blocks promotion",
                details={"reason": "overlay_section_status_blocks",
                         "span_id": span_id,
                         "section_id": section_id,
                         "current_status": current_status,
                         "required_status": list(_PROMOTE_ALLOWED_SECTION_STATUSES),
                         "operator_confirmed": confirmed},
            )
    new_cell_id = params.get("new_cell_id")
    if not isinstance(new_cell_id, str) or not new_cell_id:
        # Non-deterministic by default (Option A per promote-span atom
        # §"Cell-id determinism"). Callers who need replay-stable ids
        # MUST supply an explicit ``new_cell_id`` from a stable source.
        new_cell_id = f"prom_{span_id}_{uuid.uuid4().hex[:8]}"
    cells[new_cell_id] = {
        "kind": cell_kind,
        "bound_agent_id": None,
        "bound_span_id": span_id,
        "section_id": section_id,
        "capabilities": [],
        "turns": [],
    }
    # Position default: end of the target section (decision SD3).
    if isinstance(section_id, str) and section_id in sections:
        sec = sections[section_id]
        if isinstance(sec, dict):
            cr = sec.setdefault("cell_range", [])
            if isinstance(cr, list):
                cr.append(new_cell_id)


# PLAN-S5.5 Phase 1b — section status state machine.
#
# Allowed transitions (per PLAN-S5.5 §3 step 3 and
# decisions/v1-section-status-interruptibility):
#
#   open ↔ in_progress       (kernel-driven via RunFrame OR operator)
#   open ↔ complete          (operator only; free transition)
#   * → frozen               (operator only; freezes the section)
#   frozen → open            (operator only; explicit unfreeze)
#
# Everything else is K95 ``overlay_blocked_by_execution``. In
# particular: ``frozen → in_progress`` and ``frozen → complete``
# require unfreeze first; ``complete ↔ in_progress`` is not a
# permitted direct transition (operator should explicitly reopen).
# Same-status writes are no-ops (returned as success).
_SECTION_STATUSES: FrozenSet[str] = frozenset(
    {"open", "in_progress", "complete", "frozen"}
)
_ALLOWED_SECTION_TRANSITIONS: FrozenSet[tuple[str, str]] = frozenset({
    ("open", "in_progress"),
    ("open", "complete"),
    ("open", "frozen"),
    ("in_progress", "open"),
    ("in_progress", "frozen"),
    ("complete", "open"),
    ("complete", "frozen"),
    ("frozen", "open"),
})


def _validate_set_section_status(
    state: Dict[str, Any], params: Dict[str, Any],
) -> None:
    """``set_section_status`` operation — PLAN-S5.5 Phase 1b.

    Mutates ``sections[<id>].status`` if the (current, new) transition
    is in :data:`_ALLOWED_SECTION_TRANSITIONS`. Same-status writes
    succeed silently as a no-op.

    Rejections:

    * Unknown ``section_id`` → K90 ``unknown_section``.
    * ``new_status`` not in the 4-value enum → K90 ``invalid_status``.
    * Forbidden transition (e.g. ``frozen → in_progress``) → K95
      ``forbidden_section_transition`` (subclassed as
      ``overlay_blocked_by_execution`` per the K95 marker).
    """
    section_id = _require_str(params, "section_id", "set_section_status")
    new_status = _require_str(params, "new_status", "set_section_status")
    if new_status not in _SECTION_STATUSES:
        raise OverlayRejected(
            "K90",
            f"set_section_status: invalid status {new_status!r} "
            f"(must be one of {sorted(_SECTION_STATUSES)})",
            details={"reason": "invalid_status",
                     "section_id": section_id,
                     "new_status": new_status},
        )
    sections = state.get("sections") or {}
    if not isinstance(sections, dict) or section_id not in sections:
        raise OverlayRejected(
            "K90",
            f"set_section_status: unknown_section {section_id!r}",
            details={"reason": "unknown_section",
                     "section_id": section_id},
        )
    sec = sections[section_id]
    if not isinstance(sec, dict):
        raise OverlayRejected(
            "K90",
            f"set_section_status: section record malformed {section_id!r}",
            details={"reason": "section_record_malformed",
                     "section_id": section_id},
        )
    current_status = sec.get("status", "open")
    if not isinstance(current_status, str) or current_status not in _SECTION_STATUSES:
        # Defensive: an unrecognised legacy status; treat as ``open``
        # for transition purposes so the operator can recover.
        current_status = "open"
    if current_status == new_status:
        # Same-status write is a no-op success.
        return
    if (current_status, new_status) not in _ALLOWED_SECTION_TRANSITIONS:
        raise OverlayRejected(
            "K95",
            f"set_section_status: forbidden transition "
            f"{current_status!r} → {new_status!r}",
            details={"reason": "forbidden_section_transition",
                     "section_id": section_id,
                     "from_status": current_status,
                     "to_status": new_status},
        )
    sec["status"] = new_status


def _validate_checkpoint_section(
    state: Dict[str, Any], params: Dict[str, Any],
) -> None:
    section_id = _require_str(params, "section_id", "checkpoint_section")
    summary = params.get("summary")
    sections = state.get("sections") or {}
    if not isinstance(sections, dict) or section_id not in sections:
        raise OverlayRejected(
            "K90", f"checkpoint_section: unknown_section {section_id!r}",
            details={"reason": "unknown_section",
                     "section_id": section_id},
        )
    sec = sections[section_id]
    if isinstance(sec, dict):
        sec["summary"] = summary if isinstance(summary, str) else None
        sec["status"] = "complete"


def _validate_update_ordering(
    state: Dict[str, Any], params: Dict[str, Any],
) -> None:
    """Reorder cells inside a section per an explicit ordering list."""
    section_id = _require_str(params, "section_id", "update_ordering")
    new_order = params.get("cell_order")
    if not isinstance(new_order, list):
        raise OverlayRejected(
            "K90", "update_ordering: cell_order must be a list",
            details={"reason": "cell_order_required",
                     "section_id": section_id},
        )
    sections = state.get("sections") or {}
    if not isinstance(sections, dict) or section_id not in sections:
        raise OverlayRejected(
            "K90", f"update_ordering: unknown_section {section_id!r}",
            details={"reason": "unknown_section",
                     "section_id": section_id},
        )
    sec = sections[section_id]
    if isinstance(sec, dict):
        existing = list(sec.get("cell_range") or [])
        if sorted(existing) != sorted(new_order):
            raise OverlayRejected(
                "K90",
                "update_ordering: cell_order must be a permutation of cell_range",
                details={"reason": "cell_order_mismatch",
                         "section_id": section_id},
            )
        sec["cell_range"] = list(new_order)


def _validate_add_overlay(state: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Per-turn render-time overlay (BSP-002 §12 hook).

    V1 stores the overlay payload verbatim under
    ``state["per_turn_overlays"][turn_id][overlay_id]`` so the
    extension can re-render. The actual rendering / annotation
    semantics are X-EXT scope; the writer's job is persistence.
    """
    turn_id = _require_str(params, "turn_id", "add_overlay")
    overlay_id = _require_str(params, "overlay_id", "add_overlay")
    payload = params.get("payload") or {}
    if not isinstance(payload, dict):
        raise OverlayRejected(
            "K90", "add_overlay: payload must be a dict",
            details={"reason": "payload_invalid",
                     "turn_id": turn_id, "overlay_id": overlay_id},
        )
    overlays = state.setdefault("per_turn_overlays", {})
    if not isinstance(overlays, dict):
        overlays = {}
        state["per_turn_overlays"] = overlays
    bucket = overlays.setdefault(turn_id, {})
    bucket[overlay_id] = _deepcopy_json(payload)


def _validate_move_overlay_ref(
    state: Dict[str, Any], params: Dict[str, Any],
) -> None:
    """Move a per-turn overlay between turns (BSP-002 §12)."""
    overlay_id = _require_str(params, "overlay_id", "move_overlay_ref")
    from_turn_id = _require_str(params, "from_turn_id", "move_overlay_ref")
    to_turn_id = _require_str(params, "to_turn_id", "move_overlay_ref")
    overlays = state.get("per_turn_overlays") or {}
    if (
        not isinstance(overlays, dict)
        or from_turn_id not in overlays
        or overlay_id not in overlays[from_turn_id]
    ):
        raise OverlayRejected(
            "K90",
            f"move_overlay_ref: overlay {overlay_id!r} not found on turn "
            f"{from_turn_id!r}",
            details={"reason": "overlay_not_found",
                     "overlay_id": overlay_id, "turn_id": from_turn_id},
        )
    payload = overlays[from_turn_id].pop(overlay_id)
    bucket = overlays.setdefault(to_turn_id, {})
    bucket[overlay_id] = payload


def _validate_set_cell_metadata(
    state: Dict[str, Any], params: Dict[str, Any],
) -> None:
    """Generic cell metadata setter wrapped inside an overlay commit.

    The full BSP-005 ``set_cell_metadata`` validator (kind enum,
    per-kind constraints, normalize bound_agent_id) lives on the
    writer; for atomic-rollback safety inside the overlay commit we
    apply a minimal write to ``state["cells"][cell_id]`` here. Callers
    that need the strict validator should submit a standalone
    ``set_cell_metadata`` intent BEFORE the overlay commit (the writer
    rejects bad parameters with K42 before they reach the applier).
    """
    cell_id = _require_str(params, "cell_id", "set_cell_metadata")
    cells = state.setdefault("cells", {})
    if not isinstance(cells, dict):
        cells = {}
        state["cells"] = cells
    record = dict(cells.get(cell_id, {}))
    for key, value in params.items():
        if key in ("cell_id", "kind"):
            # ``kind`` lands verbatim only when supplied; we don't
            # invent it here.
            if key == "kind" and isinstance(value, str):
                record["kind"] = value
            continue
        record[key] = value
    cells[cell_id] = record


# Operation kind -> validator/applier dispatch.
_OPERATION_DISPATCH: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], None]] = {
    "set_pin":            lambda s, p: _apply_set_flag_op(s, "set_pin", "pinned", p),
    "set_exclude":        lambda s, p: _apply_set_flag_op(s, "set_exclude", "excluded", p),
    "set_scratch":        lambda s, p: _apply_set_flag_op(s, "set_scratch", "scratch", p),
    "set_checkpoint":     lambda s, p: _apply_set_flag_op(s, "set_checkpoint", "checkpoint", p),
    "set_cell_metadata":  _validate_set_cell_metadata,
    "update_ordering":    _validate_update_ordering,
    "add_overlay":        _validate_add_overlay,
    "move_overlay_ref":   _validate_move_overlay_ref,
    "split_cell":         _validate_split_cell,
    "merge_cells":        _validate_merge_cells,
    "move_cell":          _validate_move_cell,
    "create_section":     _validate_create_section,
    "delete_section":     _validate_delete_section,
    "rename_section":     _validate_rename_section,
    "move_cells_into_section": _validate_move_cells_into_section,
    "set_section_status": _validate_set_section_status,
    "promote_span":       _validate_promote_span,
    "checkpoint_section": _validate_checkpoint_section,
}


__all__ = [
    "_validate_split_cell",
    "_validate_merge_cells",
    "_validate_move_cell",
    "_validate_create_section",
    "_validate_delete_section",
    "_validate_rename_section",
    "_validate_move_cells_into_section",
    "_validate_promote_span",
    "_validate_set_section_status",
    "_validate_checkpoint_section",
    "_validate_update_ordering",
    "_validate_add_overlay",
    "_validate_move_overlay_ref",
    "_validate_set_cell_metadata",
    "_SECTION_STATUSES",
    "_ALLOWED_SECTION_TRANSITIONS",
    "_OPERATION_DISPATCH",
]

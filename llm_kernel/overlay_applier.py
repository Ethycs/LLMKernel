"""Overlay-graph applier (BSP-007 §4 / K-OVERLAY).

This module hosts the operator-side, git-style overlay layer that sits
ABOVE the immutable agent turn DAG. It mints overlay commits, advances
HEAD, manages named refs, and validates the 17 V1 operation kinds
enumerated in BSP-007 §3 + the per-operation atom contracts under
``docs/atoms/operations/``.

Storage shape (BSP-007 §2.2)::

    metadata.rts.zone.overlay = {
        "commits": [ <OverlayCommit>, ... ],   # append-only
        "refs": {
            "HEAD": <commit_id>,
            "<tag>": <commit_id>,
            ...
        },
    }

The applier is a thin layer over the writer's in-memory dicts: the
writer holds the lock and owns the snapshot version bump; this module
owns *what* changes inside the lock. Atomicity is provided by the
deep-copy-and-swap pattern in :func:`apply_commit` -- on first failure
no state is mutated.

Failure modes (BSP-007 §7)::

    K90  overlay_commit_invalid          (per-op validation rejected)
    K91  overlay_commit_unreachable      (commit_id not in commits[])
    K92  overlay_ref_conflict            (tag name collision in V1)
    K93  overlay_merge_rejected          (cell-merge precondition)
    K94  overlay_split_rejected          (cell-split precondition)
    K95  overlay_blocked_by_execution    (in-flight run on target cell)

The 17 V1 operation kinds dispatched within ``apply_commit``:

* Cell-flag toggles: ``set_pin``, ``set_exclude``, ``set_scratch``,
  ``set_checkpoint``. All four delegate into the writer's
  :meth:`MetadataWriter._set_cell_flag` (so the parse-derived flag
  schema and S5.0 cell-text contract stay single-sourced).
* Generic ``set_cell_metadata`` mutation (parameterized flag set).
* Cell ordering / overlay management:
  ``update_ordering``, ``add_overlay``, ``move_overlay_ref``.
* Cell-structural: ``split_cell``, ``merge_cells``, ``move_cell``.
* Section-level: ``create_section``, ``delete_section``,
  ``rename_section``, ``move_cells_into_section``.
* Promote / checkpoint: ``promote_span``, ``checkpoint_section``.

Per BSP-007 §4.1, the commit is the unit of atomicity: validation runs
ALL ops, and either every op applies or none does.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

#: V1 operation kinds dispatched within an OverlayCommit.operations[].
#: Kinds outside this set raise K90 with reason ``unknown_operation_kind``
#: (forward-compat: V2+ may add kinds; V1 readers reject them per §10).
OVERLAY_OPERATION_KINDS: FrozenSet[str] = frozenset({
    # Cell-flag toggles (delegate into MetadataWriter._set_cell_flag).
    "set_pin",
    "set_exclude",
    "set_scratch",
    "set_checkpoint",
    # Per-cell metadata writeback (parameterized flag set + kind).
    "set_cell_metadata",
    # Cell-ordering / per-turn overlay management.
    "update_ordering",
    "add_overlay",
    "move_overlay_ref",
    # Cell-structural (the meat of §6 invariants).
    "split_cell",
    "merge_cells",
    "move_cell",
    # Section-level (§3 + atoms/operations/*-section.md).
    "create_section",
    "delete_section",
    "rename_section",
    "move_cells_into_section",
    # Promote / checkpoint.
    "promote_span",
    "checkpoint_section",
})

#: Reserved ref names that operator code cannot create directly.
#: ``HEAD`` is managed by ``apply_commit`` / ``revert_to_commit``.
#: ``_*`` is the kernel-private prefix per BSP-007 §2.3.
_RESERVED_REF_NAMES: FrozenSet[str] = frozenset({"HEAD"})


# ---------------------------------------------------------------------
# Custom rejection exception. Carries the K-code so the writer's
# dispatcher can surface the spec'd failure mode (vs. the catch-all
# K42 "validation_failed" path).
# ---------------------------------------------------------------------


class OverlayRejected(Exception):
    """Overlay applier rejected a commit / primitive call.

    Carries:

    * ``code`` -- one of K90/K91/K92/K93/K94/K95.
    * ``marker`` -- the per-K marker name from BSP-007 §7.
    * ``reason`` -- short machine-friendly reason (e.g. ``pin_boundary``).
    * ``details`` -- optional dict of structured fields the operator
      UI surfaces verbatim (e.g. ``failed_operation_index``,
      ``cell_id``, ``commit_id``).
    """

    _MARKERS: Dict[str, str] = {
        "K90": "overlay_commit_invalid",
        "K91": "overlay_commit_unreachable",
        "K92": "overlay_ref_conflict",
        "K93": "overlay_merge_rejected",
        "K94": "overlay_split_rejected",
        "K95": "overlay_blocked_by_execution",
    }

    def __init__(
        self,
        code: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if code not in self._MARKERS:
            raise ValueError(f"OverlayRejected: unknown K-code {code!r}")
        self.code = code
        self.marker = self._MARKERS[code]
        self.reason = reason
        self.details = dict(details) if details else {}
        super().__init__(f"{code} {self.marker}: {reason}")


# ---------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _deepcopy_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deepcopy_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deepcopy_json(v) for v in obj]
    return obj


# Rough-monotonic counter so two commits minted in the same wall clock
# tick still sort deterministically. ULID would be ideal; the project
# does not yet vendor one (uuid.uuid4 is the convention; manifest_id in
# context_packer uses it). For stable chronological sort we prefix the
# timestamp.
_COMMIT_COUNTER: List[int] = [0]


def mint_commit_id() -> str:
    """Mint a sortable, unique overlay commit id.

    Format: ``ovc_<14-char-millis>_<hex>`` where millis is base36-ish
    timestamp. We use uuid4 hex for the entropy half; tests may
    monkeypatch this function for determinism.
    """
    _COMMIT_COUNTER[0] += 1
    millis = int(time.time() * 1000)
    return f"ovc_{millis:013d}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------
# Overlay state accessors. The writer holds ``self._zone`` -- this
# module operates on the ``self._zone["overlay"]`` substructure. We
# keep the helpers free-standing so unit tests can drive them with a
# bare dict if useful.
# ---------------------------------------------------------------------


def ensure_overlay_state(zone: Dict[str, Any]) -> Dict[str, Any]:
    """Initialise the overlay substructure in ``zone`` if missing.

    Locked-interface contract (per the brief): the dict ALWAYS has
    ``commits`` (list) and ``refs`` (dict). HEAD-less state is
    ``commits == []`` and ``refs == {}``.

    Returns the (possibly newly-created) ``zone["overlay"]`` dict.
    """
    overlay = zone.get("overlay")
    if not isinstance(overlay, dict):
        overlay = {"commits": [], "refs": {}}
        zone["overlay"] = overlay
        return overlay
    overlay.setdefault("commits", [])
    overlay.setdefault("refs", {})
    if not isinstance(overlay["commits"], list):
        overlay["commits"] = []
    if not isinstance(overlay["refs"], dict):
        overlay["refs"] = {}
    return overlay


def commit_index(overlay: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return ``commit_id -> commit`` map over the overlay's commits."""
    out: Dict[str, Dict[str, Any]] = {}
    for c in overlay.get("commits", []) or []:
        if isinstance(c, dict):
            cid = c.get("commit_id")
            if isinstance(cid, str) and cid:
                out[cid] = c
    return out


def head_commit_id(overlay: Dict[str, Any]) -> Optional[str]:
    refs = overlay.get("refs") or {}
    head = refs.get("HEAD") if isinstance(refs, dict) else None
    return head if isinstance(head, str) and head else None


# ---------------------------------------------------------------------
# Operation-kind validators / appliers. Each takes the in-flight
# ``state`` view (a dict with the writer's relevant in-memory maps) and
# the op parameters, and either mutates state in place OR raises
# :class:`OverlayRejected` with the appropriate K-code.
#
# State view contract (the writer builds this; the applier never
# touches the writer's locks directly):
#
#   {
#     "cells":    Dict[cell_id, cell_record],     # MUTATED in place
#     "sections": Dict[section_id, section_record], # MUTATED in place
#     "overlay":  Dict[str, Any],                   # MUTATED in place
#     # Closures the writer injects so we don't import its module:
#     "set_cell_flag":           Callable[[cell_id, flag, value], bool],
#     "is_cell_executing":       Callable[[cell_id], bool],
#     "deepcopy":                Callable[[Any], Any] (defaults to local),
#   }
#
# A `state["sections"]` is materialised lazily -- if the writer hasn't
# populated it yet (V1 has no first-class section table outside the
# overlay), the applier creates it on the in-state dict and the writer
# round-trips it via the snapshot path.
# ---------------------------------------------------------------------


def _require_str(params: Dict[str, Any], key: str, op_kind: str) -> str:
    val = params.get(key)
    if not isinstance(val, str) or not val:
        raise OverlayRejected(
            "K90", f"{op_kind}: {key!r} required (non-empty string)",
            details={"reason": f"missing_{key}", "op_kind": op_kind},
        )
    return val


def _check_not_executing(state: Dict[str, Any], cell_id: str, op_kind: str,
                          k_code: str = "K95") -> None:
    pred = state.get("is_cell_executing")
    if callable(pred):
        try:
            executing = bool(pred(cell_id))
        except Exception:
            executing = False
        if executing:
            raise OverlayRejected(
                k_code,
                f"{op_kind}: cell {cell_id!r} is currently executing",
                details={
                    "reason": "cell_executing",
                    "cell_id": cell_id,
                    "op_kind": op_kind,
                },
            )


def _apply_set_flag_op(
    state: Dict[str, Any],
    op_kind: str,
    flag_name: str,
    params: Dict[str, Any],
) -> None:
    """Shared body for set_pin / set_exclude / set_scratch / set_checkpoint.

    Each of those ops carries ``cell_id`` + ``value`` (default True).
    The mutation matches the writer's
    :meth:`MetadataWriter._handle_set_cell_metadata` flag-key path
    (``record[flag_key] = bool(params[flag_key])``) -- the four flag
    keys ``pinned`` / ``excluded`` / ``scratch`` / ``checkpoint`` are
    written verbatim onto the cell record, which is the canonical V1
    storage shape. We mutate the work-state ``cells`` dict directly
    so the applier's swap-on-success / rollback path covers the
    change atomically.
    """
    cell_id = _require_str(params, "cell_id", op_kind)
    value = bool(params.get("value", True))
    cells = state.get("cells")
    if not isinstance(cells, dict) or cell_id not in cells:
        raise OverlayRejected(
            "K90", f"{op_kind}: unknown cell_id {cell_id!r}",
            details={"reason": "unknown_cell", "cell_id": cell_id,
                     "op_kind": op_kind},
        )
    _check_not_executing(state, cell_id, op_kind)
    record = cells[cell_id]
    if not isinstance(record, dict):
        record = {}
        cells[cell_id] = record
    record[flag_name] = value


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


def _validate_promote_span(state: Dict[str, Any], params: Dict[str, Any]) -> None:
    span_id = _require_str(params, "span_id", "promote_span")
    cell_kind = params.get("cell_kind", "artifact")
    if cell_kind not in ("artifact", "checkpoint"):
        raise OverlayRejected(
            "K90",
            f"promote_span: cell_kind must be 'artifact' or 'checkpoint' "
            f"(got {cell_kind!r})",
            details={"reason": "cell_kind_invalid", "span_id": span_id,
                     "cell_kind": cell_kind},
        )
    section_id = params.get("section_id")
    cells = state.setdefault("cells", {})
    if not isinstance(cells, dict):
        cells = {}
        state["cells"] = cells
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
    # If a section_id is provided and exists, append the new cell.
    sections = state.get("sections") or {}
    if isinstance(section_id, str) and section_id in sections:
        sec = sections[section_id]
        if isinstance(sec, dict):
            cr = sec.setdefault("cell_range", [])
            if isinstance(cr, list):
                cr.append(new_cell_id)


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
    "promote_span":       _validate_promote_span,
    "checkpoint_section": _validate_checkpoint_section,
}


def dispatch_operation(
    state: Dict[str, Any],
    op: Dict[str, Any],
    op_index: int,
) -> None:
    """Validate + apply one operation against ``state``.

    Raises :class:`OverlayRejected` on failure; mutates ``state`` in
    place on success. ``op_index`` is the 0-based position in the
    parent commit's ``operations[]`` array; we attach it to K90 so the
    operator UI can highlight the offending op.
    """
    if not isinstance(op, dict):
        raise OverlayRejected(
            "K90", f"operations[{op_index}]: each op must be a dict",
            details={"failed_operation_index": op_index,
                     "reason": "op_not_dict"},
        )
    # Accept either ``kind`` or ``op`` for the op-type field. Atoms
    # like split-cell.md use ``op``; the §3 enumeration uses ``kind``.
    kind = op.get("kind") or op.get("op")
    if not isinstance(kind, str) or not kind:
        raise OverlayRejected(
            "K90",
            f"operations[{op_index}]: 'kind' (or 'op') required",
            details={"failed_operation_index": op_index,
                     "reason": "kind_required"},
        )
    if kind not in OVERLAY_OPERATION_KINDS:
        raise OverlayRejected(
            "K90",
            f"operations[{op_index}]: unknown_operation_kind {kind!r}",
            details={"failed_operation_index": op_index,
                     "reason": "unknown_operation_kind",
                     "op_kind": kind},
        )
    validator = _OPERATION_DISPATCH[kind]
    try:
        validator(state, op)
    except OverlayRejected as exc:
        # Augment with the operation index so the K-envelope surfaces
        # the offending position even when the validator only knew the
        # cell/section context.
        exc.details.setdefault("failed_operation_index", op_index)
        exc.details.setdefault("op_kind", kind)
        raise


# ---------------------------------------------------------------------
# Primitives. apply_commit / revert_to_commit / create_ref / diff.
# ---------------------------------------------------------------------


def _replace_dict_in_place(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Mutate ``target`` in place so it equals ``source``."""
    target.clear()
    target.update(source)


def apply_commit(
    state: Dict[str, Any],
    operations: List[Dict[str, Any]],
    *,
    message: str = "",
    author: str = "operator",
) -> Tuple[str, Dict[str, Any]]:
    """Atomic apply of one commit (BSP-007 §4.1).

    Deep-copies the ``cells`` / ``sections`` / ``overlay`` /
    ``per_turn_overlays`` substructures of ``state`` into a working
    dict, dispatches every op, mints a commit, and -- on success --
    swaps the working dict back into the live sub-dicts IN PLACE (so
    the writer's references to those dicts stay valid). The live
    sub-dicts are obtained via ``state["_install_zone_subdict"]``, a
    writer-supplied closure that creates the empty sub-dict in
    ``self._zone`` only on success (so a rejected commit leaves
    ``self._zone`` byte-identical to its pre-call state).

    Returns ``(commit_id, commit_record)`` on success.

    Raises :class:`OverlayRejected` (K90 / K93 / K94 / K95) on the
    first per-op failure. ``state``'s sub-dicts are left UNCHANGED on
    failure (the work copies are dropped on the floor).
    """
    if not isinstance(operations, list) or not operations:
        raise OverlayRejected(
            "K90", "apply_commit: operations[] must be a non-empty list",
            details={"reason": "operations_required"},
        )

    # Deep-copy into a working state. Closures (the test seam +
    # writer-side helpers) ride through unchanged. We DO NOT pass the
    # writer's ``set_cell_flag`` closure through; the dispatcher writes
    # the flag onto ``work_state["cells"][cell_id]`` directly so the
    # mutation participates in the swap-on-success / rollback path.
    work_state: Dict[str, Any] = {
        "cells":             _deepcopy_json(state.get("cells") or {}),
        "sections":          _deepcopy_json(state.get("sections") or {}),
        "overlay":           _deepcopy_json(state.get("overlay") or {}),
        "per_turn_overlays": _deepcopy_json(state.get("per_turn_overlays") or {}),
        # The execution-in-flight predicate inspects writer-level
        # state; passing it through is safe because it is read-only.
        "is_cell_executing": state.get("is_cell_executing"),
    }

    # Validate + apply every op. On the first failure, raise without
    # touching live state (the work copies are dropped on the floor).
    for idx, op in enumerate(operations):
        dispatch_operation(work_state, op, idx)

    # Mint commit + advance HEAD on the WORK overlay before swap-back.
    # ``ensure_overlay_state`` operates on a parent dict whose
    # ``overlay`` *key* it (re-)initialises; pass ``work_state`` so the
    # existing overlay (with prior commits) is preserved. Passing
    # ``work_state["overlay"]`` would have it look for a nested
    # ``overlay.overlay`` key, find none, and silently replace the
    # overlay with a fresh empty one -- losing every prior commit.
    overlay = ensure_overlay_state(work_state)
    parent_id = head_commit_id(overlay)
    commit_id = mint_commit_id()
    commit_record: Dict[str, Any] = {
        "commit_id": commit_id,
        "parent_id": parent_id,
        "author": author or "operator",
        "timestamp": _utc_now_iso(),
        "message": message or "",
        "operations": _deepcopy_json(operations),
    }
    overlay["commits"].append(commit_record)
    overlay["refs"]["HEAD"] = commit_id

    # SUCCESS PATH: install the work copies into the writer's live
    # state via the writer-supplied installer (which makes empty
    # sub-dicts on demand the first time they're needed). For
    # ``cells`` we rely on the live writer dict that ``state["cells"]``
    # already aliases; the others are new sub-dicts under
    # ``self._zone`` that the installer creates lazily.
    install = state.get("_install_zone_subdict")
    live_cells = state["cells"]
    if isinstance(live_cells, dict):
        _replace_dict_in_place(live_cells, work_state["cells"])
    if callable(install):
        live_sections = install("sections")
        live_overlay = install("overlay")
        live_overlays_per_turn = install("per_turn_overlays")
        _replace_dict_in_place(live_sections, work_state["sections"])
        _replace_dict_in_place(live_overlay, overlay)
        _replace_dict_in_place(live_overlays_per_turn,
                                work_state["per_turn_overlays"])
    else:
        # Fallback (no writer installer): mutate the state dict
        # directly. Used by direct unit tests that drive the applier
        # without the writer.
        state["sections"] = work_state["sections"]
        state["overlay"] = overlay
        state["per_turn_overlays"] = work_state["per_turn_overlays"]
    return commit_id, commit_record


def _resolve_overlay_for_mutation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return the writer-live ``overlay`` dict (creating it if absent).

    Uses the writer-supplied ``_install_zone_subdict`` closure when
    present (the production path); falls back to a direct
    ``state["overlay"]`` mutation when not (unit-test / library path).
    """
    install = state.get("_install_zone_subdict")
    if callable(install):
        live_overlay = install("overlay")
    else:
        live_overlay = state.get("overlay")
        if not isinstance(live_overlay, dict):
            live_overlay = {}
            state["overlay"] = live_overlay
    # ``live_overlay`` IS the overlay dict, not a parent. Don't call
    # ``ensure_overlay_state`` on it (that would treat ``live_overlay``
    # as a parent and try to find a nested ``overlay`` key, finding none
    # and silently building a fresh empty dict at ``live_overlay["overlay"]``).
    if not isinstance(live_overlay.get("commits"), list):
        live_overlay["commits"] = []
    if not isinstance(live_overlay.get("refs"), dict):
        live_overlay["refs"] = {}
    return live_overlay


def revert_to_commit(state: Dict[str, Any], commit_id: str) -> str:
    """Move HEAD to ``commit_id`` (BSP-007 §4.2).

    Returns the new HEAD commit_id. Raises K91 when the commit is
    unknown. Does NOT remove commits from ``commits[]`` -- the
    git-reflog semantics.
    """
    if not isinstance(commit_id, str) or not commit_id:
        raise OverlayRejected(
            "K91", "revert_to_commit: commit_id is required",
            details={"reason": "commit_id_required"},
        )
    # First check the read-only overlay; only install on the success
    # path so a K91 rejection doesn't dirty self._zone.
    read_overlay = state.get("overlay") or {}
    if not isinstance(read_overlay, dict):
        read_overlay = {}
    idx = commit_index(read_overlay)
    if commit_id not in idx:
        raise OverlayRejected(
            "K91",
            f"revert_to_commit: commit_id {commit_id!r} not in commits[]",
            details={"reason": "commit_not_found",
                     "commit_id": commit_id},
        )
    overlay = _resolve_overlay_for_mutation(state)
    overlay["refs"]["HEAD"] = commit_id
    return commit_id


def create_ref(
    state: Dict[str, Any],
    name: str,
    commit_id: str,
) -> Dict[str, Any]:
    """Create a named ref at ``commit_id`` (BSP-007 §4.4 / V1 tag).

    Raises K91 when ``commit_id`` is unknown; K92 on collision; K90 on
    invalid / reserved name.
    """
    if not isinstance(name, str) or not name:
        raise OverlayRejected(
            "K90", "create_ref: name is required",
            details={"reason": "name_required"},
        )
    if name in _RESERVED_REF_NAMES or name.startswith("_"):
        raise OverlayRejected(
            "K90", f"create_ref: name {name!r} is reserved",
            details={"reason": "name_reserved", "name": name},
        )
    if not isinstance(commit_id, str) or not commit_id:
        raise OverlayRejected(
            "K91", "create_ref: commit_id is required",
            details={"reason": "commit_id_required", "name": name},
        )
    # Read-only validation first (no mutation on rejection).
    read_overlay = state.get("overlay") or {}
    if not isinstance(read_overlay, dict):
        read_overlay = {}
    idx = commit_index(read_overlay)
    if commit_id not in idx:
        raise OverlayRejected(
            "K91",
            f"create_ref: commit_id {commit_id!r} not in commits[]",
            details={"reason": "commit_not_found",
                     "name": name, "commit_id": commit_id},
        )
    refs_read = read_overlay.get("refs") or {}
    if isinstance(refs_read, dict) and name in refs_read:
        existing = refs_read[name]
        if existing != commit_id:
            # V1 tags are immutable.
            raise OverlayRejected(
                "K92",
                f"create_ref: name {name!r} already exists at "
                f"commit_id {existing!r}",
                details={"reason": "ref_name_conflict",
                         "name": name,
                         "existing_commit_id": existing,
                         "requested_commit_id": commit_id},
            )
        # Same commit_id is a benign no-op (idempotent re-create).
        return {"name": name, "commit_id": commit_id, "created": False}
    # Apply: install live overlay (only on success path).
    overlay = _resolve_overlay_for_mutation(state)
    overlay["refs"][name] = commit_id
    return {"name": name, "commit_id": commit_id, "created": True}


def diff(
    state: Dict[str, Any],
    commit_a: str,
    commit_b: str,
) -> List[Dict[str, Any]]:
    """Return the operations between two commits on the V1 linear chain.

    Both commit_ids must exist; otherwise K91. Walks ``commits[]`` from
    the older to the newer index and concatenates ``operations``.
    """
    # Read-only access -- never mutate state.
    overlay = state.get("overlay") or {}
    if not isinstance(overlay, dict):
        overlay = {}
    commits = overlay.get("commits") or []
    idx_map: Dict[str, int] = {}
    for i, c in enumerate(commits):
        if isinstance(c, dict) and isinstance(c.get("commit_id"), str):
            idx_map[c["commit_id"]] = i
    if commit_a not in idx_map:
        raise OverlayRejected(
            "K91", f"diff: commit_a {commit_a!r} not found",
            details={"reason": "commit_not_found", "commit_id": commit_a},
        )
    if commit_b not in idx_map:
        raise OverlayRejected(
            "K91", f"diff: commit_b {commit_b!r} not found",
            details={"reason": "commit_not_found", "commit_id": commit_b},
        )
    a_idx, b_idx = idx_map[commit_a], idx_map[commit_b]
    if a_idx > b_idx:
        a_idx, b_idx = b_idx, a_idx
    out: List[Dict[str, Any]] = []
    # Per §4.3 the diff is the concatenation of operations from
    # commits BETWEEN commit_a (exclusive) and commit_b (inclusive)
    # when descending from a to b.
    for i in range(a_idx + 1, b_idx + 1):
        c = commits[i]
        if not isinstance(c, dict):
            continue
        ops = c.get("operations") or []
        if isinstance(ops, list):
            out.extend(_deepcopy_json(ops))
    return out


__all__ = [
    "OVERLAY_OPERATION_KINDS",
    "OverlayRejected",
    "apply_commit",
    "commit_index",
    "create_ref",
    "diff",
    "dispatch_operation",
    "ensure_overlay_state",
    "head_commit_id",
    "mint_commit_id",
    "revert_to_commit",
]

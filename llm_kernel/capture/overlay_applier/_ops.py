from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

from llm_kernel.capture.overlay_applier._helpers import *  # noqa: F401,F403
from llm_kernel.capture.overlay_applier._validators import *  # noqa: F401,F403

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


def _assert_dual_representation_consistent(work_state: Dict[str, Any]) -> None:
    """Enforce ``cells.section_id`` ↔ ``sections.cell_range[]`` agreement.

    PLAN-S5.5 §3 step 4. Called by :func:`apply_commit` after every op
    in a commit has applied; raises :class:`OverlayRejected` with code
    K90 / reason ``section_membership_mismatch`` if the two views
    disagree. The error details include the offending ids so an
    operator UI can show the operator exactly which cell and section
    are inconsistent.

    Three kinds of mismatch detected:

    1. ``cell.section_id`` references a section that doesn't exist.
    2. ``cell.section_id`` references a section, but the cell isn't in
       that section's ``cell_range``.
    3. A ``section.cell_range[]`` entry references a cell whose
       ``cell.section_id`` doesn't match (or the cell doesn't exist).
    """
    cells = work_state.get("cells") or {}
    sections = work_state.get("sections") or {}
    if not isinstance(cells, dict) or not isinstance(sections, dict):
        return

    # Pass 1: walk cells -> sections direction.
    for cell_id, record in cells.items():
        if not isinstance(record, dict):
            continue
        sec_id = record.get("section_id")
        if sec_id is None:
            continue
        if not isinstance(sec_id, str):
            raise OverlayRejected(
                "K90",
                f"section_membership_mismatch: cell {cell_id!r} has "
                f"non-string section_id {sec_id!r}",
                details={"reason": "section_membership_mismatch",
                         "cell_id": cell_id, "section_id": sec_id,
                         "direction": "cell_to_section"},
            )
        section = sections.get(sec_id)
        if not isinstance(section, dict):
            raise OverlayRejected(
                "K90",
                f"section_membership_mismatch: cell {cell_id!r} "
                f"references unknown section {sec_id!r}",
                details={"reason": "section_membership_mismatch",
                         "cell_id": cell_id, "section_id": sec_id,
                         "direction": "cell_references_unknown_section"},
            )
        cell_range = section.get("cell_range") or []
        if not isinstance(cell_range, list) or cell_id not in cell_range:
            raise OverlayRejected(
                "K90",
                f"section_membership_mismatch: cell {cell_id!r} has "
                f"section_id={sec_id!r} but is not in that section's "
                f"cell_range",
                details={"reason": "section_membership_mismatch",
                         "cell_id": cell_id, "section_id": sec_id,
                         "direction": "cell_to_section"},
            )

    # Pass 2: walk sections -> cells direction.
    for sec_id, section in sections.items():
        if not isinstance(section, dict):
            continue
        cell_range = section.get("cell_range") or []
        if not isinstance(cell_range, list):
            continue
        for cell_id in cell_range:
            if not isinstance(cell_id, str):
                raise OverlayRejected(
                    "K90",
                    f"section_membership_mismatch: section {sec_id!r} "
                    f"cell_range contains non-string entry {cell_id!r}",
                    details={"reason": "section_membership_mismatch",
                             "section_id": sec_id, "cell_id": cell_id,
                             "direction": "section_range_invalid"},
                )
            record = cells.get(cell_id)
            if not isinstance(record, dict):
                raise OverlayRejected(
                    "K90",
                    f"section_membership_mismatch: section {sec_id!r} "
                    f"cell_range references unknown cell {cell_id!r}",
                    details={"reason": "section_membership_mismatch",
                             "section_id": sec_id, "cell_id": cell_id,
                             "direction": "section_references_unknown_cell"},
                )
            if record.get("section_id") != sec_id:
                raise OverlayRejected(
                    "K90",
                    f"section_membership_mismatch: section {sec_id!r} "
                    f"claims cell {cell_id!r} but cell.section_id="
                    f"{record.get('section_id')!r}",
                    details={"reason": "section_membership_mismatch",
                             "section_id": sec_id, "cell_id": cell_id,
                             "direction": "section_to_cell"},
                )


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

    # PLAN-S5.5 §3 step 4 — dual-representation invariant. After every
    # op in the commit applies, ``cells[<id>].section_id`` and
    # ``sections[<id>].cell_range[]`` MUST agree. Catches any op (or
    # composition of ops) that mutates one side without the other.
    # Atomic: a failure here drops the entire commit, leaving live
    # state byte-identical to its pre-call shape.
    _assert_dual_representation_consistent(work_state)

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
    "dispatch_operation",
    "_replace_dict_in_place",
    "_assert_dual_representation_consistent",
    "apply_commit",
    "_resolve_overlay_for_mutation",
    "revert_to_commit",
    "create_ref",
    "diff",
]

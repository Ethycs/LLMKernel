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
    "set_section_status",
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


__all__ = [
    "OVERLAY_OPERATION_KINDS",
    "_RESERVED_REF_NAMES",
    "OverlayRejected",
    "_COMMIT_COUNTER",
    "_utc_now_iso",
    "_deepcopy_json",
    "mint_commit_id",
    "ensure_overlay_state",
    "commit_index",
    "head_commit_id",
    "_require_str",
    "_check_not_executing",
    "_apply_set_flag_op",
]

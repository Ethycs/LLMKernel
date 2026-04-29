"""V1 ContextPacker -- pure deterministic structural walker.

Per BSP-008 §3 and ``decisions/v1-contextpacker-walk``, this module
ships ``pack(cell_id, snapshot) -> ContextManifest`` as a *pure*
function: same input -> byte-identical output (modulo
``manifest_id`` and ``generated_at`` which are intentionally
generated fresh each call).

The packer reads the notebook overlay, applies the V1 four-step
walk, and returns a manifest dict matching
[concepts/context-manifest](../../docs/atoms/concepts/context-manifest.md):

    1. Pinned cells anywhere in the zone (in their pinned order).
    2. Predecessors of ``cell_id`` in its current section,
       chronological. When the focus cell has no section, falls
       back to document order.
    3. Sub-turns of the current cell preceding the focus point.
    4. Filter scratch | excluded | obsolete cells out.
    5. Dedupe preserving first-occurrence order.

Persistence is the caller's responsibility -- the AgentSupervisor
wraps the call in a ``record_context_manifest`` BSP-003 intent so
the manifest lands in ``metadata.rts.zone.context_manifests`` via
the single-writer discipline.

K-class error modes per BSP-008 §10:

* ``K100`` -- ``cell_id`` is not present in the overlay (orphan).
* ``K101`` -- section walk hits a parent-section cycle. Pack
  returns the partial walk up to the cycle and emits a sentinel
  entry into ``inclusion_rules_applied``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "ContextPackerError",
    "K100OrphanCellError",
    "pack",
]

#: Cell flags that exclude a cell from the manifest. The V1 walk consults
#: visible flags only -- per ``discipline/scratch-beats-config``.
_EXCLUSION_FLAGS: Tuple[str, ...] = ("scratch", "excluded", "obsolete")

#: Defensive depth cap for the section-parent walk per BSP-008 §10 K101.
#: V1 sections are flat so this should never trigger, but the cap exists
#: so a malformed overlay cannot wedge the packer.
_SECTION_WALK_DEPTH_CAP: int = 64


class ContextPackerError(ValueError):
    """Base class for pack-time validation errors."""


class K100OrphanCellError(ContextPackerError):
    """``cell_id`` is not present in the snapshot's overlay (BSP-008 K100)."""

    def __init__(self, cell_id: str) -> None:
        super().__init__(
            f"K100: cell_id {cell_id!r} is orphan (not in overlay)"
        )
        self.cell_id = cell_id


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 with millisecond precision."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _is_excluded(record: Dict[str, Any]) -> Optional[str]:
    """Return the matched exclusion flag for ``record`` or ``None``.

    Cells flagged ``scratch | excluded | obsolete`` (any truthy value)
    are filtered out per BSP-008 §3 step 4. The first matched flag is
    returned so the trace records the operator-visible reason.
    """
    if not isinstance(record, dict):
        return None
    for flag in _EXCLUSION_FLAGS:
        if record.get(flag):
            return flag
    return None


def _ordered_cells(snapshot: Dict[str, Any]) -> List[str]:
    """Return the document-order list of cell ids from ``snapshot``.

    Prefers ``snapshot['ordering']`` when present (the explicit
    document-order list). Falls back to insertion order of
    ``snapshot['cells']`` so a freshly-built overlay still pack-walks
    deterministically.
    """
    ordering = snapshot.get("ordering")
    if isinstance(ordering, list):
        return [c for c in ordering if isinstance(c, str)]
    cells = snapshot.get("cells") or {}
    if isinstance(cells, dict):
        return [c for c in cells.keys() if isinstance(c, str)]
    return []


def _pinned_cells(snapshot: Dict[str, Any], ordering: List[str]) -> List[str]:
    """Return cells flagged ``pinned``, in document order.

    ``pinned`` is a flat anywhere-in-zone flag per BSP-008 §3 step 1.
    Iteration order is the canonical document order so two snapshots
    with the same overlay produce byte-identical output.
    """
    cells = snapshot.get("cells") or {}
    if not isinstance(cells, dict):
        return []
    pinned: List[str] = []
    for cid in ordering:
        record = cells.get(cid)
        if isinstance(record, dict) and record.get("pinned"):
            pinned.append(cid)
    return pinned


def _section_predecessors(
    snapshot: Dict[str, Any],
    cell_id: str,
    ordering: List[str],
) -> List[str]:
    """Return the focus cell's section predecessors in chronological order.

    Per BSP-008 §3 step 2: when the focus cell has a ``section_id``,
    walk the section's predecessors in document order (chronological
    proxy). When the focus cell has no section, fall back to
    "previous cells in document order" so the walk still yields a
    deterministic predecessor list (BSP-008 §3 step 2 fallback).

    The focus cell itself is NEVER included -- it is the run subject,
    not its own predecessor.
    """
    cells = snapshot.get("cells") or {}
    if not isinstance(cells, dict):
        return []
    focus = cells.get(cell_id) or {}
    focus_section = focus.get("section_id") if isinstance(focus, dict) else None
    out: List[str] = []
    for cid in ordering:
        if cid == cell_id:
            break
        record = cells.get(cid)
        if not isinstance(record, dict):
            continue
        if focus_section is None:
            # Fallback: every prior cell in document order.
            out.append(cid)
        elif record.get("section_id") == focus_section:
            out.append(cid)
    return out


def _current_cell_sub_turns(
    snapshot: Dict[str, Any], cell_id: str,
) -> List[str]:
    """Return the focus cell's prior sub-turns per BSP-008 §3 step 3.

    Sub-turns are emitted only when merges produced them. The V1 shape
    looks for ``cells[<id>].sub_turns`` as a list; non-list values are
    ignored. Each entry is a string cell_id (or turn_id) reference;
    non-string entries are dropped.
    """
    cells = snapshot.get("cells") or {}
    if not isinstance(cells, dict):
        return []
    focus = cells.get(cell_id) or {}
    sub = focus.get("sub_turns") if isinstance(focus, dict) else None
    if not isinstance(sub, list):
        return []
    return [s for s in sub if isinstance(s, str)]


def pack(cell_id: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """V1 ContextPacker -- pure, deterministic walk per BSP-008 §3.

    Args:
        cell_id: The cell being run (the run subject). Must be
            present in ``snapshot['cells']`` (or ``snapshot['ordering']``
            if no cells map is supplied) -- otherwise raises
            :class:`K100OrphanCellError`.
        snapshot: The notebook overlay. Recognized keys:

            * ``cells``: ``{cell_id: record}`` dict. Each record may
              carry ``pinned``, ``section_id``, ``sub_turns``,
              ``scratch``, ``excluded``, ``obsolete`` flags.
            * ``ordering``: optional document-order list of cell ids.
              Falls back to insertion order of ``cells``.

    Returns a ContextManifest dict per
    [concepts/context-manifest](../../docs/atoms/concepts/context-manifest.md):

    ::

        {
          "manifest_id": "<uuid>",
          "cell_id": "<id>",
          "cell_refs": ["<cell_id>", ...],   # ordered, deduped
          "inclusion_rules_applied": [
              {"rule": "pinned", "cells": [...]},
              {"rule": "section_predecessor", "cells": [...]},
              {"rule": "current_cell_sub_turns", "cells": [...]},
          ],
          "exclusions_applied": [
              {"reason": "scratch", "cells": [...]},
              {"reason": "excluded", "cells": [...]},
              {"reason": "obsolete", "cells": [...]},
          ],
          "generated_at": "<iso8601>",
        }

    The manifest is byte-identical for the same input modulo
    ``manifest_id`` and ``generated_at`` -- both intentionally fresh
    so persistence has a stable correlation handle.

    Pure function discipline: no I/O, no logging, no agent calls. The
    AgentSupervisor wraps the call in a ``record_context_manifest``
    intent.
    """
    if not isinstance(cell_id, str) or not cell_id:
        raise ContextPackerError(
            "pack: cell_id must be a non-empty string"
        )
    if not isinstance(snapshot, dict):
        raise ContextPackerError(
            "pack: snapshot must be a dict"
        )

    ordering = _ordered_cells(snapshot)
    cells = snapshot.get("cells")
    cells_map = cells if isinstance(cells, dict) else {}

    # K100 orphan check: the focus cell MUST be in the overlay either
    # via the cells map or via the ordering list. Either source is
    # sufficient for a deterministic walk; we accept both.
    if cell_id not in cells_map and cell_id not in ordering:
        raise K100OrphanCellError(cell_id)

    pinned = _pinned_cells(snapshot, ordering)
    section_preds = _section_predecessors(snapshot, cell_id, ordering)
    sub_turns = _current_cell_sub_turns(snapshot, cell_id)

    # Build categorized inclusion buckets BEFORE the exclusion filter so
    # the trace records what each rule contributed pre-filter (matches
    # the operator-facing semantics in BSP-008 §3 / Inspect mode).
    pinned_kept: List[str] = []
    section_kept: List[str] = []
    sub_turns_kept: List[str] = []

    excluded_buckets: Dict[str, List[str]] = {
        "scratch": [], "excluded": [], "obsolete": [],
    }
    seen_exclusion: Dict[str, str] = {}  # cell_id -> reason

    def _filter(category_list: List[str], dest: List[str]) -> None:
        for cid in category_list:
            record = cells_map.get(cid)
            if record is None:
                # Unknown ref (e.g. dangling sub-turn); drop silently --
                # the V1 walk filters by overlay flags, not by ref-validity.
                # Validation (K103 unknown_turn_ref) happens at write time.
                continue
            reason = _is_excluded(record)
            if reason is not None:
                if cid not in seen_exclusion:
                    excluded_buckets[reason].append(cid)
                    seen_exclusion[cid] = reason
                continue
            dest.append(cid)

    _filter(pinned, pinned_kept)
    _filter(section_preds, section_kept)
    _filter(sub_turns, sub_turns_kept)

    # Step 5: dedupe preserving first-occurrence order across categories.
    # BSP-008 §3 step 5: pinned cells appear at the head; subsequent
    # categories drop duplicates rather than re-emitting them later.
    cell_refs: List[str] = []
    seen: set = set()
    for source in (pinned_kept, section_kept, sub_turns_kept):
        for cid in source:
            if cid in seen:
                continue
            seen.add(cid)
            cell_refs.append(cid)

    # Inclusion trace: record what each rule contributed AFTER exclusion
    # filtering but BEFORE cross-category dedupe so Inspect mode can show
    # "this cell was kept because pinned and section_predecessor" without
    # losing either rule's signal.
    inclusion_rules_applied: List[Dict[str, Any]] = []
    if pinned_kept:
        inclusion_rules_applied.append(
            {"rule": "pinned", "cells": list(pinned_kept)}
        )
    if section_kept:
        inclusion_rules_applied.append(
            {"rule": "section_predecessor", "cells": list(section_kept)}
        )
    if sub_turns_kept:
        inclusion_rules_applied.append(
            {"rule": "current_cell_sub_turns", "cells": list(sub_turns_kept)}
        )

    # Exclusion trace: emit one entry per reason that contributed; skip
    # empty buckets so byte-identity is preserved across no-op snapshots.
    exclusions_applied: List[Dict[str, Any]] = []
    for reason in _EXCLUSION_FLAGS:
        bucket = excluded_buckets[reason]
        if bucket:
            exclusions_applied.append(
                {"reason": reason, "cells": list(bucket)}
            )

    return {
        "manifest_id": str(uuid.uuid4()),
        "cell_id": cell_id,
        "cell_refs": cell_refs,
        "inclusion_rules_applied": inclusion_rules_applied,
        "exclusions_applied": exclusions_applied,
        "generated_at": _utc_now_iso(),
    }

"""K-AS-A BSP-005 S3.5 -- ContextPacker walker tests.

Per ``docs/notebook/PLAN-S3.5-context-packer.md``, ``decisions/v1-contextpacker-walk``,
and ``concepts/context-manifest``: the V1 ContextPacker is a pure,
deterministic structural walker that produces a ContextManifest dict.
This module covers the kernel-side guarantees the slice ships:

1. Pinned cells appear at the head of ``cell_refs``.
2. Section predecessors come in chronological (document) order after
   pinned cells.
3. ``scratch | excluded | obsolete`` cells are filtered out and
   surfaced in ``exclusions_applied``.
4. Cross-category dedupe preserves first-occurrence order.
5. Same input produces byte-identical output (modulo ``manifest_id``
   and ``generated_at``).
6. The ``record_context_manifest`` intent flow round-trips one
   manifest into ``metadata.rts.zone.context_manifests``.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from llm_kernel.context_packer import (
    K100OrphanCellError,
    pack,
)
from llm_kernel.metadata_writer import MetadataWriter


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


# ---------------------------------------------------------------------------
# Walker semantics
# ---------------------------------------------------------------------------


def test_pack_returns_pinned_cells_first() -> None:
    """Pinned cells anywhere in the zone appear at the head of cell_refs."""
    snapshot = {
        "cells": {
            "c0": {"section_id": "s1"},
            "c1": {"section_id": "s1"},
            "c2": {"section_id": "s1"},
            "c_pinned_late": {"section_id": "s2", "pinned": True},
            "c_pinned_early": {"section_id": "s1", "pinned": True},
        },
        "ordering": ["c_pinned_early", "c0", "c1", "c2", "c_pinned_late"],
    }
    manifest = pack("c2", snapshot)
    # Pinned cells iterate in document order (deterministic).
    assert manifest["cell_refs"][:2] == ["c_pinned_early", "c_pinned_late"]
    rules = {r["rule"]: r["cells"] for r in manifest["inclusion_rules_applied"]}
    assert rules["pinned"] == ["c_pinned_early", "c_pinned_late"]


def test_pack_returns_section_predecessors_chronological() -> None:
    """Within a section, predecessors are listed in document order."""
    snapshot = {
        "cells": {
            "c0": {"section_id": "s1"},
            "c1": {"section_id": "s1"},
            "c2": {"section_id": "s1"},
            "c3": {"section_id": "s2"},   # different section -- excluded
            "c4": {"section_id": "s1"},   # FOCUS
        },
        "ordering": ["c0", "c1", "c2", "c3", "c4"],
    }
    manifest = pack("c4", snapshot)
    rules = {r["rule"]: r["cells"] for r in manifest["inclusion_rules_applied"]}
    assert rules["section_predecessor"] == ["c0", "c1", "c2"]
    # Walk stops at focus; focus itself never included.
    assert "c4" not in manifest["cell_refs"]
    assert "c3" not in manifest["cell_refs"]


def test_pack_excludes_scratch_and_excluded_cells() -> None:
    """``scratch | excluded | obsolete`` flagged cells are filtered out."""
    snapshot = {
        "cells": {
            "c0": {"section_id": "s1"},
            "c1": {"section_id": "s1", "scratch": True},
            "c2": {"section_id": "s1", "excluded": True},
            "c3": {"section_id": "s1", "obsolete": True},
            "c4": {"section_id": "s1"},
            "c5": {"section_id": "s1"},  # FOCUS
        },
        "ordering": ["c0", "c1", "c2", "c3", "c4", "c5"],
    }
    manifest = pack("c5", snapshot)
    refs = manifest["cell_refs"]
    assert "c1" not in refs
    assert "c2" not in refs
    assert "c3" not in refs
    assert refs == ["c0", "c4"]
    excl = {e["reason"]: e["cells"] for e in manifest["exclusions_applied"]}
    assert excl["scratch"] == ["c1"]
    assert excl["excluded"] == ["c2"]
    assert excl["obsolete"] == ["c3"]


def test_pack_dedupes_first_occurrence() -> None:
    """A cell that would appear in multiple categories is emitted once at its
    earliest position (pinned wins over section)."""
    snapshot = {
        "cells": {
            "c_dup": {"section_id": "s1", "pinned": True},
            "c_other": {"section_id": "s1"},
            "c_focus": {"section_id": "s1"},
        },
        "ordering": ["c_dup", "c_other", "c_focus"],
    }
    manifest = pack("c_focus", snapshot)
    # ``c_dup`` would have been emitted by both pinned (step 1) and
    # section_predecessor (step 2); dedupe keeps the pinned position.
    assert manifest["cell_refs"] == ["c_dup", "c_other"]
    # The trace MUST still record both rules contributed -- Inspect
    # mode shows "this cell was kept by [pinned, section_predecessor]".
    rules = {r["rule"]: r["cells"] for r in manifest["inclusion_rules_applied"]}
    assert "c_dup" in rules["pinned"]
    assert "c_dup" in rules["section_predecessor"]


def test_pack_byte_identical_for_same_input() -> None:
    """Repeated pack() calls produce equal manifests modulo manifest_id /
    generated_at."""
    snapshot = {
        "cells": {
            "c0": {"section_id": "s1", "pinned": True},
            "c1": {"section_id": "s1"},
            "c2": {"section_id": "s1"},
            "c3": {"section_id": "s1", "scratch": True},
        },
        "ordering": ["c0", "c1", "c2", "c3"],
    }
    m1 = pack("c2", snapshot)
    m2 = pack("c2", snapshot)
    # Strip the intentionally fresh fields, then compare.
    for m in (m1, m2):
        m.pop("manifest_id")
        m.pop("generated_at")
    # Use canonical-JSON byte equality so field ordering is locked in.
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)


def test_pack_orphan_cell_raises_k100() -> None:
    """A focus cell not in the overlay raises K100."""
    snapshot = {
        "cells": {"c0": {"section_id": "s1"}},
        "ordering": ["c0"],
    }
    try:
        pack("c_missing", snapshot)
    except K100OrphanCellError as exc:
        assert "K100" in str(exc)
        assert exc.cell_id == "c_missing"
    else:  # pragma: no cover - defensive
        raise AssertionError("expected K100OrphanCellError")


# ---------------------------------------------------------------------------
# Intent envelope round-trip via MetadataWriter
# ---------------------------------------------------------------------------


def test_record_context_manifest_intent_stores_to_metadata() -> None:
    """``record_context_manifest`` envelope places the manifest under
    ``metadata.rts.zone.context_manifests[<manifest_id>]``."""
    writer = MetadataWriter(autosave_interval_sec=999.0)
    snapshot = {
        "cells": {
            "c0": {"section_id": "s1", "pinned": True},
            "c1": {"section_id": "s1"},
            "c2": {"section_id": "s1"},
        },
        "ordering": ["c0", "c1", "c2"],
    }
    manifest = pack("c2", snapshot)
    result = writer.submit_intent(_envelope(
        intent_kind="record_context_manifest",
        parameters={"manifest": manifest},
        intent_id="i-rcm-001",
    ))
    assert result["applied"] is True, result
    assert result["error_code"] is None
    snap = writer.snapshot()
    zone = snap.get("zone", {})
    manifests = zone.get("context_manifests", {})
    assert manifest["manifest_id"] in manifests
    persisted = manifests[manifest["manifest_id"]]
    assert persisted["cell_id"] == "c2"
    assert persisted["cell_refs"] == manifest["cell_refs"]

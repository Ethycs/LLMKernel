"""K-MW BSP-005 §6.1 / S0.5 -- cell-kinds typed enum.

Per ``docs/notebook/PLAN-S0.5-cell-kinds.md`` and the
[cell-kinds atom](docs/atoms/concepts/cell-kinds.md), every cell in a
V1 ``.llmnb`` carries a ``kind`` field on
``metadata.rts.cells[<cell_id>].kind`` whose value is one of the eight
enum entries (four active, four reserved).  This module covers the
writer-level guarantees the slice ships:

1. Round-trip ``set_cell_metadata`` with ``kind="agent"`` so the cell
   record is created and persisted.
2. ``markdown`` cells MUST NOT carry ``bound_agent_id``; the writer
   strips it (or rejects when the caller passed a non-null value).
3. Unknown kinds raise ``K42`` with structured ``unknown_cell_kind``.
4. Reserved kinds (``tool | artifact | control | native``) round-trip
   identically to active kinds; the writer dispatches NOTHING for
   them (renderer behaviour is X-EXT).
5. Pre-S0.5 hydrate path: cells with no ``kind`` field default to
   ``agent`` at load and the writer back-fills on the next snapshot.
"""

from __future__ import annotations

from typing import Any, Dict

from llm_kernel.metadata_writer import (
    CELL_KINDS_RESERVED,
    DEFAULT_CELL_KIND,
    MetadataWriter,
)


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


def _new_writer() -> MetadataWriter:
    return MetadataWriter(autosave_interval_sec=999.0)


def test_set_cell_metadata_with_kind_agent_round_trips() -> None:
    """``kind="agent"`` lands on the cell record; snapshot carries it."""
    writer = _new_writer()
    result = writer.submit_intent(_envelope(
        intent_kind="set_cell_metadata",
        parameters={
            "cell_id": "vscode-notebook-cell:/foo.llmnb#abc",
            "kind": "agent",
            "bound_agent_id": "alpha",
        },
        intent_id="i-cell-001",
    ))
    assert result["applied"] is True, result
    assert result["error_code"] is None
    snap = writer.snapshot()
    cells = snap.get("cells", {})
    assert "vscode-notebook-cell:/foo.llmnb#abc" in cells
    record = cells["vscode-notebook-cell:/foo.llmnb#abc"]
    assert record["kind"] == "agent"
    assert record["bound_agent_id"] == "alpha"


def test_set_cell_metadata_with_kind_markdown_strips_bound_agent_id() -> None:
    """``kind="markdown"`` MUST have ``bound_agent_id`` stripped to None.

    Per [cell-kinds atom](docs/atoms/concepts/cell-kinds.md): markdown
    cells MUST NOT carry a non-null ``bound_agent_id``.  Caller passing
    a non-null value with ``kind="markdown"`` triggers K42 with reason
    ``markdown_must_have_no_agent``; caller passing nothing or
    ``None`` succeeds with the normalized ``None``.
    """
    writer = _new_writer()
    # Happy path: explicit None.
    result = writer.submit_intent(_envelope(
        intent_kind="set_cell_metadata",
        parameters={
            "cell_id": "vscode-notebook-cell:/foo.llmnb#md1",
            "kind": "markdown",
            "bound_agent_id": None,
        },
        intent_id="i-md-001",
    ))
    assert result["applied"] is True, result
    record = writer.snapshot()["cells"][
        "vscode-notebook-cell:/foo.llmnb#md1"
    ]
    assert record["kind"] == "markdown"
    assert record["bound_agent_id"] is None

    # Failure path: markdown + non-null bound_agent_id -> K42.
    bad = writer.submit_intent(_envelope(
        intent_kind="set_cell_metadata",
        parameters={
            "cell_id": "vscode-notebook-cell:/foo.llmnb#md2",
            "kind": "markdown",
            "bound_agent_id": "alpha",
        },
        intent_id="i-md-002",
    ))
    assert bad["applied"] is False
    assert bad["error_code"] == "K42"
    assert "markdown_must_have_no_agent" in (bad["error_reason"] or "")


def test_set_cell_metadata_unknown_kind_returns_k42() -> None:
    """Unknown ``kind`` value -> K42 ``unknown_cell_kind``, no state change."""
    writer = _new_writer()
    snap_before = writer.snapshot()
    version_before = snap_before["snapshot_version"]
    result = writer.submit_intent(_envelope(
        intent_kind="set_cell_metadata",
        parameters={
            "cell_id": "vscode-notebook-cell:/foo.llmnb#x",
            "kind": "wat-this-aint-a-kind",
            "bound_agent_id": None,
        },
        intent_id="i-bad-001",
    ))
    assert result["applied"] is False
    assert result["error_code"] == "K42"
    assert "unknown_cell_kind" in (result["error_reason"] or "")
    snap_after = writer.snapshot()
    # State did not mutate (the cell map stays empty).
    assert "cells" not in snap_after or not snap_after.get("cells")
    # Version monotonic but only via the explicit snapshot() calls;
    # the rejected intent did NOT bump it.
    assert snap_after["snapshot_version"] == version_before + 1


def test_set_cell_metadata_reserved_kind_round_trips_but_does_not_dispatch() -> None:
    """All four reserved kinds round-trip; the writer dispatches NOTHING.

    The writer accepts ``tool | artifact | control | native``,
    persists them to ``metadata.rts.cells``, and produces no
    additional state mutations (the V1 renderer falls through to a
    kind-label only view, per the [cell-kinds atom invariants](docs/atoms/concepts/cell-kinds.md)).
    """
    writer = _new_writer()
    for idx, kind in enumerate(CELL_KINDS_RESERVED):
        cell_id = f"vscode-notebook-cell:/foo.llmnb#{kind}-{idx}"
        result = writer.submit_intent(_envelope(
            intent_kind="set_cell_metadata",
            parameters={
                "cell_id": cell_id,
                "kind": kind,
                "bound_agent_id": None,
            },
            intent_id=f"i-reserved-{idx:03d}",
        ))
        assert result["applied"] is True, (kind, result)
        record = writer.snapshot()["cells"][cell_id]
        assert record["kind"] == kind, kind
    # No agent dispatch / no extra side-effects on the agent graph.
    snap = writer.snapshot()
    assert snap["agents"]["nodes"] == []


def test_pre_s05_cell_no_kind_defaults_to_agent_on_load() -> None:
    """A pre-S0.5 snapshot with no ``kind`` field hydrates as ``agent``.

    The legacy snapshot has ``cells`` populated but the records are
    missing ``kind``.  After hydrate, the writer's in-memory record
    carries ``kind = "agent"`` AND a ``_kind_back_filled`` marker, and
    the writer is dirty so the next snapshot writes the kind back
    persistently (PLAN-S0.5 §3 step 3).
    """
    writer = _new_writer()
    legacy_snapshot = {
        "schema_version": "1.0.0",
        "schema_uri": "https://llmnb.dev/llmnb/v1/schema.json",
        "session_id": "s-legacy",
        "created_at": "2025-01-01T00:00:00.000Z",
        "snapshot_version": 7,
        "layout": {
            "version": 1,
            "tree": {
                "id": "root", "type": "workspace",
                "render_hints": {}, "children": [],
            },
        },
        "agents": {"version": 1, "nodes": [], "edges": []},
        "config": {
            "version": 1,
            "recoverable": {"kernel": {}, "agents": [], "mcp_servers": []},
            "volatile": {"kernel": {}, "agents": [], "mcp_servers": []},
        },
        "event_log": {"version": 1, "runs": []},
        "blobs": {},
        "drift_log": [],
        # Pre-S0.5: a cell record with no ``kind`` field.
        "cells": {
            "vscode-notebook-cell:/foo.llmnb#legacy-1": {
                "bound_agent_id": "alpha",
                "section_id": None,
                "capabilities": [],
            },
        },
    }
    writer.hydrate(legacy_snapshot)
    # Internal state: kind back-filled to default.
    record = writer._cells[  # noqa: SLF001 - test introspection
        "vscode-notebook-cell:/foo.llmnb#legacy-1"
    ]
    assert record["kind"] == DEFAULT_CELL_KIND == "agent"
    assert record.get("_kind_back_filled") is True
    # Next snapshot writes the kind out persistently and clears the
    # marker so the persisted form is canonical.
    snap = writer.snapshot()
    cells = snap["cells"]
    persisted = cells["vscode-notebook-cell:/foo.llmnb#legacy-1"]
    assert persisted["kind"] == "agent"
    assert "_kind_back_filled" not in persisted



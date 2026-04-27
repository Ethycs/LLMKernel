"""K-MW :meth:`MetadataWriter.hydrate` (RFC-005 / RFC-006 §8 mode:hydrate).

Verifies idempotency, schema_version validation, forbidden-secret
rejection, and that hydrating from a known snapshot lands the writer
in a state that re-emits an equivalent snapshot.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from llm_kernel.metadata_writer import (
    SCHEMA_VERSION,
    MetadataWriter,
)


def _baseline_snapshot() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_uri": "https://llmnb.dev/llmnb/v1/schema.json",
        "session_id": "9c1a3b2d-4e5f-4061-a072-8d9e3f4a5b6c",
        "created_at": "2026-04-26T12:14:08.221Z",
        "snapshot_version": 17,
        "layout": {
            "version": 1,
            "tree": {
                "id": "root", "type": "workspace",
                "render_hints": {"label": "monorepo"},
                "children": [
                    {"id": "zone-a", "type": "zone",
                     "render_hints": {}, "children": []},
                ],
            },
        },
        "agents": {
            "version": 1,
            "nodes": [
                {"id": "agent:alpha", "type": "agent",
                 "properties": {"status": "idle"}},
            ],
            "edges": [],
        },
        "config": {
            "version": 1,
            "recoverable": {
                "kernel": {"blob_threshold_bytes": 65536},
                "agents": [
                    {"agent_id": "alpha", "zone_id": "refactor",
                     "tools_allowed": ["notify"]},
                ],
                "mcp_servers": [],
            },
            "volatile": {
                "kernel": {
                    "model_default": "claude-sonnet-4-5",
                    "rfc_001_version": "1.0.0",
                },
                "agents": [],
                "mcp_servers": [],
            },
        },
        "event_log": {
            "version": 1,
            "runs": [
                {
                    "traceId": "5d27f5dd26ce4d619dbb9fbf36d2fe2b",
                    "spanId": "8a3c1a2e9d774f0a",
                    "parentSpanId": None,
                    "name": "notify", "kind": "SPAN_KIND_INTERNAL",
                    "startTimeUnixNano": "1745588938412000000",
                    "endTimeUnixNano": "1745588938611000000",
                    "status": {"code": "STATUS_CODE_OK", "message": ""},
                    "attributes": [
                        {"key": "llmnb.run_type",
                         "value": {"stringValue": "tool"}},
                    ],
                    "events": [], "links": [],
                },
            ],
        },
        "blobs": {},
        "drift_log": [
            {
                "detected_at": "2026-04-26T13:22:45.000Z",
                "field_path": "config.volatile.kernel.model_default",
                "previous_value": "claude-sonnet-4-5",
                "current_value": "claude-sonnet-4-6",
                "severity": "warn",
                "operator_acknowledged": False,
            },
        ],
    }


def test_hydrate_round_trip_yields_equivalent_state() -> None:
    """A hydrated writer's next snapshot mirrors the persisted shape."""
    snap = _baseline_snapshot()
    writer = MetadataWriter(autosave_interval_sec=999.0)
    writer.hydrate(snap)

    out = writer.snapshot()
    # Snapshot version increments on emission per the writer's
    # contract: the persisted version was 17 and the next emission is
    # 18 (RFC-006 §"hydrate request/response semantics").
    assert out["snapshot_version"] == snap["snapshot_version"] + 1
    assert out["session_id"] == snap["session_id"]
    assert out["created_at"] == snap["created_at"]
    assert out["layout"] == snap["layout"]
    assert out["agents"] == snap["agents"]
    # event_log runs are preserved (deduped by spanId).
    assert len(out["event_log"]["runs"]) == 1
    assert out["event_log"]["runs"][0]["spanId"] == "8a3c1a2e9d774f0a"
    assert out["drift_log"] == snap["drift_log"]


def test_hydrate_is_idempotent() -> None:
    """Hydrating with the same snapshot twice leaves observable state equal."""
    snap = _baseline_snapshot()
    a = MetadataWriter(autosave_interval_sec=999.0)
    a.hydrate(snap)
    a.hydrate(snap)

    b = MetadataWriter(autosave_interval_sec=999.0)
    b.hydrate(snap)

    # Compare by extracting state-relevant private fields.
    assert a._snapshot_version == b._snapshot_version  # noqa: SLF001
    assert a._layout == b._layout  # noqa: SLF001
    assert a._agent_graph == b._agent_graph  # noqa: SLF001
    assert a._drift_log == b._drift_log  # noqa: SLF001
    assert a._extra_runs == b._extra_runs  # noqa: SLF001
    assert a._blobs == b._blobs  # noqa: SLF001
    assert a._config == b._config  # noqa: SLF001


def test_hydrate_rejects_schema_version_major_mismatch() -> None:
    """Major-version mismatch raises :class:`ValueError`."""
    snap = _baseline_snapshot()
    snap["schema_version"] = "2.0.0"
    writer = MetadataWriter(autosave_interval_sec=999.0)
    with pytest.raises(ValueError):
        writer.hydrate(snap)


def test_hydrate_rejects_forbidden_secret_in_config() -> None:
    """A forbidden field anywhere in config raises ``ValueError``."""
    snap = _baseline_snapshot()
    snap["config"]["volatile"]["kernel"]["api_key"] = "sk-DO_NOT_LOG"
    writer = MetadataWriter(autosave_interval_sec=999.0)
    with pytest.raises(ValueError) as ei:
        writer.hydrate(snap)
    assert "forbidden secret in config" in str(ei.value)
    # The offending value MUST NOT appear in the exception text.
    assert "DO_NOT_LOG" not in str(ei.value)


def test_hydrate_replaces_state_not_merges() -> None:
    """Hydrate replaces -- pre-existing layout edits are wiped."""
    snap = _baseline_snapshot()
    writer = MetadataWriter(autosave_interval_sec=999.0)
    # Seed pre-hydrate state that the hydrate MUST clear.
    writer.apply_layout_edit(
        operation="add_zone",
        parameters={"node_spec": {"id": "pre-existing", "type": "zone"}},
    )
    writer.append_drift_event(
        field_path="x.y", previous_value=1, current_value=2, severity="info",
    )
    writer.hydrate(snap)
    out = writer.snapshot()
    ids = {c["id"] for c in out["layout"]["tree"]["children"]}
    assert "pre-existing" not in ids
    assert "zone-a" in ids
    # drift_log was reset to the hydrated shape (1 entry, not 2).
    assert len(out["drift_log"]) == 1


def test_hydrate_serializes_to_baseline_via_serialize_snapshot() -> None:
    """The post-hydrate snapshot is JSON-equivalent to the baseline.

    The next :meth:`snapshot` increments ``snapshot_version`` so the
    direct equality fails by one field; we drop that field and the
    round-trip is exact for the recoverable shape.
    """
    snap = _baseline_snapshot()
    writer = MetadataWriter(autosave_interval_sec=999.0)
    writer.hydrate(snap)
    out = writer.snapshot()
    # Strip the bumped version for the comparison.
    out_copy = dict(out)
    base_copy = dict(snap)
    out_copy["snapshot_version"] = 0
    base_copy["snapshot_version"] = 0
    assert json.dumps(out_copy, sort_keys=True) == json.dumps(
        base_copy, sort_keys=True,
    )

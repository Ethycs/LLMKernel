"""RFC-005 §"Failure modes" load-time integrity checks and blob GC.

``check_rts_integrity`` rejects F5 / F9 / F10 / F11 with
:class:`LlmnbIntegrityError` and normalizes F4 / F12 in place;
``MetadataWriter.hydrate`` applies it before any state moves.
``gc_blobs`` is RFC-005 §"Blob garbage collection".
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

import pytest

from llm_kernel.capture.metadata_writer import SCHEMA_VERSION, MetadataWriter
from llm_kernel.notebook.notebook_format import (
    LlmnbIntegrityError,
    check_rts_integrity,
    gc_blobs,
)

BODY = "x" * 100
KEY = "sha256:" + hashlib.sha256(BODY.encode("utf-8")).hexdigest()


def _rts() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "layout": {"version": 1, "tree": {
            "id": "root", "type": "workspace", "render_hints": {},
            "children": [{"id": "zone-a", "type": "zone", "render_hints": {}, "children": []}],
        }},
        "agents": {"version": 1, "nodes": [{"id": "agent:alpha", "type": "agent"}], "edges": []},
        "event_log": {"version": 1, "runs": [{
            "traceId": "5d27f5dd26ce4d619dbb9fbf36d2fe2b", "spanId": "8a3c1a2e9d774f0a",
            "name": "notify", "startTimeUnixNano": "1", "endTimeUnixNano": "2",
            "status": {"code": "STATUS_CODE_OK", "message": ""},
            "attributes": [{"key": "output.value", "value": {"stringValue": f"$blob:{KEY}"}}],
            "events": [], "links": [],
        }]},
        "blobs": {KEY: {"id": KEY, "content_hash": KEY, "body": BODY}},
    }


def test_intact_snapshot_has_no_warnings() -> None:
    assert check_rts_integrity(_rts()) == []


def test_blob_hash_mismatch_is_f5() -> None:
    rts = _rts()
    rts["blobs"][KEY]["body"] = "tampered"
    with pytest.raises(LlmnbIntegrityError) as excinfo:
        check_rts_integrity(rts)
    assert excinfo.value.code == "F5"


def test_duplicate_layout_id_is_f9() -> None:
    rts = _rts()
    rts["layout"]["tree"]["children"][0]["children"].append(
        {"id": "zone-a", "type": "zone", "children": []})
    with pytest.raises(LlmnbIntegrityError) as excinfo:
        check_rts_integrity(rts)
    assert excinfo.value.code == "F9"


def test_dangling_edge_is_f10_with_rfc_message() -> None:
    rts = _rts()
    rts["agents"]["edges"].append({"source": "agent:alpha", "target": "zone-x", "kind": "in_zone"})
    with pytest.raises(LlmnbIntegrityError) as excinfo:
        check_rts_integrity(rts)
    assert excinfo.value.code == "F10"
    assert "graph integrity violation: edge agent:alpha→zone-x kind=in_zone" in str(excinfo.value)


def test_flat_attributes_are_f11() -> None:
    rts = _rts()
    rts["event_log"]["runs"][0]["attributes"] = {"llmnb.run_type": "tool"}
    with pytest.raises(LlmnbIntegrityError, match="non-conformant OTLP encoding") as excinfo:
        check_rts_integrity(rts)
    assert excinfo.value.code == "F11"


def test_missing_blob_and_numeric_timestamps_are_normalized() -> None:
    rts = _rts()
    del rts["blobs"][KEY]
    rts["event_log"]["runs"][0]["startTimeUnixNano"] = 1745588938412000000
    warnings = check_rts_integrity(rts)
    run = rts["event_log"]["runs"][0]
    assert run["attributes"][0]["value"]["stringValue"] == f"(blob missing: {KEY})"
    assert run["startTimeUnixNano"] == "1745588938412000000"
    assert [w.split(":", 1)[0] for w in warnings] == ["F12", "F4"]


def test_hydrate_rejects_before_mutating_state() -> None:
    writer = MetadataWriter()
    rts = _rts()
    rts["agents"]["edges"].append({"source": "agent:alpha", "target": "zone-x", "kind": "in_zone"})
    with pytest.raises(LlmnbIntegrityError):
        writer.hydrate(rts)
    assert writer.snapshot()["event_log"]["runs"] == []


def test_gc_sweeps_only_unreferenced_blobs() -> None:
    rts = _rts()
    rts["blobs"]["sha256:" + "0" * 64] = {"body": "orphan"}
    removed = gc_blobs({"metadata": {"rts": rts}})
    assert removed == ["sha256:" + "0" * 64]
    assert list(rts["blobs"]) == [KEY]

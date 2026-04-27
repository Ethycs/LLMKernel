"""Contract tests for :mod:`llm_kernel.metadata_writer` (RFC-005 / RFC-006).

Covers:

* Snapshot building from in-memory state (layout / agents / config /
  event_log / blobs / drift_log).
* Forbidden-secret rejection (the security pass per RFC-005 §F2).
* Blob extraction over the OTLP attribute layout.
* Line-oriented JSON serialization (one span per line; appending one
  span MUST NOT rewrite existing spans' line content).
* Bounded-queue overflow on the event log.
* The Family F ``notebook.metadata`` envelope shape per RFC-006 §8.
"""

from __future__ import annotations

import json
import secrets
from typing import Any, Dict, List

import pytest

from llm_kernel.metadata_writer import (
    DEFAULT_BLOB_THRESHOLD_BYTES,
    DEFAULT_EVENT_LOG_QUEUE_CAP,
    MetadataWriter,
    SCHEMA_VERSION,
    SecretRejected,
    extract_blobs,
    reject_secrets,
    serialize_snapshot,
)


# ---------------------------------------------------------------------------
# reject_secrets
# ---------------------------------------------------------------------------


def test_reject_secrets_passes_clean_config() -> None:
    """A config with no forbidden field names returns without raising."""
    reject_secrets({
        "kernel": {"blob_threshold_bytes": 65536},
        "agents": [{"agent_id": "alpha", "zone_id": "z1"}],
        "mcp_servers": [{"server_id": "s1", "tools": ["notify"]}],
    })


@pytest.mark.parametrize("key", [
    "api_key", "API_KEY", "user_token", "AUTHORIZATION",
    "bearer", "Cookie", "DB_PASSWORD", "session_secret",
])
def test_reject_secrets_raises_on_forbidden_keys(key: str) -> None:
    """RFC-005 §F2: forbidden field names raise :class:`SecretRejected`."""
    with pytest.raises(SecretRejected) as ei:
        reject_secrets({"kernel": {key: "DO_NOT_LOG_THIS"}})
    # Crucially: the offending VALUE MUST NOT appear in the exception.
    assert "DO_NOT_LOG_THIS" not in str(ei.value)


def test_reject_secrets_carves_out_public_key() -> None:
    """``*_public_key`` is allowed (carve-out per RFC-005 §F2)."""
    reject_secrets({
        "kernel": {"signing_public_key": "MIIBIjANBg..."},
    })


def test_reject_secrets_walks_nested_lists_and_dicts() -> None:
    """Forbidden keys nested inside lists are also rejected."""
    with pytest.raises(SecretRejected):
        reject_secrets({
            "agents": [
                {"agent_id": "alpha"},
                {"agent_id": "beta", "secrets": {"api_token": "x"}},
            ],
        })


# ---------------------------------------------------------------------------
# extract_blobs
# ---------------------------------------------------------------------------


def test_extract_blobs_replaces_large_strings_with_sentinel() -> None:
    """Strings exceeding the threshold get hashed into ``blobs`` with a sentinel."""
    big = "x" * (DEFAULT_BLOB_THRESHOLD_BYTES + 1)
    runs: List[Dict[str, Any]] = [{
        "spanId": secrets.token_hex(8), "traceId": secrets.token_hex(16),
        "name": "report_progress", "kind": "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": "0", "endTimeUnixNano": "1",
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "attributes": [
            {"key": "input.value", "value": {"stringValue": big}},
            {"key": "input.mime_type",
             "value": {"stringValue": "application/json"}},
        ],
        "events": [], "links": [],
    }]
    blobs: Dict[str, Dict[str, Any]] = {}
    extract_blobs(runs, blobs, DEFAULT_BLOB_THRESHOLD_BYTES)
    sv = runs[0]["attributes"][0]["value"]["stringValue"]
    assert sv.startswith("$blob:sha256:")
    blob_key = sv.removeprefix("$blob:")
    assert blob_key in blobs
    assert blobs[blob_key]["data"] == big
    assert blobs[blob_key]["size_bytes"] == len(big.encode("utf-8"))
    # Small values are untouched.
    assert runs[0]["attributes"][1]["value"]["stringValue"] == "application/json"


def test_extract_blobs_is_idempotent() -> None:
    """Running ``extract_blobs`` twice does not re-hash an already-extracted value."""
    big = "y" * (DEFAULT_BLOB_THRESHOLD_BYTES + 1)
    runs: List[Dict[str, Any]] = [{
        "spanId": secrets.token_hex(8), "traceId": secrets.token_hex(16),
        "name": "x", "kind": "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": "0", "endTimeUnixNano": "1",
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "attributes": [
            {"key": "input.value", "value": {"stringValue": big}},
        ],
        "events": [], "links": [],
    }]
    blobs: Dict[str, Dict[str, Any]] = {}
    extract_blobs(runs, blobs, DEFAULT_BLOB_THRESHOLD_BYTES)
    sentinel = runs[0]["attributes"][0]["value"]["stringValue"]
    extract_blobs(runs, blobs, DEFAULT_BLOB_THRESHOLD_BYTES)
    assert runs[0]["attributes"][0]["value"]["stringValue"] == sentinel
    assert len(blobs) == 1


# ---------------------------------------------------------------------------
# serialize_snapshot (line-oriented)
# ---------------------------------------------------------------------------


def _minimal_run(name: str, span_id: str) -> Dict[str, Any]:
    return {
        "traceId": secrets.token_hex(16),
        "spanId": span_id, "parentSpanId": None,
        "name": name, "kind": "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": "0", "endTimeUnixNano": "1",
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "attributes": [
            {"key": "llmnb.run_type", "value": {"stringValue": "tool"}},
        ],
        "events": [], "links": [],
    }


def test_serialize_snapshot_one_span_per_line() -> None:
    """Each ``event_log.runs`` entry occupies its own line in the output."""
    snapshot = {
        "metadata": {"rts": {
            "schema_version": SCHEMA_VERSION,
            "session_id": "s1",
            "event_log": {
                "version": 1,
                "runs": [
                    _minimal_run("a", "1111111111111111"),
                    _minimal_run("b", "2222222222222222"),
                    _minimal_run("c", "3333333333333333"),
                ],
            },
        }},
    }
    text = serialize_snapshot(snapshot)
    # The runs array's interior MUST contain exactly three element lines.
    runs_section = text.split('"runs": [')[1].split("]\n")[0]
    inner_lines = [
        ln for ln in runs_section.splitlines() if ln.strip()
    ]
    assert len(inner_lines) == 3, runs_section
    # Each inner line is one full JSON object.
    for ln in inner_lines:
        stripped = ln.strip().rstrip(",")
        parsed = json.loads(stripped)
        assert parsed["kind"] == "SPAN_KIND_INTERNAL"


def test_serialize_snapshot_appending_a_run_only_changes_one_line() -> None:
    """RFC-005 §"Line-oriented serialization": adding one span MUST NOT
    rewrite existing spans' line content (git pack-delta efficiency).
    """
    base_runs = [
        _minimal_run("a", "1111111111111111"),
        _minimal_run("b", "2222222222222222"),
    ]
    snap_a = {"metadata": {"rts": {"event_log": {"version": 1, "runs": base_runs}}}}
    snap_b = {"metadata": {"rts": {"event_log": {"version": 1, "runs": base_runs + [
        _minimal_run("c", "3333333333333333"),
    ]}}}}
    text_a = serialize_snapshot(snap_a)
    text_b = serialize_snapshot(snap_b)
    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()
    # All lines that appear in A (modulo final-element trailing comma)
    # MUST appear identically in B's prefix.
    common = []
    for la in lines_a:
        # Strip the trailing comma if present (the previous-final
        # element gets one when a new element is appended).
        common.append(la.rstrip(","))
    for line in common[:-2]:  # ignore the closing ] and the surrounding context
        # A B-side line of equivalent stripping must exist.
        assert any(lb.rstrip(",").strip() == line.strip() for lb in lines_b), line


# ---------------------------------------------------------------------------
# MetadataWriter end-to-end
# ---------------------------------------------------------------------------


class _CaptureDispatcher:
    """Minimal Sink-like dispatcher capturing emitted envelopes."""

    def __init__(self) -> None:
        self.envelopes: List[Dict[str, Any]] = []

    def emit(self, envelope: Dict[str, Any]) -> None:
        from llm_kernel.run_envelope import validate_envelope
        validate_envelope(envelope)
        self.envelopes.append(envelope)


def test_writer_snapshot_increments_version_and_emits() -> None:
    """Each :meth:`snapshot` increments ``snapshot_version`` and emits Family F."""
    sink = _CaptureDispatcher()
    writer = MetadataWriter(dispatcher=sink, autosave_interval_sec=999.0)
    writer.update_layout({"id": "root", "type": "workspace",
                          "render_hints": {}, "children": []})
    snap1 = writer.snapshot(trigger="save")
    snap2 = writer.snapshot(trigger="save")
    assert snap1["snapshot_version"] == 1
    assert snap2["snapshot_version"] == 2
    assert len(sink.envelopes) == 2
    env = sink.envelopes[1]
    assert env["message_type"] == "notebook.metadata"
    payload = env["payload"]
    assert payload["mode"] == "snapshot"
    assert payload["trigger"] == "save"
    assert payload["snapshot"]["schema_version"] == SCHEMA_VERSION


def test_writer_rejects_secrets_in_config() -> None:
    """``update_config`` MUST refuse a forbidden field BEFORE committing."""
    writer = MetadataWriter()
    # Capture a baseline; subsequent rejected update MUST NOT overwrite it.
    writer.update_config(
        recoverable={"kernel": {}, "agents": [], "mcp_servers": []},
        volatile={"kernel": {}, "agents": [], "mcp_servers": []},
    )
    with pytest.raises(SecretRejected):
        writer.update_config(
            recoverable={"kernel": {"api_key": "leak"}},
            volatile={},
        )
    # The volatile config did not change.
    snap = writer.snapshot()
    assert "api_key" not in json.dumps(snap)


def test_writer_blob_extraction_in_snapshot() -> None:
    """A large attribute value persists as a sentinel + blob row."""
    sink = _CaptureDispatcher()
    writer = MetadataWriter(dispatcher=sink, autosave_interval_sec=999.0)
    big = "z" * (DEFAULT_BLOB_THRESHOLD_BYTES + 16)
    writer.record_run({
        "traceId": secrets.token_hex(16),
        "spanId": secrets.token_hex(8), "parentSpanId": None,
        "name": "report_progress", "kind": "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": "0", "endTimeUnixNano": "1",
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "attributes": [
            {"key": "llmnb.run_type", "value": {"stringValue": "tool"}},
            {"key": "input.value", "value": {"stringValue": big}},
        ],
        "events": [], "links": [],
    })
    snap = writer.snapshot()
    sv = snap["event_log"]["runs"][0]["attributes"][1]["value"]["stringValue"]
    assert sv.startswith("$blob:sha256:")
    blob_key = sv.removeprefix("$blob:")
    assert snap["blobs"][blob_key]["data"] == big


def test_writer_event_log_bounded_queue_overflow(
    caplog: pytest.LogCaptureFixture, tmp_path,
) -> None:
    """Event-log queue caps at 10 000; overflow logs a checkpoint warning."""
    cap = 8  # tiny cap to keep the test fast
    writer = MetadataWriter(
        autosave_interval_sec=999.0,
        event_log_queue_cap=cap,
        workspace_root=tmp_path,
    )
    import logging as _logging
    with caplog.at_level(_logging.WARNING, logger="llm_kernel.metadata_writer"):
        for i in range(cap + 5):
            writer.record_run({
                "traceId": secrets.token_hex(16),
                "spanId": secrets.token_hex(8), "parentSpanId": None,
                "name": f"run-{i}", "kind": "SPAN_KIND_INTERNAL",
                "startTimeUnixNano": "0", "endTimeUnixNano": "1",
                "status": {"code": "STATUS_CODE_OK", "message": ""},
                "attributes": [], "events": [], "links": [],
            })
    snap = writer.snapshot()
    # The event_log holds at most ``cap`` runs after overflow.
    assert len(snap["event_log"]["runs"]) <= cap
    assert any("queue overflow" in r.getMessage() for r in caplog.records)


def test_writer_envelope_correlation_id_is_session_and_version() -> None:
    """The envelope's correlation_id is ``<session_id>:<snapshot_version>``."""
    sink = _CaptureDispatcher()
    writer = MetadataWriter(
        dispatcher=sink, session_id="s-abc", autosave_interval_sec=999.0,
    )
    writer.snapshot()
    env = sink.envelopes[0]
    assert env["correlation_id"] == "s-abc:1"


def test_writer_drift_log_round_trips_in_snapshot() -> None:
    """Appended drift events appear in the next snapshot's drift_log."""
    writer = MetadataWriter(autosave_interval_sec=999.0)
    writer.append_drift_event(
        field_path="config.volatile.kernel.model_default",
        previous_value="claude-sonnet-4-5",
        current_value="claude-sonnet-4-6",
        severity="warn",
    )
    snap = writer.snapshot()
    log = snap["drift_log"]
    assert len(log) == 1
    assert log[0]["severity"] == "warn"
    assert log[0]["field_path"] == "config.volatile.kernel.model_default"
    assert log[0]["operator_acknowledged"] is False


def test_writer_session_id_and_created_at_persist_across_snapshots() -> None:
    """``session_id`` and ``created_at`` are stable across emissions."""
    writer = MetadataWriter(autosave_interval_sec=999.0)
    a = writer.snapshot()
    b = writer.snapshot()
    assert a["session_id"] == b["session_id"]
    assert a["created_at"] == b["created_at"]
    assert b["snapshot_version"] == a["snapshot_version"] + 1

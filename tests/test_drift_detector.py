"""Contract tests for :mod:`llm_kernel.drift_detector` (RFC-005 §drift_log).

Covers each volatile field RFC-005 enumerates plus in-progress span
truncation and agent-process status drift, with severity classification
per the spec.
"""

from __future__ import annotations

from typing import Any, Dict, List

from llm_kernel.drift_detector import (
    DriftDetector,
    SEVERITY_ERROR, SEVERITY_INFO, SEVERITY_WARN,
    truncate_in_progress_spans,
)


def _persisted(volatile_kernel: Dict[str, Any] = None,
               volatile_agents: List[Dict[str, Any]] = None,
               volatile_mcp: List[Dict[str, Any]] = None,
               agent_nodes: List[Dict[str, Any]] = None,
               runs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "config": {
            "volatile": {
                "kernel": volatile_kernel or {},
                "agents": volatile_agents or [],
                "mcp_servers": volatile_mcp or [],
            },
        },
        "agents": {"nodes": agent_nodes or [], "edges": []},
        "event_log": {"runs": runs or []},
    }


# ---------------------------------------------------------------------------
# Kernel volatile fields
# ---------------------------------------------------------------------------


def test_kernel_model_default_drift_is_warn() -> None:
    detector = DriftDetector()
    persisted = _persisted(volatile_kernel={"model_default": "claude-sonnet-4-5"})
    drift = detector.compare(
        persisted, current_kernel={"model_default": "claude-sonnet-4-6"},
    )
    assert len(drift) == 1
    assert drift[0]["field_path"] == "config.volatile.kernel.model_default"
    assert drift[0]["severity"] == SEVERITY_WARN
    assert drift[0]["previous_value"] == "claude-sonnet-4-5"
    assert drift[0]["current_value"] == "claude-sonnet-4-6"


def test_kernel_passthrough_mode_drift_is_warn() -> None:
    detector = DriftDetector()
    persisted = _persisted(volatile_kernel={"passthrough_mode": "litellm"})
    drift = detector.compare(
        persisted, current_kernel={"passthrough_mode": "anthropic_passthrough"},
    )
    assert len(drift) == 1
    assert drift[0]["severity"] == SEVERITY_WARN


def test_rfc_minor_version_drift_is_warn() -> None:
    detector = DriftDetector()
    persisted = _persisted(volatile_kernel={"rfc_001_version": "1.0.0"})
    drift = detector.compare(
        persisted, current_kernel={"rfc_001_version": "1.1.0"},
    )
    assert drift and drift[0]["severity"] == SEVERITY_WARN


def test_rfc_major_version_drift_is_error() -> None:
    """RFC-005 §"Resume-time RFC version check": major mismatch is error."""
    detector = DriftDetector()
    persisted = _persisted(volatile_kernel={"rfc_002_version": "1.0.1"})
    drift = detector.compare(
        persisted, current_kernel={"rfc_002_version": "2.0.0"},
    )
    assert drift and drift[0]["severity"] == SEVERITY_ERROR


def test_no_drift_when_volatile_matches_current() -> None:
    detector = DriftDetector()
    persisted = _persisted(volatile_kernel={
        "model_default": "claude-sonnet-4-5",
        "passthrough_mode": "litellm",
        "rfc_001_version": "1.0.0",
    })
    drift = detector.compare(persisted, current_kernel={
        "model_default": "claude-sonnet-4-5",
        "passthrough_mode": "litellm",
        "rfc_001_version": "1.0.0",
    })
    assert drift == []


# ---------------------------------------------------------------------------
# Per-agent volatile fields
# ---------------------------------------------------------------------------


def test_agent_model_drift_is_warn() -> None:
    detector = DriftDetector()
    persisted = _persisted(volatile_agents=[
        {"agent_id": "alpha", "model": "claude-sonnet-4-5",
         "system_prompt_template_id": "rfc-002-default",
         "system_prompt_hash": "sha256:abc"},
    ])
    drift = detector.compare(persisted, current_agents=[
        {"agent_id": "alpha", "model": "claude-sonnet-4-6",
         "system_prompt_template_id": "rfc-002-default",
         "system_prompt_hash": "sha256:abc"},
    ])
    assert any(d["field_path"].endswith(".model")
               and d["severity"] == SEVERITY_WARN for d in drift)


def test_agent_system_prompt_hash_drift_is_warn() -> None:
    detector = DriftDetector()
    persisted = _persisted(volatile_agents=[
        {"agent_id": "alpha", "system_prompt_hash": "sha256:c4f5"},
    ])
    drift = detector.compare(persisted, current_agents=[
        {"agent_id": "alpha", "system_prompt_hash": "sha256:91a2"},
    ])
    assert any(d["field_path"].endswith(".system_prompt_hash")
               and d["severity"] == SEVERITY_WARN for d in drift)


def test_agent_disappeared_is_warn() -> None:
    detector = DriftDetector()
    persisted = _persisted(volatile_agents=[
        {"agent_id": "alpha", "model": "claude-sonnet-4-5"},
    ])
    drift = detector.compare(persisted, current_agents=[])
    assert any(d["field_path"].endswith(".agent_id")
               and d["current_value"] is None for d in drift)


# ---------------------------------------------------------------------------
# MCP server drift
# ---------------------------------------------------------------------------


def test_mcp_transport_drift_is_warn() -> None:
    detector = DriftDetector()
    persisted = _persisted(volatile_mcp=[
        {"server_id": "operator-bridge", "transport": "stdio"},
    ])
    drift = detector.compare(persisted, current_mcp_servers=[
        {"server_id": "operator-bridge", "transport": "http"},
    ])
    assert any(d["field_path"].endswith(".transport")
               and d["severity"] == SEVERITY_WARN for d in drift)


def test_mcp_server_disappeared_is_error() -> None:
    """RFC-005: MCP server disappearance is resume-blocking (error)."""
    detector = DriftDetector()
    persisted = _persisted(volatile_mcp=[
        {"server_id": "operator-bridge", "transport": "stdio"},
    ])
    drift = detector.compare(persisted, current_mcp_servers=[])
    assert any(d["severity"] == SEVERITY_ERROR for d in drift)


# ---------------------------------------------------------------------------
# In-progress span truncation
# ---------------------------------------------------------------------------


def test_in_progress_span_truncation_emits_info_drift() -> None:
    """RFC-005 §"In-progress spans": truncate + emit info drift event."""
    detector = DriftDetector()
    runs = [
        {"spanId": "1111111111111111", "traceId": "0" * 32,
         "name": "x", "kind": "SPAN_KIND_INTERNAL",
         "startTimeUnixNano": "0", "endTimeUnixNano": None,
         "status": {"code": "STATUS_CODE_UNSET", "message": ""},
         "attributes": [], "events": [], "links": []},
        {"spanId": "2222222222222222", "traceId": "0" * 32,
         "name": "y", "kind": "SPAN_KIND_INTERNAL",
         "startTimeUnixNano": "0", "endTimeUnixNano": "1",
         "status": {"code": "STATUS_CODE_OK", "message": ""},
         "attributes": [], "events": [], "links": []},
    ]
    persisted = _persisted(runs=runs)
    drift = detector.compare(persisted)
    assert len(drift) == 1
    assert drift[0]["severity"] == SEVERITY_INFO
    assert drift[0]["field_path"] == "event_log.runs[0].status"
    assert "kernel restart truncated" in drift[0]["current_value"]
    # The persisted runs list MUST be mutated in place.
    assert runs[0]["endTimeUnixNano"] is not None
    assert runs[0]["status"]["code"] == "STATUS_CODE_ERROR"
    assert runs[0]["status"]["message"] == "kernel restart truncated"
    # The closed span is unchanged.
    assert runs[1]["status"]["code"] == "STATUS_CODE_OK"


def test_truncate_in_progress_spans_returns_one_event_per_open_span() -> None:
    runs = [
        {"endTimeUnixNano": None,
         "status": {"code": "STATUS_CODE_UNSET", "message": ""}},
        {"endTimeUnixNano": None,
         "status": {"code": "STATUS_CODE_UNSET", "message": ""}},
    ]
    events = truncate_in_progress_spans(runs)
    assert len(events) == 2
    assert all(e["severity"] == SEVERITY_INFO for e in events)


# ---------------------------------------------------------------------------
# Agent process status drift
# ---------------------------------------------------------------------------


def test_agent_status_drift_is_info_and_updates_node_in_place() -> None:
    detector = DriftDetector()
    nodes = [
        {"id": "agent:alpha", "type": "agent",
         "properties": {"status": "busy"}},
    ]
    persisted = _persisted(agent_nodes=nodes)
    drift = detector.compare(
        persisted, current_agent_status={"alpha": "crashed"},
    )
    assert len(drift) == 1
    assert drift[0]["severity"] == SEVERITY_INFO
    assert drift[0]["previous_value"] == "busy"
    assert drift[0]["current_value"] == "crashed"
    # Persisted node is updated in place to current reality.
    assert nodes[0]["properties"]["status"] == "crashed"


# ---------------------------------------------------------------------------
# Drift event schema
# ---------------------------------------------------------------------------


def test_drift_event_carries_required_fields() -> None:
    detector = DriftDetector()
    persisted = _persisted(volatile_kernel={"model_default": "a"})
    drift = detector.compare(persisted, current_kernel={"model_default": "b"})
    event = drift[0]
    for key in ("detected_at", "field_path", "previous_value",
                "current_value", "severity", "operator_acknowledged"):
        assert key in event
    assert event["operator_acknowledged"] is False

"""operator.action dispatch table per RFC-006 §6.

K-MCP V1 mega-round: ``_route_operator_action`` previously routed
only ``approval_response`` and log-and-dropped everything else.  This
file pins the new dispatch behavior:

* ``cell_edit`` -> INFO log line (no kernel state mutation in V1).
* ``branch_switch`` -> INFO log line carrying the new branch name.
* ``zone_select`` -> INFO log line with the focused zone id.
* ``dismiss_notification`` -> INFO log line; if the run-tracker has an
  open span at ``correlation_id``, mark ``llmnb.dismissed: true``.
* Unknown ``action_type`` -> WARN log line, no raise.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List

import pytest

from llm_kernel._attrs import decode_attrs
from llm_kernel.mcp_server import OperatorBridgeServer
from llm_kernel.run_tracker import RunTracker


class _ListSink:
    """Test sink that captures every emitted envelope."""

    def __init__(self) -> None:
        self.envelopes: List[Dict[str, Any]] = []

    def emit(self, env: Dict[str, Any]) -> None:
        self.envelopes.append(env)


@pytest.fixture
def bridge_with_tracker() -> tuple[OperatorBridgeServer, RunTracker]:
    sink = _ListSink()
    tracker = RunTracker(
        trace_id=str(uuid.uuid4()), sink=sink,
        agent_id="test-agent", zone_id="test-zone",
    )
    bridge = OperatorBridgeServer(
        agent_id="test-agent", zone_id="test-zone",
        trace_id=str(uuid.uuid4()), run_tracker=tracker,
    )
    return bridge, tracker


def _envelope(action_type: str, params: Dict[str, Any], **extra) -> Dict[str, Any]:
    """Build an RFC-006 §6 operator.action envelope (extension->kernel)."""
    payload = {"action_type": action_type, "parameters": params}
    payload.update(extra)
    env: Dict[str, Any] = {"type": "operator.action", "payload": payload}
    return env


def test_cell_edit_emits_info_log(
    bridge_with_tracker: tuple[OperatorBridgeServer, RunTracker],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """cell_edit MUST log an INFO record with the params; no exceptions."""
    bridge, _ = bridge_with_tracker
    with caplog.at_level(logging.INFO, logger="llm_kernel.mcp"):
        bridge._route_operator_action(_envelope(
            "cell_edit", {"cell_id": "c1", "diff": "+x"},
        ))
    assert any(
        rec.levelno == logging.INFO and "cell_edit" in rec.getMessage()
        for rec in caplog.records
    )


def test_branch_switch_emits_info_log(
    bridge_with_tracker: tuple[OperatorBridgeServer, RunTracker],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """branch_switch MUST log INFO carrying ``new_branch``."""
    bridge, _ = bridge_with_tracker
    with caplog.at_level(logging.INFO, logger="llm_kernel.mcp"):
        bridge._route_operator_action(_envelope(
            "branch_switch", {"new_branch": "feature/foo"},
        ))
    matching = [r for r in caplog.records if "branch_switch" in r.getMessage()]
    assert matching, "expected at least one branch_switch log record"
    # ``new_branch`` is set as an extra on the LogRecord.
    assert any(getattr(r, "new_branch", None) == "feature/foo" for r in matching)


def test_zone_select_emits_info_log(
    bridge_with_tracker: tuple[OperatorBridgeServer, RunTracker],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """zone_select MUST log INFO; the selected zone id is on the record."""
    bridge, _ = bridge_with_tracker
    with caplog.at_level(logging.INFO, logger="llm_kernel.mcp"):
        bridge._route_operator_action(_envelope(
            "zone_select", {"zone_id": "zone-A"},
        ))
    matching = [r for r in caplog.records if "zone_select" in r.getMessage()]
    assert matching
    assert any(getattr(r, "selected_zone_id", None) == "zone-A" for r in matching)


def test_dismiss_notification_marks_open_span_dismissed(
    bridge_with_tracker: tuple[OperatorBridgeServer, RunTracker],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """dismiss_notification on an open span MUST add ``llmnb.dismissed: true``."""
    bridge, tracker = bridge_with_tracker
    span_id = tracker.start_run(
        name="notify", run_type="tool", inputs={"observation": "x", "importance": "info"},
    )
    with caplog.at_level(logging.INFO, logger="llm_kernel.mcp"):
        bridge._route_operator_action(_envelope(
            "dismiss_notification",
            {"notification_id": "n-1", "correlation_id": span_id},
        ))
    span = tracker.get_run(span_id)
    attrs = decode_attrs(list(span.attributes))
    assert attrs.get("llmnb.dismissed") is True
    assert any(
        "dismiss_notification" in r.getMessage() for r in caplog.records
    )


def test_dismiss_notification_unknown_correlation_logs_only(
    bridge_with_tracker: tuple[OperatorBridgeServer, RunTracker],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unknown correlation_id MUST NOT raise; logs only."""
    bridge, _ = bridge_with_tracker
    with caplog.at_level(logging.INFO, logger="llm_kernel.mcp"):
        bridge._route_operator_action(_envelope(
            "dismiss_notification",
            {"notification_id": "n-2", "correlation_id": "deadbeefdeadbeef"},
        ))
    assert any(
        "dismiss_notification" in r.getMessage() for r in caplog.records
    )


def test_unknown_action_type_logs_warning_no_raise(
    bridge_with_tracker: tuple[OperatorBridgeServer, RunTracker],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unknown action_type MUST log a WARN and not raise."""
    bridge, _ = bridge_with_tracker
    with caplog.at_level(logging.WARNING, logger="llm_kernel.mcp"):
        bridge._route_operator_action(_envelope(
            "v1.5_future_action", {"x": "y"},
        ))
    assert any(
        rec.levelno == logging.WARNING and "unknown action_type" in rec.getMessage()
        for rec in caplog.records
    )


def test_approval_response_still_routes(
    bridge_with_tracker: tuple[OperatorBridgeServer, RunTracker],
) -> None:
    """approval_response routing MUST be preserved after the dispatch overhaul."""
    import asyncio
    bridge, _ = bridge_with_tracker

    async def _runner() -> None:
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Dict[str, Any]] = loop.create_future()
        run_id = "abc123"
        # Register the future on the bridge's pending table directly
        # (simulating an in-flight ask/clarify/propose round-trip).
        with bridge._pending_lock:
            bridge._pending_responses[run_id] = future
        bridge._route_operator_action(_envelope(
            "approval_response",
            {"request_id": run_id, "decision": "approve"},
        ))
        result = await asyncio.wait_for(future, timeout=2.0)
        assert result["decision"] == "approve"

    asyncio.run(_runner())

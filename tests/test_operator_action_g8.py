"""V1 Kernel Gap Closure G8 — operator.action handler routing per kind.

Companion to ``test_operator_action_dispatch.py``.  This file pins the
G8-acceptance routing decisions for the four "remaining" action kinds
the closure brief enumerates:

* ``cell_edit``        -> log-only (V1: no kernel state mutation;
  cell-edit semantics owned by the K-CM cell-manager slice not yet
  shipped — see PLAN-S5.5-sections.md).
* ``branch_switch``    -> log-only (zone branch routing owned by the
  K-MW branch slice — see PLAN-S5-branch-revert-stop.md).
* ``zone_select``      -> log-only (renderer-side state; the kernel's
  zone-select handler is purely observational in V1).
* ``dismiss_notification`` -> log + ``llmnb.dismissed`` attribute on
  the matching open span (real handler).

Each kind is a "real" handler in the sense that the dispatcher routes
the kind explicitly and emits a structured log line — but for the
three K-class-blocked kinds (cell_edit, branch_switch, zone_select)
the V1 handler intentionally avoids mutating kernel state until the
owning slice lands.  The brief calls these "clean K-class refusals";
in our implementation they're documented log-only handlers carrying
the params verbatim, so a downstream operator surface (renderer)
sees the action and may act on it.

K-MCP V1 mega-round Round B G8 closure.  See
``docs/atoms/protocols/operator-action.md``.
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


def _envelope(action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "operator.action",
        "payload": {"action_type": action_type, "parameters": params},
    }


# --- G8 routing: each kind dispatches without raising and without leaking
# the action into a different kind's handler. ------------------------------


def test_cell_edit_routes_or_returns_clean_kclass(
    bridge_with_tracker: tuple[OperatorBridgeServer, RunTracker],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """cell_edit MUST route to its own log line; not raise; not mutate state.

    The K-CM cell-manager slice owns mutation; the V1 K-MCP handler
    is observational.  We pin: an INFO record carrying ``cell_edit``
    text, and the params are forwarded as a record extra.
    """
    bridge, _ = bridge_with_tracker
    with caplog.at_level(logging.INFO, logger="llm_kernel.mcp"):
        bridge._route_operator_action(_envelope(
            "cell_edit",
            {"cell_id": "cell-g8", "diff": "+inserted"},
        ))
    matching = [
        r for r in caplog.records
        if r.levelno == logging.INFO and "cell_edit" in r.getMessage()
    ]
    assert matching, "expected cell_edit INFO log"
    record = matching[0]
    params_extra = getattr(record, "parameters", {})
    assert params_extra.get("cell_id") == "cell-g8"
    assert params_extra.get("diff") == "+inserted"


def test_branch_switch_routes_or_returns_clean_kclass(
    bridge_with_tracker: tuple[OperatorBridgeServer, RunTracker],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """branch_switch MUST route to its own log line; carries ``new_branch``.

    Branch routing is owned by the K-MW slice (see
    PLAN-S5-branch-revert-stop.md); the V1 K-MCP handler is
    observational and surfaces the chosen branch on the log record.
    Also accepts the legacy ``branch`` parameter alias.
    """
    bridge, _ = bridge_with_tracker
    with caplog.at_level(logging.INFO, logger="llm_kernel.mcp"):
        bridge._route_operator_action(_envelope(
            "branch_switch", {"new_branch": "wip/g8-routing"},
        ))
        bridge._route_operator_action(_envelope(
            "branch_switch", {"branch": "legacy/alias"},
        ))
    branches_seen = [
        getattr(r, "new_branch", None)
        for r in caplog.records if "branch_switch" in r.getMessage()
    ]
    assert "wip/g8-routing" in branches_seen
    assert "legacy/alias" in branches_seen  # alias accepted


def test_zone_select_routes_or_returns_clean_kclass(
    bridge_with_tracker: tuple[OperatorBridgeServer, RunTracker],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """zone_select MUST route to its own log line; carries ``selected_zone_id``.

    V1 zone-select is renderer-side state; the kernel does not switch
    its operating zone in response.  The log line is the contract
    surface that lets renderers / observability tools see the
    selection.
    """
    bridge, _ = bridge_with_tracker
    with caplog.at_level(logging.INFO, logger="llm_kernel.mcp"):
        bridge._route_operator_action(_envelope(
            "zone_select", {"zone_id": "zone-g8-target"},
        ))
    matching = [r for r in caplog.records if "zone_select" in r.getMessage()]
    assert matching, "expected zone_select INFO log"
    assert any(
        getattr(r, "selected_zone_id", None) == "zone-g8-target"
        for r in matching
    )
    # Sanity: did NOT route into the cell_edit / branch_switch handlers.
    assert not any("cell_edit" in r.getMessage() for r in matching)
    assert not any("branch_switch" in r.getMessage() for r in matching)


def test_dismiss_notification_routes_or_returns_clean_kclass(
    bridge_with_tracker: tuple[OperatorBridgeServer, RunTracker],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """dismiss_notification MUST route, log, and tag the open span.

    This handler IS implemented end-to-end (no K-class blocker): the
    matching open span gets ``llmnb.dismissed: true`` so the
    operator surface can render the dismissal.
    """
    bridge, tracker = bridge_with_tracker
    span_id = tracker.start_run(
        name="notify", run_type="tool",
        inputs={"observation": "test", "importance": "info"},
    )
    with caplog.at_level(logging.INFO, logger="llm_kernel.mcp"):
        bridge._route_operator_action(_envelope(
            "dismiss_notification",
            {"notification_id": "n-g8", "correlation_id": span_id},
        ))
    span = tracker.get_run(span_id)
    attrs = decode_attrs(list(span.attributes))
    assert attrs.get("llmnb.dismissed") is True
    matching = [
        r for r in caplog.records
        if "dismiss_notification" in r.getMessage()
    ]
    assert matching, "expected dismiss_notification INFO log"
    record = matching[0]
    assert getattr(record, "notification_id", None) == "n-g8"
    assert getattr(record, "correlation_id", None) == span_id

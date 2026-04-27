"""Per-scenario contract tests (deterministic; no Hypothesis randomization).

Each of the eight scenarios from :mod:`scenarios` runs through
:class:`EventSequencer`; the test asserts the captured envelopes match
the documented :class:`ScenarioOutcome` (run count, terminal status,
the subset of RFC-004 §"Property-based invariants" the scenario
specifically validates).

Hypothesis is for the property-based generalization in
:mod:`test_invariants`; this module is the deterministic baseline that
fails fast on a regression in any single scenario.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from .harness import EventSequencer
from .invariants import (
    InvariantResult,
    i1_run_lifecycle_closed,
    i2_request_approval_resolved,
    i3_state_reconstructable,
    i4_no_zone_overlap,
    i5_one_run_per_tool_call,
    i6_correlation_ids_unique_uuid4,
    i7_heartbeat_cadence,
    i8_layout_tree_invariants,
    i9_agent_graph_invariants,
)
from .scenarios import SCENARIOS, ScenarioFactory

_INVARIANT_FNS: Dict[str, object] = {
    "I1": i1_run_lifecycle_closed,
    "I2": i2_request_approval_resolved,
    "I3": i3_state_reconstructable,
    "I4": i4_no_zone_overlap,
    "I5": None,  # I5 is parameterized — handled below.
    "I6": i6_correlation_ids_unique_uuid4,
    "I7": i7_heartbeat_cadence,
    "I8": i8_layout_tree_invariants,
    "I9": i9_agent_graph_invariants,
}


def _expected_tool_call_count(factory: ScenarioFactory) -> int:
    """Count tool_calls that should produce a run record."""
    return sum(
        1 for e in factory.build()
        if e.kind == "tool_call" and e.expected_outcome != "method_not_found"
    )


def _expected_model_call_count(factory: ScenarioFactory) -> int:
    """Count model calls — the streaming case is one run record."""
    return sum(1 for e in factory.build() if e.kind == "model_call")


@pytest.mark.parametrize(
    "scenario_id", sorted(SCENARIOS.keys()),
    ids=sorted(SCENARIOS.keys()),
)
def test_scenario_run_count_matches_outcome(scenario_id: str) -> None:
    """The run-tracker's run count MUST equal ``expected_run_count``."""
    factory = SCENARIOS[scenario_id]
    result = EventSequencer(factory.build()).run()
    starts = [e for e in result.events if e["message_type"] == "run.start"]
    assert len(starts) == factory.outcome.expected_run_count, (
        f"{scenario_id}: expected {factory.outcome.expected_run_count} "
        f"run.start, observed {len(starts)}"
    )


@pytest.mark.parametrize(
    "scenario_id", sorted(SCENARIOS.keys()),
    ids=sorted(SCENARIOS.keys()),
)
def test_scenario_terminal_statuses_match(scenario_id: str) -> None:
    """The terminal status of each run MUST match ``expected_status_for_each``."""
    factory = SCENARIOS[scenario_id]
    result = EventSequencer(factory.build()).run()
    # Walk run.complete envelopes in insertion order; for each
    # correlation_id keep the *last* status (timeout scenario stamps a
    # follow-up run.error). The list is keyed to insertion order of the
    # FIRST run.complete per run, mirroring how the kernel commits.
    status_by_run: Dict[str, str] = {}
    insertion_order: List[str] = []
    for env in result.events:
        if env["message_type"] != "run.complete":
            continue
        cid = env["correlation_id"]
        status = env["payload"]["status"]
        if cid not in status_by_run:
            insertion_order.append(cid)
        status_by_run[cid] = status
    observed = [status_by_run[cid] for cid in insertion_order]
    assert observed == factory.outcome.expected_status_for_each, (
        f"{scenario_id}: expected {factory.outcome.expected_status_for_each}, "
        f"observed {observed}"
    )


@pytest.mark.parametrize(
    "scenario_id", sorted(SCENARIOS.keys()),
    ids=sorted(SCENARIOS.keys()),
)
def test_scenario_validates_its_documented_invariants(scenario_id: str) -> None:
    """Each scenario asserts the invariants in its ``expected_invariants_pass``."""
    factory = SCENARIOS[scenario_id]
    result = EventSequencer(factory.build()).run()
    n_tool = _expected_tool_call_count(factory)
    failures: List[str] = []
    for inv_id in factory.outcome.expected_invariants_pass:
        if inv_id == "I5":
            outcome: InvariantResult = i5_one_run_per_tool_call(result, n_tool)
        else:
            fn = _INVARIANT_FNS.get(inv_id)
            if fn is None:
                continue
            outcome = fn(result)  # type: ignore[operator]
        if not outcome.ok:
            failures.append(f"{inv_id}: {outcome.message}")
    assert not failures, (
        f"{scenario_id} failed invariants: " + "; ".join(failures)
    )


def test_unknown_tool_yields_no_run_record() -> None:
    """RFC-001 §Failure modes — unknown tool ⇒ -32601, no run record."""
    factory = SCENARIOS["unknown_tool_rejected"]
    result = EventSequencer(factory.build()).run()
    starts = [e for e in result.events if e["message_type"] == "run.start"]
    assert starts == [], f"unknown tool should NOT create a run.start, got {len(starts)}"
    # The errors list MAY contain the McpError if the harness re-raised
    # it, but the sequencer swallows -32601 specifically; assert nothing
    # else escaped.
    assert result.errors == []


def test_request_approval_timeout_marks_error() -> None:
    """RFC-001 §-32002 — approval timeout MUST close the run as error."""
    factory = SCENARIOS["request_approval_timeout"]
    result = EventSequencer(factory.build()).run()
    completes = [e for e in result.events if e["message_type"] == "run.complete"]
    assert completes, "timeout scenario should emit run.complete"
    # Final status across the run lifecycle MUST be error.
    terminal = completes[-1]
    assert terminal["payload"]["status"] == "error"
    assert terminal["payload"]["error"]["message"] == "approval_timeout"


def test_streaming_model_call_emits_token_events() -> None:
    """RFC-003 §run.event(token) — five chunks ⇒ five token events."""
    factory = SCENARIOS["litellm_streaming_call"]
    result = EventSequencer(factory.build()).run()
    token_events = [
        e for e in result.events
        if e["message_type"] == "run.event"
        and e["payload"]["event_type"] == "token"
    ]
    assert len(token_events) == 5, f"expected 5 token events, got {len(token_events)}"
    completes = [e for e in result.events if e["message_type"] == "run.complete"]
    assert len(completes) == 1, "streaming model call should emit one run.complete"

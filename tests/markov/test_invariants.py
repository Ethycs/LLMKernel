"""Hypothesis-based property tests for RFC-004 invariants.

Each test takes a randomly-sampled scenario from :data:`SCENARIOS`,
drives it through :class:`EventSequencer`, and asserts the relevant
RFC-004 invariant holds. The fault-injection variants exercise the
Markov scheduler from :class:`FaultInjector`.

Profile: the default ``max_examples=100`` runs in CI; a dedicated
``--hypothesis-profile=nightly`` profile is registered below for the
slow-suite cadence (1000 examples) per RFC-004 §"Doc-driven contract
test families".
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from .harness import EventSequencer, FaultInjector, FaultMatrix, ReplayHarness, ReplayMode
from .invariants import (
    UUIDV4_RE,
    i1_run_lifecycle_closed,
    i3_state_reconstructable,
    i5_one_run_per_tool_call,
    i6_correlation_ids_unique_uuid4,
)
from .scenarios import SCENARIOS, ScenarioFactory

# Hypothesis profiles: fast (CI default) and nightly (slow suite).
settings.register_profile("ci", max_examples=100, deadline=None,
                          suppress_health_check=[HealthCheck.too_slow,
                                                 HealthCheck.function_scoped_fixture])
settings.register_profile("nightly", max_examples=1000, deadline=None,
                          suppress_health_check=[HealthCheck.too_slow,
                                                 HealthCheck.function_scoped_fixture])
settings.load_profile("ci")


_factory_strategy = st.sampled_from(sorted(SCENARIOS.values(), key=lambda f: f.scenario_id))


def _expected_tool_calls(factory: ScenarioFactory) -> int:
    """Count tool_call kinds in a scenario's event list."""
    return sum(1 for e in factory.build() if e.kind == "tool_call"
               and e.expected_outcome != "method_not_found")


@given(factory=_factory_strategy)
def test_invariant_I1_holds_under_random_scenarios(
    factory: ScenarioFactory,
) -> None:
    """RFC-004 §I1 — every run.start has run.complete within T."""
    sequencer = EventSequencer(factory.build())
    result = sequencer.run()
    outcome = i1_run_lifecycle_closed(result)
    assert outcome.ok, (
        f"I1 violated for {factory.scenario_id}: {outcome.message}"
    )


@given(factory=_factory_strategy)
def test_invariant_I3_state_reconstruction(
    factory: ScenarioFactory,
) -> None:
    """RFC-004 §I3 — DRY-replay reconstructs the same final state."""
    sequencer = EventSequencer(factory.build())
    result = sequencer.run()
    # Replay the captured log through DRY mode and assert the fold matches.
    replay = ReplayHarness(result.events, mode=ReplayMode.DRY).run()
    assert replay.final_state == result.final_state, (
        f"I3 violated for {factory.scenario_id}: "
        f"replay={replay.final_state} live={result.final_state}"
    )
    outcome = i3_state_reconstructable(result)
    assert outcome.ok, outcome.message


def test_invariant_I5_one_run_per_tool_call() -> None:
    """RFC-004 §I5 — exactly one run record per MCP tool call.

    Two scenarios with deterministic shape (no Hypothesis): notify_smoke
    has one tool call ⇒ one run; report_progress_stream has six ⇒ six.
    """
    for scenario_id in ("notify_smoke", "report_progress_stream"):
        factory = SCENARIOS[scenario_id]
        result = EventSequencer(factory.build()).run()
        n_calls = _expected_tool_calls(factory)
        outcome = i5_one_run_per_tool_call(result, n_calls)
        assert outcome.ok, (
            f"I5 violated for {scenario_id}: {outcome.message} "
            f"(events={len(result.events)})"
        )


@given(factory=_factory_strategy)
def test_invariant_I6_correlation_id_uniqueness(
    factory: ScenarioFactory,
) -> None:
    """RFC-004 §I6 — correlation_ids are UUIDv4 and unique within a session."""
    result = EventSequencer(factory.build()).run()
    outcome = i6_correlation_ids_unique_uuid4(result)
    assert outcome.ok, (
        f"I6 violated for {factory.scenario_id}: {outcome.message}"
    )


@given(seed=st.integers(min_value=0, max_value=1 << 30))
@settings(max_examples=25, deadline=None)
def test_fault_injection_drop_does_not_violate_I3(seed: int) -> None:
    """Drops shrink the log but I3 still holds — fail-closed posture.

    RFC-004 §"Recovery posture": V1 fails CLOSED. A dropped run.start
    means the run never appears in state; a dropped run.complete means
    the run stays in ``open``. In both cases ``fold(log)`` and the
    final state agree (because final_state is itself derived from the
    log we captured AFTER the drop).
    """
    factory = SCENARIOS["report_progress_stream"]
    sequencer = EventSequencer(factory.build())
    matrix = FaultMatrix(p_drop=0.05, p_delay=0.0, p_corrupt=0.0,
                         p_disconnect=0.0)
    result = FaultInjector(sequencer, matrix=matrix, seed=seed).run()
    outcome = i3_state_reconstructable(result)
    assert outcome.ok, (
        f"I3 violated under p_drop=0.05 (seed={seed}): {outcome.message}"
    )


@given(seed=st.integers(min_value=0, max_value=1 << 30))
@settings(max_examples=25, deadline=None)
def test_fault_injection_corrupt_correlation_id_is_caught(seed: int) -> None:
    """Corrupt envelopes appear in the captured log but NOT in iter_runs.

    The injector pushes corrupted envelopes directly onto the captured
    list (bypassing the run-tracker so its validate_envelope call does
    not fire); the run-tracker's run records remain UUIDv4-keyed. This
    matches RFC-003 §F1 + RFC-004 §I6: the receiver MUST log and drop
    the bad envelope; the in-memory state is unaffected.
    """
    factory = SCENARIOS["report_progress_stream"]
    sequencer = EventSequencer(factory.build())
    matrix = FaultMatrix(p_drop=0.0, p_delay=0.0, p_corrupt=0.10,
                         p_disconnect=0.0)
    result = FaultInjector(sequencer, matrix=matrix, seed=seed).run()
    # Run-tracker MUST hold only UUIDv4 ids — corruptions did not poison it.
    assert result.run_tracker is not None
    for record in result.run_tracker.iter_runs():
        assert UUIDV4_RE.match(record.id), (
            f"corrupt id leaked into run-tracker: {record.id!r}"
        )


def test_replay_partial_mode_filters_log() -> None:
    """RFC-004 §"Partial replay" — selector filters to a sub-population."""
    factory = SCENARIOS["multi_agent_concurrent"]
    result = EventSequencer(factory.build()).run()
    # Pick the first run.start's correlation_id and assert PARTIAL keeps
    # only that family of envelopes.
    target_cid = next(
        e["correlation_id"] for e in result.events
        if e["message_type"] == "run.start"
    )
    replay = ReplayHarness(
        result.events, mode=ReplayMode.PARTIAL,
        selector=lambda env: env.get("correlation_id") == target_cid,
    ).run()
    assert all(e["correlation_id"] == target_cid for e in replay.events)
    assert len(replay.events) > 0


def test_replay_live_mode_reconstructs_run_count() -> None:
    """RFC-004 §"Live replay" — live re-driving yields one run per recorded run."""
    factory = SCENARIOS["notify_smoke"]
    original = EventSequencer(factory.build()).run()
    replay = ReplayHarness(original.events, mode=ReplayMode.LIVE).run()
    n_starts_orig = sum(1 for e in original.events if e["message_type"] == "run.start")
    n_starts_replay = sum(1 for e in replay.events if e["message_type"] == "run.start")
    assert n_starts_orig == n_starts_replay


def test_partial_replay_requires_selector() -> None:
    """PARTIAL mode without a selector MUST raise ValueError per the harness contract."""
    with pytest.raises(ValueError, match="selector"):
        ReplayHarness([], mode=ReplayMode.PARTIAL)


def test_fault_matrix_validate_rejects_out_of_range() -> None:
    """FaultMatrix MUST reject probabilities outside [0, 1] (RFC-004 schema)."""
    with pytest.raises(ValueError):
        FaultMatrix(p_drop=1.5).validate()
    with pytest.raises(ValueError):
        FaultMatrix(p_drop=-0.1).validate()


__all__: List[str] = []

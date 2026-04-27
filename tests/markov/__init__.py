"""Stage 4 Track T2 — RFC-004 Markov fault-injection harness (kernel side).

This package implements the kernel-side test harness that RFC-004
("Failure-mode analysis and fault-injection harness") names as the
Stage 4 deliverable for the V1 reliability foundation. The harness has
four pieces, one per public module below:

* :mod:`scenarios` — pre-baked event sequences (the eight known-good
  interaction patterns the harness drives, per RFC-004 §"Worked example"
  and §"Replay harness modes").
* :mod:`harness` — the driver: an in-memory event sequencer wired
  against a real :class:`llm_kernel.mcp_server.OperatorBridgeServer`,
  plus the fault injector and the three RFC-004 replay modes
  (LIVE / DRY / PARTIAL).
* :mod:`invariants` — the nine RFC-004 §"Property-based invariants"
  predicates as pure functions over a captured envelope log.
* :mod:`test_invariants` / :mod:`test_scenarios` — the Hypothesis-based
  property suite plus the deterministic per-scenario contract suite.

All tests under this package run inside the kernel pixi feature
(``hypothesis`` is declared there). The fast suite (default Hypothesis
profile, 100 examples) runs on every CI commit; the nightly suite
(1000 examples) runs in the slow-suite cadence per RFC-004 §"Doc-driven
contract test families".
"""

from __future__ import annotations

from .harness import EventSequencer, FaultInjector, ReplayHarness, ReplayMode, RunResult
from .invariants import InvariantResult, run_all_invariants
from .scenarios import SCENARIOS, Event, ScenarioOutcome

__all__ = [
    "Event",
    "EventSequencer",
    "FaultInjector",
    "InvariantResult",
    "ReplayHarness",
    "ReplayMode",
    "RunResult",
    "SCENARIOS",
    "ScenarioOutcome",
    "run_all_invariants",
]

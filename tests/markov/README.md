# RFC-004 Markov fault-injection harness (Stage 4 Track T2)

This package implements the kernel-side test harness specified by
[RFC-004 — Failure-mode analysis and fault-injection harness](../../../../docs/rfcs/RFC-004-failure-modes.md).

## Why

RFC-004 §"Specification" applies Bell-System fault-tree discipline to
the kernel/notebook split: every failure mode has a documented
recovery path, and the harness MUST verify that the system fails
CLOSED (agent operations halt; the operator surface shows a structured
error; no silent retries). Track T2 is the kernel-side suite that
exercises the nine RFC-004 §"Property-based invariants" against
thousands of randomized sequences.

## What

- **`scenarios.py`** — eight pre-baked happy-path event sequences
  (notify smoke, request_approval round-trip, request_approval timeout,
  report_progress streaming, multi-agent concurrency, LiteLLM
  passthrough, LiteLLM streaming, unknown-tool rejection).
- **`harness.py`** — the driver: an `EventSequencer` against a real
  `OperatorBridgeServer` + `RunTracker`; a Markov-style `FaultInjector`
  with a probability matrix per RFC-004 §"Fault-injection scheduler";
  and a `ReplayHarness` covering the three RFC-004 modes (LIVE / DRY /
  PARTIAL).
- **`invariants.py`** — pure predicates for I1..I9 over a captured
  envelope log; each cites its RFC-004 section.
- **`test_invariants.py`** — Hypothesis property tests over
  `SCENARIOS`. Profile `ci` (default, 100 examples) for fast CI;
  profile `nightly` (1000 examples) for the slow suite.
- **`test_scenarios.py`** — deterministic per-scenario contract tests
  that fail fast on a regression in any single scenario.

## How to run

Fast suite (default Hypothesis profile, 100 examples):

```bash
pixi run -e kernel pytest vendor/LLMKernel/tests/markov/ -v
```

Nightly slow suite (1000 examples):

```bash
pixi run -e kernel pytest vendor/LLMKernel/tests/markov/ \
  --hypothesis-profile=nightly -v
```

Reproduce a specific Hypothesis failure (per RFC-004 §"Fault-injection
scheduler" — same seed + same scenario MUST produce the same schedule):

```bash
pixi run -e kernel pytest vendor/LLMKernel/tests/markov/ \
  --hypothesis-seed=42 -v
```

## The nine invariants

| ID | Witness | V1 status |
|----|---------|-----------|
| I1 | every run.start has matching run.complete within T=300s | real (every scenario) |
| I2 | every request_approval has operator response or documented timeout | real (request_approval scenarios only) |
| I3 | in-memory state reconstructable from append-only event log | real (every scenario) |
| I4 | no two simultaneous agent operations on the same zone | real (multi_agent_concurrent only) |
| I5 | every MCP tool call produces exactly one run record | real (every scenario) |
| I6 | correlation_ids are UUIDv4 and unique within a session | real (every scenario) |
| I7 | heartbeats arrive at 10s ± 2s when kernel healthy | n/a in V1 markov suite (no synthesized heartbeats) |
| I8 | layout-tree mutations preserve tree invariants | real (layout-edit scenarios only; no V1 scenario triggers) |
| I9 | agent-graph mutations preserve graph invariants | real (agent_graph scenarios only; no V1 scenario triggers) |

V1 summary: 4 real on every scenario (I1, I3, I5, I6), 4 scenario-specific
(I2, I4, I8, I9), 1 n/a (I7).

## TODO(T2-coverage)

- Coverage-guided sequence generation per RFC-004 §"Open extensions":
  feed the `FaultInjector`'s injection log back into Hypothesis's
  shrinker so failing examples minimize on the *injection point*,
  not the scenario shape.
- Synthesize heartbeat envelopes so I7 becomes assertable here rather
  than deferred to the extension-side suite.
- Synthesize `layout.update` / `layout.edit` and `agent_graph.query` /
  `agent_graph.response` traffic so I8 and I9 become assertable on
  every replayed sequence.

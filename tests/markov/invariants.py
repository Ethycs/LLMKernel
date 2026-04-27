"""RFC-004 §"Property-based invariants" — pure predicates over a RunResult.

Each function takes a :class:`~.harness.RunResult` and returns an
:class:`InvariantResult` ``(ok, message)``. The nine RFC-004 invariants
are implemented exactly as the RFC's witness predicates state.

V1 status (per the prompt's "REQUIRED INPUTS" §"For V1"):

* I1, I3, I5, I6 — real, asserted on every scenario.
* I2, I4, I8, I9 — real, but only triggered by scenarios that mark the
  relevant signal (request_approval for I2, ``Event.zone`` for I4,
  layout/agent-graph mutations for I8 / I9).
* I7 — n/a in the V1 markov suite (the harness does not synthesize
  heartbeats; RFC-004 §"Property-based invariants" — I7 footnote).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

from .harness import RunResult

#: UUIDv4 regex per RFC-003 §Envelope.correlation_id and RFC-004 §I6.
UUIDV4_RE: re.Pattern[str] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

#: Default I1 timeout — RFC-004 §I1 ``T = 300s``.
DEFAULT_TIMEOUT_SEC: float = 300.0


@dataclass(frozen=True)
class InvariantResult:
    """Outcome of one invariant check.

    Attributes:
        invariant_id: ``"I1".."I9"`` per RFC-004 numbering.
        ok: ``True`` iff the witness predicate held.
        message: Human-readable status; on failure, names the
            divergence witness so the harness reproducer block (RFC-004
            §"Fault-injection scheduler") can quote it verbatim.
    """

    invariant_id: str
    ok: bool
    message: str


def _parse_iso(timestamp: str) -> float:
    """Return UNIX seconds for ``timestamp``; ``0.0`` if malformed."""
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return 0.0


def i1_run_lifecycle_closed(
    result: RunResult, timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> InvariantResult:
    """RFC-004 §I1 — every run.start has a matching run.complete/run.error within T."""
    starts: Dict[str, float] = {}
    completes: Dict[str, float] = {}
    for env in result.events:
        mt = env.get("message_type")
        cid = env.get("correlation_id")
        ts = _parse_iso(env.get("timestamp", ""))
        if mt == "run.start" and isinstance(cid, str):
            starts[cid] = ts
        elif mt == "run.complete" and isinstance(cid, str):
            completes[cid] = ts
    for run_id, start_ts in starts.items():
        if run_id not in completes:
            return InvariantResult("I1", False, f"run {run_id} has no run.complete")
        if completes[run_id] - start_ts > timeout_sec:
            return InvariantResult(
                "I1", False,
                f"run {run_id} took {completes[run_id] - start_ts:.3f}s > T={timeout_sec}s",
            )
    return InvariantResult("I1", True, f"all {len(starts)} run.start envelopes closed within {timeout_sec}s")


def i2_request_approval_resolved(
    result: RunResult, timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> InvariantResult:
    """RFC-004 §I2 — every request_approval has an operator response or documented timeout."""
    approval_runs: List[str] = []
    for env in result.events:
        if env.get("message_type") != "run.start":
            continue
        if env.get("payload", {}).get("name") == "request_approval":
            cid = env.get("correlation_id")
            if isinstance(cid, str):
                approval_runs.append(cid)
    if not approval_runs:
        return InvariantResult("I2", True, "no request_approval calls to validate")
    # Each approval run MUST close as success (operator responded) or
    # error (documented approval_timeout per RFC-001 §-32002).
    for run_id in approval_runs:
        completions = [
            e for e in result.events
            if e.get("message_type") == "run.complete"
            and e.get("correlation_id") == run_id
        ]
        if not completions:
            return InvariantResult("I2", False, f"approval run {run_id} has no completion")
        terminal = completions[-1]
        status = terminal.get("payload", {}).get("status")
        if status not in {"success", "error", "timeout"}:
            return InvariantResult(
                "I2", False, f"approval run {run_id} status={status!r} unrecognized",
            )
    return InvariantResult(
        "I2", True, f"all {len(approval_runs)} request_approval calls resolved",
    )


def i3_state_reconstructable(result: RunResult) -> InvariantResult:
    """RFC-004 §I3 — in-memory state reconstructable from the append-only log."""
    # Re-fold a copy of the events; assert it equals result.final_state.
    from .harness import fold_state  # local import avoids cycle on package init
    refolded = fold_state(list(result.events))
    if refolded == result.final_state:
        return InvariantResult(
            "I3", True, f"fold(log) == final_state ({len(refolded.get('runs', {}))} runs)",
        )
    return InvariantResult(
        "I3", False, f"fold mismatch: refolded={refolded} vs final_state={result.final_state}",
    )


def i4_no_zone_overlap(result: RunResult) -> InvariantResult:
    """RFC-004 §I4 — no two simultaneous agent operations on the same zone."""
    if not result.zone_lifetimes:
        return InvariantResult("I4", True, "no zone-tagged operations to validate")
    by_zone: Dict[str, List[Dict[str, Any]]] = {}
    for entry in result.zone_lifetimes:
        by_zone.setdefault(entry["zone"], []).append(entry)
    for zone, entries in by_zone.items():
        # The harness drives events sequentially per scenario, so
        # lifetimes share a single timeline; assert pairwise no overlap.
        sorted_entries = sorted(entries, key=lambda e: _parse_iso(e["start_ts"]))
        for left, right in zip(sorted_entries, sorted_entries[1:]):
            left_end = _parse_iso(left.get("end_ts") or left["start_ts"])
            right_start = _parse_iso(right["start_ts"])
            if right_start < left_end:
                return InvariantResult(
                    "I4", False,
                    f"zone {zone!r} overlap: {left['run_id']} ends "
                    f"after {right['run_id']} begins",
                )
    return InvariantResult(
        "I4", True, f"{len(result.zone_lifetimes)} zone-tagged operations all serial",
    )


def i5_one_run_per_tool_call(
    result: RunResult, expected_tool_calls: int,
) -> InvariantResult:
    """RFC-004 §I5 — every MCP tool call produces exactly one run record."""
    starts = sum(
        1 for e in result.events if e.get("message_type") == "run.start"
        and not (e.get("payload", {}).get("name", "").startswith("litellm:"))
    )
    if starts == expected_tool_calls:
        return InvariantResult(
            "I5", True, f"{starts} tool-call run.start envelopes for {expected_tool_calls} tool calls",
        )
    return InvariantResult(
        "I5", False, f"expected {expected_tool_calls} run.start, observed {starts}",
    )


def i6_correlation_ids_unique_uuid4(result: RunResult) -> InvariantResult:
    """RFC-004 §I6 — correlation_ids are UUIDv4 and unique within a session."""
    cids: List[str] = [str(e.get("correlation_id", "")) for e in result.events]
    if not cids:
        return InvariantResult("I6", True, "no envelopes to validate")
    # Per Family A (RFC-003 §run.event/run.complete) the same correlation_id
    # is reused across all envelopes for one run. So uniqueness is asserted
    # at the *run* level (one cid per run.start), not per envelope.
    start_cids = [
        str(e["correlation_id"]) for e in result.events
        if e.get("message_type") == "run.start"
    ]
    for cid in start_cids:
        if not UUIDV4_RE.match(cid):
            return InvariantResult(
                "I6", False, f"correlation_id {cid!r} is not a UUIDv4",
            )
    if len(set(start_cids)) != len(start_cids):
        dups = {c for c in start_cids if start_cids.count(c) > 1}
        return InvariantResult("I6", False, f"duplicate correlation_ids: {sorted(dups)}")
    return InvariantResult(
        "I6", True, f"{len(start_cids)} run.start correlation_ids all UUIDv4 + unique",
    )


def i7_heartbeat_cadence(result: RunResult) -> InvariantResult:
    """RFC-004 §I7 — heartbeats arrive within 8s ≤ Δ ≤ 12s when healthy.

    Marked n/a in V1 markov suite per the prompt: the harness does not
    synthesize heartbeats. Returns ``ok=True`` with an explicit n/a
    note so callers can trace the skip in the reproducer block.
    """
    return InvariantResult(
        "I7", True, "n/a in V1 markov suite (harness does not synthesize heartbeats)",
    )


def i8_layout_tree_invariants(result: RunResult) -> InvariantResult:
    """RFC-004 §I8 — layout-tree mutations preserve tree invariants."""
    layout_envs = [
        e for e in result.events
        if e.get("message_type") in {"layout.update", "layout.edit"}
    ]
    if not layout_envs:
        return InvariantResult("I8", True, "no layout envelopes to validate")
    # Walk the most-recent layout.update tree; assert acyclic + unique ids.
    snapshots = [e for e in layout_envs if e["message_type"] == "layout.update"]
    if not snapshots:
        return InvariantResult("I8", True, "no layout.update snapshots to validate")
    tree = snapshots[-1].get("payload", {}).get("tree", {})
    seen_ids: set[str] = set()
    stack: List[Dict[str, Any]] = [tree]
    while stack:
        node = stack.pop()
        nid = node.get("id")
        if not isinstance(nid, str):
            return InvariantResult("I8", False, f"layout node missing id: {node}")
        if nid in seen_ids:
            return InvariantResult("I8", False, f"layout duplicate id {nid!r}")
        seen_ids.add(nid)
        for child in node.get("children", []) or []:
            stack.append(child)
    return InvariantResult(
        "I8", True, f"layout tree valid ({len(seen_ids)} unique nodes)",
    )


def i9_agent_graph_invariants(result: RunResult) -> InvariantResult:
    """RFC-004 §I9 — agent-graph mutations preserve graph invariants."""
    graph_envs = [
        e for e in result.events
        if e.get("message_type") == "agent_graph.response"
    ]
    if not graph_envs:
        return InvariantResult("I9", True, "no agent_graph.response envelopes to validate")
    valid_kinds = {
        "spawned", "in_zone", "has_tool", "connects_to",
        "supervises", "collaborates_with", "has_capability", "configured_with",
    }
    for env in graph_envs:
        payload = env.get("payload", {})
        node_ids = {n["id"] for n in payload.get("nodes", [])}
        for edge in payload.get("edges", []):
            if edge.get("source") not in node_ids:
                return InvariantResult("I9", False, f"dangling edge source: {edge}")
            if edge.get("target") not in node_ids:
                return InvariantResult("I9", False, f"dangling edge target: {edge}")
            if edge.get("kind") not in valid_kinds:
                return InvariantResult("I9", False, f"unknown edge kind: {edge}")
    return InvariantResult(
        "I9", True, f"{len(graph_envs)} agent_graph.response envelopes all consistent",
    )


def run_all_invariants(
    result: RunResult, expected_tool_calls: int = 0,
) -> List[InvariantResult]:
    """Run all nine invariants; return the full :class:`InvariantResult` list."""
    return [
        i1_run_lifecycle_closed(result),
        i2_request_approval_resolved(result),
        i3_state_reconstructable(result),
        i4_no_zone_overlap(result),
        i5_one_run_per_tool_call(result, expected_tool_calls),
        i6_correlation_ids_unique_uuid4(result),
        i7_heartbeat_cadence(result),
        i8_layout_tree_invariants(result),
        i9_agent_graph_invariants(result),
    ]


__all__ = [
    "DEFAULT_TIMEOUT_SEC", "InvariantResult", "UUIDV4_RE",
    "i1_run_lifecycle_closed", "i2_request_approval_resolved",
    "i3_state_reconstructable", "i4_no_zone_overlap",
    "i5_one_run_per_tool_call", "i6_correlation_ids_unique_uuid4",
    "i7_heartbeat_cadence", "i8_layout_tree_invariants",
    "i9_agent_graph_invariants", "run_all_invariants",
]

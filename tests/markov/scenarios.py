"""RFC-004 happy-path scenarios for the Markov harness.

Each scenario is a function returning ``list[Event]`` plus a
:class:`ScenarioOutcome` describing the post-conditions the harness
asserts when it replays the event list. The eight scenarios below cover
the V1 minimum named by RFC-004 §"Worked example" plus the round-trips
demanded by RFC-001 §Failure modes (timeout, unknown tool) and RFC-003
Family A (streaming).

Per RFC-004, scenarios are the *known-good* sequences the
:class:`~.harness.FaultInjector` mutates at random points; the
deterministic per-scenario contract tests in
:mod:`test_scenarios` assert the unmutated baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class Event:
    """One step in a scenario.

    Attributes:
        kind: The interaction surface this step exercises. One of
            ``"tool_call"`` (agent → kernel MCP tool call),
            ``"operator.action"`` (extension → kernel inbound envelope),
            ``"model_call"`` (agent → LiteLLM proxy via HTTP), or
            ``"sleep"`` (a no-op spacer used by the fault injector).
        target: For ``tool_call``, the RFC-001 tool name; for
            ``operator.action``, the ``action_type`` value; for
            ``model_call``, the model id.
        payload: The arguments / parameters dict.
        expected_outcome: A free-form key the per-scenario contract
            tests assert against (e.g. ``"success"``, ``"timeout"``,
            ``"method_not_found"``).
        zone: Optional zone id used by I4 (no two simultaneous agent
            operations on the same zone). Default ``None`` ⇒ I4 inert.
    """

    kind: str
    target: str
    payload: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = "success"
    zone: Optional[str] = None


@dataclass(frozen=True)
class ScenarioOutcome:
    """Structured assertions a scenario commits to passing.

    Attributes:
        scenario_id: Stable string identifier the harness emits in
            reproducer blocks (RFC-004 §"Fault-injection scheduler").
        expected_run_count: Number of LangSmith run records the run
            tracker MUST hold after replay.
        expected_status_for_each: One status per run record, in run
            insertion order. Values are the RFC-003 ``run.complete``
            status enum (``"success"`` / ``"error"`` / ``"timeout"``).
            Empty list when the scenario produces *no* run records (e.g.
            the unknown-tool scenario where -32601 short-circuits).
        expected_invariants_pass: Subset of ``"I1".."I9"`` this
            scenario specifically validates.
    """

    scenario_id: str
    expected_run_count: int
    expected_status_for_each: List[str]
    expected_invariants_pass: List[str]


# RFC-004 §"Worked example" — Stage 3 paper-telephone notify smoke.
def notify_smoke() -> List[Event]:
    """Single-tool ``notify`` call → run.start → run.complete."""
    return [
        Event(
            kind="tool_call",
            target="notify",
            payload={"observation": "smoke", "importance": "info"},
            expected_outcome="success",
            zone="refactor",
        ),
    ]


# RFC-001 §request_approval — round-trip with operator approving.
def request_approval_round_trip() -> List[Event]:
    """Agent emits request_approval; operator approves; tool returns."""
    return [
        Event(
            kind="tool_call",
            target="request_approval",
            payload={
                "action": "extract validateJwt() to src/auth/jwt.rs",
                "diff_preview": {"kind": "unified_diff", "body": "@@..."},
                "risk_level": "medium",
            },
            expected_outcome="success",
            zone="refactor",
        ),
        Event(
            kind="operator.action",
            target="approval_response",
            payload={"decision": "approve"},
            expected_outcome="approve",
        ),
    ]


# RFC-001 §Failure modes — -32002 operator response timeout.
def request_approval_timeout() -> List[Event]:
    """Agent emits request_approval; no operator response; timeout."""
    return [
        Event(
            kind="tool_call",
            target="request_approval",
            payload={
                "action": "rm -rf /",
                "diff_preview": {"kind": "command", "body": "rm -rf /"},
                "risk_level": "critical",
            },
            expected_outcome="timeout",
            zone="refactor",
        ),
    ]


# RFC-001 §report_progress — non-blocking streaming progress updates.
def report_progress_stream() -> List[Event]:
    """Five report_progress calls then one report_completion."""
    events: List[Event] = []
    for i in range(5):
        events.append(
            Event(
                kind="tool_call",
                target="report_progress",
                payload={
                    "status": f"step {i + 1}/5",
                    "percent": (i + 1) * 20,
                    "display_id": "alpha-progress-1",
                },
                expected_outcome="success",
                zone="refactor",
            )
        )
    events.append(
        Event(
            kind="tool_call",
            target="report_completion",
            payload={"summary": "five steps done", "outcome": "success"},
            expected_outcome="success",
            zone="refactor",
        )
    )
    return events


# RFC-004 §I6 — correlation_id uniqueness across simultaneous agents.
def multi_agent_concurrent() -> List[Event]:
    """Two agents simultaneously emit notify; assert no run_id collision."""
    return [
        Event(
            kind="tool_call",
            target="notify",
            payload={"observation": "alpha says hi", "importance": "info"},
            expected_outcome="success",
            zone="zone-a",
        ),
        Event(
            kind="tool_call",
            target="notify",
            payload={"observation": "beta says hi", "importance": "info"},
            expected_outcome="success",
            zone="zone-b",
        ),
    ]


# RFC-002 / RFC-004 §"Worked example" — proxy receives a /v1/messages
# request, run.start → run.complete with model output.
def litellm_call_through_proxy() -> List[Event]:
    """Proxy receives a /v1/messages request; one model call run record."""
    return [
        Event(
            kind="model_call",
            target="claude-sonnet-4-6",
            payload={
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
            expected_outcome="success",
        ),
    ]


# RFC-003 §run.event with event_type=token — streaming tokens.
def litellm_streaming_call() -> List[Event]:
    """Proxy streams 5 chunks; assert 5 run.event(token) + 1 run.complete."""
    return [
        Event(
            kind="model_call",
            target="claude-sonnet-4-6",
            payload={
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "_chunks": 5,  # harness hint: how many chunks to stream
            },
            expected_outcome="success",
        ),
    ]


# RFC-001 §Failure modes — -32601 method-not-found for unknown tools.
def unknown_tool_rejected() -> List[Event]:
    """Agent calls a tool not in RFC-001; assert -32601, no run record."""
    return [
        Event(
            kind="tool_call",
            target="totally_fake_tool",
            payload={"any": "args"},
            expected_outcome="method_not_found",
            zone="refactor",
        ),
    ]


#: Public registry used by the Hypothesis property tests and the
#: per-scenario contract tests. Maps scenario_id → (factory, outcome).
SCENARIOS: Dict[str, "ScenarioFactory"] = {}


@dataclass(frozen=True)
class ScenarioFactory:
    """Bundle a scenario's event-builder with its expected outcome."""

    scenario_id: str
    build: Callable[[], List[Event]]
    outcome: ScenarioOutcome


def _register(
    scenario_id: str,
    build: Callable[[], List[Event]],
    expected_run_count: int,
    expected_status_for_each: List[str],
    expected_invariants_pass: List[str],
) -> None:
    """Register a scenario factory in :data:`SCENARIOS`."""
    SCENARIOS[scenario_id] = ScenarioFactory(
        scenario_id=scenario_id,
        build=build,
        outcome=ScenarioOutcome(
            scenario_id=scenario_id,
            expected_run_count=expected_run_count,
            expected_status_for_each=expected_status_for_each,
            expected_invariants_pass=expected_invariants_pass,
        ),
    )


_register("notify_smoke", notify_smoke, 1, ["success"], ["I1", "I3", "I5", "I6"])
_register(
    "request_approval_round_trip",
    request_approval_round_trip,
    1,
    ["success"],
    ["I1", "I2", "I5", "I6"],
)
_register(
    "request_approval_timeout",
    request_approval_timeout,
    1,
    ["error"],
    ["I1", "I2", "I6"],
)
_register(
    "report_progress_stream",
    report_progress_stream,
    6,
    ["success"] * 6,
    ["I1", "I3", "I5", "I6"],
)
_register(
    "multi_agent_concurrent",
    multi_agent_concurrent,
    2,
    ["success", "success"],
    ["I1", "I4", "I5", "I6"],
)
_register(
    "litellm_call_through_proxy",
    litellm_call_through_proxy,
    1,
    ["success"],
    ["I1", "I3", "I5", "I6"],
)
_register(
    "litellm_streaming_call",
    litellm_streaming_call,
    1,
    ["success"],
    ["I1", "I3", "I6"],
)
# Unknown tool: zero run records — RFC-001 §Failure modes line "-32601;
# no run record; kernel emits a run.policy audit event".
_register("unknown_tool_rejected", unknown_tool_rejected, 0, [], ["I5", "I6"])


__all__ = [
    "Event",
    "SCENARIOS",
    "ScenarioFactory",
    "ScenarioOutcome",
    "litellm_call_through_proxy",
    "litellm_streaming_call",
    "multi_agent_concurrent",
    "notify_smoke",
    "report_progress_stream",
    "request_approval_round_trip",
    "request_approval_timeout",
    "unknown_tool_rejected",
]

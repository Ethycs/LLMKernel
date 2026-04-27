"""RFC-004 Markov harness driver — the :class:`EventSequencer`.

Wires the eight :mod:`scenarios` against an in-memory kernel-side
stack: a real :class:`llm_kernel.run_tracker.RunTracker` whose sink
captures every RFC-003 envelope, plus a real
:class:`llm_kernel.mcp_server.OperatorBridgeServer` invoked through
``server.request_handlers[CallToolRequest]``.

The fault injector lives in :mod:`.faults` and the replay harness in
:mod:`.replay`; they're re-exported from the package root.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from mcp import types
from mcp.shared.exceptions import McpError

from llm_kernel.custom_messages import CustomMessageDispatcher
from llm_kernel.mcp_server import OperatorBridgeServer
from llm_kernel.run_envelope import (
    DIRECTION_EXTENSION_TO_KERNEL, make_envelope, validate_envelope,
)
from llm_kernel.run_tracker import RunTracker

from .scenarios import Event


@dataclass
class RunResult:
    """Outcome of a sequencer or replay run.

    Attributes:
        events: Every RFC-003 envelope captured, in insertion order.
        errors: Exceptions raised mid-run (recorded, not aborting).
        final_state: ``{"runs": {run_id: status}}`` per :func:`fold_state`.
        wall_clock_ms: Approximate duration in milliseconds.
        zone_lifetimes: Per-run zone open/close intervals (RFC-004 §I4).
        run_tracker: The live :class:`RunTracker` (None for replay results).
    """

    events: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[BaseException] = field(default_factory=list)
    final_state: Dict[str, Any] = field(default_factory=dict)
    wall_clock_ms: float = 0.0
    zone_lifetimes: List[Dict[str, Any]] = field(default_factory=list)
    run_tracker: Optional[RunTracker] = None


class CapturingSink:
    """List-sink that records every envelope the run-tracker emits."""

    def __init__(self) -> None:
        self.envelopes: List[Dict[str, Any]] = []
        self._lock: threading.Lock = threading.Lock()

    def emit(self, envelope: Dict[str, Any]) -> None:
        """Append a copy of ``envelope`` to the captured list."""
        with self._lock:
            self.envelopes.append(dict(envelope))


def fold_state(envelopes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reconstruct kernel-visible state from an append-only envelope log.

    RFC-004 §I3: ``fold(initial_state, e)`` — runs map spanId to
    ``"open"`` after ``run.start`` and to the ``run.complete``
    ``status.code`` once closed.

    Post-OTLP refactor (R1-K): payload identity is ``spanId`` (16-hex)
    and status is the OTel object ``{code, message}``.  The reduced
    state stores the OTel ``status.code`` string for I3 fold parity.
    """
    runs: Dict[str, str] = {}
    for env in envelopes:
        mt = env.get("message_type")
        payload = env.get("payload") or {}
        if mt == "run.start":
            run_id = payload.get("spanId") or env.get("correlation_id")
            if isinstance(run_id, str):
                runs[run_id] = "open"
        elif mt == "run.complete":
            run_id = payload.get("spanId") or env.get("correlation_id")
            if isinstance(run_id, str):
                status = payload.get("status") or {}
                if isinstance(status, dict):
                    runs[run_id] = str(status.get("code", "STATUS_CODE_OK"))
                else:
                    runs[run_id] = str(status)
    return {"runs": runs}


def utc_now_iso() -> str:
    """ISO 8601 UTC timestamp, millisecond precision."""
    return datetime.now(timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def _build_bridge(
    sink: CapturingSink, agent_id: str, zone_id: str,
) -> OperatorBridgeServer:
    """Build an :class:`OperatorBridgeServer` wired to ``sink``."""
    tracker = RunTracker(
        trace_id=str(uuid.uuid4()), sink=sink,
        agent_id=agent_id, zone_id=zone_id,
    )
    return OperatorBridgeServer(
        agent_id=agent_id, zone_id=zone_id,
        trace_id=tracker.trace_id, run_tracker=tracker, dispatcher=None,
    )


async def _invoke_tool(
    bridge: OperatorBridgeServer, tool_name: str, arguments: Dict[str, Any],
) -> types.CallToolResult:
    """Round-trip one MCP tool call through ``bridge``."""
    handler = bridge.server.request_handlers[types.CallToolRequest]
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=tool_name, arguments=arguments),
    )
    return (await handler(request)).root  # type: ignore[return-value]


class EventSequencer:
    """Drive a scenario's :class:`Event` list against the kernel-side stack."""

    def __init__(
        self, events: List[Event], agent_id: str = "test-agent",
        zone_id: str = "test-zone",
    ) -> None:
        """Bind to fresh sink + bridge; events list is consumed by :meth:`run`."""
        self.events: List[Event] = list(events)
        self.sink: CapturingSink = CapturingSink()
        self.bridge: OperatorBridgeServer = _build_bridge(
            self.sink, agent_id=agent_id, zone_id=zone_id,
        )
        self._zone_lifetimes: List[Dict[str, Any]] = []

    def run(self) -> RunResult:
        """Drive the events synchronously; return the captured envelopes."""
        start = time.monotonic()
        errors: List[BaseException] = []
        for event in self.events:
            try:
                self._dispatch(event)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        return RunResult(
            events=list(self.sink.envelopes), errors=errors,
            final_state=fold_state(self.sink.envelopes),
            wall_clock_ms=(time.monotonic() - start) * 1000.0,
            zone_lifetimes=list(self._zone_lifetimes),
            run_tracker=self.bridge.run_tracker,
        )

    def _dispatch(self, event: Event) -> None:
        """Route one :class:`Event` to its handler family."""
        if event.kind == "tool_call":
            self._dispatch_tool_call(event)
        elif event.kind == "operator.action":
            self._dispatch_operator_action(event)
        elif event.kind == "model_call":
            self._dispatch_model_call(event)
        elif event.kind == "sleep":
            time.sleep(0.0)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown Event.kind {event.kind!r}")

    def _dispatch_tool_call(self, event: Event) -> None:
        """Round-trip one MCP tool call; honor the timeout-expecting case.

        RFC-001 §Failure modes: -32601 for unknown tools means no run
        record. ``request_approval_timeout`` stamps a synthetic run.error
        to mirror the kernel's -32002 emission (RFC-004 §I2).
        """
        before = len(self.sink.envelopes)
        try:
            asyncio.run(_invoke_tool(self.bridge, event.target, event.payload))
        except McpError as exc:
            if exc.error.code == types.METHOD_NOT_FOUND:
                return
            raise
        after = self.sink.envelopes[before:]
        if event.zone is not None:
            self._record_zone_lifetime(after, event.zone)
        if event.expected_outcome == "timeout":
            for env in reversed(after):
                if env["message_type"] == "run.start":
                    self._stamp_timeout(env["correlation_id"])
                    break

    def _dispatch_operator_action(self, event: Event) -> None:
        """Synthesize and capture an inbound RFC-003 ``operator.action``."""
        envelope = make_envelope(
            "operator.action",
            {"action_type": event.target, "parameters": dict(event.payload)},
            correlation_id=str(uuid.uuid4()),
            direction=DIRECTION_EXTENSION_TO_KERNEL,
        )
        validate_envelope(envelope)
        self.sink.envelopes.append(envelope)

    def _dispatch_model_call(self, event: Event) -> None:
        """Drive one LiteLLM proxy /v1/messages request via the run-tracker.

        Patches ``litellm.acompletion`` so no real provider call happens.
        """
        tracker = self.bridge.run_tracker
        assert tracker is not None
        run_id = tracker.start_run(
            name=f"litellm:{event.target}", run_type="llm",
            inputs=dict(event.payload),
            metadata={"endpoint": "v1/messages",
                      "stream": bool(event.payload.get("stream"))},
        )
        if event.payload.get("stream"):
            chunks = int(event.payload.get("_chunks", 5))
            with patch("litellm.acompletion"):
                for i in range(chunks):
                    tracker.event(run_id, "token",
                                  {"chunk": {"delta": f"t{i}"}})
            tracker.complete_run(run_id, outputs={"streamed": True},
                                 status="success")
        else:
            with patch("litellm.acompletion"):
                tracker.complete_run(run_id, outputs={"id": "msg_x"},
                                     status="success")

    def _record_zone_lifetime(
        self, envelopes: List[Dict[str, Any]], zone: str,
    ) -> None:
        """Capture the start/end times of each new run's zone window."""
        for env in envelopes:
            if env["message_type"] == "run.start":
                self._zone_lifetimes.append({
                    "run_id": env["correlation_id"], "zone": zone,
                    "start_ts": env["timestamp"], "end_ts": None,
                })
        for env in envelopes:
            if env["message_type"] != "run.complete":
                continue
            for entry in self._zone_lifetimes:
                if (entry["run_id"] == env["correlation_id"]
                        and entry["end_ts"] is None):
                    entry["end_ts"] = env["timestamp"]

    def _stamp_timeout(self, run_id: str) -> None:
        """Append a synthetic run.complete(status=ERROR) for a timed-out run.

        Post-OTLP refactor (R1-K): the payload now carries OTLP shape —
        ``spanId``, ``endTimeUnixNano`` (decimal-nanoseconds string),
        OTel ``status`` object, and OTel exception attributes
        (``exception.type`` / ``exception.message``).
        """
        import time as _time
        from llm_kernel._attrs import encode_attrs
        envelope = make_envelope(
            "run.complete",
            {
                "spanId": run_id,
                "endTimeUnixNano": str(_time.time_ns()),
                "status": {
                    "code": "STATUS_CODE_ERROR",
                    "message": "approval_timeout",
                },
                "attributes": encode_attrs({
                    "exception.type": "TimeoutError",
                    "exception.message": "approval_timeout",
                    "exception.stacktrace": "",
                }),
            },
            correlation_id=run_id,
        )
        self.sink.envelopes.append(envelope)


def build_bridge_with_dispatcher(
    kernel: Any, agent_id: str = "test-agent", zone_id: str = "test-zone",
) -> tuple[OperatorBridgeServer, CustomMessageDispatcher]:
    """Build a bridge whose run-tracker sink is a real dispatcher."""
    dispatcher = CustomMessageDispatcher(kernel)
    tracker = RunTracker(
        trace_id=str(uuid.uuid4()), sink=dispatcher,
        agent_id=agent_id, zone_id=zone_id,
    )
    bridge = OperatorBridgeServer(
        agent_id=agent_id, zone_id=zone_id, trace_id=tracker.trace_id,
        run_tracker=tracker, dispatcher=dispatcher,
    )
    return bridge, dispatcher


# Re-export FaultInjector / ReplayHarness for backward compat with the
# package's __init__ imports.
from .faults import FaultInjector, FaultMatrix  # noqa: E402
from .replay import ReplayHarness, ReplayMode  # noqa: E402

__all__ = [
    "CapturingSink", "EventSequencer",
    "FaultInjector", "FaultMatrix",
    "ReplayHarness", "ReplayMode", "RunResult",
    "build_bridge_with_dispatcher", "fold_state", "utc_now_iso",
]

"""K-AS G11 — restart window per RFC-002 §"Process lifecycle" 5.

Per-agent sliding-window: ≤3 restart attempts in 300 seconds. The
fourth attempt MUST be refused with a synthetic ``report_problem``;
after the window slides past, restart attempts MUST become possible
again.

These tests exercise :meth:`AgentSupervisor._watchdog` directly with a
controllable monotonic clock so the window arithmetic is observable
without sleeping for 5 minutes.
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from llm_kernel import agent_supervisor as supervisor_mod
from llm_kernel.agent_supervisor import AgentHandle, AgentSupervisor


def _make_supervisor() -> AgentSupervisor:
    from llm_kernel.run_tracker import RunTracker

    class _ListSink:
        def __init__(self) -> None:
            self.envelopes: List[Dict[str, Any]] = []

        def emit(self, env: Dict[str, Any]) -> None:
            self.envelopes.append(env)

    tracker = RunTracker(
        trace_id=str(uuid.uuid4()), sink=_ListSink(),
        agent_id="alpha", zone_id="z1",
    )
    dispatcher = MagicMock()
    return AgentSupervisor(
        run_tracker=tracker, dispatcher=dispatcher,
        litellm_endpoint_url="http://127.0.0.1:9999/v1",
    )


def _fake_handle(agent_id: str = "alpha") -> AgentHandle:
    """Build a bare AgentHandle with the minimum fields the watchdog reads."""
    popen = MagicMock()
    popen.returncode = -1
    popen.wait.return_value = -1  # non-zero exit -> trigger restart logic
    return AgentHandle(
        agent_id=agent_id, zone_id="z1", popen=popen,
        started_at=0.0, work_dir=Path("/tmp/x"),
        stdout_thread=threading.Thread(),
        stderr_thread=threading.Thread(),
        _last_stdout_ts=0.0,
    )


def test_three_rapid_restarts_then_fourth_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three crashes inside 300s -> fourth MUST be refused."""
    sup = _make_supervisor()
    handle = _fake_handle()
    # Freeze time at t=1000.0; advance manually between calls.
    clock = [1000.0]

    def now() -> float:
        return clock[0]

    monkeypatch.setattr(supervisor_mod.time, "monotonic", now)
    monkeypatch.setattr(supervisor_mod.time, "sleep", lambda s: None)

    # Stub respawn so it's a no-op (the test cares about bookkeeping).
    monkeypatch.setattr(
        sup, "_respawn_in_place", lambda h, t: None,
    )
    # The watchdog recurses on _respawn_in_place; cap recursion by
    # invoking the bookkeeping inline rather than calling _watchdog.

    # Drive three restart attempts directly via the bookkeeping logic.
    # We observe state transitions and the deque fill.
    for i in range(3):
        clock[0] += 1.0  # 1s apart
        # Simulate one entry of crash-restart bookkeeping.
        handle._restart_history.append(now())
    assert len(handle._restart_history) == 3

    # Now the 4th attempt should be refused. Drive watchdog with a
    # popen.wait that returns immediately and a respawn that should
    # NEVER be called.
    handle.popen.wait = MagicMock(return_value=-1)
    respawn_calls: List[Any] = []

    def fake_respawn(h: Any, t: Any) -> None:
        respawn_calls.append((h, t))

    monkeypatch.setattr(sup, "_respawn_in_place", fake_respawn)
    # Capture the synthetic problem report.
    problems: List[Any] = []
    real_record = sup._record_synthetic_problem

    def record(*args: Any, **kwargs: Any) -> Any:
        problems.append((args, kwargs))
        return real_record(*args, **kwargs)

    monkeypatch.setattr(sup, "_record_synthetic_problem", record)

    sup._watchdog(handle, "task")

    assert handle.state == "failed"
    assert respawn_calls == []  # 4th attempt refused — no respawn
    assert problems, "synthetic report_problem was not emitted"
    desc = problems[0][0][2]  # signature: (agent_id, zone_id, description, log_sig)
    assert "3 restart attempts in 5 minutes" in desc
    log_sig = problems[0][0][3]
    assert log_sig == "agent.unrestartable"


def test_window_slides_after_300s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After 300s elapses, old entries are pruned; restart works again."""
    sup = _make_supervisor()
    handle = _fake_handle()
    clock = [1000.0]

    def now() -> float:
        return clock[0]

    monkeypatch.setattr(supervisor_mod.time, "monotonic", now)
    monkeypatch.setattr(supervisor_mod.time, "sleep", lambda s: None)

    # Pre-fill 3 entries at t=1000..1002 (recent enough to be in the window).
    for i in range(3):
        handle._restart_history.append(1000.0 + i)
    # Slide clock past the window (300s + slop).
    clock[0] = 1000.0 + 305.0

    respawned: List[Any] = []

    def fake_respawn(h: Any, t: Any) -> None:
        respawned.append((h, t))
        # Stop the recursion: mark stop-event so the next _watchdog call
        # short-circuits via the popen.wait check.
        h._stop_event.set()

    monkeypatch.setattr(sup, "_respawn_in_place", fake_respawn)
    # popen.wait returns -1 once (crash), then with stop_event set the
    # watchdog short-circuits after the recursive call.
    handle.popen.wait = MagicMock(return_value=-1)

    sup._watchdog(handle, "task")

    # Old entries pruned. New entry appended. Respawn invoked.
    assert respawned, "respawn was refused even though window had slid"
    assert len(handle._restart_history) == 1
    assert handle._restart_history[0] == 1305.0


def test_restart_window_is_per_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each agent has its OWN restart deque (not zone-wide)."""
    sup = _make_supervisor()
    a = _fake_handle(agent_id="alpha")
    b = _fake_handle(agent_id="beta")
    clock = [1000.0]
    monkeypatch.setattr(supervisor_mod.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(supervisor_mod.time, "sleep", lambda s: None)

    # Saturate alpha's window.
    for i in range(3):
        a._restart_history.append(1000.0 + i)

    # Beta has its own — should be empty and a fresh restart succeeds.
    monkeypatch.setattr(sup, "_respawn_in_place", lambda h, t: h._stop_event.set())
    b.popen.wait = MagicMock(return_value=-1)
    sup._watchdog(b, "task-b")
    assert len(b._restart_history) == 1
    assert b.state != "failed"
    # Alpha's deque is untouched.
    assert len(a._restart_history) == 3

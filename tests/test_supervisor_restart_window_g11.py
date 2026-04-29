"""K-AS-B / G11 — restart-window reset (≤3 in 5 minutes) acceptance tests.

The G11 audit found the sliding-window restart cap is already
implemented in :meth:`AgentSupervisor._watchdog`:

* Each ``AgentHandle`` carries a ``_restart_history: Deque[float]`` of
  monotonic timestamps.
* On each crash, the watchdog prunes entries older than
  ``_RESTART_WINDOW_SEC = 300.0`` and refuses if the remaining length
  is ``>= _RESTART_WINDOW_MAX = 3``.
* A refused restart records a synthetic ``report_problem`` with
  ``log_signature="agent.unrestartable"`` and parks the handle in
  ``state="failed"``.

These acceptance tests pin the contract from RFC-002 §"Process
lifecycle" 5: under-cap restarts proceed, the at-cap restart is
refused, and the window slides — i.e., entries older than 5 minutes
no longer count.

The watchdog is exercised directly with a controllable monotonic clock
so the 300-second window arithmetic is observable without sleeping.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from llm_kernel import agent_supervisor as supervisor_mod
from llm_kernel.agent_supervisor import (
    AgentHandle,
    AgentSupervisor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """AgentHandle with the minimum fields the watchdog reads."""
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


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------


def test_restart_allowed_under_3_in_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under-cap (1st and 2nd) crashes MUST be restarted, not refused.

    Contract: ``_RESTART_WINDOW_MAX = 3`` defines the at-cap boundary;
    while ``len(_restart_history) < 3`` the watchdog calls
    ``_respawn_in_place`` and appends a fresh timestamp.
    """
    # Pin the canonical 300s / 3-attempt window so a future RFC bump
    # triggers the contract test rather than silently shipping.
    assert supervisor_mod._RESTART_WINDOW_SEC == 300.0
    assert supervisor_mod._RESTART_WINDOW_MAX == 3

    sup = _make_supervisor()
    handle = _fake_handle()
    clock = [1000.0]
    monkeypatch.setattr(supervisor_mod.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(supervisor_mod.time, "sleep", lambda s: None)

    respawn_calls: List[Any] = []

    def fake_respawn(h: Any, t: Any) -> None:
        respawn_calls.append((h, t))
        # Stop the watchdog's recursion tail.
        h._stop_event.set()

    monkeypatch.setattr(sup, "_respawn_in_place", fake_respawn)

    # Pre-populate two recent restarts (1st and 2nd attempt).
    handle._restart_history.append(1000.0)
    handle._restart_history.append(1001.0)
    clock[0] = 1002.0

    # Drive the watchdog: under-cap, MUST respawn.
    sup._watchdog(handle, "task")

    assert respawn_calls, (
        "3rd restart attempt was refused while under the cap"
    )
    assert len(handle._restart_history) == 3
    assert handle.state != "failed"


def test_restart_rejected_at_3_in_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At cap (3 entries already in window) the next crash MUST be refused.

    Contract: a 4th attempt synchronously emits a synthetic
    ``report_problem`` with ``log_signature="agent.unrestartable"``,
    sets ``handle.state="failed"``, and does NOT call respawn.
    """
    sup = _make_supervisor()
    handle = _fake_handle()
    clock = [1000.0]
    monkeypatch.setattr(supervisor_mod.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(supervisor_mod.time, "sleep", lambda s: None)

    respawn_calls: List[Any] = []

    def fake_respawn(h: Any, t: Any) -> None:  # pragma: no cover - must NOT run
        respawn_calls.append((h, t))

    monkeypatch.setattr(sup, "_respawn_in_place", fake_respawn)

    problems: List[Any] = []
    real_record = sup._record_synthetic_problem

    def record(*args: Any, **kwargs: Any) -> Any:
        problems.append((args, kwargs))
        return real_record(*args, **kwargs)

    monkeypatch.setattr(sup, "_record_synthetic_problem", record)

    # Saturate the window: three entries within 300s.
    handle._restart_history.append(1000.0)
    handle._restart_history.append(1050.0)
    handle._restart_history.append(1200.0)
    clock[0] = 1250.0  # still inside the 300s window

    sup._watchdog(handle, "task")

    assert handle.state == "failed", (
        f"4th restart should have parked agent in failed; "
        f"state={handle.state!r}"
    )
    assert respawn_calls == [], (
        "4th restart was attempted despite saturated window"
    )
    assert problems, "synthetic report_problem was not emitted at cap"
    args, _ = problems[0]
    # Signature: (agent_id, zone_id, description, log_signature)
    description = args[2]
    log_signature = args[3]
    assert "3 restart attempts in 5 minutes" in description
    assert log_signature == "agent.unrestartable"


def test_restart_window_resets_after_5_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After 300s elapses, old entries are pruned and restart succeeds.

    Contract: the deque is a SLIDING window — a saturated agent can
    restart again once the oldest entry has aged past
    ``_RESTART_WINDOW_SEC``. The pruning happens INSIDE the watchdog
    under ``self._lock``; this test fast-forwards the clock to verify
    the pruning + append behavior.
    """
    sup = _make_supervisor()
    handle = _fake_handle()
    clock = [1000.0]
    monkeypatch.setattr(supervisor_mod.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(supervisor_mod.time, "sleep", lambda s: None)

    # Pre-fill 3 saturating entries clustered at t=1000..1002.
    for offset in range(3):
        handle._restart_history.append(1000.0 + offset)

    respawned: List[Any] = []

    def fake_respawn(h: Any, t: Any) -> None:
        respawned.append((h, t))
        h._stop_event.set()

    monkeypatch.setattr(sup, "_respawn_in_place", fake_respawn)

    # Advance the clock past the 5-minute window (300s + slop).
    clock[0] = 1000.0 + 305.0

    sup._watchdog(handle, "task")

    assert respawned, (
        "restart was refused after the 5-minute window slid past"
    )
    # All three pre-existing entries are older than the cutoff and
    # MUST have been pruned; only the freshly-appended one remains.
    assert len(handle._restart_history) == 1
    assert handle._restart_history[0] == 1305.0
    assert handle.state != "failed"

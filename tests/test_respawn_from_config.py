"""K-AS Round-1 G1 — ``AgentSupervisor.respawn_from_config``.

Contract for K-CM hydrate handler: the supervisor accepts a list of
``config.recoverable.agents[]`` entries (joined with the volatile
half by the caller) and spawns each one. Returns
``{agent_id: status}`` where status is ``'spawned'`` | ``'failed'`` |
``'skipped'``. Idempotent: already-running agent_ids are skipped.

These tests stub :meth:`AgentSupervisor.spawn` so the unit test does
not invoke real subprocess.Popen and remains fast / dependency-free.
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

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


def _stub_handle(agent_id: str, returncode: int | None = None) -> AgentHandle:
    popen = MagicMock()
    popen.returncode = returncode
    popen.poll = MagicMock(return_value=returncode)
    return AgentHandle(
        agent_id=agent_id, zone_id="z1", popen=popen,
        started_at=0.0, work_dir=Path("/tmp/x"),
        stdout_thread=threading.Thread(),
        stderr_thread=threading.Thread(),
    )


def test_respawn_two_valid_entries(tmp_path: Path) -> None:
    """Both well-formed entries MUST be spawned with the resolved fields."""
    sup = _make_supervisor()
    spawn_calls: List[Dict[str, Any]] = []

    def fake_spawn(**kwargs: Any) -> AgentHandle:
        spawn_calls.append(kwargs)
        return _stub_handle(kwargs["agent_id"])

    with patch.object(sup, "spawn", side_effect=fake_spawn):
        results = sup.respawn_from_config([
            {
                "agent_id": "alpha", "zone_id": "z1",
                "tools_allowed": ["notify"],
                "task": "task-a", "work_dir": str(tmp_path / "a"),
                "model": "claude-sonnet-4-5", "api_key": "sk-x",
            },
            {
                "agent_id": "beta", "zone_id": "z2",
                "tools_allowed": ["ask"],
                "task": "task-b", "work_dir": str(tmp_path / "b"),
                "model": "claude-haiku-4-5", "api_key": "sk-y",
            },
        ])
    assert results == {"alpha": "spawned", "beta": "spawned"}
    assert len(spawn_calls) == 2
    assert spawn_calls[0]["agent_id"] == "alpha"
    assert spawn_calls[0]["model"] == "claude-sonnet-4-5"
    assert spawn_calls[0]["task"] == "task-a"
    assert spawn_calls[1]["agent_id"] == "beta"
    assert spawn_calls[1]["model"] == "claude-haiku-4-5"


def test_one_malformed_does_not_block_others(tmp_path: Path) -> None:
    """A malformed entry MUST NOT prevent subsequent entries from spawning."""
    sup = _make_supervisor()

    def fake_spawn(**kwargs: Any) -> AgentHandle:
        return _stub_handle(kwargs["agent_id"])

    with patch.object(sup, "spawn", side_effect=fake_spawn):
        results = sup.respawn_from_config([
            # Missing required 'task' -> 'failed'.
            {
                "agent_id": "alpha", "zone_id": "z1",
                "tools_allowed": ["notify"],
                "work_dir": str(tmp_path / "a"),
            },
            # Well-formed -> 'spawned'.
            {
                "agent_id": "beta", "zone_id": "z2",
                "tools_allowed": ["ask"],
                "task": "task-b", "work_dir": str(tmp_path / "b"),
            },
        ])
    assert results.get("alpha") == "failed"
    assert results.get("beta") == "spawned"


def test_already_running_agent_is_skipped(tmp_path: Path) -> None:
    """Idempotency: an agent_id with a live popen MUST be skipped, not respawned."""
    sup = _make_supervisor()
    # Pre-register a "running" handle with popen.poll() returning None.
    running = _stub_handle("alpha", returncode=None)
    sup._agents["alpha"] = running

    spawn_called = False

    def fake_spawn(**kwargs: Any) -> AgentHandle:
        nonlocal spawn_called
        spawn_called = True
        return _stub_handle(kwargs["agent_id"])

    with patch.object(sup, "spawn", side_effect=fake_spawn):
        results = sup.respawn_from_config([
            {
                "agent_id": "alpha", "zone_id": "z1",
                "tools_allowed": ["notify"],
                "task": "task-a", "work_dir": str(tmp_path / "a"),
            },
        ])
    assert results == {"alpha": "skipped"}
    assert not spawn_called


def test_dead_agent_id_respawns(tmp_path: Path) -> None:
    """An agent_id whose popen has reaped (returncode set) is NOT skipped."""
    sup = _make_supervisor()
    dead = _stub_handle("alpha", returncode=1)
    sup._agents["alpha"] = dead

    new_handles: List[AgentHandle] = []

    def fake_spawn(**kwargs: Any) -> AgentHandle:
        h = _stub_handle(kwargs["agent_id"])
        new_handles.append(h)
        return h

    with patch.object(sup, "spawn", side_effect=fake_spawn):
        results = sup.respawn_from_config([
            {
                "agent_id": "alpha", "zone_id": "z1",
                "tools_allowed": ["notify"],
                "task": "task-a", "work_dir": str(tmp_path / "a"),
            },
        ])
    assert results == {"alpha": "spawned"}
    assert len(new_handles) == 1


def test_missing_agent_id_yields_failed_sentinel(tmp_path: Path) -> None:
    """Entry with no agent_id -> failed status under a sentinel key."""
    sup = _make_supervisor()
    with patch.object(sup, "spawn") as spawn_mock:
        results = sup.respawn_from_config([
            {"zone_id": "z1", "task": "x", "work_dir": str(tmp_path)},
        ])
    assert any(v == "failed" for v in results.values())
    assert spawn_mock.call_count == 0


def test_spawn_raising_records_failed_per_entry(tmp_path: Path) -> None:
    """If spawn() raises, that agent records 'failed'; other entries proceed."""
    sup = _make_supervisor()

    def fake_spawn(**kwargs: Any) -> AgentHandle:
        if kwargs["agent_id"] == "alpha":
            raise RuntimeError("simulated spawn failure")
        return _stub_handle(kwargs["agent_id"])

    with patch.object(sup, "spawn", side_effect=fake_spawn):
        results = sup.respawn_from_config([
            {
                "agent_id": "alpha", "zone_id": "z1",
                "tools_allowed": ["notify"],
                "task": "task-a", "work_dir": str(tmp_path / "a"),
            },
            {
                "agent_id": "beta", "zone_id": "z2",
                "tools_allowed": ["ask"],
                "task": "task-b", "work_dir": str(tmp_path / "b"),
            },
        ])
    assert results["alpha"] == "failed"
    assert results["beta"] == "spawned"

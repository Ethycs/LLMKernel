"""PLAN-S5 §3 step 6 -- operator.action routing for branch / revert / stop.

The ``operator.action`` router in :mod:`llm_kernel.mcp.mcp_server`
dispatches ``agent_branch`` / ``agent_revert`` / ``agent_stop`` to
``AgentSupervisor.fork`` / ``revert`` / ``stop`` and answers K-class
failures with an ``operator.action`` error envelope (``k_code`` +
``cell_id``). Also pins the supervisor's ancestry walk to the agent's
own head and the writer's acceptance of ``True`` handler outcomes
(``record_event`` agent_ref_move after the first snapshot bump).
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from llm_kernel.capture.metadata_writer import MetadataWriter
from llm_kernel.capture.run_tracker import RunTracker
from llm_kernel.mcp.mcp_server import OperatorBridgeServer
from llm_kernel.supervision.agent_supervisor import (
    AgentHandle, AgentOperationError, AgentSupervisor,
)


class _RecordingSupervisor:
    def __init__(self, raises: Optional[Exception] = None) -> None:
        self.calls: List[Tuple[str, Tuple[Any, ...]]] = []
        self.raises = raises

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))
        if self.raises is not None:
            raise self.raises

    def fork(self, source: str, at_turn_id: Optional[str], new_agent_id: str) -> None:
        self._record("fork", source, at_turn_id, new_agent_id)

    def revert(self, agent_id: str, target_turn_id: str) -> None:
        self._record("revert", agent_id, target_turn_id)

    def stop(self, agent_id: str) -> None:
        self._record("stop", agent_id)


def _bridge(supervisor: Any) -> Tuple[OperatorBridgeServer, MagicMock]:
    dispatcher = MagicMock()
    bridge = OperatorBridgeServer(
        agent_id="kernel", zone_id="z1", trace_id=str(uuid.uuid4()),
    )
    bridge.dispatcher = dispatcher
    bridge.agent_supervisor = supervisor
    return bridge, dispatcher


def _act(bridge: OperatorBridgeServer, action_type: str, params: Dict[str, Any]) -> None:
    bridge._route_operator_action({
        "type": "operator.action",
        "payload": {"action_type": action_type, "parameters": params},
    })


def _replies(dispatcher: MagicMock) -> List[Dict[str, Any]]:
    return [c.args[0]["payload"] for c in dispatcher.emit.call_args_list]


def test_branch_revert_stop_reach_supervisor() -> None:
    sup = _RecordingSupervisor()
    bridge, dispatcher = _bridge(sup)
    _act(bridge, "agent_branch", {
        "source_agent": "alpha", "at_turn_id": "t_3", "new_agent_id": "beta",
        "cell_id": "c1",
    })
    _act(bridge, "agent_revert", {"agent_id": "alpha", "target_turn_id": "t_2", "cell_id": "c2"})
    _act(bridge, "agent_stop", {"agent_id": "beta", "cell_id": "c3"})
    assert sup.calls == [
        ("fork", ("alpha", "t_3", "beta")),
        ("revert", ("alpha", "t_2")),
        ("stop", ("beta",)),
    ]
    assert _replies(dispatcher) == []


def test_branch_source_agent_id_alias_and_default_turn() -> None:
    sup = _RecordingSupervisor()
    bridge, _ = _bridge(sup)
    _act(bridge, "agent_branch", {"source_agent_id": "alpha", "new_agent_id": "beta"})
    assert sup.calls == [("fork", ("alpha", None, "beta"))]


def test_supervisor_k_code_becomes_error_envelope() -> None:
    sup = _RecordingSupervisor(raises=AgentOperationError("K22", "not in ancestry"))
    bridge, dispatcher = _bridge(sup)
    _act(bridge, "agent_revert", {"agent_id": "alpha", "target_turn_id": "t_9", "cell_id": "c9"})
    (envelope,) = [c.args[0] for c in dispatcher.emit.call_args_list]
    assert envelope["message_type"] == "operator.action"
    assert envelope["direction"] == "kernel→extension"
    payload = envelope["payload"]
    assert payload["k_code"] == "K22"
    assert payload["status"] == "error"
    assert payload["cell_id"] == "c9"
    assert payload["action_type"] == "agent_revert"


def test_missing_parameters_are_k42_without_supervisor_call() -> None:
    sup = _RecordingSupervisor()
    bridge, dispatcher = _bridge(sup)
    _act(bridge, "agent_stop", {"cell_id": "c1"})
    _act(bridge, "agent_revert", {"agent_id": "alpha", "cell_id": "c2"})
    assert sup.calls == []
    assert [p["k_code"] for p in _replies(dispatcher)] == ["K42", "K42"]


# ---------------------------------------------------------------------------
# Supervisor: ancestry is walked from the agent's own head
# ---------------------------------------------------------------------------


class _AlivePopen:
    pid = 77
    stdin = None
    stdout = None
    stderr = None

    def __init__(self) -> None:
        self.returncode: Optional[int] = None

    def poll(self) -> Optional[int]:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0


def _supervisor_with(*agent_ids: str) -> Tuple[AgentSupervisor, MetadataWriter]:
    tracker = RunTracker(
        trace_id=str(uuid.uuid4()), sink=MagicMock(), agent_id="k", zone_id="z1",
    )
    sup = AgentSupervisor(
        run_tracker=tracker, dispatcher=MagicMock(),
        litellm_endpoint_url="http://127.0.0.1:9999/v1",
    )
    writer = MetadataWriter(autosave_interval_sec=999.0)
    sup.set_metadata_writer(writer)
    for agent_id in agent_ids:
        handle = AgentHandle(
            agent_id=agent_id, zone_id="z1", popen=_AlivePopen(),  # type: ignore[arg-type]
            started_at=0.0, work_dir=None,  # type: ignore[arg-type]
            stdout_thread=threading.Thread(), stderr_thread=threading.Thread(),
        )
        with sup._lock:
            sup._agents[agent_id] = handle
    return sup, writer


def _turn(writer: MetadataWriter, tid: str, agent: str, parent: Optional[str], ts: str) -> None:
    result = writer.submit_intent({"payload": {
        "action_type": "zone_mutate", "intent_kind": "append_turn",
        "parameters": {"id": tid, "agent_id": agent, "role": "agent",
                       "body": tid, "parent_id": parent, "created_at": ts},
        "intent_id": f"seed-{tid}",
    }})
    assert result["applied"], result


def test_revert_rejects_turn_from_another_lineage_k22() -> None:
    sup, writer = _supervisor_with("alpha")
    _turn(writer, "a1", "alpha", None, "2026-11-01T00:00:00.000Z")
    _turn(writer, "g1", "gamma", None, "2026-11-01T00:00:09.000Z")
    with pytest.raises(AgentOperationError) as info:
        sup.revert("alpha", "g1")
    assert info.value.k_code == "K22"


def test_fork_defaults_to_source_head_and_k21_without_head() -> None:
    sup, writer = _supervisor_with("alpha", "idle")
    _turn(writer, "a1", "alpha", None, "2026-11-01T00:00:00.000Z")
    _turn(writer, "a2", "alpha", "a1", "2026-11-01T00:00:01.000Z")
    _turn(writer, "g1", "gamma", None, "2026-11-01T00:00:09.000Z")
    handle = sup.fork("alpha", None, "beta")
    assert handle.last_seen_turn_id == "a2"
    snap = writer.snapshot()
    assert snap["zone"]["agents"]["beta"]["session"]["head_turn_id"] == "a2"
    with pytest.raises(AgentOperationError) as info:
        sup.fork("idle", None, "delta")
    assert info.value.k_code == "K21"


def test_agent_ref_move_event_persists_after_version_bumps() -> None:
    sup, writer = _supervisor_with("alpha")
    _turn(writer, "a1", "alpha", None, "2026-11-01T00:00:00.000Z")
    _turn(writer, "a2", "alpha", "a1", "2026-11-01T00:00:01.000Z")
    sup.revert("alpha", "a1")
    log = writer.snapshot()["zone"].get("event_log") or []
    moves = [e for e in log if e.get("kind") == "agent_ref_move"]
    assert [(m["reason"], m["from_turn_id"], m["to_turn_id"]) for m in moves] == [
        ("operator_revert", "a2", "a1"),
    ]

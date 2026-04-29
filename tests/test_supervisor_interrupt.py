"""K-AS-A BSP-005 S9 -- AgentSupervisor.interrupt + dispatcher routing.

Per ``docs/atoms/contracts/agent-supervisor.md`` and
``docs/atoms/operations/stop-agent.md``: the kernel-side half of the
interrupt button (X-EXT commit 5de3401) is

* :meth:`AgentSupervisor.interrupt` -- send SIGINT to the live PID.
* the ``operator.action`` dispatcher in :mod:`llm_kernel.mcp_server`
  routes ``{action_type: "agent_interrupt", agent_id: ...}`` to it.

Tests are sync, dependency-free, and patch ``os.kill`` so no real
SIGINT crosses the process boundary.
"""

from __future__ import annotations

import signal
import threading
import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from llm_kernel.agent_supervisor import AgentHandle, AgentSupervisor


class _FakePopen:
    """Minimal Popen substitute exposing pid + poll() for the interrupt path."""

    def __init__(self, pid: int = 4242, returncode: Optional[int] = None) -> None:
        self.pid = pid
        self.returncode = returncode
        self.stdin = None
        self.stdout = None
        self.stderr = None

    def poll(self) -> Optional[int]:
        return self.returncode


def _make_supervisor() -> AgentSupervisor:
    """Build a supervisor with stub run-tracker + dispatcher (no real kernel)."""
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


def _seed_handle(
    sup: AgentSupervisor, agent_id: str, *, pid: int = 4242,
    returncode: Optional[int] = None,
) -> AgentHandle:
    """Insert a fake handle into the supervisor's registry."""
    fake = _FakePopen(pid=pid, returncode=returncode)
    handle = AgentHandle(
        agent_id=agent_id, zone_id="z1", popen=fake,  # type: ignore[arg-type]
        started_at=0.0, work_dir=None,  # type: ignore[arg-type]
        stdout_thread=threading.Thread(),
        stderr_thread=threading.Thread(),
    )
    handle.state = "running"
    with sup._lock:
        sup._agents[agent_id] = handle
    return handle


# ---------------------------------------------------------------------------
# AgentSupervisor.interrupt -- direct method tests
# ---------------------------------------------------------------------------


def test_interrupt_running_agent_sends_sigint() -> None:
    """A live agent receives SIGINT via os.kill(pid, signal.SIGINT)."""
    sup = _make_supervisor()
    _seed_handle(sup, "alpha", pid=4242, returncode=None)
    with patch("llm_kernel.agent_supervisor.os.kill") as fake_kill:
        result = sup.interrupt("alpha")
    fake_kill.assert_called_once_with(4242, signal.SIGINT)
    assert result == {"agent_id": "alpha", "status": "interrupted"}


def test_interrupt_idle_agent_returns_not_running() -> None:
    """A registered agent whose process has exited yields ``not_running``.

    Per ``atoms/operations/stop-agent.md``: idle / exited agents have
    ``runtime_status: idle``; SIGINT to a stale or recycled PID would
    misfire so the supervisor refuses.
    """
    sup = _make_supervisor()
    _seed_handle(sup, "alpha", pid=4242, returncode=0)
    with patch("llm_kernel.agent_supervisor.os.kill") as fake_kill:
        result = sup.interrupt("alpha")
    fake_kill.assert_not_called()
    assert result == {"agent_id": "alpha", "status": "not_running"}


def test_interrupt_unknown_agent_returns_unknown() -> None:
    """An agent_id with no spawn record yields ``unknown``."""
    sup = _make_supervisor()
    with patch("llm_kernel.agent_supervisor.os.kill") as fake_kill:
        result = sup.interrupt("ghost")
    fake_kill.assert_not_called()
    assert result == {"agent_id": "ghost", "status": "unknown"}


def test_interrupt_pid_gone_downgrades_to_not_running() -> None:
    """Race: ``os.kill`` raises ProcessLookupError -> status not_running.

    The agent reaped between ``poll()`` and ``os.kill()``; the V1 API
    treats this as ``not_running`` rather than raising so the operator
    surface still gets a deterministic response shape.
    """
    sup = _make_supervisor()
    _seed_handle(sup, "alpha", pid=4242, returncode=None)
    with patch(
        "llm_kernel.agent_supervisor.os.kill",
        side_effect=ProcessLookupError(),
    ):
        result = sup.interrupt("alpha")
    assert result == {"agent_id": "alpha", "status": "not_running"}


# ---------------------------------------------------------------------------
# Dispatcher routing -- envelope from the wire arrives and reaches supervisor
# ---------------------------------------------------------------------------


class _RecordingSupervisor:
    """Stand-in supervisor that captures the interrupt(...) call."""

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.return_value: Dict[str, Any] = {
            "agent_id": "alpha", "status": "interrupted",
        }

    def interrupt(self, agent_id: str) -> Dict[str, Any]:
        self.calls.append(agent_id)
        return self.return_value


def _make_dispatcher_target(supervisor: Optional[Any]):
    """Build a minimal :class:`OperatorBridgeServer` routing harness.

    The full server pulls in the asyncio MCP machinery; we exercise just
    ``_route_operator_action`` by binding it to a tiny fake instance so
    the test stays sync and dependency-free. We override
    ``_resolve_agent_supervisor`` to return our recording stand-in.
    """
    from llm_kernel.mcp_server import OperatorBridgeServer

    class _Stub(OperatorBridgeServer):  # type: ignore[misc]
        def __init__(self) -> None:  # noqa: D401 -- bypass parent __init__
            self.agent_id = "alpha"
            self.zone_id = "z1"
            self._pending_responses = {}
            self._pending_lock = threading.Lock()
            self.run_tracker = None
            self._supervisor = supervisor
            self._writer = None
            self.metadata_writer = None
            self.kernel = None

        def _resolve_agent_supervisor(self):
            return self._supervisor

        def _resolve_metadata_writer(self):
            return self._writer

    return _Stub()


def test_dispatcher_routes_agent_interrupt_envelope_to_supervisor() -> None:
    """An ``agent_interrupt`` envelope reaches ``supervisor.interrupt(agent_id)``."""
    sup = _RecordingSupervisor()
    server = _make_dispatcher_target(sup)
    envelope = {
        "type": "operator.action",
        "payload": {
            "action_type": "agent_interrupt",
            "parameters": {"agent_id": "alpha", "cell_id": "vscode-notebook-cell:/x.llmnb#0"},
        },
    }
    server._route_operator_action(envelope)
    assert sup.calls == ["alpha"]


def test_interrupt_malformed_envelope_returns_kclass_error() -> None:
    """Empty/non-string ``agent_id`` is K42-rejected without calling interrupt."""
    sup = _RecordingSupervisor()
    server = _make_dispatcher_target(sup)
    # Empty string.
    server._route_operator_action({
        "type": "operator.action",
        "payload": {
            "action_type": "agent_interrupt",
            "parameters": {"agent_id": ""},
        },
    })
    # Missing key entirely.
    server._route_operator_action({
        "type": "operator.action",
        "payload": {
            "action_type": "agent_interrupt",
            "parameters": {},
        },
    })
    # Non-string.
    server._route_operator_action({
        "type": "operator.action",
        "payload": {
            "action_type": "agent_interrupt",
            "parameters": {"agent_id": 123},
        },
    })
    assert sup.calls == []

"""K-AS S3 — ``AgentSupervisor.send_user_turn`` and the dispatcher route.

Contract for BSP-005 S3 (multi-turn cells via ``@<agent>``). The
supervisor accepts an operator's continuation message for an existing
agent and writes one stream-json user-turn line to the agent's stdin.

Per atoms/operations/continue-turn.md and atoms/contracts/agent-supervisor.md:

* alive agent  -> single stdin write, return ``status: "sent"``.
* idle/exited  -> resume via S2 ``resume_claude_session_id`` plumbing,
                  then stdin write; return ``status: "resumed_then_sent"``.
* unknown id   -> KeyError (dispatcher translates to K23).
* empty text   -> ValueError (dispatcher translates to K42).

These tests stub :class:`subprocess.Popen` so the unit suite does not
spawn a real claude process; only the wire-side write is asserted.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_kernel.agent_supervisor import AgentHandle, AgentSupervisor


# ---------------------------------------------------------------------------
# Fixture / helper plumbing — mirrors test_agent_supervisor.py's pattern.
# ---------------------------------------------------------------------------


class _StubStdin:
    """Captures every write/flush so tests can assert one JSON line landed."""

    def __init__(self) -> None:
        self.writes: List[str] = []
        self.flushed: int = 0

    def write(self, data: str) -> int:
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:  # pragma: no cover - exercised only at teardown
        pass


class _FakePopen:
    """Minimal Popen substitute with a writable stdin and seekable iterators."""

    def __init__(
        self,
        stdout_lines: List[str] | None = None,
        stderr_lines: List[str] | None = None,
        exit_code: int = 0,
    ) -> None:
        self.stdout = iter(list(stdout_lines or []) + [""])
        self.stderr = iter(list(stderr_lines or []) + [""])
        self.stdin = _StubStdin()
        self._exit_code = exit_code
        self._exited = threading.Event()
        self.returncode: Optional[int] = None
        self.pid = 13579

    def poll(self) -> Optional[int]:
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        self._exited.wait(timeout=timeout if timeout is not None else 0.5)
        if self.returncode is None:
            self.returncode = self._exit_code
        return self.returncode

    def terminate(self) -> None:
        self.returncode = self.returncode or 0
        self._exited.set()

    def kill(self) -> None:  # pragma: no cover - not exercised
        self.returncode = self.returncode or -9
        self._exited.set()


def _patch_health(status_code: int = 200) -> Any:
    fake = MagicMock()
    fake.status_code = status_code
    return patch("llm_kernel._provisioning.httpx.head", return_value=fake)


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


def _stub_handle(
    agent_id: str,
    *,
    returncode: Optional[int] = None,
    claude_session_id: str = "",
    work_dir: Path = Path("/tmp/x"),
) -> AgentHandle:
    """Build an AgentHandle whose popen is dead/alive per ``returncode``.

    ``returncode is None`` simulates an alive agent; an int simulates an
    idle/exited agent whose poll() returns the exit code.
    """
    popen = MagicMock()
    popen.returncode = returncode
    popen.poll = MagicMock(return_value=returncode)
    popen.stdin = _StubStdin()
    return AgentHandle(
        agent_id=agent_id, zone_id="z1", popen=popen,
        started_at=0.0, work_dir=work_dir,
        stdout_thread=threading.Thread(),
        stderr_thread=threading.Thread(),
        claude_session_id=claude_session_id,
    )


# ---------------------------------------------------------------------------
# 1. send_user_turn writes a single JSON line to an alive agent's stdin.
# ---------------------------------------------------------------------------


def test_send_user_turn_writes_to_alive_agent_stdin(tmp_path: Path) -> None:
    """An alive agent receives one ``{"type":"user",...}`` line on stdin."""
    sup = _make_supervisor()
    fake = _FakePopen(stdout_lines=[])
    fake.returncode = None  # alive
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        sup.spawn(
            zone_id="z1", agent_id="alpha", task="seed",
            work_dir=tmp_path, api_key="sk-x",
        )

    result = sup.send_user_turn(
        agent_id="alpha",
        text="now optimize for read performance",
        cell_id="vscode-notebook-cell:test#c1",
    )
    assert result["agent_id"] == "alpha"
    assert result["status"] == "sent"
    assert result["cell_id"] == "vscode-notebook-cell:test#c1"
    # Exactly one line written, valid JSON of the expected shape.
    assert len(fake.stdin.writes) == 1, fake.stdin.writes
    line = fake.stdin.writes[0]
    assert line.endswith("\n")
    parsed = json.loads(line)
    assert parsed["type"] == "user"
    assert parsed["message"]["role"] == "user"
    assert parsed["message"]["content"] == "now optimize for read performance"
    assert fake.stdin.flushed == 1
    fake._exited.set()
    sup.terminate_all(grace_seconds=0.05)


# ---------------------------------------------------------------------------
# 2. Idle agent: send_user_turn resumes via S2 plumbing, then writes stdin.
# ---------------------------------------------------------------------------


def test_send_user_turn_resumes_idle_agent_then_sends(tmp_path: Path) -> None:
    """An idle (popen reaped, session_id preserved) agent triggers --resume."""
    sup = _make_supervisor()
    # Pre-seed an idle handle: popen.poll() returns 0; claude_session_id set.
    idle = _stub_handle(
        "alpha", returncode=0,
        claude_session_id="9d4f-prior-session-uuid",
        work_dir=tmp_path / "alpha",
    )
    sup._agents["alpha"] = idle

    spawn_calls: List[Dict[str, Any]] = []
    new_fake = _FakePopen()
    new_fake.returncode = None  # resume succeeds (alive)

    def fake_spawn(**kwargs: Any) -> AgentHandle:
        spawn_calls.append(dict(kwargs))
        # Return a fresh stub handle so send_user_turn writes to a
        # different stdin than the original idle handle.
        h = _stub_handle(
            kwargs["agent_id"],
            returncode=None,
            claude_session_id=kwargs.get("resume_claude_session_id", "")
            or "fresh-uuid",
            work_dir=kwargs["work_dir"],
        )
        sup._agents[kwargs["agent_id"]] = h
        return h

    with patch.object(sup, "spawn", side_effect=fake_spawn):
        result = sup.send_user_turn(
            agent_id="alpha",
            text="continue with new criteria",
            cell_id="vscode-notebook-cell:test#c2",
        )

    assert result["status"] == "resumed_then_sent"
    assert len(spawn_calls) == 1
    # The S2 plumbing MUST be invoked: resume_claude_session_id is the
    # idle handle's preserved claude_session_id.
    assert spawn_calls[0]["resume_claude_session_id"] == "9d4f-prior-session-uuid"
    # The stdin write happened on the freshly-resumed handle.
    new_handle = sup._agents["alpha"]
    assert len(new_handle.popen.stdin.writes) == 1
    parsed = json.loads(new_handle.popen.stdin.writes[0])
    assert parsed["type"] == "user"
    assert parsed["message"]["content"] == "continue with new criteria"


# ---------------------------------------------------------------------------
# 3. Unknown agent_id raises KeyError (dispatcher -> K23).
# ---------------------------------------------------------------------------


def test_send_user_turn_unknown_agent_raises_keyerror() -> None:
    """An unknown agent_id MUST raise :class:`KeyError`."""
    sup = _make_supervisor()
    with pytest.raises(KeyError):
        sup.send_user_turn(agent_id="ghost", text="hello", cell_id=None)


# ---------------------------------------------------------------------------
# 4. Empty / whitespace-only text rejected with ValueError (dispatcher -> K42).
# ---------------------------------------------------------------------------


def test_send_user_turn_empty_text_rejected(tmp_path: Path) -> None:
    """Empty or whitespace-only text MUST raise :class:`ValueError`."""
    sup = _make_supervisor()
    sup._agents["alpha"] = _stub_handle(
        "alpha", returncode=None, claude_session_id="s1",
    )

    for bad in ("", "   ", "\n\t  "):
        with pytest.raises(ValueError):
            sup.send_user_turn(agent_id="alpha", text=bad)
    # Non-string also rejected.
    with pytest.raises(ValueError):
        sup.send_user_turn(agent_id="alpha", text=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. Dispatcher route: operator.action {action_type:"agent_continue",
#    intent_kind:"send_user_turn"} -> supervisor.send_user_turn(...).
# ---------------------------------------------------------------------------


def test_dispatcher_routes_send_user_turn_intent_to_supervisor() -> None:
    """The full envelope path MUST land at supervisor.send_user_turn."""
    from llm_kernel.mcp_server import OperatorBridgeServer
    from llm_kernel.run_tracker import RunTracker

    class _ListSink:
        def __init__(self) -> None:
            self.envelopes: List[Dict[str, Any]] = []

        def emit(self, env: Dict[str, Any]) -> None:
            self.envelopes.append(env)

    tracker = RunTracker(
        trace_id=str(uuid.uuid4()), sink=_ListSink(),
        agent_id="kernel-agent", zone_id="z1",
    )
    bridge = OperatorBridgeServer(
        agent_id="kernel-agent", zone_id="z1",
        trace_id=str(uuid.uuid4()), run_tracker=tracker,
    )
    fake_supervisor = MagicMock()
    fake_supervisor.send_user_turn = MagicMock(return_value={
        "agent_id": "alpha", "status": "sent",
        "cell_id": "vscode-notebook-cell:test#c1",
    })
    bridge.metadata_writer = None  # not exercised here
    # Inject the supervisor via the documented attribute slot used by
    # ``_resolve_agent_supervisor``.

    class _FakeKernel:
        _llmnb_agent_supervisor = fake_supervisor
        _llmnb_metadata_writer = None
        shell = None

    bridge.kernel = _FakeKernel()  # type: ignore[assignment]

    envelope = {
        "type": "operator.action",
        "payload": {
            "action_type": "agent_continue",
            "intent_kind": "send_user_turn",
            "parameters": {
                "agent_id": "alpha",
                "text": "now optimize for read performance",
                "cell_id": "vscode-notebook-cell:test#c1",
            },
            "originating_cell_id": "vscode-notebook-cell:test#c1",
        },
    }
    bridge._route_operator_action(envelope)
    fake_supervisor.send_user_turn.assert_called_once()
    kwargs = fake_supervisor.send_user_turn.call_args.kwargs
    assert kwargs["agent_id"] == "alpha"
    assert kwargs["text"] == "now optimize for read performance"
    assert kwargs["cell_id"] == "vscode-notebook-cell:test#c1"

    # K23 path: unknown agent_id -> KeyError -> dispatcher logs and drops.
    fake_supervisor.send_user_turn = MagicMock(side_effect=KeyError("ghost"))
    bridge._route_operator_action({
        "type": "operator.action",
        "payload": {
            "action_type": "agent_continue",
            "intent_kind": "send_user_turn",
            "parameters": {"agent_id": "ghost", "text": "hello"},
        },
    })
    fake_supervisor.send_user_turn.assert_called_once()

    # K42 path: empty text -> ValueError -> dispatcher logs and drops.
    fake_supervisor.send_user_turn = MagicMock(side_effect=ValueError("empty"))
    bridge._route_operator_action({
        "type": "operator.action",
        "payload": {
            "action_type": "agent_continue",
            "intent_kind": "send_user_turn",
            "parameters": {"agent_id": "alpha", "text": "   "},
        },
    })
    fake_supervisor.send_user_turn.assert_called_once()

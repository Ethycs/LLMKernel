"""BSP-002 §4 / §4.3 — persistent claude lifecycle, Phase 2 (`--resume`).

Phase 2 scope (this test file): Plumb ``claude --resume <session_id>``
through ``AgentSupervisor.respawn_from_config`` so an idle agent's
conversation actually picks up where it left off.

Covered behaviors:

* ``_provisioning.build_argv`` accepts a ``resume_session_id`` kwarg
  that emits ``--resume <id>`` (mutually exclusive with ``--session-id``
  per the claude CLI grammar).
* ``AgentSupervisor.spawn`` accepts ``resume_claude_session_id`` and
  threads it through ``build_argv`` without minting a new UUID.
* ``AgentSupervisor.respawn_from_config`` selects the resume branch
  whenever the entry has a ``claude_session_id`` AND a recoverable
  ``runtime_status`` (``alive`` / ``idle`` / ``exited``); per
  ``decisions/no-rebind-popen``, an ``alive`` snapshot is treated as
  ``idle`` because the prior PID is volatile.
* On resume failure (claude exits non-zero — e.g. session cache
  expired), the supervisor emits the K24 marker per BSP-002 §7 and
  retries as a fresh spawn.

Out of scope (later phases):

* Stdin-based continuation via ``--input-format=stream-json`` (§4.2).
* Full transcript replay rebuild after K24 (BSP-002 §4.4 Case B is
  queued for a later slice; V1 V1's K24 fallback is just a fresh spawn).
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from llm_kernel._provisioning import build_argv
from llm_kernel.agent_supervisor import AgentHandle, AgentSupervisor


class _FakePopen:
    """Minimal ``subprocess.Popen`` substitute (mirrors the shape used
    by ``test_persistent_claude.py`` / ``test_agent_supervisor.py``).
    """

    def __init__(
        self,
        stdout_lines: Iterable[str] = (),
        stderr_lines: Iterable[str] = (),
        exit_code: int = 0,
    ) -> None:
        self.stdout = iter(list(stdout_lines) + [""])
        self.stderr = iter(list(stderr_lines) + [""])
        self._exit_code = exit_code
        self._exited = threading.Event()
        self.returncode: Optional[int] = None
        self.pid = 12345

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

    def kill(self) -> None:  # pragma: no cover
        self.returncode = self.returncode or -9
        self._exited.set()


def _patch_health(status_code: int = 200):
    fake = MagicMock(); fake.status_code = status_code
    return patch("llm_kernel._provisioning.httpx.head", return_value=fake)


def _make_supervisor() -> AgentSupervisor:
    """Stub run-tracker + dispatcher (no real kernel)."""
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
    claude_session_id: str = "",
    returncode: Optional[int] = None,
) -> AgentHandle:
    popen = MagicMock()
    popen.returncode = returncode
    popen.poll = MagicMock(return_value=returncode)
    return AgentHandle(
        agent_id=agent_id, zone_id="z1", popen=popen,
        started_at=0.0, work_dir=Path("/tmp/x"),
        stdout_thread=threading.Thread(),
        stderr_thread=threading.Thread(),
        claude_session_id=claude_session_id,
    )


# -- S2.1: build_argv -------------------------------------------------


def test_build_argv_emits_resume_when_session_id_provided(tmp_path: Path) -> None:
    """``resume_session_id`` -> ``--resume <id>``; no ``--session-id`` flag."""
    sp = tmp_path / "system-prompt.txt"; sp.write_text("x")
    mc = tmp_path / "mcp-config.json"; mc.write_text("{}")

    argv = build_argv(
        sp, mc, "task body",
        resume_session_id="9d4f-abcd-1234",
    )

    assert "--resume" in argv, f"--resume missing from argv: {argv}"
    idx = argv.index("--resume")
    assert idx + 1 < len(argv)
    assert argv[idx + 1] == "9d4f-abcd-1234"
    assert "--session-id" not in argv, (
        f"--session-id leaked into resume argv: {argv}"
    )


def test_build_argv_session_id_and_resume_mutually_exclusive(
    tmp_path: Path,
) -> None:
    """Passing BOTH ``session_id`` and ``resume_session_id`` raises ValueError.

    The claude CLI rejects the combination; the docstring on
    ``build_argv`` documents this. Catching it at the kernel boundary
    rather than letting claude exit with a confusing message is the
    point of the guard.
    """
    sp = tmp_path / "system-prompt.txt"; sp.write_text("x")
    mc = tmp_path / "mcp-config.json"; mc.write_text("{}")

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_argv(
            sp, mc, "task body",
            session_id="aaaa-1111",
            resume_session_id="bbbb-2222",
        )


# -- S2.2: spawn resume branch ---------------------------------------


def test_spawn_resume_branch_skips_uuid_mint(tmp_path: Path) -> None:
    """``resume_claude_session_id`` -> handle's ``claude_session_id`` is
    the supplied id; argv carries ``--resume`` (NOT ``--session-id``).
    """
    sup = _make_supervisor()
    fake = _FakePopen()
    fake.returncode = 0  # exit promptly so threads don't linger
    captured: Dict[str, Any] = {}

    def _capture(argv, **kwargs):  # type: ignore[no-untyped-def]
        captured["argv"] = list(argv)
        return fake

    resumed_id = "11111111-2222-3333-4444-555555555555"
    with _patch_health(), \
            patch("subprocess.Popen", side_effect=_capture), \
            patch("uuid.uuid4") as uuid4_mock:
        uuid4_mock.side_effect = AssertionError(
            "spawn(resume_claude_session_id=...) MUST NOT mint a new UUID"
        )
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="continue",
            work_dir=tmp_path, api_key="sk-x",
            resume_claude_session_id=resumed_id,
        )

    assert handle.claude_session_id == resumed_id
    argv = captured.get("argv", [])
    assert "--resume" in argv, f"--resume missing from argv: {argv}"
    assert argv[argv.index("--resume") + 1] == resumed_id
    assert "--session-id" not in argv, (
        f"--session-id leaked into resume argv: {argv}"
    )
    handle.terminate()


# -- S2.3: respawn_from_config threads claude_session_id -------------


def _resumable_entry(
    *, agent_id: str, runtime_status: str, session_id: Optional[str],
    work_dir: Path, task: str = "task-x",
) -> Dict[str, Any]:
    """Build a recoverable+volatile config entry shape for tests."""
    entry: Dict[str, Any] = {
        "agent_id": agent_id, "zone_id": "z1",
        "tools_allowed": ["notify"],
        "task": task, "work_dir": str(work_dir),
    }
    if session_id is not None:
        entry["claude_session_id"] = session_id
    if runtime_status is not None:
        entry["runtime_status"] = runtime_status
    return entry


def test_respawn_from_config_uses_resume_for_idle_entry(tmp_path: Path) -> None:
    """An ``idle`` entry with a stored ``claude_session_id`` MUST
    forward it to ``spawn(..., resume_claude_session_id=...)``.
    """
    sup = _make_supervisor()
    spawn_calls: List[Dict[str, Any]] = []

    def fake_spawn(**kwargs: Any) -> AgentHandle:
        spawn_calls.append(kwargs)
        return _stub_handle(
            kwargs["agent_id"],
            claude_session_id=kwargs.get("resume_claude_session_id") or "",
        )

    with patch.object(sup, "spawn", side_effect=fake_spawn):
        results = sup.respawn_from_config([
            _resumable_entry(
                agent_id="alpha", runtime_status="idle",
                session_id="sess-idle-123",
                work_dir=tmp_path / "alpha",
            ),
        ])

    assert results == {"alpha": "spawned"}
    assert len(spawn_calls) == 1
    assert spawn_calls[0]["resume_claude_session_id"] == "sess-idle-123"


def test_respawn_from_config_uses_resume_for_exited_entry(tmp_path: Path) -> None:
    """An ``exited`` entry with a session id also routes through resume.

    Per ``concepts/agent.md`` ``runtime_status`` semantics, ``exited``
    means "process exited and cannot be resumed." The supervisor still
    attempts ``--resume`` first; the K24 fallback path
    (``test_resume_failure_emits_k24_marker_and_falls_back_to_fresh``)
    handles the case where claude rejects the session.
    """
    sup = _make_supervisor()
    spawn_calls: List[Dict[str, Any]] = []

    def fake_spawn(**kwargs: Any) -> AgentHandle:
        spawn_calls.append(kwargs)
        return _stub_handle(
            kwargs["agent_id"],
            claude_session_id=kwargs.get("resume_claude_session_id") or "",
        )

    with patch.object(sup, "spawn", side_effect=fake_spawn):
        results = sup.respawn_from_config([
            _resumable_entry(
                agent_id="alpha", runtime_status="exited",
                session_id="sess-exited-456",
                work_dir=tmp_path / "alpha",
            ),
        ])

    assert results == {"alpha": "spawned"}
    assert spawn_calls[0]["resume_claude_session_id"] == "sess-exited-456"


def test_respawn_from_config_treats_alive_snapshot_as_idle(
    tmp_path: Path,
) -> None:
    """Snapshot ``runtime_status: "alive"`` is treated as ``idle`` per
    ``decisions/no-rebind-popen`` -- the prior PID is gone, the session
    is durable, so ``--resume`` is the right reattach path.
    """
    sup = _make_supervisor()
    spawn_calls: List[Dict[str, Any]] = []

    def fake_spawn(**kwargs: Any) -> AgentHandle:
        spawn_calls.append(kwargs)
        return _stub_handle(
            kwargs["agent_id"],
            claude_session_id=kwargs.get("resume_claude_session_id") or "",
        )

    with patch.object(sup, "spawn", side_effect=fake_spawn):
        results = sup.respawn_from_config([
            _resumable_entry(
                agent_id="alpha", runtime_status="alive",
                session_id="sess-alive-789",
                work_dir=tmp_path / "alpha",
            ),
        ])

    assert results == {"alpha": "spawned"}
    assert spawn_calls[0]["resume_claude_session_id"] == "sess-alive-789"


def test_respawn_from_config_falls_back_fresh_when_no_session_id(
    tmp_path: Path,
) -> None:
    """No ``claude_session_id`` -> spawn called WITHOUT
    ``resume_claude_session_id`` (current Phase 1 behavior).
    """
    sup = _make_supervisor()
    spawn_calls: List[Dict[str, Any]] = []

    def fake_spawn(**kwargs: Any) -> AgentHandle:
        spawn_calls.append(kwargs)
        return _stub_handle(kwargs["agent_id"])

    with patch.object(sup, "spawn", side_effect=fake_spawn):
        results = sup.respawn_from_config([
            _resumable_entry(
                agent_id="alpha", runtime_status="idle",
                session_id=None,  # no session id stored
                work_dir=tmp_path / "alpha",
            ),
        ])

    assert results == {"alpha": "spawned"}
    assert len(spawn_calls) == 1
    # Critical: resume kwarg must be absent (None or unset) so spawn
    # mints a fresh UUID instead of attempting --resume.
    assert spawn_calls[0].get("resume_claude_session_id") in (None, "")


# -- S2.4: K24 fallback path -----------------------------------------


def test_resume_failure_emits_k24_marker_and_falls_back_to_fresh(
    tmp_path: Path,
) -> None:
    """``claude --resume <id>`` exiting non-zero -> K24 ``report_problem``
    is recorded AND a fresh spawn is attempted (without the resume kwarg).

    The first ``spawn(resume_claude_session_id=...)`` call returns a
    handle whose popen has already exited 1; the supervisor must
    detect that, emit K24, and retry with a fresh-spawn invocation.
    """
    sup = _make_supervisor()
    # Tighten the verify timeout so the test runs in milliseconds.
    sup._RESUME_VERIFY_TIMEOUT_SEC = 0.1  # type: ignore[attr-defined]

    spawn_calls: List[Dict[str, Any]] = []

    def fake_spawn(**kwargs: Any) -> AgentHandle:
        spawn_calls.append(kwargs)
        if "resume_claude_session_id" in kwargs and kwargs["resume_claude_session_id"]:
            # Simulate claude rejecting the resume (session expired) —
            # popen has already exited with code 1.
            handle = _stub_handle(
                kwargs["agent_id"],
                claude_session_id=kwargs["resume_claude_session_id"],
                returncode=1,
            )
        else:
            # Fresh-spawn fallback: handle is alive (returncode=None).
            handle = _stub_handle(
                kwargs["agent_id"],
                claude_session_id=str(uuid.uuid4()),
                returncode=None,
            )
        return handle

    problem_calls: List[Dict[str, Any]] = []

    def fake_report_problem(
        agent_id: str, zone_id: str, description: str, log_signature: str,
    ) -> None:
        problem_calls.append({
            "agent_id": agent_id, "zone_id": zone_id,
            "description": description, "log_signature": log_signature,
        })

    marker_calls: List[Dict[str, Any]] = []

    def fake_mark(stage: str, **kw: Any) -> None:
        marker_calls.append({"stage": stage, **kw})

    with patch.object(sup, "spawn", side_effect=fake_spawn), \
            patch.object(sup, "_record_synthetic_problem",
                         side_effect=fake_report_problem), \
            patch("llm_kernel._diagnostics.mark", side_effect=fake_mark):
        results = sup.respawn_from_config([
            _resumable_entry(
                agent_id="alpha", runtime_status="idle",
                session_id="sess-stale-999",
                work_dir=tmp_path / "alpha",
            ),
        ])

    # Status reports as "spawned" — the fresh-spawn retry succeeded.
    assert results == {"alpha": "spawned"}
    # Two spawn calls: one resume (which "failed"), one fresh fallback.
    assert len(spawn_calls) == 2
    assert spawn_calls[0].get("resume_claude_session_id") == "sess-stale-999"
    assert spawn_calls[1].get("resume_claude_session_id") in (None, "")
    # The K24 marker landed on both surfaces.
    assert any(
        p["log_signature"] == "agent.resume_failed.k24"
        and "sess-stale-999" in p["description"]
        for p in problem_calls
    ), f"K24 report_problem missing; saw: {problem_calls}"
    assert any(
        m["stage"] == "supervisor_resume_failed_k24"
        and m.get("attempted_session_id") == "sess-stale-999"
        for m in marker_calls
    ), f"K24 diagnostic marker missing; saw: {marker_calls}"

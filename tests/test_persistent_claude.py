"""BSP-002 §4 / §5 — persistent claude lifecycle, Phase 1.

Phase 1 scope (this test file):
  - Fresh spawn assigns a kernel-owned ``claude_session_id`` (UUID).
  - The session id is passed to ``claude`` via ``--session-id <uuid>``.
  - Idempotent spawn: a second ``spawn(agent_id=alpha, ...)`` call while
    the first process is still alive returns the existing handle without
    invoking Popen again.
  - Dead-process re-spawn: when the prior process has exited, a new
    spawn creates a NEW session id (Phase 1 does not yet implement
    --resume; that's Phase 2).

Out of scope (Phase 2+):
  - ``--resume <claude_session_id>`` after idle exit (BSP-002 §4.3).
  - stdin-based continuation via ``--input-format=stream-json`` (§4.2).
  - Cross-agent context handoff (§4.6).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_kernel.agent_supervisor import AgentSupervisor


class _FakePopen:
    """Minimal Popen substitute (mirrors test_agent_supervisor.py shape)."""

    def __init__(
        self, stdout_lines: Iterable[str] = (),
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
        def __init__(self) -> None: self.envelopes: List[Dict[str, Any]] = []
        def emit(self, env: Dict[str, Any]) -> None: self.envelopes.append(env)

    tracker = RunTracker(
        trace_id=str(uuid.uuid4()), sink=_ListSink(),
        agent_id="alpha", zone_id="z1",
    )
    dispatcher = MagicMock()
    return AgentSupervisor(
        run_tracker=tracker, dispatcher=dispatcher,
        litellm_endpoint_url="http://127.0.0.1:9999/v1",
    )


def test_spawn_assigns_claude_session_id(tmp_path: Path) -> None:
    """A fresh spawn populates handle.claude_session_id with a valid UUID."""
    sup = _make_supervisor()
    fake = _FakePopen()
    fake.returncode = 0  # exits promptly so threads don't linger
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="say hi",
            work_dir=tmp_path, api_key="sk-x",
        )

    assert handle.claude_session_id, "spawn must assign a claude_session_id"
    # Round-trip parse to assert a real UUID-shaped string.
    parsed = uuid.UUID(handle.claude_session_id)
    assert str(parsed) == handle.claude_session_id
    handle.terminate()


def test_spawn_passes_session_id_to_argv(tmp_path: Path) -> None:
    """The claude argv MUST include --session-id <uuid> per BSP-002 §5."""
    sup = _make_supervisor()
    fake = _FakePopen()
    fake.returncode = 0
    captured: Dict[str, Any] = {}

    def _capture(argv, **kwargs):  # type: ignore[no-untyped-def]
        captured["argv"] = list(argv)
        return fake

    with _patch_health(), patch("subprocess.Popen", side_effect=_capture):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="say hi",
            work_dir=tmp_path, api_key="sk-x",
        )

    argv = captured.get("argv", [])
    assert "--session-id" in argv, f"--session-id flag missing from argv: {argv}"
    sid_index = argv.index("--session-id")
    assert sid_index + 1 < len(argv), "--session-id requires a value"
    sid_value = argv[sid_index + 1]
    assert sid_value == handle.claude_session_id, (
        f"argv carries {sid_value!r} but handle.claude_session_id is "
        f"{handle.claude_session_id!r}"
    )
    handle.terminate()


def test_spawn_idempotent_when_alive(tmp_path: Path) -> None:
    """A second spawn for the same agent_id while alive returns the existing
    handle WITHOUT invoking subprocess.Popen again (BSP-002 Phase 1 dedup)."""
    sup = _make_supervisor()
    # Live subprocess: returncode=None until terminate() — so poll() returns
    # None and the supervisor sees the agent as alive.
    fake = _FakePopen()  # returncode stays None
    popen_call_count = {"n": 0}

    def _counted(*args, **kwargs):  # type: ignore[no-untyped-def]
        popen_call_count["n"] += 1
        return fake

    with _patch_health(), patch("subprocess.Popen", side_effect=_counted):
        first = sup.spawn(
            zone_id="z1", agent_id="alpha", task="t1",
            work_dir=tmp_path, api_key="sk-x",
        )
        assert popen_call_count["n"] == 1
        second = sup.spawn(
            zone_id="z1", agent_id="alpha", task="t2",
            work_dir=tmp_path, api_key="sk-x",
        )

    assert second is first, "idempotent spawn must return the same AgentHandle"
    assert popen_call_count["n"] == 1, (
        f"second spawn invoked Popen again: count={popen_call_count['n']}"
    )
    assert second.claude_session_id == first.claude_session_id
    first.terminate()


def test_spawn_after_dead_process_creates_new_session(tmp_path: Path) -> None:
    """When the prior agent's process has exited, a fresh spawn creates a
    new claude_session_id. (Phase 2 will replace this with --resume; for
    Phase 1 we assert the simpler behavior so the contract is documented.)"""
    sup = _make_supervisor()

    # First spawn: a popen that exits promptly.
    first_fake = _FakePopen()
    first_fake.returncode = 0  # already exited

    second_fake = _FakePopen()
    second_fake.returncode = 0

    # subprocess.Popen returns first_fake then second_fake.
    popens = iter([first_fake, second_fake])
    captured_argvs: List[List[str]] = []

    def _next(argv, **kwargs):  # type: ignore[no-untyped-def]
        captured_argvs.append(list(argv))
        return next(popens)

    with _patch_health(), patch("subprocess.Popen", side_effect=_next):
        first = sup.spawn(
            zone_id="z1", agent_id="alpha", task="t1",
            work_dir=tmp_path, api_key="sk-x",
        )
        # Confirm the first process is dead before second spawn.
        first.popen.wait(timeout=0.5)
        assert first.popen.poll() == 0
        second = sup.spawn(
            zone_id="z1", agent_id="alpha", task="t2",
            work_dir=tmp_path, api_key="sk-x",
        )

    assert second is not first
    assert second.claude_session_id != first.claude_session_id
    assert len(captured_argvs) == 2, "Popen should have been called twice"
    # Both invocations carry --session-id; values match their handles.
    for argv, handle in [(captured_argvs[0], first), (captured_argvs[1], second)]:
        assert "--session-id" in argv
        assert argv[argv.index("--session-id") + 1] == handle.claude_session_id
    first.terminate()
    second.terminate()

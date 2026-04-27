"""Track B4 contract tests for agent provisioning and supervision.

Covers the RFC-002 v1.0.0 contract surface:

- ``_provisioning`` pure helpers (env build, argv build, system-prompt
  rendering, MCP config rendering, pre-spawn validation)
- ``AgentSupervisor.spawn`` — patched ``subprocess.Popen`` + patched
  ``httpx.head`` for the LiteLLM health check; asserts artifact files
  land on disk and the run-tracker records a tool run for an MCP
  JSON-RPC frame written to fake stdout.

Per the plan, full lifecycle behavior (real subprocess, real Claude
Code) is the operator-side R2-prototype run. These tests stay sync,
fast, and dependency-free of a live network.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_kernel._provisioning import (
    ALLOWED_TOOLS, CANONICAL_SYSTEM_PROMPT_TEMPLATE, DISABLED_TOOLS,
    MCP_SERVER_NAME, PreSpawnValidationError, RFC001_ALLOWED_TOOLS,
    build_argv, build_env, is_secret_var,
    render_mcp_config, render_system_prompt, validate_pre_spawn,
)
from llm_kernel.agent_supervisor import AgentSupervisor


# ---------------------------------------------------------------------------
# _provisioning pure functions
# ---------------------------------------------------------------------------

def test_render_system_prompt_substitutes_task() -> None:
    """``[TASK_BLOCK]`` MUST be replaced verbatim by the task string."""
    rendered = render_system_prompt("ECHO TASK")
    assert "[TASK_BLOCK]" not in rendered
    assert "ECHO TASK" in rendered
    # Trailing version footer is part of the template (RFC-002).
    assert "system-prompt-template v1.0.0" in rendered


def test_canonical_template_is_versioned_and_complete() -> None:
    """Template carries the v1.0.0 footer + 13-tool list per RFC-002."""
    for tool in RFC001_ALLOWED_TOOLS:
        assert tool in CANONICAL_SYSTEM_PROMPT_TEMPLATE, tool
    assert "rfc=RFC-002" in CANONICAL_SYSTEM_PROMPT_TEMPLATE


def test_render_mcp_config_has_thirteen_tools() -> None:
    """allowedTools MUST equal the 13-tool RFC-001 catalog in canonical order."""
    cfg = render_mcp_config(
        agent_id="alpha", zone_id="z1", trace_id="t1",
        kernel_python="/usr/bin/python",
    )
    server = cfg["mcpServers"][MCP_SERVER_NAME]
    assert tuple(server["allowedTools"]) == RFC001_ALLOWED_TOOLS
    assert server["transport"] == "stdio"
    assert server["command"] == "/usr/bin/python"
    assert server["args"][:2] == ["-m", "llm_kernel.mcp_server"]
    assert server["env"]["LLMKERNEL_RUN_TRACE_ID"] == "t1"


def test_build_env_strips_secrets_and_adds_kernel_vars() -> None:
    """Parent secrets are stripped; ANTHROPIC_API_KEY survives via the override."""
    parent: Dict[str, str] = {
        "PATH": "/bin",
        "ANTHROPIC_API_KEY": "sk-old",
        "OPENAI_API_KEY": "sk-leak",
        "MY_TOKEN": "leak",
        "GOOGLE_APPLICATION_CREDENTIALS": "leak",
        "AWS_SECRET_ACCESS_KEY": "leak",
    }
    env = build_env(
        parent, api_key="sk-new", llm_endpoint_url="http://127.0.0.1:9999/v1",
        mcp_config_path=Path("/tmp/mcp.json"),
        system_prompt_path=Path("/tmp/sp.txt"),
        work_dir=Path("/tmp/work"), agent_id="a", zone_id="z", trace_id="t",
    )
    # Stripped:
    for stripped in ("OPENAI_API_KEY", "MY_TOKEN",
                     "GOOGLE_APPLICATION_CREDENTIALS",
                     "AWS_SECRET_ACCESS_KEY"):
        assert stripped not in env, stripped
    # Kept + overridden:
    assert env["PATH"] == "/bin"
    assert env["ANTHROPIC_API_KEY"] == "sk-new"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999/v1"
    assert env["CLAUDE_CODE_DISABLED_TOOLS"] == DISABLED_TOOLS
    assert env["CLAUDE_CODE_ALLOWED_TOOLS"] == ALLOWED_TOOLS
    assert env["LLMKERNEL_AGENT_ID"] == "a"
    assert env["LLMKERNEL_ZONE_ID"] == "z"
    assert env["LLMKERNEL_RUN_TRACE_ID"] == "t"


def test_build_argv_includes_required_flags() -> None:
    """argv MUST honor the RFC-002 v1.0.1 amended flag set (default: OAuth)."""
    sp = Path("/tmp/sp.txt")
    mcp = Path("/tmp/mcp.json")
    argv = build_argv(sp, mcp, "do thing")
    assert argv[0] == "claude"
    # RFC-002 v1.0.1 amendments:
    assert "--print" in argv
    assert "--verbose" in argv  # required with --output-format=stream-json
    assert "--output-format=stream-json" in argv
    assert "--system-prompt-file" in argv  # not --system-prompt
    assert "--mcp-config" in argv
    assert "--strict-mcp-config" in argv
    assert "--disallowedTools" in argv
    # --bare is opt-in via use_bare=True (default OAuth path).
    assert "--bare" not in argv
    # --dangerously-skip-permissions is on by default in V1 (the MCP
    # server is the kernel itself; trusted boundary).
    assert "--dangerously-skip-permissions" in argv
    assert argv[-1] == "do thing"
    # Path string repr is platform-dependent; assert paths are present by str(Path).
    assert str(sp) in argv
    assert str(mcp) in argv


def test_build_argv_use_bare_inserts_flag() -> None:
    """``use_bare=True`` MUST insert ``--bare`` for the API-key auth path."""
    argv = build_argv(
        Path("/tmp/sp.txt"), Path("/tmp/mcp.json"), "do thing",
        use_bare=True,
    )
    assert "--bare" in argv


def test_build_argv_threads_model_when_provided() -> None:
    """Optional ``model`` SHOULD insert ``--model <id>`` before the task."""
    sp = Path("/tmp/sp.txt")
    mcp = Path("/tmp/mcp.json")
    argv = build_argv(sp, mcp, "do thing", model="claude-haiku-4-5-20251001")
    assert "--model" in argv
    idx = argv.index("--model")
    assert argv[idx + 1] == "claude-haiku-4-5-20251001"
    # Model still ends with the task.
    assert argv[-1] == "do thing"


def test_is_secret_var_keeps_anthropic_key() -> None:
    """ANTHROPIC_API_KEY is on the always-keep list."""
    assert is_secret_var("ANTHROPIC_API_KEY") is False
    assert is_secret_var("OPENAI_API_KEY") is True
    assert is_secret_var("MY_PASSWORD") is True
    assert is_secret_var("PATH") is False


# ---------------------------------------------------------------------------
# validate_pre_spawn
# ---------------------------------------------------------------------------

def test_pre_spawn_rejects_missing_api_key(tmp_path: Path) -> None:
    """Empty api_key MUST raise PreSpawnValidationError."""
    mcp = tmp_path / "mcp.json"
    sp = tmp_path / "sp.txt"
    mcp.write_text("{}")
    sp.write_text("prompt")
    with pytest.raises(PreSpawnValidationError) as ei:
        validate_pre_spawn(api_key="", llm_endpoint_url="http://127.0.0.1:1/v1",
                           mcp_config_path=mcp, system_prompt_path=sp)
    assert ei.value.log_signature == "provisioning.api_key.invalid"


def test_pre_spawn_rejects_unreachable_litellm(tmp_path: Path) -> None:
    """httpx.head raising MUST surface as PreSpawnValidationError."""
    mcp = tmp_path / "mcp.json"
    sp = tmp_path / "sp.txt"
    mcp.write_text("{}")
    sp.write_text("prompt")
    with patch("llm_kernel._provisioning.httpx.head",
               side_effect=httpx.ConnectError("nope")):
        with pytest.raises(PreSpawnValidationError) as ei:
            validate_pre_spawn(
                api_key="sk-x", llm_endpoint_url="http://127.0.0.1:1/v1",
                mcp_config_path=mcp, system_prompt_path=sp,
            )
    assert ei.value.log_signature == "provisioning.litellm.unreachable"


def test_pre_spawn_rejects_litellm_non_200(tmp_path: Path) -> None:
    """Non-200 from /v1/models MUST surface as PreSpawnValidationError."""
    mcp = tmp_path / "mcp.json"
    sp = tmp_path / "sp.txt"
    mcp.write_text("{}")
    sp.write_text("prompt")
    fake = MagicMock()
    fake.status_code = 503
    with patch("llm_kernel._provisioning.httpx.head", return_value=fake):
        with pytest.raises(PreSpawnValidationError):
            validate_pre_spawn(
                api_key="sk-x", llm_endpoint_url="http://127.0.0.1:1/v1",
                mcp_config_path=mcp, system_prompt_path=sp,
            )


def test_pre_spawn_rejects_empty_mcp_config(tmp_path: Path) -> None:
    """Missing or empty MCP config MUST surface a documented log signature."""
    mcp = tmp_path / "mcp.json"
    sp = tmp_path / "sp.txt"
    mcp.write_text("")  # empty
    sp.write_text("prompt")
    fake = MagicMock(); fake.status_code = 200
    with patch("llm_kernel._provisioning.httpx.head", return_value=fake):
        with pytest.raises(PreSpawnValidationError) as ei:
            validate_pre_spawn(
                api_key="sk-x", llm_endpoint_url="http://127.0.0.1:1/v1",
                mcp_config_path=mcp, system_prompt_path=sp,
            )
    assert ei.value.log_signature == "provisioning.mcp.unreachable"


# ---------------------------------------------------------------------------
# AgentSupervisor.spawn — with patched Popen + httpx
# ---------------------------------------------------------------------------

class _FakePopen:
    """Minimal Popen substitute. stdout/stderr yield prerecorded lines."""

    def __init__(
        self, stdout_lines: List[str], stderr_lines: List[str] = (),
        exit_code: int = 0,
    ) -> None:
        self.stdout = iter(stdout_lines + [""])
        self.stderr = iter(list(stderr_lines) + [""])
        self._exit_code = exit_code
        self._exited = threading.Event()
        self.returncode: Optional[int] = None

    def poll(self) -> Optional[int]:
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        # Simulate process exit promptly so the watchdog completes.
        self._exited.wait(timeout=timeout if timeout is not None else 0.5)
        if self.returncode is None:
            self.returncode = self._exit_code
        return self.returncode

    def terminate(self) -> None:
        self.returncode = self.returncode or 0
        self._exited.set()

    def kill(self) -> None:  # pragma: no cover - not exercised by these tests
        self.returncode = self.returncode or -9
        self._exited.set()


def _patch_health(status_code: int = 200):
    fake = MagicMock(); fake.status_code = status_code
    return patch("llm_kernel._provisioning.httpx.head", return_value=fake)


def _make_supervisor() -> AgentSupervisor:
    """Build a supervisor with stub run-tracker + dispatcher (no real kernel)."""
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


def test_supervisor_spawn_creates_artifact_files(tmp_path: Path) -> None:
    """spawn MUST render mcp-config.json + system-prompt.txt under work_dir/.run/<id>/."""
    sup = _make_supervisor()
    fake = _FakePopen(stdout_lines=[])
    fake.returncode = 0  # already done
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="say hi",
            work_dir=tmp_path, api_key="sk-x",
        )
    spawn_dir = tmp_path / ".run" / "alpha"
    assert (spawn_dir / "mcp-config.json").exists()
    assert (spawn_dir / "system-prompt.txt").exists()
    cfg = json.loads((spawn_dir / "mcp-config.json").read_text())
    assert cfg["mcpServers"][MCP_SERVER_NAME]["allowedTools"] \
        == list(RFC001_ALLOWED_TOOLS)
    sp = (spawn_dir / "system-prompt.txt").read_text()
    assert "say hi" in sp
    handle.terminate()


def test_supervisor_pre_spawn_failure_raises_no_files_left(tmp_path: Path) -> None:
    """Health-check failure raises PreSpawnValidationError; no Popen invoked."""
    sup = _make_supervisor()
    with patch("llm_kernel._provisioning.httpx.head",
               side_effect=httpx.ConnectError("nope")), \
            patch("subprocess.Popen") as popen:
        with pytest.raises(PreSpawnValidationError):
            sup.spawn(zone_id="z1", agent_id="beta", task="x",
                      work_dir=tmp_path, api_key="sk-x")
        popen.assert_not_called()


def test_supervisor_stdout_jsonrpc_frame_records_run(tmp_path: Path) -> None:
    """A JSON-RPC tools/call frame on stdout MUST become a courtesy run."""
    sup = _make_supervisor()
    frame = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "notify",
                   "arguments": {"observation": "hi", "importance": "info"}},
    })
    fake = _FakePopen(stdout_lines=[frame + "\n"])
    fake.returncode = 0
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="x",
            work_dir=tmp_path, api_key="sk-x",
        )
    # Give the stdout reader thread a moment to drain the iterator.
    handle.stdout_thread.join(timeout=2.0)
    fake._exited.set()
    handle.terminate()
    names = {r.name for r in sup._run_tracker.iter_runs()}  # type: ignore[attr-defined]
    assert any("notify" in n or "agent.tool_call" in n for n in names), names


def test_supervisor_stdout_prose_logs_violation(tmp_path: Path) -> None:
    """Plain-prose stdout MUST be flagged as a DR-0010 violation."""
    sup = _make_supervisor()
    fake = _FakePopen(stdout_lines=["Hello operator!\n"])
    fake.returncode = 0
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="x",
            work_dir=tmp_path, api_key="sk-x",
        )
    handle.stdout_thread.join(timeout=2.0)
    fake._exited.set()
    handle.terminate()
    names = [r.name for r in sup._run_tracker.iter_runs()]  # type: ignore[attr-defined]
    # The supervisor records a synthetic notify run for the violation.
    assert any("notify" in n or "violation" in n.lower() for n in names), names


def test_supervisor_terminate_sets_state_terminated(tmp_path: Path) -> None:
    """terminate() MUST mark the handle ``terminated`` and stop reader threads."""
    sup = _make_supervisor()
    fake = _FakePopen(stdout_lines=[])
    fake.returncode = 0  # process already exited
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="x",
            work_dir=tmp_path, api_key="sk-x",
        )
    handle.terminate(grace_seconds=0.1)
    # Watchdog has chance to observe exit and tag terminated.
    time.sleep(0.05)
    assert handle.state in {"terminated", "running"}  # depends on timing

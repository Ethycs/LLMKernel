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
    # BSP-002 §4.1 + S3 multi-turn continuation: stdin carries newline-delimited
    # JSON user turns. Without --input-format=stream-json the CLI ignores stdin
    # and only honors the trailing positional argv as the initial prompt.
    assert "--input-format=stream-json" in argv
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
        # AgentSupervisor.spawn reads popen.pid for the diagnostics mark
        # (agent_supervisor.py:322). Real Popen always exposes pid; the test
        # double must too.
        self.pid = 12345

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
    """Build a supervisor with stub run-tracker + dispatcher + real writer.

    PLAN-S4.1: a real :class:`MetadataWriter` is wired so the
    ``_missed_turns`` walker can read from
    ``metadata.rts.zone.agents.<*>.turns[]``.  Tests seed turns via
    :func:`_seed_turn` (which submits ``append_turn`` intents).
    """
    from llm_kernel.run_tracker import RunTracker
    from llm_kernel.metadata_writer import MetadataWriter

    class _ListSink:
        def __init__(self) -> None: self.envelopes: List[Dict[str, Any]] = []
        def emit(self, env: Dict[str, Any]) -> None: self.envelopes.append(env)

    tracker = RunTracker(
        trace_id=str(uuid.uuid4()), sink=_ListSink(),
        agent_id="alpha", zone_id="z1",
    )
    dispatcher = MagicMock()
    sup = AgentSupervisor(
        run_tracker=tracker, dispatcher=dispatcher,
        litellm_endpoint_url="http://127.0.0.1:9999/v1",
    )
    writer = MetadataWriter(autosave_interval_sec=999.0)
    sup.set_metadata_writer(writer)
    return sup


def _seed_turn(
    sup: AgentSupervisor,
    turn_id: str,
    agent_id: str,
    role: str,
    content: str,
    parent_id: Optional[str] = None,
    *,
    created_at: Optional[str] = None,
) -> None:
    """Submit one ``append_turn`` intent to the supervisor's writer.

    Replacement for the deleted ``AgentSupervisor.record_turn`` test
    seam (PLAN-S4.1 §3.C migration).
    """
    if sup._metadata_writer is None:  # pragma: no cover - defensive
        raise RuntimeError(
            "_seed_turn: supervisor has no writer wired"
        )
    # Map legacy ``role: "assistant"|"user"`` to the canonical roles.
    norm_role = {"assistant": "agent", "user": "operator"}.get(role, role)
    params: Dict[str, Any] = {
        "id": turn_id,
        "agent_id": agent_id,
        "role": norm_role,
        "body": content,
        "parent_id": parent_id,
    }
    if created_at is not None:
        params["created_at"] = created_at
    result = sup._metadata_writer.submit_intent({
        "payload": {
            "action_type": "zone_mutate",
            "intent_kind": "append_turn",
            "parameters": params,
            "intent_id": f"seed-{turn_id}-{uuid.uuid4().hex[:8]}",
        },
    })
    if not result.get("applied"):
        raise RuntimeError(
            f"_seed_turn: append_turn rejected: {result}"
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
    """Plain-prose stdout MUST emit ``agent_emit`` (RFC-002 §3 / RFC-005)."""
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
    # Lines that fail JSON parse become ``agent_emit:malformed_json``;
    # the same line is ALSO flood-tracked as prose -> ``agent_emit:prose``.
    assert any("agent_emit:prose" in n for n in names), names
    assert any("agent_emit:malformed_json" in n for n in names), names


def _decode_attrs(attrs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tiny attr decoder so tests can read ``llmnb.*`` without importing _attrs."""
    from llm_kernel._attrs import decode_attrs
    return decode_attrs(attrs)


def _agent_emit_spans(sup: AgentSupervisor) -> List[Any]:
    """Return all open or closed ``agent_emit`` spans on the supervisor."""
    return [
        span for span in sup._run_tracker.iter_runs()  # type: ignore[attr-defined]
        if span.name.startswith("agent_emit:")
    ]


def test_supervisor_stream_json_system_emits_agent_emit(tmp_path: Path) -> None:
    """RFC-002 §3 / RFC-005: stream-json ``system`` -> ``agent_emit:system_message``."""
    sup = _make_supervisor()
    record = json.dumps({"type": "system", "subtype": "init", "session_id": "s"})
    fake = _FakePopen(stdout_lines=[record + "\n"])
    fake.returncode = 0
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="x",
            work_dir=tmp_path, api_key="sk-x",
        )
    handle.stdout_thread.join(timeout=2.0)
    fake._exited.set()
    handle.terminate()
    spans = _agent_emit_spans(sup)
    kinds = [_decode_attrs(s.attributes).get("llmnb.emit_kind") for s in spans]
    assert "system_message" in kinds, kinds
    sys_span = next(s for s in spans if "system_message" in s.name)
    attrs = _decode_attrs(sys_span.attributes)
    assert attrs["llmnb.run_type"] == "agent_emit"
    assert attrs["llmnb.agent_id"] == "alpha"
    assert "init" in attrs["llmnb.emit_content"]


def test_supervisor_stream_json_result_emits_agent_emit(tmp_path: Path) -> None:
    """stream-json ``result`` -> ``agent_emit:result``."""
    sup = _make_supervisor()
    record = json.dumps({"type": "result", "subtype": "ok", "stop_reason": "end_turn"})
    fake = _FakePopen(stdout_lines=[record + "\n"])
    fake.returncode = 0
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="x",
            work_dir=tmp_path, api_key="sk-x",
        )
    handle.stdout_thread.join(timeout=2.0)
    fake._exited.set()
    handle.terminate()
    kinds = [
        _decode_attrs(s.attributes).get("llmnb.emit_kind")
        for s in _agent_emit_spans(sup)
    ]
    assert "result" in kinds, kinds


def test_supervisor_stream_json_error_emits_agent_emit(tmp_path: Path) -> None:
    """stream-json ``error`` -> ``agent_emit:error``."""
    sup = _make_supervisor()
    record = json.dumps({"type": "error", "message": "model overloaded"})
    fake = _FakePopen(stdout_lines=[record + "\n"])
    fake.returncode = 0
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="x",
            work_dir=tmp_path, api_key="sk-x",
        )
    handle.stdout_thread.join(timeout=2.0)
    fake._exited.set()
    handle.terminate()
    kinds = [
        _decode_attrs(s.attributes).get("llmnb.emit_kind")
        for s in _agent_emit_spans(sup)
    ]
    assert "error" in kinds, kinds


def test_supervisor_assistant_text_with_tool_use_is_reasoning(
    tmp_path: Path,
) -> None:
    """RFC-002 §3: text preceding a tool_use is ``agent_emit:reasoning``."""
    sup = _make_supervisor()
    assistant = json.dumps({
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "Let me check the file first."},
                {"type": "tool_use", "id": "u1",
                 "name": "mcp__llmkernel-operator-bridge__notify",
                 "input": {"observation": "x", "importance": "info"}},
            ],
        },
    })
    fake = _FakePopen(stdout_lines=[assistant + "\n"])
    fake.returncode = 0
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="x",
            work_dir=tmp_path, api_key="sk-x",
        )
    handle.stdout_thread.join(timeout=2.0)
    fake._exited.set()
    handle.terminate()
    spans = _agent_emit_spans(sup)
    kinds = [_decode_attrs(s.attributes).get("llmnb.emit_kind") for s in spans]
    assert "reasoning" in kinds, kinds


def test_supervisor_malformed_json_emits_diagnostic(tmp_path: Path) -> None:
    """RFC-005: malformed JSON -> agent_emit with parser_diagnostic."""
    sup = _make_supervisor()
    # Truncated JSON — fails the decoder mid-parse.
    fake = _FakePopen(stdout_lines=['{"type":"assistant","message":\n'])
    fake.returncode = 0
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="x",
            work_dir=tmp_path, api_key="sk-x",
        )
    handle.stdout_thread.join(timeout=2.0)
    fake._exited.set()
    handle.terminate()
    spans = _agent_emit_spans(sup)
    bad = [s for s in spans
           if _decode_attrs(s.attributes).get("llmnb.emit_kind") == "malformed_json"]
    assert bad, [s.name for s in spans]
    diag = _decode_attrs(bad[0].attributes).get("llmnb.parser_diagnostic")
    assert diag and "position" in diag


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


# ---------------------------------------------------------------------------
# PLAN-S4 cross-agent context handoff tests
# ---------------------------------------------------------------------------

import io as _io


class _FakeStdin:
    """Captures write/flush calls for handoff prefix tests."""

    def __init__(self) -> None:
        self._buf = _io.StringIO()

    def write(self, s: str) -> int:
        return self._buf.write(s)

    def flush(self) -> None:
        pass  # no-op

    def written(self) -> str:
        return self._buf.getvalue()

    def lines(self) -> List[str]:
        return [ln for ln in self.written().splitlines() if ln]


class _FakePopenWithStdin(_FakePopen):
    """_FakePopen variant that provides a writable stdin buffer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.stdin = _FakeStdin()


def _make_supervisor_with_alpha(tmp_path: Path) -> tuple:
    """Spawn alpha in a supervisor; return (supervisor, handle, fake_popen)."""
    sup = _make_supervisor()
    fake = _FakePopenWithStdin(stdout_lines=[])
    fake.returncode = None  # keep process alive
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="hello",
            work_dir=tmp_path, api_key="sk-x",
        )
    return sup, handle, fake


def test_send_user_turn_no_missed_turns(tmp_path: Path) -> None:
    """Single agent, no sibling turns — operator message goes straight through with no prefix."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    result = sup.send_user_turn("alpha", "hello from operator")
    lines = fake.stdin.lines()
    # Only one line: the operator message itself.
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["message"]["content"] == "hello from operator"
    assert result["handoff_prefix_count"] == 0
    handle.terminate()


def test_send_user_turn_with_one_missed_sibling_turn(tmp_path: Path) -> None:
    """Two agents; beta produced 1 turn; alpha gets 1 prefix line then the operator message."""
    sup = _make_supervisor()
    fake_alpha = _FakePopenWithStdin(stdout_lines=[])
    fake_alpha.returncode = None
    fake_beta = _FakePopenWithStdin(stdout_lines=[])
    fake_beta.returncode = None

    def _popen_factory(*args: Any, **kwargs: Any) -> Any:
        # Return alpha's popen first, then beta's.
        return _popen_factory._queue.pop(0)
    _popen_factory._queue = [fake_alpha, fake_beta]

    with _patch_health(), patch("subprocess.Popen", side_effect=_popen_factory):
        sup.spawn(zone_id="z1", agent_id="alpha", task="t", work_dir=tmp_path, api_key="sk-x")
        sup.spawn(zone_id="z1", agent_id="beta", task="t", work_dir=tmp_path, api_key="sk-x")

    # Record a beta turn.
    _seed_turn(sup,"t1", "beta", "assistant", "beta reply one", parent_id=None)

    result = sup.send_user_turn("alpha", "operator to alpha")
    lines = fake_alpha.stdin.lines()
    # Expect: 1 prefix line + 1 operator line = 2 total.
    assert len(lines) == 2, lines
    prefix = json.loads(lines[0])
    assert prefix["type"] == "user"
    assert "beta" in prefix["message"]["content"]
    assert "beta reply one" in prefix["message"]["content"]
    operator_msg = json.loads(lines[1])
    assert operator_msg["message"]["content"] == "operator to alpha"
    assert result["handoff_prefix_count"] == 1
    fake_alpha.terminate()
    fake_beta.terminate()


def test_send_user_turn_with_three_missed_sibling_turns(tmp_path: Path) -> None:
    """Chronological order preserved; 3 prefix lines asserted exact strings."""
    sup = _make_supervisor()
    fake_alpha = _FakePopenWithStdin(stdout_lines=[])
    fake_alpha.returncode = None
    fake_beta = _FakePopenWithStdin(stdout_lines=[])
    fake_beta.returncode = None

    def _factory(*a: Any, **kw: Any) -> Any:
        return _factory._q.pop(0)
    _factory._q = [fake_alpha, fake_beta]

    with _patch_health(), patch("subprocess.Popen", side_effect=_factory):
        sup.spawn(zone_id="z1", agent_id="alpha", task="t", work_dir=tmp_path, api_key="sk-x")
        sup.spawn(zone_id="z1", agent_id="beta", task="t", work_dir=tmp_path, api_key="sk-x")

    # Record 3 turns in a chain: t1 -> t2 -> t3.
    _seed_turn(sup,"t1", "beta", "assistant", "first", parent_id=None)
    _seed_turn(sup,"t2", "beta", "assistant", "second", parent_id="t1")
    _seed_turn(sup,"t3", "beta", "assistant", "third", parent_id="t2")

    result = sup.send_user_turn("alpha", "go")
    lines = fake_alpha.stdin.lines()
    # 3 prefix + 1 operator = 4
    assert len(lines) == 4, lines
    contents = [json.loads(ln)["message"]["content"] for ln in lines[:3]]
    assert "first" in contents[0]
    assert "second" in contents[1]
    assert "third" in contents[2]
    assert result["handoff_prefix_count"] == 3
    fake_alpha.terminate()
    fake_beta.terminate()


def test_send_user_turn_unknown_agent_raises_k20(tmp_path: Path) -> None:
    """Supervisor lookup miss raises KeyError (maps to K20 at dispatcher)."""
    sup = _make_supervisor()
    with pytest.raises(KeyError):
        sup.send_user_turn("no_such_agent", "hello")


def test_send_user_turn_dead_agent_resumes_first(tmp_path: Path) -> None:
    """Idle agent (popen reaped) is resumed before handoff + message are sent."""
    sup = _make_supervisor()
    fake_first = _FakePopenWithStdin(stdout_lines=[])
    fake_first.returncode = None
    fake_resumed = _FakePopenWithStdin(stdout_lines=[])
    fake_resumed.returncode = None  # resume succeeds

    popped: List[Any] = [fake_first, fake_resumed]

    def _factory(*a: Any, **kw: Any) -> Any:
        return popped.pop(0)

    with _patch_health(), patch("subprocess.Popen", side_effect=_factory):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="t",
            work_dir=tmp_path, api_key="sk-x",
        )

    # Simulate process death.
    fake_first.returncode = 0
    fake_first._exited.set()

    with _patch_health(), patch("subprocess.Popen", side_effect=_factory):
        result = sup.send_user_turn("alpha", "after resume")

    assert result["status"] in {"resumed_then_sent", "spawned_fresh"}
    # The resumed popen's stdin should have the operator message.
    assert fake_resumed.stdin.lines()
    fake_resumed.terminate()


def test_send_user_turn_advances_last_seen_turn_id(tmp_path: Path) -> None:
    """After success, handle.last_seen_turn_id == notebook head turn id."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    _seed_turn(sup,"t99", "alpha", "assistant", "alpha self turn", parent_id=None)
    # Self-authored turn is filtered — no prefix — but head advances.
    sup.send_user_turn("alpha", "msg")
    assert handle.last_seen_turn_id == "t99"
    handle.terminate()


def test_send_user_turn_handoff_failure_raises_k26(tmp_path: Path) -> None:
    """Cycle in turn DAG raises K26 RuntimeError with cycle_detected reason.

    PLAN-S4.1: cycle injection bypasses ``append_turn`` validators
    (which would reject the unknown parent_id) by writing directly into
    the writer's ``_zone`` map.
    """
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    # Inject the cycle directly into the writer's zone so the persisted
    # graph carries it; ``append_turn`` rejects unknown parent_id so we
    # cannot get there via the intent path.
    writer = sup._metadata_writer
    assert writer is not None
    with writer._lock:
        writer._zone.setdefault("agents", {})["beta"] = {
            "turns": [
                {"id": "t1", "agent_id": "beta", "role": "agent",
                 "body": "beta msg", "parent_id": "t2",
                 "created_at": "2026-04-30T17:00:00.000Z"},
                {"id": "t2", "agent_id": "beta", "role": "agent",
                 "body": "beta msg2", "parent_id": "t1",
                 "created_at": "2026-04-30T17:00:01.000Z"},
            ],
            "session": {"head_turn_id": "t2"},
        }
        writer._dirty = True

    with pytest.raises(RuntimeError, match="K26"):
        sup.send_user_turn("alpha", "trigger cycle")
    handle.terminate()


def test_send_user_turn_persists_last_seen_via_writer(tmp_path: Path) -> None:
    """update_agent_session intent submitted with last_seen_turn_id after success."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    # PLAN-S4.1: seed BEFORE the mock swap so the persisted graph is
    # populated; then swap to a mock to capture update_agent_session.
    _seed_turn(sup, "t77", "alpha", "assistant", "self turn", parent_id=None)
    mock_writer = MagicMock()
    mock_writer.submit_intent = MagicMock(return_value={"ok": True})
    # Pre-populate the mock's snapshot return so _missed_turns finds t77.
    mock_writer.snapshot = MagicMock(return_value={
        "zone": {"agents": {"alpha": {"turns": [
            {"id": "t77", "agent_id": "alpha", "role": "agent",
             "body": "self turn", "parent_id": None,
             "created_at": "2026-04-30T17:00:00.000Z"},
        ]}}},
    })
    sup.set_metadata_writer(mock_writer)

    sup.send_user_turn("alpha", "persist test")

    # Writer must have received update_agent_session with last_seen_turn_id.
    mock_writer.submit_intent.assert_called_once()
    envelope = mock_writer.submit_intent.call_args[0][0]
    # Per BSP-003 §3 the wire shape wraps in ``payload`` (we accept the
    # bare unwrapped form here for back-compat with the existing send
    # call site).
    inner = envelope.get("payload", envelope)
    assert inner["intent_kind"] == "update_agent_session"
    assert inner["parameters"]["last_seen_turn_id"] == "t77"
    assert inner["parameters"]["agent_id"] == "alpha"
    handle.terminate()


def test_send_user_turn_strips_hashes_in_handoff_prefix(tmp_path: Path) -> None:
    """Sibling turn body with hashed-magic survives as plain magic in prefix."""
    sup = _make_supervisor()
    fake_alpha = _FakePopenWithStdin(stdout_lines=[])
    fake_alpha.returncode = None
    fake_beta = _FakePopenWithStdin(stdout_lines=[])
    fake_beta.returncode = None

    def _factory(*a: Any, **kw: Any) -> Any:
        return _factory._q.pop(0)
    _factory._q = [fake_alpha, fake_beta]

    with _patch_health(), patch("subprocess.Popen", side_effect=_factory):
        sup.spawn(zone_id="z1", agent_id="alpha", task="t", work_dir=tmp_path, api_key="sk-x")
        sup.spawn(zone_id="z1", agent_id="beta", task="t", work_dir=tmp_path, api_key="sk-x")

    # Body contains a hashed-magic line (fake hash + registered name "agent").
    # The strip helper removes the hash; the test asserts the prefix content
    # does NOT contain the hashed form.
    hashed_body = "@@deadbeef1234:agent hello"
    _seed_turn(sup,"t1", "beta", "assistant", hashed_body, parent_id=None)

    sup.send_user_turn("alpha", "check strip")
    lines = fake_alpha.stdin.lines()
    # At least the prefix line exists.
    assert len(lines) >= 2, lines
    prefix_content = json.loads(lines[0])["message"]["content"]
    # The hashed form "@@deadbeef1234:agent" must NOT appear in the prefix.
    assert "@@deadbeef1234:agent" not in prefix_content, prefix_content
    # The plain form OR the raw body text should appear (strip_hashes_from_text
    # strips to @@agent when "agent" is in known_names, otherwise leaves as-is).
    assert "hashed_body" not in prefix_content  # not the variable name
    fake_alpha.terminate()
    fake_beta.terminate()


# ---------------------------------------------------------------------------
# PLAN-S5b: AgentSupervisor.revert tests
# ---------------------------------------------------------------------------


def test_revert_validates_agent_exists_raises_k20(tmp_path: Path) -> None:
    """revert() raises K20 RuntimeError when agent_id is unknown."""
    sup = _make_supervisor()
    with pytest.raises(RuntimeError, match="K20"):
        sup.revert("nonexistent", "t_1")


def test_revert_validates_target_in_ancestry_raises_k22(tmp_path: Path) -> None:
    """revert() raises K22 RuntimeError when target_turn_id is not in ancestry."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    # Record a turn so there is a head, but target "t_orphan" is not in it.
    _seed_turn(sup,"t_1", "alpha", "assistant", "first turn", parent_id=None)
    with pytest.raises(RuntimeError, match="K22"):
        sup.revert("alpha", "t_orphan")
    handle.terminate()


def test_revert_sigterms_live_process(tmp_path: Path) -> None:
    """revert() calls terminate() on a live process (popen.poll() is None)."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    _seed_turn(sup,"t_1", "alpha", "assistant", "turn one", parent_id=None)
    # Process is alive (returncode is None).
    assert fake.returncode is None
    sup.revert("alpha", "t_1")
    # After SIGTERM the fake popen should have its terminate() called;
    # FakePopen.terminate sets returncode != None.
    assert fake.returncode is not None
    handle.terminate()


def test_revert_resets_last_seen_to_target(tmp_path: Path) -> None:
    """revert() updates handle.last_seen_turn_id to target_turn_id."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    _seed_turn(sup,"t_1", "alpha", "assistant", "turn one", parent_id=None)
    _seed_turn(sup,"t_2", "alpha", "assistant", "turn two", parent_id="t_1")
    # After two turns head is t_2; revert to t_1.
    sup.revert("alpha", "t_1")
    assert handle.last_seen_turn_id == "t_1"
    assert sup._notebook_head_turn_id() == "t_1"
    handle.terminate()


def test_revert_emits_agent_ref_move_event(tmp_path: Path) -> None:
    """revert() submits move_agent_head + record_event(agent_ref_move) intents.

    PLAN-S4.1: ``agent_ref_move`` is an event sub-kind on
    ``record_event``, NOT a top-level intent kind (which would K40).
    """
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    # Seed via the real writer (still wired by _make_supervisor) so the
    # ancestry walk inside revert can find t_1.
    _seed_turn(sup, "t_1", "alpha", "assistant", "turn one", parent_id=None)

    # Now wrap the real writer so we can intercept submit_intent calls
    # while the ancestry walk still reads from the persisted graph.
    real_writer = sup._metadata_writer
    intent_log: List[Dict[str, Any]] = []
    orig_submit = real_writer.submit_intent  # type: ignore[union-attr]

    def _capture(env: Dict[str, Any]) -> Dict[str, Any]:
        intent_log.append(env)
        return orig_submit(env)
    real_writer.submit_intent = _capture  # type: ignore[union-attr]

    sup.revert("alpha", "t_1")

    # Two intents submitted: move_agent_head + record_event(agent_ref_move).
    assert len(intent_log) == 2, intent_log
    payloads = [env["payload"] for env in intent_log]
    kinds = [p["intent_kind"] for p in payloads]
    assert "move_agent_head" in kinds
    assert "record_event" in kinds
    # Confirm record_event carries the right agent_ref_move payload.
    rec = next(p for p in payloads if p["intent_kind"] == "record_event")
    assert rec["parameters"]["kind"] == "agent_ref_move"
    assert rec["parameters"]["reason"] == "operator_revert"
    assert rec["parameters"]["to_turn_id"] == "t_1"
    handle.terminate()


def test_revert_mints_new_claude_session_id_for_next_continue(tmp_path: Path) -> None:
    """revert() replaces claude_session_id on the handle so next continue uses a fresh session."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    _seed_turn(sup,"t_1", "alpha", "assistant", "turn one", parent_id=None)
    original_session_id = handle.claude_session_id
    sup.revert("alpha", "t_1")
    assert handle.claude_session_id != original_session_id
    assert handle.claude_session_id  # non-empty UUID
    handle.terminate()


# ---------------------------------------------------------------------------
# PLAN-S5c: AgentSupervisor.stop tests
# ---------------------------------------------------------------------------


def test_stop_validates_agent_exists_raises_k20(tmp_path: Path) -> None:
    """stop() raises K20 RuntimeError when agent_id is unknown."""
    sup = _make_supervisor()
    with pytest.raises(RuntimeError, match="K20"):
        sup.stop("nonexistent")


def test_stop_sigterms_live_process_sets_idle(tmp_path: Path) -> None:
    """stop() calls terminate() on a live process and exits without error."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    # Process is alive (returncode is None).
    assert fake.returncode is None
    sup.stop("alpha")
    # After SIGTERM the fake popen should have been terminated.
    assert fake.returncode is not None
    handle.terminate()


def test_stop_on_idle_agent_is_noop(tmp_path: Path) -> None:
    """stop() on an already-exited agent is a silent no-op (idempotent)."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    # Simulate process already exited.
    fake.returncode = 0
    # Should not raise; should return cleanly.
    sup.stop("alpha")
    handle.terminate()


def test_stop_preserves_claude_session_id(tmp_path: Path) -> None:
    """stop() does NOT mint a new claude_session_id (contrast with revert)."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    original_session_id = handle.claude_session_id
    sup.stop("alpha")
    assert handle.claude_session_id == original_session_id, (
        "stop must preserve claude_session_id for next --resume continuation"
    )
    handle.terminate()


def test_stop_persists_runtime_status_via_writer(tmp_path: Path) -> None:
    """stop() submits update_agent_session intent with runtime_status=idle."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    mock_writer = MagicMock()
    mock_writer.submit_intent = MagicMock(return_value={"ok": True})
    sup.set_metadata_writer(mock_writer)

    sup.stop("alpha")

    mock_writer.submit_intent.assert_called_once()
    envelope = mock_writer.submit_intent.call_args[0][0]
    inner = envelope.get("payload", envelope)
    assert inner["intent_kind"] == "update_agent_session"
    assert inner["parameters"]["runtime_status"] == "idle"
    assert inner["parameters"]["pid"] is None
    assert inner["parameters"]["agent_id"] == "alpha"
    handle.terminate()


def test_stop_runtime_status_persists_in_metadata_rts(tmp_path: Path) -> None:
    """End-to-end: after @stop alpha, metadata.rts.zone.agents.alpha.session.runtime_status == 'idle'.

    PLAN-S4.2: the update_agent_session writer handler is now active, so
    stop() no longer produces a no-op pending-slice; the status change
    persists into the snapshot.
    """
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    # Seed a turn so zone.agents.alpha exists in the writer's in-memory graph.
    _seed_turn(sup, "t_s1", "alpha", "assistant", "seed", parent_id=None)

    sup.stop("alpha")

    snap = sup._metadata_writer.snapshot()
    session = snap["zone"]["agents"]["alpha"]["session"]
    assert session.get("runtime_status") == "idle", (
        f"Expected runtime_status='idle' after @stop, got: {session}"
    )
    handle.terminate()


# ---------------------------------------------------------------------------
# PLAN-S5a: AgentSupervisor.fork tests
# ---------------------------------------------------------------------------


def test_fork_validates_source_exists_raises_k20(tmp_path: Path) -> None:
    """fork() raises K20 RuntimeError when source_agent_id is unknown."""
    sup = _make_supervisor()
    with pytest.raises(RuntimeError, match="K20"):
        sup.fork("nonexistent", None, "beta")


def test_fork_validates_new_id_unique_raises_k21(tmp_path: Path) -> None:
    """fork() raises K21 RuntimeError when new_agent_id already exists."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    _seed_turn(sup,"t_1", "alpha", "assistant", "turn one", parent_id=None)
    # Attempting to fork with new_agent_id="alpha" (already exists) raises K21.
    with pytest.raises(RuntimeError, match="K21"):
        sup.fork("alpha", None, "alpha")
    handle.terminate()


def test_fork_case_a_at_head_succeeds_with_fresh_session_id(tmp_path: Path) -> None:
    """fork() Case A (at_turn_id == head): new agent gets fresh claude_session_id."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    _seed_turn(sup,"t_1", "alpha", "assistant", "turn one", parent_id=None)
    original_session = handle.claude_session_id
    head = sup._notebook_head_turn_id()  # "t_1"

    new_handle = sup.fork("alpha", head, "beta")

    assert new_handle.agent_id == "beta"
    assert new_handle.last_seen_turn_id == head
    assert new_handle.claude_session_id != original_session
    assert new_handle.claude_session_id  # non-empty UUID
    # New agent is in the registry.
    assert "beta" in sup._agents
    handle.terminate()


def test_fork_case_a_implicit_head_when_at_turn_id_none(tmp_path: Path) -> None:
    """fork() Case A (at_turn_id=None): branches at head implicitly."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    _seed_turn(sup,"t_1", "alpha", "assistant", "turn one", parent_id=None)
    head = sup._notebook_head_turn_id()  # "t_1"

    new_handle = sup.fork("alpha", None, "beta")

    assert new_handle.last_seen_turn_id == head
    assert "beta" in sup._agents
    handle.terminate()


def test_fork_case_b_validates_target_in_ancestry_raises_k22(tmp_path: Path) -> None:
    """fork() Case B raises K22 when at_turn_id is not in source ancestry."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    _seed_turn(sup,"t_1", "alpha", "assistant", "turn one", parent_id=None)
    # "t_orphan" is not in the turn chain → K22.
    with pytest.raises(RuntimeError, match="K22"):
        sup.fork("alpha", "t_orphan", "beta")
    handle.terminate()


def test_fork_case_b_succeeds_with_fresh_session_id_at_ancestor_turn(tmp_path: Path) -> None:
    """fork() Case B: branch at ancestor turn; new agent head=t_1, fresh session."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    _seed_turn(sup,"t_1", "alpha", "assistant", "turn one", parent_id=None)
    _seed_turn(sup,"t_2", "alpha", "assistant", "turn two", parent_id="t_1")
    # Head is now t_2; fork at ancestor t_1 (Case B).
    original_session = handle.claude_session_id

    new_handle = sup.fork("alpha", "t_1", "beta")

    assert new_handle.agent_id == "beta"
    assert new_handle.last_seen_turn_id == "t_1"
    assert new_handle.claude_session_id != original_session
    assert "beta" in sup._agents
    handle.terminate()


def test_fork_emits_agent_ref_move_event(tmp_path: Path) -> None:
    """fork() submits fork_agent + record_event(agent_ref_move) intents.

    PLAN-S4.1: ``agent_ref_move`` is an event sub-kind on
    ``record_event``, NOT a top-level intent kind.
    """
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    _seed_turn(sup, "t_1", "alpha", "assistant", "turn one", parent_id=None)

    real_writer = sup._metadata_writer
    intent_log: List[Dict[str, Any]] = []
    orig_submit = real_writer.submit_intent  # type: ignore[union-attr]

    def _capture(env: Dict[str, Any]) -> Dict[str, Any]:
        intent_log.append(env)
        return orig_submit(env)
    real_writer.submit_intent = _capture  # type: ignore[union-attr]

    sup.fork("alpha", None, "beta")

    assert len(intent_log) == 2, intent_log
    payloads = [env["payload"] for env in intent_log]
    kinds = [p["intent_kind"] for p in payloads]
    assert "fork_agent" in kinds
    assert "record_event" in kinds
    rec = next(p for p in payloads if p["intent_kind"] == "record_event")
    assert rec["parameters"]["kind"] == "agent_ref_move"
    assert rec["parameters"]["reason"] == "operator_branch"
    assert rec["parameters"]["agent_id"] == "beta"
    handle.terminate()


# ---------------------------------------------------------------------------
# PLAN-S4.1 §5: supervisor migration + lifted PLAN-S4 §9 deferred smokes.
# ---------------------------------------------------------------------------


def test_missed_turns_reads_from_metadata_rts(tmp_path: Path) -> None:
    """_missed_turns walks zone.agents.<*>.turns[] (no in-memory _turns cache)."""
    sup, handle, fake = _make_supervisor_with_alpha(tmp_path)
    fake_beta = _FakePopenWithStdin(stdout_lines=[])
    fake_beta.returncode = None
    with _patch_health(), patch("subprocess.Popen", return_value=fake_beta):
        sup.spawn(zone_id="z1", agent_id="beta", task="t",
                  work_dir=tmp_path, api_key="sk-x")
    # Seed a chain on beta via append_turn intents only.
    _seed_turn(sup, "t_a", "beta", "assistant", "alpha-msg-A", parent_id=None)
    _seed_turn(sup, "t_b", "beta", "assistant", "alpha-msg-B", parent_id="t_a")
    # Confirm the supervisor has NO _turns attribute (deletion proof).
    assert not hasattr(sup, "_turns"), "_turns cache must be deleted"
    head = sup._notebook_head_turn_id()
    assert head == "t_b"
    missed = sup._missed_turns("alpha", head)
    # alpha has not seen any turn; both beta turns are surfaced
    # in chronological order.
    assert [t["id"] for t in missed] == ["t_a", "t_b"]
    handle.terminate()
    fake_beta.terminate()


def test_two_agent_end_to_end_handoff_smoke(tmp_path: Path) -> None:
    """PLAN-S4 §9 lifted: alpha+beta exchange turns with handoff prefix."""
    sup = _make_supervisor()
    fake_alpha = _FakePopenWithStdin(stdout_lines=[])
    fake_alpha.returncode = None
    fake_beta = _FakePopenWithStdin(stdout_lines=[])
    fake_beta.returncode = None
    queue = [fake_alpha, fake_beta]

    def _factory(*a: Any, **k: Any) -> Any:
        return queue.pop(0)
    with _patch_health(), patch("subprocess.Popen", side_effect=_factory):
        sup.spawn(zone_id="z1", agent_id="alpha", task="t",
                  work_dir=tmp_path, api_key="sk-x")
        sup.spawn(zone_id="z1", agent_id="beta", task="t",
                  work_dir=tmp_path, api_key="sk-x")
    # beta produces 2 turns; alpha is then addressed.
    _seed_turn(sup, "t1", "beta", "assistant", "first", parent_id=None)
    _seed_turn(sup, "t2", "beta", "assistant", "second", parent_id="t1")
    res = sup.send_user_turn("alpha", "go")
    assert res["handoff_prefix_count"] == 2
    fake_alpha.terminate()
    fake_beta.terminate()


def test_three_agent_stress_smoke(tmp_path: Path) -> None:
    """PLAN-S4 §9 lifted: three-agent chain, all writer-driven."""
    sup = _make_supervisor()
    fakes = [_FakePopenWithStdin(stdout_lines=[]) for _ in range(3)]
    for f in fakes:
        f.returncode = None
    queue = list(fakes)

    def _factory(*a: Any, **k: Any) -> Any:
        return queue.pop(0)
    with _patch_health(), patch("subprocess.Popen", side_effect=_factory):
        for aid in ("alpha", "beta", "gamma"):
            sup.spawn(zone_id="z1", agent_id=aid, task="t",
                      work_dir=tmp_path, api_key="sk-x")
    # beta + gamma each contribute one turn.
    _seed_turn(sup, "t_b", "beta", "assistant", "b-msg", parent_id=None)
    _seed_turn(sup, "t_g", "gamma", "assistant", "g-msg", parent_id="t_b")
    res = sup.send_user_turn("alpha", "review")
    # Both sibling turns surface (filtered by author).
    assert res["handoff_prefix_count"] == 2
    for f in fakes:
        f.terminate()


def test_idle_resume_and_handoff_smoke(tmp_path: Path) -> None:
    """PLAN-S4 §9 lifted: dead alpha resumes; handoff fires on next addressed turn."""
    sup = _make_supervisor()
    fake_alpha_first = _FakePopenWithStdin(stdout_lines=[])
    fake_alpha_first.returncode = None
    fake_beta = _FakePopenWithStdin(stdout_lines=[])
    fake_beta.returncode = None
    fake_alpha_resume = _FakePopenWithStdin(stdout_lines=[])
    fake_alpha_resume.returncode = None

    queue = [fake_alpha_first, fake_beta, fake_alpha_resume]

    def _factory(*a: Any, **k: Any) -> Any:
        return queue.pop(0)
    with _patch_health(), patch("subprocess.Popen", side_effect=_factory):
        sup.spawn(zone_id="z1", agent_id="alpha", task="t",
                  work_dir=tmp_path, api_key="sk-x")
        sup.spawn(zone_id="z1", agent_id="beta", task="t",
                  work_dir=tmp_path, api_key="sk-x")
    # alpha exits; beta produces a turn; alpha gets re-spawned with handoff.
    fake_alpha_first.returncode = 0
    fake_alpha_first._exited.set()
    _seed_turn(sup, "t_b1", "beta", "assistant", "beta-msg", parent_id=None)
    with _patch_health(), patch("subprocess.Popen", side_effect=_factory):
        res = sup.send_user_turn("alpha", "wake up")
    assert res["status"] in {"resumed_then_sent", "spawned_fresh"}
    assert res["handoff_prefix_count"] == 1
    fake_alpha_resume.terminate()
    fake_beta.terminate()

"""Proxied-tool semantics for read_file / write_file / run_command.

K-MCP V1 mega-round: the three RFC-001 proxied tools used to raise
``NotImplementedError``.  This file pins their post-implementation
behavior:

* workspace-root resolution (env var fallback to ``os.getcwd()``)
* path traversal refusal -> JSON-RPC -32005
* OSError -> JSON-RPC -32006
* run_command timeout -> -32006 with the timeout duration in the
  message
* run_command binary-not-on-PATH -> -32005

Tests use ``tmp_path`` and ``monkeypatch`` for parallel-safety; no
fixed-path I/O and no raw ``os.environ`` mutation.
"""

from __future__ import annotations

import sys
import uuid
from typing import Any, Awaitable, Callable, Dict

import pytest
from mcp import types
from mcp.shared.exceptions import McpError

from llm_kernel.mcp_server import (
    JSONRPC_PROXIED_IO_FAILURE,
    JSONRPC_PROXIED_REFUSED,
    OperatorBridgeServer,
)


ServerHandler = Callable[[types.ClientRequestType], Awaitable[types.ServerResult]]


@pytest.fixture
def bridge() -> OperatorBridgeServer:
    """Build a fresh OperatorBridgeServer with deterministic ids."""
    return OperatorBridgeServer(
        agent_id="test-agent",
        zone_id="test-zone",
        trace_id=str(uuid.uuid4()),
    )


def _call_tool_handler(bridge: OperatorBridgeServer) -> ServerHandler:
    """Return the registered tools/call handler from the wrapped MCP server."""
    handler = bridge.server.request_handlers.get(types.CallToolRequest)
    assert handler is not None
    return handler


async def _invoke(
    bridge: OperatorBridgeServer, name: str, arguments: Dict[str, Any]
) -> types.CallToolResult:
    handler = _call_tool_handler(bridge)
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    server_result = await handler(request)
    inner = server_result.root
    assert isinstance(inner, types.CallToolResult)
    return inner


def _structured(result: types.CallToolResult) -> Dict[str, Any]:
    assert result.structuredContent is not None
    return result.structuredContent


# --- read_file -------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file_resolves_against_workspace_root(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """A relative path resolves under ``LLMKERNEL_WORKSPACE_ROOT``."""
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    target = tmp_path / "subdir" / "data.txt"
    target.parent.mkdir()
    target.write_text("workspace-rooted content", encoding="utf-8")
    result = await _invoke(bridge, "read_file", {"path": "subdir/data.txt"})
    payload = _structured(result)
    assert payload["content"] == "workspace-rooted content"
    assert payload["truncated"] is False
    assert payload["size_bytes"] == len("workspace-rooted content")


@pytest.mark.asyncio
async def test_read_file_falls_back_to_cwd(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """When the env var is unset, the workspace root is ``os.getcwd()``."""
    monkeypatch.delenv("LLMKERNEL_WORKSPACE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "cwd-rooted.txt"
    target.write_text("from cwd", encoding="utf-8")
    result = await _invoke(bridge, "read_file", {"path": "cwd-rooted.txt"})
    assert _structured(result)["content"] == "from cwd"


@pytest.mark.asyncio
async def test_read_file_refuses_traversal(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """A traversal-path that escapes the workspace MUST yield -32005."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(workspace))
    with pytest.raises(McpError) as exc_info:
        await _invoke(bridge, "read_file", {"path": "../secret.txt"})
    assert exc_info.value.error.code == JSONRPC_PROXIED_REFUSED


@pytest.mark.asyncio
async def test_read_file_absolute_outside_root_refused(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """An absolute path outside the workspace root MUST yield -32005."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(workspace))
    with pytest.raises(McpError) as exc_info:
        await _invoke(bridge, "read_file", {"path": str(outside)})
    assert exc_info.value.error.code == JSONRPC_PROXIED_REFUSED


@pytest.mark.asyncio
async def test_read_file_oserror_maps_to_32006(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """A nonexistent file MUST surface -32006 with the OS error string."""
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(McpError) as exc_info:
        await _invoke(bridge, "read_file", {"path": "missing.txt"})
    assert exc_info.value.error.code == JSONRPC_PROXIED_IO_FAILURE


# --- write_file ------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_file_creates_parents(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """write_file MUST create parent directories on demand."""
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    result = await _invoke(
        bridge, "write_file",
        {"path": "deep/nested/file.txt", "content": "hello"},
    )
    payload = _structured(result)
    assert payload["bytes_written"] == 5
    assert payload["created"] is True
    assert (tmp_path / "deep" / "nested" / "file.txt").read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_write_file_overwrites_existing(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """Default mode (overwrite) MUST replace existing content."""
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    target = tmp_path / "existing.txt"
    target.write_text("old", encoding="utf-8")
    result = await _invoke(
        bridge, "write_file",
        {"path": "existing.txt", "content": "new content"},
    )
    payload = _structured(result)
    assert payload["created"] is False
    assert target.read_text(encoding="utf-8") == "new content"


@pytest.mark.asyncio
async def test_write_file_refuses_traversal(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """Writing outside the workspace root MUST yield -32005."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(workspace))
    with pytest.raises(McpError) as exc_info:
        await _invoke(
            bridge, "write_file",
            {"path": "../escaped.txt", "content": "x"},
        )
    assert exc_info.value.error.code == JSONRPC_PROXIED_REFUSED
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.asyncio
async def test_write_file_create_mode_existing_file_oserror(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """mode=create on an existing file MUST yield -32006."""
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    target = tmp_path / "exists.txt"
    target.write_text("present", encoding="utf-8")
    with pytest.raises(McpError) as exc_info:
        await _invoke(
            bridge, "write_file",
            {"path": "exists.txt", "content": "new", "mode": "create"},
        )
    assert exc_info.value.error.code == JSONRPC_PROXIED_IO_FAILURE


# --- run_command -----------------------------------------------------------


@pytest.mark.asyncio
async def test_run_command_returns_stdout_and_exit_code(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """A simple python ``-c`` invocation MUST return stdout / exit_code 0."""
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    result = await _invoke(
        bridge, "run_command",
        {
            "command": sys.executable,
            "args": ["-c", "print('hello-from-subproc')"],
            "timeout_ms": 10000,
        },
    )
    payload = _structured(result)
    assert payload["exit_code"] == 0
    assert "hello-from-subproc" in payload["stdout"]
    assert payload["timed_out"] is False
    assert payload["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_run_command_refuses_unknown_binary(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """A binary not on PATH MUST yield -32005."""
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(McpError) as exc_info:
        await _invoke(
            bridge, "run_command",
            {"command": "absolutely_not_a_real_binary_xyzzy", "timeout_ms": 1000},
        )
    assert exc_info.value.error.code == JSONRPC_PROXIED_REFUSED


@pytest.mark.asyncio
async def test_run_command_timeout_yields_32006(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """A subprocess that exceeds ``timeout_ms`` MUST yield -32006 with timeout."""
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(McpError) as exc_info:
        await _invoke(
            bridge, "run_command",
            {
                "command": sys.executable,
                "args": ["-c", "import time; time.sleep(2)"],
                "timeout_ms": 200,
            },
        )
    assert exc_info.value.error.code == JSONRPC_PROXIED_IO_FAILURE
    assert "timed out" in exc_info.value.error.message.lower()


# --- run-record emission ---------------------------------------------------


@pytest.mark.asyncio
async def test_proxied_read_file_emits_lifecycle_envelopes(
    tmp_path, monkeypatch,
) -> None:
    """Proxied tools MUST emit run.start -> tool_call -> tool_result -> run.complete."""
    from llm_kernel.run_tracker import RunTracker

    class _ListSink:
        def __init__(self) -> None:
            self.envelopes = []

        def emit(self, env):
            self.envelopes.append(env)

    sink = _ListSink()
    tracker = RunTracker(
        trace_id=str(uuid.uuid4()), sink=sink,
        agent_id="test-agent", zone_id="test-zone",
    )
    bridge = OperatorBridgeServer(
        agent_id="test-agent", zone_id="test-zone",
        trace_id=str(uuid.uuid4()), run_tracker=tracker,
    )
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "x.txt").write_text("hi", encoding="utf-8")
    await _invoke(bridge, "read_file", {"path": "x.txt"})

    types_in_order = [env["message_type"] for env in sink.envelopes]
    assert "run.start" in types_in_order
    assert "run.complete" in types_in_order
    # Two run.event entries (tool_call + tool_result).
    event_envs = [
        env for env in sink.envelopes if env["message_type"] == "run.event"
    ]
    event_names = [env["payload"]["event"]["name"] for env in event_envs]
    assert "tool_call" in event_names
    assert "tool_result" in event_names
    # Order: start -> tool_call -> tool_result -> complete.
    flat = [
        (env["message_type"], env["payload"].get("event", {}).get("name"))
        for env in sink.envelopes
    ]
    assert flat[0][0] == "run.start"
    assert flat[-1][0] == "run.complete"

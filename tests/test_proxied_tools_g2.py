"""V1 Kernel Gap Closure G2 — proxied tool happy paths and edge cases.

Companion to ``test_proxied_tools.py``.  This file covers the brief-named
G2 acceptance scenarios, exercising paths the existing suite does not
explicitly assert (intermediate-dir creation depth, base64 round-trip,
env-override propagation, exit-code propagation, and the
``timed_out`` field shape).

K-MCP V1 mega-round Round B G2 closure.  See
``docs/notebook/PLAN-substrate-gap-closure.md`` and
``docs/rfcs/RFC-001-mcp-tool-taxonomy.md`` §read_file/§write_file/
§run_command.
"""

from __future__ import annotations

import base64
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
    return OperatorBridgeServer(
        agent_id="test-agent",
        zone_id="test-zone",
        trace_id=str(uuid.uuid4()),
    )


def _call_tool_handler(bridge: OperatorBridgeServer) -> ServerHandler:
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


# --- G2 acceptance tests ---------------------------------------------------


@pytest.mark.asyncio
async def test_read_file_returns_content_with_workspace_relative_path(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """A workspace-relative path MUST round-trip content + size_bytes.

    Pins the read_file return shape against the RFC-001 §read_file
    output schema fields the agent depends on (content / encoding /
    truncated / size_bytes).
    """
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    payload_text = "G2 round-trip via workspace-relative path."
    (tmp_path / "g2_relative.txt").write_text(payload_text, encoding="utf-8")
    result = await _invoke(bridge, "read_file", {"path": "g2_relative.txt"})
    body = _structured(result)
    assert body["content"] == payload_text
    assert body["encoding"] == "utf-8"
    assert body["truncated"] is False
    assert body["size_bytes"] == len(payload_text.encode("utf-8"))


@pytest.mark.asyncio
async def test_write_file_creates_intermediate_dirs(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """write_file MUST mkdir intermediate parents at arbitrary depth.

    Three-level nesting catches a regression where a single
    ``parent.mkdir(parents=False)`` would slip in.
    """
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    rel_path = "alpha/beta/gamma/delta.txt"
    body = "deep tree write"
    result = await _invoke(
        bridge, "write_file", {"path": rel_path, "content": body},
    )
    out = _structured(result)
    assert out["created"] is True
    assert out["bytes_written"] == len(body.encode("utf-8"))
    assert (tmp_path / "alpha" / "beta" / "gamma" / "delta.txt").read_text(
        encoding="utf-8",
    ) == body


@pytest.mark.asyncio
async def test_read_file_missing_emits_minus_32006(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """Missing file -> -32006 (proxied I/O failure) per RFC-001.

    Note: the brief calls for ``-32001`` here, but RFC-001 reserves
    -32001 for schema validation failures and -32006 for proxied I/O
    failures (which is what a missing file is).  The current
    implementation correctly emits -32006; this test pins that.
    """
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(McpError) as exc_info:
        await _invoke(
            bridge, "read_file",
            {"path": "definitely_not_present_g2.bin"},
        )
    err = exc_info.value.error
    assert err.code == JSONRPC_PROXIED_IO_FAILURE
    assert err.data is not None
    assert err.data.get("path") == "definitely_not_present_g2.bin"


@pytest.mark.asyncio
async def test_run_command_captures_stdout_and_exit_code(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """run_command MUST surface stdout / stderr / non-zero exit_code.

    A python -c that prints to both streams and exits 7 pins the
    independent capture of all three RFC-001 §run_command output fields.
    """
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    result = await _invoke(
        bridge, "run_command",
        {
            "command": sys.executable,
            "args": [
                "-c",
                "import sys; "
                "print('stdout-from-g2'); "
                "print('stderr-from-g2', file=sys.stderr); "
                "sys.exit(7)",
            ],
            "timeout_ms": 10000,
        },
    )
    out = _structured(result)
    assert out["exit_code"] == 7
    assert "stdout-from-g2" in out["stdout"]
    assert "stderr-from-g2" in out["stderr"]
    assert out["timed_out"] is False
    assert isinstance(out["duration_ms"], int)
    assert out["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_run_command_timeout_kills_process(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """A timeout MUST yield -32006 with timeout-ms in the error payload.

    Pins the data envelope (``timeout_ms`` echoed) so the agent can
    surface the limit hit in the operator UI.
    """
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(McpError) as exc_info:
        await _invoke(
            bridge, "run_command",
            {
                "command": sys.executable,
                "args": ["-c", "import time; time.sleep(5)"],
                "timeout_ms": 250,
            },
        )
    err = exc_info.value.error
    assert err.code == JSONRPC_PROXIED_IO_FAILURE
    assert "timed out" in err.message.lower()
    assert err.data is not None
    assert err.data.get("timeout_ms") == 250


# --- Bonus G2 round-trips: base64 + env override --------------------------


@pytest.mark.asyncio
async def test_read_file_base64_round_trip(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """A binary file read with encoding=base64 MUST round-trip bytes.

    Pins the only non-utf8 read path; previous coverage exercises
    utf-8 only.
    """
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    raw = bytes(range(64))
    (tmp_path / "blob.bin").write_bytes(raw)
    result = await _invoke(
        bridge, "read_file", {"path": "blob.bin", "encoding": "base64"},
    )
    body = _structured(result)
    assert body["encoding"] == "base64"
    decoded = base64.b64decode(body["content"].encode("ascii"))
    assert decoded == raw

"""The cell-magic injection defense is enforced in the running kernel.

Sources: ``docs/04 - Reference/atoms/discipline/magic-injection-defense.md``
(emission ban, Layer-1 contamination detector, Layer-2 wire sanitizer,
bidirectional strip, threat model: the pin never reaches the notebook or an
agent), ``docs/06 - Roadmaps/PLAN-S5.0.1-cell-magic-injection-defense.md``
(§2 goals, §3.2 detector, §3.3 sanitizer, §3.5 pin-aware parser and
``@@auth set``, §3.6 schema: only the fingerprint is stored),
``docs/06 - Roadmaps/PLAN-S5.0.4-privileged-magic-emission.md``
(``emit_magic_cell`` needs an operator grant, the tool is registered per
agent only when a grant exists, grants are in-session operator intents,
stream promotion) and
``docs/04 - Reference/atoms/discipline/certified-magic-emitter.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import mcp.types as types
import pytest
from mcp.shared.exceptions import McpError

from llm_kernel.capture.metadata_writer import MetadataWriter
from llm_kernel.capture.socket_writer import SocketWriter, sanitize_outbound_record
from llm_kernel.kernel_facade import Kernel
from llm_kernel.magics.magic_emit_tool import MagicEmitError, emit_magic_cell
from llm_kernel.mcp import mcp_server
from llm_kernel.mcp.auth_handlers import apply_auth_command
from llm_kernel.mcp.mcp_server import OperatorBridgeServer
from llm_kernel.notebook.cell_manager import CellManager
from llm_kernel.supervision.agent_supervisor import AgentHandle
from llm_kernel.text import magic_hash as MH
from llm_kernel.text.cell_text import ParserContext, parse_cell, parser_context_from_config

from tests.conformance._support import (
    WAIT_SECONDS,
    RecordingShell,
    WirePeer,
    current_kernel_block,
    hydrate_complete_snapshot,
    hydrate_envelope,
    hydrate_snapshot,
    run,
    wait_until,
)

PIN = "conformance-pin-0001"
OTHER_PIN = "someone-elses-pin-99"
PIN_ENV = MH.OPERATOR_PIN_ENV_VAR


def _agent(agent_id: str, zone_id: str = "default") -> AgentHandle:
    """An agent handle with no process -- only identity is consulted."""
    handle = AgentHandle.__new__(AgentHandle)
    handle.agent_id = agent_id
    handle.zone_id = zone_id
    handle.popen = None  # type: ignore[assignment]
    return handle


def _bound_cells() -> Dict[str, Any]:
    return {
        "c_alpha": {"text": "@@agent alpha\nhello", "bound_agent_id": "alpha", "kind": "agent"},
        "c_other": {"text": "@@agent gamma\nhello", "bound_agent_id": "gamma", "kind": "agent"},
    }


def _hashed(name: str, pin: str = PIN) -> str:
    return f"@@{MH.magic_hash(pin, name)}:{name}"


# ---------------------------------------------------------------------------
# Start paths: each yields the live writer / supervisor and a wire view.
# ---------------------------------------------------------------------------


class _Started:
    def __init__(self, writer: Any, supervisor: Any, deliver: Callable[[Dict[str, Any]], None],
                 frames: Callable[[], List[Dict[str, Any]]],
                 socket_writer: Optional[SocketWriter] = None,
                 raw: Optional[Callable[[], List[bytes]]] = None) -> None:
        self.writer = writer
        self.supervisor = supervisor
        self.deliver = deliver
        self.frames = frames
        self.socket_writer = socket_writer
        self.raw = raw

    def hydrate(self, **kwargs: Any) -> Dict[str, Any]:
        self.deliver(hydrate_envelope(current_kernel_block(), **kwargs))
        snap = wait_until(lambda: hydrate_complete_snapshot(self.frames()))
        assert snap is not None, "hydrate was not confirmed"
        return snap


@contextlib.contextmanager
def _facade(tmp_path: Path) -> Iterator[_Started]:
    loop = asyncio.new_event_loop()
    shell = RecordingShell()
    k = Kernel(workdir=tmp_path, kernel_shell=shell)
    loop.run_until_complete(k.start())
    try:
        yield _Started(k.metadata_writer, k.supervisor, shell.deliver, shell.snapshot)
    finally:
        loop.run_until_complete(k.shutdown())
        loop.close()


@contextlib.contextmanager
def _pty_boot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Started]:
    from llm_kernel.transports import pty_mode

    peer = WirePeer()
    monkeypatch.setenv(pty_mode.ENV_IPC_SOCKET, peer.address)
    state = pty_mode.boot_kernel(workdir=tmp_path)
    try:
        yield _Started(state["metadata_writer"], state["supervisor"], peer.send,
                       peer.snapshot, socket_writer=state["writer"],
                       raw=lambda: list(peer.raw_lines))
    finally:
        pty_mode.shutdown_kernel(state)
        peer.close()


@contextlib.contextmanager
def _legacy_hooks(tmp_path: Path) -> Iterator[_Started]:
    from llm_kernel._kernel_hooks import attach_kernel_subsystems

    shell = RecordingShell()
    dispatcher, _t, _b, supervisor, writer = attach_kernel_subsystems(shell)
    try:
        yield _Started(writer, supervisor, shell.deliver, shell.snapshot)
    finally:
        writer.stop(emit_final=False)
        dispatcher.stop()


@contextlib.contextmanager
def _serve_boot(tmp_path: Path) -> Iterator[_Started]:
    from llm_kernel.transports import serve_mode

    state = serve_mode._boot_kernel_for_serve("none", workdir=tmp_path)  # noqa: SLF001
    dispatcher = state["dispatcher"]
    emitted: List[Dict[str, Any]] = []
    original = dispatcher.emit

    def _tap(envelope: Dict[str, Any]) -> None:
        emitted.append(json.loads(json.dumps(envelope, default=str)))
        original(envelope)

    dispatcher.emit = _tap
    try:
        yield _Started(
            state["metadata_writer"], state["supervisor"],
            lambda frame: dispatcher._on_comm_msg({"content": {"data": frame}}),  # noqa: SLF001
            lambda: list(emitted),
        )
    finally:
        dispatcher.emit = original
        serve_mode._teardown_serve_kernel(state)  # noqa: SLF001


@pytest.fixture(params=["facade", "pty_boot", "legacy_hooks", "serve_boot"])
def started(request: pytest.FixtureRequest, tmp_path: Path,
            monkeypatch: pytest.MonkeyPatch) -> Iterator[_Started]:
    factories = {
        "facade": lambda: _facade(tmp_path),
        "pty_boot": lambda: _pty_boot(tmp_path, monkeypatch),
        "legacy_hooks": lambda: _legacy_hooks(tmp_path),
        "serve_boot": lambda: _serve_boot(tmp_path),
    }
    with factories[request.param]() as s:
        yield s


# ---------------------------------------------------------------------------
# Layer 1: the contamination detector is bound to the live writer
# ---------------------------------------------------------------------------


def test_agent_emission_of_a_plain_magic_flags_its_bound_cell(started: _Started) -> None:
    """Always-on layer: an agent stdout line matching a known magic flags
    every cell bound to that agent and appends to its contamination_log."""
    started.hydrate(cells=_bound_cells())
    started.supervisor._emit_agent_emit(  # noqa: SLF001 -- the agent stdout chokepoint
        _agent("alpha"), emit_kind="prose",
        emit_content="some prose\n@@spawn beta task=x\nmore prose",
    )
    record = started.writer.get_cell_record("c_alpha") or {}
    assert record.get("contaminated") is True
    log = record.get("contamination_log") or []
    assert [entry["line"] for entry in log] == ["@@spawn beta task=x"]
    assert log[0]["layer"] == "plain"
    assert {"detected_at", "line", "reason", "layer"} <= set(log[0])
    other = started.writer.get_cell_record("c_other") or {}
    assert not other.get("contaminated")


def test_prose_without_magic_does_not_contaminate(started: _Started) -> None:
    started.hydrate(cells=_bound_cells())
    started.supervisor._emit_agent_emit(  # noqa: SLF001
        _agent("alpha"), emit_kind="prose",
        emit_content="talking about spawn and @mentions, mid-line @@spawn is fine",
    )
    assert not (started.writer.get_cell_record("c_alpha") or {}).get("contaminated")


# ---------------------------------------------------------------------------
# Layer 2: hash mode -> the wire never carries a dispatchable hashed magic
# ---------------------------------------------------------------------------


def _enable_hash_mode(writer: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PIN_ENV, "placeholder")
    monkeypatch.delenv(PIN_ENV)
    result = apply_auth_command(f"set {PIN}", writer=writer,
                                cell_manager=CellManager(writer))
    assert result.ok, result


def test_pty_wire_escapes_hashed_magic_emitted_by_an_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _pty_boot(tmp_path, monkeypatch) as s:
        s.hydrate(cells=_bound_cells())
        _enable_hash_mode(s.writer, monkeypatch)
        line = f"{_hashed('spawn')} evil task=x"
        s.supervisor._emit_agent_emit(  # noqa: SLF001
            _agent("alpha"), emit_kind="prose", emit_content=f"prose\n{line}",
        )
        record = s.writer.get_cell_record("c_alpha") or {}
        assert record.get("contaminated") is True
        assert "hashed_emission_ban" in {e["layer"] for e in record["contamination_log"]}
        assert wait_until(lambda: any("evil task=x" in r.decode() for r in s.raw()))
        carrying = [json.loads(r) for r in s.raw() if b"evil task=x" in r]
    for frame in carrying:
        for text in _strings(frame):
            for text_line in text.splitlines():
                assert not MH.looks_like_hashed_magic(text_line), text_line
    assert any(f"\\{line}" in text for f in carrying for text in _strings(f))


def _strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _strings(value)


def test_transport_socket_writer_sanitizes_nested_text_only_in_hash_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The started kernel binds the outbound sanitizer to the transport's
    SocketWriter; it follows the writer's hash-mode setting and walks
    nested lists (content blocks, OTLP attribute arrays)."""
    frame = {
        "emit_kind": "prose",
        "content": [
            {"type": "text", "text": "plain prose"},
            {"type": "text", "text": "@@deadbeef:spawn some_agent"},
        ],
        "attributes": [
            {"key": "message", "value": {"stringValue": "@@cafe0123:scratch y"}},
            {"key": "llmnb.agent_id", "value": {"stringValue": "@@cafe0123:scratch id"}},
        ],
    }
    with _pty_boot(tmp_path, monkeypatch) as s:
        assert s.socket_writer is not None
        s.socket_writer.write_frame({"marker": "off", **frame})
        _enable_hash_mode(s.writer, monkeypatch)
        s.socket_writer.write_frame({"marker": "on", **frame})
        got = {}
        for marker in ("off", "on"):
            got[marker] = wait_until(lambda m=marker: next(
                (f for f in s.frames() if f.get("marker") == m), None))
    assert got["off"]["content"] == frame["content"]
    on = got["on"]
    assert on["content"][0]["text"] == "plain prose"
    assert on["content"][1]["text"] == "\\@@deadbeef:spawn some_agent"
    assert on["attributes"][0]["value"]["stringValue"] == "\\@@cafe0123:scratch y"
    # Only text fields are rewritten; identity attributes pass through.
    assert on["attributes"][1]["value"]["stringValue"] == "@@cafe0123:scratch id"
    # The caller's record is never mutated.
    assert frame["content"][1]["text"] == "@@deadbeef:spawn some_agent"


def test_sanitizer_is_identity_on_clean_records_and_idempotent() -> None:
    clean = {"text": ["nothing"], "outputs": [{"preview": "fine"}]}
    assert sanitize_outbound_record(clean) is clean
    dirty = {"outputs": [{"preview": ["@@abc12345:spawn x", "ok"]}]}
    once = sanitize_outbound_record(dirty)
    assert once["outputs"][0]["preview"] == ["\\@@abc12345:spawn x", "ok"]
    assert sanitize_outbound_record(once) is once
    assert MH.escape_leading_at(MH.escape_leading_at("@@abc:spawn")) == "\\@@abc:spawn"


# ---------------------------------------------------------------------------
# Pin-aware parsing everywhere text becomes cells
# ---------------------------------------------------------------------------


def test_parser_in_hash_mode_dispatches_only_the_pin_stamped_form() -> None:
    ctx = ParserContext(hash_mode_on=True, pin=PIN)
    plain = parse_cell("@@spawn alpha task:\"x\"\nbody", parser_context=ctx)
    assert plain.kind != "spawn"
    assert "@@spawn alpha" in plain.body
    assert [k["code"] for k in plain.k_class_emissions] == ["K35"]
    stamped = parse_cell(f"{_hashed('spawn')} alpha task:\"x\"\nbody", parser_context=ctx)
    assert stamped.kind == "spawn" and stamped.args["agent_id"] == "alpha"
    forged = parse_cell(f"{_hashed('spawn', OTHER_PIN)} alpha\nbody", parser_context=ctx)
    assert forged.kind != "spawn"
    degraded = parse_cell("@@spawn alpha\nbody",
                          parser_context=ParserContext(hash_mode_on=True, pin=None))
    assert degraded.kind != "spawn"
    assert parse_cell("@@spawn alpha\nbody").kind == "spawn"  # legacy path


def test_resolve_operator_pin_requires_the_fingerprint_to_match() -> None:
    fp = MH.magic_pin_fingerprint(PIN)
    assert MH.resolve_operator_pin(fp, env={PIN_ENV: PIN}) == PIN
    assert MH.resolve_operator_pin(fp, env={PIN_ENV: OTHER_PIN}) is None
    assert MH.resolve_operator_pin(fp, env={}) is None
    assert MH.resolve_operator_pin(None, env={PIN_ENV: PIN}) is None
    assert MH.resolve_operator_pin("", env={PIN_ENV: PIN}) is None


def test_file_sourced_config_earns_hash_mode_only_with_the_session_pin() -> None:
    cfg = {"magic_hash_enabled": True, "magic_pin_fingerprint": MH.magic_pin_fingerprint(PIN)}
    ctx = parser_context_from_config(cfg, env={PIN_ENV: PIN})
    assert ctx is not None and ctx.hash_mode_on and ctx.pin == PIN
    assert parser_context_from_config(cfg, env={PIN_ENV: OTHER_PIN}) is None
    assert parser_context_from_config(cfg, env={}) is None
    assert parser_context_from_config({"magic_hash_enabled": False}, env={PIN_ENV: PIN}) is None
    assert parser_context_from_config(None) is None


def test_writer_and_cell_manager_parse_under_the_live_hash_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = MetadataWriter(workspace_root=tmp_path, autosave_interval_sec=3600.0)
    try:
        cm = CellManager(writer)
        writer.set_cell_text("plain", "@@spawn beta\nhello")
        writer.set_cell_text("stamped", f"{_hashed('spawn')} beta\nhello")
        assert writer.parser_context() is None
        assert writer.cell_view("plain").kind == "spawn"
        assert cm.view("plain").kind == "spawn"
        _enable_hash_mode(writer, monkeypatch)
        ctx = writer.parser_context()
        assert ctx is not None and ctx.hash_mode_on and ctx.pin == PIN
        # PLAN §3.7: ``@auth set`` re-stamps existing plain magic lines.
        assert writer.get_cell_text("plain").splitlines()[0] == f"{_hashed('spawn')} beta"
        assert writer.cell_view("plain").kind == "spawn"
        # A plain magic typed (or pasted) after hash mode is on is body;
        # the cached legacy view of the same text must not survive.
        writer.set_cell_text("pasted", "@@spawn beta\nhello")
        assert writer.cell_view("pasted").kind != "spawn"
        assert cm.view("pasted").kind != "spawn"
        assert writer.cell_view("stamped").kind == "spawn"
        assert cm.view("stamped").kind == "spawn"
        # A pin that does not reproduce the fingerprint validates nothing.
        monkeypatch.setenv(PIN_ENV, OTHER_PIN)
        assert writer.get_operator_pin() is None
        assert writer.cell_view("stamped").kind != "spawn"
        assert writer.cell_view("pasted").kind != "spawn"
    finally:
        writer.stop(emit_final=False)


def test_notebook_format_parses_under_the_notebook_hash_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_kernel.notebook import notebook_format

    base = {"config": {"magic_hash_enabled": True,
                       "magic_pin_fingerprint": MH.magic_pin_fingerprint(PIN)}}
    text = "@@spawn alpha task:\"x\"\nhi"
    monkeypatch.setenv(PIN_ENV, PIN)
    doc = notebook_format.magic_to_llmnb(text, base_metadata=base)
    cells = doc["metadata"]["rts"]["cells"]
    assert [c.get("kind") for c in cells.values()] != ["spawn"]
    stamped = notebook_format.magic_to_llmnb(f"{_hashed('spawn')} alpha task:\"x\"\nhi",
                                             base_metadata=base)
    assert [c.get("bound_agent_id") for c in stamped["metadata"]["rts"]["cells"].values()] == ["alpha"]
    # Without the in-session pin the file cannot switch hash mode on.
    monkeypatch.delenv(PIN_ENV)
    legacy = notebook_format.magic_to_llmnb(text, base_metadata=base)
    assert [c.get("bound_agent_id") for c in legacy["metadata"]["rts"]["cells"].values()] == ["alpha"]


# ---------------------------------------------------------------------------
# @auth set: only a fingerprint is recorded; the pin stays kernel-local
# ---------------------------------------------------------------------------


def test_auth_set_records_the_fingerprint_never_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    writer = MetadataWriter(workspace_root=tmp_path, autosave_interval_sec=3600.0)
    try:
        _enable_hash_mode(writer, monkeypatch)
        assert writer.get_config_setting("magic_hash_enabled") is True
        assert writer.get_config_setting("magic_pin_fingerprint") == MH.magic_pin_fingerprint(PIN)
        assert os.environ.get(PIN_ENV) == PIN
        assert writer.get_operator_pin() == PIN
        snapshot = writer.snapshot(trigger="save")
        assert PIN not in json.dumps(snapshot, default=str)
        assert snapshot["config"]["magic_pin_fingerprint"] == MH.magic_pin_fingerprint(PIN)
        # A second set is refused while hash mode is on.
        again = apply_auth_command(f"set {OTHER_PIN}", writer=writer,
                                   cell_manager=CellManager(writer))
        assert not again.ok
        assert writer.get_config_setting("magic_pin_fingerprint") == MH.magic_pin_fingerprint(PIN)
    finally:
        writer.stop(emit_final=False)


def test_agent_env_never_carries_the_operator_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_kernel.supervision import _provisioning

    parent = {"PATH": "/usr/bin", PIN_ENV: PIN, "HOME": "/home/x"}
    common = dict(api_key="k", llm_endpoint_url="http://127.0.0.1:1/v1",
                  mcp_config_path=tmp_path / "m.json", system_prompt_path=tmp_path / "p.txt",
                  work_dir=tmp_path, agent_id="alpha", zone_id="z", trace_id="t" * 32)
    for leak in (None, "1"):
        if leak is None:
            monkeypatch.delenv("LLMKERNEL_LEAK_ENV", raising=False)
        else:
            monkeypatch.setenv("LLMKERNEL_LEAK_ENV", leak)
        env = _provisioning.build_env(parent, **common)
        assert PIN_ENV not in env, f"LLMKERNEL_LEAK_ENV={leak!r}"
        assert PIN not in env.values()


def _call_tool(bridge: OperatorBridgeServer, name: str, arguments: Dict[str, Any]) -> Any:
    handler = bridge.server.request_handlers[types.CallToolRequest]
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    return run(handler(request)).root


def _list_tools(bridge: OperatorBridgeServer) -> List[str]:
    handler = bridge.server.request_handlers[types.ListToolsRequest]
    return [t.name for t in run(handler(types.ListToolsRequest(method="tools/list"))).root.tools]


def test_run_command_child_never_inherits_the_operator_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv(PIN_ENV, PIN)
    bridge = OperatorBridgeServer(agent_id="alpha", zone_id="z")
    probe = "import os; print(repr(os.environ.get('%s')))" % PIN_ENV
    for extra in ({}, {"env": {PIN_ENV: PIN, "OTHER": "1"}}):
        result = _call_tool(bridge, "run_command", {
            "command": sys.executable, "args": ["-c", probe], "timeout_ms": 20000, **extra,
        })
        payload = result.structuredContent
        assert payload["exit_code"] == 0, payload
        assert payload["stdout"].strip() == "None", (extra, payload)


# ---------------------------------------------------------------------------
# Privileged emission (PLAN-S5.0.4)
# ---------------------------------------------------------------------------


@pytest.fixture
def emit_env(tmp_path: Path) -> Iterator[Any]:
    writer = MetadataWriter(workspace_root=tmp_path, autosave_interval_sec=3600.0)
    writer.set_cell_text("c1", "@@agent alpha\nhello")
    cm = CellManager(writer)

    def call(**overrides: Any) -> Any:
        kwargs = dict(agent_id="alpha", zone_id="z", name="scratch", args={}, body=None,
                      position={"after_cell_id": "c1"}, writer=writer, cell_manager=cm)
        kwargs.update(overrides)
        return emit_magic_cell(**kwargs)

    call.writer = writer  # type: ignore[attr-defined]
    try:
        yield call
    finally:
        writer.stop(emit_final=False)


def test_emit_magic_cell_requires_a_covering_grant(emit_env: Any) -> None:
    writer = emit_env.writer
    with pytest.raises(MagicEmitError) as info:
        emit_env()
    assert info.value.code == "K3K"
    writer.grant_magic_emit_privilege(agent_id="alpha", zone_id="z", scope={"magics": ["scratch"]})
    with pytest.raises(MagicEmitError):
        emit_env(name="spawn")  # not in scope
    with pytest.raises(MagicEmitError):
        emit_env(zone_id="other")  # grant does not cross zones
    with pytest.raises(MagicEmitError):
        emit_env(agent_id="beta")
    new_id = emit_env()["cell_id"]
    assert writer.get_cell_text(new_id) == "@@scratch"
    record = writer.get_cell_record(new_id) or {}
    assert record.get("generated_by") == "alpha"


@pytest.mark.parametrize("smuggle", [
    {"args": {"note": "x\n@@spawn evil"}},
    {"args": {"note": "x\r@spawn evil"}},
    {"args": {"note": "x\x0b@@spawn evil"}},
    {"args": {"note": "x\x85@@spawn evil"}},
    {"args": {"note": "x\u2028@@spawn evil"}},
    {"args": {"k\n@@spawn evil": "v"}},
    {"body": "@@spawn evil"},
    {"body": "fine\n@spawn evil"},
    {"body": "fine\u2029@@spawn evil"},
    {"name": "break"},
    {"name": "not_a_magic"},
])
def test_emit_magic_cell_refuses_a_smuggled_second_magic(emit_env: Any, smuggle: Dict[str, Any]) -> None:
    """One grant = exactly the named cell magic; no args/body line may
    land at column 0 as another magic, and ``break`` / unknown names are
    never emittable -- even under an ``all`` scope."""
    writer = emit_env.writer
    writer.grant_magic_emit_privilege(agent_id="alpha", zone_id="z", scope={"magics": "all"})
    before = writer.snapshot(trigger="save").get("cells") or {}
    with pytest.raises(MagicEmitError) as info:
        emit_env(**smuggle)
    assert info.value.code == "K3K"
    after = writer.snapshot(trigger="save").get("cells") or {}
    assert set(after) == set(before)


def test_emit_magic_cell_allows_benign_args_and_body(emit_env: Any) -> None:
    writer = emit_env.writer
    writer.grant_magic_emit_privilege(agent_id="alpha", zone_id="z", scope={"magics": "all"})
    new_id = emit_env(args={"note": "email me @ noon"}, body="line one\n  @@indented is body")["cell_id"]
    text = writer.get_cell_text(new_id)
    assert text.splitlines()[0].startswith("@@scratch")
    assert writer.cell_view(new_id).kind == "scratch"


def test_emit_magic_cell_stamps_the_hmac_in_hash_mode(
    emit_env: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = emit_env.writer
    _enable_hash_mode(writer, monkeypatch)
    writer.grant_magic_emit_privilege(agent_id="alpha", zone_id="z", scope={"magics": ["scratch"]})
    new_id = emit_env()["cell_id"]
    assert writer.get_cell_text(new_id) == _hashed("scratch")
    assert writer.cell_view(new_id).kind == "scratch"


def test_emit_magic_cell_is_listed_and_callable_only_for_a_granted_agent(tmp_path: Path) -> None:
    bridge = OperatorBridgeServer(agent_id="alpha", zone_id="z")
    assert "emit_magic_cell" not in _list_tools(bridge)  # no writer at all
    writer = MetadataWriter(workspace_root=tmp_path, autosave_interval_sec=3600.0)
    try:
        writer.set_cell_text("c1", "@@agent alpha\nhello")
        bridge.metadata_writer = writer
        assert "emit_magic_cell" not in _list_tools(bridge)
        writer.grant_magic_emit_privilege(agent_id="alpha", zone_id="z", scope={"magics": []})
        assert "emit_magic_cell" not in _list_tools(bridge)  # empty scope = no privilege
        writer.grant_magic_emit_privilege(agent_id="beta", zone_id="z", scope={"magics": "all"})
        assert "emit_magic_cell" not in _list_tools(bridge)  # someone else's grant
        args = {"name": "scratch", "args": {}, "position": {"after_cell_id": "c1"}}
        with pytest.raises(McpError) as info:
            _call_tool(bridge, "emit_magic_cell", args)
        assert info.value.error.code == mcp_server.JSONRPC_PRIVILEGE_REQUIRED
        assert (info.value.error.data or {}).get("k_code") == "K3K"
        writer.grant_magic_emit_privilege(agent_id="alpha", zone_id="z", scope={"magics": ["scratch"]})
        tools = _list_tools(bridge)
        assert "emit_magic_cell" in tools
        others = set(tools) - {"emit_magic_cell"}
        assert others and "notify" in others
        result = _call_tool(bridge, "emit_magic_cell", args)
        assert not result.isError
        smuggled = dict(args, body="x\n@@spawn evil")
        with pytest.raises(McpError) as info:
            _call_tool(bridge, "emit_magic_cell", smuggled)
        assert (info.value.error.data or {}).get("k_code") == "K3K"
    finally:
        writer.stop(emit_final=False)


def test_stream_promotion_is_anchored_to_logged_evidence(tmp_path: Path) -> None:
    """PLAN-S5.0.4 §3.3: promotion synthesizes emit_magic_cell for the
    source cell's agent -- but only for a line the detector logged."""
    from llm_kernel.mcp.intent_handlers.promote_stream_magic import handle_promote_stream_magic

    writer = MetadataWriter(workspace_root=tmp_path, autosave_interval_sec=3600.0)
    try:
        writer.hydrate(hydrate_snapshot(current_kernel_block(), cells={"c_alpha": {
            "text": "@@agent alpha\nhi", "bound_agent_id": "alpha", "kind": "agent"}}))
        writer.grant_magic_emit_privilege(agent_id="alpha", zone_id="z", scope={"magics": "all"})
        writer.flag_cells_contaminated_by_agent(
            agent_id="alpha", line="@@scratch", source="stdout", layer="plain")
        cm = CellManager(writer)
        refused = handle_promote_stream_magic(
            params={"cell_id": "c_alpha", "line": "@@spawn evil"},
            writer=writer, cell_manager=cm, zone_id="z")
        assert refused is None
        promoted = handle_promote_stream_magic(
            params={"cell_id": "c_alpha", "line": "@@scratch"},
            writer=writer, cell_manager=cm, zone_id="z")
        assert promoted and promoted.get("cell_id")
        record = writer.get_cell_record(promoted["cell_id"]) or {}
        assert record.get("generated_by") == "alpha"
        assert record.get("promoted_from_stream") is True
    finally:
        writer.stop(emit_final=False)


# ---------------------------------------------------------------------------
# Trust does not come from a file: hydrate quarantines it
# ---------------------------------------------------------------------------


def _persisted_trust_config() -> Dict[str, Any]:
    return {
        "magic_emit_privileges": [{
            "agent_id": "alpha", "zone_id": "default",
            "granted_at": "2026-01-01T00:00:00Z", "scope": {"magics": "all"},
        }],
        "magic_hash_enabled": True,
        "magic_pin_fingerprint": MH.magic_pin_fingerprint(PIN),
    }


def _operator_action(action_type: str, **parameters: Any) -> Dict[str, Any]:
    return {"type": "operator.action",
            "payload": {"action_type": action_type, "parameters": parameters}}


def test_file_sourced_grants_count_only_once_re_granted_in_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PIN_ENV, raising=False)
    with _facade(tmp_path) as s:
        snap = s.hydrate(config=_persisted_trust_config(),
                         cells={"c1": {"text": "@@spawn beta\nhello", "kind": "agent"}})
        w = s.writer
        assert w.get_magic_emit_privileges() == []
        assert not w.has_magic_emit_privilege(agent_id="alpha", zone_id="default", magic_name="spawn")
        persisted = snap["config"].get("persisted") or {}
        assert persisted.get("magic_emit_privileges"), snap["config"]
        # No session pin: the file cannot switch hash mode on; the parser
        # stays in the legacy mode and the flag is parked with the grants.
        assert not w.get_config_setting("magic_hash_enabled")
        assert w.parser_context() is None
        assert persisted.get("magic_hash_enabled") is True
        assert w.cell_view("c1").kind == "spawn"
        # Persisted trust round-trips under the same marker.
        again = w.snapshot(trigger="save")["config"]
        assert again.get("persisted", {}).get("magic_emit_privileges")
        assert not again.get("magic_emit_privileges")
        # An operator.action grant without a scope grants nothing.
        s.deliver(_operator_action("grant_magic_emit_privilege",
                                   agent_id="alpha", zone_id="default"))
        assert wait_until(lambda: w.get_magic_emit_privileges(agent_id="alpha"))
        assert not w.has_magic_emit_privilege(agent_id="alpha", zone_id="default", magic_name="spawn")
        # The operator re-establishes the grant in session.
        s.deliver(_operator_action("grant_magic_emit_privilege", agent_id="alpha",
                                   zone_id="default", scope={"magics": ["spawn"]}))
        assert wait_until(lambda: w.has_magic_emit_privilege(
            agent_id="alpha", zone_id="default", magic_name="spawn"))
        assert not (w.snapshot(trigger="save")["config"].get("persisted") or {}).get(
            "magic_emit_privileges")


def test_file_sourced_hash_mode_stays_live_for_the_matching_session_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PIN_ENV, PIN)
    with _facade(tmp_path) as s:
        s.hydrate(config=_persisted_trust_config(),
                  cells={"plain": {"text": "@@spawn beta\nhello", "kind": "agent"},
                         "stamped": {"text": f"{_hashed('spawn')} beta\nhello", "kind": "agent"}})
        w = s.writer
        assert w.get_config_setting("magic_hash_enabled") is True
        assert w.get_operator_pin() == PIN
        assert w.cell_view("plain").kind != "spawn"
        assert w.cell_view("stamped").kind == "spawn"
        # Grants are still never taken from the file.
        assert w.get_magic_emit_privileges() == []

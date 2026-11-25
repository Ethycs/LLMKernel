"""Production wiring of the cell-magic injection defense (PLAN-S5.0.1).

The library pieces (Layer-1 contamination detector, outbound hashed-magic
sanitizer, pin-aware parser) shipped with unit tests but nothing bound
them to the live metadata writer. These tests pin the wiring: a started
kernel (facade, and the shared ``wire_magic_defense`` call the pty-mode
boot paths make) leaves the supervisor holding the writer and the
transport's ``SocketWriter`` holding a hash-mode provider, and shows the
two consequences that follow.
"""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path

from llm_kernel._kernel_hooks import wire_magic_defense
from llm_kernel.capture.metadata_writer import MetadataWriter
from llm_kernel.capture.socket_writer import SocketWriter, sanitize_outbound_record
from llm_kernel.kernel_facade import Kernel
from llm_kernel.supervision.agent_supervisor import AgentHandle
from llm_kernel.text.magic_hash import magic_pin_fingerprint
from llm_kernel.transports.pty_mode import _PtyKernel


def _stub_handle(agent_id: str, zone_id: str) -> AgentHandle:
    handle = AgentHandle.__new__(AgentHandle)
    handle.agent_id = agent_id
    handle.zone_id = zone_id
    handle.popen = None  # type: ignore[assignment]
    return handle


def _bind_cell(writer: MetadataWriter, cell_id: str, text: str, agent_id: str) -> None:
    writer.set_cell_text(cell_id, text)
    record = dict(writer.get_cell_record(cell_id) or {})
    record["bound_agent_id"] = agent_id
    writer._cells[cell_id] = record  # noqa: SLF001 -- test seam, as in sibling tests


def test_wire_magic_defense_binds_supervisor_and_socket_provider(tmp_path: Path) -> None:
    """The shared boot-path call wires both collaborators to the live writer."""

    class _Supervisor:
        writer = None

        def set_metadata_writer(self, writer):  # noqa: ANN001
            self.writer = writer

    writer = MetadataWriter(workspace_root=tmp_path)
    supervisor = _Supervisor()
    socket_writer = SocketWriter()
    wire_magic_defense(
        supervisor=supervisor, metadata_writer=writer, socket_writer=socket_writer,
    )
    assert supervisor.writer is writer
    provider = socket_writer._hash_mode_provider  # noqa: SLF001
    assert provider is not None and provider() is False
    writer.set_hash_mode(True, magic_pin_fingerprint("wiring-pin-1234"))
    assert provider() is True
    # No transport writer (embedded sink): still wires the supervisor.
    wire_magic_defense(supervisor=supervisor, metadata_writer=writer)
    assert supervisor.writer is writer


def test_facade_start_wires_supervisor_writer_and_socket_hash_provider(
    tmp_path: Path, monkeypatch,
) -> None:
    """``Kernel.start()`` with a pty-shaped shell wires both defenses."""
    socket_writer = SocketWriter()  # unconnected: frames are dropped, no socket needed
    shell = _PtyKernel(socket_writer)

    async def main() -> None:
        k = Kernel(workdir=tmp_path, kernel_shell=shell)
        await k.start()
        try:
            writer = k._metadata_writer  # noqa: SLF001
            assert k._supervisor._metadata_writer is writer  # noqa: SLF001
            provider = socket_writer._hash_mode_provider  # noqa: SLF001
            assert provider is not None
            assert provider() is False
            pin = "wiring-pin-1234"
            monkeypatch.setenv("LLMNB_OPERATOR_PIN", pin)
            writer.set_hash_mode(True, magic_pin_fingerprint(pin))
            assert provider() is True
        finally:
            await k.shutdown()

    asyncio.run(main())


def test_facade_started_kernel_flags_bound_cell_on_magic_emission(tmp_path: Path) -> None:
    """Consequence (a): an agent emission carrying a magic line lands in
    the bound cell's ``contamination_log`` on a facade-started kernel."""

    async def main() -> None:
        k = Kernel(workdir=tmp_path)
        await k.start()
        try:
            writer = k._metadata_writer  # noqa: SLF001
            _bind_cell(writer, "c_alpha", "@@agent alpha\nhello", "alpha")
            supervisor = k._supervisor  # noqa: SLF001
            assert supervisor._metadata_writer is writer  # noqa: SLF001
            supervisor._scan_and_rewrite_emit_content(  # noqa: SLF001
                _stub_handle("alpha", "default"),
                "some prose\n@@spawn beta task=x\nmore prose",
                source="stdout",
            )
            record = writer.get_cell_record("c_alpha") or {}
            assert record.get("contaminated") is True
            log = record.get("contamination_log") or []
            assert [entry["line"] for entry in log] == ["@@spawn beta task=x"]
            assert log[0]["layer"] == "plain"
        finally:
            await k.shutdown()

    asyncio.run(main())


def test_hash_mode_escapes_hashed_magic_nested_in_list_on_the_wire(tmp_path: Path) -> None:
    """Consequence (b): with hash mode on, a hashed magic inside a
    ``content`` list of text blocks is escaped before it hits the wire."""
    writer = MetadataWriter(workspace_root=tmp_path)
    writer.set_hash_mode(True, magic_pin_fingerprint("wiring-pin-1234"))
    socket_writer = SocketWriter()
    wire_magic_defense(
        supervisor=type("S", (), {"set_metadata_writer": lambda self, w: None})(),
        metadata_writer=writer, socket_writer=socket_writer,
    )
    sender, receiver = socket.socketpair()
    socket_writer._sock = sender  # noqa: SLF001 -- bypass connect(); same seam as sibling tests
    try:
        frame = {
            "emit_kind": "prose",
            "content": [
                {"type": "text", "text": "plain prose"},
                {"type": "text", "text": "@@deadbeef:spawn some_agent"},
            ],
        }
        socket_writer.write_frame(frame)
        line = b""
        while not line.endswith(b"\n"):
            line += receiver.recv(4096)
        on_wire = json.loads(line.decode("utf-8"))
    finally:
        socket_writer._sock = None  # noqa: SLF001
        sender.close()
        receiver.close()
    assert on_wire["content"][0]["text"] == "plain prose"
    assert on_wire["content"][1]["text"] == "\\@@deadbeef:spawn some_agent"
    # The caller's record is never mutated.
    assert frame["content"][1]["text"] == "@@deadbeef:spawn some_agent"


def test_sanitizer_walks_lists_under_text_fields_and_leaves_identity() -> None:
    record = {"text": ["@@deadbeef:spawn x", "ok"], "outputs": [{"preview": "@@cafe0123:scratch y"}]}
    out = sanitize_outbound_record(record)
    assert out is not record
    assert out["text"] == ["\\@@deadbeef:spawn x", "ok"]
    assert out["outputs"][0]["preview"] == "\\@@cafe0123:scratch y"
    clean = {"text": ["nothing here"], "outputs": [{"preview": "fine"}]}
    assert sanitize_outbound_record(clean) is clean

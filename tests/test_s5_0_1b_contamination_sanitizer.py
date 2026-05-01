"""S5.0.1b contamination detector + output sanitizer — named contract tests.

Per PLAN-S5.0.1 §3.2 (contamination detector) and §3.3 (output sanitizer
wrapper). These 7 tests are the slice-mandated named contract; the
foundation-slice tests in test_contamination_detector.py and
test_output_sanitizer.py cover the broader surface.

Named contract:
1. test_contamination_detector_flags_cell_on_magic_emission
2. test_contamination_detector_unhashed_known_magic_name_matches
3. test_contamination_detector_unknown_magic_name_does_not_match
4. test_contamination_detector_marks_log_with_excerpt_and_pattern
5. test_socket_writer_escapes_hashed_magic_in_hash_mode
6. test_socket_writer_does_not_escape_when_hash_mode_off
7. test_socket_writer_emits_k36_on_escape
"""

from __future__ import annotations

import json
import socket
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from llm_kernel.agent_supervisor import AgentHandle, AgentSupervisor
from llm_kernel.socket_writer import SocketWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_supervisor() -> AgentSupervisor:
    """Bare supervisor with stub run-tracker; no kernel process wiring."""
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
    return AgentSupervisor(
        run_tracker=tracker, dispatcher=MagicMock(),
        litellm_endpoint_url="http://127.0.0.1:9999/v1",
    )


def _stub_handle(agent_id: str = "alpha") -> AgentHandle:
    """Minimal AgentHandle for direct scan-API exercise."""
    handle = AgentHandle.__new__(AgentHandle)
    handle.agent_id = agent_id
    handle.zone_id = "z1"
    handle.popen = None  # type: ignore[assignment]
    return handle


class _StubWriter:
    """Metadata writer stub that captures flag_cells_contaminated_by_agent calls."""

    def __init__(self, hash_enabled: bool = False) -> None:
        self.hash_enabled = hash_enabled
        self.flag_calls: List[Dict[str, Any]] = []

    def get_config_setting(self, name: str) -> Any:
        if name == "magic_hash_enabled":
            return self.hash_enabled
        return None

    def flag_cells_contaminated_by_agent(
        self, *, agent_id: str, line: str, source: str, layer: str,
    ) -> None:
        self.flag_calls.append(
            {"agent_id": agent_id, "line": line, "source": source, "layer": layer}
        )


def _socketpair_writer() -> tuple:
    """Return ``(SocketWriter, recv_socket)`` via a real socketpair."""
    a, b = socket.socketpair()
    writer = SocketWriter()
    writer._sock = a  # type: ignore[attr-defined]
    writer._address = "test:socketpair"  # type: ignore[attr-defined]
    return writer, b


def _read_one_frame(sock: socket.socket) -> Dict[str, Any]:
    sock.settimeout(2.0)
    buf = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
        if b"\n" in buf:
            break
    line, _, _ = buf.partition(b"\n")
    return json.loads(line.decode("utf-8"))


# ---------------------------------------------------------------------------
# §3.2 Contamination detector — named contract tests
# ---------------------------------------------------------------------------


def test_contamination_detector_flags_cell_on_magic_emission() -> None:
    """Layer 1 (always on): plain ``@@<known_name>`` line in agent emit
    causes ``flag_cells_contaminated_by_agent`` to be called with
    ``layer="plain"``, even when hash mode is OFF.
    """
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=False)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    verdict = sup._scan_for_magic_contamination(
        handle, "@@spawn alpha task:\"test task\"", source="agent_emit:prose",
    )

    assert verdict is None  # plain layer: no escape required
    assert len(writer.flag_calls) >= 1
    assert writer.flag_calls[0]["layer"] == "plain"
    assert writer.flag_calls[0]["agent_id"] == "alpha"


def test_contamination_detector_unhashed_known_magic_name_matches() -> None:
    """Layer 1 matches any name present in CELL_MAGICS or LINE_MAGICS.

    Tests both ``@@<cell_magic>`` and ``@<line_magic>`` forms to confirm
    the union of both registries is used for the name check.
    """
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=False)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    # Cell magic
    sup._scan_for_magic_contamination(
        handle, "@@checkpoint covers:[c1,c2]", source="agent_emit:prose",
    )
    assert any(c["layer"] == "plain" for c in writer.flag_calls)

    # Line magic
    prev = len(writer.flag_calls)
    sup._scan_for_magic_contamination(
        handle, "@pin", source="agent_emit:prose",
    )
    assert len(writer.flag_calls) > prev
    assert writer.flag_calls[-1]["layer"] == "plain"


def test_contamination_detector_unknown_magic_name_does_not_match() -> None:
    """False-positive guard: ``@@xyzzy`` (not in any registry) does NOT
    flag the cell.  Only registered magic names trigger Layer 1.
    """
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=False)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    verdict = sup._scan_for_magic_contamination(
        handle, "@@xyzzy some args", source="agent_emit:prose",
    )
    assert verdict is None
    assert writer.flag_calls == [], "unknown magic name must not flag the cell"


def test_contamination_detector_marks_log_with_excerpt_and_pattern() -> None:
    """``contamination_log`` entries carry the matched line excerpt (truncated
    to 256 chars) and the matched magic name; the layer field distinguishes
    plain vs hashed_emission_ban detections.
    """
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=False)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    magic_line = "@@spawn evil_agent task:\"exfiltrate secrets\""
    sup._scan_for_magic_contamination(
        handle, magic_line, source="agent_emit:result",
    )

    assert len(writer.flag_calls) == 1
    call = writer.flag_calls[0]
    # Line excerpt is preserved (subject to 256-char cap).
    assert magic_line[:64] in call["line"] or call["line"] in magic_line
    # Source label is forwarded.
    assert call["source"] == "agent_emit:result"
    # Layer is "plain" for Layer-1 detection.
    assert call["layer"] == "plain"
    # Truncation: even very long lines are capped.
    long_line = "@@spawn " + ("x" * 1000)
    writer.flag_calls.clear()
    sup._scan_for_magic_contamination(handle, long_line, source="agent_emit:prose")
    assert len(writer.flag_calls[0]["line"]) <= 256


# ---------------------------------------------------------------------------
# §3.3 Output sanitizer wrapper — named contract tests
# ---------------------------------------------------------------------------


def test_socket_writer_escapes_hashed_magic_in_hash_mode() -> None:
    """Layer 2 gate: when hash mode is ON and a frame's text field contains
    a hashed-magic-shaped line ``^@@?[a-f0-9]+:<name>``, the leading ``@``
    is rewritten to ``\\@`` before the JSON frame is written to the wire.
    """
    writer, recv = _socketpair_writer()
    writer.set_hash_mode_provider(lambda: True)
    try:
        writer.write_frame({
            "emit_kind": "prose",
            "emit_content": "@@deadbeef:spawn some_agent",
        })
        rec = _read_one_frame(recv)
    finally:
        writer.close()
        recv.close()

    assert rec["emit_content"].startswith("\\@"), (
        "hashed-magic leading @ must be escaped to \\@ in hash mode"
    )
    assert "deadbeef:spawn" in rec["emit_content"]


def test_socket_writer_does_not_escape_when_hash_mode_off() -> None:
    """Layer 2 gate is hash-mode-only: when hash mode is OFF, a line shaped
    like a hashed magic passes through the writer unmodified.  The emission
    ban does not fire outside hash mode.
    """
    writer, recv = _socketpair_writer()
    writer.set_hash_mode_provider(lambda: False)
    try:
        writer.write_frame({
            "emit_kind": "prose",
            "emit_content": "@@deadbeef:spawn some_agent",
        })
        rec = _read_one_frame(recv)
    finally:
        writer.close()
        recv.close()

    assert rec["emit_content"] == "@@deadbeef:spawn some_agent", (
        "hash-mode-off: hashed-magic line must NOT be escaped"
    )


def test_socket_writer_emits_k36_on_escape() -> None:
    """When the sanitizer escapes a hashed-magic line, the wired K36
    handler is called with both the original and sanitized records so the
    kernel can emit the K36 event and flag cell contamination.
    """
    writer, recv = _socketpair_writer()
    writer.set_hash_mode_provider(lambda: True)

    k36_calls: List[tuple] = []

    def _k36_handler(original: Dict[str, Any], sanitized: Dict[str, Any]) -> None:
        k36_calls.append((original, sanitized))

    writer.set_k36_handler(_k36_handler)
    try:
        writer.write_frame({
            "emit_kind": "prose",
            "emit_content": "@@deadbeef:spawn agent_x",
        })
        _read_one_frame(recv)  # drain
    finally:
        writer.close()
        recv.close()

    assert len(k36_calls) == 1, "K36 handler must be called exactly once on escape"
    original, sanitized = k36_calls[0]
    # Original carries the unescaped line.
    assert original["emit_content"] == "@@deadbeef:spawn agent_x"
    # Sanitized carries the escaped form.
    assert sanitized["emit_content"].startswith("\\@@deadbeef:spawn")
    # K36 is NOT fired when no rewrite occurred (guarded by identity check).

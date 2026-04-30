"""Output sanitizer tests — S5.0.1a foundation slice.

Per PLAN-S5.0.1 §3.3. The socket_writer's outbound path gains a
defense-in-depth sanitizer: when hash mode is on, any line in a
text-bearing field that matches the canonical hashed-magic shape is
escaped (``@`` → ``\\@``) before the JSON frame is written. This
catches kernel-emitted paths that bypass the agent_supervisor scan
(notify spans, escalate text, synthetic markers).

Three primary tests per the slice spec:
1. hash-mode + hashed line → escaped before write
2. hash-mode + plain line → write-through (no false positive)
3. hash-mode-off + hashed line → write-through (defense is opt-in)

Plus tests for the standalone helpers and identity short-circuit.
"""

from __future__ import annotations

import json
import socket
import threading
from typing import Any, Dict, List

import pytest

from llm_kernel.socket_writer import (
    SocketWriter,
    sanitize_outbound_line, sanitize_outbound_record,
    sanitize_outbound_text,
)


# ---------------------------------------------------------------------------
# Standalone sanitizer helpers
# ---------------------------------------------------------------------------


def test_sanitize_line_escapes_hashed_magic() -> None:
    out = sanitize_outbound_line("@@deadbeef:spawn alpha")
    assert out == "\\@@deadbeef:spawn alpha"


def test_sanitize_line_passthrough_on_plain_text() -> None:
    assert sanitize_outbound_line("hello world") == "hello world"
    assert sanitize_outbound_line("@@spawn alpha") == "@@spawn alpha"
    assert sanitize_outbound_line("") == ""


def test_sanitize_line_idempotent() -> None:
    once = sanitize_outbound_line("@@deadbeef:spawn x")
    twice = sanitize_outbound_line(once)
    # Already escaped: looks_like_hashed_magic is False on \\@@... so
    # the helper returns the input unchanged.
    assert once == twice


def test_sanitize_text_multiline() -> None:
    text = (
        "prose line\n"
        "@@deadbeef:spawn alpha\n"
        "more prose\n"
        "@@feedface:checkpoint covers:c_3\n"
    )
    out = sanitize_outbound_text(text)
    assert "\\@@deadbeef:spawn alpha" in out
    assert "\\@@feedface:checkpoint covers:c_3" in out
    assert "prose line" in out and "more prose" in out


def test_sanitize_text_no_at_short_circuits_identity() -> None:
    text = "no at sign anywhere\nplain prose\n"
    assert sanitize_outbound_text(text) is text


def test_sanitize_text_no_match_short_circuits_identity() -> None:
    text = "@@spawn alpha\n@@agent beta\n"  # plain magics, not hashed
    assert sanitize_outbound_text(text) is text


def test_sanitize_record_rewrites_known_text_field() -> None:
    rec = {"emit_kind": "prose", "emit_content": "@@deadbeef:spawn x"}
    out = sanitize_outbound_record(rec)
    assert out is not rec
    assert out["emit_content"].startswith("\\@@")
    assert rec["emit_content"] == "@@deadbeef:spawn x"  # untouched original


def test_sanitize_record_passes_through_unknown_fields() -> None:
    rec = {"foo": "@@deadbeef:spawn x"}  # not in _OUTPUT_TEXT_FIELDS
    out = sanitize_outbound_record(rec)
    # ``foo`` is not in the sanitizer's text-field list; the field is
    # not scanned (the kernel doesn't route cell-output text through
    # arbitrary keys).
    assert out is rec
    assert out["foo"] == "@@deadbeef:spawn x"


def test_sanitize_record_recurses_into_nested_dict() -> None:
    rec = {
        "metadata": {"emit_content": "@@deadbeef:spawn x"},
        "other": "ignored",
    }
    out = sanitize_outbound_record(rec)
    assert out is not rec
    assert out["metadata"]["emit_content"].startswith("\\@@")
    # Original untouched.
    assert rec["metadata"]["emit_content"] == "@@deadbeef:spawn x"


def test_sanitize_record_identity_when_no_match() -> None:
    rec = {"emit_kind": "prose", "emit_content": "harmless prose"}
    assert sanitize_outbound_record(rec) is rec


def test_sanitize_record_handles_non_dict() -> None:
    assert sanitize_outbound_record("not a dict") == "not a dict"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SocketWriter integration — hash-mode provider + write_frame
# ---------------------------------------------------------------------------


def _socketpair_writer():
    """Return ``(writer, recv_sock)`` connected via socketpair.

    Uses :class:`socket.socketpair` so the test runs on POSIX without
    a UDS path. On Windows :func:`socket.socketpair` falls back to
    AF_INET loopback per CPython 3.5+; both work for the read-side
    assertions below.
    """
    import socket as _s
    a, b = _s.socketpair()
    writer = SocketWriter()
    # Bypass the address-parsing path: directly install ``a`` as the
    # connected socket. The writer treats this as a connected stream.
    writer._sock = a  # type: ignore[attr-defined]
    writer._address = "test:socketpair"  # type: ignore[attr-defined]
    return writer, b


def _read_one_frame(sock: socket.socket) -> Dict[str, Any]:
    """Read one newline-delimited JSON frame from ``sock``."""
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


def test_writer_hash_mode_on_escapes_hashed_line_in_emit_content() -> None:
    """hash-mode + hashed line → wire payload is escaped."""
    writer, recv = _socketpair_writer()
    writer.set_hash_mode_provider(lambda: True)
    try:
        writer.write_frame({
            "emit_kind": "prose",
            "emit_content": "before\n@@deadbeef:spawn x\nafter",
        })
        rec = _read_one_frame(recv)
    finally:
        writer.close()
        recv.close()
    assert rec["emit_kind"] == "prose"
    assert "\\@@deadbeef:spawn x" in rec["emit_content"]
    assert "before" in rec["emit_content"] and "after" in rec["emit_content"]


def test_writer_hash_mode_on_plain_line_passes_through() -> None:
    """hash-mode + plain magic line → wire payload unchanged."""
    writer, recv = _socketpair_writer()
    writer.set_hash_mode_provider(lambda: True)
    try:
        writer.write_frame({
            "emit_kind": "prose",
            "emit_content": "@@spawn alpha task:\"x\"",
        })
        rec = _read_one_frame(recv)
    finally:
        writer.close()
        recv.close()
    # Plain magic is the contamination detector's domain; the
    # emission ban is hash-mode-only and ONLY targets hashed shapes.
    assert rec["emit_content"] == "@@spawn alpha task:\"x\""
    assert "\\" not in rec["emit_content"]


def test_writer_hash_mode_off_hashed_line_passes_through() -> None:
    """hash-mode OFF + hashed line → no false positive; line unchanged."""
    writer, recv = _socketpair_writer()
    writer.set_hash_mode_provider(lambda: False)
    try:
        writer.write_frame({
            "emit_kind": "prose",
            "emit_content": "@@deadbeef:spawn x",
        })
        rec = _read_one_frame(recv)
    finally:
        writer.close()
        recv.close()
    assert rec["emit_content"] == "@@deadbeef:spawn x"


def test_writer_no_provider_passes_through() -> None:
    """No hash_mode_provider wired → sanitization no-op (V1 default)."""
    writer, recv = _socketpair_writer()
    # Don't call set_hash_mode_provider — leaves it as None.
    try:
        writer.write_frame({
            "emit_kind": "prose",
            "emit_content": "@@deadbeef:spawn x",
        })
        rec = _read_one_frame(recv)
    finally:
        writer.close()
        recv.close()
    assert rec["emit_content"] == "@@deadbeef:spawn x"


def test_writer_provider_raises_falls_back_to_disabled() -> None:
    """A raising provider is treated as hash-mode-off (defensive)."""
    writer, recv = _socketpair_writer()

    def _bad_provider() -> bool:
        raise RuntimeError("config unavailable")

    writer.set_hash_mode_provider(_bad_provider)
    try:
        writer.write_frame({
            "emit_kind": "prose",
            "emit_content": "@@deadbeef:spawn x",
        })
        rec = _read_one_frame(recv)
    finally:
        writer.close()
        recv.close()
    # Provider raised → treated as False → no escape.
    assert rec["emit_content"] == "@@deadbeef:spawn x"


def test_writer_hash_mode_does_not_disturb_other_fields() -> None:
    """Sanitization touches only known text fields."""
    writer, recv = _socketpair_writer()
    writer.set_hash_mode_provider(lambda: True)
    try:
        writer.write_frame({
            "run_id": "abc-123",
            "emit_kind": "prose",
            "emit_content": "@@deadbeef:spawn x",
            "ts": "2026-04-29T00:00:00Z",
        })
        rec = _read_one_frame(recv)
    finally:
        writer.close()
        recv.close()
    assert rec["run_id"] == "abc-123"
    assert rec["ts"] == "2026-04-29T00:00:00Z"
    assert rec["emit_content"].startswith("\\@@")


def test_set_hash_mode_provider_to_none_disables() -> None:
    """Wiring the provider then clearing it reverts to no-op."""
    writer, recv = _socketpair_writer()
    writer.set_hash_mode_provider(lambda: True)
    writer.set_hash_mode_provider(None)
    try:
        writer.write_frame({
            "emit_kind": "prose",
            "emit_content": "@@deadbeef:spawn x",
        })
        rec = _read_one_frame(recv)
    finally:
        writer.close()
        recv.close()
    assert rec["emit_content"] == "@@deadbeef:spawn x"

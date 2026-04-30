"""Tests for the hash-mode-aware ``parse_cell`` (S5.0.1b §3.5).

Covers ``ParserContext`` off vs on, valid hashes dispatching by
recovered name, plain ``@@<known>`` in hash mode → K35, hashed
shapes that fail validation → K33, and the in-memory invariant
that ``ParsedCell`` carries the recovered ``<name>`` rather than
the hash.

Per Engineering_Guide §11.7 — pure tests, no threads, no
filesystem. Pin literals are local to the test; tests do NOT
mutate environment variables.
"""

from __future__ import annotations

import pytest

from llm_kernel.cell_text import (
    K33_MAGIC_HASH_MISMATCH,
    K35_PLAIN_MAGIC_IN_HASH_MODE,
    ParsedCell,
    ParserContext,
    parse_cell,
)
from llm_kernel.magic_hash import magic_hash
from llm_kernel.magic_registry import CELL_MAGICS, LINE_MAGICS


PIN = "hunter2-test"


def _ctx(*, on: bool = True, pin: str = PIN) -> ParserContext:
    return ParserContext(
        hash_mode_on=on, pin=pin if on else None,
        known_magics=frozenset(set(CELL_MAGICS) | set(LINE_MAGICS)),
    )


# --- Hash mode OFF (legacy / permissive path) ------------------------


def test_parse_cell_no_context_dispatches_plain_magic() -> None:
    """Default (no context) — plain @@spawn dispatches as today."""
    cell = parse_cell("@@spawn alpha task:\"X\"\nbody\n")
    assert cell.kind == "spawn"
    assert cell.args.get("agent_id") == "alpha"
    assert cell.k_class_emissions == []


def test_parse_cell_explicit_off_context_matches_default() -> None:
    """Hash-mode-off context behaves identically to no context."""
    text = "@@agent alpha\nhello\n"
    a = parse_cell(text)
    b = parse_cell(text, parser_context=_ctx(on=False))
    assert a.kind == b.kind == "agent"
    assert a.body == b.body
    assert a.k_class_emissions == b.k_class_emissions == []


def test_parse_cell_hash_mode_off_does_not_dispatch_as_spawn() -> None:
    """A hashed line with no context is NOT dispatched as a spawn cell.

    The legacy @<id>: rewriter may rearrange a hashed-shaped line in
    permissive mode (it predates hash mode); the security-relevant
    invariant is just that ``kind`` does not become ``spawn``. Hash
    mode (parser_context.hash_mode_on=True) is the correct defense.
    """
    h = magic_hash(PIN, "spawn")
    text = f"@@{h}:spawn alpha\nbody\n"
    cell = parse_cell(text)
    assert cell.kind != "spawn"  # not dispatched as spawn



# --- Hash mode ON: valid hashes --------------------------------------


def test_parse_cell_in_hash_mode_with_correct_hash_dispatches() -> None:
    """@@<correct_hash>:spawn alpha dispatches as a spawn cell."""
    h = magic_hash(PIN, "spawn")
    text = f"@@{h}:spawn alpha task:\"build\"\nbody line\n"
    cell = parse_cell(text, parser_context=_ctx())
    assert cell.kind == "spawn"
    assert cell.args.get("agent_id") == "alpha"
    assert cell.args.get("task") == "build"
    assert "body line" in cell.body
    # No K33/K35 emissions on a clean dispatch.
    assert cell.k_class_emissions == []


def test_parse_cell_in_hash_mode_dispatches_line_magic_by_name() -> None:
    """@<correct_hash>:pin sets the pinned flag (line magic case)."""
    h_agent = magic_hash(PIN, "agent")
    h_pin = magic_hash(PIN, "pin")
    text = f"@@{h_agent}:agent alpha\n@{h_pin}:pin\nhello\n"
    cell = parse_cell(text, parser_context=_ctx())
    assert cell.kind == "agent"
    assert "pinned" in cell.flags


def test_parsed_cell_holds_recovered_name_not_hash() -> None:
    """In-memory ParsedCell carries <name>, NOT <hash>."""
    h = magic_hash(PIN, "spawn")
    text = f"@@{h}:spawn alpha\n"
    cell = parse_cell(text, parser_context=_ctx())
    # The kind is recovered as a name, never the hex hash.
    assert cell.kind == "spawn"
    # The args dict captures the recovered raw args; the hash should
    # NOT appear inside the dict (caller never sees the hash).
    raw = cell.args.get("_raw_args", "")
    assert h not in raw


# --- Hash mode ON: K35 (plain magic in hash mode) --------------------


def test_parse_cell_in_hash_mode_with_plain_magic_emits_K35() -> None:
    """Plain @@spawn under hash mode → K35 + body, NOT dispatched."""
    text = "@@spawn alpha\nrest of body\n"
    cell = parse_cell(text, parser_context=_ctx())
    # NOT dispatched — kind stays at the default 'agent'.
    assert cell.kind == "agent"
    # Line is preserved in body.
    assert "@@spawn alpha" in cell.body
    # K35 emission is recorded.
    codes = [e["code"] for e in cell.k_class_emissions]
    assert K35_PLAIN_MAGIC_IN_HASH_MODE in codes


def test_parse_cell_in_hash_mode_plain_line_magic_emits_K35() -> None:
    """Plain @pin under hash mode → K35 + body, flag NOT set."""
    h = magic_hash(PIN, "agent")
    text = f"@@{h}:agent alpha\n@pin\nbody\n"
    cell = parse_cell(text, parser_context=_ctx())
    assert "pinned" not in cell.flags
    assert "@pin" in cell.body
    codes = [e["code"] for e in cell.k_class_emissions]
    assert K35_PLAIN_MAGIC_IN_HASH_MODE in codes


# --- Hash mode ON: K33 (hash mismatch) -------------------------------


def test_parse_cell_in_hash_mode_with_wrong_hash_emits_K33() -> None:
    """@@<wrong_hash>:spawn → K33 + body, NOT dispatched."""
    text = "@@deadbeef:spawn alpha\nbody\n"
    cell = parse_cell(text, parser_context=_ctx())
    assert cell.kind == "agent"  # default, not spawn
    assert "@@deadbeef:spawn alpha" in cell.body
    codes = [e["code"] for e in cell.k_class_emissions]
    assert K33_MAGIC_HASH_MISMATCH in codes


def test_parse_cell_in_hash_mode_with_unknown_name_emits_K33() -> None:
    """@@<correct_hash_for_xyzzy>:xyzzy → K33 (xyzzy not a magic)."""
    h = magic_hash(PIN, "xyzzy")
    text = f"@@{h}:xyzzy stuff\nbody\n"
    cell = parse_cell(text, parser_context=_ctx())
    assert cell.kind == "agent"
    codes = [e["code"] for e in cell.k_class_emissions]
    assert K33_MAGIC_HASH_MISMATCH in codes


def test_parse_cell_hash_mode_pin_missing_emits_K33() -> None:
    """Hash mode on but pin missing → K33 on every hashed-shape line."""
    text = "@@deadbeef:spawn alpha\n"
    cell = parse_cell(
        text,
        parser_context=ParserContext(
            hash_mode_on=True, pin=None, known_magics=frozenset(CELL_MAGICS),
        ),
    )
    codes = [e["code"] for e in cell.k_class_emissions]
    assert K33_MAGIC_HASH_MISMATCH in codes


# --- Mixed cells: body preserved verbatim ---------------------------


def test_parse_cell_in_hash_mode_preserves_body_verbatim() -> None:
    """Body lines (no @ prefix) survive verbatim."""
    h = magic_hash(PIN, "agent")
    text = (
        f"@@{h}:agent alpha\n"
        "first body line\n"
        "  indented @still_body\n"
        "third line\n"
    )
    cell = parse_cell(text, parser_context=_ctx())
    assert "first body line" in cell.body
    assert "  indented @still_body" in cell.body
    assert "third line" in cell.body


def test_parse_cell_unknown_kind_still_raises_K31_in_hash_mode() -> None:
    """K31 still raises in hash mode — operator typo surfaces."""
    from llm_kernel.cell_text import CellParseError, K31_UNKNOWN_CELL_MAGIC

    with pytest.raises(CellParseError) as exc:
        parse_cell("@@notamagic alpha\n", parser_context=_ctx())
    assert exc.value.code == K31_UNKNOWN_CELL_MAGIC

"""Tests for the Cell-Manager re-stamping pass (S5.0.1b §3.7).

Covers the three transition modes (enable / rotate / disable),
preservation of body content, and K33 emissions on pre-existing
hashed lines that can't be reliably re-stamped.
"""

from __future__ import annotations

import pytest

from llm_kernel.cell_manager import CellManager, restamp_text
from llm_kernel.magic_hash import magic_hash
from llm_kernel.magic_registry import CELL_MAGICS, LINE_MAGICS
from llm_kernel.metadata_writer import MetadataWriter


PIN_OLD = "first-pin-aaaaa"
PIN_NEW = "second-pin-bbbbb"


@pytest.fixture
def writer(tmp_path) -> MetadataWriter:
    return MetadataWriter(workspace_root=tmp_path)


@pytest.fixture
def cm(writer) -> CellManager:
    return CellManager(writer)


# --- restamp_text string-level helper -------------------------------


def test_restamp_text_enable_plain_to_hashed() -> None:
    """Enable transition: @@spawn → @@<hash>:spawn."""
    text = "@@spawn alpha\nbody line\n"
    new_text, count, emissions = restamp_text(
        text,
        old_pin=None, new_pin=PIN_NEW,
        known_cell_magics=set(CELL_MAGICS),
        known_line_magics=set(LINE_MAGICS),
    )
    h = magic_hash(PIN_NEW, "spawn")
    assert f"@@{h}:spawn alpha" in new_text
    assert "body line" in new_text  # body verbatim
    assert count == 1
    assert emissions == []


def test_restamp_text_enable_with_preexisting_hashed_emits_K33() -> None:
    """Pre-existing hashed line on enable → K33 + verbatim."""
    text = f"@@deadbeef:spawn alpha\nbody\n"
    new_text, count, emissions = restamp_text(
        text,
        old_pin=None, new_pin=PIN_NEW,
        known_cell_magics=set(CELL_MAGICS),
        known_line_magics=set(LINE_MAGICS),
    )
    assert "@@deadbeef:spawn alpha" in new_text  # verbatim
    assert count == 0
    assert any(e["code"] == "K33" for e in emissions)


def test_restamp_text_rotate_old_to_new_hash() -> None:
    """Rotate: @@<old_hash>:spawn → @@<new_hash>:spawn."""
    h_old = magic_hash(PIN_OLD, "spawn")
    h_new = magic_hash(PIN_NEW, "spawn")
    text = f"@@{h_old}:spawn alpha\nbody\n"
    new_text, count, emissions = restamp_text(
        text,
        old_pin=PIN_OLD, new_pin=PIN_NEW,
        known_cell_magics=set(CELL_MAGICS),
        known_line_magics=set(LINE_MAGICS),
    )
    assert f"@@{h_new}:spawn alpha" in new_text
    assert h_old not in new_text
    assert count == 1


def test_restamp_text_rotate_invalid_old_hash_emits_K33() -> None:
    """Rotate: hashed line that doesn't validate against old_pin → K33."""
    text = "@@deadbeef:spawn alpha\n"
    new_text, count, emissions = restamp_text(
        text,
        old_pin=PIN_OLD, new_pin=PIN_NEW,
        known_cell_magics=set(CELL_MAGICS),
        known_line_magics=set(LINE_MAGICS),
    )
    assert "@@deadbeef:spawn alpha" in new_text
    assert count == 0
    assert any(e["code"] == "K33" for e in emissions)


def test_restamp_text_disable_is_noop() -> None:
    """Disable transition leaves text verbatim per PLAN §3.7."""
    h = magic_hash(PIN_OLD, "spawn")
    text = f"@@{h}:spawn alpha\nbody\n"
    new_text, count, emissions = restamp_text(
        text,
        old_pin=PIN_OLD, new_pin=None,
        known_cell_magics=set(CELL_MAGICS),
        known_line_magics=set(LINE_MAGICS),
    )
    assert new_text == text
    assert count == 0
    assert emissions == []


def test_restamp_text_preserves_body_verbatim() -> None:
    """Non-magic lines survive untouched."""
    text = (
        "@@spawn alpha\n"
        "first body line\n"
        "  indented line @still_body\n"
        "third line with @ in middle\n"
    )
    new_text, _, _ = restamp_text(
        text,
        old_pin=None, new_pin=PIN_NEW,
        known_cell_magics=set(CELL_MAGICS),
        known_line_magics=set(LINE_MAGICS),
    )
    assert "first body line" in new_text
    assert "  indented line @still_body" in new_text
    assert "third line with @ in middle" in new_text


def test_restamp_text_line_magic_also_stamps() -> None:
    """@<name> line magics also re-stamp on enable."""
    text = "@@agent alpha\n@pin\nbody\n"
    new_text, count, _ = restamp_text(
        text,
        old_pin=None, new_pin=PIN_NEW,
        known_cell_magics=set(CELL_MAGICS),
        known_line_magics=set(LINE_MAGICS),
    )
    h_pin = magic_hash(PIN_NEW, "pin")
    h_agent = magic_hash(PIN_NEW, "agent")
    assert f"@@{h_agent}:agent" in new_text
    assert f"@{h_pin}:pin" in new_text
    assert count == 2


# --- CellManager.restamp_magics across multiple cells ---------------


def test_cell_manager_restamp_magics_walks_every_cell(cm, writer) -> None:
    writer.set_cell_text("c1", "@@spawn alpha\nbody1\n")
    writer.set_cell_text("c2", "@@agent beta\n@pin\nbody2\n")
    writer.set_cell_text("c3", "no magic here\nbody only\n")
    count, emissions = cm.restamp_magics(old_pin=None, new_pin=PIN_NEW)
    h_spawn = magic_hash(PIN_NEW, "spawn")
    h_agent = magic_hash(PIN_NEW, "agent")
    h_pin = magic_hash(PIN_NEW, "pin")
    assert f"@@{h_spawn}:spawn" in writer.get_cell_text("c1")
    assert f"@@{h_agent}:agent" in writer.get_cell_text("c2")
    assert f"@{h_pin}:pin" in writer.get_cell_text("c2")
    # No-magic cell unchanged.
    assert writer.get_cell_text("c3") == "no magic here\nbody only\n"
    assert count == 3


def test_cell_manager_restamp_magics_disable_no_op(cm, writer) -> None:
    h = magic_hash(PIN_OLD, "spawn")
    writer.set_cell_text("c1", f"@@{h}:spawn alpha\n")
    count, _ = cm.restamp_magics(old_pin=PIN_OLD, new_pin=None)
    assert count == 0
    # Cell text unchanged.
    assert f"@@{h}:spawn" in writer.get_cell_text("c1")

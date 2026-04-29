"""Tests for ``llm_kernel.cell_manager`` — BSP-005 S5.0 text-mutation ops."""

from __future__ import annotations

from llm_kernel.cell_manager import (
    insert_line_magic_text,
    merge_cells_text,
    remove_line_magic_text,
    set_cell_kind_text,
    split_at_break_text,
)


def test_split_at_break_inserts_marker() -> None:
    """split_at_break_text(text, position) returns two cells."""
    text = "@@agent alpha\nfirst half\nsecond half"
    left, right = split_at_break_text(text, len("@@agent alpha\nfirst half"))
    assert left == "@@agent alpha\nfirst half"
    assert right == "second half"


def test_merge_cells_strips_intervening_break() -> None:
    """merge_cells_text drops a leading/trailing @@break at the join."""
    a = "first\n@@break"
    b = "@@break\nsecond"
    merged = merge_cells_text(a, b)
    assert merged == "first\nsecond"
    assert "@@break" not in merged


def test_insert_line_magic_at_top() -> None:
    """insert_line_magic_text prepends @<magic> below the cell-magic line."""
    text = "@@agent alpha\nbody text"
    out = insert_line_magic_text(text, "pin")
    lines = out.splitlines()
    assert lines[0] == "@@agent alpha"
    assert lines[1] == "@pin"
    assert "body text" in out


def test_insert_line_magic_with_args() -> None:
    """insert_line_magic_text writes ``@<magic> <args>``."""
    text = "@@agent alpha\nbody"
    out = insert_line_magic_text(text, "affinity", "primary,cheap")
    assert "@affinity primary,cheap" in out


def test_insert_line_magic_idempotent() -> None:
    """Inserting a magic that already exists is a no-op."""
    text = "@@agent alpha\n@pin\nbody"
    out = insert_line_magic_text(text, "pin")
    assert out == text


def test_set_cell_kind_replaces_existing() -> None:
    """set_cell_kind_text replaces the leading @@<kind> line."""
    text = "@@agent alpha\nbody"
    out = set_cell_kind_text(text, "markdown", "")
    assert out.startswith("@@markdown")
    assert "@@agent" not in out


def test_set_cell_kind_inserts_when_missing() -> None:
    """set_cell_kind_text on a kind-less cell prepends the declaration."""
    text = "just some prose"
    out = set_cell_kind_text(text, "scratch", "")
    assert out.startswith("@@scratch")
    assert "just some prose" in out


def test_remove_line_magic_idempotent() -> None:
    """Removing a non-existent magic is a no-op."""
    text = "@@agent alpha\nbody"
    out = remove_line_magic_text(text, "pin")
    assert out == text


def test_remove_line_magic_strips_matching_lines() -> None:
    """remove_line_magic_text removes column-0 ``@<magic>`` lines."""
    text = "@@agent alpha\n@pin\n@exclude\nbody"
    out = remove_line_magic_text(text, "pin")
    assert "@pin" not in out
    assert "@exclude" in out
    assert "body" in out

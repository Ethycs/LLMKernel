"""Tests for ``llm_kernel.cell_text`` — BSP-005 S5.0 parser."""

from __future__ import annotations

import pytest

from llm_kernel.cell_text import (
    CellParseError,
    K30_MULTIPLE_KINDS,
    K31_UNKNOWN_CELL_MAGIC,
    parse_cell,
    rewrite_legacy_directives,
    split_at_breaks,
)


# --- split_at_breaks -------------------------------------------------


def test_split_at_breaks_basic() -> None:
    """Three @@break-separated cells round-trip as three list entries."""
    text = "cell one\n@@break\ncell two\n@@break\ncell three"
    cells = split_at_breaks(text)
    assert cells == ["cell one", "cell two", "cell three"]


def test_split_at_breaks_implicit_file_start() -> None:
    """A cell at the file head (no leading @@break) is preserved."""
    text = "leading cell\n@@break\nsecond"
    cells = split_at_breaks(text)
    assert cells[0] == "leading cell"
    assert len(cells) == 2


def test_split_at_breaks_drops_empty_cells() -> None:
    """Back-to-back @@break and whitespace-only cells are dropped."""
    text = "@@break\n@@break\n   \n@@break\nactual"
    cells = split_at_breaks(text)
    assert cells == ["actual"]


def test_split_at_breaks_break_must_be_alone_on_line() -> None:
    """A line of literal '  @@break  ' is a separator; '@@break foo' is not."""
    text = "before\n  @@break  \nmiddle\n@@break extra\nstill body"
    cells = split_at_breaks(text)
    assert cells[0] == "before"
    # The ``@@break extra`` line was NOT a separator — it's body text.
    assert "middle" in cells[1]
    assert "@@break extra" in cells[1]


# --- parse_cell ------------------------------------------------------


def test_parse_cell_kind_declaration() -> None:
    """``@@agent alpha`` sets kind=agent, agent_id=alpha."""
    cell = parse_cell("@@agent alpha\nhello world")
    assert cell.kind == "agent"
    assert cell.args.get("agent_id") == "alpha"
    assert cell.body == "hello world"


def test_parse_cell_default_kind_when_no_magic() -> None:
    """A cell starting with prose defaults to kind=agent."""
    cell = parse_cell("just some prose\nmore prose")
    assert cell.kind == "agent"
    assert cell.kind_was_default is True
    assert "just some prose" in cell.body


def test_parse_cell_line_magics_mutate_flags() -> None:
    """@pin and @exclude accumulate into the flags set."""
    cell = parse_cell("@@agent alpha\n@pin\n@exclude\nbody text")
    assert "pinned" in cell.flags
    assert "excluded" in cell.flags
    assert cell.body == "body text"


def test_parse_cell_unpin_removes_pinned() -> None:
    """@unpin removes the pinned flag a prior @pin had set."""
    cell = parse_cell("@@agent alpha\n@pin\n@unpin\nbody")
    assert "pinned" not in cell.flags


def test_parse_cell_body_verbatim() -> None:
    """Body text containing unknown @something stays as body, verbatim."""
    cell = parse_cell("@@agent alpha\nemail me at @user before noon")
    assert "@user before noon" in cell.body


def test_parse_cell_k30_on_duplicate_kinds() -> None:
    """Two @@<kind> declarations in one cell raise K30."""
    with pytest.raises(CellParseError) as ei:
        parse_cell("@@agent alpha\n@@spawn beta task:\"x\"")
    assert ei.value.code == K30_MULTIPLE_KINDS


def test_parse_cell_k31_on_unknown_cell_magic() -> None:
    """``@@xyzzy`` at the kind position raises K31."""
    with pytest.raises(CellParseError) as ei:
        parse_cell("@@xyzzy something\nbody")
    assert ei.value.code == K31_UNKNOWN_CELL_MAGIC


def test_parse_cell_parametric_line_magic_recorded() -> None:
    """``@affinity primary,cheap`` lands in cell.line_magics, not flags."""
    cell = parse_cell("@@agent alpha\n@affinity primary,cheap\nbody")
    assert ("affinity", "primary,cheap") in cell.line_magics
    # Affinity is not a flag — it's parametric.
    assert "primary,cheap" not in cell.flags


def test_parse_cell_spawn_args_extracted() -> None:
    """``@@spawn alpha task:"…"`` extracts agent_id and task."""
    cell = parse_cell("@@spawn alpha task:\"design recipe schema\"")
    assert cell.kind == "spawn"
    assert cell.args.get("agent_id") == "alpha"
    assert cell.args.get("task") == "design recipe schema"


# --- legacy compat ---------------------------------------------------


def test_legacy_slash_spawn_alias() -> None:
    """``/spawn alpha task:"X"`` parses identically to @@spawn form."""
    legacy = parse_cell('/spawn alpha task:"X"')
    canonical = parse_cell('@@spawn alpha task:"X"')
    assert legacy.kind == canonical.kind == "spawn"
    assert legacy.args.get("agent_id") == canonical.args.get("agent_id") == "alpha"
    assert legacy.args.get("task") == canonical.args.get("task") == "X"
    assert legacy.legacy_alias_used is True


def test_legacy_at_id_colon_alias() -> None:
    """``@alpha: hello`` parses to an agent-bound cell with body=hello."""
    cell = parse_cell("@alpha: hello there")
    assert cell.kind == "agent"
    assert cell.args.get("agent_id") == "alpha"
    assert cell.body == "hello there"
    assert cell.legacy_alias_used is True


def test_rewrite_legacy_directives_no_match() -> None:
    """rewrite_legacy_directives returns text unchanged when no legacy form."""
    text = "@@agent alpha\nbody"
    out, was_legacy = rewrite_legacy_directives(text)
    assert out == text
    assert was_legacy is False

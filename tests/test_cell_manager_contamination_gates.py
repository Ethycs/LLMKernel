"""Tests for ``cell_manager.CellManager`` contamination precondition gates.

PLAN-S5.0.1c §3.10. For each structural op (split / merge / delete /
move / set_cell_kind / set_cell_text): set up a contaminated cell and
verify the op raises ``CellManagerPreconditionError`` with K3E. Also
verifies ``reset_contamination`` clears the flag and unblocks the
gates idempotently.
"""

from __future__ import annotations

from typing import Optional

import pytest

from llm_kernel.cell_manager import (
    CellManager,
    CellManagerPreconditionError,
    K3E_CONTAMINATED_CELL_STRUCTURAL_OP_BLOCKED,
)
from llm_kernel.metadata_writer import MetadataWriter


@pytest.fixture
def writer(tmp_path) -> MetadataWriter:
    return MetadataWriter(workspace_root=tmp_path)


def _seed_cell(
    writer: MetadataWriter,
    cell_id: str,
    text: str,
    bound_agent_id: Optional[str],
) -> None:
    writer.set_cell_text(cell_id, text)
    with writer._lock:
        record = dict(writer._cells.get(cell_id, {}))
        record["bound_agent_id"] = bound_agent_id
        writer._cells[cell_id] = record


def _contaminate(writer: MetadataWriter, cell_id: str) -> None:
    """Convenience: mark a cell contaminated via the agent-flag path."""
    bound = None
    with writer._lock:
        record = writer._cells.get(cell_id, {})
        bound = record.get("bound_agent_id")
    if not bound:
        # The flag-by-agent path requires a bound_agent_id; fall back
        # to direct mutation when no agent is bound.
        with writer._lock:
            record = dict(writer._cells.get(cell_id, {}))
            record["contaminated"] = True
            record["contamination_log"] = [
                {"detected_at": "2026-04-29T00:00:00.000Z",
                 "line": "@@spawn evil", "reason": "test",
                 "layer": "plain"},
            ]
            writer._cells[cell_id] = record
        return
    writer.flag_cells_contaminated_by_agent(
        agent_id=bound, line="@@spawn evil",
        source="stdout", layer="plain",
    )


# --- structural ops on contaminated cells: K3E ----------------------


def test_split_contaminated_raises_K3E(writer) -> None:
    _seed_cell(writer, "c1", "@@agent alpha\nbody", bound_agent_id="alpha")
    _contaminate(writer, "c1")
    cm = CellManager(writer)  # no supervisor => no running gate
    with pytest.raises(CellManagerPreconditionError) as exc:
        cm.split_at_break("c1", 5)
    assert exc.value.k_code == K3E_CONTAMINATED_CELL_STRUCTURAL_OP_BLOCKED


def test_merge_contaminated_either_side_raises_K3E(writer) -> None:
    _seed_cell(writer, "a", "first", bound_agent_id="alpha")
    _seed_cell(writer, "b", "second", bound_agent_id="beta")
    _contaminate(writer, "b")
    cm = CellManager(writer)
    with pytest.raises(CellManagerPreconditionError) as exc:
        cm.merge_cells("a", "b")
    assert exc.value.k_code == K3E_CONTAMINATED_CELL_STRUCTURAL_OP_BLOCKED


def test_delete_contaminated_raises_K3E(writer) -> None:
    _seed_cell(writer, "c1", "body", bound_agent_id="alpha")
    _contaminate(writer, "c1")
    cm = CellManager(writer)
    with pytest.raises(CellManagerPreconditionError) as exc:
        cm.delete_cell("c1")
    assert exc.value.k_code == K3E_CONTAMINATED_CELL_STRUCTURAL_OP_BLOCKED


def test_move_contaminated_raises_K3E(writer) -> None:
    _seed_cell(writer, "c1", "body", bound_agent_id="alpha")
    _contaminate(writer, "c1")
    cm = CellManager(writer)
    with pytest.raises(CellManagerPreconditionError) as exc:
        cm.move_cell("c1", 0)
    assert exc.value.k_code == K3E_CONTAMINATED_CELL_STRUCTURAL_OP_BLOCKED


def test_set_cell_kind_contaminated_raises_K3E(writer) -> None:
    _seed_cell(writer, "c1", "@@agent alpha\nbody", bound_agent_id="alpha")
    _contaminate(writer, "c1")
    cm = CellManager(writer)
    with pytest.raises(CellManagerPreconditionError) as exc:
        cm.set_cell_kind("c1", "scratch")
    assert exc.value.k_code == K3E_CONTAMINATED_CELL_STRUCTURAL_OP_BLOCKED


def test_set_cell_text_contaminated_raises_K3E(writer) -> None:
    _seed_cell(writer, "c1", "old", bound_agent_id="alpha")
    _contaminate(writer, "c1")
    cm = CellManager(writer)
    with pytest.raises(CellManagerPreconditionError) as exc:
        cm.set_cell_text("c1", "new body")
    assert exc.value.k_code == K3E_CONTAMINATED_CELL_STRUCTURAL_OP_BLOCKED


# --- reset_contamination unblocks ------------------------------------


def test_reset_contamination_clears_flag_and_log(writer) -> None:
    _seed_cell(writer, "c1", "body", bound_agent_id="alpha")
    _contaminate(writer, "c1")
    assert writer.is_cell_contaminated("c1") is True
    cm = CellManager(writer)
    assert cm.reset_contamination("c1") is True
    assert writer.is_cell_contaminated("c1") is False
    # log fully cleared (per AMBIGUITY-FLAG choice in the writer)
    record = writer._cells["c1"]
    assert "contamination_log" not in record


def test_reset_contamination_idempotent(writer) -> None:
    _seed_cell(writer, "c1", "body", bound_agent_id=None)
    cm = CellManager(writer)
    # Cell never contaminated → no-op, returns False.
    assert cm.reset_contamination("c1") is False


def test_reset_contamination_unblocks_subsequent_split(writer) -> None:
    _seed_cell(writer, "c1", "abc\ndef", bound_agent_id="alpha")
    _contaminate(writer, "c1")
    cm = CellManager(writer)
    with pytest.raises(CellManagerPreconditionError):
        cm.split_at_break("c1", 3)
    cm.reset_contamination("c1")
    left, right = cm.split_at_break("c1", 3)
    assert left == "abc"
    assert right == "def"


def test_reset_contamination_unknown_cell_returns_false(writer) -> None:
    cm = CellManager(writer)
    assert cm.reset_contamination("missing") is False


# --- writer-level helpers -------------------------------------------


def test_is_cell_contaminated_unknown_cell_returns_false(writer) -> None:
    assert writer.is_cell_contaminated("missing") is False


def test_is_cell_contaminated_uncontaminated_returns_false(writer) -> None:
    _seed_cell(writer, "c1", "body", bound_agent_id=None)
    assert writer.is_cell_contaminated("c1") is False


def test_reset_cell_contamination_returns_false_on_clean_cell(writer) -> None:
    _seed_cell(writer, "c1", "body", bound_agent_id=None)
    assert writer.reset_cell_contamination("c1") is False

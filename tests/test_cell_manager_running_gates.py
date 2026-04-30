"""Tests for ``cell_manager.CellManager`` running-cell precondition gates.

PLAN-S5.0.1c §3.10. For each structural op (split / merge / delete /
move / set_cell_kind / reset_contamination): set up a cell whose
bound agent is in an active state and verify the op raises
``CellManagerPreconditionError`` with K3C (or K3D for set_cell_kind).
Also verifies the text-edit-during-run path is allowed and emits the
K3F info marker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from llm_kernel import _diagnostics
from llm_kernel.cell_manager import (
    CellManager,
    CellManagerPreconditionError,
    K3C_RUNNING_CELL_STRUCTURAL_OP_BLOCKED,
    K3D_RUNNING_CELL_KIND_CHANGE_BLOCKED,
    K3F_RUNNING_CELL_EDIT_TEXT_ONLY_PATH,
)
from llm_kernel.metadata_writer import MetadataWriter


@dataclass
class _StubHandle:
    """Minimal stand-in for AgentHandle.state used by the predicate."""

    state: str = "running"


class _StubSupervisor:
    """Read-only stub: maps agent_id -> state via a dict."""

    def __init__(self, agents: Optional[dict] = None) -> None:
        self._agents = dict(agents or {})

    def get(self, agent_id: str):  # noqa: D401 - mirrors real API
        return self._agents.get(agent_id)


@pytest.fixture
def writer(tmp_path) -> MetadataWriter:
    return MetadataWriter(workspace_root=tmp_path)


def _seed_cell(
    writer: MetadataWriter,
    cell_id: str,
    text: str,
    bound_agent_id: Optional[str],
) -> None:
    """Set up a cell record with the given bound agent."""
    writer.set_cell_text(cell_id, text)
    # Patch bound_agent_id directly into the record (the higher-level
    # set_cell_metadata path requires a kind which is orthogonal here).
    with writer._lock:
        record = dict(writer._cells.get(cell_id, {}))
        record["bound_agent_id"] = bound_agent_id
        writer._cells[cell_id] = record


# --- split ----------------------------------------------------------


def test_split_running_cell_raises_K3C(writer) -> None:
    _seed_cell(writer, "c1", "@@agent alpha\nbody\n", bound_agent_id="alpha")
    sup = _StubSupervisor({"alpha": _StubHandle(state="running")})
    cm = CellManager(writer, supervisor=sup)
    with pytest.raises(CellManagerPreconditionError) as exc:
        cm.split_at_break("c1", 5)
    assert exc.value.k_code == K3C_RUNNING_CELL_STRUCTURAL_OP_BLOCKED


def test_split_starting_state_also_raises_K3C(writer) -> None:
    _seed_cell(writer, "c1", "@@agent alpha\nbody\n", bound_agent_id="alpha")
    sup = _StubSupervisor({"alpha": _StubHandle(state="starting")})
    cm = CellManager(writer, supervisor=sup)
    with pytest.raises(CellManagerPreconditionError) as exc:
        cm.split_at_break("c1", 5)
    assert exc.value.k_code == K3C_RUNNING_CELL_STRUCTURAL_OP_BLOCKED


def test_split_terminated_state_allowed(writer) -> None:
    _seed_cell(writer, "c1", "@@agent alpha\nbody\n", bound_agent_id="alpha")
    sup = _StubSupervisor({"alpha": _StubHandle(state="terminated")})
    cm = CellManager(writer, supervisor=sup)
    left, right = cm.split_at_break("c1", len("@@agent alpha"))
    assert left == "@@agent alpha"
    assert "body" in right


def test_split_no_supervisor_allowed(writer) -> None:
    """Back-compat: a CellManager without a supervisor never gates."""
    _seed_cell(writer, "c1", "abc\ndef", bound_agent_id="alpha")
    cm = CellManager(writer)
    left, right = cm.split_at_break("c1", 3)
    assert left == "abc"
    assert right == "def"


# --- merge ----------------------------------------------------------


def test_merge_either_running_raises_K3C(writer) -> None:
    _seed_cell(writer, "a", "first", bound_agent_id="alpha")
    _seed_cell(writer, "b", "second", bound_agent_id="beta")
    sup = _StubSupervisor({"beta": _StubHandle(state="running")})
    cm = CellManager(writer, supervisor=sup)
    with pytest.raises(CellManagerPreconditionError) as exc:
        cm.merge_cells("a", "b")
    assert exc.value.k_code == K3C_RUNNING_CELL_STRUCTURAL_OP_BLOCKED


def test_merge_both_idle_allowed(writer) -> None:
    _seed_cell(writer, "a", "first", bound_agent_id=None)
    _seed_cell(writer, "b", "second", bound_agent_id=None)
    sup = _StubSupervisor({})
    cm = CellManager(writer, supervisor=sup)
    merged = cm.merge_cells("a", "b")
    assert "first" in merged and "second" in merged


# --- delete ---------------------------------------------------------


def test_delete_running_cell_raises_K3C(writer) -> None:
    _seed_cell(writer, "c1", "body", bound_agent_id="alpha")
    sup = _StubSupervisor({"alpha": _StubHandle(state="running")})
    cm = CellManager(writer, supervisor=sup)
    with pytest.raises(CellManagerPreconditionError) as exc:
        cm.delete_cell("c1")
    assert exc.value.k_code == K3C_RUNNING_CELL_STRUCTURAL_OP_BLOCKED


def test_delete_idle_cell_succeeds(writer) -> None:
    _seed_cell(writer, "c1", "body", bound_agent_id=None)
    sup = _StubSupervisor({})
    cm = CellManager(writer, supervisor=sup)
    assert cm.delete_cell("c1") is True
    assert writer.get_cell_text("c1") is None


# --- set_cell_kind --> K3D (separate code) --------------------------


def test_set_cell_kind_running_raises_K3D(writer) -> None:
    _seed_cell(writer, "c1", "@@agent alpha\nbody", bound_agent_id="alpha")
    sup = _StubSupervisor({"alpha": _StubHandle(state="running")})
    cm = CellManager(writer, supervisor=sup)
    with pytest.raises(CellManagerPreconditionError) as exc:
        cm.set_cell_kind("c1", "scratch")
    assert exc.value.k_code == K3D_RUNNING_CELL_KIND_CHANGE_BLOCKED


def test_set_cell_kind_idle_succeeds(writer) -> None:
    _seed_cell(writer, "c1", "@@agent alpha\nbody", bound_agent_id="alpha")
    sup = _StubSupervisor({"alpha": _StubHandle(state="terminated")})
    cm = CellManager(writer, supervisor=sup)
    out = cm.set_cell_kind("c1", "scratch")
    assert out.startswith("@@scratch")


# --- move -----------------------------------------------------------


def test_move_running_cell_raises_K3C(writer) -> None:
    _seed_cell(writer, "c1", "body", bound_agent_id="alpha")
    sup = _StubSupervisor({"alpha": _StubHandle(state="running")})
    cm = CellManager(writer, supervisor=sup)
    with pytest.raises(CellManagerPreconditionError) as exc:
        cm.move_cell("c1", 0)
    assert exc.value.k_code == K3C_RUNNING_CELL_STRUCTURAL_OP_BLOCKED


def test_move_idle_cell_returns_index(writer) -> None:
    _seed_cell(writer, "c1", "body", bound_agent_id=None)
    sup = _StubSupervisor({})
    cm = CellManager(writer, supervisor=sup)
    assert cm.move_cell("c1", 7) == 7


# --- set_cell_text on running cell: ALLOWED + K3F info marker ------


def test_set_cell_text_on_running_cell_allowed(writer) -> None:
    _seed_cell(writer, "c1", "old", bound_agent_id="alpha")
    sup = _StubSupervisor({"alpha": _StubHandle(state="running")})
    cm = CellManager(writer, supervisor=sup)
    cm.set_cell_text("c1", "new body")
    assert writer.get_cell_text("c1") == "new body"


def test_set_cell_text_running_emits_K3F_marker(writer, monkeypatch) -> None:
    """The text-edit-during-run path emits K3F at info level."""
    _seed_cell(writer, "c1", "old", bound_agent_id="alpha")
    sup = _StubSupervisor({"alpha": _StubHandle(state="running")})
    cm = CellManager(writer, supervisor=sup)
    captured: list = []

    def _stub_mark(stage: str, **kw) -> None:
        captured.append((stage, kw))

    monkeypatch.setattr(_diagnostics, "mark", _stub_mark)
    cm.set_cell_text("c1", "new body")
    assert any(
        kw.get("k_class") == K3F_RUNNING_CELL_EDIT_TEXT_ONLY_PATH
        for _, kw in captured
    )


def test_set_cell_text_idle_no_K3F(writer, monkeypatch) -> None:
    _seed_cell(writer, "c1", "old", bound_agent_id=None)
    sup = _StubSupervisor({})
    cm = CellManager(writer, supervisor=sup)
    captured: list = []

    def _stub_mark(stage: str, **kw) -> None:
        captured.append((stage, kw))

    monkeypatch.setattr(_diagnostics, "mark", _stub_mark)
    cm.set_cell_text("c1", "new body")
    assert not any(
        kw.get("k_class") == K3F_RUNNING_CELL_EDIT_TEXT_ONLY_PATH
        for _, kw in captured
    )


# --- reset_contamination on running cell raises K3C -----------------


def test_reset_contamination_running_raises_K3C(writer) -> None:
    _seed_cell(writer, "c1", "body", bound_agent_id="alpha")
    # Mark contaminated so the reset has something to clear.
    writer.flag_cells_contaminated_by_agent(
        agent_id="alpha", line="@@spawn evil",
        source="stdout", layer="plain",
    )
    sup = _StubSupervisor({"alpha": _StubHandle(state="running")})
    cm = CellManager(writer, supervisor=sup)
    with pytest.raises(CellManagerPreconditionError) as exc:
        cm.reset_contamination("c1")
    assert exc.value.k_code == K3C_RUNNING_CELL_STRUCTURAL_OP_BLOCKED


# --- predicate helpers ----------------------------------------------


def test_is_cell_running_no_bound_agent_returns_false(writer) -> None:
    _seed_cell(writer, "c1", "x", bound_agent_id=None)
    sup = _StubSupervisor({"alpha": _StubHandle(state="running")})
    cm = CellManager(writer, supervisor=sup)
    assert cm._is_cell_running("c1") is False


def test_is_cell_running_unknown_cell_returns_false(writer) -> None:
    sup = _StubSupervisor({"alpha": _StubHandle(state="running")})
    cm = CellManager(writer, supervisor=sup)
    assert cm._is_cell_running("missing") is False


def test_is_cell_running_handle_missing_returns_false(writer) -> None:
    _seed_cell(writer, "c1", "x", bound_agent_id="ghost")
    sup = _StubSupervisor({})  # no handle for "ghost"
    cm = CellManager(writer, supervisor=sup)
    assert cm._is_cell_running("c1") is False

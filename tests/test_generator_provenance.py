"""Provenance preservation tests — PLAN-S5.0.2 §6."""

from __future__ import annotations

import pytest

from llm_kernel.cell_manager import (
    CellManager,
    CellManagerPreconditionError,
    K3J_GENERATOR_PROVENANCE_MISSING,
)
from llm_kernel.magic_generators import dispatch_generator
from llm_kernel.metadata_writer import MetadataWriter


@pytest.fixture
def writer(tmp_path) -> MetadataWriter:
    w = MetadataWriter(workspace_root=tmp_path)
    w.set_cell_text("c_gen", "@@template greet")
    return w


@pytest.fixture
def cell_manager(writer) -> CellManager:
    return CellManager(writer)


def test_generated_cells_carry_provenance(writer, cell_manager) -> None:
    writer.set_config_template(
        "greet", "@@scratch hello\n@@break\n@@scratch world",
    )
    args = {"positional": ["greet"], "named": {}}
    new_ids = dispatch_generator(
        "c_gen", "template", args, "", writer, cell_manager,
    )
    for cid in new_ids:
        rec = writer.get_cell_record(cid)
        assert rec["generated_by"] == "c_gen"
        assert isinstance(rec["generated_at"], str)
        assert rec["generated_at"].endswith("Z")


def test_round_trip_through_snapshot_preserves_provenance(
    writer, cell_manager,
) -> None:
    writer.set_config_template(
        "greet", "@@scratch hello\n@@break\n@@scratch world",
    )
    args = {"positional": ["greet"], "named": {}}
    new_ids = dispatch_generator(
        "c_gen", "template", args, "", writer, cell_manager,
    )
    snap = writer._build_snapshot()
    cells_out = snap.get("cells") or {}
    for cid in new_ids:
        rec = cells_out[cid]
        assert rec["generated_by"] == "c_gen"
        assert rec["generated_at"]
        assert "text" in rec


def test_chain_depth_2_preserves_immediate_parent(writer, cell_manager) -> None:
    """A generator that emits a generator-cell preserves immediate parent."""
    writer.set_config_template("inner", "@@scratch leaf_one")
    writer.set_config_template("outer", "@@template inner")
    # First-level dispatch — emits a @@template inner cell.
    args_outer = {"positional": ["outer"], "named": {}}
    level1_ids = dispatch_generator(
        "c_gen", "template", args_outer, "", writer, cell_manager,
    )
    assert len(level1_ids) == 1
    inner_gen_id = level1_ids[0]
    rec_inner_gen = writer.get_cell_record(inner_gen_id)
    assert rec_inner_gen["generated_by"] == "c_gen"
    # Second-level dispatch — emits a leaf cell.
    args_inner = {"positional": ["inner"], "named": {}}
    level2_ids = dispatch_generator(
        inner_gen_id, "template", args_inner, "", writer, cell_manager,
    )
    assert len(level2_ids) == 1
    leaf_id = level2_ids[0]
    rec_leaf = writer.get_cell_record(leaf_id)
    # The immediate parent is the inner generator cell, NOT c_gen.
    assert rec_leaf["generated_by"] == inner_gen_id


def test_insert_without_generated_by_raises_K3J(writer, cell_manager) -> None:
    with pytest.raises(CellManagerPreconditionError) as exc:
        cell_manager.insert_cells_with_provenance(
            after_cell_id="c_gen",
            magic_texts=["@@scratch x"],
            generated_by="",
            generated_at="2025-01-01T00:00:00Z",
        )
    assert exc.value.k_code == K3J_GENERATOR_PROVENANCE_MISSING


def test_insert_without_generated_at_raises_K3J(writer, cell_manager) -> None:
    with pytest.raises(CellManagerPreconditionError) as exc:
        cell_manager.insert_cells_with_provenance(
            after_cell_id="c_gen",
            magic_texts=["@@scratch x"],
            generated_by="c_gen",
            generated_at="",
        )
    assert exc.value.k_code == K3J_GENERATOR_PROVENANCE_MISSING


def test_writer_rejects_unknown_generated_by(writer) -> None:
    """Writer-level validation rejects generated_by referencing unknown cell."""
    with pytest.raises(ValueError) as exc:
        writer.insert_generated_cell(
            "c_new", "@@scratch x",
            after_cell_id="c_gen",
            generated_by="c_does_not_exist",
            generated_at="2025-01-01T00:00:00Z",
        )
    assert "unknown cell" in str(exc.value)

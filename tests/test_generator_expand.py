"""Tests for the ``@@expand`` generator — PLAN-S5.0.2 §5.2."""

from __future__ import annotations

import pytest

from llm_kernel.cell_manager import CellManager
from llm_kernel.magic_generators import (
    GeneratorError,
    K30_GENERATOR_INPUT_INVALID,
    dispatch_generator,
)
from llm_kernel.metadata_writer import MetadataWriter


@pytest.fixture
def writer(tmp_path) -> MetadataWriter:
    w = MetadataWriter(workspace_root=tmp_path)
    w.set_cell_text("c_gen", "@@expand")
    return w


@pytest.fixture
def cell_manager(writer) -> CellManager:
    return CellManager(writer)


def test_expand_happy_path(writer, cell_manager) -> None:
    body = "@@scratch first cell\n@@break\n@@scratch second cell"
    args = {"positional": [], "named": {}}
    new_ids = dispatch_generator(
        "c_gen", "expand", args, body, writer, cell_manager,
    )
    assert len(new_ids) == 2
    rec0 = writer.get_cell_record(new_ids[0])
    rec1 = writer.get_cell_record(new_ids[1])
    assert "first cell" in rec0["text"]
    assert "second cell" in rec1["text"]
    assert rec0["generated_by"] == "c_gen"
    assert rec1["generated_by"] == "c_gen"


def test_expand_empty_body_raises_K30(writer, cell_manager) -> None:
    args = {"positional": [], "named": {}}
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "expand", args, "", writer, cell_manager,
        )
    assert exc.value.code == K30_GENERATOR_INPUT_INVALID
    assert "non_empty_body" in exc.value.reason


def test_expand_whitespace_only_body_raises_K30(writer, cell_manager) -> None:
    args = {"positional": [], "named": {}}
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "expand", args, "   \n  \n", writer, cell_manager,
        )
    assert exc.value.code == K30_GENERATOR_INPUT_INVALID


def test_expand_bad_fragment_atomic_rejection(writer, cell_manager) -> None:
    """A fragment that fails parse rejects the entire invocation."""
    body = "@@scratch ok\n@@break\n@@nonexistent_kind bad"
    args = {"positional": [], "named": {}}
    # Parser raises K31 on unknown @@<kind>; CellManager's pre-flight
    # parse sweep catches it atomically.
    with pytest.raises(Exception):
        dispatch_generator(
            "c_gen", "expand", args, body, writer, cell_manager,
        )
    # No generated cells — only the seeded c_gen.
    assert list(writer._cells.keys()) == ["c_gen"]


def test_expand_single_cell(writer, cell_manager) -> None:
    """Body with no @@break is a single fragment."""
    body = "@@scratch only_one"
    args = {"positional": [], "named": {}}
    new_ids = dispatch_generator(
        "c_gen", "expand", args, body, writer, cell_manager,
    )
    assert len(new_ids) == 1

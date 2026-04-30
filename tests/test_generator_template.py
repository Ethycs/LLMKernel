"""Tests for the ``@@template`` generator — PLAN-S5.0.2 §5.1."""

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
    # Seed a generator cell that the dispatcher inserts AFTER. The
    # generator cell must already exist in writer._cells so the
    # provenance back-pointer references a known cell.
    w.set_cell_text("c_gen", "@@template greet")
    return w


@pytest.fixture
def cell_manager(writer) -> CellManager:
    return CellManager(writer)


def test_template_happy_path_kwarg_substitution(writer, cell_manager) -> None:
    writer.set_config_template(
        "greet",
        "@@scratch hello ${name}\n@@break\n@@scratch goodbye ${name}",
    )
    args = {"positional": ["greet"], "named": {"name": "alpha"}}
    new_ids = dispatch_generator(
        "c_gen", "template", args, "", writer, cell_manager,
    )
    assert len(new_ids) == 2
    rec0 = writer.get_cell_record(new_ids[0])
    rec1 = writer.get_cell_record(new_ids[1])
    assert "hello alpha" in rec0["text"]
    assert "goodbye alpha" in rec1["text"]
    assert rec0["generated_by"] == "c_gen"
    assert rec1["generated_by"] == "c_gen"
    assert rec0["generated_at"]
    assert rec1["generated_at"]


def test_template_missing_template_raises_K30(writer, cell_manager) -> None:
    args = {"positional": ["nonexistent"], "named": {}}
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "template", args, "", writer, cell_manager,
        )
    assert exc.value.code == K30_GENERATOR_INPUT_INVALID
    assert "unknown_template" in exc.value.reason


def test_template_placeholder_unresolved_raises_K30(writer, cell_manager) -> None:
    writer.set_config_template("with_placeholder", "@@scratch ${missing}")
    args = {"positional": ["with_placeholder"], "named": {}}
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "template", args, "", writer, cell_manager,
        )
    assert exc.value.code == K30_GENERATOR_INPUT_INVALID
    assert "placeholder_unresolved" in exc.value.reason


def test_template_no_positional_raises_K30(writer, cell_manager) -> None:
    args = {"positional": [], "named": {}}
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "template", args, "", writer, cell_manager,
        )
    assert exc.value.code == K30_GENERATOR_INPUT_INVALID
    assert "template_name_required" in exc.value.reason


def test_template_fragment_parse_failure_raises(writer, cell_manager) -> None:
    """A template that yields a syntactically-bad fragment rejects all."""
    writer.set_config_template("bad", "@@nonexistent_kind body")
    args = {"positional": ["bad"], "named": {}}
    # Parser raises K31 on unknown @@<kind>; the dispatcher's pre-flight
    # parse_cell sweep surfaces the failure atomically.
    with pytest.raises(Exception):
        dispatch_generator(
            "c_gen", "template", args, "", writer, cell_manager,
        )
    # No generated cells inserted (atomic) — only the seeded c_gen.
    assert list(writer._cells.keys()) == ["c_gen"]


def test_template_yields_no_cells_raises_K30(writer, cell_manager) -> None:
    """Template body that splits to nothing → K30."""
    writer.set_config_template("empty", "")
    args = {"positional": ["empty"], "named": {}}
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "template", args, "", writer, cell_manager,
        )
    assert exc.value.code == K30_GENERATOR_INPUT_INVALID

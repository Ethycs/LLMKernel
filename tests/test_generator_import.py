"""Tests for the ``@@import`` generator — PLAN-S5.0.2 §5.3."""

from __future__ import annotations

import json

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
    w.set_cell_text("c_gen", "@@import other.llmnb")
    return w


@pytest.fixture
def cell_manager(writer) -> CellManager:
    return CellManager(writer)


def _write_llmnb(path, cells_in_order):
    """Write a minimal `.llmnb`-shaped JSON file."""
    data = {
        "metadata": {
            "rts": {
                "cells": {
                    cid: {"text": text} for cid, text in cells_in_order
                },
                "layout": {
                    "tree": {
                        "id": "root",
                        "children": [
                            {"id": cid, "type": "cell"}
                            for cid, _text in cells_in_order
                        ],
                    },
                },
            },
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_import_happy_path(tmp_path, writer, cell_manager) -> None:
    target = tmp_path / "other.llmnb"
    _write_llmnb(target, [
        ("c1", "@@scratch first"),
        ("c2", "@@scratch second"),
        ("c3", "@@scratch third"),
    ])
    args = {"positional": ["other.llmnb"], "named": {}}
    new_ids = dispatch_generator(
        "c_gen", "import", args, "", writer, cell_manager,
    )
    assert len(new_ids) == 3
    rec0 = writer.get_cell_record(new_ids[0])
    rec2 = writer.get_cell_record(new_ids[2])
    assert "first" in rec0["text"]
    assert "third" in rec2["text"]
    assert rec0["generated_by"] == "c_gen"


def test_import_missing_file_raises_K30(tmp_path, writer, cell_manager) -> None:
    args = {"positional": ["does_not_exist.llmnb"], "named": {}}
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "import", args, "", writer, cell_manager,
        )
    assert exc.value.code == K30_GENERATOR_INPUT_INVALID
    assert "import_file_missing" in exc.value.reason


def test_import_non_llmnb_raises_K30(tmp_path, writer, cell_manager) -> None:
    bad = tmp_path / "garbage.json"
    bad.write_text("{\"hello\": \"world\"}", encoding="utf-8")
    args = {"positional": ["garbage.json"], "named": {}}
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "import", args, "", writer, cell_manager,
        )
    assert exc.value.code == K30_GENERATOR_INPUT_INVALID
    assert "not_llmnb" in exc.value.reason or "no metadata" in exc.value.reason


def test_import_no_path_raises_K30(writer, cell_manager) -> None:
    args = {"positional": [], "named": {}}
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "import", args, "", writer, cell_manager,
        )
    assert exc.value.code == K30_GENERATOR_INPUT_INVALID
    assert "requires_file" in exc.value.reason


def test_import_path_traversal_rejected(tmp_path, writer, cell_manager) -> None:
    """``..`` traversal that escapes workspace_root → K30."""
    args = {"positional": ["../escaped.llmnb"], "named": {}}
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "import", args, "", writer, cell_manager,
        )
    assert exc.value.code == K30_GENERATOR_INPUT_INVALID
    assert "escapes_workspace" in exc.value.reason


def test_import_cycle_detection_single_level(tmp_path, writer, cell_manager) -> None:
    """File already in import_chain → K30 cycle detected."""
    target = tmp_path / "other.llmnb"
    _write_llmnb(target, [("c1", "@@scratch x")])
    chain = {str(target.resolve())}
    args = {"positional": ["other.llmnb"], "named": {}}
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "import", args, "", writer, cell_manager,
            import_chain=chain,
        )
    assert exc.value.code == K30_GENERATOR_INPUT_INVALID
    assert "cycle_detected" in exc.value.reason


def test_import_invalid_json_raises_K30(tmp_path, writer, cell_manager) -> None:
    bad = tmp_path / "broken.llmnb"
    bad.write_text("not valid json {", encoding="utf-8")
    args = {"positional": ["broken.llmnb"], "named": {}}
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "import", args, "", writer, cell_manager,
        )
    assert exc.value.code == K30_GENERATOR_INPUT_INVALID
    assert "not_json" in exc.value.reason

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


# ---------------------------------------------------------------------------
# PLAN-S5.0.5 — multi-format @@import (.magic / .ipynb)
# ---------------------------------------------------------------------------


def test_import_magic_format_happy_path(tmp_path, writer, cell_manager) -> None:
    """@@import sample.magic → cells split at @@break, inserted with provenance."""
    target = tmp_path / "sample.magic"
    target.write_text(
        "@@scratch\nfirst\n@@break\n@@scratch\nsecond\n",
        encoding="utf-8",
    )
    args = {"positional": ["sample.magic"], "named": {}}
    new_ids = dispatch_generator(
        "c_gen", "import", args, "", writer, cell_manager,
    )
    assert len(new_ids) == 2
    rec0 = writer.get_cell_record(new_ids[0])
    rec1 = writer.get_cell_record(new_ids[1])
    assert "first" in rec0["text"]
    assert "second" in rec1["text"]
    assert rec0["generated_by"] == "c_gen"
    assert rec1["generated_by"] == "c_gen"


def test_import_ipynb_format_happy_path(tmp_path, writer, cell_manager) -> None:
    """@@import sample.ipynb → code cells map to @@scratch, markdown to @@markdown."""
    ipynb = {
        "cells": [
            {"cell_type": "code", "source": "print('hello')", "outputs": [],
             "execution_count": None, "metadata": {}},
            {"cell_type": "markdown", "source": "# heading", "metadata": {}},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = tmp_path / "sample.ipynb"
    target.write_text(json.dumps(ipynb), encoding="utf-8")
    args = {"positional": ["sample.ipynb"], "named": {}}
    new_ids = dispatch_generator(
        "c_gen", "import", args, "", writer, cell_manager,
    )
    assert len(new_ids) == 2
    rec0 = writer.get_cell_record(new_ids[0])
    rec1 = writer.get_cell_record(new_ids[1])
    assert rec0["text"].startswith("@@scratch")
    assert "print('hello')" in rec0["text"]
    assert rec1["text"].startswith("@@markdown")
    assert "heading" in rec1["text"]


def test_import_explicit_format_overrides_extension(
    tmp_path, writer, cell_manager,
) -> None:
    """`format:` kwarg supersedes extension-based detection."""
    target = tmp_path / "sample.magic"
    target.write_text(
        "@@scratch\nfrom-magic-via-explicit-kwarg\n",
        encoding="utf-8",
    )
    args = {"positional": ["sample.magic"], "named": {"format": "magic"}}
    new_ids = dispatch_generator(
        "c_gen", "import", args, "", writer, cell_manager,
    )
    assert len(new_ids) == 1
    assert "from-magic-via-explicit-kwarg" in writer.get_cell_record(new_ids[0])["text"]


def test_import_explicit_format_mismatch_routes_to_specified_format(
    tmp_path, writer, cell_manager,
) -> None:
    """A .magic-extension file with format:"llmnb" tries JSON parse → K30."""
    target = tmp_path / "looks_like_magic.magic"
    target.write_text("@@scratch\nbody\n", encoding="utf-8")
    args = {
        "positional": ["looks_like_magic.magic"],
        "named": {"format": "llmnb"},
    }
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "import", args, "", writer, cell_manager,
        )
    assert exc.value.code == K30_GENERATOR_INPUT_INVALID
    assert "not_json" in exc.value.reason


def test_import_unsupported_format_raises_K30(
    tmp_path, writer, cell_manager,
) -> None:
    """format:"bogus" rejected before file read."""
    target = tmp_path / "x.llmnb"
    target.write_text("{}", encoding="utf-8")
    args = {"positional": ["x.llmnb"], "named": {"format": "bogus"}}
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "import", args, "", writer, cell_manager,
        )
    assert exc.value.code == K30_GENERATOR_INPUT_INVALID
    assert "unsupported_format" in exc.value.reason


def test_import_magic_empty_file_raises_K30(
    tmp_path, writer, cell_manager,
) -> None:
    """An empty .magic file yields no cells → K30."""
    target = tmp_path / "empty.magic"
    target.write_text("", encoding="utf-8")
    args = {"positional": ["empty.magic"], "named": {}}
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "import", args, "", writer, cell_manager,
        )
    assert exc.value.code == K30_GENERATOR_INPUT_INVALID
    assert "yielded_no_cells" in exc.value.reason

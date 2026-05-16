"""Tests for ``llm_kernel.notebook_format`` — PLAN-S5.0.5 §3.1.

The bulk of these converters were promoted from
``llm_client/notebook.py``; equivalence with the driver-side shim is
covered by the parent repo's ``tests/test_format_converters.py``.

This module focuses on:

* New ``llmnb_to_ipynb`` function (added in S5.0.5).
* Round-trip invariants the magic-driven @@import path depends on.
"""

from __future__ import annotations

import json

from llm_kernel.notebook_format import (
    detect_format,
    ipynb_to_llmnb,
    llmnb_to_ipynb,
    llmnb_to_magic,
    magic_to_llmnb,
)


# ---------------------------------------------------------------------------
# detect_format — extension + content probe
# ---------------------------------------------------------------------------


def test_detect_format_extensions(tmp_path) -> None:
    for ext, expected in [
        (".llmnb", "llmnb"),
        (".ipynb", "ipynb"),
        (".magic", "magic"),
        (".txt", "magic"),
    ]:
        path = tmp_path / f"x{ext}"
        path.write_text("anything", encoding="utf-8")
        assert detect_format(path) == expected, f"extension {ext}"


def test_detect_format_probes_magic_content(tmp_path) -> None:
    path = tmp_path / "no_ext"
    path.write_text("@@scratch\nbody\n", encoding="utf-8")
    assert detect_format(path) == "magic"


def test_detect_format_probes_llmnb_json(tmp_path) -> None:
    path = tmp_path / "no_ext"
    path.write_text(
        json.dumps({"cells": [], "metadata": {"rts": {}}}),
        encoding="utf-8",
    )
    assert detect_format(path) == "llmnb"


def test_detect_format_probes_ipynb_json(tmp_path) -> None:
    path = tmp_path / "no_ext"
    path.write_text(
        json.dumps({"cells": [], "metadata": {"kernelspec": {}}}),
        encoding="utf-8",
    )
    assert detect_format(path) == "ipynb"


def test_detect_format_unknown(tmp_path) -> None:
    path = tmp_path / "no_ext"
    path.write_text("just some prose with no markers", encoding="utf-8")
    assert detect_format(path) == "unknown"


# ---------------------------------------------------------------------------
# magic ↔ llmnb round-trip
# ---------------------------------------------------------------------------


def test_magic_to_llmnb_to_magic_round_trip() -> None:
    src = "@@scratch\nhello\n@@break\n@@scratch\nworld\n"
    llmnb = magic_to_llmnb(src)
    back = llmnb_to_magic(llmnb)
    # The round-trip preserves cell contents and the @@break separator;
    # trailing newline policy is "final \n preserved when cells exist".
    assert "hello" in back
    assert "world" in back
    assert "@@break" in back


def test_magic_to_llmnb_produces_one_cell_per_fragment() -> None:
    src = "@@scratch\na\n@@break\n@@scratch\nb\n@@break\n@@scratch\nc"
    llmnb = magic_to_llmnb(src)
    cells = llmnb["metadata"]["rts"]["cells"]
    assert len(cells) == 3


# ---------------------------------------------------------------------------
# ipynb → llmnb (PLAN-S5.0.3 §6.3)
# ---------------------------------------------------------------------------


def test_ipynb_to_llmnb_code_maps_to_scratch() -> None:
    ipynb = {
        "cells": [{"cell_type": "code", "source": "print('x')", "outputs": [],
                   "execution_count": None, "metadata": {}}],
        "metadata": {},
        "nbformat": 4, "nbformat_minor": 5,
    }
    llmnb = ipynb_to_llmnb(ipynb)
    cells = llmnb["metadata"]["rts"]["cells"]
    assert len(cells) == 1
    first = next(iter(cells.values()))
    assert first["text"].startswith("@@scratch")
    assert "print('x')" in first["text"]


def test_ipynb_to_llmnb_markdown_maps_to_markdown() -> None:
    ipynb = {
        "cells": [{"cell_type": "markdown", "source": "# title", "metadata": {}}],
        "metadata": {},
        "nbformat": 4, "nbformat_minor": 5,
    }
    llmnb = ipynb_to_llmnb(ipynb)
    cells = llmnb["metadata"]["rts"]["cells"]
    first = next(iter(cells.values()))
    assert first["text"].startswith("@@markdown")
    assert "title" in first["text"]


def test_ipynb_to_llmnb_source_as_list() -> None:
    """ipynb cells often store ``source`` as a list of lines."""
    ipynb = {
        "cells": [{"cell_type": "code", "source": ["line1\n", "line2"],
                   "outputs": [], "execution_count": None, "metadata": {}}],
        "metadata": {},
        "nbformat": 4, "nbformat_minor": 5,
    }
    llmnb = ipynb_to_llmnb(ipynb)
    first = next(iter(llmnb["metadata"]["rts"]["cells"].values()))
    assert "line1" in first["text"] and "line2" in first["text"]


# ---------------------------------------------------------------------------
# llmnb → ipynb (NEW in S5.0.5)
# ---------------------------------------------------------------------------


def test_llmnb_to_ipynb_code_cells_preserve_magic_declaration() -> None:
    llmnb = magic_to_llmnb("@@scratch\nbody\n")
    ipynb = llmnb_to_ipynb(llmnb)
    assert ipynb["nbformat"] == 4
    assert len(ipynb["cells"]) == 1
    cell = ipynb["cells"][0]
    assert cell["cell_type"] == "code"
    # Source preserves the @@scratch declaration so a round-trip back
    # through ipynb_to_llmnb yields a scratch cell again.
    assert "@@scratch" in cell["source"]


def test_llmnb_to_ipynb_markdown_strips_declaration() -> None:
    """Markdown cells drop the leading @@markdown line so the rendered
    output is operator-readable in Jupyter."""
    llmnb = magic_to_llmnb("@@markdown\n# heading\n")
    ipynb = llmnb_to_ipynb(llmnb)
    cell = ipynb["cells"][0]
    assert cell["cell_type"] == "markdown"
    assert "@@markdown" not in cell["source"]
    assert "# heading" in cell["source"]


def test_llmnb_to_ipynb_drops_non_jupyter_outputs() -> None:
    """Cell outputs without Jupyter ``output_type`` are dropped silently."""
    llmnb = magic_to_llmnb("@@scratch\nx\n")
    # Inject a non-Jupyter output shape into the cell record.
    rts = llmnb["metadata"]["rts"]
    cell_id = next(iter(rts["cells"]))
    rts["cells"][cell_id]["outputs"] = [
        {"output_type": "stream", "name": "stdout", "text": "ok"},
        {"shape": "custom-not-jupyter"},  # dropped
    ]
    ipynb = llmnb_to_ipynb(llmnb)
    code_cell = ipynb["cells"][0]
    assert len(code_cell["outputs"]) == 1
    assert code_cell["outputs"][0]["output_type"] == "stream"


def test_llmnb_to_ipynb_drops_provenance() -> None:
    """generated_by / generated_at do not appear in the ipynb output."""
    llmnb = magic_to_llmnb("@@scratch\ndata\n")
    rts = llmnb["metadata"]["rts"]
    cell_id = next(iter(rts["cells"]))
    rts["cells"][cell_id]["generated_by"] = "c_parent"
    rts["cells"][cell_id]["generated_at"] = "2026-01-01T00:00:00Z"
    ipynb = llmnb_to_ipynb(llmnb)
    cell = ipynb["cells"][0]
    assert "generated_by" not in (cell.get("metadata") or {})
    assert "generated_at" not in (cell.get("metadata") or {})


def test_llmnb_to_ipynb_then_back_round_trips_code_cells() -> None:
    """The lossy round-trip (llmnb → ipynb → llmnb) preserves cell texts
    for scratch cells. (Markdown round-trips lose the @@markdown
    declaration line; that's the documented lossy behavior.)"""
    src = "@@scratch\nhello\n@@break\n@@scratch\nworld\n"
    llmnb = magic_to_llmnb(src)
    ipynb = llmnb_to_ipynb(llmnb)
    back = ipynb_to_llmnb(ipynb)
    back_cells = list(back["metadata"]["rts"]["cells"].values())
    assert len(back_cells) == 2
    assert "hello" in back_cells[0]["text"]
    assert "world" in back_cells[1]["text"]

"""Tests for ``llm_kernel.file_actions.apply_export`` — PLAN-S5.0.5 Phase 2.

Covers:

* Happy path for each format (.llmnb / .magic / .ipynb).
* K3M — path escapes workspace_root.
* K3N — target exists and overwrite=False.
* K3O — invalid input, unsupported format, unknown extension.
* Format inference from extension.
* Atomic write (no stray .tmp files on the happy path).
* Round-trip with multi-format ``@@import``.
"""

from __future__ import annotations

import json

import pytest

from llm_kernel.file_actions import apply_export, ExportOutcome
from llm_kernel.wire.tools import (
    K3M_NOTEBOOK_EXPORT_PATH_OUTSIDE_WORKSPACE,
    K3N_NOTEBOOK_EXPORT_REFUSED_OVERWRITE,
    K3O_NOTEBOOK_IO_FAILED,
)
from llm_kernel.notebook_format import magic_to_llmnb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def notebook_state() -> dict:
    """A small but realistic ``metadata.rts`` snapshot."""
    nb = magic_to_llmnb(
        "@@scratch\nfirst\n@@break\n@@scratch\nsecond\n@@break\n@@markdown\n# title\n"
    )
    return nb["metadata"]["rts"]


# ---------------------------------------------------------------------------
# Happy paths — one per format
# ---------------------------------------------------------------------------


def test_apply_export_llmnb_happy_path(tmp_path, notebook_state) -> None:
    out = apply_export(
        cell_id="c_export",
        path="out.llmnb",
        format=None,
        overwrite=False,
        notebook_state=notebook_state,
        workspace_root=tmp_path,
    )
    assert out.status == "ok"
    assert out.format == "llmnb"
    assert out.path == (tmp_path / "out.llmnb").resolve()
    assert out.cells_written == 3
    # Round-trip the JSON: re-parse and confirm metadata.rts is preserved.
    written = json.loads((tmp_path / "out.llmnb").read_text(encoding="utf-8"))
    assert written["metadata"]["rts"] == notebook_state


def test_apply_export_magic_happy_path(tmp_path, notebook_state) -> None:
    out = apply_export(
        cell_id="c_export",
        path="out.magic",
        format=None,
        overwrite=False,
        notebook_state=notebook_state,
        workspace_root=tmp_path,
    )
    assert out.status == "ok"
    assert out.format == "magic"
    content = (tmp_path / "out.magic").read_text(encoding="utf-8")
    assert "@@scratch" in content
    assert "first" in content
    assert "@@break" in content
    # Lossy-format warning surfaces.
    assert any("outputs dropped" in w for w in out.warnings)


def test_apply_export_ipynb_happy_path(tmp_path, notebook_state) -> None:
    out = apply_export(
        cell_id="c_export",
        path="out.ipynb",
        format=None,
        overwrite=False,
        notebook_state=notebook_state,
        workspace_root=tmp_path,
    )
    assert out.status == "ok"
    assert out.format == "ipynb"
    data = json.loads((tmp_path / "out.ipynb").read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert len(data["cells"]) == 3
    assert any("provenance dropped" in w for w in out.warnings)


def test_apply_export_explicit_format_overrides_extension(
    tmp_path, notebook_state,
) -> None:
    """``format:`` kwarg supersedes extension-based detection."""
    out = apply_export(
        cell_id="c_export",
        path="out.weird",
        format="magic",
        overwrite=False,
        notebook_state=notebook_state,
        workspace_root=tmp_path,
    )
    assert out.status == "ok"
    assert out.format == "magic"
    content = (tmp_path / "out.weird").read_text(encoding="utf-8")
    assert "@@scratch" in content


# ---------------------------------------------------------------------------
# K3M — path-outside-workspace
# ---------------------------------------------------------------------------


def test_apply_export_path_traversal_rejected(tmp_path, notebook_state) -> None:
    """``../escape`` resolves outside workspace_root → K3M."""
    out = apply_export(
        cell_id="c_export",
        path="../escape.llmnb",
        format=None,
        overwrite=False,
        notebook_state=notebook_state,
        workspace_root=tmp_path,
    )
    assert out.status == "error"
    assert out.k_code == K3M_NOTEBOOK_EXPORT_PATH_OUTSIDE_WORKSPACE
    assert "escapes_workspace" in (out.message or "")
    # No file written outside workspace.
    assert not (tmp_path.parent / "escape.llmnb").exists()


def test_apply_export_absolute_path_outside_rejected(
    tmp_path, notebook_state,
) -> None:
    """Absolute path that doesn't share workspace_root prefix → K3M."""
    # An absolute path under workspace_root IS valid (re-relativized);
    # one that points elsewhere is not.
    out = apply_export(
        cell_id="c_export",
        path=str(tmp_path.parent / "outside.llmnb"),
        format=None,
        overwrite=False,
        notebook_state=notebook_state,
        workspace_root=tmp_path,
    )
    # The join (ws_root / abs_path) yields the abs_path itself; .resolve()
    # then sits outside ws_root.
    assert out.status == "error"
    assert out.k_code == K3M_NOTEBOOK_EXPORT_PATH_OUTSIDE_WORKSPACE


# ---------------------------------------------------------------------------
# K3N — overwrite refused
# ---------------------------------------------------------------------------


def test_apply_export_existing_target_refused_without_overwrite(
    tmp_path, notebook_state,
) -> None:
    target = tmp_path / "out.llmnb"
    target.write_text("preexisting", encoding="utf-8")
    out = apply_export(
        cell_id="c_export",
        path="out.llmnb",
        format=None,
        overwrite=False,
        notebook_state=notebook_state,
        workspace_root=tmp_path,
    )
    assert out.status == "error"
    assert out.k_code == K3N_NOTEBOOK_EXPORT_REFUSED_OVERWRITE
    # Original content untouched.
    assert target.read_text(encoding="utf-8") == "preexisting"


def test_apply_export_existing_target_replaced_with_overwrite(
    tmp_path, notebook_state,
) -> None:
    target = tmp_path / "out.llmnb"
    target.write_text("preexisting", encoding="utf-8")
    out = apply_export(
        cell_id="c_export",
        path="out.llmnb",
        format=None,
        overwrite=True,
        notebook_state=notebook_state,
        workspace_root=tmp_path,
    )
    assert out.status == "ok"
    # New content is JSON-shaped llmnb, not "preexisting".
    new_content = target.read_text(encoding="utf-8")
    assert new_content != "preexisting"
    assert new_content.lstrip().startswith("{")


# ---------------------------------------------------------------------------
# K3O — invalid input / unsupported format / unknown extension
# ---------------------------------------------------------------------------


def test_apply_export_empty_path_returns_K3O(tmp_path, notebook_state) -> None:
    out = apply_export(
        cell_id="c_export",
        path="",
        format=None,
        overwrite=False,
        notebook_state=notebook_state,
        workspace_root=tmp_path,
    )
    assert out.status == "error"
    assert out.k_code == K3O_NOTEBOOK_IO_FAILED
    assert out.cause == "invalid_input"


def test_apply_export_unsupported_format_returns_K3O(
    tmp_path, notebook_state,
) -> None:
    out = apply_export(
        cell_id="c_export",
        path="out.llmnb",
        format="bogus",
        overwrite=False,
        notebook_state=notebook_state,
        workspace_root=tmp_path,
    )
    assert out.status == "error"
    assert out.k_code == K3O_NOTEBOOK_IO_FAILED
    assert out.cause == "unsupported_format"


def test_apply_export_unknown_extension_no_format_returns_K3O(
    tmp_path, notebook_state,
) -> None:
    """Cannot infer format from extension and no explicit ``format:``."""
    out = apply_export(
        cell_id="c_export",
        path="out.unknown",
        format=None,
        overwrite=False,
        notebook_state=notebook_state,
        workspace_root=tmp_path,
    )
    assert out.status == "error"
    assert out.k_code == K3O_NOTEBOOK_IO_FAILED
    assert out.cause == "unsupported_format"


def test_apply_export_non_dict_notebook_state_returns_K3O(tmp_path) -> None:
    out = apply_export(
        cell_id="c_export",
        path="out.llmnb",
        format=None,
        overwrite=False,
        notebook_state="not a dict",  # type: ignore[arg-type]
        workspace_root=tmp_path,
    )
    assert out.status == "error"
    assert out.k_code == K3O_NOTEBOOK_IO_FAILED
    assert out.cause == "invalid_input"


# ---------------------------------------------------------------------------
# Atomic write semantics
# ---------------------------------------------------------------------------


def test_apply_export_no_stray_tmp_file_on_success(
    tmp_path, notebook_state,
) -> None:
    out = apply_export(
        cell_id="c_export",
        path="out.llmnb",
        format=None,
        overwrite=False,
        notebook_state=notebook_state,
        workspace_root=tmp_path,
    )
    assert out.status == "ok"
    # No leftover <name>.tmp file.
    assert not (tmp_path / "out.llmnb.tmp").exists()
    assert (tmp_path / "out.llmnb").exists()


def test_apply_export_creates_subdirectories(tmp_path, notebook_state) -> None:
    out = apply_export(
        cell_id="c_export",
        path="subdir/nested/out.llmnb",
        format=None,
        overwrite=False,
        notebook_state=notebook_state,
        workspace_root=tmp_path,
    )
    assert out.status == "ok"
    assert (tmp_path / "subdir" / "nested" / "out.llmnb").exists()


# ---------------------------------------------------------------------------
# Round-trip via @@export → @@import (proves the symmetry)
# ---------------------------------------------------------------------------


def test_export_then_import_round_trips_cell_text(
    tmp_path, notebook_state,
) -> None:
    """@@export then @@import reproduces cell text byte-for-byte."""
    # Step 1: export.
    out = apply_export(
        cell_id="c_export",
        path="round_trip.llmnb",
        format=None,
        overwrite=False,
        notebook_state=notebook_state,
        workspace_root=tmp_path,
    )
    assert out.status == "ok"

    # Step 2: import via the magic_generators handler (already shipped
    # in Phase 1). We call the function directly with a constructed ctx
    # rather than going through dispatch_generator (which needs a full
    # cell_manager fixture).
    from llm_kernel.magic_generators import _handle_import, GeneratorContext
    from datetime import datetime, timezone

    ctx: GeneratorContext = {
        "cell_id": "c_import",
        "pin": None,
        "workspace_root": tmp_path,
        "config_templates": {},
        "now_iso": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "import_chain": set(),
    }
    args = {"positional": ["round_trip.llmnb"], "named": {}}
    fragments = _handle_import("import", args, "", ctx)

    # Original cell texts:
    original_texts = [
        rec.get("text", "")
        for rec in notebook_state.get("cells", {}).values()
        if isinstance(rec, dict) and rec.get("text", "").strip()
    ]
    assert fragments == original_texts


# ---------------------------------------------------------------------------
# Magic-registry integration — @@export reserves the name
# ---------------------------------------------------------------------------


def test_export_registered_in_cell_magics() -> None:
    from llm_kernel.magic_registry import CELL_MAGICS, RESERVED_NAMES
    assert "export" in CELL_MAGICS
    assert CELL_MAGICS["export"].kind == "export"
    assert CELL_MAGICS["export"].status == "active"
    assert "export" in RESERVED_NAMES


def test_export_parses_args_to_typed_fields() -> None:
    """``@@export path:"x.llmnb" format:"magic" overwrite:true`` populates
    cell.args with typed fields."""
    from llm_kernel.cell_text import parse_cell

    cell = parse_cell(
        '@@export path:"x.llmnb" format:"magic" overwrite:true\n'
    )
    assert cell.kind == "export"
    assert cell.args.get("path") == "x.llmnb"
    assert cell.args.get("format") == "magic"
    assert cell.args.get("overwrite") is True


def test_export_overwrite_defaults_false() -> None:
    from llm_kernel.cell_text import parse_cell

    cell = parse_cell('@@export path:"x.llmnb"\n')
    assert cell.args.get("overwrite") is False

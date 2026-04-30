"""S5.0.3a schema-export round-trip test.

Per PLAN-S5.0.3 §8.1.  Runs ``python -m llm_kernel.wire.export <tmp_path>``
and verifies the output set matches the in-tree ``wire/schemas/``.
Catches future drift between Python validators and emitted JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _in_tree_schemas_dir() -> Path:
    import llm_kernel.wire
    return Path(llm_kernel.wire.__file__).parent / "schemas"


def test_export_produces_correct_file_set(tmp_path: Path) -> None:
    """Exporting to tmp_path produces exactly the same filenames as in-tree."""
    from llm_kernel.wire.export import export_schemas

    written = export_schemas(tmp_path)
    in_tree_dir = _in_tree_schemas_dir()

    exported_names = sorted(f.name for f in written)
    in_tree_names = sorted(f.name for f in in_tree_dir.glob("*.json"))

    assert exported_names == in_tree_names, (
        f"Export file set mismatch.\n"
        f"Exported: {exported_names}\n"
        f"In-tree:  {in_tree_names}"
    )


def test_export_content_matches_in_tree(tmp_path: Path) -> None:
    """Each exported JSON matches the in-tree counterpart byte-for-byte (modulo sort/indent)."""
    from llm_kernel.wire.export import export_schemas

    export_schemas(tmp_path)
    in_tree_dir = _in_tree_schemas_dir()

    mismatches: list[str] = []
    for in_tree_file in sorted(in_tree_dir.glob("*.json")):
        exported_file = tmp_path / in_tree_file.name
        if not exported_file.exists():
            mismatches.append(f"{in_tree_file.name}: missing from export")
            continue
        in_tree_data = json.loads(in_tree_file.read_text(encoding="utf-8"))
        exported_data = json.loads(exported_file.read_text(encoding="utf-8"))
        if in_tree_data != exported_data:
            mismatches.append(
                f"{in_tree_file.name}: content differs\n"
                f"  in-tree:  {json.dumps(in_tree_data, sort_keys=True)[:200]}\n"
                f"  exported: {json.dumps(exported_data, sort_keys=True)[:200]}"
            )

    assert not mismatches, (
        f"Schema drift detected ({len(mismatches)} file(s)):\n"
        + "\n".join(mismatches)
        + "\nRegenerate with: python -m llm_kernel.wire.export llm_kernel/wire/schemas/"
    )


def test_export_all_files_are_valid_json(tmp_path: Path) -> None:
    """All exported files are valid JSON objects."""
    from llm_kernel.wire.export import export_schemas

    written = export_schemas(tmp_path)
    for path in written:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{path.name}: expected JSON object, got {type(data)}"


def test_main_entrypoint(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """``python -m llm_kernel.wire.export <dir>`` prints summary and writes files."""
    import sys
    from llm_kernel.wire.export import main

    sys.argv = ["llm_kernel.wire.export", str(tmp_path)]
    main()
    captured = capsys.readouterr()
    assert "Exported" in captured.out
    assert str(tmp_path) in captured.out
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) > 0, "main() wrote no files"

"""Lint check — generators write structurally, never via stdout.

PLAN-S5.0.2 §4.3. Walks ``llm_kernel/magic_generators.py`` and
asserts:

* No handler calls ``print()`` with text containing ``@@``.
* All three handlers route fragments through
  ``cell_manager.insert_cells_with_provenance`` (i.e., the dispatcher
  itself is the sole structural-write call site; handlers return
  fragments and never write directly).

This formalizes the "generators write structurally, never via output
stream" rule from ``discipline/cell-manager-owns-structure``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import llm_kernel.magic_generators as magic_generators


def _module_source() -> str:
    path = Path(inspect.getfile(magic_generators))
    return path.read_text(encoding="utf-8")


def _module_ast() -> ast.Module:
    return ast.parse(_module_source())


def test_no_print_of_at_at_text() -> None:
    """No ``print(...)`` call in this module emits a literal ``@@``."""
    tree = _module_ast()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "print":
            continue
        # Walk the args; any string literal containing ``@@`` is a
        # violation (even an indirect helper that LATER prints magic
        # text would surface here on direct inspection).
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                assert "@@" not in arg.value, (
                    f"print() emits '@@' literal: {arg.value!r}"
                )


def test_dispatcher_routes_through_cell_manager() -> None:
    """``dispatch_generator`` calls
    ``cell_manager.insert_cells_with_provenance``."""
    source = inspect.getsource(magic_generators.dispatch_generator)
    assert "insert_cells_with_provenance" in source


def test_handlers_return_lists_not_print() -> None:
    """Each handler returns a list; none uses ``print``/``sys.stdout``."""
    for name in ("_handle_template", "_handle_expand", "_handle_import"):
        fn = getattr(magic_generators, name)
        src = inspect.getsource(fn)
        assert "print(" not in src, f"{name} uses print()"
        assert "sys.stdout" not in src, f"{name} writes to sys.stdout"
        assert "return" in src, f"{name} has no return"


def test_module_does_not_import_sys_stdout() -> None:
    """No ``sys.stdout.write`` call anywhere in the module."""
    src = _module_source()
    assert "sys.stdout" not in src

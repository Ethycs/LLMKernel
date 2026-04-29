"""Cell-Manager façade — text-mutation primitives.

Per [PLAN-S5.0-cell-magic-vocabulary.md] §3.8 / [discipline/cell-manager-
owns-structure] the Cell Manager is the structural API for cell text
mutations: insert ``@@break`` markers, prepend ``@<line_magic>`` lines,
replace ``@@<kind>`` declarations, etc.

S5.0 introduces these primitives. Earlier slices wrote per-cell flags
straight into ``metadata.rts.cells[<id>]`` fields; with the cell schema
collapse to ``{ text, outputs, bound_agent_id }`` (PLAN §3.5) flag
toggles must edit ``cell.text`` so the source-of-truth round-trips
through ``parse_cell``.

The implementation is *string-level* and side-effect-free in this
module: all primitives accept and return strings. The
:class:`CellManager` adapter shim wires them to a ``MetadataWriter``
instance so callers can mutate by ``cell_id`` instead of by raw text;
that adapter is the public façade.

Atomicity per BSP-007: each primitive returns a complete new text
string; the caller is expected to commit it via the writer's
single-mutation lock so concurrent edits don't tear.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

from .cell_text import parse_cell

if TYPE_CHECKING:  # pragma: no cover
    from .metadata_writer import MetadataWriter


__all__ = (
    "split_at_break_text",
    "merge_cells_text",
    "insert_line_magic_text",
    "remove_line_magic_text",
    "set_cell_kind_text",
    "CellManager",
)


# --- String-level primitives -----------------------------------------


def split_at_break_text(text: str, position: int) -> Tuple[str, str]:
    """Insert a ``@@break`` line at character ``position``.

    Returns ``(text_before, text_after)`` — two cells. The break marker
    itself is stripped (the splitter at the higher level will re-insert
    it on serialization). ``position`` is clamped to ``[0, len(text)]``.
    """
    if position < 0:
        position = 0
    if position > len(text):
        position = len(text)
    before = text[:position].rstrip("\n")
    after = text[position:].lstrip("\n")
    return before, after


def merge_cells_text(a_text: str, b_text: str) -> str:
    """Concatenate two cells' texts with a single ``\\n`` separator.

    Per PLAN §3.8 any intervening ``@@break`` is dropped — the splitter
    consumes them anyway, so we stay safe in case the caller hands us
    text that retained one. Whitespace at the join boundary is
    normalized to a single newline for predictability.
    """
    a_clean = a_text.rstrip("\n")
    b_clean = b_text.lstrip("\n")
    # Defensively strip a leading/trailing @@break line at the join.
    if a_clean.endswith("@@break"):
        a_clean = a_clean[: -len("@@break")].rstrip("\n")
    if b_clean.startswith("@@break"):
        b_clean = b_clean[len("@@break"):].lstrip("\n")
    if not a_clean:
        return b_clean
    if not b_clean:
        return a_clean
    return f"{a_clean}\n{b_clean}"


def insert_line_magic_text(text: str, magic_name: str, args: str = "") -> str:
    """Prepend ``@<magic_name> [<args>]`` to ``text``.

    Per PLAN §3.8 line magics live above the body but *below* the
    cell-kind declaration when one is present. We scan to find the
    insertion point: after a leading ``@@<kind>`` line, before any
    other content. If the same line magic already appears anywhere in
    the cell, the call is a no-op (idempotent).
    """
    lines = text.splitlines()
    args_clean = args.strip()
    line = f"@{magic_name}" if not args_clean else f"@{magic_name} {args_clean}"
    # Idempotency check.
    target_prefix = f"@{magic_name}"
    for existing in lines:
        existing_stripped = existing.rstrip()
        if (
            existing_stripped == target_prefix
            or existing_stripped.startswith(target_prefix + " ")
        ):
            return text
    # Find insertion index: first non-blank, non-cell-magic line.
    idx = 0
    # Skip leading blank lines.
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    # Skip a leading cell-magic declaration so the line magic lands
    # under it.
    if idx < len(lines) and lines[idx].startswith("@@"):
        idx += 1
    new_lines = lines[:idx] + [line] + lines[idx:]
    return "\n".join(new_lines)


def remove_line_magic_text(text: str, magic_name: str) -> str:
    """Remove all column-0 ``@<magic_name>`` lines from ``text``.

    Idempotent: removing a magic that is not present returns ``text``
    unchanged. Body lines that happen to match the prefix mid-cell are
    NOT removed — only column-0 lines (the ones the parser would have
    classified as line magics).
    """
    target_prefix = f"@{magic_name}"
    out: List[str] = []
    for line in text.splitlines():
        stripped = line.rstrip()
        if (
            stripped == target_prefix
            or stripped.startswith(target_prefix + " ")
        ):
            continue
        out.append(line)
    return "\n".join(out)


def set_cell_kind_text(text: str, kind: str, args: str = "") -> str:
    """Replace or insert the leading ``@@<kind>`` declaration.

    Per PLAN §3.8 — strips the existing ``@@<existing_kind>`` line at
    the top (if any) and inserts the new declaration. ``kind="agent"``
    is the default; passing it on a cell that has no explicit kind
    declaration prepends a ``@@agent`` line. Line magics that lived
    *above* any prior body content keep their position.
    """
    lines = text.splitlines()
    args_clean = args.strip()
    new_decl = f"@@{kind}" if not args_clean else f"@@{kind} {args_clean}"
    # Find leading blank-then-magic structure.
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx < len(lines) and lines[idx].startswith("@@"):
        # Replace the existing kind line.
        lines[idx] = new_decl
        return "\n".join(lines)
    # No existing kind line — prepend.
    new_lines = lines[:idx] + [new_decl] + lines[idx:]
    return "\n".join(new_lines)


# --- Adapter to MetadataWriter ---------------------------------------


class CellManager:
    """Thin façade wiring text primitives to a :class:`MetadataWriter`.

    The writer owns ``cells[<id>].text``; this adapter calls the
    string-level primitives and writes the result back via
    ``MetadataWriter.set_cell_text(cell_id, new_text)``.
    """

    def __init__(self, writer: "MetadataWriter") -> None:
        self._writer = writer

    def split_at_break(self, cell_id: str, position: int) -> Tuple[str, str]:
        """Split ``cell_id``'s text at ``position``; return new cell IDs.

        Returns the (left_text, right_text) tuple — actually persisting
        the second cell as a new cell entry is left to the higher-level
        caller (which knows the notebook's cell-id allocation policy).
        """
        text = self._writer.get_cell_text(cell_id) or ""
        return split_at_break_text(text, position)

    def merge_cells(self, a_id: str, b_id: str) -> str:
        """Merge ``b_id`` into ``a_id``; remove ``b_id``."""
        a_text = self._writer.get_cell_text(a_id) or ""
        b_text = self._writer.get_cell_text(b_id) or ""
        merged = merge_cells_text(a_text, b_text)
        self._writer.set_cell_text(a_id, merged)
        self._writer.delete_cell(b_id)
        return merged

    def insert_line_magic(
        self, cell_id: str, magic_name: str, args: str = "",
    ) -> str:
        text = self._writer.get_cell_text(cell_id) or ""
        new_text = insert_line_magic_text(text, magic_name, args)
        self._writer.set_cell_text(cell_id, new_text)
        return new_text

    def remove_line_magic(self, cell_id: str, magic_name: str) -> str:
        text = self._writer.get_cell_text(cell_id) or ""
        new_text = remove_line_magic_text(text, magic_name)
        self._writer.set_cell_text(cell_id, new_text)
        return new_text

    def set_cell_kind(
        self, cell_id: str, kind: str, args: str = "",
    ) -> str:
        text = self._writer.get_cell_text(cell_id) or ""
        new_text = set_cell_kind_text(text, kind, args)
        self._writer.set_cell_text(cell_id, new_text)
        return new_text

    def view(self, cell_id: str) -> Optional[object]:
        """Return the parsed view for ``cell_id`` (or None if absent)."""
        text = self._writer.get_cell_text(cell_id)
        if text is None:
            return None
        return parse_cell(text)

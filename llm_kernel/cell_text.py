"""Cell-text splitter and parser — BSP-005 S5.0 magic vocabulary.

Per [PLAN-S5.0-cell-magic-vocabulary.md] §3.1 / §3.2 the notebook adopts
an IPython-style two-tier grammar:

* ``@@<cell_magic>`` at column 0 — declares the cell's *kind* (one per
  cell). The shipped V1 cell magics are ``@@agent | @@spawn | @@markdown
  | @@scratch | @@checkpoint | @@endpoint | @@compare | @@section`` plus
  the V2+ reserved ``@@tool | @@artifact | @@native``. ``@@break`` is
  the *only* cell separator (the splitter consumes it; it never reaches
  the parser as a kind).
* ``@<line_magic>`` at column 0 — mutates per-cell flags (e.g. ``@pin``,
  ``@exclude``, ``@affinity primary,cheap``, ``@handoff alpha``).

The cell schema collapses to ``{ text, outputs, bound_agent_id }`` per
PLAN §3.5; ``kind``, ``flags``, ``args`` are *parse-derived* via
:func:`parse_cell` and cached by ``MetadataWriter.cell_view`` with
text-hash invalidation.

This module is pure (no I/O, no logging). The registries are imported
from ``magic_registry`` to avoid an import cycle: this module only
*classifies* lines by syntax, the registry maps known names to handlers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


__all__ = (
    "ParsedCell",
    "split_at_breaks",
    "parse_cell",
    "CellParseError",
    "rewrite_legacy_directives",
)


#: K-class identifiers per PLAN §4 K-class additions.
K30_MULTIPLE_KINDS: str = "K30"
K31_UNKNOWN_CELL_MAGIC: str = "K31"
K34_INCOMPATIBLE_KIND_CHANGE: str = "K34"


class CellParseError(ValueError):
    """Raised on K30/K31/K34 — visible at the parse-cell boundary.

    Carries a structured ``code`` (K30 / K31 / K34) so the caller can
    wrap it in a kernel-side ``intent_failure`` envelope without string
    matching. The exception message starts with the code so log lines
    are immediately classifiable.
    """

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(f"{code}: {reason}")
        self.code: str = code
        self.reason: str = reason


@dataclass
class ParsedCell:
    """The parse-derived view of a cell's canonical ``text``.

    Per PLAN §3.2: ``kind`` defaults to ``"agent"`` when no
    ``@@<cell_magic>`` is declared (the prose-only cell). ``args`` is
    the per-cell-magic structured args (e.g. ``{"agent_id": "alpha"}``
    for ``@@agent alpha``). ``flags`` accumulates set-bit
    ``@<line_magic>`` effects (``"pinned" | "excluded" | …``).
    ``line_magics`` retains the *non-flag* line magics (e.g.
    ``[("affinity", "primary,cheap")]``) for downstream consumers.
    ``body`` is the joined verbatim non-magic content.
    """

    kind: str = "agent"
    args: Dict[str, Any] = field(default_factory=dict)
    flags: Set[str] = field(default_factory=set)
    line_magics: List[Tuple[str, str]] = field(default_factory=list)
    body: str = ""
    #: When True the cell originally had no ``@@<kind>`` declaration; the
    #: parser inferred ``kind="agent"`` (the default). Used by the
    #: schema migrator to decide whether to insert an explicit ``@@agent``
    #: line during the one-shot upgrade pass.
    kind_was_default: bool = True
    #: When the operator typed ``/spawn``/``@<id>:`` (legacy column-0
    #: aliases) we rewrite to canonical magic form *before* parsing per
    #: PLAN §3.9. The parser sets this to True so the round-trip layer
    #: can preserve the original text on emission.
    legacy_alias_used: bool = False


# --- Legacy-directive rewrite ----------------------------------------

# ``/spawn <id> task:"…"`` and ``@<id>: <message>`` are V1 shipped
# directives. PLAN §3.9 specifies they continue to parse — we rewrite
# them to canonical ``@@<kind>`` form before walking. The rewrite is
# *line-local*: a body line that happens to start with ``/spawn`` is NOT
# rewritten because we only inspect line[0] up to the first whitespace
# / colon; if it appears mid-cell after a ``@@<kind>`` declaration the
# parser already classifies it as body.

_LEGACY_SPAWN_RE: re.Pattern[str] = re.compile(
    r'^/spawn\s+(\S+)\s+task:"([^"]*)"\s*$'
)
_LEGACY_AT_ID_RE: re.Pattern[str] = re.compile(
    r"^@([^\s:]+)\s*:\s*([\s\S]+)$"
)


def rewrite_legacy_directives(text: str) -> Tuple[str, bool]:
    """Rewrite legacy column-0 ``/spawn`` and ``@<id>:`` to magic form.

    Per PLAN §3.9 — applied to a *full* cell text before parsing. Only
    the first non-blank line is considered (the legacy grammar is
    line-1-only). Returns ``(rewritten_text, was_legacy)``.

    * ``/spawn alpha task:"X"`` → ``@@spawn alpha task:"X"``
    * ``@alpha: hello\\nmore`` → ``@@agent alpha\\nhello\\nmore``

    When neither pattern matches the input is returned verbatim.
    """
    lines = text.splitlines()
    # Find first non-blank line.
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        return text, False
    first = lines[idx]
    spawn_m = _LEGACY_SPAWN_RE.match(first.strip())
    if spawn_m is not None:
        agent_id, task = spawn_m.group(1), spawn_m.group(2)
        replacement = f'@@spawn {agent_id} task:"{task}"'
        new_lines = list(lines)
        new_lines[idx] = replacement
        return "\n".join(new_lines), True
    at_m = _LEGACY_AT_ID_RE.match(first.strip())
    if at_m is not None:
        agent_id = at_m.group(1)
        body_first = at_m.group(2).strip()
        if not agent_id or not body_first:
            return text, False
        # Rewrite to a two-line head: @@agent <id> + body. Subsequent
        # lines (if any) are appended below. We dedent the leading
        # blanks so the canonical form has a deterministic shape.
        head = [f"@@agent {agent_id}", body_first]
        tail = lines[idx + 1:]
        return "\n".join(head + list(tail)), True
    return text, False


# --- Splitter --------------------------------------------------------


def split_at_breaks(text: str) -> List[str]:
    """Split a notebook's full text at ``@@break`` lines.

    Per PLAN §3.1 — file start and file end are *implicit* breaks.
    Empty cells (back-to-back ``@@break``s, or whitespace-only bodies)
    are dropped. The ``@@break`` lines themselves are consumed; they
    never appear in any returned cell.
    """
    cells: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.strip() == "@@break":
            if current:
                cells.append("\n".join(current))
            current = []
        else:
            current.append(line)
    if current:
        cells.append("\n".join(current))
    return [c for c in cells if c.strip()]


# --- Per-cell parser -------------------------------------------------

# Column-0 detection: the magic must start at character 0 of the line
# (no indentation). Body lines that happen to start with ``@`` after
# leading whitespace are *body*, not magic, per PLAN §6 escape rule.
_CELL_MAGIC_RE: re.Pattern[str] = re.compile(r"^@@([A-Za-z_][\w]*)\s*(.*)$")
_LINE_MAGIC_RE: re.Pattern[str] = re.compile(r"^@([A-Za-z_][\w]*)\s*(.*)$")


def parse_cell(text: str) -> ParsedCell:
    """Parse one cell's text into a :class:`ParsedCell` view.

    Walking rule per PLAN §3.2:

    1. Apply the legacy-directive rewrite first so ``/spawn`` and
       ``@<id>:`` shipping cells parse identically to their canonical
       magic form.
    2. Walk lines top-down. A column-0 ``@@<name>`` line is a *cell
       magic*; the FIRST one sets :attr:`ParsedCell.kind` and parses
       its arg-string via the registry. A SECOND ``@@<known>`` line
       raises K30. An ``@@<unknown>`` at the kind position raises K31.
    3. A column-0 ``@<name>`` line is a *line magic*; the registry's
       handler mutates :attr:`flags` or appends to
       :attr:`line_magics`. Unknown line magics are *body* (so a typed
       email address ``@user`` mid-cell still round-trips).
    4. Anything else is body. The body is joined with ``\\n``,
       preserving the operator's whitespace and blank lines verbatim.

    The parser imports the registries at call time to avoid an import
    cycle (the registry's K32 hook in turn imports back from this
    module's :class:`ParsedCell`).
    """
    # Lazy import: registry calls back into this module's dataclass.
    from .magic_registry import (
        CELL_MAGICS,
        LINE_MAGICS,
        FLAG_SETTING_LINE_MAGICS,
    )

    rewritten, was_legacy = rewrite_legacy_directives(text)
    cell = ParsedCell()
    cell.legacy_alias_used = was_legacy

    body_lines: List[str] = []
    saw_kind = False

    for line in rewritten.splitlines():
        # Column-0 cell magic? `@@<name> [args]` at line[0:2] == '@@'.
        if line.startswith("@@"):
            m = _CELL_MAGIC_RE.match(line)
            if m is not None:
                name, args_str = m.group(1), m.group(2)
                args_str = args_str.strip()
                # ``@@break`` would normally be consumed by the splitter
                # — defensive K30 here in case a downstream caller hands
                # us a single-cell text that still contains one.
                if name == "break":
                    # Defensive: ``@@break`` should never reach the
                    # parser. Treat as no-op (splitter is canonical).
                    continue
                if name not in CELL_MAGICS:
                    # K31 only fires at the *kind* position. After a
                    # valid kind, an unknown ``@@<x>`` is body (escape
                    # for operators who type literal ``@@`` text).
                    if not saw_kind:
                        raise CellParseError(
                            K31_UNKNOWN_CELL_MAGIC,
                            f"unknown cell magic @@{name!r}",
                        )
                    body_lines.append(line)
                    continue
                if saw_kind:
                    raise CellParseError(
                        K30_MULTIPLE_KINDS,
                        f"multiple cell-kind declarations: "
                        f"saw @@{cell.kind!r} then @@{name!r}",
                    )
                handler = CELL_MAGICS[name]
                handler.apply(cell, args_str)
                cell.kind_was_default = False
                saw_kind = True
                continue
            # Otherwise fall through and treat as body (e.g. ``@@1``
            # typo; the regex above only matches identifier-like names).
            body_lines.append(line)
            continue
        if line.startswith("@"):
            m = _LINE_MAGIC_RE.match(line)
            if m is not None:
                name, args_str = m.group(1), m.group(2)
                args_str = args_str.strip()
                if name in LINE_MAGICS:
                    handler = LINE_MAGICS[name]
                    handler.apply(cell, args_str)
                    if name not in FLAG_SETTING_LINE_MAGICS:
                        cell.line_magics.append((name, args_str))
                    continue
            # Unknown ``@<name>`` is body — preserves email addresses,
            # at-mentions in prose, etc.
            body_lines.append(line)
            continue
        body_lines.append(line)

    cell.body = "\n".join(body_lines)
    return cell

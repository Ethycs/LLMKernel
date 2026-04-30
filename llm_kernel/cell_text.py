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
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


__all__ = (
    "ParsedCell",
    "ParserContext",
    "split_at_breaks",
    "parse_cell",
    "CellParseError",
    "rewrite_legacy_directives",
)


#: K-class identifiers per PLAN §4 K-class additions.
K30_MULTIPLE_KINDS: str = "K30"
K31_UNKNOWN_CELL_MAGIC: str = "K31"
K33_MAGIC_HASH_MISMATCH: str = "K33"
K34_INCOMPATIBLE_KIND_CHANGE: str = "K34"
K35_PLAIN_MAGIC_IN_HASH_MODE: str = "K35"


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
class ParserContext:
    """Carrier for hash-mode awareness across :func:`parse_cell`.

    PLAN-S5.0.1b §3.5 PIN-AWARE PARSER. The parser is a pure function;
    the dispatcher / Cell Manager builds a context from the writer's
    ``metadata.rts.config`` (``magic_hash_enabled``,
    ``magic_pin_fingerprint``) plus the registered magic-name set,
    then passes it in. ``pin`` is *not* read from the notebook — it
    arrives via env / OS-keychain in V1 (see ``auth_handlers``); the
    parser only needs the value at parse time.

    Contract:

    * ``hash_mode_on=False`` (or ``parser_context=None``): legacy /
      permissive path. Plain ``@@<name>`` and ``@<name>`` dispatch as
      today. Hashed-shaped lines are body unless they happen to also
      satisfy the legacy regex (they don't, the regex is identifier-
      strict).
    * ``hash_mode_on=True`` AND ``pin`` non-empty: only
      ``@@<hash>:<name> [args]`` and ``@<hash>:<name> [args]``
      dispatch, validated via ``magic_hash.validate_hashed_magic``.
      Plain ``@@<known>`` lines emit K35 (in
      ``cell.k_class_emissions``) and become body. Hashed-shaped
      lines that fail validation emit K33 and become body.
    * ``hash_mode_on=True`` BUT ``pin`` is empty/None: degraded path.
      The notebook says hash mode is on but the kernel has no pin
      loaded; we treat plain magics as K35-body (defensive — a no-pin
      operator session must NOT silently dispatch). The auth
      handlers gate the pin-load surface; this is just the parser's
      defensive fallback.

    The in-memory ``ParsedCell`` holds the recovered ``<name>`` (not
    the hash). Storage retains the canonical hashed form via
    ``cell.text``.
    """

    hash_mode_on: bool = False
    pin: Optional[str] = None
    known_magics: FrozenSet[str] = field(default_factory=frozenset)


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
    #: PLAN-S5.0.1b §3.5 — append-only audit of K-class emissions
    #: produced *during* this cell's parse pass. Each entry:
    #: ``{"code": "K33"|"K35", "line": <str>, "reason": <str>}``.
    #: The parser does NOT raise on K33/K35 (unlike K30/K31 which do
    #: raise via :class:`CellParseError`) — they are advisory and the
    #: dispatcher / drift-detector consumer iterates over them after
    #: a successful parse to decide whether to surface a warning chip
    #: or a contamination flag.
    k_class_emissions: List[Dict[str, str]] = field(default_factory=list)
    #: PLAN-S5.0.2 — True iff ``kind`` resolves to a registered magic
    #: code generator (``@@template`` / ``@@expand`` / ``@@import``).
    #: The dispatcher reads this to route to
    #: :func:`llm_kernel.magic_generators.dispatch_generator` instead
    #: of the regular cell-execution path.
    is_generator: bool = False


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


#: Hashed-magic line shape — ``@@<hash>:<name> [args]`` / ``@<hash>:<name>``.
#: Mirrors :data:`magic_hash.HASHED_MAGIC_LINE` but kept local so the
#: parser doesn't import the magic_hash module unconditionally (the
#: legacy/permissive path doesn't need it).
_HASHED_CELL_MAGIC_RE: re.Pattern[str] = re.compile(
    r"^@@([a-f0-9]+):([A-Za-z_][\w]*)\s*(.*)$"
)
_HASHED_LINE_MAGIC_RE: re.Pattern[str] = re.compile(
    r"^@([a-f0-9]+):([A-Za-z_][\w]*)\s*(.*)$"
)


def parse_cell(
    text: str,
    *,
    parser_context: Optional[ParserContext] = None,
) -> ParsedCell:
    """Parse one cell's text into a :class:`ParsedCell` view.

    Walking rule per PLAN §3.2 (S5.0) + §3.5 PIN-AWARE PARSER (S5.0.1b):

    1. Apply the legacy-directive rewrite first so ``/spawn`` and
       ``@<id>:`` shipping cells parse identically to their canonical
       magic form.
    2. **Hash mode off** (``parser_context=None`` or
       ``parser_context.hash_mode_on=False``): walk lines top-down.
       Plain ``@@<name>`` / ``@<name>`` dispatch via the registry as
       today. Hashed-shaped lines (``@@<hex>:<name>``) are body in
       this path; the legacy parser regex only matches identifier-like
       names, so a hashed line is *naturally* body without a special
       case.
    3. **Hash mode on** (``parser_context.hash_mode_on=True``):
       * ``@@<hash>:<name> [args]`` validates via
         ``magic_hash.validate_hashed_magic``. Match → dispatch by
         ``<name>``. Mismatch (or unknown ``<name>``) → emit K33,
         line becomes body.
       * Plain ``@@<known>`` (no hash) → emit K35, line becomes body.
       * Same shape pair for ``@<hash>:<name>`` line magics.
       * Plain ``@@<unknown>`` at the kind position still raises K31
         (preserves the kind-required invariant; a typo is a typo
         regardless of hash mode).
    4. Anything else is body. The body is joined with ``\\n``,
       preserving the operator's whitespace and blank lines verbatim.

    K30/K31 still raise (cell unparseable). K33/K35 are advisory — the
    parser appends entries to ``cell.k_class_emissions`` and the
    line falls through to body. The dispatcher / drift-detector
    consumer iterates over emissions after a successful parse.

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

    # Hash-mode prelude (must come BEFORE legacy rewrite — the legacy
    # @<id>:<msg> rewrite would otherwise mangle a hashed-magic line
    # whose shape ``@@<hex>:<name> ...`` accidentally matches the
    # legacy pattern). When hash mode is on we skip the legacy
    # rewrite entirely; legacy directives predate hash mode and a
    # hash-mode notebook is canonical-only.
    ctx_probe = parser_context
    _hash_on_probe = bool(ctx_probe is not None and ctx_probe.hash_mode_on)
    if _hash_on_probe:
        rewritten, was_legacy = text, False
    else:
        rewritten, was_legacy = rewrite_legacy_directives(text)
    cell = ParsedCell()
    cell.legacy_alias_used = was_legacy

    # Hash-mode prelude. We treat a context-with-empty-pin as "hash
    # mode active but pin missing" — defensive: plain magics still
    # emit K35 + become body; the parser must NOT silently dispatch
    # in a no-pin session (PLAN §3.5 invariant).
    ctx = parser_context
    hash_mode_on = bool(ctx is not None and ctx.hash_mode_on)
    pin = ctx.pin if (ctx is not None) else None
    known_magics = (ctx.known_magics if ctx is not None else frozenset())
    # In hash mode we need the union of cell + line magic names for
    # the validator's name-set guard. We default to the live registry
    # union when the context didn't pre-supply it (caller convenience).
    if hash_mode_on and not known_magics:
        known_magics = frozenset(set(CELL_MAGICS.keys()) | set(LINE_MAGICS.keys()))

    body_lines: List[str] = []
    saw_kind = False

    # Local helper — record a K33/K35 emission. Caller decides
    # whether to also append to body_lines (always yes for these
    # codes — they're advisory).
    def _emit_k(code: str, line: str, reason: str) -> None:
        cell.k_class_emissions.append({
            "code": code, "line": line[:256], "reason": reason,
        })

    # Lazy validator import — only needed in the hash-mode branch.
    if hash_mode_on:
        from .magic_hash import validate_hashed_magic

    for line in rewritten.splitlines():
        # --- Hash-mode branches ---------------------------------------
        if hash_mode_on:
            # @@<hash>:<name> cell magic.
            if line.startswith("@@"):
                hm = _HASHED_CELL_MAGIC_RE.match(line)
                if hm is not None:
                    if not pin:
                        # Hash mode on but pin missing → defensive K33.
                        _emit_k(
                            K33_MAGIC_HASH_MISMATCH, line,
                            "hash_mode_on_but_pin_missing",
                        )
                        body_lines.append(line)
                        continue
                    ok, recovered = validate_hashed_magic(
                        line, pin, known_magics,
                    )
                    if ok and recovered is not None and recovered in CELL_MAGICS:
                        if recovered == "break":
                            continue
                        if saw_kind:
                            raise CellParseError(
                                K30_MULTIPLE_KINDS,
                                f"multiple cell-kind declarations: "
                                f"saw @@{cell.kind!r} then @@{recovered!r}",
                            )
                        # The args portion is everything AFTER the
                        # ``<name>`` token; the regex's group(3) is
                        # the trimmed remainder.
                        args_str = hm.group(3).strip()
                        handler = CELL_MAGICS[recovered]
                        handler.apply(cell, args_str)
                        cell.kind_was_default = False
                        saw_kind = True
                        continue
                    # Hashed shape but didn't validate — K33 + body.
                    _emit_k(
                        K33_MAGIC_HASH_MISMATCH, line,
                        "invalid_hash_or_unknown_name",
                    )
                    body_lines.append(line)
                    continue
                # @@<plain_name> in hash mode → K35 + body (unless it's
                # an @@<unknown> at kind position, which still raises K31
                # so operator typos surface uniformly).
                m = _CELL_MAGIC_RE.match(line)
                if m is not None:
                    name = m.group(1)
                    if name == "break":
                        continue
                    if name in CELL_MAGICS:
                        _emit_k(
                            K35_PLAIN_MAGIC_IN_HASH_MODE, line,
                            "plain_cell_magic_in_hash_mode",
                        )
                        body_lines.append(line)
                        continue
                    # Unknown @@<x> still raises K31 at kind position.
                    if not saw_kind:
                        raise CellParseError(
                            K31_UNKNOWN_CELL_MAGIC,
                            f"unknown cell magic @@{name!r}",
                        )
                    body_lines.append(line)
                    continue
                body_lines.append(line)
                continue
            # @<hash>:<name> line magic.
            if line.startswith("@"):
                hm = _HASHED_LINE_MAGIC_RE.match(line)
                if hm is not None:
                    if not pin:
                        _emit_k(
                            K33_MAGIC_HASH_MISMATCH, line,
                            "hash_mode_on_but_pin_missing",
                        )
                        body_lines.append(line)
                        continue
                    ok, recovered = validate_hashed_magic(
                        line, pin, known_magics,
                    )
                    if ok and recovered is not None and recovered in LINE_MAGICS:
                        args_str = hm.group(3).strip()
                        handler = LINE_MAGICS[recovered]
                        handler.apply(cell, args_str)
                        if recovered not in FLAG_SETTING_LINE_MAGICS:
                            cell.line_magics.append((recovered, args_str))
                        continue
                    _emit_k(
                        K33_MAGIC_HASH_MISMATCH, line,
                        "invalid_hash_or_unknown_name",
                    )
                    body_lines.append(line)
                    continue
                # Plain @<name> in hash mode → K35 if known, else body.
                m = _LINE_MAGIC_RE.match(line)
                if m is not None and m.group(1) in LINE_MAGICS:
                    _emit_k(
                        K35_PLAIN_MAGIC_IN_HASH_MODE, line,
                        "plain_line_magic_in_hash_mode",
                    )
                    body_lines.append(line)
                    continue
                body_lines.append(line)
                continue
            body_lines.append(line)
            continue

        # --- Legacy / permissive (hash-mode-off) path -----------------
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
    # PLAN-S5.0.2 — mark generator cells so the dispatcher can route
    # to ``magic_generators.dispatch_generator`` instead of the regular
    # cell execution path. We import lazily to avoid the registry
    # bootstrap circular import.
    try:
        from .magic_registry import is_generator as _is_gen
        if _is_gen(cell.kind):
            cell.is_generator = True
    except Exception:  # pragma: no cover - defensive
        pass
    return cell

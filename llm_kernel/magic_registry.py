"""Magic-name registry — BSP-005 S5.0 cell magic vocabulary.

Per [PLAN-S5.0-cell-magic-vocabulary.md] §3.3 / §3.4. The registries
are *static* dicts keyed by magic name (without the ``@`` / ``@@``
sigil); each value is a Handler instance with an :meth:`apply` method
that mutates a :class:`cell_text.ParsedCell` in place.

Two registries:

* :data:`CELL_MAGICS` — column-0 ``@@<name>`` declarations. One per
  cell. Sets ``cell.kind`` and ``cell.args``.
* :data:`LINE_MAGICS` — column-0 ``@<name>`` flag mutators. May appear
  multiple times in one cell. Sets ``cell.flags`` (set-bit flags) or
  appends to ``cell.line_magics`` (parametric flags).

:data:`RESERVED_NAMES` is the union of both registries' keys *plus*
``break`` (the splitter sentinel) — :class:`AgentSupervisor.spawn`
checks an operator-supplied ``agent_id`` against this set and rejects
with K32 if there's a collision.

V1-active handlers fully apply effects. Handlers reserved for S5 / S5.5
/ V1.5 / V2+ slices are *registered* (so the name round-trips and the
K32 reservation holds) but their handlers stub out: cell magics record
``cell.args["_pending"] = True`` and a slice label; line magics still
append to ``line_magics`` so re-emission preserves the operator's text.
The kernel-side dispatcher returns K42 ("not yet implemented") when a
stubbed magic's effect is requested at run time — see
``metadata_writer._dispatch_intent`` for the K42 path.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .cell_text import ParsedCell


__all__ = (
    "CellMagicHandler",
    "LineMagicHandler",
    "CELL_MAGICS",
    "LINE_MAGICS",
    "GENERATORS",
    "RESERVED_NAMES",
    "FLAG_SETTING_LINE_MAGICS",
    "K32_RESERVED_MAGIC_NAME",
    "K42_NOT_YET_IMPLEMENTED",
    "is_generator",
    "is_reserved_name",
    "parse_kv_args",
)


K32_RESERVED_MAGIC_NAME: str = "K32"
K42_NOT_YET_IMPLEMENTED: str = "K42"


# --- Arg-string mini-parser -------------------------------------------

# Per PLAN §3.3 the cell-magic arg shape is loosely ``key:"value"`` or
# ``key:value`` separated by whitespace, with a positional first
# argument allowed (``@@spawn alpha task:"…"``). We use shlex for
# whitespace tokenization and detect the ``key:value`` shape per token.

_KV_RE: re.Pattern[str] = re.compile(r"^([A-Za-z_][\w]*):(.*)$")


def parse_kv_args(args_str: str) -> Tuple[Tuple[str, ...], Dict[str, str]]:
    """Parse a magic's arg string into ``(positional, named)`` parts.

    Whitespace tokenization with shell-like quoting (``"…"`` and
    ``'…'``) so ``task:"design recipe schema"`` survives as one token.
    A token of the form ``key:value`` (where key is an identifier)
    becomes a named arg; everything else is positional in order. The
    parser is permissive: bad shlex (e.g. unmatched quote) falls back
    to a whitespace split so the parse never raises.

    Per PLAN — magics decide for themselves which positional slots
    they expect; the registry only delivers the parsed parts.
    """
    if not args_str:
        return (), {}
    try:
        tokens = shlex.split(args_str, posix=True)
    except ValueError:
        # Bad quoting — fall back to whitespace split. The handler
        # validates downstream.
        tokens = args_str.split()
    positional: list = []
    named: Dict[str, str] = {}
    for tok in tokens:
        m = _KV_RE.match(tok)
        if m is not None:
            key, value = m.group(1), m.group(2)
            named[key] = value
        else:
            positional.append(tok)
    return tuple(positional), named


# --- Handler base classes --------------------------------------------


@dataclass
class CellMagicHandler:
    """One ``@@<name>`` cell-magic registry entry.

    ``status`` is an informational tag — ``"active"`` for V1 magics,
    ``"stub:<slice>"`` for those reserved for a later slice. The
    presence of a stub handler in :data:`CELL_MAGICS` ensures the name
    round-trips (parser accepts it, K32 reserves it against agent IDs)
    even when the runtime effect lands in a future slice.
    """

    name: str
    kind: str  # the cell-kind tag this magic declares
    status: str = "active"
    pending_slice: str = ""

    def apply(self, cell: "ParsedCell", args_str: str) -> None:
        """Mutate ``cell.kind`` / ``cell.args`` from ``args_str``."""
        cell.kind = self.kind
        positional, named = parse_kv_args(args_str)
        cell.args = {
            "_raw_args": args_str,
            "positional": list(positional),
            "named": dict(named),
        }
        if self.status != "active":
            cell.args["_pending"] = True
            cell.args["_pending_slice"] = self.pending_slice
        # Per-magic structured args: subclasses override.
        self._refine_args(cell, positional, named)

    def _refine_args(
        self,
        cell: "ParsedCell",
        positional: Tuple[str, ...],
        named: Dict[str, str],
    ) -> None:
        """Hook for subclasses to extract typed fields."""
        # Default: no refinement.
        return None


@dataclass
class LineMagicHandler:
    """One ``@<name>`` line-magic registry entry.

    Per PLAN §3.4: line magics are *flag mutators*. The base class
    discriminates between *set-bit flags* (``@pin``, ``@exclude``) which
    add a string to ``cell.flags``, and *parametric* line magics
    (``@affinity primary,cheap``, ``@handoff alpha``) which append a
    record to ``cell.line_magics`` (the parser already takes care of
    appending — this handler only implements the flag-mutation side).
    """

    name: str
    flag_name: str = ""  # non-empty -> sets a flag in cell.flags
    flag_unsets: Tuple[str, ...] = ()  # flags this magic *removes*
    status: str = "active"
    pending_slice: str = ""

    def apply(self, cell: "ParsedCell", args_str: str) -> None:
        if self.flag_name:
            cell.flags.add(self.flag_name)
        for flag in self.flag_unsets:
            cell.flags.discard(flag)
        # Parametric magics record into line_magics in the *parser* (so
        # we don't double-record here). Stub status is purely
        # advisory; the parse still succeeds.


# --- Refined cell-magic handlers -------------------------------------


class _AgentCellMagic(CellMagicHandler):
    def _refine_args(self, cell, positional, named):
        if positional:
            cell.args["agent_id"] = positional[0]


class _SpawnCellMagic(CellMagicHandler):
    def _refine_args(self, cell, positional, named):
        if positional:
            cell.args["agent_id"] = positional[0]
        if "task" in named:
            cell.args["task"] = named["task"]
        if "endpoint" in named:
            cell.args["endpoint"] = named["endpoint"]


class _EndpointCellMagic(CellMagicHandler):
    def _refine_args(self, cell, positional, named):
        if positional:
            cell.args["endpoint_name"] = positional[0]
        for k in ("provider", "model", "api_key_env"):
            if k in named:
                cell.args[k] = named[k]


class _CompareCellMagic(CellMagicHandler):
    def _refine_args(self, cell, positional, named):
        if "endpoints" in named:
            cell.args["endpoints"] = [
                e.strip() for e in named["endpoints"].split(",") if e.strip()
            ]


class _SectionCellMagic(CellMagicHandler):
    def _refine_args(self, cell, positional, named):
        if positional:
            cell.args["section_name"] = positional[0]


class _CheckpointCellMagic(CellMagicHandler):
    def _refine_args(self, cell, positional, named):
        if "covers" in named:
            # Plain comma-list — operator types ``covers:[c_3,c_4]`` or
            # ``covers:c_3,c_4``; we strip surrounding ``[]`` and split.
            value = named["covers"].strip().lstrip("[").rstrip("]")
            cell.args["covers"] = [
                s.strip() for s in value.split(",") if s.strip()
            ]


# --- Refined line-magic handlers -------------------------------------


class _MarkLineMagic(LineMagicHandler):
    """``@mark <kind>`` flips ``cell.kind`` to ``<kind>``.

    Per PLAN §3.4 K34 fires when the new kind is incompatible with the
    cell's body content — V1 only validates the kind name itself; the
    full body-compat check is left to the cell-manager call path.
    """

    def apply(self, cell, args_str):
        new_kind = args_str.strip()
        if not new_kind:
            return
        # Validate: target kind must exist as a cell magic. (We can't
        # circular-import metadata_writer.CELL_KINDS here cheaply.)
        if new_kind not in CELL_MAGICS or new_kind == "break":
            from .cell_text import CellParseError, K34_INCOMPATIBLE_KIND_CHANGE
            raise CellParseError(
                K34_INCOMPATIBLE_KIND_CHANGE,
                f"@mark target kind {new_kind!r} is not a known cell kind",
            )
        cell.kind = new_kind


# --- Build registries -------------------------------------------------


CELL_MAGICS: Dict[str, CellMagicHandler] = {
    # break is owned by the splitter; we register it so name reservation
    # picks it up but the parser short-circuits on it.
    "break": CellMagicHandler(name="break", kind="_separator", status="active"),
    "agent": _AgentCellMagic(name="agent", kind="agent"),
    "spawn": _SpawnCellMagic(name="spawn", kind="spawn"),
    "markdown": CellMagicHandler(name="markdown", kind="markdown"),
    "scratch": CellMagicHandler(name="scratch", kind="scratch"),
    "checkpoint": _CheckpointCellMagic(name="checkpoint", kind="checkpoint"),
    "endpoint": _EndpointCellMagic(name="endpoint", kind="endpoint"),
    # V1.5+ — registered so the name reserves but apply marks pending.
    "compare": _CompareCellMagic(
        name="compare", kind="compare", status="stub", pending_slice="V1.5+",
    ),
    # S5.5 — sections.
    "section": _SectionCellMagic(
        name="section", kind="section", status="stub", pending_slice="S5.5",
    ),
    # V2+ reserved kinds — round-trip identically; renderer falls
    # through to kind-label-only view per cell-kinds atom invariants.
    "tool": CellMagicHandler(
        name="tool", kind="tool", status="stub", pending_slice="V2+",
    ),
    "artifact": CellMagicHandler(
        name="artifact", kind="artifact", status="stub", pending_slice="V2+",
    ),
    "native": CellMagicHandler(
        name="native", kind="native", status="stub", pending_slice="V2+",
    ),
    # PLAN-S5.0.2 — magic code generators. The cell-magic registry
    # entry exists so the parser classifies ``@@template foo`` as a
    # cell with ``kind=template`` (and friends); the dispatcher routes
    # to ``magic_generators.dispatch_generator`` based on the
    # ``GENERATORS`` mapping below.
    "template": CellMagicHandler(name="template", kind="template"),
    "expand": CellMagicHandler(name="expand", kind="expand"),
    "import": CellMagicHandler(name="import", kind="import"),
}


LINE_MAGICS: Dict[str, LineMagicHandler] = {
    "pin": LineMagicHandler(name="pin", flag_name="pinned"),
    "unpin": LineMagicHandler(name="unpin", flag_unsets=("pinned",)),
    "exclude": LineMagicHandler(name="exclude", flag_name="excluded"),
    "include": LineMagicHandler(name="include", flag_unsets=("excluded",)),
    "mark": _MarkLineMagic(name="mark"),
    # Parametric magics — handler is a no-op; the parser appends to
    # cell.line_magics so the body of "@affinity primary,cheap" lands
    # there.
    "affinity": LineMagicHandler(name="affinity"),
    "handoff": LineMagicHandler(name="handoff"),
    "status": LineMagicHandler(name="status"),
    # PLAN-S5.0.1b §3.5 — pin lifecycle line magic. Subcommands
    # (set / rotate / off / verify) are dispatched at runtime via
    # ``llm_kernel.auth_handlers.apply_auth_command``; the parser's
    # job is just to record the line in ``cell.line_magics`` so the
    # dispatcher sees it. The handler is intentionally a no-op on
    # apply (no flag mutation) — the side effect is operator-typed
    # and runs through the runtime dispatch surface.
    "auth": LineMagicHandler(name="auth"),
    # S5 stubs — register so the name reserves; apply is a no-op.
    "revert": LineMagicHandler(
        name="revert", status="stub", pending_slice="S5",
    ),
    "stop": LineMagicHandler(
        name="stop", status="stub", pending_slice="S5",
    ),
    "branch": LineMagicHandler(
        name="branch", status="stub", pending_slice="S5",
    ),
}


#: The set of line-magic names whose effect is *only* a flag mutation
#: (``cell.flags``). The parser uses this to decide whether to also
#: append to ``cell.line_magics`` — flag-only magics don't need that
#: additional record because the flag set is the whole story.
FLAG_SETTING_LINE_MAGICS: FrozenSet[str] = frozenset(
    name for name, h in LINE_MAGICS.items()
    if (h.flag_name or h.flag_unsets) and name != "mark"
)


#: Union of cell + line magic names. PLAN §3 reserved-prefix
#: ``llmnb_*`` is *also* reserved (validated by :func:`is_reserved_name`)
#: but enumerating it here would be infinite — handled at validation
#: time.
RESERVED_NAMES: FrozenSet[str] = frozenset(
    set(CELL_MAGICS.keys()) | set(LINE_MAGICS.keys())
)


# --- PLAN-S5.0.2 — magic code generator registry --------------------
#
# The ``GENERATORS`` mapping points each generator-magic name at its
# handler in :mod:`llm_kernel.magic_generators`. Lazy-imported at
# build time to keep this module's import graph minimal (and to avoid
# a circular dep — the generator handlers import back into this
# module's :data:`CELL_MAGICS` for parse-side classification).
#
# V1 ships the three built-ins exactly as listed in
# [magic-code-generator atom](../docs/atoms/concepts/magic-code-generator.md).
# V2+ operator-registered custom generators append to this dict via
# the registration intent (deferred slice).


def _build_generators() -> Dict[str, Any]:
    from . import magic_generators

    return {
        "template": magic_generators._handle_template,
        "expand": magic_generators._handle_expand,
        "import": magic_generators._handle_import,
    }


# Populated lazily on first read via :func:`is_generator` /
# :data:`GENERATORS` access; the build-time function is invoked when
# the module finishes importing.
GENERATORS: Dict[str, Any] = {}


def _ensure_generators_loaded() -> None:
    """Populate :data:`GENERATORS` on first access (lazy import)."""
    global GENERATORS
    if not GENERATORS:
        GENERATORS = _build_generators()


def is_generator(name: str) -> bool:
    """Return True iff ``name`` resolves to a registered generator.

    PLAN-S5.0.2 §4. The dispatcher (``magic_generators.dispatch_generator``)
    consults this to route a parsed ``@@<name>`` cell. Used also by
    ``cell_text.parse_cell`` to mark ``parsed.is_generator = True``
    on classify (so the dispatcher can fast-path the routing decision)
    and by ``agent_supervisor`` to upgrade a contamination K-tag from
    K35 to K3H when the agent emitted a generator-magic-name line.
    """
    if not isinstance(name, str) or not name:
        return False
    _ensure_generators_loaded()
    return name in GENERATORS


def is_reserved_name(name: str) -> bool:
    """Check whether ``name`` is reserved as a magic identifier.

    Per PLAN §3 — ``RESERVED_NAMES`` plus the ``llmnb_*`` future-
    reservation prefix. Used by ``AgentSupervisor.spawn`` (K32) and the
    extension's parser-side validator.
    """
    if not isinstance(name, str) or not name:
        return False
    if name in RESERVED_NAMES:
        return True
    if name.startswith("llmnb_"):
        return True
    return False


# Eager-load the generator registry at import time. We do this AFTER
# this module has finished defining ``CELL_MAGICS`` / ``LINE_MAGICS``
# so the import inside ``_build_generators`` (which pulls
# :mod:`llm_kernel.magic_generators`, which itself imports back into
# this module for classification) doesn't see a partially-built state.
try:
    _ensure_generators_loaded()
except Exception:  # pragma: no cover - defensive
    # If the generator module fails to import for any reason we keep
    # ``GENERATORS`` empty; ``is_generator`` returns False and the
    # parser falls through (the cell-magic registry entry still sets
    # the kind so the cell round-trips).
    pass

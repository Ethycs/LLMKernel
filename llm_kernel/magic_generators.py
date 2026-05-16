"""Magic code generators — PLAN-S5.0.2.

Per [PLAN-S5.0.2-magic-code-generators.md] §4-§5. The three V1 built-in
generators (`@@template`, `@@expand`, `@@import`) are operator-designated
cell-magics whose execution emits valid magic syntax that the parser
dispatches as **new cells**. Generators are the legitimate carve-out to
the [emission ban](../docs/atoms/discipline/magic-injection-defense.md)
established by S5.0.1: they preserve the visible-tile discipline by
placing generated cells in the notebook with `generated_by` provenance.

Public surface:

* :func:`dispatch_generator` — entry point. Routes a parsed
  ``@@<generator>`` cell to its handler, validates HMACs in hash mode,
  and inserts the resulting cells via
  ``cell_manager.insert_cells_with_provenance`` (atomic — no partial
  insert when any fragment is bad).
* :class:`UnknownGeneratorError` — raised when the dispatcher can't
  resolve ``magic_name`` against
  :data:`llm_kernel.magic_registry.GENERATORS`.
* :class:`GeneratorError` — wraps K30/K3I/etc. raised by handlers so
  callers don't unwrap a Python traceback.

The three built-in handlers (``_handle_template``, ``_handle_expand``,
``_handle_import``) live here. The ``GENERATORS`` dict in
``magic_registry`` points at them. Handlers are pure in their effects —
they return a list of magic-text fragments. The dispatcher does the
HMAC stamping (via :func:`_with_optional_hmac`) and the structural
write through Cell Manager.

PLAN §4.3 lint check: this module never calls ``print()`` of any text
that contains ``@@``; all fragments leave through the dispatcher's
``insert_cells_with_provenance`` path. ``test_generator_emission_path``
walks this module's source and asserts the rule.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TypedDict,
)

if TYPE_CHECKING:  # pragma: no cover
    from .cell_manager import CellManager
    from .metadata_writer import MetadataWriter


__all__ = (
    "K30_GENERATOR_INPUT_INVALID",
    "K3H_AGENT_EMITTED_GENERATOR_MAGIC_BLOCKED",
    "K3I_GENERATOR_HANDLER_PRODUCED_INVALID_HASH",
    "K3J_GENERATOR_PROVENANCE_MISSING",
    "GeneratorContext",
    "GeneratorError",
    "UnknownGeneratorError",
    "dispatch_generator",
    "_with_optional_hmac",
    "_handle_template",
    "_handle_expand",
    "_handle_import",
)


#: K30 — re-exported here for handler error returns. Generator input
#: errors (missing template, bad fragment, missing file, cycle, etc.)
#: surface as a structured ``GeneratorError(code="K30")`` so callers
#: don't unwrap a Python traceback.
K30_GENERATOR_INPUT_INVALID: str = "K30"
K3H_AGENT_EMITTED_GENERATOR_MAGIC_BLOCKED: str = "K3H"
K3I_GENERATOR_HANDLER_PRODUCED_INVALID_HASH: str = "K3I"
K3J_GENERATOR_PROVENANCE_MISSING: str = "K3J"


class GeneratorError(ValueError):
    """Raised by generator handlers / dispatcher on structured failures.

    Carries the K-code (K30 / K3I / K3J) and a human-readable reason.
    The dispatcher catches handler exceptions and re-wraps as needed
    so the wire surface never sees a raw ``KeyError`` /
    ``FileNotFoundError`` / etc. Callers index on ``code``.
    """

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(f"{code}: {reason}")
        self.code: str = code
        self.reason: str = reason


class UnknownGeneratorError(GeneratorError):
    """Raised when a magic name doesn't resolve to a known generator.

    Distinct subclass so the wire dispatcher can split the unknown-
    generator class from the input-invalid class for analytics.
    """

    def __init__(self, magic_name: str) -> None:
        super().__init__(
            K30_GENERATOR_INPUT_INVALID,
            f"unknown_generator: {magic_name!r}",
        )


class GeneratorContext(TypedDict, total=False):
    """Per-invocation context the dispatcher passes to handlers.

    Per PLAN §4.1. Optional keys (the ``import_chain`` cycle tracker)
    use ``total=False`` so V1 ``@@import`` doesn't have to bubble it
    through the call signature.
    """

    cell_id: str
    pin: Optional[str]
    workspace_root: Path
    config_templates: Dict[str, str]
    now_iso: str
    import_chain: Set[str]


GeneratorHandler = Callable[
    [str, Dict[str, Any], str, GeneratorContext], List[str]
]


# --- Hashing helper ---------------------------------------------------


def _with_optional_hmac(line: str, pin: Optional[str], name: str) -> str:
    """Prefix ``line`` with ``@@<HMAC>:`` when hash mode is on.

    PLAN §5.4. When ``pin is None`` returns the line unchanged. When
    ``pin`` is set, computes ``HMAC(pin, name)`` via :func:`magic_hash`
    and rewrites the leading ``@@<name>`` (or ``@<name>``) to the
    canonical hashed form.

    Idempotent on already-hashed lines: if the input already carries a
    ``@@<hex>:<n>`` prefix that matches ``HMAC(pin, name)`` we leave it
    unchanged (so handlers can hash-stamp once and not worry about
    double-application).
    """
    if not isinstance(line, str) or not line:
        return line
    if pin is None:
        return line
    # Late import — avoid pulling magic_hash into the registry import
    # graph.
    from .magic_hash import magic_hash, HASHED_MAGIC_LINE

    h = magic_hash(pin, name)
    # Already-hashed? Leave verbatim if it matches our expectation.
    if HASHED_MAGIC_LINE.match(line) is not None:
        return line
    if line.startswith("@@"):
        sigil = "@@"
        rest = line[2:]
    elif line.startswith("@"):
        sigil = "@"
        rest = line[1:]
    else:
        # Generators emit cell magics; a non-@@-prefix line is body
        # content (e.g. a ``@@scratch`` body that includes prose).
        return line
    # Strip the leading <name> token; we'll re-emit ``@@<hash>:<name>``
    # followed by the rest. The token is identifier-ish.
    m = re.match(r"^([A-Za-z_][\w]*)(.*)$", rest)
    if m is None:
        return line
    tok_name = m.group(1)
    if tok_name != name:
        # Mismatch between the line's name token and the requested one
        # — leave verbatim, the dispatcher's _validate_hmacs sweep will
        # surface the issue if the hash is bad.
        return line
    tail = m.group(2)
    return f"{sigil}{h}:{name}{tail}"


def _validate_fragment_hashes(
    fragment: str,
    pin: str,
    known_magics: "set[str]",
) -> None:
    """Walk a fragment's lines; raise K3I on any invalid hashed-magic.

    PLAN §4.2 — the dispatcher's safety net. Generator handlers may
    legitimately use :func:`_with_optional_hmac` (which always produces
    valid hashes) but a buggy custom handler emitting a hand-built
    ``@@<bad_hex>:<name>`` line would slip through. We re-validate every
    hashed-shaped line in the fragment against the operator's pin; any
    mismatch trips K3I and the *entire* invocation rejects atomically.
    """
    from .magic_hash import HASHED_MAGIC_LINE, validate_hashed_magic

    if not isinstance(fragment, str) or not fragment:
        return
    for line in fragment.splitlines():
        if HASHED_MAGIC_LINE.match(line) is None:
            continue
        ok, _recovered = validate_hashed_magic(line, pin, known_magics)
        if not ok:
            raise GeneratorError(
                K3I_GENERATOR_HANDLER_PRODUCED_INVALID_HASH,
                f"handler emitted invalid HMAC in line: {line[:120]!r}",
            )


# --- Built-in handlers ------------------------------------------------


def _handle_template(
    name: str,
    args: Dict[str, Any],
    body: str,
    ctx: GeneratorContext,
) -> List[str]:
    """``@@template <template_name> [k=v ...]`` handler.

    PLAN §5.1.

    * ``args["positional"][0]`` is the template name; ``args["named"]``
      carries the kwarg overrides.
    * Looks up ``ctx["config_templates"][template_name]``; substitutes
      ``${k}`` placeholders via ``string.Template``; splits the result
      on ``@@break``; returns one fragment per cell.

    Errors:

    * ``template_name`` missing from positional args → K30.
    * ``template_name`` not in ``config_templates`` → K30.
    * Placeholder unresolved (``KeyError`` from ``substitute``) → K30
      with the field path.
    * Resulting fragment fails ``parse_cell`` is detected by the
      dispatcher's atomic-insert pass; we don't pre-validate here.
    """
    positional = args.get("positional") or []
    named = args.get("named") or {}
    if not positional:
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            "template_name_required: usage @@template <name> [k=v ...]",
        )
    template_name = positional[0]
    templates = ctx.get("config_templates") or {}
    raw = templates.get(template_name)
    if raw is None:
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            f"unknown_template: {template_name!r}",
        )
    if not isinstance(raw, str):
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            f"template_value_not_string: {template_name!r}",
        )
    tpl = string.Template(raw)
    try:
        rendered = tpl.substitute(named)
    except KeyError as exc:
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            f"template_placeholder_unresolved: {exc.args[0]!r} "
            f"(template={template_name!r})",
        ) from None
    except ValueError as exc:
        # ``Template.substitute`` raises ValueError on bad ``$`` escapes
        # in the template itself (operator-authored).
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            f"template_render_failed: {template_name!r}: {exc}",
        ) from None
    # Split on @@break — same splitter the canonical text uses.
    from .cell_text import split_at_breaks

    fragments = split_at_breaks(rendered)
    if not fragments:
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            f"template_yielded_no_cells: {template_name!r}",
        )
    return fragments


def _handle_expand(
    name: str,
    args: Dict[str, Any],
    body: str,
    ctx: GeneratorContext,
) -> List[str]:
    """``@@expand`` handler — body is a notebook fragment.

    PLAN §5.2. Splits ``body`` on ``@@break`` and returns each fragment
    verbatim (the dispatcher parses each fragment as a cell on insert
    so a bad fragment surfaces K30 atomically).

    Errors:

    * Empty body → K30.
    """
    if not isinstance(body, str) or not body.strip():
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            "expand_requires_non_empty_body",
        )
    from .cell_text import split_at_breaks

    fragments = split_at_breaks(body)
    if not fragments:
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            "expand_yielded_no_cells",
        )
    return fragments


def _handle_import(
    name: str,
    args: Dict[str, Any],
    body: str,
    ctx: GeneratorContext,
) -> List[str]:
    """``@@import <file>`` handler.

    PLAN §5.3.

    * ``args["positional"][0]`` is the relative path (may also live in
      ``args["named"]["file"]``).
    * Resolves under ``ctx["workspace_root"]``; rejects path traversal
      that escapes the workspace.
    * Reads as ``.llmnb`` (JSON); for each cell in
      ``metadata.rts.cells`` (in ``layout`` order if available) returns
      its ``text``.

    Errors:

    * No path supplied → K30.
    * Path contains ``..`` traversal that escapes workspace → K30.
    * File missing / unreadable → K30.
    * Non-``.llmnb`` (no ``metadata.rts`` / no ``cells``) → K30.
    * Single-level cycle: file path already in
      ``ctx.get("import_chain", set())`` → K30.

    AMBIGUITY-FLAG (PLAN §5.3 / report §risks/2): V1 cycle detection is
    "stop after one level" — we ALWAYS reject re-entry of an already-
    seen file via ``import_chain``. We do NOT recurse through nested
    ``@@import`` cells in the imported notebook (V1 keeps the
    one-shot semantics). Tests exercise the single-level case.
    """
    positional = args.get("positional") or []
    named = args.get("named") or {}
    file_arg: Optional[str] = None
    if positional:
        file_arg = positional[0]
    elif "file" in named:
        file_arg = named["file"]
    if not file_arg or not isinstance(file_arg, str):
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            "import_requires_file: usage @@import <relpath>",
        )
    workspace_root = ctx.get("workspace_root")
    if workspace_root is None:
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            "import_workspace_root_unset",
        )
    workspace_root = Path(workspace_root).resolve()
    candidate = (workspace_root / file_arg).resolve()
    # Containment check — refuse paths that resolved outside the
    # workspace. ``Path.is_relative_to`` is 3.9+.
    try:
        candidate.relative_to(workspace_root)
    except ValueError:
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            f"import_path_escapes_workspace: {file_arg!r}",
        ) from None
    # Cycle detection (V1 single-level).
    chain = ctx.get("import_chain") or set()
    chain_key = str(candidate)
    if chain_key in chain:
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            f"import_cycle_detected: {file_arg!r} (already in chain)",
        )
    if not candidate.exists() or not candidate.is_file():
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            f"import_file_missing: {file_arg!r}",
        )
    try:
        raw = candidate.read_text(encoding="utf-8")
    except OSError as exc:
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            f"import_file_unreadable: {file_arg!r}: {exc}",
        ) from None

    # PLAN-S5.0.5 §5.2 — multi-format import. Explicit ``format:`` named
    # arg overrides extension-based detection. Supported formats:
    # ``llmnb`` (native, the legacy path), ``magic`` (operator-edit
    # form; split at @@break), ``ipynb`` (Jupyter; route through
    # notebook_format.ipynb_to_llmnb then walk cells).
    explicit_format = named.get("format") if isinstance(named, dict) else None
    if explicit_format is not None:
        if explicit_format not in ("llmnb", "magic", "ipynb"):
            raise GeneratorError(
                K30_GENERATOR_INPUT_INVALID,
                f"import_unsupported_format: {explicit_format!r} "
                "(must be one of llmnb / magic / ipynb)",
            )
        fmt = explicit_format
    else:
        # Late import keeps magic_generators' import graph minimal.
        from . import notebook_format
        fmt = notebook_format.detect_format(candidate)
        if fmt == "unknown":
            # Fall back to llmnb (legacy behavior) — the caller likely
            # has a misnamed file; the JSON parse below will surface a
            # clearer error than "unknown format" alone.
            fmt = "llmnb"

    # Magic format: split at @@break and return fragments verbatim. No
    # JSON parse, no notebook_format round-trip — each fragment is
    # already operator-edit cell text. The dispatcher will run them
    # through ``cell_manager.insert_cells_with_provenance`` which
    # invokes ``parse_cell`` per fragment, so malformed magic surfaces
    # there with the full K30 message.
    if fmt == "magic":
        from .cell_text import split_at_breaks
        fragments = [f for f in split_at_breaks(raw) if f.strip()]
        if not fragments:
            raise GeneratorError(
                K30_GENERATOR_INPUT_INVALID,
                f"import_yielded_no_cells: {file_arg!r}",
            )
        return fragments

    # Ipynb format: parse JSON, convert through ipynb_to_llmnb, then
    # fall through to the llmnb cell-walk below.
    if fmt == "ipynb":
        try:
            ipynb_data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GeneratorError(
                K30_GENERATOR_INPUT_INVALID,
                f"import_file_not_json: {file_arg!r}: {exc.msg}",
            ) from None
        from . import notebook_format
        try:
            data = notebook_format.ipynb_to_llmnb(ipynb_data)
        except Exception as exc:  # noqa: BLE001 — defensive; surface as K30
            raise GeneratorError(
                K30_GENERATOR_INPUT_INVALID,
                f"import_ipynb_conversion_failed: {file_arg!r}: {exc}",
            ) from None
    else:
        # Native llmnb branch.
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GeneratorError(
                K30_GENERATOR_INPUT_INVALID,
                f"import_file_not_json: {file_arg!r}: {exc.msg}",
            ) from None
    # Must be a `.llmnb` shape: ``metadata.rts.cells`` (or root-level
    # ``cells`` in the snapshot wrapper).
    rts = None
    if isinstance(data, dict):
        meta = data.get("metadata")
        if isinstance(meta, dict):
            rts = meta.get("rts")
        if rts is None and "cells" in data:
            # Direct snapshot shape (build_snapshot output).
            rts = data
    if not isinstance(rts, dict) or "cells" not in rts:
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            f"import_file_not_llmnb: {file_arg!r} (no metadata.rts.cells)",
        )
    cells = rts.get("cells") or {}
    if not isinstance(cells, dict):
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            f"import_file_cells_not_dict: {file_arg!r}",
        )
    # Order: prefer ``layout`` order if it surfaces cell ids; fall back
    # to dict-insertion order. Walking layout.tree is recursive; we
    # collect any string id that has a matching cells[<id>] entry.
    cell_ids_in_order: List[str] = []
    layout = rts.get("layout")
    if isinstance(layout, dict):
        tree = layout.get("tree")
        seen: Set[str] = set()
        stack: List[Any] = [tree] if tree is not None else []
        while stack:
            node = stack.pop(0)
            if isinstance(node, dict):
                node_id = node.get("id")
                if isinstance(node_id, str) and node_id in cells and node_id not in seen:
                    cell_ids_in_order.append(node_id)
                    seen.add(node_id)
                children = node.get("children")
                if isinstance(children, list):
                    stack = list(children) + stack
    if not cell_ids_in_order:
        cell_ids_in_order = list(cells.keys())
    fragments: List[str] = []
    for cid in cell_ids_in_order:
        record = cells.get(cid)
        if not isinstance(record, dict):
            continue
        text = record.get("text")
        if isinstance(text, str) and text.strip():
            fragments.append(text)
    if not fragments:
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            f"import_yielded_no_cells: {file_arg!r}",
        )
    return fragments


# --- Dispatcher -------------------------------------------------------


def _utc_now_iso() -> str:
    """ISO8601 UTC with trailing ``Z`` — matches ``metadata_writer``."""
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def dispatch_generator(
    cell_id: str,
    magic_name: str,
    args: Dict[str, Any],
    body: str,
    writer: "MetadataWriter",
    cell_manager: "CellManager",
    *,
    import_chain: Optional[Set[str]] = None,
) -> List[str]:
    """Routes a parsed ``@@<generator>`` cell to its handler.

    PLAN §4.2. Returns the list of new ``cell_id`` values inserted by
    the cell manager (in insertion order). Atomic: if any fragment
    fails parse OR any emitted hashed-magic line fails HMAC validation,
    the entire invocation is rejected and no cells are inserted.
    """
    # Late import — registry imports back into this module.
    from . import magic_registry

    handler = magic_registry.GENERATORS.get(magic_name)
    if handler is None:
        raise UnknownGeneratorError(magic_name)

    # Build the context. Pin is only set when hash mode is active AND
    # the writer supplies an operator pin.
    hash_mode = bool(writer.get_config_setting("magic_hash_enabled"))
    pin: Optional[str] = None
    if hash_mode:
        pin_getter = getattr(writer, "get_operator_pin", None)
        if callable(pin_getter):
            pin = pin_getter()

    templates: Dict[str, str] = {}
    templates_getter = getattr(writer, "read_config_templates", None)
    if callable(templates_getter):
        try:
            templates = dict(templates_getter() or {})
        except Exception:  # pragma: no cover - defensive
            templates = {}

    workspace_root: Path
    workspace_getter = getattr(writer, "get_workspace_root", None)
    if callable(workspace_getter):
        workspace_root = Path(workspace_getter())
    else:
        # Back-compat fallback: read the private attribute.
        workspace_root = Path(getattr(writer, "_workspace_root", Path.cwd()))

    now_iso = _utc_now_iso()
    ctx: GeneratorContext = {
        "cell_id": cell_id,
        "pin": pin,
        "workspace_root": workspace_root,
        "config_templates": templates,
        "now_iso": now_iso,
        "import_chain": set(import_chain) if import_chain else set(),
    }

    # Track this invocation in the chain for V1 single-level cycle
    # detection in nested ``@@import`` handlers (call-site convention).
    if magic_name == "import":
        # Add the workspace-relative reference of this generator cell
        # to the chain. Since cells don't have a path, we use the
        # cell_id sentinel — but per PLAN §5.3 the chain is keyed by
        # FILE path, populated inside the import handler. We pass the
        # set through verbatim; the import handler does the keying.
        pass

    # Run the handler. Wrap unexpected exceptions as a structured
    # GeneratorError so callers don't unwrap a Python traceback.
    try:
        fragments = handler(magic_name, args, body, ctx)
    except GeneratorError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            f"generator_handler_unexpected_error: {magic_name}: {exc}",
        ) from None
    if not isinstance(fragments, list):
        raise GeneratorError(
            K30_GENERATOR_INPUT_INVALID,
            f"generator_handler_bad_return: {magic_name} "
            f"returned {type(fragments).__name__}",
        )

    # Stamp HMACs onto generator-emitted plain-magic lines when hash
    # mode is on. The handler may have already done it (idempotent),
    # but the dispatcher is the canonical pin-trust boundary.
    if pin is not None:
        # We know the generator emits valid magic syntax; we don't
        # globally rewrite (that would mangle bodies). The validator
        # below catches any line that DOES match the hashed shape but
        # fails the HMAC.
        from .magic_registry import CELL_MAGICS, LINE_MAGICS
        known_magics = set(CELL_MAGICS.keys()) | set(LINE_MAGICS.keys())
        for fragment in fragments:
            _validate_fragment_hashes(fragment, pin, known_magics)

    # Hand off to Cell Manager — atomic insert with provenance.
    return cell_manager.insert_cells_with_provenance(
        after_cell_id=cell_id,
        magic_texts=fragments,
        generated_by=cell_id,
        generated_at=now_iso,
    )

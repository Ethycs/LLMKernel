"""llm_kernel.notebook.notebook_format — public notebook format converters.

Promoted from ``llm_client/notebook.py`` per PLAN-S5.0.5 §3.1. The
converters are public kernel surface (alongside ``wire`` and
``cell_text``); drivers consume them via that module's allow-list and
the back-compat shim at ``llm_client/notebook.py``.

Public API:

* :func:`detect_format` — sniff a path's notebook format.
* :func:`llmnb_to_magic` — encode parsed .llmnb dict → operator-edit magic text.
* :func:`magic_to_llmnb` — decode magic text → parsed .llmnb dict.
* :func:`ipynb_to_llmnb` — Jupyter notebook → .llmnb (one-way; lossy).
* :func:`llmnb_to_ipynb` — .llmnb → Jupyter notebook (one-way; lossy on provenance).
* :func:`check_rts_integrity` — RFC-005 load-time integrity checks (F4/F5/F9-F12).
* :func:`gc_blobs` — RFC-005 manual blob garbage collection.

Rationale — kernel-speaks-magic (PLAN-S5.0.5 §1):

The format converters operate on the canonical ``cell_text`` parser and
the ``metadata.rts`` schema. Their natural home is the kernel package
where those primitives already live; the prior driver-side location was
historical (S5.0.3 driver extraction landed the converters in
``llm_client`` because the kernel didn't expose them yet). Promotion
here means both kernel-side magic handlers (``@@export`` / multi-format
``@@import`` per PLAN-S5.0.5) and driver-side CLI (``llmnb convert``)
consume the same byte-stable implementation.

Round-trip property:

    llmnb_to_magic(magic_to_llmnb(t)) == t (modulo final newline)

for any ``t`` whose cells are non-empty. Outputs cannot round-trip
through magic-text (magic-text doesn't carry them). Ipynb round-trip
is lossy: provenance (``generated_by`` / ``generated_at``),
``bound_agent_id``, and ``metadata.rts.config`` are dropped on export
and synthesized fresh on import.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Literal

from llm_kernel.text.cell_text import (
    ParserContext,
    parse_cell,
    parser_context_from_config,
    split_at_breaks,
)
from llm_kernel.wire import versions as _versions


__all__ = [
    "detect_format",
    "llmnb_to_magic",
    "magic_to_llmnb",
    "ipynb_to_llmnb",
    "llmnb_to_ipynb",
    "LlmnbIntegrityError",
    "check_rts_integrity",
    "gc_blobs",
]


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_format(
    path: Path,
) -> Literal["llmnb", "magic", "ipynb", "unknown"]:
    """Detect the on-disk notebook format from extension + first-line probe.

    Resolution:

    * ``.llmnb`` extension → ``"llmnb"``.
    * ``.ipynb`` extension → ``"ipynb"``.
    * ``.magic`` or ``.txt`` extension → ``"magic"``.
    * Any other extension or no extension: probe contents.

      - First non-blank line starting with ``@@`` or ``@`` → ``"magic"``.
      - JSON-shaped (first non-blank char is ``{``) with a top-level
        ``"cells"`` key → ``"llmnb"`` if ``metadata.rts`` present,
        else ``"ipynb"``.
      - Otherwise ``"unknown"``.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".llmnb":
        return "llmnb"
    if ext == ".ipynb":
        return "ipynb"
    if ext in (".magic", ".txt"):
        return "magic"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unknown"
    stripped = text.lstrip()
    if not stripped:
        return "unknown"
    first = stripped.splitlines()[0] if stripped else ""
    if first.startswith("@@") or (
        first.startswith("@") and not first.startswith("@@")
    ):
        return "magic"
    if stripped[0] == "{":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return "unknown"
        if isinstance(obj, dict) and "cells" in obj:
            md = obj.get("metadata") or {}
            if isinstance(md, dict) and "rts" in md:
                return "llmnb"
            return "ipynb"
    return "unknown"


# ---------------------------------------------------------------------------
# llmnb ↔ magic
# ---------------------------------------------------------------------------


def _rts_namespace(llmnb: dict) -> dict:
    """Return ``metadata.rts`` from a parsed .llmnb dict (or empty dict)."""
    md = llmnb.get("metadata") or {}
    if not isinstance(md, dict):
        return {}
    rts = md.get("rts") or {}
    return rts if isinstance(rts, dict) else {}


def _layout_walk_ids(layout_tree: Any) -> list[str]:
    """Walk ``layout.tree`` collecting cell ids in document order.

    Returns ids in the order they appear in a depth-first walk of
    ``children`` arrays. Mirrors MetadataWriter.get_cell_layout_order
    semantics (PLAN-S5.0.2 §3) but pure-functional on the snapshot dict.
    """
    if not isinstance(layout_tree, dict):
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    stack: list[Any] = [layout_tree]
    while stack:
        node = stack.pop(0)
        if isinstance(node, dict):
            nid = node.get("id")
            if isinstance(nid, str) and nid not in seen:
                if node.get("kind") != "tree":
                    ordered.append(nid)
                    seen.add(nid)
            children = node.get("children")
            if isinstance(children, list):
                stack = list(children) + stack
    return ordered


def llmnb_to_magic(llmnb: dict) -> str:
    """Convert a parsed .llmnb dict to operator-edit magic text.

    Rules (PLAN-S5.0.3 §6.3):

    * Walk ``metadata.rts.cells`` in ``metadata.rts.layout.tree`` order.
    * Cells unreferenced from the layout follow in dict-insertion order.
    * Each cell's ``text`` is emitted verbatim (it already includes the
      ``@@<kind>`` declaration when set; ``magic_to_llmnb`` round-trip
      preserves whatever the operator typed).
    * Cells are separated by a single ``@@break\\n`` line (one cell per
      magic-text segment per PLAN-S5.0 §3.1).
    * Outputs are dropped (magic-text is operator-edit form, not a
      snapshot).
    """
    rts = _rts_namespace(llmnb)
    cells_dict = rts.get("cells") or {}
    if not isinstance(cells_dict, dict):
        cells_dict = {}
    layout = rts.get("layout") or {}
    layout_tree = layout.get("tree") if isinstance(layout, dict) else None

    ordered_ids = _layout_walk_ids(layout_tree)
    seen = set(ordered_ids)
    for cid in cells_dict:
        if cid not in seen:
            ordered_ids.append(cid)
            seen.add(cid)

    parts: list[str] = []
    for cid in ordered_ids:
        rec = cells_dict.get(cid)
        if not isinstance(rec, dict):
            continue
        text = rec.get("text", "")
        if not isinstance(text, str):
            continue
        if not text.strip():
            continue
        parts.append(text.rstrip("\n"))

    if not parts:
        return ""
    return "\n@@break\n".join(parts) + "\n"


def magic_to_llmnb(
    magic_text: str,
    *,
    base_metadata: dict | None = None,
    parser_context: ParserContext | None = None,
) -> dict:
    """Convert magic-text to a parsed .llmnb dict (a JSON object).

    Splits at ``@@break`` per PLAN-S5.0 §3.1, then for each fragment runs
    ``parse_cell`` to derive ``kind`` for the cell record. Builds a fresh
    ``metadata.rts`` namespace with ``cells`` keyed by deterministic ids
    (``cell-0``, ``cell-1``, …) and a flat ``layout.tree`` containing all
    cells in order.

    ``base_metadata`` (when provided) is deep-copied and used as the
    starting ``rts`` namespace; useful for round-tripping where the
    caller wants to preserve config / agents / event_log.

    ``parser_context`` governs hash mode for the per-fragment parse
    (PLAN-S5.0.1b §3.5). When omitted it is derived from
    ``base_metadata["config"]`` via :func:`parser_context_from_config`,
    so a hash-mode notebook's plain ``@@<name>`` lines classify as
    K35-body here exactly as they do in the kernel.
    """
    fragments = split_at_breaks(magic_text or "")
    if parser_context is None and isinstance(base_metadata, dict):
        parser_context = parser_context_from_config(base_metadata.get("config"))

    cells_dict: dict[str, dict[str, Any]] = {}
    layout_children: list[dict[str, Any]] = []
    # PLAN-S5.5 Phase 5 — section cells render as markdown-typed cells in
    # the nbformat shape so VS Code's native markdown-header folding kicks
    # in on the section range. The original ``@@section <title>`` magic
    # text is preserved verbatim in ``rts.cells[<id>].text`` (canonical
    # operator-edit form); the outer nbformat ``source`` field carries
    # the synthesized ``# <title>`` heading for display + folding. The
    # divergence is intentional: text-is-canonical for the kernel, source
    # is the renderer view. ``display_sources`` is local to this function;
    # it does NOT leak into the persisted ``rts.cells`` snapshot.
    display_sources: dict[str, str] = {}
    for idx, text in enumerate(fragments):
        cell_id = f"cell-{idx}"
        try:
            parsed = parse_cell(text, parser_context=parser_context)
            kind = parsed.kind
            args = parsed.args if isinstance(parsed.args, dict) else {}
            bound_agent_id = args.get("agent_id")
            section_title = args.get("title") if kind == "section" else None
        except Exception:  # noqa: BLE001 — defensive; converter must not crash
            kind = "agent"
            bound_agent_id = None
            section_title = None
        record: dict[str, Any] = {
            "text": text,
            "outputs": [],
            "kind": kind,
        }
        if bound_agent_id and kind in ("agent", "spawn"):
            record["bound_agent_id"] = bound_agent_id
        cells_dict[cell_id] = record
        layout_children.append({"id": cell_id, "kind": "cell"})
        if (
            kind == "section"
            and isinstance(section_title, str)
            and section_title.strip()
        ):
            display_sources[cell_id] = f"# {section_title}"

    if base_metadata is not None and isinstance(base_metadata, dict):
        rts: dict[str, Any] = json.loads(json.dumps(base_metadata))  # deep copy
    else:
        rts = {
            "schema_version": _versions.SCHEMA_VERSION,
            "schema_uri": "https://llmnb.dev/schemas/rts/v1",
            "session_id": "00000000-0000-0000-0000-000000000000",
            "created_at": "1970-01-01T00:00:00Z",
            "snapshot_version": 0,
            "agents": {"nodes": [], "edges": []},
            "config": {},
            "event_log": [],
            "blobs": {},
            "drift_log": [],
        }
    rts["cells"] = cells_dict
    rts["layout"] = {"tree": {"id": "root", "kind": "tree", "children": layout_children}}

    return {
        "cells": [
            {
                # PLAN-S5.5 Phase 5 — section cells with a parseable title
                # render as markdown so VS Code's native fold kicks in.
                # Section cells without a title (bare ``@@section``) stay
                # code-typed so the operator sees the raw magic.
                "cell_type": (
                    "markdown"
                    if cells_dict[cid].get("kind") == "markdown"
                    or (
                        cells_dict[cid].get("kind") == "section"
                        and cid in display_sources
                    )
                    else "code"
                ),
                "source": display_sources.get(cid, cells_dict[cid]["text"]),
                "metadata": {"rts": {"cell": {"kind": cells_dict[cid]["kind"]}}},
                "outputs": [],
                "execution_count": None,
            }
            for cid in cells_dict
        ],
        "metadata": {"rts": rts},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# ipynb ↔ llmnb
# ---------------------------------------------------------------------------


def ipynb_to_llmnb(ipynb: dict) -> dict:
    """One-way Jupyter ipynb → .llmnb conversion (PLAN-S5.0.3 §6.3, §10 risk #5).

    Mapping:

    * ``cell_type == "code"``  → ``@@scratch`` cell (V1 — no agent binding).
    * ``cell_type == "markdown"`` → ``@@markdown`` cell.
    * ``cell_type == "raw"``  → ``@@scratch`` (treated as code-shaped).
    * Outputs are dropped.
    * ``kernelspec`` is NOT preserved (drivers don't run Python; they
      ship envelopes — V2+ may revisit per PLAN §10 risk #5).

    A WARNING summary of dropped data is the caller's responsibility
    (see ``llm_client/cli/convert.py``); this function is pure.
    """
    cells_in = ipynb.get("cells") or []
    if not isinstance(cells_in, list):
        cells_in = []

    fragments: list[str] = []
    for c in cells_in:
        if not isinstance(c, dict):
            continue
        ctype = c.get("cell_type")
        source = c.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if not isinstance(source, str):
            source = str(source)
        source = source.rstrip("\n")
        if ctype == "markdown":
            fragments.append(f"@@markdown\n{source}" if source else "@@markdown")
        elif ctype in ("code", "raw"):
            fragments.append(f"@@scratch\n{source}" if source else "@@scratch")
        else:
            fragments.append(f"@@scratch\n{source}" if source else "@@scratch")

    magic_text = "\n@@break\n".join(fragments)
    return magic_to_llmnb(magic_text)


def llmnb_to_ipynb(llmnb: dict) -> dict:
    """One-way .llmnb → Jupyter ipynb conversion (PLAN-S5.0.5 §3.1).

    Mapping (inverse of :func:`ipynb_to_llmnb`):

    * ``@@markdown`` cells → Jupyter ``cell_type: "markdown"``; source is
      the cell text minus the leading ``@@markdown`` line (if present).
    * All other kinds → ``cell_type: "code"``; source is the verbatim
      cell text (preserves the ``@@<kind>`` declaration so the cell
      round-trips through ``ipynb_to_llmnb`` → ``magic_to_llmnb``).
    * Cell outputs are preserved as Jupyter ``outputs`` when they parse
      as Jupyter-shaped output records; otherwise dropped (warned).
    * ``metadata.rts.cells[<id>].generated_by`` / ``generated_at`` are
      dropped — provenance is not part of the Jupyter schema. Operator
      sees the warning in the export cell output.

    Returns a Jupyter-conformant dict (``nbformat`` 4, minor 5).
    """
    rts = _rts_namespace(llmnb)
    cells_dict = rts.get("cells") or {}
    if not isinstance(cells_dict, dict):
        cells_dict = {}
    layout = rts.get("layout") or {}
    layout_tree = layout.get("tree") if isinstance(layout, dict) else None

    ordered_ids = _layout_walk_ids(layout_tree)
    seen = set(ordered_ids)
    for cid in cells_dict:
        if cid not in seen:
            ordered_ids.append(cid)
            seen.add(cid)

    ipynb_cells: list[dict[str, Any]] = []
    for cid in ordered_ids:
        rec = cells_dict.get(cid)
        if not isinstance(rec, dict):
            continue
        text = rec.get("text", "")
        if not isinstance(text, str):
            continue
        kind = rec.get("kind") or "scratch"
        is_markdown = kind == "markdown"

        if is_markdown:
            # Strip a leading "@@markdown" line so the rendered markdown
            # doesn't carry the magic directive into the Jupyter cell.
            lines = text.splitlines()
            if lines and lines[0].strip() == "@@markdown":
                source = "\n".join(lines[1:])
            else:
                source = text
            cell_type = "markdown"
        else:
            source = text
            cell_type = "code"

        outputs_raw = rec.get("outputs") or []
        if not isinstance(outputs_raw, list):
            outputs_raw = []
        # Preserve only outputs that look Jupyter-shaped (dict with
        # output_type). Anything else is driver/agent-specific and gets
        # dropped silently.
        outputs: list[dict[str, Any]] = []
        for o in outputs_raw:
            if isinstance(o, dict) and "output_type" in o:
                outputs.append(o)

        ipynb_cells.append({
            "cell_type": cell_type,
            "source": source,
            "metadata": {},
            "outputs": outputs if cell_type == "code" else [],
            "execution_count": None if cell_type == "code" else None,
        } if cell_type == "code" else {
            "cell_type": "markdown",
            "source": source,
            "metadata": {},
        })

    return {
        "cells": ipynb_cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# metadata.rts integrity (RFC-005 §"Failure modes") and blob GC
# ---------------------------------------------------------------------------


#: ``$blob:`` sentinel prefix (RFC-005 §"metadata.rts.blobs").
BLOB_SENTINEL_PREFIX = "$blob:"


class LlmnbIntegrityError(ValueError):
    """A ``.llmnb`` file violates an RFC-005 "Reject load" failure mode.

    ``code`` is the RFC-005 failure-mode row (``"F5"``, ``"F9"``,
    ``"F10"``, ``"F11"``); the message ends with ``(RFC-005 <code>)``.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{message} (RFC-005 {code})")
        self.code = code


def _blob_key(value: Any) -> str | None:
    """Return ``sha256:<hex>`` for a ``$blob:`` sentinel string, else None."""
    if isinstance(value, str) and value.startswith(BLOB_SENTINEL_PREFIX):
        return value[len(BLOB_SENTINEL_PREFIX):]
    return None


def _iter_blob_refs(node: Any) -> Iterator[str]:
    """Yield every ``$blob:`` sentinel key anywhere under ``node``."""
    if isinstance(node, dict):
        for value in node.values():
            yield from _iter_blob_refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_blob_refs(value)
    else:
        key = _blob_key(node)
        if key is not None:
            yield key


def _replace_missing_blobs(node: Any, blobs: dict, warnings: list[str]) -> None:
    """F4: swap sentinels whose blob is absent for the RFC placeholder."""
    if isinstance(node, dict):
        items = list(node.items())
    elif isinstance(node, list):
        items = list(enumerate(node))
    else:
        return
    for k, value in items:
        key = _blob_key(value)
        if key is not None and key not in blobs:
            node[k] = f"(blob missing: {key})"
            warnings.append(
                f"F4: file corruption: {key} is missing from "
                f"metadata.rts.blobs; rendered {node[k]}"
            )
        else:
            _replace_missing_blobs(value, blobs, warnings)


def _check_otlp_attributes(attrs: Any, where: str) -> None:
    """F11: attributes MUST be ``[{key, value: {<AnyValue>}}]``."""
    if attrs is None:
        return
    if not isinstance(attrs, list) or not all(
        isinstance(a, dict) and isinstance(a.get("key"), str)
        and isinstance(a.get("value"), dict)
        for a in attrs
    ):
        raise LlmnbIntegrityError(
            "F11", f"non-conformant OTLP encoding in {where}.attributes",
        )


def check_rts_integrity(rts: dict) -> list[str]:
    """Apply the RFC-005 load-time integrity checks to ``metadata.rts``.

    Rejections raise :class:`LlmnbIntegrityError`:

    * F5 -- a ``blobs[<key>]`` body is null or its SHA-256 is not ``<key>``.
    * F9 -- the layout tree carries a duplicate ``id``.
    * F10 -- an agent-graph edge references a nonexistent node.
    * F11 -- a run (or run event) uses flat-object attribute encoding.

    Tolerated conditions mutate ``rts`` in place and come back as
    warning strings:

    * F4 -- a ``$blob:`` sentinel whose key is absent from ``blobs`` is
      replaced by ``"(blob missing: sha256:<hex>)"``.
    * F12 -- a numeric ``startTimeUnixNano`` / ``endTimeUnixNano`` is
      converted to its decimal string.
    """
    warnings: list[str] = []
    blobs = rts.get("blobs")
    if not isinstance(blobs, dict):
        blobs = {}
    for key, entry in blobs.items():
        # RFC-005 ArtifactRef rows carry ``body``; the metadata writer's
        # extract_blobs rows carry ``data``.
        body = entry.get("body", entry.get("data")) if isinstance(entry, dict) else None
        if not isinstance(body, str):
            raise LlmnbIntegrityError("F5", f"blob {key} has no body")
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if key != f"sha256:{digest}":
            raise LlmnbIntegrityError(
                "F5", f"blob integrity violation: body of {key} hashes to sha256:{digest}",
            )

    layout = rts.get("layout")
    tree = layout.get("tree") if isinstance(layout, dict) else None
    seen: set[Any] = set()
    stack: list[Any] = [tree]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        nid = node.get("id")
        if nid in seen:
            raise LlmnbIntegrityError(
                "F9", f"tree integrity violation: duplicate layout id {nid!r}",
            )
        seen.add(nid)
        children = node.get("children")
        if isinstance(children, list):
            stack.extend(children)

    agents = rts.get("agents")
    if isinstance(agents, dict):
        node_ids = {
            n.get("id") for n in agents.get("nodes") or [] if isinstance(n, dict)
        }
        for edge in agents.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            source, target = edge.get("source"), edge.get("target")
            if source not in node_ids or target not in node_ids:
                raise LlmnbIntegrityError(
                    "F10",
                    f"graph integrity violation: edge {source}→{target} "
                    f"kind={edge.get('kind')}",
                )

    event_log = rts.get("event_log")
    runs = event_log.get("runs") if isinstance(event_log, dict) else None
    for index, run in enumerate(runs if isinstance(runs, list) else []):
        if not isinstance(run, dict):
            continue
        where = f"event_log.runs[{index}]"
        _check_otlp_attributes(run.get("attributes"), where)
        for e_index, event in enumerate(run.get("events") or []):
            if isinstance(event, dict):
                _check_otlp_attributes(
                    event.get("attributes"), f"{where}.events[{e_index}]",
                )
        for name in ("startTimeUnixNano", "endTimeUnixNano"):
            value = run.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                run[name] = str(int(value))
                warnings.append(
                    f"F12: {where}.{name} was a JSON number; converted to a string"
                )
        _replace_missing_blobs(run, blobs, warnings)
    return warnings


def gc_blobs(llmnb: dict) -> list[str]:
    """Drop unreferenced entries from ``metadata.rts.blobs`` in place.

    RFC-005 §"Blob garbage collection": mark every ``$blob:`` sentinel
    reachable from the notebook (``event_log.runs[*]`` and, so a cell
    output's copy of a span keeps its blob alive, everything else outside
    the blob table), then sweep the rest. Returns the removed keys, sorted.
    """
    rts = _rts_namespace(llmnb)
    blobs = rts.get("blobs")
    if not isinstance(blobs, dict) or not blobs:
        return []
    reachable = set(_iter_blob_refs({k: v for k, v in rts.items() if k != "blobs"}))
    reachable.update(_iter_blob_refs(llmnb.get("cells")))
    removed = sorted(k for k in blobs if k not in reachable)
    for key in removed:
        del blobs[key]
    return removed

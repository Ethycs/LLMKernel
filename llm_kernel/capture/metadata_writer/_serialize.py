from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from llm_kernel.capture import overlay_applier as _overlay_applier
from llm_kernel.capture.overlay_applier import OverlayRejected as _OverlayRejected
from llm_kernel.wire import versions as _versions

if TYPE_CHECKING:  # pragma: no cover
    from llm_kernel.capture.custom_messages import CustomMessageDispatcher
    from llm_kernel.capture.run_tracker import RunTracker

logger: logging.Logger = logging.getLogger("llm_kernel.capture.metadata_writer")

#: RFC-005 schema version this writer emits.  Receivers MUST reject any
#: snapshot whose major differs (RFC-005 §"Top-level structure" /
#: RFC-006 W7).  ``SCHEMA_VERSION`` is forwarded from the single source
#: :mod:`llm_kernel.wire.versions` via ``__getattr__`` below (and again
#: by the ``metadata_writer`` package), so it stays importable from both
#: places while always reflecting the live source.  It is deliberately
#: absent from ``__all__`` so the star-imports do not snapshot a copy.


def __getattr__(name: str) -> Any:
    if name == "SCHEMA_VERSION":
        return _versions.SCHEMA_VERSION
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

#: Schema URI per RFC-005 §"Top-level structure"; SHOULD point at the
#: published JSON-Schema for this version.
SCHEMA_URI: str = "https://llmnb.dev/llmnb/v1/schema.json"

#: Default blob-extraction threshold in bytes.  Mirrored into
#: ``config.recoverable.kernel.blob_threshold_bytes`` so loaders use
#: the same threshold the writer used.
DEFAULT_BLOB_THRESHOLD_BYTES: int = 65536

#: Bounded-queue cap for queued event-log entries when no extension is
#: attached (RFC-005 §"When no extension is attached").
DEFAULT_EVENT_LOG_QUEUE_CAP: int = 10_000

#: Periodic snapshot cadence in seconds (RFC-005 §"Snapshot triggers"
#: case 3).  Kernel restarts lose at most this much activity; V1
#: accepts that.
DEFAULT_AUTOSAVE_INTERVAL_SEC: float = 30.0

#: V1 Kernel Gap Closure G13 -- intent-queue overflow threshold (RFC-005
#: §F13 disk-fallback rule, applied to the buffered ``submit_intent``
#: queue).  RFC-005 §F13 caps the *event-log* queue at 10 000; the
#: brief asks for a separate threshold on the *intent* queue.  No
#: published number; we INVENT 1 000 as a sane default and flag it in
#: the slice report.  Operators can tune via the constructor kwarg.
DEFAULT_INTENT_QUEUE_OVERFLOW_THRESHOLD: int = 1_000

#: BSP-005 §6.1 / [cell-kinds atom](docs/atoms/concepts/cell-kinds.md) --
#: the eight cell kinds the V1 writer accepts.  ``agent | markdown |
#: scratch | checkpoint`` are *active* (renderer dispatches per kind);
#: ``tool | artifact | control | native`` are *reserved* (V1 stores
#: them verbatim and renders inert).  Anything outside the eight values
#: is rejected with K42 ``unknown_cell_kind``.
CELL_KINDS_ACTIVE: Tuple[str, ...] = (
    "agent",
    "markdown",
    "scratch",
    "checkpoint",
)
CELL_KINDS_RESERVED: Tuple[str, ...] = (
    "tool",
    "artifact",
    "control",
    "native",
)
CELL_KINDS: FrozenSet[str] = frozenset(CELL_KINDS_ACTIVE + CELL_KINDS_RESERVED)
DEFAULT_CELL_KIND: str = "agent"

#: Sentinel used by :meth:`MetadataWriter._handle_set_cell_metadata` to
#: distinguish "caller did not pass this key" from "caller explicitly
#: passed ``None``".  Distinct sentinel object so isinstance(x, type)
#: checks can never collide with a legitimate ``None`` payload.
_UNSET: Any = object()

#: Forbidden-field name regexes per RFC-005 §"Forbidden fields
#: (security)".  Matched case-insensitively against the FIELD NAME (not
#: the value) at every depth of the ``config`` substructure.  The
#: ``*_public_key`` carve-out is handled by the dedicated check below.
_FORBIDDEN_FIELD_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"^.*_token$", re.IGNORECASE),
    re.compile(r"^.*_password$", re.IGNORECASE),
    re.compile(r"^.*_secret$", re.IGNORECASE),
    re.compile(r"^authorization$", re.IGNORECASE),
    re.compile(r"^bearer$", re.IGNORECASE),
    re.compile(r"^cookie$", re.IGNORECASE),
    re.compile(r"^api_key$", re.IGNORECASE),
)
_KEY_SUFFIX_RE: re.Pattern[str] = re.compile(r"^.*_key$", re.IGNORECASE)
_PUBLIC_KEY_SUFFIX_RE: re.Pattern[str] = re.compile(
    r"^.*_public_key$", re.IGNORECASE,
)


class SecretRejected(ValueError):
    """Raised by :func:`reject_secrets` when a forbidden field is detected.

    The offending VALUE is never logged or attached to the exception
    payload (RFC-005 §F2).  The exception's ``args[0]`` carries only
    the field path and the matched pattern name so failure is
    diagnosable without leaking the secret.
    """

    def __init__(self, field_path: str, pattern: str) -> None:
        super().__init__(
            f"forbidden field at {field_path!r} matches pattern {pattern!r}; "
            f"refusing to write metadata.rts (RFC-005 F2)"
        )
        self.field_path = field_path
        self.pattern = pattern


def _is_forbidden_key(key: str) -> Optional[str]:
    """Return the matched pattern name iff ``key`` is forbidden, else ``None``.

    RFC-005 carves out ``*_public_key`` from the ``*_key`` rule; this
    helper applies the carve-out before failing the broader
    ``*_key`` match.
    """
    if _PUBLIC_KEY_SUFFIX_RE.match(key):
        return None
    if _KEY_SUFFIX_RE.match(key):
        return "*_key (excluding *_public_key)"
    for pat in _FORBIDDEN_FIELD_PATTERNS:
        if pat.match(key):
            return pat.pattern
    return None


def reject_secrets(node: Any, path: str = "config") -> None:
    """Walk ``node`` recursively, raising on any forbidden field name.

    This is the security pass per RFC-005 §"Forbidden fields".  The
    walk descends into dicts and lists; on a forbidden key (matched
    case-insensitively against the field NAME, not the value) it
    raises :class:`SecretRejected` with the dot-path of the offender
    and never logs the value.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if not isinstance(key, str):
                continue
            matched = _is_forbidden_key(key)
            if matched is not None:
                raise SecretRejected(f"{path}.{key}", matched)
            reject_secrets(value, f"{path}.{key}")
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            reject_secrets(value, f"{path}[{idx}]")
    # Scalars are not recursed; their VALUES are not subject to the
    # name-match rule and MUST NOT be inspected (avoids accidental
    # secret-text logging if a future change misfires).


def _sha256_hex(data: str) -> str:
    """Return the lowercase-hex SHA-256 of the UTF-8 encoding of ``data``."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def extract_blobs(
    runs: List[Dict[str, Any]],
    blobs: Dict[str, Dict[str, Any]],
    threshold_bytes: int = DEFAULT_BLOB_THRESHOLD_BYTES,
) -> None:
    """Mutate ``runs`` and ``blobs`` in place per RFC-005 §"blob extraction".

    For every OTLP attribute value of shape ``{"stringValue": "..."}``
    whose UTF-8 byte length exceeds ``threshold_bytes``, the value is
    SHA-256 hashed; the blob is recorded under
    ``blobs["sha256:<hex>"]`` (idempotent: a re-encountered hash
    short-circuits); and the original attribute's ``stringValue`` is
    rewritten to ``$blob:sha256:<hex>`` so the OTLP span is loadable
    without the blob table (the renderer SHOULD resolve it on demand).

    Operates on the OTLP attribute layout (a list of ``{key, value}``
    pairs where ``value`` carries one of the AnyValue tagged-union
    keys).  Sentinels carry the same MIME and encoding metadata as
    the original blob row so a reader can faithfully reconstruct
    ``output.value`` etc.
    """
    for span in runs:
        for attr in span.get("attributes", []) or []:
            value = attr.get("value")
            if not isinstance(value, dict):
                continue
            if "stringValue" not in value:
                continue
            sv = value["stringValue"]
            if not isinstance(sv, str):
                continue
            if sv.startswith("$blob:sha256:"):
                continue  # already extracted on a prior pass
            if len(sv.encode("utf-8")) <= threshold_bytes:
                continue
            digest = _sha256_hex(sv)
            blob_key = f"sha256:{digest}"
            if blob_key not in blobs:
                blobs[blob_key] = {
                    "content_type": "text/plain",
                    "encoding": "utf-8",
                    "size_bytes": len(sv.encode("utf-8")),
                    "data": sv,
                }
            value["stringValue"] = f"$blob:{blob_key}"


# ----------------------------------------------------------------------
# Line-oriented JSON serializer.  RFC-005 §"Line-oriented serialization"
# requires each entry of ``event_log.runs[]``,
# ``layout.tree.children[]``, and ``agents.{nodes,edges}[]`` occupy ITS
# OWN line in the serialized JSON, so git's pack-delta compression and
# nbdime / git-diff produce useful per-record diffs.  Adding one run
# record SHOULD rewrite only one line of the serialized form.
# ----------------------------------------------------------------------


_LINE_ORIENTED_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("metadata", "rts", "event_log", "runs"),
    ("metadata", "rts", "agents", "nodes"),
    ("metadata", "rts", "agents", "edges"),
    # ``layout.tree.children`` is recursive; the encoder special-cases
    # every ``children`` array under ``layout.tree`` to keep the rule
    # simple even at deep nesting.  Marker is the literal "tree" path
    # element appearing anywhere under ``layout``.
)


def serialize_snapshot(snapshot: Dict[str, Any], indent: int = 2) -> str:
    """Serialize a full ``.llmnb``-shaped dict with line-oriented arrays.

    Equivalent to ``json.dumps(snapshot, indent=indent)`` except the
    elements of the line-oriented arrays each occupy a single line
    (no inner indentation, no inner newlines) so additive growth in
    ``event_log.runs`` produces a single-line git diff per appended
    span.  All other arrays use the standard pretty-print form for
    readability.

    Implementation: emit standard pretty JSON, then post-process to
    flatten the targeted arrays.  Post-processing locates each target
    by its bracket depth + key path + ``[`` opener, then walks to the
    matching closing ``]`` collapsing every interior newline-and-indent
    into the empty string.  Line breaks between elements are
    re-injected so each element still occupies its own line.
    """
    text = json.dumps(snapshot, indent=indent, ensure_ascii=False)
    text = _flatten_array_at_path(
        text, ("event_log", "runs"), indent=indent,
    )
    text = _flatten_array_at_path(
        text, ("agents", "nodes"), indent=indent,
    )
    text = _flatten_array_at_path(
        text, ("agents", "edges"), indent=indent,
    )
    text = _flatten_children_arrays(text, indent=indent)
    return text


def _flatten_array_at_path(
    text: str, path: Tuple[str, ...], indent: int,
) -> str:
    """Flatten one named JSON array so each element occupies one line.

    Walks the serialized form looking for the literal key name of the
    last path element followed by ``": [``.  Once located, scans
    forward tracking bracket depth (with quote-awareness) until the
    matching ``]`` is found; rewrites the slice to a one-element-per-
    line form.

    This is a best-effort textual flattener; it does NOT enforce the
    full path -- the path's last element is what's matched.  This is
    acceptable because the kernel's snapshot dict is constructed by
    this module and known not to reuse those names elsewhere.
    """
    last_key = path[-1]
    needle = f'"{last_key}": ['
    out_parts: List[str] = []
    cursor = 0
    while True:
        idx = text.find(needle, cursor)
        if idx == -1:
            out_parts.append(text[cursor:])
            break
        # Append the prefix unchanged (keeps surrounding indentation).
        out_parts.append(text[cursor:idx + len(needle)])
        # Find the matching closing bracket with quote-aware depth tracking.
        depth = 1
        i = idx + len(needle)
        in_string = False
        escape = False
        while i < len(text) and depth > 0:
            ch = text[i]
            if escape:
                escape = False
            elif ch == "\\" and in_string:
                escape = True
            elif ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        break
            i += 1
        body = text[idx + len(needle):i]
        # Locate the array's outer indentation by walking back from idx
        # to the start of the line.  The closing ``]`` is placed at
        # that indentation; each element is at outer + one indent step.
        line_start = text.rfind("\n", 0, idx) + 1
        outer_indent = " " * (idx - line_start - 0)
        # Compute outer_indent based on leading spaces on the line.
        leading_spaces = 0
        while line_start + leading_spaces < idx \
                and text[line_start + leading_spaces] == " ":
            leading_spaces += 1
        outer_indent = " " * leading_spaces
        elem_indent = outer_indent + " " * indent
        # If the array is empty (or whitespace only), preserve as-is.
        if body.strip() == "":
            out_parts.append(body + "]")
            cursor = i + 1
            continue
        # Split the body into top-level elements (depth-aware) then
        # collapse each to a single line.
        elements = _split_top_level_elements(body)
        flat_elements = [_collapse_whitespace(e.strip()) for e in elements]
        rebuilt = (
            "\n"
            + ",\n".join(elem_indent + e for e in flat_elements)
            + "\n"
            + outer_indent
            + "]"
        )
        out_parts.append(rebuilt)
        cursor = i + 1
    return "".join(out_parts)


def _flatten_children_arrays(text: str, indent: int) -> str:
    """Flatten every ``children`` array under ``layout``.

    The layout tree is recursive; ``children`` appears at every node.
    We flatten EVERY ``children`` array in the serialized form -- the
    only ``children`` keys in a snapshot live under ``metadata.rts.
    layout.tree.children`` (RFC-005 §`metadata.rts.layout`) so the
    naming is unambiguous in this writer's output.
    """
    return _flatten_array_at_path(text, ("layout", "tree", "children"), indent)


def _split_top_level_elements(body: str) -> List[str]:
    """Split a JSON-array body string into its top-level elements.

    Tracks brace + bracket + quote depth; commas at depth zero are
    element separators.  Returns a list of element strings (no
    surrounding commas, no trimmed whitespace).
    """
    elements: List[str] = []
    depth = 0
    in_string = False
    escape = False
    start = 0
    for i, ch in enumerate(body):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "," and depth == 0:
            elements.append(body[start:i])
            start = i + 1
    tail = body[start:]
    if tail.strip():
        elements.append(tail)
    return elements


def _collapse_whitespace(element: str) -> str:
    """Collapse a JSON element's interior whitespace to a single line.

    Quote-aware: characters inside strings are preserved verbatim.
    Outside strings, every run of whitespace (spaces, tabs, newlines)
    is replaced by a single space, except whitespace adjacent to
    structural punctuation (``{}[],:``) which is removed entirely
    so the output looks like compact JSON with one-space separators
    after ``,`` and ``:``.
    """
    out: List[str] = []
    in_string = False
    escape = False
    prev_was_space = False
    for ch in element:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            prev_was_space = False
            continue
        if in_string:
            out.append(ch)
            continue
        if ch in (" ", "\t", "\n", "\r"):
            if not prev_was_space and out and out[-1] not in "{[(":
                out.append(" ")
                prev_was_space = True
            continue
        # Drop trailing space before structural close.
        if ch in "}]),:" and out and out[-1] == " ":
            out.pop()
        out.append(ch)
        prev_was_space = False
    return "".join(out).strip()


# ----------------------------------------------------------------------


def _deepcopy_json(obj: Any) -> Any:
    """Deep-copy a JSON-shaped tree without importing ``copy.deepcopy``."""
    if isinstance(obj, dict):
        return {k: _deepcopy_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deepcopy_json(v) for v in obj]
    return obj


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 with millisecond precision."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


__all__ = [
    "logger",
    "SCHEMA_URI",
    "DEFAULT_BLOB_THRESHOLD_BYTES",
    "DEFAULT_EVENT_LOG_QUEUE_CAP",
    "DEFAULT_AUTOSAVE_INTERVAL_SEC",
    "DEFAULT_INTENT_QUEUE_OVERFLOW_THRESHOLD",
    "CELL_KINDS_ACTIVE",
    "CELL_KINDS_RESERVED",
    "CELL_KINDS",
    "DEFAULT_CELL_KIND",
    "_UNSET",
    "_FORBIDDEN_FIELD_PATTERNS",
    "_KEY_SUFFIX_RE",
    "_PUBLIC_KEY_SUFFIX_RE",
    "_LINE_ORIENTED_PATHS",
    "SecretRejected",
    "_is_forbidden_key",
    "reject_secrets",
    "_sha256_hex",
    "extract_blobs",
    "serialize_snapshot",
    "_flatten_array_at_path",
    "_flatten_children_arrays",
    "_split_top_level_elements",
    "_collapse_whitespace",
    "_deepcopy_json",
    "_utc_now_iso",
]

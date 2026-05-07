"""LLMKernel metadata writer (RFC-005 single logical writer of ``metadata.rts``).

The kernel is the single logical writer of ``metadata.rts`` per
RFC-005 §"Persistence strategy"; this module is that writer.  It
builds full snapshots of the ``metadata.rts`` namespace from
in-memory state and emits them to the extension over the RFC-006
Family F ``notebook.metadata`` wire (the dispatcher's
:meth:`emit` path).  When no extension is attached, the dispatcher
buffers; on bounded-queue overflow this writer drops a checkpoint
marker per RFC-005 §F13 (kernel-internal disk path; the
operator-facing direct-write tool is queued for V1.5).

What this module provides:

* :class:`MetadataWriter` -- the single-instance class wired into the
  kernel.  Accumulates layout / agents / config state and event-log
  entries, applies the security and blob-extraction passes, and
  emits ``notebook.metadata`` envelopes on the four RFC-005
  triggers: operator save, clean shutdown, periodic 30s timer, and
  end-of-run.
* :func:`extract_blobs` -- the recoverable blob-extraction pass.
  Any string attribute value whose serialized form exceeds
  ``config.kernel.blob_threshold_bytes`` is hashed (SHA-256) and
  stored in a content-addressed table; the originating attribute is
  rewritten to a ``$blob:sha256:<hex>`` sentinel.
* :func:`reject_secrets` -- the forbidden-field pass per RFC-005
  §"Forbidden fields (security)".  Raises :class:`SecretRejected`
  with a sanitized log signature; the offending value is never
  logged.
* :class:`SnapshotEncoder` -- the line-oriented JSON encoder.  One
  span per line in ``event_log.runs[]``; same for
  ``layout.tree.children[]`` and ``agents.{nodes,edges}[]``.  This is
  what makes ``git diff`` over ``.llmnb`` files produce useful
  output (RFC-005 §"Line-oriented serialization").

Thread-safe: callers may invoke :meth:`update_layout`,
:meth:`update_agents`, :meth:`record_run` from any thread.  The
periodic-timer trigger runs in a daemon thread and respects
:meth:`stop`.
"""

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

from . import overlay_applier as _overlay_applier
from .overlay_applier import OverlayRejected as _OverlayRejected

if TYPE_CHECKING:  # pragma: no cover
    from .custom_messages import CustomMessageDispatcher
    from .run_tracker import RunTracker

logger: logging.Logger = logging.getLogger("llm_kernel.metadata_writer")

#: RFC-005 schema version this writer emits.  Receivers MUST reject any
#: snapshot whose major differs (RFC-005 §"Top-level structure" /
#: RFC-006 W7).
SCHEMA_VERSION: str = "1.0.0"

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
# MetadataWriter
# ----------------------------------------------------------------------


class MetadataWriter:
    """Single-instance writer that builds and emits ``metadata.rts`` snapshots.

    Wired into :func:`llm_kernel._kernel_hooks.attach_kernel_subsystems`;
    receives closed-span notifications from the run-tracker (or pulls
    them on each emission) and runs a 30-second autosave timer.
    Emissions go through ``dispatcher.emit`` as a Family F
    ``notebook.metadata`` envelope per RFC-006 §8.

    Attributes are mutated through the dedicated ``update_*`` methods
    so the lock scope is well-defined; callers MUST NOT poke at the
    private snapshot dicts directly.
    """

    def __init__(
        self,
        dispatcher: "Optional[CustomMessageDispatcher]" = None,
        run_tracker: "Optional[RunTracker]" = None,
        session_id: Optional[str] = None,
        blob_threshold_bytes: int = DEFAULT_BLOB_THRESHOLD_BYTES,
        autosave_interval_sec: float = DEFAULT_AUTOSAVE_INTERVAL_SEC,
        event_log_queue_cap: int = DEFAULT_EVENT_LOG_QUEUE_CAP,
        workspace_root: Optional[Path] = None,
        intent_queue_overflow_threshold: int = (
            DEFAULT_INTENT_QUEUE_OVERFLOW_THRESHOLD
        ),
    ) -> None:
        """Construct an unstarted writer.

        ``dispatcher`` may be ``None`` for tests / smokes that capture
        emissions through :meth:`take_last_envelope`.  ``run_tracker``
        is the source of truth for ``event_log`` if set; otherwise the
        writer relies entirely on :meth:`record_run`.  ``session_id``
        defaults to a fresh UUIDv4; reopening an existing file would
        pass the persisted UUID through.

        ``workspace_root`` is the directory the queue-overflow disk
        fallback (RFC-005 §F13) writes into.  Defaults to the current
        working directory; tests usually pass ``tmp_path``.
        """
        self._dispatcher = dispatcher
        self._run_tracker = run_tracker
        self._lock: threading.RLock = threading.RLock()
        self._session_id: str = session_id or str(uuid.uuid4())
        self._created_at: str = _utc_now_iso()
        self._snapshot_version: int = 0
        self._blob_threshold_bytes: int = blob_threshold_bytes
        self._autosave_interval_sec: float = autosave_interval_sec
        self._event_log_queue_cap: int = event_log_queue_cap
        self._workspace_root: Path = (
            Path(workspace_root) if workspace_root is not None else Path.cwd()
        )

        self._layout: Dict[str, Any] = {
            "version": 1,
            "tree": {
                "id": "root", "type": "workspace",
                "render_hints": {}, "children": [],
            },
        }
        self._agent_graph: Dict[str, Any] = {
            "version": 1, "nodes": [], "edges": [],
        }
        self._config: Dict[str, Any] = {
            "version": 1,
            "recoverable": {
                "kernel": {"blob_threshold_bytes": blob_threshold_bytes},
                "agents": [],
                "mcp_servers": [],
            },
            "volatile": {
                "kernel": {},
                "agents": [],
                "mcp_servers": [],
            },
        }
        # event_log.runs is the in-memory list of OTLP spans the writer
        # has seen.  When run_tracker is set we drain its open+closed
        # spans on every snapshot; record_run() additionally appends
        # spans pushed in by external producers (e.g. the agent
        # supervisor's agent_emit emissions).
        self._extra_runs: List[Dict[str, Any]] = []
        self._blobs: Dict[str, Dict[str, Any]] = {}
        self._drift_log: List[Dict[str, Any]] = []
        # BSP-005 §6.1 / S0.5 -- per-cell metadata keyed by cell_id.
        # Stored at ``metadata.rts.cells`` in the snapshot.  Each entry:
        #   {"kind": "agent" | ..., "bound_agent_id": str|None,
        #    "section_id": str|None, "capabilities": [], ...flags}
        # The writer is the source of truth; the extension reads via
        # the Family F snapshot.
        self._cells: Dict[str, Dict[str, Any]] = {}
        # BSP-008 §3 / S3.5 -- per-zone substructure rooted at
        # ``metadata.rts.zone``. Today carries only the
        # ``context_manifests`` map keyed by manifest_id; future slices
        # add ``run_frames`` (S6), ``sections`` (S5.5), and the rest of
        # the zone substructure listed in
        # [concepts/context-manifest](docs/atoms/concepts/context-manifest.md).
        # The manifests are append-only -- Inspect mode needs historical
        # access -- so the writer never deletes from this map.
        self._zone: Dict[str, Any] = {}
        # V1 Kernel Gap Closure G13 -- intent-queue overflow state.
        # ``_pending_intents`` is the buffered (deferred) intent queue
        # filled by :meth:`enqueue_intent` and drained by
        # :meth:`flush_pending_intents`.  When buffered count exceeds
        # ``_intent_queue_overflow_threshold`` the buffer spills to a
        # JSON-line file under ``<workspace_root>/.llmnb-intent-queue/``
        # and a marker is recorded at
        # ``metadata.rts.queues['intents'].overflow``.  ``_queues`` is
        # the persistent record of overflow markers (cleared once the
        # corresponding spill is fully drained).
        self._intent_queue_overflow_threshold: int = max(
            1, int(intent_queue_overflow_threshold),
        )
        self._pending_intents: List[Dict[str, Any]] = []
        self._queues: Dict[str, Dict[str, Any]] = {}
        self._intent_overflow_count: int = 0

        self._timer_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._dirty: bool = False
        self._last_emitted_envelope: Optional[Dict[str, Any]] = None
        self._overflow_checkpoint_count: int = 0

        # BSP-003 §6 intent registry state.  ``_intent_applied`` is the
        # idempotency set keyed on ``intent_id`` -> ``snapshot_version``
        # at the time the intent was applied (BSP-003 §6 step 2).
        # ``_intent_log`` accumulates the per-intent ``intent_applied``
        # event-log entries emitted by :meth:`submit_intent` (BSP-003
        # §6 step 7).  ``_intent_queue_lock`` is the FIFO serialization
        # gate (BSP-003 §6 final paragraph: single thread of writes
        # within one zone).
        self._intent_applied: Dict[str, int] = {}
        self._intent_log: List[Dict[str, Any]] = []
        self._intent_queue_lock: threading.RLock = threading.RLock()

    # -- PLAN-S6.0 in-tree event-log substrate -----------------------
    #
    # ``metadata.rts.zone.event_log[]`` was introduced in PLAN-S4.1 as
    # an ``agent_ref_move``-only array.  PLAN-S6.0 broadens it to carry
    # the full RFC-006 envelope stream (the **internal v1 envelope**
    # shape per ``run_envelope.make_envelope`` -- preserves
    # ``rfc_version`` for replay branching).  Existing readers MUST
    # filter by envelope ``kind``: pre-S6.0 entries carry
    # ``kind == "agent_ref_move"`` (the legacy shape), post-S6.0
    # envelope captures carry ``kind`` equal to the ``message_type``
    # (e.g. ``operator.action``, ``layout.update``,
    # ``notebook.metadata``).  Two filter helpers are exposed below.
    #
    # OTLP run.* lifecycle envelopes (Family A run.start | run.event |
    # run.complete) are EXCLUDED from this log per PLAN-S6.0 §3.B
    # non-goal -- they ride a separate observability sink and the
    # closed span lands in ``metadata.rts.event_log.runs[]``.
    #
    # Family F ``mode: "patch"`` envelopes are also EXCLUDED -- they
    # are transient deltas; replay reconstructs from
    # ``mode: "snapshot"`` checkpoints + intermediate Family A/D
    # events.

    _EVENT_LOG_EXCLUDED_TYPES: FrozenSet[str] = frozenset({
        "run.start", "run.event", "run.complete",
    })

    def capture_envelope(self, envelope: Dict[str, Any]) -> None:
        """Append ``envelope`` to ``metadata.rts.zone.event_log[]``.

        Honors ``metadata.rts.config.recoverable.kernel.event_log_enabled``
        (default ``True``) and
        ``metadata.rts.config.recoverable.kernel.event_log_max_entries``
        (default ``None`` = unbounded).  When the cap is set and the
        log exceeds it, the oldest entries are sliced off and appended
        to ``zone.event_log_archive[]``.

        Skips:
          * Family A run.start/run.event/run.complete (PLAN-S6.0 §3.B
            non-goal -- separate observability sink).
          * Family F ``notebook.metadata`` envelopes whose
            ``payload.mode == "patch"`` (transient deltas; replay
            uses snapshots + Family A/D between them).

        Validation: ``envelope`` MUST be a dict with the internal v1
        shape (``message_type``, ``payload``, ``rfc_version``...).
        Malformed inputs are silently dropped (defensive on the hot
        path; the wire layer already validated).
        """
        if not isinstance(envelope, dict):
            return
        message_type = envelope.get("message_type")
        if not isinstance(message_type, str):
            return
        if message_type in self._EVENT_LOG_EXCLUDED_TYPES:
            return
        # Family F mode=patch exclusion.
        if message_type == "notebook.metadata":
            payload = envelope.get("payload") or {}
            if isinstance(payload, dict) and payload.get("mode") == "patch":
                return
        with self._lock:
            kernel_cfg = (
                self._config.get("recoverable", {}).get("kernel", {})
                if isinstance(self._config, dict) else {}
            )
            enabled = kernel_cfg.get("event_log_enabled", True)
            if not enabled:
                return
            max_entries = kernel_cfg.get("event_log_max_entries")
            log = self._zone.setdefault("event_log", [])
            log.append(_deepcopy_json(envelope))
            if isinstance(max_entries, int) and max_entries > 0:
                if len(log) > max_entries:
                    overflow = len(log) - max_entries
                    archived = log[:overflow]
                    del log[:overflow]
                    archive = self._zone.setdefault("event_log_archive", [])
                    archive.extend(archived)
            self._dirty = True

    @staticmethod
    def is_legacy_event_log_entry(entry: Dict[str, Any]) -> bool:
        """Return True iff ``entry`` is a pre-S6.0 ``agent_ref_move`` record.

        Filter helper for downstream readers of
        ``zone.event_log[]``.  Post-S6.0 the array is mixed: legacy
        entries (the ``agent_ref_move`` shape from PLAN-S4.1) and
        captured envelopes (the v1 envelope shape from PLAN-S6.0).
        Use this to filter to the legacy-only subset.
        """
        if not isinstance(entry, dict):
            return False
        return entry.get("kind") == "agent_ref_move" and "message_type" not in entry

    @staticmethod
    def is_envelope_event_log_entry(entry: Dict[str, Any]) -> bool:
        """Return True iff ``entry`` is a captured-envelope (post-S6.0) record.

        Filter helper for downstream readers.  Recognized by the
        presence of ``message_type`` (the v1 envelope shape).
        """
        if not isinstance(entry, dict):
            return False
        return isinstance(entry.get("message_type"), str)

    # -- Public mutators ---------------------------------------------

    def update_layout(self, tree: Dict[str, Any]) -> None:
        """Replace the layout tree (last-writer-wins).  Marks the writer dirty."""
        with self._lock:
            self._layout["tree"] = tree
            self._dirty = True

    def update_agents(
        self, nodes: Iterable[Dict[str, Any]],
        edges: Iterable[Dict[str, Any]],
    ) -> None:
        """Replace the agent graph (last-writer-wins)."""
        with self._lock:
            self._agent_graph["nodes"] = list(nodes)
            self._agent_graph["edges"] = list(edges)
            self._dirty = True

    def update_config(
        self, recoverable: Dict[str, Any], volatile: Dict[str, Any],
    ) -> None:
        """Replace the config substructures.

        Both ``recoverable`` and ``volatile`` are validated for
        forbidden fields (RFC-005 §F2) BEFORE being committed.  A
        :class:`SecretRejected` exception leaves the writer's existing
        config untouched.
        """
        reject_secrets(recoverable, path="config.recoverable")
        reject_secrets(volatile, path="config.volatile")
        with self._lock:
            # Preserve the kernel-only settings the writer manages.
            recoverable_kernel = dict(recoverable.get("kernel", {}))
            recoverable_kernel.setdefault(
                "blob_threshold_bytes", self._blob_threshold_bytes,
            )
            self._config["recoverable"] = {
                **recoverable, "kernel": recoverable_kernel,
            }
            self._config["volatile"] = dict(volatile)
            self._dirty = True

    def record_run(self, span: Dict[str, Any]) -> None:
        """Append one OTLP/JSON span to the event-log queue.

        Bounded per RFC-005 §"When no extension is attached".  On
        overflow (>``event_log_queue_cap`` entries) the writer drops a
        checkpoint marker file and direct-writes the snapshot per
        RFC-005 §F13, then drains the in-memory queue so subsequent
        ``record_run`` calls have headroom.

        Threading note: the disk write happens inside the lock (so the
        snapshot is consistent), but the post-overflow ``logger.warning``
        is deliberately moved OUTSIDE the lock per the Engineering
        Guide §11.7 RLock-on-logging anti-pattern.
        """
        overflow_log_args: Optional[Tuple[int, int]] = None
        marker: Optional[Dict[str, Any]] = None
        snapshot_to_write: Optional[Dict[str, Any]] = None
        with self._lock:
            self._extra_runs.append(span)
            self._dirty = True
            if len(self._extra_runs) > self._event_log_queue_cap:
                queue_size = len(self._extra_runs)
                # Build the snapshot UNDER the RLock (re-entrant; safe
                # per Engineering Guide §11.7).  The
                # _build_snapshot call increments _snapshot_version, so
                # the marker carries the post-build version (the
                # version of the on-disk overflow snapshot).
                snapshot_to_write = self._build_snapshot()
                marker = {
                    "kernel_session_id": self._session_id,
                    "overflow_at": _utc_now_iso(),
                    "snapshot_version": snapshot_to_write["snapshot_version"],
                    "queue_size_at_overflow": queue_size,
                }
                # Drain the in-memory queue so subsequent record_run
                # calls have headroom.  We do this BEFORE releasing the
                # lock to keep the queue/version state coherent.
                self._extra_runs = []
                self._overflow_checkpoint_count += 1
                overflow_log_args = (
                    self._overflow_checkpoint_count, queue_size,
                )
        # Disk write + log MUST happen AFTER releasing the lock.  The
        # disk write because §F13 specifies the helper runs outside
        # the lock; the log because Engineering Guide §11.7 forbids
        # logger calls inside a lock that a logging handler may
        # re-enter.
        if marker is not None and snapshot_to_write is not None:
            try:
                self._write_overflow_fallback(marker, snapshot_to_write)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "metadata writer: overflow disk fallback raised; "
                    "queue already drained, continuing"
                )
        if overflow_log_args is not None:
            checkpoint_count, queue_size = overflow_log_args
            logger.warning(
                "event_log queue overflow: queue=%d cap=%d checkpoints=%d; "
                "checkpoint marker + direct-write per RFC-005 §F13",
                queue_size, self._event_log_queue_cap, checkpoint_count,
                extra={
                    "event.name": "metadata.queue_overflow",
                    "llmnb.checkpoint_count": checkpoint_count,
                    "llmnb.queue_size_at_overflow": queue_size,
                },
            )

    def _write_overflow_fallback(
        self, marker: Dict[str, Any], snapshot: Dict[str, Any],
    ) -> None:
        """Direct-write the queue-overflow marker + snapshot to disk.

        Per RFC-005 §F13 the fallback path lives sibling to the
        ``.llmnb`` file at ``<workspace_root>/.llmnb-overflow-marker.json``
        and ``<workspace_root>/.llmnb-overflow-snapshot.json``.  The
        operator/extension may merge or discard on next file-open.

        Implementation note: writes are done with ``json.dumps`` +
        atomic-style "write to .tmp, then os.replace" so a crash mid-
        write does not leave a half-written marker that the next open
        would interpret as a successful overflow.
        """
        self._workspace_root.mkdir(parents=True, exist_ok=True)
        marker_path = self._workspace_root / ".llmnb-overflow-marker.json"
        snapshot_path = self._workspace_root / ".llmnb-overflow-snapshot.json"
        marker_tmp = marker_path.with_name(marker_path.name + ".tmp")
        snapshot_tmp = snapshot_path.with_name(snapshot_path.name + ".tmp")
        # Snapshot first so the marker, when present, points at a
        # fully written snapshot.
        snapshot_tmp.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(snapshot_tmp, snapshot_path)
        marker_tmp.write_text(
            json.dumps(marker, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(marker_tmp, marker_path)

    def append_drift_event(
        self, *, field_path: str, previous_value: Any, current_value: Any,
        severity: str, detected_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append one drift event.  Returns the appended dict for testability.

        Drift events are NEVER removed (RFC-005 §"drift_log").  This
        method mutates the in-memory drift log; the next snapshot
        emits the new entry.
        """
        event = {
            "detected_at": detected_at or _utc_now_iso(),
            "field_path": field_path,
            "previous_value": previous_value,
            "current_value": current_value,
            "severity": severity,
            "operator_acknowledged": False,
        }
        with self._lock:
            self._drift_log.append(event)
            self._dirty = True
        return event

    def acknowledge_drift(
        self, field_path: str, detected_at: str,
    ) -> bool:
        """Mark one drift_log entry as operator-acknowledged.

        Locates the entry whose ``field_path`` AND ``detected_at`` BOTH
        equal the arguments and sets ``operator_acknowledged = True``.
        Returns ``True`` on a hit, ``False`` when no entry matches
        (idempotent: this method NEVER raises on miss; the caller -- e.g.
        the MCP ``drift_acknowledged`` operator action -- is allowed to
        send the request twice without causing an error).

        Threadsafe: takes the writer's RLock for the duration of the
        scan + mutate so a concurrent ``append_drift_event`` cannot
        interleave a partial state.
        """
        with self._lock:
            for entry in self._drift_log:
                if (
                    entry.get("field_path") == field_path
                    and entry.get("detected_at") == detected_at
                ):
                    entry["operator_acknowledged"] = True
                    self._dirty = True
                    return True
        return False

    # -- BSP-003 §10 intent dispatcher -------------------------------

    #: BSP-003 §5 enumeration of legal ``intent_kind`` values.  Adding
    #: a kind requires a BSP/RFC amendment; an unknown kind triggers
    #: K40.
    _BSP003_INTENT_KINDS: FrozenSet[str] = frozenset({  # type: ignore[name-defined]
        "append_turn",
        "create_agent",
        "move_agent_head",
        "fork_agent",
        "update_agent_session",
        "add_overlay",
        "move_overlay_ref",
        "set_cell_metadata",
        "update_ordering",
        "add_blob",
        "record_event",
        # K-MW slice extensions: these wrap the existing apply functions
        # so the dispatcher is the single mutation entrypoint per
        # BSP-003 §2 ("all writes serialize through one queue").  Other
        # agents call the public methods directly today; the registry
        # entries here let external clients submit the same operations
        # via the intent envelope.
        "apply_layout_edit",
        "apply_agent_graph_command",
        "acknowledge_drift",
        # 2026-04-28 amendment kinds: BSP-007 K-OVERLAY + BSP-008
        # ContextPacker handlers ship with their respective slices.
        # The overlay-commit kinds dispatch to overlay_applier.py
        # (BSP-007 K-OVERLAY); the context-manifest / run-frame kinds
        # dispatch to context_packer.py (BSP-008 K-AS-A / K-CTXR).
        "apply_overlay_commit",        # BSP-007 §4.1 / §8 — overlay_applier.py
        "revert_overlay_to_commit",    # BSP-007 §4.2 — overlay_applier.py
        "create_overlay_ref",          # BSP-007 §4.4 — overlay_applier.py
        "record_context_manifest",     # BSP-008 §3 — context_packer.py
        "record_run_frame",            # BSP-008 §7 — context_packer.py
    })

    def submit_intent(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """Submit one BSP-003 intent envelope; apply via the registry.

        Envelope shape (BSP-003 §3):

        ::

            {
              "type": "operator.action",
              "payload": {
                "action_type": "zone_mutate",
                "intent_kind": "<kind>",
                "parameters": { ... },
                "intent_id": "<ulid|uuid>",
                "expected_snapshot_version": <int>   # optional, CAS
              }
            }

        Returns a result dict carrying the disposition:

        ``{"applied": bool, "intent_id": str, "snapshot_version": int,
        "already_applied": bool, "error_code": str|None,
        "error_reason": str|None, "response": dict|None}``

        Failure modes (BSP-003 §8):

        * ``K40`` -- unknown ``intent_kind``.  No state change.
        * ``K41`` -- ``expected_snapshot_version`` mismatch (CAS).  No
          state change.
        * ``K42`` -- intent validator rejected (e.g., missing required
          parameters, target node not found).  No state change.
        * ``K43`` -- atomic file write failed.  In-memory state still
          consistent; surface as degraded.

        The method is the public mutation entrypoint per BSP-003 §10.
        FIFO serialization is provided by ``_intent_queue_lock``; the
        existing autosave-timer thread handles drainage of the
        downstream debounced file write.

        Atomic-write step (BSP-003 §6 step 8) is exposed as a no-op in
        V1 -- the actual file write is the dispatcher's
        :meth:`snapshot` path which uses ``json.dumps`` over a
        complete materialized snapshot.  A direct ``tmp + rename``
        write to ``metadata.rts`` is queued for the ``.llmnb``
        save-pipeline integration (X-EXT slice, file format owner) and
        not in the K-MW V1 scope.  This method emits a
        ``notebook.metadata`` snapshot envelope (BSP-003 §6 step 9)
        which the dispatcher carries on the wire.
        """
        # FIFO gate: serialize all intents through a single re-entrant
        # lock so concurrent submissions interleave deterministically.
        with self._intent_queue_lock:
            return self._dispatch_intent(envelope)

    def _dispatch_intent(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """Validate + route one intent envelope.  Caller holds the FIFO lock."""
        # Envelope validation (BSP-003 §3 + RFC-006 §"operator.action").
        if not isinstance(envelope, dict):
            return self._intent_failure(
                intent_id="", code="K42",
                reason="envelope must be a dict",
            )
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            return self._intent_failure(
                intent_id="", code="K42",
                reason="envelope.payload must be a dict",
            )
        intent_id = payload.get("intent_id")
        if not isinstance(intent_id, str) or not intent_id:
            return self._intent_failure(
                intent_id="", code="K42",
                reason="payload.intent_id must be a non-empty string",
            )
        intent_kind = payload.get("intent_kind")
        parameters = payload.get("parameters") or {}
        if not isinstance(parameters, dict):
            return self._intent_failure(
                intent_id=intent_id, code="K42",
                reason="payload.parameters must be a dict",
            )
        expected_version = payload.get("expected_snapshot_version")
        action_type = payload.get("action_type")
        if action_type is not None and action_type != "zone_mutate":
            # We accept only the canonical action_type per BSP-003 §3.
            # Receivers SHOULD be permissive (RFC-003 §back-compat); we
            # log a marker and continue rather than fail.
            logger.debug(
                "submit_intent: unexpected action_type %r (continuing)",
                action_type,
            )

        # BSP-003 §6 step 2: idempotency check.  A re-submission of an
        # already-applied intent_id is a no-op + emits an
        # already-applied response.
        with self._lock:
            if intent_id in self._intent_applied:
                applied_version = self._intent_applied[intent_id]
                return {
                    "applied": False,
                    "already_applied": True,
                    "intent_id": intent_id,
                    "snapshot_version": applied_version,
                    "error_code": None,
                    "error_reason": None,
                    "response": None,
                }

        # BSP-003 §6 step 3: CAS check.
        if expected_version is not None:
            try:
                expected_int = int(expected_version)
            except (TypeError, ValueError):
                return self._intent_failure(
                    intent_id=intent_id, code="K42",
                    reason="expected_snapshot_version must be an integer",
                )
            with self._lock:
                actual = self._snapshot_version
            if actual != expected_int:
                # K41 emit a numbered-marker log entry per BSP-003 §8.
                logger.warning(
                    "intent_cas_rejected expected=%d actual=%d intent_id=%s",
                    expected_int, actual, intent_id,
                    extra={
                        "event.name": "intent_cas_rejected",
                        "llmnb.intent_id": intent_id,
                        "llmnb.expected_snapshot_version": expected_int,
                        "llmnb.actual_snapshot_version": actual,
                    },
                )
                return {
                    "applied": False,
                    "already_applied": False,
                    "intent_id": intent_id,
                    "snapshot_version": actual,
                    "error_code": "K41",
                    "error_reason": (
                        f"CAS rejected: expected {expected_int} actual {actual}"
                    ),
                    "response": None,
                }

        # BSP-003 §6 step 1: dispatch through the registry.
        if intent_kind not in self._BSP003_INTENT_KINDS:
            logger.warning(
                "intent_unknown_kind intent_kind=%r intent_id=%s",
                intent_kind, intent_id,
                extra={
                    "event.name": "intent_unknown_kind",
                    "llmnb.intent_id": intent_id,
                    "llmnb.intent_kind": str(intent_kind),
                },
            )
            with self._lock:
                version_at_reject = self._snapshot_version
            return {
                "applied": False,
                "already_applied": False,
                "intent_id": intent_id,
                "snapshot_version": version_at_reject,
                "error_code": "K40",
                "error_reason": f"unknown intent_kind: {intent_kind!r}",
                "response": None,
            }

        # Apply via the per-kind handler.  Handlers return either:
        #   - True/False/int (truthy = applied; int = post-bumped version)
        #   - dict response (for query-style commands)
        handler = self._intent_handler_for(intent_kind)
        with self._lock:
            pre_version = self._snapshot_version
        try:
            outcome = handler(parameters)
        except _OverlayRejected as overlay_exc:  # BSP-007 §7 K90+ codes.
            # The overlay applier raises a structured rejection that
            # carries the K-code (K90/K91/K92/K93/K94/K95) plus the
            # per-K marker and detail dict the operator UI surfaces.
            logger.warning(
                "%s intent_kind=%s intent_id=%s reason=%s",
                overlay_exc.marker, intent_kind, intent_id, overlay_exc.reason,
                extra={
                    "event.name": overlay_exc.marker,
                    "llmnb.intent_id": intent_id,
                    "llmnb.intent_kind": intent_kind,
                    "llmnb.overlay_rejection": dict(overlay_exc.details),
                },
            )
            with self._lock:
                version_at_reject = self._snapshot_version
            return {
                "applied": False,
                "already_applied": False,
                "intent_id": intent_id,
                "snapshot_version": version_at_reject,
                "error_code": overlay_exc.code,
                "error_reason": overlay_exc.reason,
                "response": {
                    "marker": overlay_exc.marker,
                    "details": dict(overlay_exc.details),
                },
            }
        except Exception as exc:  # K42 -- validation / apply error.
            logger.warning(
                "intent_validation_failed intent_kind=%s intent_id=%s reason=%s",
                intent_kind, intent_id, str(exc),
                extra={
                    "event.name": "intent_validation_failed",
                    "llmnb.intent_id": intent_id,
                    "llmnb.intent_kind": intent_kind,
                },
            )
            with self._lock:
                version_at_reject = self._snapshot_version
            return {
                "applied": False,
                "already_applied": False,
                "intent_id": intent_id,
                "snapshot_version": version_at_reject,
                "error_code": "K42",
                "error_reason": str(exc),
                "response": None,
            }

        # Decode the handler outcome.  For dict outcomes (query-style),
        # the outcome is the response payload and we do NOT bump the
        # version unless the dict carries ``"applied": True``.
        ok: bool
        response_payload: Optional[Dict[str, Any]] = None
        if isinstance(outcome, dict):
            response_payload = outcome
            # Mutation-style outcomes carry "ok"; query-style do not.
            ok = bool(outcome.get("ok", True))
        elif isinstance(outcome, int):
            # apply_layout_edit returns the post-bump version; if the
            # version DIDN'T change the apply was a no-op (K42-like).
            ok = outcome > pre_version
            # Note: for layout the bump already happened inside the
            # apply call; record the new version verbatim.
        else:
            ok = bool(outcome)

        if not ok:
            logger.warning(
                "intent_validation_failed intent_kind=%s intent_id=%s "
                "reason=apply_returned_false",
                intent_kind, intent_id,
                extra={
                    "event.name": "intent_validation_failed",
                    "llmnb.intent_id": intent_id,
                    "llmnb.intent_kind": intent_kind,
                },
            )
            with self._lock:
                version_at_reject = self._snapshot_version
            return {
                "applied": False,
                "already_applied": False,
                "intent_id": intent_id,
                "snapshot_version": version_at_reject,
                "error_code": "K42",
                "error_reason": "apply rejected the intent (validator returned false)",
                "response": response_payload,
            }

        # BSP-003 §6 steps 6-7: bump version (if the handler hasn't
        # already), record the intent_applied event, mark dirty.
        with self._lock:
            # Layout / agent_graph mutators bump _snapshot_version
            # themselves; record_event / acknowledge_drift do not.
            # Bump iff the version is unchanged from before the apply.
            # We approximate by bumping only when the handler did NOT
            # already increment (intent_kind in the explicit set).
            if intent_kind in {
                "record_event", "acknowledge_drift",
                # PLAN-S4.1 turn-graph handlers: in-memory mutators
                # that flip ``_dirty`` but do not bump the snapshot
                # version themselves.  We bump here so each successful
                # apply produces a fresh snapshot id.
                "append_turn", "fork_agent", "move_agent_head",
                # PLAN-S4.2: update_agent_session bumps version so
                # runtime_status / pid changes produce a fresh snapshot.
                "update_agent_session",
                # BSP-007 K-OVERLAY: overlay primitives flip _dirty
                # but rely on the dispatcher to bump version, so each
                # apply / revert / create-ref produces a fresh snapshot.
                "apply_overlay_commit", "revert_overlay_to_commit",
                "create_overlay_ref",
            }:
                self._snapshot_version += 1
            new_version = self._snapshot_version
            self._intent_applied[intent_id] = new_version
            entry = {
                "type": "intent_applied",
                "intent_id": intent_id,
                "intent_kind": intent_kind,
                "snapshot_version": new_version,
                "recorded_at": _utc_now_iso(),
            }
            self._intent_log.append(entry)
            self._dirty = True

        # BSP-003 §6 step 9: emit the post-apply Family F snapshot
        # envelope to subscribers.  We wrap exceptions because the
        # envelope emit must not break the apply contract -- the
        # in-memory state is already updated by this point.
        try:
            self.snapshot(trigger="intent_applied")
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "submit_intent: post-apply snapshot emit raised; "
                "in-memory state already updated"
            )

        return {
            "applied": True,
            "already_applied": False,
            "intent_id": intent_id,
            "snapshot_version": new_version,
            "error_code": None,
            "error_reason": None,
            "response": response_payload,
        }

    def _intent_handler_for(self, intent_kind: str):
        """Return the bound apply function for ``intent_kind``."""
        # Bridge handlers: existing apply_* methods are the V1 truth
        # for layout / agent_graph; we accept their parameter shapes.
        if intent_kind == "apply_layout_edit":
            def _h(params: Dict[str, Any]) -> int:
                op = params.get("operation")
                inner = params.get("parameters", {})
                if not isinstance(op, str):
                    raise ValueError("apply_layout_edit: missing operation")
                if not isinstance(inner, dict):
                    inner = {}
                return self.apply_layout_edit(operation=op, parameters=inner)
            return _h
        if intent_kind == "apply_agent_graph_command":
            def _h(params: Dict[str, Any]) -> Dict[str, Any]:
                cmd = params.get("command")
                inner = params.get("parameters", {})
                if not isinstance(cmd, str):
                    raise ValueError("apply_agent_graph_command: missing command")
                if not isinstance(inner, dict):
                    inner = {}
                return self.apply_agent_graph_command(command=cmd, parameters=inner)
            return _h
        if intent_kind == "acknowledge_drift":
            def _h(params: Dict[str, Any]) -> bool:
                fp = params.get("field_path")
                da = params.get("detected_at")
                if not isinstance(fp, str) or not isinstance(da, str):
                    raise ValueError(
                        "acknowledge_drift: field_path and detected_at "
                        "must be strings"
                    )
                return self.acknowledge_drift(field_path=fp, detected_at=da)
            return _h
        if intent_kind == "record_event":
            return self._handle_record_event
        if intent_kind == "append_turn":
            return self._handle_append_turn
        if intent_kind == "fork_agent":
            return self._handle_fork_agent
        if intent_kind == "move_agent_head":
            return self._handle_move_agent_head
        if intent_kind == "update_agent_session":
            return self._handle_update_agent_session
        if intent_kind == "add_blob":
            def _h(params: Dict[str, Any]) -> bool:
                key = params.get("key")
                blob = params.get("blob")
                if not isinstance(key, str) or not key:
                    raise ValueError("add_blob: key is required")
                if not isinstance(blob, dict):
                    raise ValueError("add_blob: blob must be a dict")
                with self._lock:
                    if key not in self._blobs:
                        self._blobs[key] = dict(blob)
                    self._dirty = True
                return True
            return _h
        if intent_kind == "set_cell_metadata":
            return self._handle_set_cell_metadata
        if intent_kind == "record_context_manifest":
            return self._handle_record_context_manifest
        if intent_kind == "record_run_frame":
            return self._handle_record_run_frame
        # BSP-007 K-OVERLAY slice: overlay-commit primitives.
        if intent_kind == "apply_overlay_commit":
            return self._handle_apply_overlay_commit
        if intent_kind == "revert_overlay_to_commit":
            return self._handle_revert_overlay_to_commit
        if intent_kind == "create_overlay_ref":
            return self._handle_create_overlay_ref

        # Registered-but-not-yet-implemented kinds: the registry
        # accepts the envelope so callers see K42 ("not yet implemented")
        # rather than K40 ("unknown kind"), which preserves the protocol
        # contract while individual handlers ship with later slices.
        _PENDING_SLICE = {
            # BSP-002 turn graph kinds (writer carries the agent graph
            # but not the turn-level conversation graph yet).
            # PLAN-S4.1: ``append_turn``, ``fork_agent``, ``move_agent_head``
            # active above; ``record_event`` reshaped above.
            "create_agent":             "BSP-002 turn graph slice",
            # PLAN-S4.2: ``update_agent_session`` active above.
            # Note: ``add_overlay`` / ``move_overlay_ref`` / ``update_ordering``
            # are reachable INSIDE ``apply_overlay_commit`` (BSP-007) via the
            # overlay applier dispatch; the standalone (BSP-002 turn-graph)
            # entrypoints below remain pending until that slice lands.
            "add_overlay":              "BSP-002 turn graph slice",
            "move_overlay_ref":         "BSP-002 turn graph slice",
            "update_ordering":          "BSP-002 turn graph slice",
            # BSP-007 overlay-commit kinds: handlers wired above
            # (apply_overlay_commit / revert_overlay_to_commit /
            # create_overlay_ref). K-OVERLAY slice landed.
            # BSP-008 ContextPacker / RunFrame kinds. Both ``record_context_manifest``
            # (K-AS-A / S3.5) and ``record_run_frame`` (K-CTXR / S6) ship with
            # active handlers above; this map is now empty for BSP-008 entries.
        }
        slice_label = _PENDING_SLICE.get(intent_kind, "future slice")
        def _stub(params: Dict[str, Any]) -> bool:
            raise ValueError(
                f"intent_kind {intent_kind!r} is registered but its "
                f"handler ships with the {slice_label}; not yet "
                "implemented in V1 MetadataWriter."
            )
        return _stub

    def _intent_failure(
        self, *, intent_id: str, code: str, reason: str,
    ) -> Dict[str, Any]:
        """Build a uniform K-coded failure response and log a marker."""
        marker = {
            "K40": "intent_unknown_kind",
            "K41": "intent_cas_rejected",
            "K42": "intent_validation_failed",
            "K43": "zone_write_failed",
        }.get(code, "intent_failed")
        logger.warning(
            "%s intent_id=%s reason=%s",
            marker, intent_id, reason,
            extra={
                "event.name": marker,
                "llmnb.intent_id": intent_id,
            },
        )
        with self._lock:
            version_at_reject = self._snapshot_version
        return {
            "applied": False,
            "already_applied": False,
            "intent_id": intent_id,
            "snapshot_version": version_at_reject,
            "error_code": code,
            "error_reason": reason,
            "response": None,
        }

    def iter_intent_log(self) -> List[Dict[str, Any]]:
        """Return a snapshot of the in-memory intent_applied log entries."""
        with self._lock:
            return [dict(entry) for entry in self._intent_log]

    # -- PLAN-S4.1 turn-graph persistence ---------------------------

    # Recognized roles per [concepts/turn](docs/atoms/concepts/turn.md)
    # §Schema.  Used as a soft validator on ``append_turn``; unknown
    # roles raise K42.
    _TURN_ROLES: FrozenSet[str] = frozenset({"operator", "agent", "system", "user", "assistant"})  # type: ignore[name-defined]
    _AGENT_REF_MOVE_REASONS: FrozenSet[str] = frozenset({"operator_revert", "operator_branch"})  # type: ignore[name-defined]

    def _zone_agents(self) -> Dict[str, Any]:
        """Return ``metadata.rts.zone.agents`` map, creating it if absent.

        Caller MUST hold ``self._lock``.  The map is keyed by
        ``agent_id`` per [concepts/agent](docs/atoms/concepts/agent.md);
        each value carries ``turns[]`` plus a ``session`` substructure
        with ``head_turn_id`` / ``last_seen_turn_id`` per PLAN-S4.1
        §3.B'.
        """
        return self._zone.setdefault("agents", {})

    def _all_persisted_turn_ids(self) -> set:
        """Return the set of all turn ids across all agents in the zone.

        Caller MUST hold ``self._lock``.  Used by ``append_turn`` to
        enforce zone-wide turn_id uniqueness per PLAN-S4.1 §3.A.
        """
        agents = self._zone.get("agents", {})
        out: set = set()
        for agent_state in agents.values():
            if not isinstance(agent_state, dict):
                continue
            for t in agent_state.get("turns", []) or []:
                tid = t.get("id") if isinstance(t, dict) else None
                if isinstance(tid, str):
                    out.add(tid)
        return out

    def _turn_in_agent_ancestry(
        self, agent_id: str, target_turn_id: str,
    ) -> bool:
        """Return True iff ``target_turn_id`` is reachable from the agent's head.

        Caller MUST hold ``self._lock``.  Walks ``parent_id`` from the
        agent's current ``head_turn_id`` (or last persisted turn) back
        through the union of all agents' ``turns[]`` arrays.  Used by
        ``move_agent_head`` and ``fork_agent`` per PLAN-S4.1 §3.A.
        """
        agents = self._zone.get("agents", {})
        agent_state = agents.get(agent_id)
        if not isinstance(agent_state, dict):
            return False
        # Build a global turn-by-id index across all agents (the chain
        # may cross agent boundaries by parent_id).
        all_turns: Dict[str, Dict[str, Any]] = {}
        for st in agents.values():
            if not isinstance(st, dict):
                continue
            for t in st.get("turns", []) or []:
                if isinstance(t, dict) and isinstance(t.get("id"), str):
                    all_turns[t["id"]] = t
        # Resolve starting cursor: prefer session.head_turn_id, else the
        # last appended turn for this agent.
        session = agent_state.get("session", {}) or {}
        cursor = session.get("head_turn_id")
        if not isinstance(cursor, str):
            agent_turns = agent_state.get("turns", []) or []
            cursor = agent_turns[-1].get("id") if agent_turns else None
        # Special case: target equals current cursor.
        visited: set = set()
        depth = 0
        max_depth = 10_000  # generous guard
        while isinstance(cursor, str) and depth < max_depth:
            if cursor in visited:
                return False  # cycle guard
            visited.add(cursor)
            if cursor == target_turn_id:
                return True
            t = all_turns.get(cursor)
            if t is None:
                return False
            cursor = t.get("parent_id")
            depth += 1
        return False

    def _handle_append_turn(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply one ``append_turn`` intent (PLAN-S4.1 §3.A).

        Appends an immutable turn record to
        ``metadata.rts.zone.agents.<agent_id>.turns[]``.  Validators
        (raise ``ValueError`` -> K42):

        * ``id`` (str, non-empty) — the turn id.
        * ``agent_id`` (str, non-empty) — the author.
        * ``role`` (str) — must be in :attr:`_TURN_ROLES`.
        * ``parent_id`` (str|None) — if non-null, MUST resolve to an
          existing turn in any agent's ``turns[]`` chain.
        * ``id`` MUST be unique zone-wide (no other agent's chain
          carries it).

        Returns ``{"ok": True, "turn_id": ..., "agent_id": ...}``.
        """
        # ``turn_id`` accepted as alias for ``id`` (the wire shape uses
        # ``id`` per concepts/turn.md but PLAN-S4.1 examples in the
        # supervisor migration use both).
        turn_id = params.get("id")
        if not isinstance(turn_id, str) or not turn_id:
            turn_id = params.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            raise ValueError("append_turn: id (or turn_id) is required")
        agent_id = params.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("append_turn: agent_id is required")
        role = params.get("role")
        if not isinstance(role, str) or role not in self._TURN_ROLES:
            raise ValueError(
                f"append_turn: role must be one of {sorted(self._TURN_ROLES)} "
                f"(got {role!r})"
            )
        parent_id = params.get("parent_id")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError(
                "append_turn: parent_id must be a string or null"
            )
        body = params.get("body")
        if body is None:
            body = params.get("content", "")
        if not isinstance(body, str):
            raise ValueError("append_turn: body must be a string")

        with self._lock:
            existing_ids = self._all_persisted_turn_ids()
            if turn_id in existing_ids:
                raise ValueError(
                    f"append_turn: duplicate turn id {turn_id!r} "
                    "already in zone"
                )
            if parent_id is not None and parent_id not in existing_ids:
                raise ValueError(
                    f"append_turn: unknown parent_id {parent_id!r} "
                    "(not in any agent's persisted turns[])"
                )
            agents = self._zone_agents()
            agent_state = agents.setdefault(agent_id, {
                "turns": [],
                "session": {},
            })
            turns_list = agent_state.setdefault("turns", [])
            record: Dict[str, Any] = {
                "id": turn_id,
                "parent_id": parent_id,
                "agent_id": agent_id,
                "claude_session_id": params.get("claude_session_id"),
                "role": role,
                "body": body,
                "spans": list(params.get("spans") or []),
                "cell_id": params.get("cell_id"),
                "created_at": params.get("created_at") or _utc_now_iso(),
            }
            if "provider" in params:
                record["provider"] = params["provider"]
            else:
                record["provider"] = "claude-code"
            turns_list.append(record)
            self._dirty = True
        return {"ok": True, "turn_id": turn_id, "agent_id": agent_id}

    def _handle_fork_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply one ``fork_agent`` intent (PLAN-S4.1 §3.A).

        Creates ``metadata.rts.zone.agents.<new_agent_id>`` with
        ``head_turn_id = at_turn_id`` and a fresh session.  Validators
        (raise ``ValueError`` -> K42):

        * ``source_agent_id`` (str, non-empty) — MUST already exist in
          ``zone.agents`` (or the source has zero persisted turns yet —
          in that case ``at_turn_id`` MUST also be null).
        * ``new_agent_id`` (str, non-empty) — MUST NOT already exist.
        * ``at_turn_id`` (str|None) — if non-null, MUST be in the source
          agent's ancestry.
        * ``case`` (``"A"`` or ``"B"``) — informational; not validated.
        * ``claude_session_id`` (str) — recorded on the new agent.
        """
        source_agent_id = params.get("source_agent_id")
        if not isinstance(source_agent_id, str) or not source_agent_id:
            raise ValueError("fork_agent: source_agent_id is required")
        new_agent_id = params.get("new_agent_id")
        if not isinstance(new_agent_id, str) or not new_agent_id:
            raise ValueError("fork_agent: new_agent_id is required")
        at_turn_id = params.get("at_turn_id")
        if at_turn_id is not None and not isinstance(at_turn_id, str):
            raise ValueError(
                "fork_agent: at_turn_id must be a string or null"
            )
        new_session_id = params.get("claude_session_id")
        # case is informational; we accept it without strict checking
        # (the supervisor sets it; the writer trusts).
        case = params.get("case")

        with self._lock:
            agents = self._zone_agents()
            if new_agent_id in agents:
                raise ValueError(
                    f"fork_agent: new_agent_id {new_agent_id!r} already "
                    "exists in zone.agents"
                )
            # Source-existence check is permissive: if the source agent
            # has not yet been persisted (e.g., the very first fork in a
            # zone with no prior append_turn calls) AND at_turn_id is
            # null, we accept the fork as bootstrapping the agent map.
            source_present = source_agent_id in agents
            if not source_present and at_turn_id is not None:
                raise ValueError(
                    f"fork_agent: source_agent_id {source_agent_id!r} not "
                    "found in zone.agents (cannot fork at a non-null turn "
                    "from an absent source)"
                )
            if at_turn_id is not None and source_present:
                if not self._turn_in_agent_ancestry(
                    source_agent_id, at_turn_id,
                ):
                    raise ValueError(
                        f"fork_agent: at_turn_id {at_turn_id!r} is not in "
                        f"agent {source_agent_id!r}'s ancestry"
                    )
            agents[new_agent_id] = {
                "turns": [],
                "session": {
                    "head_turn_id": at_turn_id,
                    "last_seen_turn_id": at_turn_id,
                    "claude_session_id": new_session_id,
                    "runtime_status": "idle",
                },
            }
            if case is not None:
                agents[new_agent_id]["session"]["fork_case"] = case
            self._dirty = True
        return {
            "ok": True,
            "new_agent_id": new_agent_id,
            "source_agent_id": source_agent_id,
            "at_turn_id": at_turn_id,
        }

    def _handle_move_agent_head(
        self, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply one ``move_agent_head`` intent (PLAN-S4.1 §3.A).

        Sets ``agents.<agent_id>.session.head_turn_id`` and
        ``last_seen_turn_id`` to the new value.  Validators (raise
        ``ValueError`` -> K42):

        * ``agent_id`` (str, non-empty) — MUST exist in zone.agents.
        * ``head_turn_id`` (str) — MUST be in the agent's ancestry.
        * ``last_seen_turn_id`` (str, optional) — defaults to
          ``head_turn_id``.
        """
        agent_id = params.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("move_agent_head: agent_id is required")
        head_turn_id = params.get("head_turn_id")
        if not isinstance(head_turn_id, str) or not head_turn_id:
            raise ValueError("move_agent_head: head_turn_id is required")
        last_seen = params.get("last_seen_turn_id", head_turn_id)
        if not isinstance(last_seen, str) or not last_seen:
            raise ValueError(
                "move_agent_head: last_seen_turn_id must be a non-empty "
                "string when provided"
            )

        with self._lock:
            agents = self._zone_agents()
            agent_state = agents.get(agent_id)
            if not isinstance(agent_state, dict):
                raise ValueError(
                    f"move_agent_head: agent {agent_id!r} not found in "
                    "zone.agents"
                )
            if not self._turn_in_agent_ancestry(agent_id, head_turn_id):
                raise ValueError(
                    f"move_agent_head: head_turn_id {head_turn_id!r} is "
                    f"not in agent {agent_id!r}'s ancestry"
                )
            session = agent_state.setdefault("session", {})
            session["head_turn_id"] = head_turn_id
            session["last_seen_turn_id"] = last_seen
            self._dirty = True
        return {
            "ok": True,
            "agent_id": agent_id,
            "head_turn_id": head_turn_id,
            "last_seen_turn_id": last_seen,
        }

    # -- PLAN-S4.2 update_agent_session handler ----------------------

    #: Fields that ``update_agent_session`` is permitted to patch.
    #: All are optional; the handler applies only those that are non-None
    #: in ``params`` (partial-update semantics).
    _SESSION_MUTABLE_FIELDS: FrozenSet[str] = frozenset({  # type: ignore[name-defined]
        "head_turn_id",
        "last_seen_turn_id",
        "runtime_status",
        "pid",
        "claude_session_id",
    })

    def _handle_update_agent_session(
        self, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply one ``update_agent_session`` intent (PLAN-S4.2).

        Updates one or more session fields on
        ``metadata.rts.zone.agents.<agent_id>.session``.  Uses partial-
        update semantics: only fields present and non-``None`` in
        ``params`` are written (except ``pid`` which accepts ``None``
        explicitly to clear it).

        Validators (raise ``ValueError`` -> K42):

        * ``agent_id`` (str, non-empty) — MUST exist in ``zone.agents``;
          raises K20 if absent.
        * ``runtime_status`` (str, optional) — when present, MUST be one
          of ``{"idle", "alive", "exited", "terminated"}``.

        Returns ``{"ok": True, "agent_id": ..., "updated": [...]}``.
        """
        agent_id = params.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("update_agent_session: agent_id is required")

        runtime_status = params.get("runtime_status")
        if runtime_status is not None:
            _VALID_RUNTIME_STATUSES = frozenset(
                {"idle", "alive", "exited", "terminated"}
            )
            if runtime_status not in _VALID_RUNTIME_STATUSES:
                raise ValueError(
                    f"update_agent_session: runtime_status must be one of "
                    f"{sorted(_VALID_RUNTIME_STATUSES)} (got {runtime_status!r})"
                )

        with self._lock:
            agents = self._zone_agents()
            agent_state = agents.get(agent_id)
            if not isinstance(agent_state, dict):
                raise ValueError(
                    f"K20: update_agent_session: agent {agent_id!r} not "
                    "found in zone.agents"
                )
            session = agent_state.setdefault("session", {})
            updated: List[str] = []

            # Apply each non-None field (pid is special: None is a valid
            # value meaning "clear the pid").
            for field in (
                "head_turn_id",
                "last_seen_turn_id",
                "runtime_status",
                "claude_session_id",
            ):
                if field in params and params[field] is not None:
                    session[field] = params[field]
                    updated.append(field)

            # pid explicitly accepts None (idle agent has pid=None).
            if "pid" in params:
                session["pid"] = params["pid"]
                updated.append("pid")

            self._dirty = True

        return {
            "ok": True,
            "agent_id": agent_id,
            "updated": updated,
        }

    def _handle_record_event(self, params: Dict[str, Any]) -> bool:
        """Apply one ``record_event`` intent (PLAN-S4.1 §3.B).

        Dispatches on ``parameters.kind``:

        * ``kind == "agent_ref_move"``: append a structured entry to
          ``metadata.rts.event_log[]`` per
          [protocols/family-d-event-log](docs/atoms/protocols/family-d-event-log.md).
          Required parameters: ``reason``, ``agent_id``, ``from_turn_id``,
          ``to_turn_id``.  ``reason`` MUST be in
          :attr:`_AGENT_REF_MOVE_REASONS`.
        * Otherwise: legacy drift-event path.  ``field_path`` required;
          routes through :meth:`append_drift_event` for backward
          compatibility (the original V1 shape).
        """
        kind = params.get("kind")
        if kind == "agent_ref_move":
            reason = params.get("reason")
            if not isinstance(reason, str) or reason not in self._AGENT_REF_MOVE_REASONS:
                raise ValueError(
                    f"record_event[agent_ref_move]: reason must be one of "
                    f"{sorted(self._AGENT_REF_MOVE_REASONS)} (got {reason!r})"
                )
            agent_id = params.get("agent_id")
            if not isinstance(agent_id, str) or not agent_id:
                raise ValueError(
                    "record_event[agent_ref_move]: agent_id is required"
                )
            entry = {
                "kind": "agent_ref_move",
                "reason": reason,
                "agent_id": agent_id,
                "from_turn_id": params.get("from_turn_id"),
                "to_turn_id": params.get("to_turn_id"),
                "recorded_at": params.get("recorded_at") or _utc_now_iso(),
            }
            with self._lock:
                rts_event_log = self._zone.setdefault("event_log", [])
                rts_event_log.append(entry)
                self._dirty = True
            return True
        # Legacy drift-event path (backward compat).
        fp = params.get("field_path")
        if not isinstance(fp, str) or not fp:
            raise ValueError(
                "record_event: either kind='agent_ref_move' or "
                "field_path (legacy drift) is required"
            )
        self.append_drift_event(
            field_path=fp,
            previous_value=params.get("previous_value"),
            current_value=params.get("current_value"),
            severity=str(params.get("severity", "info")),
            detected_at=params.get("detected_at"),
        )
        return True

    # -- BSP-005 §6.1 / S0.5 cell-kinds -----------------------------

    # Recognized parameter keys on the ``set_cell_metadata`` intent.
    # Anything else is preserved on the cell record verbatim (forward-
    # compat with future per-cell flags) but is not validated here.
    _SET_CELL_METADATA_FLAG_KEYS: Tuple[str, ...] = (
        "pinned", "excluded", "scratch", "checkpoint", "read_only",
    )

    def _handle_set_cell_metadata(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply one ``set_cell_metadata`` intent.

        Per BSP-005 §6.1 / [cell-kinds atom](docs/atoms/concepts/cell-kinds.md)
        and [cell atom](docs/atoms/concepts/cell.md) §"Schema":

        * Required: ``cell_id`` (string).
        * Required from S0.5 forward: ``kind`` (one of the eight enum
          values).
        * Validators (raise ``ValueError`` -> K42 with structured
          reason):

          - ``unknown_cell_kind``: ``kind`` not in
            :data:`CELL_KINDS`.
          - ``markdown_must_have_no_agent``: ``kind == "markdown"``
            with non-null ``bound_agent_id``.

        * Per-kind soft rules (NOT errors, the writer normalizes):

          - ``markdown`` cells have ``bound_agent_id`` stripped
            (forced to ``None``).
          - ``scratch`` / ``checkpoint`` cells without an explicit
            ``bound_agent_id`` get ``None`` written through.

        * Reserved kinds (``tool | artifact | control | native``)
          round-trip identically to active kinds; the writer does NOT
          dispatch them anywhere -- the renderer falls through to the
          kind-label-only view per the cell-kinds atom invariants.

        Returns ``{"ok": True, "cell_id": ..., "kind": ...}`` on
        success.
        """
        cell_id = params.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("set_cell_metadata: cell_id is required")
        # ``kind`` is the new field S0.5 lands.  We accept its absence
        # iff the cell already exists with a kind (operator is editing
        # flags only); otherwise it's required.
        kind = params.get("kind", _UNSET)
        bound_agent_id = params.get("bound_agent_id", _UNSET)
        with self._lock:
            existing = self._cells.get(cell_id)
        if kind is _UNSET:
            if existing is None or "kind" not in existing:
                raise ValueError(
                    "set_cell_metadata: kind is required (kind_required)"
                )
            kind_value: str = existing["kind"]
        else:
            if not isinstance(kind, str) or kind not in CELL_KINDS:
                raise ValueError(
                    f"set_cell_metadata: unknown_cell_kind {kind!r} "
                    f"(allowed: {sorted(CELL_KINDS)})"
                )
            kind_value = kind

        # Per-kind constraints.  ``markdown`` MUST NOT carry a non-null
        # bound_agent_id (cell-kinds atom invariant).  ``scratch`` and
        # ``checkpoint`` SHOULD have a null bound_agent_id; the writer
        # normalizes to None when not provided.
        if kind_value == "markdown":
            if bound_agent_id is not _UNSET and bound_agent_id is not None:
                raise ValueError(
                    "set_cell_metadata: markdown_must_have_no_agent "
                    f"(kind=markdown, bound_agent_id={bound_agent_id!r})"
                )
            normalized_bound: Optional[str] = None
        elif kind_value in ("scratch", "checkpoint"):
            if bound_agent_id is _UNSET:
                normalized_bound = (
                    existing.get("bound_agent_id")
                    if existing is not None else None
                )
            else:
                normalized_bound = bound_agent_id  # type: ignore[assignment]
            # SHOULD-rule: not enforced as K42, but normalize None when
            # the caller explicitly passed None.
        elif kind_value == "agent":
            if bound_agent_id is _UNSET:
                normalized_bound = (
                    existing.get("bound_agent_id")
                    if existing is not None else None
                )
            else:
                normalized_bound = bound_agent_id  # type: ignore[assignment]
        else:
            # Reserved kinds: round-trip whatever the caller sent.
            if bound_agent_id is _UNSET:
                normalized_bound = (
                    existing.get("bound_agent_id")
                    if existing is not None else None
                )
            else:
                normalized_bound = bound_agent_id  # type: ignore[assignment]

        # Build / merge the cell record.
        with self._lock:
            record = dict(self._cells.get(cell_id, {}))
            record["kind"] = kind_value
            record["bound_agent_id"] = normalized_bound
            # Optional fields: section_id, capabilities, plus the flag
            # set.  Only overwrite when the caller passed a value.
            if "section_id" in params:
                record["section_id"] = params["section_id"]
            elif "section_id" not in record:
                record["section_id"] = None
            if "capabilities" in params:
                caps = params["capabilities"]
                record["capabilities"] = list(caps) if isinstance(caps, list) else []
            elif "capabilities" not in record:
                record["capabilities"] = []
            for flag_key in self._SET_CELL_METADATA_FLAG_KEYS:
                if flag_key in params:
                    record[flag_key] = bool(params[flag_key])
            # Clear the back-fill marker (if any) once the operator has
            # explicitly written the kind.
            record.pop("_kind_back_filled", None)
            self._cells[cell_id] = record
            self._dirty = True
        return {"ok": True, "cell_id": cell_id, "kind": kind_value}

    # -- BSP-005 S5.0 cell-text canonical accessors ------------------

    # PLAN-S5.0 §3.5: cells[<id>] schema collapses to ``{ text, outputs,
    # bound_agent_id }``. The ``kind``, ``pinned``, ``excluded`` fields
    # become *parse-derived* from ``text``. ``cell_view`` returns the
    # parsed view with text-hash caching so repeated reads of an
    # unchanged cell don't re-walk the parser.
    #
    # We keep a parallel ``_cell_view_cache`` keyed by cell_id; entries
    # invalidate when ``set_cell_text`` writes a new text. The cache is
    # populated lazily on first ``cell_view`` call.

    def get_cell_text(self, cell_id: str) -> Optional[str]:
        """Return the canonical cell ``text`` (PLAN-S5.0 §3.5) or None."""
        with self._lock:
            record = self._cells.get(cell_id)
            if record is None:
                return None
            return record.get("text")

    def set_cell_text(self, cell_id: str, text: str) -> None:
        """Write ``text`` as the canonical source for ``cell_id``.

        Invalidates the parsed-view cache for the cell. Marks the
        writer dirty so the next snapshot persists the change.
        """
        if not isinstance(text, str):
            raise ValueError("set_cell_text: text must be a str")
        with self._lock:
            record = dict(self._cells.get(cell_id, {}))
            record["text"] = text
            self._cells[cell_id] = record
            # Invalidate the cached parse.
            if hasattr(self, "_cell_view_cache"):
                self._cell_view_cache.pop(cell_id, None)
            self._dirty = True

    def delete_cell(self, cell_id: str) -> bool:
        """Remove a cell record. Returns True iff it was present."""
        with self._lock:
            existed = cell_id in self._cells
            self._cells.pop(cell_id, None)
            if hasattr(self, "_cell_view_cache"):
                self._cell_view_cache.pop(cell_id, None)
            if existed:
                self._dirty = True
            return existed

    # -- PLAN-S5.0.1b §3.6 — hash-mode config + contamination schema --

    #: Config keys living at ``metadata.rts.config[key]`` (not under
    #: ``recoverable``/``volatile``). Hash-mode settings are kernel-
    #: scoped and round-trip through ``get_config_setting`` /
    #: ``set_hash_mode``. The pin itself is NEVER stored in the
    #: notebook — only the fingerprint (a one-way hash). See
    #: :func:`llm_kernel.magic_hash.magic_pin_fingerprint`.
    _HASH_MODE_CONFIG_KEYS: FrozenSet[str] = frozenset({
        "magic_hash_enabled",
        "magic_pin_fingerprint",
    })

    def get_config_setting(self, name: str) -> Any:
        """Read a top-level ``metadata.rts.config[name]`` setting.

        PLAN-S5.0.1b §3.6 — convenience accessor used by S5.0.1a's
        contamination detector (``agent_supervisor._magic_hash_enabled``)
        and the parser/dispatcher hash-mode plumbing in 5.0.1b. Returns
        the stored value or ``None`` if absent. The reader is
        forward-compatible: callers must coerce to bool / str as
        appropriate.

        Defaults: ``magic_hash_enabled`` defaults to ``False``;
        ``magic_pin_fingerprint`` defaults to ``None``. All other keys
        return ``None`` when unset.
        """
        if not isinstance(name, str) or not name:
            return None
        with self._lock:
            if name == "magic_hash_enabled":
                return bool(self._config.get("magic_hash_enabled", False))
            if name == "magic_pin_fingerprint":
                return self._config.get("magic_pin_fingerprint")
            return self._config.get(name)

    def set_hash_mode(
        self, enabled: bool, fingerprint: Optional[str],
    ) -> None:
        """Atomically set the hash-mode pair under ``metadata.rts.config``.

        PLAN-S5.0.1b §3.6 — used by ``@auth set`` / ``@auth rotate`` /
        ``@auth off``. Both fields move together: enabling without a
        fingerprint, or disabling while leaving a fingerprint, is a
        contract violation by the caller (the auth handlers enforce
        the invariant).

        Idempotent: writing the same pair twice is a no-op apart from
        the dirty flag (already-dirty stays dirty).
        """
        if not isinstance(enabled, bool):
            raise TypeError(
                f"enabled must be bool; got {type(enabled).__name__}"
            )
        if fingerprint is not None and not isinstance(fingerprint, str):
            raise TypeError(
                f"fingerprint must be str|None; got "
                f"{type(fingerprint).__name__}"
            )
        with self._lock:
            self._config["magic_hash_enabled"] = bool(enabled)
            if fingerprint is None:
                # Clear the fingerprint entirely when hash mode is off
                # so a stale pin can't survive.
                self._config.pop("magic_pin_fingerprint", None)
            else:
                self._config["magic_pin_fingerprint"] = fingerprint
            self._dirty = True

    def flag_cells_contaminated_by_agent(
        self,
        *,
        agent_id: str,
        line: str,
        source: str,
        layer: str,
    ) -> List[str]:
        """Mark every cell bound to ``agent_id`` as contaminated.

        PLAN-S5.0.1b §3.6 — first-class realization of the duck-typed
        method S5.0.1a's ``agent_supervisor._flag_contaminated``
        already calls (it falls back to ``_diagnostics.mark`` when the
        method is missing). This implementation:

        * Walks ``self._cells`` for entries where
          ``record["bound_agent_id"] == agent_id``.
        * Sets ``record["contaminated"] = True``.
        * Appends a ``{detected_at, line, reason, layer}`` entry to
          ``record["contamination_log"]`` (append-only audit; the
          caller passes a ``layer`` of ``"plain"`` /
          ``"hashed_emission_ban"``).
        * Returns the list of cell_ids that were flagged.

        The line is truncated to 256 chars before storage (mirrors
        ``agent_supervisor._flag_contaminated``'s own bound) so a
        flooded contamination path can't unbounded-grow the notebook.

        Cell-Manager precondition gates that *use* the contaminated
        flag (K3E, K3F) are 5.0.1c scope; this slice only adds the
        schema + writer method.
        """
        if not isinstance(agent_id, str) or not agent_id:
            return []
        truncated = line[:256] if isinstance(line, str) else ""
        flagged: List[str] = []
        ts = _utc_now_iso()
        with self._lock:
            for cell_id, record in self._cells.items():
                if not isinstance(record, dict):
                    continue
                if record.get("bound_agent_id") != agent_id:
                    continue
                record["contaminated"] = True
                log = record.get("contamination_log")
                if not isinstance(log, list):
                    log = []
                log.append({
                    "detected_at": ts,
                    "line": truncated,
                    "reason": f"agent_emit:{source}",
                    "layer": layer,
                })
                record["contamination_log"] = log
                self._cells[cell_id] = record
                flagged.append(cell_id)
            if flagged:
                self._dirty = True
        return flagged

    # -- PLAN-S5.0.1c §3.10 — contamination flag query / clear ---------

    #: Verbatim acceptance string format (PLAN-S5.0.1c §3.11). The
    #: prefix is FIXED — every operator opening the notebook in any
    #: text editor sees the same plain-English statement. The
    #: validator below rejects any other shape so a proxied write
    #: cannot smuggle arbitrary text into this slot.
    _INJECTION_ACCEPTANCE_PREFIX: str = (
        "The Operator Has Accepted Arbitrary Code Injection at "
    )

    def is_cell_contaminated(self, cell_id: str) -> bool:
        """Return True iff ``cells[cell_id].contaminated == True``.

        PLAN-S5.0.1c §3.10 helper consumed by ``CellManager``'s
        precondition gates. Read-only, lock-protected, returns False
        for unknown cell ids (callers don't need to check first).
        """
        if not isinstance(cell_id, str) or not cell_id:
            return False
        with self._lock:
            record = self._cells.get(cell_id)
            if not isinstance(record, dict):
                return False
            return bool(record.get("contaminated", False))

    def reset_cell_contamination(self, cell_id: str) -> bool:
        """Clear ``cells[cell_id].contaminated`` + ``contamination_log``.

        PLAN-S5.0.1c §3.10. Operator-click entry point exposed via
        ``CellManager.reset_contamination``; this is the writer-side
        implementation. Returns True iff a flag was actually flipped
        from True to False (idempotent on already-clean cells:
        returns False, no mutation, no dirty-flag flip).

        AMBIGUITY-FLAG: PLAN §3.10 says the operator click "resets"
        the flag but does not specify log retention. We chose **full
        clear** (delete the ``contamination_log`` list) so a fresh
        contamination event after reset starts from empty — operator
        intent on click is "this cell is clean now, treat it as such".
        Audit history of the original detection lives in the kernel
        diagnostics marker stream (``_diagnostics.mark`` calls in
        ``agent_supervisor._flag_contaminated``), not in the per-cell
        log.
        """
        if not isinstance(cell_id, str) or not cell_id:
            return False
        with self._lock:
            record = self._cells.get(cell_id)
            if not isinstance(record, dict):
                return False
            if not record.get("contaminated"):
                return False
            record["contaminated"] = False
            record.pop("contamination_log", None)
            self._cells[cell_id] = record
            self._dirty = True
            return True

    # -- PLAN-S5.0.1c §3.11 — verbatim injection-acceptance flag -------

    def get_injection_acceptance(self) -> Optional[str]:
        """Return the ``injection_acceptance`` string or None.

        PLAN-S5.0.1c §3.11. Read-only accessor. Returns the verbatim
        string previously written by :meth:`accept_injection_risk`,
        or ``None`` when the operator has never accepted.
        """
        with self._lock:
            value = self._config.get("injection_acceptance")
        if isinstance(value, str) and value.startswith(
            self._INJECTION_ACCEPTANCE_PREFIX
        ):
            return value
        return None

    def accept_injection_risk(self) -> str:
        """Persist the verbatim operator-acceptance string. Returns it.

        PLAN-S5.0.1c §3.11. Writes the literal phrase

            ``"The Operator Has Accepted Arbitrary Code Injection at <ISO8601>"``

        to ``metadata.rts.config.injection_acceptance``. The verbatim
        format is FIXED — the validator below rejects any other shape
        so a proxied write attempting to smuggle arbitrary text fails.

        **Idempotent on first set**: if the field is already populated
        with a valid verbatim string, this method is a NO-OP and
        returns the EXISTING string (preserving the original "accepted
        at" timestamp). The caller-side K3G emit should also be
        skipped on the no-op branch — callers can compare the returned
        string against a pre-call ``get_injection_acceptance()`` to
        detect the first-set case.
        """
        with self._lock:
            existing = self._config.get("injection_acceptance")
            if isinstance(existing, str) and existing.startswith(
                self._INJECTION_ACCEPTANCE_PREFIX
            ):
                # Idempotent: return the original record verbatim.
                return existing
            new_value = (
                f"{self._INJECTION_ACCEPTANCE_PREFIX}{_utc_now_iso()}"
            )
            # Validator: ensure the constructed string round-trips
            # through our own verifier (defense against a future
            # refactor that breaks the format invariant).
            if not self._validate_injection_acceptance(new_value):
                raise RuntimeError(
                    "accept_injection_risk: constructed string failed "
                    "format validator (kernel bug, not operator input)"
                )
            self._config["injection_acceptance"] = new_value
            self._dirty = True
            return new_value

    # -- PLAN-S5.0.2 — magic code generator config + provenance --------

    #: Built-in V1 generator names. Operators may extend this in V2+
    #: via the registration intent (deferred slice). The list lives at
    #: ``metadata.rts.config.magic_code_generators``.
    _DEFAULT_GENERATORS: Tuple[str, ...] = ("template", "expand", "import")

    def read_config_generators(self) -> List[str]:
        """Read ``metadata.rts.config.magic_code_generators``.

        PLAN-S5.0.2 §6. Defaults to the V1 built-ins when the field is
        unset (so an operator who hasn't pinned the list still gets
        ``@@template`` / ``@@expand`` / ``@@import``). Returns a copy
        so callers can mutate without racing the writer.
        """
        with self._lock:
            value = self._config.get("magic_code_generators")
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return list(value)
        return list(self._DEFAULT_GENERATORS)

    def read_config_templates(self) -> Dict[str, str]:
        """Read ``metadata.rts.config.templates``.

        PLAN-S5.0.2 §6 — operator-defined named templates that
        ``@@template <name>`` looks up. Returns a fresh dict.
        """
        with self._lock:
            value = self._config.get("templates")
        if isinstance(value, dict):
            return {
                k: v for k, v in value.items()
                if isinstance(k, str) and isinstance(v, str)
            }
        return {}

    def set_config_template(self, name: str, body: str) -> None:
        """Write one template under ``metadata.rts.config.templates``.

        PLAN-S5.0.2 §6 — operator helper for tests / programmatic
        seeding. The operator-typed path is editing
        ``metadata.rts.config.templates`` directly via the canonical
        notebook editor; this method is the kernel-side mirror.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("set_config_template: name must be a non-empty str")
        if not isinstance(body, str):
            raise ValueError("set_config_template: body must be a str")
        with self._lock:
            templates = self._config.get("templates")
            if not isinstance(templates, dict):
                templates = {}
            templates[name] = body
            self._config["templates"] = templates
            self._dirty = True

    def get_operator_pin(self) -> Optional[str]:
        """Return the operator pin when hash mode is on, else None.

        PLAN-S5.0.2 §4.2. The pin itself is NEVER stored in the
        notebook (only its fingerprint, per S5.0.1b). V1 sources the
        pin from the ``LLMNB_OPERATOR_PIN`` env var; V1.5+ may add an
        OS-keychain integration. Returns None when hash mode is off
        OR when the env var is unset / empty.
        """
        if not bool(self.get_config_setting("magic_hash_enabled")):
            return None
        import os

        pin = os.environ.get("LLMNB_OPERATOR_PIN")
        if isinstance(pin, str) and pin:
            return pin
        return None

    def get_workspace_root(self) -> Path:
        """Return the workspace root path. PLAN-S5.0.2 §4.2 helper."""
        return self._workspace_root

    def get_cell_record(self, cell_id: str) -> Optional[Dict[str, Any]]:
        """Return a shallow copy of ``cells[cell_id]`` (or None).

        PLAN-S5.0.2 — convenience accessor used by the cell-manager
        precondition checks and by tests asserting on provenance
        fields. Read-only; mutations on the returned dict don't
        affect the writer.
        """
        if not isinstance(cell_id, str) or not cell_id:
            return None
        with self._lock:
            record = self._cells.get(cell_id)
            if record is None:
                return None
            return dict(record)

    def get_cell_layout_order(self) -> List[str]:
        """Return cell ids in layout-walk order, fallback dict order.

        PLAN-S5.0.2 §3 — used by ``CellManager.insert_cells_with_provenance``
        to find the position of ``after_cell_id`` and append new cells
        right after it. We walk ``layout.tree.children`` recursively
        collecting any node whose ``id`` is a known cell key; cells
        not referenced from the layout fall to the end in
        dict-insertion order.
        """
        with self._lock:
            tree = self._layout.get("tree")
            cell_keys: List[str] = list(self._cells.keys())
        ordered: List[str] = []
        seen: Set[str] = set()
        if isinstance(tree, dict):
            stack: List[Any] = [tree]
            while stack:
                node = stack.pop(0)
                if isinstance(node, dict):
                    nid = node.get("id")
                    if isinstance(nid, str) and nid in cell_keys and nid not in seen:
                        ordered.append(nid)
                        seen.add(nid)
                    children = node.get("children")
                    if isinstance(children, list):
                        stack = list(children) + stack
        for cid in cell_keys:
            if cid not in seen:
                ordered.append(cid)
                seen.add(cid)
        return ordered

    def insert_generated_cell(
        self,
        new_cell_id: str,
        text: str,
        *,
        after_cell_id: str,
        generated_by: str,
        generated_at: str,
    ) -> None:
        """Persist a generator-emitted cell with provenance.

        PLAN-S5.0.2 §3 / §6. Writes the canonical fields:

        * ``cells[new_cell_id].text = text``
        * ``cells[new_cell_id].generated_by = generated_by``
        * ``cells[new_cell_id].generated_at = generated_at``

        Validates: ``generated_by`` (when non-null) must reference a
        known cell id; the ISO timestamp must parse. Marks the writer
        dirty.

        Raises ``ValueError`` (mapped to K3J at the dispatch boundary)
        when ``generated_by`` is None / empty.
        """
        if not isinstance(new_cell_id, str) or not new_cell_id:
            raise ValueError("insert_generated_cell: new_cell_id required")
        if not isinstance(text, str):
            raise ValueError("insert_generated_cell: text must be a str")
        if not isinstance(generated_by, str) or not generated_by:
            raise ValueError(
                "insert_generated_cell: generated_by required (K3J)"
            )
        if not isinstance(generated_at, str) or not generated_at:
            raise ValueError(
                "insert_generated_cell: generated_at required"
            )
        with self._lock:
            if generated_by not in self._cells:
                raise ValueError(
                    "insert_generated_cell: generated_by references "
                    f"unknown cell {generated_by!r}"
                )
            record = dict(self._cells.get(new_cell_id, {}))
            record["text"] = text
            record["generated_by"] = generated_by
            record["generated_at"] = generated_at
            self._cells[new_cell_id] = record
            if hasattr(self, "_cell_view_cache"):
                self._cell_view_cache.pop(new_cell_id, None)
            self._dirty = True

    @classmethod
    def _validate_injection_acceptance(cls, value: Any) -> bool:
        """Return True iff ``value`` is a valid verbatim acceptance string.

        PLAN-S5.0.1c §3.11. Format: the literal prefix
        ``"The Operator Has Accepted Arbitrary Code Injection at "``
        followed by an ISO-8601 timestamp. Rejects ``None``, empty
        strings, and any other prose — the wording is intentionally
        un-localizable so operator searches across a corpus surface a
        single canonical phrase.
        """
        if not isinstance(value, str):
            return False
        if not value.startswith(cls._INJECTION_ACCEPTANCE_PREFIX):
            return False
        ts_part = value[len(cls._INJECTION_ACCEPTANCE_PREFIX):]
        if not ts_part:
            return False
        # Sanity-check the timestamp parses as ISO-8601.
        try:
            from datetime import datetime
            # Trim trailing 'Z' (datetime.fromisoformat doesn't accept
            # it on Python < 3.11) for portability.
            normalized = ts_part[:-1] if ts_part.endswith("Z") else ts_part
            datetime.fromisoformat(normalized)
        except (ValueError, ImportError):
            return False
        return True

    def cell_view(self, cell_id: str):
        """Return the parsed :class:`cell_text.ParsedCell` for ``cell_id``.

        Per PLAN-S5.0 §3.5: text-hash caching. Cache hit when the
        text's SHA-256 matches the entry; miss re-runs ``parse_cell``.
        Returns ``None`` when the cell does not exist.
        """
        from .cell_text import parse_cell  # lazy import

        with self._lock:
            record = self._cells.get(cell_id)
            if record is None:
                return None
            text = record.get("text", "")
            cache = getattr(self, "_cell_view_cache", None)
            if cache is None:
                self._cell_view_cache: Dict[str, Tuple[str, Any]] = {}
                cache = self._cell_view_cache
            text_hash = _sha256_hex(text or "")
            cached = cache.get(cell_id)
            if cached is not None and cached[0] == text_hash:
                return cached[1]
        # Parse outside the lock so a future K30/K31 raise doesn't hold
        # writes against the snapshot lock.
        view = parse_cell(text or "")
        with self._lock:
            self._cell_view_cache[cell_id] = (text_hash, view)
        return view

    def migrate_cells_to_canonical_text(self) -> Dict[str, Any]:
        """One-shot: re-emit pre-S5.0 cell records as canonical text form.

        PLAN-S5.0 §3.5 — a cell record carrying explicit ``kind`` /
        ``pinned`` / ``excluded`` / ``scratch`` / ``checkpoint`` /
        ``bound_agent_id`` fields without a ``text`` field is migrated:
        the canonical text is rebuilt from those fields and stored in
        the new ``text`` slot. The pre-existing fields are NOT removed
        (back-compat: older readers may still want them); the writer's
        ``cell_view`` accessor henceforth derives them from ``text``.

        Idempotent: a cell that already carries ``text`` is skipped.
        Returns a marker dict logging the cells migrated.
        """
        migrated: List[Dict[str, Any]] = []
        with self._lock:
            for cell_id, record in list(self._cells.items()):
                if not isinstance(record, dict):
                    continue
                if "text" in record and isinstance(record["text"], str):
                    continue
                kind = record.get("kind", DEFAULT_CELL_KIND) or DEFAULT_CELL_KIND
                bound_agent_id = record.get("bound_agent_id")
                pieces: List[str] = []
                # Cell-magic declaration. ``agent`` cells default to no
                # explicit ``@@agent`` line — but if a bound_agent_id is
                # set, emit the declaration so the binding round-trips
                # through the parser.
                if kind == "agent" and bound_agent_id:
                    pieces.append(f"@@agent {bound_agent_id}")
                elif kind != "agent":
                    pieces.append(f"@@{kind}")
                # Flag line magics.
                if record.get("pinned"):
                    pieces.append("@pin")
                if record.get("excluded"):
                    pieces.append("@exclude")
                # Body — whatever the existing record stored as a
                # source field. Pre-S5.0 records didn't have a single
                # canonical body slot; we accept ``source`` if present
                # (that's the field the extension serializer writes).
                source = record.get("source") or ""
                if source:
                    pieces.append(source)
                new_text = "\n".join(p for p in pieces if p)
                record["text"] = new_text
                self._cells[cell_id] = record
                migrated.append({
                    "cell_id": cell_id,
                    "from_kind": kind,
                    "had_pinned": bool(record.get("pinned")),
                    "had_excluded": bool(record.get("excluded")),
                })
            if migrated:
                self._dirty = True
        marker = {
            "migration": "BSP-005-S5.0-cell-text-canonical",
            "migrated_count": len(migrated),
            "cells": migrated,
        }
        if migrated:
            try:
                self._workspace_root.mkdir(parents=True, exist_ok=True)
                marker_path = (
                    self._workspace_root / ".llmnb-s5-0-cell-text-migration.json"
                )
                marker_path.write_text(
                    json.dumps(marker, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError:  # pragma: no cover - defensive
                logger.warning(
                    "metadata writer: could not write S5.0 cell-text migration "
                    "marker; continuing"
                )
        return marker

    # -- BSP-008 §3 / S3.5 record_context_manifest ------------------

    def _handle_record_context_manifest(
        self, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Persist one ContextManifest under ``metadata.rts.zone.context_manifests``.

        Per BSP-008 §3 / [concepts/context-manifest](docs/atoms/concepts/context-manifest.md):
        the ``record_context_manifest`` intent carries a fully-formed
        :class:`context_packer.ContextManifest` dict -- the writer's job
        is single-writer persistence into the zone substructure, not
        re-validation of the V1 walk (the packer is the source of truth
        per :func:`context_packer.pack`).

        Validation surface (raises ``ValueError`` -> K42):

        * ``params['manifest']`` MUST be a dict (or ``params`` itself
          MAY carry the manifest fields directly per the brief's
          "store ``params`` into ..." phrasing -- we accept both).
        * ``manifest['manifest_id']`` MUST be a non-empty string.
        * ``manifest['cell_id']`` MUST be a non-empty string.

        Storage key: ``metadata.rts.zone.context_manifests[<manifest_id>]``.
        Manifests are append-only per the atom's invariants; submitting
        the same ``manifest_id`` twice through the intent registry hits
        the BSP-003 §6 step 2 idempotency check (``intent_id`` keyed) so
        re-submission is a no-op even if the manifest payload were
        re-derived.
        """
        # Accept both shapes: params={"manifest": {...}} OR params={...}
        # (the brief's "store params" phrasing). The former matches the
        # contract atom's envelope example; the latter is what the
        # AgentSupervisor will likely build for symmetry with other
        # intent shapes once S6 wires it up.
        manifest = params.get("manifest")
        if not isinstance(manifest, dict):
            manifest = params
        if not isinstance(manifest, dict):
            raise ValueError(
                "record_context_manifest: manifest must be a dict"
            )
        manifest_id = manifest.get("manifest_id")
        if not isinstance(manifest_id, str) or not manifest_id:
            raise ValueError(
                "record_context_manifest: manifest_id is required"
            )
        cell_id = manifest.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError(
                "record_context_manifest: cell_id is required"
            )
        # Defensive copy so the caller's dict is not mutated by later
        # snapshot serialization passes.
        record = _deepcopy_json(manifest)
        with self._lock:
            zone = self._zone.setdefault("context_manifests", {})
            zone[manifest_id] = record
            self._dirty = True
        return {
            "ok": True,
            "manifest_id": manifest_id,
            "cell_id": cell_id,
        }

    # -- BSP-007 K-OVERLAY overlay-commit handlers ------------------

    def _set_cell_flag(
        self, cell_id: str, flag: str, value: bool,
    ) -> bool:
        """Mutate one of the canonical cell flags on ``self._cells``.

        The four flags pin / exclude / scratch / checkpoint are V1's
        single-cell-property toggles per the
        [pin-exclude-scratch-checkpoint atom]
        (docs/atoms/operations/pin-exclude-scratch-checkpoint.md).

        Used by:
          * ``_handle_set_cell_metadata`` (the BSP-003 §5 entrypoint),
            which writes flags through the ``params`` dict directly;
          * the BSP-007 overlay applier, which wraps the same mutation
            in an overlay commit so the operator timeline records it.

        Returns True iff the flag's stored value changed. Callers
        should also flip ``self._dirty`` (this helper does NOT bump
        ``_snapshot_version``; the dispatcher does that on success).
        """
        if flag not in self._SET_CELL_METADATA_FLAG_KEYS:
            raise ValueError(
                f"_set_cell_flag: unknown flag {flag!r} "
                f"(allowed: {self._SET_CELL_METADATA_FLAG_KEYS})"
            )
        coerced = bool(value)
        with self._lock:
            record = dict(self._cells.get(cell_id, {}))
            previous = record.get(flag)
            record[flag] = coerced
            self._cells[cell_id] = record
            if hasattr(self, "_cell_view_cache"):
                self._cell_view_cache.pop(cell_id, None)
            self._dirty = True
        return previous != coerced

    def _zone_overlay_state(self) -> Dict[str, Any]:
        """Return ``metadata.rts.zone.overlay`` substructure (init if absent).

        The locked-interface contract per the BSP-007 K-OVERLAY brief:
        the dict ALWAYS has ``commits`` (list) and ``refs`` (dict).
        Caller MUST hold ``self._lock``.
        """
        return _overlay_applier.ensure_overlay_state(self._zone)

    def _build_overlay_state_view(self) -> Dict[str, Any]:
        """Build the per-call state view the applier mutates.

        Caller MUST hold ``self._lock``. The returned dict references
        the writer's live in-memory dicts directly (no copy). The
        applier deep-copies into a work_state, applies ops, and on
        success swaps the work copies back IN PLACE so the writer's
        references stay valid.

        We inject only the read-only ``is_cell_executing`` predicate
        as a closure so the applier never imports the writer module.
        Mutating closures (``set_cell_flag``, ``set_cell_metadata``)
        are NOT passed through -- the applier mutates the work_state's
        dicts directly so atomic rollback covers them.

        The zone substructures (``sections``, ``overlay``,
        ``per_turn_overlays``) are NOT initialised here -- the applier
        does that lazily inside :func:`overlay_applier.apply_commit`
        only when an apply actually succeeds, so a rejected commit
        leaves the writer's ``self._zone`` byte-identical to its
        pre-call state.
        """
        return {
            "cells":             self._cells,
            "sections":          self._zone.get("sections") or {},
            "overlay":           self._zone.get("overlay") or {},
            "per_turn_overlays": self._zone.get("per_turn_overlays") or {},
            # Markers the applier reads to know whether the sub-dicts
            # are aliases of live state vs fresh defaults. We pass a
            # callable the applier invokes on success to actually
            # install the live sub-dicts.
            "_install_zone_subdict": self._install_zone_subdict,
            "is_cell_executing": self._is_cell_executing_for_overlay,
        }

    def _install_zone_subdict(self, key: str) -> Dict[str, Any]:
        """Install a writer-owned sub-dict at ``self._zone[key]`` if absent.

        Caller MUST hold ``self._lock``. Returns the live dict.
        Used by the overlay applier only on the success path so a
        rejected commit doesn't dirty ``self._zone``.
        """
        existing = self._zone.get(key)
        if not isinstance(existing, dict):
            existing = {}
            self._zone[key] = existing
        return existing

    def _is_cell_executing_for_overlay(self, cell_id: str) -> bool:
        """Predicate: is a run currently in flight on ``cell_id``?

        BSP-007 §7 K95 fires when an overlay commit would touch a cell
        the kernel is actively executing. The writer doesn't own the
        run state directly; the AgentSupervisor / RunTracker does. We
        consult those if attached, else fall back to a per-cell
        "_executing" flag the test harness can set via
        :meth:`set_cell_execution_state`.
        """
        # Check the per-cell test seam first.
        with self._lock:
            record = self._cells.get(cell_id)
            if isinstance(record, dict) and bool(record.get("_executing")):
                return True
        # Run tracker: if any open span is bound to this cell, we
        # treat the cell as executing. The tracker's open spans live
        # in ``iter_runs()`` with ``endTimeUnixNano: None``.
        tracker = self._run_tracker
        if tracker is not None:
            try:
                for span in tracker.iter_runs():
                    span_dict = (
                        span.model_dump() if hasattr(span, "model_dump")
                        else span
                    )
                    if not isinstance(span_dict, dict):
                        continue
                    if span_dict.get("endTimeUnixNano") is not None:
                        continue
                    attrs = span_dict.get("attributes") or {}
                    if isinstance(attrs, dict):
                        if attrs.get("llmnb.cell_id") == cell_id:
                            return True
            except Exception:  # pragma: no cover - defensive
                pass
        return False

    def set_cell_execution_state(
        self, cell_id: str, executing: bool,
    ) -> None:
        """Test-only seam: mark a cell as currently executing.

        Used by BSP-007 §9 K95 tests
        (``test_overlay_blocked_during_execution``) when the kernel's
        actual run-tracker isn't attached. Production code drives this
        via :meth:`_is_cell_executing_for_overlay`.
        """
        with self._lock:
            record = dict(self._cells.get(cell_id, {}))
            if executing:
                record["_executing"] = True
            else:
                record.pop("_executing", None)
            self._cells[cell_id] = record
            self._dirty = True

    def _handle_apply_overlay_commit(
        self, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply one ``apply_overlay_commit`` intent (BSP-007 §4.1).

        Atomic: the applier deep-copies the writer's live cell /
        section / overlay sub-dicts, validates every operation in
        ``params['operations']`` against the work copy, and on success
        swaps the work copy back IN PLACE. On failure (any K-code) the
        work copy is dropped and the writer's live state is unchanged
        -- partial application is impossible.

        Returns ``{"ok": True, "commit_id": ..., "head_commit_id": ...}``.
        """
        operations = params.get("operations")
        if not isinstance(operations, list) or not operations:
            raise _OverlayRejected(
                "K90",
                "apply_overlay_commit: operations[] must be a non-empty list",
                details={"reason": "operations_required"},
            )
        message = params.get("message", "")
        author = params.get("author", "operator")

        with self._lock:
            state_view = self._build_overlay_state_view()
            commit_id, _record = _overlay_applier.apply_commit(
                state_view,
                operations,
                message=str(message) if message is not None else "",
                author=str(author) if author else "operator",
            )
            # On failure the applier raises; this code only runs on
            # success. Live cells / sections / overlay were mutated in
            # place by the applier's swap-back step.
            self._dirty = True
            # Invalidate any cached cell views the per-op flag mutations
            # may have invalidated.
            if hasattr(self, "_cell_view_cache"):
                self._cell_view_cache.clear()
            head_id = _overlay_applier.head_commit_id(
                self._zone_overlay_state(),
            )
        return {
            "ok": True,
            "commit_id": commit_id,
            "head_commit_id": head_id,
        }

    def _handle_revert_overlay_to_commit(
        self, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply one ``revert_overlay_to_commit`` intent (BSP-007 §4.2).

        K91 when ``commit_id`` is unknown.
        """
        commit_id = params.get("commit_id")
        if not isinstance(commit_id, str) or not commit_id:
            raise _OverlayRejected(
                "K91",
                "revert_overlay_to_commit: commit_id is required",
                details={"reason": "commit_id_required"},
            )
        with self._lock:
            state_view = self._build_overlay_state_view()
            head = _overlay_applier.revert_to_commit(state_view, commit_id)
            self._dirty = True
        return {
            "ok": True,
            "commit_id": commit_id,
            "head_commit_id": head,
        }

    def _handle_create_overlay_ref(
        self, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply one ``create_overlay_ref`` intent (BSP-007 §4.4 / V1 tag)."""
        name = params.get("name")
        commit_id = params.get("commit_id")
        with self._lock:
            state_view = self._build_overlay_state_view()
            outcome = _overlay_applier.create_ref(
                state_view, name=name, commit_id=commit_id,
            )
            self._dirty = True
        return {
            "ok": True,
            "name": outcome["name"],
            "commit_id": outcome["commit_id"],
            "created": outcome["created"],
        }

    def diff_overlay_commits(
        self, commit_a: str, commit_b: str,
    ) -> List[Dict[str, Any]]:
        """Read-only diff primitive (BSP-007 §4.3).

        Not exposed via ``submit_intent`` because it is read-only;
        consumers (History mode UI, audit trails) call this directly.
        Raises :class:`OverlayRejected` (K91) when either commit is
        unknown.
        """
        with self._lock:
            state_view = self._build_overlay_state_view()
            return _overlay_applier.diff(state_view, commit_a, commit_b)

    # -- BSP-008 §7 / S6 record_run_frame ---------------------------

    #: BSP-008 §7 RunFrame status enum. ``running`` is permitted as an
    #: intermediate state per the §8 lifecycle ("a single run produces
    #: 2+ intents over its lifetime — start with status=running, terminal
    #: with same run_id and final status"). The §7 schema documents only
    #: the three terminal values; the AgentSupervisor needs ``running``
    #: to record the start frame before the run completes.
    #: FLAGGED: the spec atom (concepts/run-frame.md) lists only the
    #: three terminals; we accept ``running`` as a permitted intermediate
    #: state because the §8 lifecycle paragraph requires it.
    _RUN_FRAME_STATUSES: FrozenSet[str] = frozenset({  # type: ignore[name-defined]
        "running", "complete", "failed", "interrupted",
    })

    def _handle_record_run_frame(
        self, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Persist one RunFrame under ``metadata.rts.zone.run_frames``.

        Per BSP-008 §7 / [concepts/run-frame](docs/atoms/concepts/run-frame.md):
        the ``record_run_frame`` intent carries a fully-formed RunFrame
        dict. The AgentSupervisor submits TWO intents per run:

        1. A start frame at run dispatch (``status: "running"``,
           ``ended_at: null``).
        2. A terminal frame at run completion (``status: "complete" |
           "failed" | "interrupted"``, ``turn_head_after`` set,
           ``ended_at`` set). Both frames carry the same ``run_id``;
           idempotency-on-``run_id`` allows update-in-place per §8.

        Validation surface (raises ``ValueError`` -> K42 / K102):

        * ``params['run_frame']`` MUST be a dict (or ``params`` itself
          MAY carry the run-frame fields directly per the
          ``record_context_manifest`` precedent — we accept both).
        * ``run_id``, ``cell_id``, ``executor_id``,
          ``context_manifest_id``, ``status``, and ``started_at`` are
          required non-empty strings.
        * ``status`` MUST be one of ``running | complete | failed |
          interrupted`` per BSP-008 §7 + the §8 lifecycle ``running``
          intermediate.
        * **K102**: a ``run_id`` that already exists with a DIFFERENT
          ``cell_id`` is rejected. Same-``cell_id`` resubmission is the
          terminal-status update path described in the atom invariants.

        Storage key: ``metadata.rts.zone.run_frames[<run_id>]``.
        RunFrames are append-only across distinct ``run_id`` values per
        the atom's invariants; the only mutation a RunFrame undergoes is
        the start->terminal status update (idempotent on ``run_id``).
        """
        # Accept both shapes: params={"run_frame": {...}} OR params={...}
        # (matches the ``record_context_manifest`` handler's accept-both
        # discipline so the AgentSupervisor builder is symmetric).
        frame = params.get("run_frame")
        if not isinstance(frame, dict):
            frame = params
        if not isinstance(frame, dict):
            raise ValueError(
                "record_run_frame: run_frame must be a dict"
            )
        run_id = frame.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(
                "record_run_frame: run_id is required"
            )
        cell_id = frame.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError(
                "record_run_frame: cell_id is required"
            )
        executor_id = frame.get("executor_id")
        if not isinstance(executor_id, str) or not executor_id:
            raise ValueError(
                "record_run_frame: executor_id is required"
            )
        context_manifest_id = frame.get("context_manifest_id")
        if not isinstance(context_manifest_id, str) or not context_manifest_id:
            raise ValueError(
                "record_run_frame: context_manifest_id is required"
            )
        status = frame.get("status")
        if not isinstance(status, str) or status not in self._RUN_FRAME_STATUSES:
            raise ValueError(
                f"record_run_frame: status must be one of "
                f"{sorted(self._RUN_FRAME_STATUSES)!r} (got {status!r})"
            )
        started_at = frame.get("started_at")
        if not isinstance(started_at, str) or not started_at:
            raise ValueError(
                "record_run_frame: started_at is required"
            )
        # ``ended_at`` MAY be present on terminal frames; per §7 it is
        # null while running. Terminal frames without ``ended_at`` are
        # accepted (the supervisor may not have populated it server-side
        # if the wall clock was unavailable) — the writer logs but does
        # not reject. Per the brief: "treat missing ended_at on terminal
        # frames as accept + log".
        ended_at = frame.get("ended_at")
        if status != "running" and (
            not isinstance(ended_at, str) or not ended_at
        ):
            logger.info(
                "record_run_frame: terminal status %s without ended_at "
                "(run_id=%s); accepting without ended_at",
                status, run_id,
                extra={
                    "event.name": "runframe_terminal_missing_ended_at",
                    "llmnb.run_id": run_id,
                    "llmnb.runframe_status": status,
                },
            )

        record = _deepcopy_json(frame)
        with self._lock:
            zone = self._zone.setdefault("run_frames", {})
            existing = zone.get(run_id)
            if existing is not None:
                existing_cell_id = existing.get("cell_id")
                if existing_cell_id != cell_id:
                    # K102: idempotency-on-run_id permits same-cell
                    # update-in-place (terminal status), but a different
                    # cell_id under the same run_id is a corruption /
                    # collision. Surface as K42 (intent_validation_failed)
                    # with the K102 marker per BSP-008 §10.
                    logger.warning(
                        "runframe_write_rejected run_id=%s reason=cell_id_mismatch "
                        "existing_cell=%s submitted_cell=%s",
                        run_id, existing_cell_id, cell_id,
                        extra={
                            "event.name": "runframe_write_rejected",
                            "llmnb.run_id": run_id,
                            "llmnb.k_class": "K102",
                            "llmnb.runframe_reason": "cell_id_mismatch",
                        },
                    )
                    raise ValueError(
                        f"K102: record_run_frame: run_id {run_id!r} already "
                        f"exists with cell_id {existing_cell_id!r}; refusing "
                        f"to rebind to {cell_id!r}"
                    )
            zone[run_id] = record
            self._dirty = True
        return {
            "ok": True,
            "run_id": run_id,
            "cell_id": cell_id,
            "status": status,
        }

    # -- V1 Kernel Gap Closure G13 -- intent-queue overflow ----------

    #: The single logical "intents" queue name (used as the key in
    #: ``metadata.rts.queues``).  V1 has only one such queue; the
    #: ``queues[*]`` indirection in the schema reserves room for V2+
    #: per-zone queues.
    _INTENT_QUEUE_NAME: str = "intents"

    def enqueue_intent(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """Buffer an intent envelope for deferred application.

        Unlike :meth:`submit_intent` (which applies synchronously),
        this method appends the envelope to the in-memory pending
        queue.  Callers drain via :meth:`flush_pending_intents`
        (typically on the next dispatcher round-trip / autosave tick).

        Per RFC-005 §F13 the queue is bounded; on overflow
        (``len(buffer) > intent_queue_overflow_threshold``) the writer
        spills the buffered intents to a JSON-line file under
        ``<workspace_root>/.llmnb-intent-queue/`` and records a marker
        at ``metadata.rts.queues[<queue>].overflow``.  Subsequent
        :meth:`flush_pending_intents` calls drain the disk spill IN
        ORDER before any in-memory entries.

        Returns a small status dict: ``{"buffered": True,
        "overflow": bool, "buffer_size": int}``.
        """
        if not isinstance(envelope, dict):
            raise ValueError("enqueue_intent: envelope must be a dict")
        marker_to_log: Optional[Tuple[int, int, str]] = None
        with self._lock:
            self._pending_intents.append(envelope)
            buffer_size = len(self._pending_intents)
            overflow = buffer_size > self._intent_queue_overflow_threshold
            if overflow:
                checkpoint_id, disk_path = self._spill_intent_buffer_locked()
                marker_to_log = (
                    self._intent_overflow_count, buffer_size, checkpoint_id,
                )
                # Buffer is now empty (spill_intent_buffer_locked
                # drained it into the disk file).
                buffer_size = 0
        # Log AFTER releasing the lock per Engineering Guide §11.7.
        if marker_to_log is not None:
            count, spilled_size, checkpoint_id = marker_to_log
            logger.warning(
                "intent queue overflow: spilled=%d threshold=%d "
                "checkpoint=%s; spill-to-disk per RFC-005 §F13",
                spilled_size, self._intent_queue_overflow_threshold,
                checkpoint_id,
                extra={
                    "event.name": "metadata.intent_queue_overflow",
                    "llmnb.intent_overflow_count": count,
                    "llmnb.spilled_intent_count": spilled_size,
                    "llmnb.checkpoint_id": checkpoint_id,
                },
            )
        return {
            "buffered": True,
            "overflow": marker_to_log is not None,
            "buffer_size": buffer_size,
        }

    def _intent_overflow_dir(self) -> Path:
        """Return the directory the intent-queue overflow spills land in."""
        return self._workspace_root / ".llmnb-intent-queue"

    def _spill_intent_buffer_locked(self) -> Tuple[str, str]:
        """Spill ``self._pending_intents`` to disk and record the marker.

        Caller MUST hold ``self._lock``.  Returns ``(checkpoint_id,
        disk_path)`` for the marker.

        On-disk format: one JSON object per line, each line a complete
        intent envelope.  The file lives at
        ``<workspace_root>/.llmnb-intent-queue/<checkpoint_id>.jsonl``.
        Atomic write via ``<file>.tmp + os.replace``.

        After spill, the in-memory buffer is emptied; the marker
        persists in ``metadata.rts.queues[intents].overflow`` until the
        spill is fully drained (see :meth:`flush_pending_intents`).
        """
        self._intent_overflow_count += 1
        # Compose a checkpoint ID that's both unique and deterministic-
        # enough to debug.  ``ovf-<session>-<count>-<timestamp>``.
        ts = _utc_now_iso().replace(":", "").replace(".", "").replace("-", "")
        checkpoint_id = (
            f"ovf-{self._session_id[:8]}-{self._intent_overflow_count:06d}-{ts}"
        )
        target_dir = self._intent_overflow_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{checkpoint_id}.jsonl"
        tmp_path = target_path.with_name(target_path.name + ".tmp")
        # Serialize each envelope as a single line.
        lines: List[str] = []
        for env in self._pending_intents:
            lines.append(json.dumps(env, ensure_ascii=False))
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp_path, target_path)
        # Record / merge the queue overflow marker.  We store the disk
        # path RELATIVE to the workspace root so the marker is
        # workspace-portable (the operator may move the workspace; the
        # absolute path would not).
        try:
            disk_rel = target_path.relative_to(self._workspace_root).as_posix()
        except ValueError:  # pragma: no cover - workspace mismatch
            disk_rel = str(target_path)
        queue_record = self._queues.setdefault(
            self._INTENT_QUEUE_NAME,
            {"version": 1, "spills": []},
        )
        spill_entry = {
            "checkpoint_id": checkpoint_id,
            "disk_path": disk_rel,
            "spilled_at": _utc_now_iso(),
            "intent_count": len(self._pending_intents),
        }
        queue_record.setdefault("spills", []).append(spill_entry)
        # Per the brief: store the *current* overflow marker as a
        # single-object slot at queues[<queue>].overflow.  When the
        # spill is drained the slot is cleared but the spill row stays
        # in ``spills[]`` for audit.
        queue_record["overflow"] = {
            "checkpoint_id": checkpoint_id,
            "disk_path": disk_rel,
        }
        # Drain the in-memory buffer.
        self._pending_intents = []
        self._dirty = True
        return checkpoint_id, disk_rel

    def flush_pending_intents(self) -> List[Dict[str, Any]]:
        """Drain disk-spilled intents (in order), then in-memory buffer.

        Per the V1 Kernel Gap Closure G13 brief: "On next successful
        flush, drain disk-spilled intents in order, then resume in-
        memory queueing."  We honour that order:

        1. For every overflow marker on this queue (oldest first),
           read the JSONL file and ``submit_intent`` each line.
        2. After all spill files are drained, clear the
           ``queues[<queue>].overflow`` marker (the spill rows in
           ``spills[]`` stay for audit).
        3. Drain the in-memory ``_pending_intents`` buffer through
           ``submit_intent`` in arrival order.

        Returns the list of per-intent results (the same dict shape
        :meth:`submit_intent` returns).  Failures of individual
        intents (K40/K42/K43) do NOT abort the flush; they appear in
        the result list with their error codes.
        """
        results: List[Dict[str, Any]] = []
        # Step 1: drain disk spills.
        with self._lock:
            queue_record = self._queues.get(self._INTENT_QUEUE_NAME)
            spills_to_drain: List[Dict[str, Any]] = []
            if queue_record is not None:
                spills_to_drain = list(queue_record.get("spills", []))
        # The drain itself MUST happen outside the lock because each
        # submit_intent acquires the FIFO gate.
        drained_paths: List[str] = []
        for spill in spills_to_drain:
            disk_path = spill.get("disk_path")
            if not isinstance(disk_path, str) or not disk_path:
                continue
            spill_file = self._resolve_spill_path(disk_path)
            if not spill_file.exists():
                # Already drained on a prior flush; skip but remember
                # for marker cleanup.
                drained_paths.append(disk_path)
                continue
            try:
                contents = spill_file.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover - defensive
                logger.exception(
                    "flush_pending_intents: failed reading spill %s",
                    spill_file,
                )
                continue
            for line in contents.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    env = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "flush_pending_intents: malformed spill line skipped"
                    )
                    continue
                result = self.submit_intent(env)
                results.append(result)
            # Remove the on-disk file once successfully replayed.
            try:
                spill_file.unlink()
            except OSError:  # pragma: no cover - defensive
                pass
            drained_paths.append(disk_path)
        # Step 2: clear marker / spill rows for fully-drained files.
        if drained_paths:
            with self._lock:
                queue_record = self._queues.get(self._INTENT_QUEUE_NAME)
                if queue_record is not None:
                    spills = queue_record.get("spills", [])
                    queue_record["spills"] = [
                        s for s in spills
                        if s.get("disk_path") not in drained_paths
                    ]
                    if not queue_record["spills"]:
                        # All spills drained: clear the active overflow
                        # marker so the next enqueue resumes in-memory.
                        queue_record.pop("overflow", None)
                    self._dirty = True
        # Step 3: drain in-memory buffer.
        with self._lock:
            in_memory = list(self._pending_intents)
            self._pending_intents = []
        for env in in_memory:
            results.append(self.submit_intent(env))
        return results

    def _resolve_spill_path(self, disk_path: str) -> Path:
        """Resolve a disk_path stored in the queue marker to an absolute Path.

        Markers store paths *relative* to ``workspace_root`` so they
        survive workspace moves; absolute paths are also tolerated for
        manual debug spills.
        """
        candidate = Path(disk_path)
        if candidate.is_absolute():
            return candidate
        return self._workspace_root / candidate

    def get_queue_overflow_marker(
        self, queue_name: str = "intents",
    ) -> Optional[Dict[str, Any]]:
        """Return the active overflow marker for ``queue_name`` or None."""
        with self._lock:
            queue_record = self._queues.get(queue_name)
            if queue_record is None:
                return None
            marker = queue_record.get("overflow")
            return dict(marker) if isinstance(marker, dict) else None

    # -- Family B (layout) state machine -----------------------------

    def apply_layout_edit(
        self, operation: str, parameters: Dict[str, Any],
    ) -> int:
        """Apply one ``layout.edit`` operation to the in-memory tree.

        ``operation`` ∈ ``add_zone | remove_node | move_node |
        rename_node | update_render_hints`` per RFC-006 §"Family B".
        On success the writer's ``_layout`` is mutated in place,
        ``_snapshot_version`` is incremented, ``_dirty`` is set, and
        the new ``_snapshot_version`` is returned.

        On invalid operation (unknown op, missing required parameters,
        target node not found, duplicate ID) the call leaves state
        unchanged and returns the CURRENT (un-incremented)
        ``_snapshot_version``.  Per the K-CM brief: this method does
        NOT raise; the dispatcher's RFC-006 W4 fail-closed behavior
        relies on the no-op path being silent at this level.
        """
        if not isinstance(parameters, dict):
            parameters = {}
        with self._lock:
            tree = self._layout.get("tree")
            if not isinstance(tree, dict):
                return self._snapshot_version
            ok = False
            try:
                if operation == "add_zone":
                    ok = self._layout_add_zone(tree, parameters)
                elif operation == "remove_node":
                    ok = self._layout_remove_node(tree, parameters)
                elif operation == "move_node":
                    ok = self._layout_move_node(tree, parameters)
                elif operation == "rename_node":
                    ok = self._layout_rename_node(tree, parameters)
                elif operation == "update_render_hints":
                    ok = self._layout_update_render_hints(tree, parameters)
                else:
                    ok = False
            except Exception:  # pragma: no cover - defensive
                ok = False
            if ok:
                self._snapshot_version += 1
                self._dirty = True
            return self._snapshot_version

    def emit_layout_update(self) -> Dict[str, Any]:
        """Return the ``layout.update`` payload per RFC-006 §"Family B".

        Shape: ``{"snapshot_version": int, "tree": <tree>}`` -- a
        DEEP COPY of the current layout tree so callers can mutate
        the returned object without touching writer state.
        """
        with self._lock:
            return {
                "snapshot_version": self._snapshot_version,
                "tree": _deepcopy_json(self._layout.get("tree", {})),
            }

    # -- Layout helpers (called under self._lock) --------------------

    @staticmethod
    def _layout_collect_ids(node: Dict[str, Any]) -> List[str]:
        """Return all node IDs in the subtree rooted at ``node``."""
        out: List[str] = []
        stack: List[Dict[str, Any]] = [node]
        while stack:
            cur = stack.pop()
            nid = cur.get("id")
            if isinstance(nid, str):
                out.append(nid)
            for child in cur.get("children", []) or []:
                if isinstance(child, dict):
                    stack.append(child)
        return out

    @staticmethod
    def _layout_find(
        node: Dict[str, Any], target_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the node with ``id == target_id`` or ``None``."""
        if node.get("id") == target_id:
            return node
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                hit = MetadataWriter._layout_find(child, target_id)
                if hit is not None:
                    return hit
        return None

    @staticmethod
    def _layout_find_parent(
        node: Dict[str, Any], target_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the parent of the node with ``id == target_id``."""
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                if child.get("id") == target_id:
                    return node
                hit = MetadataWriter._layout_find_parent(child, target_id)
                if hit is not None:
                    return hit
        return None

    def _layout_add_zone(
        self, tree: Dict[str, Any], params: Dict[str, Any],
    ) -> bool:
        """Add a zone (or any non-root node) under a named parent."""
        spec = params.get("node_spec")
        parent_id = params.get("new_parent_id") or params.get("parent_id")
        if not isinstance(spec, dict):
            return False
        node_id = spec.get("id")
        if not isinstance(node_id, str) or not node_id:
            return False
        # ID uniqueness across the tree.
        if node_id in self._layout_collect_ids(tree):
            return False
        new_node: Dict[str, Any] = {
            "id": node_id,
            "type": spec.get("type", "zone"),
            "render_hints": dict(spec.get("render_hints", {})),
            "children": list(spec.get("children", [])),
        }
        if parent_id is None:
            parent = tree
        else:
            parent = self._layout_find(tree, parent_id)
            if parent is None:
                return False
        children = parent.setdefault("children", [])
        if not isinstance(children, list):
            return False
        children.append(new_node)
        return True

    def _layout_remove_node(
        self, tree: Dict[str, Any], params: Dict[str, Any],
    ) -> bool:
        node_id = params.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            return False
        if node_id == tree.get("id"):
            return False  # cannot remove the root
        parent = self._layout_find_parent(tree, node_id)
        if parent is None:
            return False
        children = parent.get("children", [])
        if not isinstance(children, list):
            return False
        new_children = [
            c for c in children
            if not (isinstance(c, dict) and c.get("id") == node_id)
        ]
        if len(new_children) == len(children):
            return False
        parent["children"] = new_children
        return True

    def _layout_move_node(
        self, tree: Dict[str, Any], params: Dict[str, Any],
    ) -> bool:
        node_id = params.get("node_id")
        new_parent_id = params.get("new_parent_id")
        if not isinstance(node_id, str) or not isinstance(new_parent_id, str):
            return False
        if node_id == tree.get("id"):
            return False  # cannot move root
        if node_id == new_parent_id:
            return False  # cannot parent self
        node = self._layout_find(tree, node_id)
        if node is None:
            return False
        new_parent = self._layout_find(tree, new_parent_id)
        if new_parent is None:
            return False
        # Descendant check: cannot move a node under one of its own
        # descendants without breaking the tree invariant.
        if MetadataWriter._layout_find(node, new_parent_id) is not None:
            return False
        old_parent = self._layout_find_parent(tree, node_id)
        if old_parent is None:
            return False
        old_children = old_parent.get("children", [])
        if not isinstance(old_children, list):
            return False
        old_parent["children"] = [
            c for c in old_children
            if not (isinstance(c, dict) and c.get("id") == node_id)
        ]
        new_children = new_parent.setdefault("children", [])
        if not isinstance(new_children, list):
            return False
        new_children.append(node)
        return True

    def _layout_rename_node(
        self, tree: Dict[str, Any], params: Dict[str, Any],
    ) -> bool:
        node_id = params.get("node_id")
        new_name = params.get("new_name") or params.get("new_id")
        if not isinstance(node_id, str) or not isinstance(new_name, str):
            return False
        if not new_name:
            return False
        if new_name == node_id:
            return True  # no-op
        node = self._layout_find(tree, node_id)
        if node is None:
            return False
        if new_name in self._layout_collect_ids(tree):
            return False
        node["id"] = new_name
        return True

    def _layout_update_render_hints(
        self, tree: Dict[str, Any], params: Dict[str, Any],
    ) -> bool:
        node_id = params.get("node_id")
        hints = params.get("render_hints")
        if not isinstance(node_id, str) or not isinstance(hints, dict):
            return False
        node = self._layout_find(tree, node_id)
        if node is None:
            return False
        existing = node.get("render_hints")
        if not isinstance(existing, dict):
            existing = {}
        existing.update(hints)
        node["render_hints"] = existing
        return True

    # -- Family C (agent graph) state machine ------------------------

    def apply_agent_graph_command(
        self, command: str, parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply one Family C command and return the response payload.

        ``command`` enumerates BOTH the RFC-006 §"Family C" query types
        (``neighbors | paths | subgraph | full_snapshot``) AND a minimal
        CRUD set the kernel needs internally to maintain the graph
        (``upsert_node | remove_node | upsert_edge | remove_edge``).
        See the slice's ambiguity flag in the report -- RFC-006 is
        silent on the mutation set; this is the recommended minimal
        union and may be tightened by an RFC-006 erratum.

        Returned payload shape:

        * Mutation commands -- ``{"ok": bool, "snapshot_version": int}``.
        * ``neighbors`` / ``paths`` / ``subgraph`` -- ``{"nodes": [...],
          "edges": [...], "truncated": bool}``.
        * ``full_snapshot`` -- ``{"nodes": [...], "edges": [...],
          "truncated": False}`` containing the entire graph.

        Unknown commands return ``{"ok": false, "snapshot_version": <current>}``
        (no exception per the dispatcher contract).
        """
        if not isinstance(parameters, dict):
            parameters = {}
        with self._lock:
            mutators = {
                "upsert_node": self._graph_upsert_node,
                "remove_node": self._graph_remove_node,
                "upsert_edge": self._graph_upsert_edge,
                "remove_edge": self._graph_remove_edge,
            }
            if command in mutators:
                ok = False
                try:
                    ok = mutators[command](parameters)
                except Exception:  # pragma: no cover - defensive
                    ok = False
                if ok:
                    self._snapshot_version += 1
                    self._dirty = True
                return {
                    "ok": bool(ok),
                    "snapshot_version": self._snapshot_version,
                }
            if command == "full_snapshot":
                return {
                    "nodes": _deepcopy_json(
                        self._agent_graph.get("nodes", []),
                    ),
                    "edges": _deepcopy_json(
                        self._agent_graph.get("edges", []),
                    ),
                    "truncated": False,
                }
            if command == "neighbors":
                return self._graph_neighbors(parameters)
            if command == "paths":
                return self._graph_paths(parameters)
            if command == "subgraph":
                return self._graph_subgraph(parameters)
            # Unknown -- mutation-style response with ok=False so the
            # caller can detect the no-op.
            return {
                "ok": False,
                "snapshot_version": self._snapshot_version,
            }

    # -- Agent-graph helpers (called under self._lock) ---------------

    def _graph_upsert_node(self, params: Dict[str, Any]) -> bool:
        node = params.get("node") if isinstance(params.get("node"), dict) else params
        nid = node.get("id") if isinstance(node, dict) else None
        if not isinstance(nid, str) or not nid:
            return False
        ntype = node.get("type")
        if not isinstance(ntype, str):
            return False
        properties = node.get("properties") or {}
        if not isinstance(properties, dict):
            properties = {}
        nodes = self._agent_graph.setdefault("nodes", [])
        for existing in nodes:
            if isinstance(existing, dict) and existing.get("id") == nid:
                existing["type"] = ntype
                existing["properties"] = dict(properties)
                return True
        nodes.append({"id": nid, "type": ntype, "properties": dict(properties)})
        return True

    def _graph_remove_node(self, params: Dict[str, Any]) -> bool:
        nid = params.get("node_id") or params.get("id")
        if not isinstance(nid, str) or not nid:
            return False
        nodes = self._agent_graph.get("nodes", [])
        if not any(isinstance(n, dict) and n.get("id") == nid for n in nodes):
            return False
        self._agent_graph["nodes"] = [
            n for n in nodes
            if not (isinstance(n, dict) and n.get("id") == nid)
        ]
        # Remove edges incident to the removed node.
        edges = self._agent_graph.get("edges", [])
        self._agent_graph["edges"] = [
            e for e in edges
            if not (
                isinstance(e, dict)
                and (e.get("source") == nid or e.get("target") == nid)
            )
        ]
        return True

    def _graph_upsert_edge(self, params: Dict[str, Any]) -> bool:
        edge = params.get("edge") if isinstance(params.get("edge"), dict) else params
        if not isinstance(edge, dict):
            return False
        source = edge.get("source")
        target = edge.get("target")
        kind = edge.get("kind")
        if not (
            isinstance(source, str) and isinstance(target, str)
            and isinstance(kind, str) and source and target and kind
        ):
            return False
        # RFC-005 §`metadata.rts.agents`: edge endpoints MUST exist
        # in nodes[].
        node_ids = {
            n.get("id") for n in self._agent_graph.get("nodes", [])
            if isinstance(n, dict)
        }
        if source not in node_ids or target not in node_ids:
            return False
        properties = edge.get("properties") or {}
        if not isinstance(properties, dict):
            properties = {}
        edges = self._agent_graph.setdefault("edges", [])
        for existing in edges:
            if (
                isinstance(existing, dict)
                and existing.get("source") == source
                and existing.get("target") == target
                and existing.get("kind") == kind
            ):
                existing["properties"] = dict(properties)
                return True
        edges.append({
            "source": source, "target": target,
            "kind": kind, "properties": dict(properties),
        })
        return True

    def _graph_remove_edge(self, params: Dict[str, Any]) -> bool:
        source = params.get("source")
        target = params.get("target")
        kind = params.get("kind")
        if not (
            isinstance(source, str) and isinstance(target, str)
            and isinstance(kind, str)
        ):
            return False
        edges = self._agent_graph.get("edges", [])
        new_edges = [
            e for e in edges
            if not (
                isinstance(e, dict)
                and e.get("source") == source
                and e.get("target") == target
                and e.get("kind") == kind
            )
        ]
        if len(new_edges) == len(edges):
            return False
        self._agent_graph["edges"] = new_edges
        return True

    def _graph_neighbors(self, params: Dict[str, Any]) -> Dict[str, Any]:
        nid = params.get("node_id")
        if not isinstance(nid, str) or not nid:
            return {"nodes": [], "edges": [], "truncated": False}
        try:
            hops = int(params.get("hops", 1) or 1)
        except (TypeError, ValueError):
            hops = 1
        hops = max(1, min(hops, 16))
        edge_filters = params.get("edge_filters")
        if isinstance(edge_filters, list):
            allowed_kinds = {k for k in edge_filters if isinstance(k, str)}
        else:
            allowed_kinds = None
        nodes_by_id = {
            n.get("id"): n for n in self._agent_graph.get("nodes", [])
            if isinstance(n, dict)
        }
        edges = [
            e for e in self._agent_graph.get("edges", [])
            if isinstance(e, dict)
        ]
        if allowed_kinds is not None:
            edges = [e for e in edges if e.get("kind") in allowed_kinds]
        visited = {nid}
        frontier = {nid}
        included_edges: List[Dict[str, Any]] = []
        for _ in range(hops):
            next_frontier: set = set()
            for edge in edges:
                src = edge.get("source")
                dst = edge.get("target")
                if src in frontier and dst not in visited:
                    next_frontier.add(dst)
                    included_edges.append(edge)
                elif dst in frontier and src not in visited:
                    next_frontier.add(src)
                    included_edges.append(edge)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        result_nodes = [
            nodes_by_id[n] for n in visited if n in nodes_by_id
        ]
        return {
            "nodes": _deepcopy_json(result_nodes),
            "edges": _deepcopy_json(included_edges),
            "truncated": False,
        }

    def _graph_paths(self, params: Dict[str, Any]) -> Dict[str, Any]:
        source = params.get("node_id") or params.get("source")
        target = params.get("target_node_id") or params.get("target")
        if not (isinstance(source, str) and isinstance(target, str)):
            return {"nodes": [], "edges": [], "truncated": False}
        try:
            hops = int(params.get("hops", 4) or 4)
        except (TypeError, ValueError):
            hops = 4
        hops = max(1, min(hops, 16))
        # Simple BFS from source to target up to ``hops`` edges, return
        # the union of nodes and edges on the discovered path (one
        # shortest path).
        edges = [
            e for e in self._agent_graph.get("edges", [])
            if isinstance(e, dict)
        ]
        adj: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for e in edges:
            s = e.get("source")
            t = e.get("target")
            if isinstance(s, str) and isinstance(t, str):
                adj.setdefault(s, []).append((t, e))
                adj.setdefault(t, []).append((s, e))  # undirected for V1
        prev: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        seen = {source}
        frontier = [source]
        depth = 0
        found = source == target
        while frontier and depth < hops and not found:
            next_frontier: List[str] = []
            for cur in frontier:
                for nbr, edge in adj.get(cur, []):
                    if nbr in seen:
                        continue
                    seen.add(nbr)
                    prev[nbr] = (cur, edge)
                    if nbr == target:
                        found = True
                        break
                    next_frontier.append(nbr)
                if found:
                    break
            frontier = next_frontier
            depth += 1
        if not found:
            return {"nodes": [], "edges": [], "truncated": False}
        # Reconstruct path.
        path_node_ids: List[str] = [target]
        path_edges: List[Dict[str, Any]] = []
        cur = target
        while cur != source:
            parent, edge = prev[cur]
            path_node_ids.append(parent)
            path_edges.append(edge)
            cur = parent
        nodes_by_id = {
            n.get("id"): n for n in self._agent_graph.get("nodes", [])
            if isinstance(n, dict)
        }
        path_nodes = [
            nodes_by_id[n] for n in path_node_ids if n in nodes_by_id
        ]
        return {
            "nodes": _deepcopy_json(path_nodes),
            "edges": _deepcopy_json(path_edges),
            "truncated": False,
        }

    def _graph_subgraph(self, params: Dict[str, Any]) -> Dict[str, Any]:
        ids = params.get("node_ids")
        if not isinstance(ids, list):
            return {"nodes": [], "edges": [], "truncated": False}
        wanted = {n for n in ids if isinstance(n, str)}
        nodes = [
            n for n in self._agent_graph.get("nodes", [])
            if isinstance(n, dict) and n.get("id") in wanted
        ]
        edges = [
            e for e in self._agent_graph.get("edges", [])
            if isinstance(e, dict)
            and e.get("source") in wanted and e.get("target") in wanted
        ]
        return {
            "nodes": _deepcopy_json(nodes),
            "edges": _deepcopy_json(edges),
            "truncated": False,
        }

    # -- Hydration ---------------------------------------------------

    def hydrate(self, snapshot: Dict[str, Any]) -> None:
        """Reset in-memory state from a persisted ``metadata.rts`` snapshot.

        Idempotent: hydrating with the same snapshot twice leaves the
        writer in the same observable state (``snapshot_version``,
        ``_layout``, ``_agent_graph``, etc. all match).

        Steps:

        1. Validate ``schema_version`` major equals RFC-005 v1
           (``"1"``).  Mismatch raises :class:`ValueError`.
        2. Validate the inner ``config`` block contains no forbidden
           secret fields (RFC-005 §F2).  Forbidden field raises
           :class:`ValueError("forbidden secret in config")`.
        3. Replace ``_layout``, ``_agent_graph``, ``_extra_runs``,
           ``_blobs``, ``_drift_log``, ``_config``, and
           ``_snapshot_version`` with values from the snapshot.

        ``snapshot_version`` semantics: the writer stores the
        persisted value verbatim.  The next call to
        :meth:`_build_snapshot` (via :meth:`snapshot`,
        :meth:`apply_layout_edit` follow-up emission, etc.) will
        increment so the kernel emits the next version (per RFC-006
        §"hydrate request/response semantics": "the kernel resumes its
        counter from this value + 1").  Writers MUST NOT decrement
        ``snapshot_version`` -- this method is the only legitimate way
        to assign a smaller value, and it does so only at session
        start.

        This method does NOT call :class:`DriftDetector.compare` (the
        K-CM dispatcher does that against current-environment values
        K-MW does not have access to) and does NOT respawn agents
        (K-AS / K-CM do that via ``AgentSupervisor.respawn_from_config``).
        """
        if not isinstance(snapshot, dict):
            raise ValueError("hydrate: snapshot must be a dict")
        # Validate schema_version major matches.
        sv = snapshot.get("schema_version", "")
        if not isinstance(sv, str):
            raise ValueError(
                "hydrate: schema_version must be a string"
            )
        major = sv.split(".", 1)[0] if sv else ""
        expected_major = SCHEMA_VERSION.split(".", 1)[0]
        if major != expected_major:
            raise ValueError(
                f"hydrate: schema_version major {major!r} does not match "
                f"this writer's RFC-005 major {expected_major!r}"
            )
        # Validate forbidden-secret fields in config.  We use a
        # ValueError with the brief-mandated message rather than the
        # in-process ``SecretRejected`` because the brief locks the
        # exception type to ``ValueError("forbidden secret in config")``.
        config = snapshot.get("config")
        if isinstance(config, dict):
            try:
                reject_secrets(config, path="config")
            except SecretRejected as exc:
                raise ValueError("forbidden secret in config") from exc
        # Apply.  All state under the writer's RLock so a concurrent
        # ``snapshot()`` does not see a partial hydrate.
        with self._lock:
            layout = snapshot.get("layout")
            if isinstance(layout, dict):
                self._layout = _deepcopy_json(layout)
            else:
                self._layout = {
                    "version": 1,
                    "tree": {
                        "id": "root", "type": "workspace",
                        "render_hints": {}, "children": [],
                    },
                }
            # RFC-005's snapshot key is "agents"; the writer's
            # in-memory name is _agent_graph (per the K-MW slice
            # rename for clarity against the brief).
            agents = snapshot.get("agents")
            if isinstance(agents, dict):
                self._agent_graph = _deepcopy_json(agents)
            else:
                self._agent_graph = {
                    "version": 1, "nodes": [], "edges": [],
                }
            event_log = snapshot.get("event_log") or {}
            runs = event_log.get("runs") if isinstance(event_log, dict) else None
            if isinstance(runs, list):
                self._extra_runs = [_deepcopy_json(r) for r in runs]
            else:
                self._extra_runs = []
            blobs = snapshot.get("blobs")
            if isinstance(blobs, dict):
                self._blobs = _deepcopy_json(blobs)
            else:
                self._blobs = {}
            drift_log = snapshot.get("drift_log")
            if isinstance(drift_log, list):
                self._drift_log = [_deepcopy_json(d) for d in drift_log]
            else:
                self._drift_log = []
            cfg = snapshot.get("config")
            if isinstance(cfg, dict):
                self._config = _deepcopy_json(cfg)
            persisted_version = snapshot.get("snapshot_version", 0)
            try:
                self._snapshot_version = int(persisted_version)
            except (TypeError, ValueError):
                self._snapshot_version = 0
            persisted_session = snapshot.get("session_id")
            if isinstance(persisted_session, str) and persisted_session:
                self._session_id = persisted_session
            # Reset BSP-003 §10 intent dispatcher state.  Re-hydrating
            # is idempotent: a previously-applied intent_id appears as
            # an unseen ID after hydrate (the registry tracks
            # in-process state only).  This is acceptable because the
            # idempotency set is for replay-within-a-session, not
            # cross-session de-duplication; the latter is V3 work.
            event_log_dict = snapshot.get("event_log") or {}
            persisted_intents = (
                event_log_dict.get("intent_log")
                if isinstance(event_log_dict, dict) else None
            )
            if isinstance(persisted_intents, list):
                self._intent_log = [
                    _deepcopy_json(e) for e in persisted_intents
                ]
                # Rebuild the idempotency set so a freshly-hydrated
                # writer rejects re-submissions of in-snapshot intents.
                self._intent_applied = {
                    e.get("intent_id"): e.get("snapshot_version", 0)
                    for e in self._intent_log
                    if isinstance(e, dict) and isinstance(e.get("intent_id"), str)
                }
            else:
                self._intent_log = []
                self._intent_applied = {}
            persisted_created_at = snapshot.get("created_at")
            if isinstance(persisted_created_at, str) and persisted_created_at:
                self._created_at = persisted_created_at
            # BSP-005 §6.1 / S0.5 -- hydrate the cells map.  Pre-S0.5
            # snapshots carry no ``cells`` key, or carry cells without
            # a ``kind`` field; we default to ``agent`` and mark the
            # record so the next snapshot writes the resolved value
            # back persistently (PLAN-S0.5 §3 step 3).
            cells_raw = snapshot.get("cells")
            self._cells = {}
            back_filled_any = False
            if isinstance(cells_raw, dict):
                for cell_id, record in cells_raw.items():
                    if not isinstance(cell_id, str) or not cell_id:
                        continue
                    if not isinstance(record, dict):
                        continue
                    record_copy = _deepcopy_json(record)
                    if (
                        "kind" not in record_copy
                        or not isinstance(record_copy.get("kind"), str)
                        or record_copy.get("kind") not in CELL_KINDS
                    ):
                        record_copy["kind"] = DEFAULT_CELL_KIND
                        record_copy["_kind_back_filled"] = True
                        back_filled_any = True
                    self._cells[cell_id] = record_copy
            # V1 Kernel Gap Closure G13 -- hydrate the queues marker
            # so a session that crashed mid-overflow knows there's a
            # disk spill awaiting drain.  ``flush_pending_intents`` on
            # the next dispatcher round-trip will replay the file.
            queues_raw = snapshot.get("queues")
            if isinstance(queues_raw, dict):
                self._queues = _deepcopy_json(queues_raw)
            else:
                self._queues = {}
            # BSP-008 §3 / S3.5 -- hydrate the zone substructure. Pre-S3.5
            # snapshots carry no ``zone`` key; we default to an empty dict
            # and the next ``record_context_manifest`` populates it.
            zone_raw = snapshot.get("zone")
            if isinstance(zone_raw, dict):
                self._zone = _deepcopy_json(zone_raw)
            else:
                self._zone = {}
            self._pending_intents = []
            # Mark dirty iff we back-filled at least one cell.kind so
            # the next snapshot writes the resolved values out.
            self._dirty = back_filled_any

    # -- Snapshot triggers -------------------------------------------

    def snapshot(self, trigger: str = "save") -> Dict[str, Any]:
        """Build, validate, and emit one ``notebook.metadata`` snapshot.

        ``trigger`` is one of ``save | shutdown | timer | end_of_run``
        per RFC-005 §"Snapshot triggers"; receivers MUST tolerate
        unknown values.  Returns the inner ``metadata.rts`` snapshot
        dict (for tests; the wire envelope wraps it).
        """
        snapshot_dict = self._build_snapshot()
        envelope = self._build_envelope(snapshot_dict, trigger=trigger)
        self._last_emitted_envelope = envelope
        if self._dispatcher is not None:
            try:
                self._dispatcher.emit(envelope)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "metadata writer: dispatcher.emit raised; envelope dropped"
                )
        with self._lock:
            self._dirty = False
        return snapshot_dict

    def take_last_envelope(self) -> Optional[Dict[str, Any]]:
        """Return the most recent emitted envelope (for tests / smokes)."""
        return self._last_emitted_envelope

    def start(self) -> None:
        """Start the 30-second autosave timer thread.  Idempotent."""
        with self._lock:
            if self._timer_thread is not None and self._timer_thread.is_alive():
                return
            self._stop_event.clear()
            self._timer_thread = threading.Thread(
                target=self._timer_loop,
                name="llmnb.metadata-writer.timer",
                daemon=True,
            )
            self._timer_thread.start()

    def stop(self, *, emit_final: bool = True) -> None:
        """Stop the autosave timer; OPTIONALLY emit a shutdown snapshot.

        Per RFC-005 §"Snapshot triggers" case 2 the kernel emits one
        last snapshot on clean shutdown so the operator never loses
        more than the timer cadence.  ``emit_final`` exists for tests
        that want to stop without emitting.
        """
        self._stop_event.set()
        thread = self._timer_thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._timer_thread = None
        if emit_final and self._dirty:
            try:
                self.snapshot(trigger="shutdown")
            except Exception:  # pragma: no cover - defensive
                logger.exception("metadata writer: final snapshot raised")

    def _timer_loop(self) -> None:
        """Periodic-timer trigger thread.  Emits while dirty."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self._autosave_interval_sec)
            if self._stop_event.is_set():
                return
            if self._dirty:
                try:
                    self.snapshot(trigger="timer")
                except Exception:  # pragma: no cover - defensive
                    logger.exception("metadata writer: timer snapshot raised")

    # -- Sink interface ----------------------------------------------

    def on_run_closed(self, span: Dict[str, Any]) -> None:
        """Sink entry point: a run-tracker's closed-span notification.

        Records the closed span and emits a snapshot per RFC-005
        §"Snapshot triggers" case 4 (end_of_run).  Open spans (those
        with ``endTimeUnixNano: null``) trigger no emission to avoid
        flooding the wire on every event.
        """
        end_time = span.get("endTimeUnixNano")
        self.record_run(span)
        if end_time is not None:
            try:
                self.snapshot(trigger="end_of_run")
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "metadata writer: end_of_run snapshot raised",
                )

    # -- Snapshot building -------------------------------------------

    def _build_snapshot(self) -> Dict[str, Any]:
        """Produce one full ``metadata.rts`` snapshot.

        Walks the in-memory state, runs the secret-rejection pass,
        runs the blob-extraction pass on the merged event log, and
        returns the outermost ``metadata.rts`` dict (the wire shape
        wraps this under ``payload.snapshot``).
        """
        with self._lock:
            # Validate config BEFORE serializing any of it.  A failure
            # here MUST raise; we never persist a snapshot that contains
            # forbidden fields.
            reject_secrets(self._config, path="config")

            # Drain runs from the run-tracker (if any) and merge with
            # external recordings.  Order: run-tracker first
            # (insertion order), then extras in arrival order.  Spans
            # with the same spanId from the tracker take precedence.
            seen: Dict[str, Dict[str, Any]] = {}
            order: List[str] = []
            if self._run_tracker is not None:
                for span in self._run_tracker.iter_runs():
                    obj = span.model_dump()
                    sid = obj.get("spanId")
                    if isinstance(sid, str) and sid not in seen:
                        seen[sid] = obj
                        order.append(sid)
            for span in self._extra_runs:
                sid = span.get("spanId")
                if isinstance(sid, str) and sid not in seen:
                    seen[sid] = dict(span)
                    order.append(sid)
            runs: List[Dict[str, Any]] = [seen[sid] for sid in order]

            # Blob extraction is in place on the run list and the blob
            # table; the original ``self._blobs`` is preserved across
            # snapshots so receivers can resolve historical references.
            extract_blobs(runs, self._blobs, self._blob_threshold_bytes)

            self._snapshot_version += 1

            snapshot_out: Dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "schema_uri": SCHEMA_URI,
                "session_id": self._session_id,
                "created_at": self._created_at,
                "snapshot_version": self._snapshot_version,
                "layout": dict(self._layout),
                "agents": dict(self._agent_graph),
                "config": _deepcopy_json(self._config),
                "event_log": self._build_event_log(runs),
                "blobs": dict(self._blobs),
                "drift_log": list(self._drift_log),
            }
            # BSP-005 §6.1 / S0.5 -- emit ``cells`` ONLY when non-empty
            # so existing baseline-snapshot tests stay byte-identical.
            if self._cells:
                # Back-filled records need their _kind_back_filled
                # marker cleared on emission so the *persisted* snapshot
                # carries only the canonical kind field.  We mutate
                # in place: the marker is an in-memory hint, not part
                # of the on-the-wire schema. Same treatment for the
                # BSP-007 ``_executing`` test seam (set via
                # :meth:`set_cell_execution_state`) -- it lives in
                # memory only and never lands on the wire.
                cells_out: Dict[str, Dict[str, Any]] = {}
                _IN_MEMORY_ONLY_KEYS = ("_kind_back_filled", "_executing")
                for cell_id, record in self._cells.items():
                    persisted = {
                        k: v for k, v in record.items()
                        if k not in _IN_MEMORY_ONLY_KEYS
                    }
                    cells_out[cell_id] = persisted
                    # Clear the back-fill marker now that we've written
                    # the kind field into a snapshot the persistence
                    # layer will emit.
                    record.pop("_kind_back_filled", None)
                snapshot_out["cells"] = cells_out
            # V1 Kernel Gap Closure G13 -- emit ``queues`` ONLY when
            # non-empty for the same byte-identity reason.
            if self._queues:
                snapshot_out["queues"] = _deepcopy_json(self._queues)
            # BSP-008 §3 / S3.5 -- emit ``zone`` ONLY when non-empty so
            # baseline-snapshot tests stay byte-identical for unmutated
            # writers. The hydrate path round-trips this verbatim.
            if self._zone:
                snapshot_out["zone"] = _deepcopy_json(self._zone)
            return snapshot_out

    def _build_event_log(
        self, runs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compose the ``event_log`` substructure.

        ``intent_log`` (BSP-003 §6 step 7) is included only when
        non-empty so on-the-wire snapshots from a never-mutated writer
        remain byte-identical to a freshly-built baseline (existing
        hydrate round-trip tests rely on this).
        """
        out: Dict[str, Any] = {"version": 1, "runs": runs}
        if self._intent_log:
            out["intent_log"] = [
                _deepcopy_json(e) for e in self._intent_log
            ]
        return out

    def _build_envelope(
        self, snapshot: Dict[str, Any], *, trigger: str,
    ) -> Dict[str, Any]:
        """Wrap a snapshot in the RFC-006 §8 ``notebook.metadata`` envelope.

        The dispatcher's :meth:`emit` accepts the internal v1
        envelope shape; the dispatcher flattens to the RFC-006 v2
        thin form on egress.  We use ``correlation_id`` =
        ``session_id`` + ``snapshot_version`` so the wire form has a
        stable identifier per emission for log correlation.
        """
        from .run_envelope import make_envelope
        payload: Dict[str, Any] = {
            "mode": "snapshot",
            "snapshot_version": snapshot["snapshot_version"],
            "snapshot": snapshot,
            "trigger": trigger,
        }
        correlation_id = f"{self._session_id}:{snapshot['snapshot_version']}"
        # ``make_envelope`` synthesizes the timestamp / direction /
        # rfc_version fields the internal validator requires; the
        # dispatcher's outbound flattener drops them per RFC-006 §3.
        return make_envelope(
            "notebook.metadata", payload,
            correlation_id=correlation_id,
        )


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
    "DEFAULT_AUTOSAVE_INTERVAL_SEC",
    "DEFAULT_BLOB_THRESHOLD_BYTES",
    "DEFAULT_EVENT_LOG_QUEUE_CAP",
    "MetadataWriter",
    "SCHEMA_URI",
    "SCHEMA_VERSION",
    "SecretRejected",
    "extract_blobs",
    "reject_secrets",
    "serialize_snapshot",
]

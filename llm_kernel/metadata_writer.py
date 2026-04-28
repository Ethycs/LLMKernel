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
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

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
            if intent_kind in {"record_event", "acknowledge_drift"}:
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
            def _h(params: Dict[str, Any]) -> bool:
                # Append a structured event-log entry; we fold these
                # into the intent_log so the snapshot carries them.
                fp = params.get("field_path")
                if not isinstance(fp, str) or not fp:
                    raise ValueError("record_event: field_path is required")
                self.append_drift_event(
                    field_path=fp,
                    previous_value=params.get("previous_value"),
                    current_value=params.get("current_value"),
                    severity=str(params.get("severity", "info")),
                    detected_at=params.get("detected_at"),
                )
                return True
            return _h
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

        # BSP-002 turn graph kinds: not yet wired into MetadataWriter
        # state (the writer carries the agent graph but not the
        # turn-level conversation graph).  We return a stub that
        # raises K42-style validation so the dispatcher returns a
        # well-formed error rather than silently ignoring the call.
        # When BSP-002 §"Implementation slice" lands, this stub is
        # replaced by the real apply functions.
        def _stub(params: Dict[str, Any]) -> bool:
            raise ValueError(
                f"intent_kind {intent_kind!r} is not yet implemented in "
                "the V1 MetadataWriter (BSP-002 turn graph slice pending)"
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
            self._dirty = False

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

            return {
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

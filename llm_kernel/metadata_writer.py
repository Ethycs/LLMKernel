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
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

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
    ) -> None:
        """Construct an unstarted writer.

        ``dispatcher`` may be ``None`` for tests / smokes that capture
        emissions through :meth:`take_last_envelope`.  ``run_tracker``
        is the source of truth for ``event_log`` if set; otherwise the
        writer relies entirely on :meth:`record_run`.  ``session_id``
        defaults to a fresh UUIDv4; reopening an existing file would
        pass the persisted UUID through.
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

        self._layout: Dict[str, Any] = {
            "version": 1,
            "tree": {
                "id": "root", "type": "workspace",
                "render_hints": {}, "children": [],
            },
        }
        self._agents: Dict[str, Any] = {
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
            self._agents["nodes"] = list(nodes)
            self._agents["edges"] = list(edges)
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
        overflow (>``event_log_queue_cap`` entries with no extension
        attached AND no successful emission yet) the writer logs a
        checkpoint marker and continues; the operator-facing direct-
        write tool (RFC-005 §F13) is queued for V1.5.
        """
        with self._lock:
            self._extra_runs.append(span)
            if len(self._extra_runs) > self._event_log_queue_cap:
                # Trim the oldest entries to keep memory bounded.
                drop = len(self._extra_runs) - self._event_log_queue_cap
                self._extra_runs = self._extra_runs[drop:]
                self._overflow_checkpoint_count += 1
                logger.warning(
                    "event_log queue overflow; dropped %d oldest spans "
                    "(checkpoints=%d).  TODO(V1.5): direct-write fallback "
                    "per RFC-005 F13.", drop, self._overflow_checkpoint_count,
                )
            self._dirty = True

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
                "agents": dict(self._agents),
                "config": _deepcopy_json(self._config),
                "event_log": {"version": 1, "runs": runs},
                "blobs": dict(self._blobs),
                "drift_log": list(self._drift_log),
            }

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

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

from llm_kernel.capture.metadata_writer._serialize import *  # noqa: F401,F403
from llm_kernel.capture.metadata_writer._writer import *  # noqa: F401,F403

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

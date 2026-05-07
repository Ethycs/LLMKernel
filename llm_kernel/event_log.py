"""PLAN-S6.0 in-tree event-log replayer.

Consumes ``metadata.rts.zone.event_log[]`` (the in-tree array broadened
by PLAN-S6.0 from ``agent_ref_move``-only to the full RFC-006 envelope
stream) and projects state via the same dispatcher routing the live
path uses, in *read-only* mode (no agent processes spawn).

Public surface (locked at PLAN-S6.0 §4):

* :class:`EventLogReplayer` -- consumes a list, no file I/O.
  - :meth:`latest_snapshot` -- return the most-recent
    ``notebook.metadata`` ``mode == "snapshot"`` envelope.
  - :meth:`envelopes_after_snapshot` -- yield envelopes following the
    latest snapshot, in order.
  - :meth:`project_state` -- load the latest snapshot, replay
    subsequent envelopes through a *read-only* dispatcher, return the
    resulting ``metadata.rts``.

Schema-version branching: each envelope carries ``rfc_version``
(``run_envelope.RFC003_VERSION = "1.0.0"``).  On load:

* Major mismatch with the kernel's :data:`llm_kernel.wire.WIRE_VERSION`
  raises :class:`EventLogVersionMismatchError`.
* Minor mismatch logs a warning and proceeds (newer minor MUST be
  backward-compatible per RFC-006 v2.1.0).

When no snapshot exists in the event log, :meth:`project_state` raises
:class:`EventLogReplayError` ("no checkpoint").

Determinism contract (PLAN-S6.0 §3.C):
    same ``event_log[]`` prefix -> byte-identical projection, every time.

This module has NO dependence on the file system; persistence lives in
the ``.llmnb`` JSON tree and is owned by ``MetadataWriter`` /
``executor.run_notebook``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional

from .wire.version import WIRE_MAJOR, WIRE_MINOR

logger: logging.Logger = logging.getLogger("llm_kernel.event_log")

#: ``zone.event_log[]`` entry kinds the replayer recognises as
#: "captured envelopes" (post-S6.0 shape).  Pre-S6.0 entries
#: (``agent_ref_move`` only) are skipped during replay because the
#: writer's own intent path is the source of truth for them.
_ENVELOPE_SHAPE_KEY: str = "message_type"


class EventLogReplayError(RuntimeError):
    """Raised when the event log lacks the data required for replay.

    Concrete cases:
    * No ``notebook.metadata`` ``mode == "snapshot"`` envelope present
      (the replayer needs a checkpoint to start from).
    """


class EventLogVersionMismatchError(RuntimeError):
    """Raised when an envelope's ``rfc_version`` major differs from the kernel.

    Mirrors the runtime wire-handshake major-mismatch behavior per
    RFC-006 §"version negotiation".  Minor mismatch is *not* an error;
    it logs a warning and replay proceeds.
    """


class EventLogReplayUnsafeError(RuntimeError):
    """Raised when ``project_state`` is called with a writable dispatcher.

    Per PLAN-S6.0 §3.D the replay path MUST drive a dispatcher whose
    ``is_writable()`` returns ``False``; a writable dispatcher would
    re-emit captured envelopes on the wire and double-log them in the
    in-tree event log.  The check is a contract assertion at the boundary,
    not an emission gate -- callers are still responsible for honoring
    the read-only contract internally.
    """


def _parse_semver_major_minor(version: str) -> Optional[tuple[int, int]]:
    """Return ``(major, minor)`` for a semver triple, or ``None`` on parse error."""
    if not isinstance(version, str):
        return None
    parts = version.split(".")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        return None


def _check_envelope_version(envelope: Dict[str, Any]) -> None:
    """Branch on the envelope's ``rfc_version`` per PLAN-S6.0 §3.G.

    Raises :class:`EventLogVersionMismatchError` on major mismatch;
    logs a warning and returns on minor mismatch; silent on exact
    match or missing/malformed version (defensive: pre-S6.0 entries
    have no version field and shouldn't trip replay).
    """
    raw = envelope.get("rfc_version")
    parsed = _parse_semver_major_minor(raw) if isinstance(raw, str) else None
    if parsed is None:
        return
    major, minor = parsed
    if major != WIRE_MAJOR:
        raise EventLogVersionMismatchError(
            f"event-log envelope rfc_version={raw!r} major differs from "
            f"kernel WIRE_MAJOR={WIRE_MAJOR}; replay refused (PLAN-S6.0 §3.G)"
        )
    if minor != WIRE_MINOR:
        logger.warning(
            "event-log envelope rfc_version=%r minor differs from "
            "kernel WIRE_MINOR=%d; proceeding (PLAN-S6.0 §3.G)",
            raw, WIRE_MINOR,
        )


class EventLogReplayer:
    """Read-only projection of ``metadata.rts.zone.event_log[]`` to state.

    Constructed against an in-memory list (no path argument; no file
    I/O).  Callers obtain the list from
    ``metadata.rts["zone"]["event_log"]`` after loading the ``.llmnb``
    JSON tree via the existing ``MetadataWriter.hydrate`` path.
    """

    def __init__(self, event_log: List[Dict[str, Any]]) -> None:
        if not isinstance(event_log, list):
            raise TypeError(
                f"event_log MUST be a list of envelope dicts; got "
                f"{type(event_log).__name__}"
            )
        # Eagerly check versions so callers see a mismatch immediately
        # (PLAN-S6.0 §3.G "schema-version reads").
        for entry in event_log:
            if isinstance(entry, dict) and _ENVELOPE_SHAPE_KEY in entry:
                _check_envelope_version(entry)
        self._event_log: List[Dict[str, Any]] = event_log

    # -- Locked surface (PLAN-S6.0 §4) --------------------------------

    def latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Return the most-recent envelope where the payload is a snapshot.

        Recognised by ``message_type == "notebook.metadata"`` AND
        ``payload.mode == "snapshot"`` per PLAN-S6.0 §3.C step 1-2.
        Returns ``None`` if no such envelope exists.
        """
        latest: Optional[Dict[str, Any]] = None
        for entry in self._event_log:
            if not isinstance(entry, dict):
                continue
            if entry.get(_ENVELOPE_SHAPE_KEY) != "notebook.metadata":
                continue
            payload = entry.get("payload")
            if not isinstance(payload, dict):
                continue
            if payload.get("mode") == "snapshot":
                latest = entry
        return latest

    def envelopes_after_snapshot(self) -> Iterator[Dict[str, Any]]:
        """Yield envelopes that follow the latest snapshot, in order.

        Iterates in the original log order; only the trailing tail
        after the most-recent snapshot envelope is emitted.  Pre-S6.0
        ``agent_ref_move`` legacy entries are skipped (they have no
        ``message_type`` field and are not envelopes).
        """
        # Locate the latest-snapshot index via reverse scan.
        snapshot_index: int = -1
        for idx in range(len(self._event_log) - 1, -1, -1):
            entry = self._event_log[idx]
            if not isinstance(entry, dict):
                continue
            if entry.get(_ENVELOPE_SHAPE_KEY) != "notebook.metadata":
                continue
            payload = entry.get("payload")
            if not isinstance(payload, dict):
                continue
            if payload.get("mode") == "snapshot":
                snapshot_index = idx
                break
        if snapshot_index < 0:
            return
        for entry in self._event_log[snapshot_index + 1:]:
            if not isinstance(entry, dict):
                continue
            if _ENVELOPE_SHAPE_KEY not in entry:
                continue
            yield entry

    def project_state(self, *, dispatcher: Any) -> Dict[str, Any]:
        """Project ``metadata.rts`` from the latest snapshot + tail.

        Algorithm (PLAN-S6.0 §3.C):
          1. Assert ``dispatcher.is_writable()`` returns ``False`` --
             writable dispatchers would re-emit captured envelopes on
             every reopen, exponentially duplicating the event log
             (PLAN-S6.0 §3.D).  Default-deny: a dispatcher missing the
             ``is_writable`` method is treated as writable.
          2. Find the latest snapshot envelope.
          3. ``working_rts := snapshot.payload.snapshot``.
          4. Replay each subsequent envelope through ``dispatcher``
             (which the caller MUST have configured in *read-only*
             mode -- see ``llm_client.boot.boot_minimal_kernel``'s
             ``read_only=True`` kwarg, which gates AgentSupervisor
             instantiation, and ``CustomMessageDispatcher(read_only=True)``
             which makes ``is_writable()`` return ``False``).
          5. Return ``working_rts``.

        Raises:
            :class:`EventLogReplayUnsafeError` if the dispatcher is
                writable or lacks ``is_writable``.
            :class:`EventLogReplayError` if no snapshot is present.
        """
        is_writable = getattr(dispatcher, "is_writable", None)
        if not callable(is_writable):
            raise EventLogReplayUnsafeError(
                f"dispatcher of type {type(dispatcher).__name__!r} lacks a "
                "callable is_writable() method; refusing replay (PLAN-S6.0 "
                "§3.D default-deny).  Construct with read_only=True or wrap "
                "in a stub that reports is_writable() == False."
            )
        if is_writable():
            raise EventLogReplayUnsafeError(
                f"dispatcher of type {type(dispatcher).__name__!r} reports "
                "is_writable() == True; refusing replay (PLAN-S6.0 §3.D).  "
                "Pass read_only=True at construction or call "
                "set_read_only(True) before driving project_state -- "
                "otherwise captured envelopes re-emit on the wire and "
                "double-log on every reopen."
            )

        snapshot_envelope = self.latest_snapshot()
        if snapshot_envelope is None:
            raise EventLogReplayError(
                "no checkpoint: event_log[] contains no "
                "notebook.metadata mode=snapshot envelope"
            )
        payload = snapshot_envelope.get("payload") or {}
        working_rts = payload.get("snapshot")
        if not isinstance(working_rts, dict):
            raise EventLogReplayError(
                "snapshot envelope payload.snapshot is not a dict; "
                "event log is corrupt"
            )
        # Deep-copy so the caller's mutations during replay don't
        # bleed back into the source ``event_log[]`` -- replay must be
        # idempotent across calls.
        working_rts = _deepcopy_json(working_rts)

        for envelope in self.envelopes_after_snapshot():
            self._replay_one(envelope, dispatcher=dispatcher)

        return working_rts

    # -- Internals ----------------------------------------------------

    def _replay_one(
        self, envelope: Dict[str, Any], *, dispatcher: Any,
    ) -> None:
        """Re-route one captured envelope through the read-only dispatcher.

        The dispatcher's outbound ``emit`` path tees envelopes back to
        the writer's ``capture_envelope``; during replay we drive the
        inbound handler chain instead so writer state mutations are
        re-applied without re-emitting on the wire.

        Best-effort: a malformed envelope or a missing handler logs and
        continues -- replay never raises mid-stream (the version check
        in ``__init__`` already guarded against major-version drift).
        """
        message_type = envelope.get("message_type")
        if not isinstance(message_type, str):
            return
        # Snapshot envelopes within the tail are checkpoints -- we do
        # NOT re-apply them; the most recent snapshot is already the
        # base.  Skip silently.
        if message_type == "notebook.metadata":
            payload = envelope.get("payload") or {}
            if isinstance(payload, dict) and payload.get("mode") == "snapshot":
                return
        # Drive the dispatcher's registered handler chain.  When no
        # handler exists for this message_type the dispatcher's own
        # log-and-drop path runs (RFC-006 W4 fail-closed).  We use a
        # synthetic comm_msg so existing handlers see a familiar shape.
        on_comm_msg = getattr(dispatcher, "_on_comm_msg", None)
        if not callable(on_comm_msg):
            return
        try:
            on_comm_msg({"content": {"data": dict(envelope)}})
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "event-log replay: dispatcher._on_comm_msg raised for "
                "type=%s; continuing", message_type,
            )


def _deepcopy_json(obj: Any) -> Any:
    """Deep-copy a JSON-shaped tree without ``copy.deepcopy``."""
    if isinstance(obj, dict):
        return {k: _deepcopy_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deepcopy_json(v) for v in obj]
    return obj


__all__ = [
    "EventLogReplayer",
    "EventLogReplayError",
    "EventLogReplayUnsafeError",
    "EventLogVersionMismatchError",
]

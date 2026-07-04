"""RFC-003 envelope helpers (Stage 2 Track B2).

Pure stateless helpers shaped against ``docs/rfcs/RFC-003-custom-message-format.md``
§Envelope schema. Track B3 (custom-message dispatcher) reuses
:func:`make_envelope` / :func:`validate_envelope` without depending on
the run-tracker's stateful class.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, Optional

#: Lowercase-hex regex used to recognize OTLP span ids (16 hex chars,
#: 8 bytes) and trace ids (32 hex chars, 16 bytes).  ``correlation_id``
#: accepts either of these or a UUIDv4 string for backward-compat with
#: receivers that still produce UUIDs.
_HEX_ID_RE: re.Pattern[str] = re.compile(r"^[0-9a-f]{16}$|^[0-9a-f]{32}$")

#: RFC-003 semver this module emits and accepts; receivers MUST reject
#: envelopes whose major version differs.
RFC003_VERSION: str = "1.0.0"

#: Direction string for kernel-emitted envelopes.
DIRECTION_KERNEL_TO_EXTENSION: str = "kernel→extension"
#: Direction string for extension-emitted envelopes.
DIRECTION_EXTENSION_TO_KERNEL: str = "extension→kernel"

#: Every ``message_type`` the kernel emits or accepts.  Spans Families
#: A (run lifecycle), B (layout), C (agent graph), D (operator action),
#: E (heartbeat / liveness), and the RFC-006 v2 addition F (notebook
#: metadata persistence).  The internal envelope contract still uses
#: the wider v1 shape; ``CustomMessageDispatcher`` flattens to the
#: RFC-006 thin v2 envelope on egress.
RFC003_MESSAGE_TYPES: FrozenSet[str] = frozenset(
    {
        "run.start", "run.event", "run.complete",
        "layout.update", "layout.edit",
        "agent_graph.query", "agent_graph.response",
        "operator.action",
        "heartbeat.kernel", "heartbeat.extension",
        "notebook.metadata",
        # RFC-006 v2.0.x additive: graceful kernel shutdown request
        # (extension→kernel) per RFC-008 §4 step 6. EOF remains the
        # fallback when this message is not received.
        "kernel.shutdown_request",
    }
)

_REQUIRED_ENVELOPE_KEYS: FrozenSet[str] = frozenset(
    {"message_type", "direction", "correlation_id", "timestamp", "rfc_version", "payload"}
)
_VALID_DIRECTIONS: FrozenSet[str] = frozenset(
    {DIRECTION_KERNEL_TO_EXTENSION, DIRECTION_EXTENSION_TO_KERNEL}
)


def _is_uuid(value: str) -> bool:
    """Return True iff ``value`` parses as a UUID."""
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError):
        return False
    return True


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def make_envelope(
    message_type: str,
    payload: Dict[str, Any],
    correlation_id: str,
    direction: str = DIRECTION_KERNEL_TO_EXTENSION,
    timestamp: Optional[str] = None,
    rfc_version: str = RFC003_VERSION,
) -> Dict[str, Any]:
    """Build an RFC-003 envelope with the canonical defaults.

    ``message_type`` MUST be one of :data:`RFC003_MESSAGE_TYPES`.
    ``correlation_id`` is a UUIDv4 string (the ``run_id`` for Family A).
    Payload validity is the caller's responsibility.
    """
    return {
        "message_type": message_type,
        "direction": direction,
        "correlation_id": correlation_id,
        "timestamp": timestamp or _utc_now_iso(),
        "rfc_version": rfc_version,
        "payload": payload,
    }


def validate_envelope(envelope: Dict[str, Any]) -> None:
    """Raise :class:`ValueError` if ``envelope`` violates RFC-003 §Envelope.

    Strict on the sender side per RFC-003 §Backward-compatibility (the
    schema's strictness is for senders; receivers should be permissive).
    """
    if not isinstance(envelope, dict):
        raise ValueError(f"envelope MUST be a dict; got {type(envelope).__name__}")
    missing = _REQUIRED_ENVELOPE_KEYS - envelope.keys()
    if missing:
        raise ValueError(f"envelope missing required keys: {sorted(missing)}")

    message_type = envelope["message_type"]
    if message_type not in RFC003_MESSAGE_TYPES:
        raise ValueError(
            f"envelope.message_type {message_type!r} not in RFC-003 v1.0.0 catalog"
        )

    direction = envelope["direction"]
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"envelope.direction {direction!r} is not a valid RFC-003 direction")

    correlation_id = envelope["correlation_id"]
    if not isinstance(correlation_id, str) or not correlation_id:
        raise ValueError(
            "envelope.correlation_id MUST be a non-empty string"
        )
    # Family A (run lifecycle) MUST use the OTLP spanId or traceId.
    # Other families MAY use any non-empty correlation string -- the
    # metadata writer uses ``<session_id>:<snapshot_version>`` for
    # log correlation, the agent_graph pair uses UUIDv4 per RFC-006
    # §5, and the legacy heartbeats UUID4 per RFC-006 §7.  We only
    # enforce the OTLP shape on Family A.
    if message_type in {"run.start", "run.event", "run.complete"}:
        if not (_HEX_ID_RE.match(correlation_id) or _is_uuid(correlation_id)):
            raise ValueError(
                f"envelope.correlation_id {correlation_id!r} is not a "
                "valid OTLP spanId/traceId or UUID for Family A"
            )

    timestamp = envelope["timestamp"]
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("envelope.timestamp MUST be a non-empty ISO 8601 string")

    rfc_version = envelope["rfc_version"]
    if not isinstance(rfc_version, str):
        raise ValueError("envelope.rfc_version MUST be a string")
    parts = rfc_version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"envelope.rfc_version {rfc_version!r} is not a semver triple")

    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise ValueError(
            f"envelope.payload MUST be an object; got {type(payload).__name__}"
        )

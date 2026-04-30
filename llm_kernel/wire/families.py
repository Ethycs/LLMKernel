"""RFC-006 v2 envelope family shapes — public wire surface.

Five families per RFC-006.  TypedDicts use ``total=False`` (all fields
optional at type-check time) to stay loose for V1; the actual wire
validation happens on the tool-schema side (``wire.tools`` validators)
and the dispatcher's ``validate_envelope`` call.  Exhaustive payload
typing is a V2+ concern (see PLAN-S5.0.3 §10 risk #2).

These shapes are derived from the envelope patterns dispatched in
``llm_kernel.custom_messages``.  That module's dispatcher STAYS
unchanged; it continues to route inbound traffic.  The *shape
definitions* live here as the public contract so external drivers can
construct correct envelopes without reading kernel internals.
"""

from __future__ import annotations

from typing import Any, Literal, Union

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover  (Python <3.8 fallback -- not expected)
    from typing_extensions import TypedDict  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Family A: operator-action envelopes (parser -> kernel)
# RFC-006 §3; rides on IOPub display_data / update_display_data.
# ---------------------------------------------------------------------------

class FamilyA_OperatorAction(TypedDict, total=False):
    """Operator-action envelope — kind-discriminated by payload.kind.

    V1 payload kinds: ``run.start``, ``run.event``, ``run.complete``
    (the OTLP run lifecycle).  Full kind set is in RFC-006 §1.
    """
    type: Literal["operator.action"]
    payload: dict[str, Any]   # kind-discriminated; validators in wire.tools


# ---------------------------------------------------------------------------
# Family B: layout edits (extension -> kernel)
# RFC-006 §4; rides on Comm ``llmnb.rts.v2``.
# ---------------------------------------------------------------------------

class FamilyB_LayoutEdit(TypedDict, total=False):
    """Layout-edit envelope — mutates the cell/zone layout tree.

    Kernel echoes the new state as a ``layout.update`` after applying.
    Payload keys: ``operation`` (str), ``parameters`` (dict).
    """
    type: Literal["layout.edit"]
    payload: dict[str, Any]   # {operation: str, parameters: dict}


# ---------------------------------------------------------------------------
# Family C: agent-graph commands (bidirectional)
# RFC-006 §5; correlation_id required for request/response pairing.
# ---------------------------------------------------------------------------

class FamilyC_AgentGraphCommand(TypedDict, total=False):
    """Agent-graph command envelope (query / response).

    ``agent_graph.query`` goes extension -> kernel;
    ``agent_graph.response`` goes kernel -> extension.
    ``correlation_id`` is required for this family (RFC-006 §5).
    """
    type: Literal["agent_graph.command"]
    payload: dict[str, Any]   # {query_type: str, ...}
    correlation_id: str


# ---------------------------------------------------------------------------
# Family F: notebook metadata snapshots / patches / hydrate
# RFC-006 §8; bidirectional in v2.0.2.
# ---------------------------------------------------------------------------

class FamilyF_NotebookSnapshot(TypedDict, total=False):
    """Notebook-metadata envelope.

    ``mode`` discriminates behavior:
    - ``"hydrate"`` — extension -> kernel on file-open; triggers
      MetadataWriter.hydrate + DriftDetector.compare + agent respawn.
    - ``"snapshot"`` — kernel -> extension; carries full metadata state.
    - ``"patch"`` — V1.5+ (RFC-006 §8); rejected in V1.

    Payload keys vary by mode; see RFC-006 §8 for full shape.
    """
    type: Literal["notebook.metadata"]
    payload: dict[str, Any]   # {mode: str, snapshot?: dict, trigger?: str}


# ---------------------------------------------------------------------------
# Family G: lifecycle (kernel.shutdown_request, heartbeat.kernel, etc.)
# RFC-006 §7; includes heartbeat (v2.0.2 amendment) and shutdown.
# ---------------------------------------------------------------------------

class FamilyG_Lifecycle(TypedDict, total=False):
    """Lifecycle envelope for kernel state management.

    Known ``type`` values:
    - ``"kernel.shutdown_request"`` — extension -> kernel graceful shutdown.
    - ``"heartbeat.kernel"`` — kernel -> extension every 5s (RFC-006 §7).
    - ``"kernel.handshake"`` — V1.5+ (PLAN-S5.0.3d); reserved here.

    ``payload`` shape is type-specific; see RFC-006 §7 for full detail.
    """
    type: str   # Literal union deferred; full set grows with RFC amendments
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Handshake envelope (S5.0.3d).
#
# First envelope on any connection per docs/atoms/protocols/wire-handshake.md.
# Negotiates wire version, declares the driver's capabilities, and (for TCP)
# carries bearer-token auth. The kernel responds with its own version + a
# session id + the accepted capability set. On mismatched WIRE_MAJOR or auth
# failure the kernel sends an error envelope (HandshakeResponse with
# ``error`` set in payload) and closes the transport. No Family A/B/C/F/G
# frames flow until handshake succeeds.
# ---------------------------------------------------------------------------


class HandshakeAuth(TypedDict, total=False):
    """Auth block on handshake request payload.

    Present iff ``transport == "tcp"``; absent for ``pty``/``unix``.
    Token comparison is constant-time (``hmac.compare_digest``); see
    PLAN-S5.0.3 §5.2.
    """
    scheme: Literal["bearer"]
    token: str


class HandshakeRequestPayload(TypedDict, total=False):
    """Driver -> kernel handshake payload."""
    client_name: str            # "llmnb-cli" | "vscode-extension" | <custom>
    client_version: str         # semver
    wire_version: str           # semver; matched against kernel WIRE_VERSION
    transport: Literal["pty", "unix", "tcp"]
    auth: HandshakeAuth         # required for tcp; absent for pty/unix
    capabilities: list[str]     # ["family_a", "family_b", ...] -- V1: full set


class HandshakeResponsePayload(TypedDict, total=False):
    """Kernel -> driver handshake payload (success).

    On error, ``error`` is set instead of ``session_id`` /
    ``accepted_capabilities``.  Known error codes (see
    docs/atoms/protocols/wire-handshake.md):
        ``version_mismatch_major`` -- WIRE_MAJOR differs
        ``auth_failed``            -- TCP token missing/invalid
        ``kernel_busy``            -- second client to single-client kernel
        ``wire-failure``           -- malformed handshake payload
    """
    kernel_version: str
    wire_version: str
    session_id: str
    accepted_capabilities: list[str]
    warnings: list[str]
    error: str                  # set iff handshake rejected


class HandshakeRequest(TypedDict, total=False):
    """Driver -> kernel handshake envelope.

    First envelope on any connection per
    [protocols/wire-handshake](../../../../docs/atoms/protocols/wire-handshake.md).
    """
    type: Literal["kernel.handshake"]
    payload: HandshakeRequestPayload


class HandshakeResponse(TypedDict, total=False):
    """Kernel -> driver handshake response."""
    type: Literal["kernel.handshake"]
    payload: HandshakeResponsePayload


# ---------------------------------------------------------------------------
# Union alias
# ---------------------------------------------------------------------------

Envelope = Union[
    FamilyA_OperatorAction,
    FamilyB_LayoutEdit,
    FamilyC_AgentGraphCommand,
    FamilyF_NotebookSnapshot,
    FamilyG_Lifecycle,
]

__all__ = [
    "FamilyA_OperatorAction",
    "FamilyB_LayoutEdit",
    "FamilyC_AgentGraphCommand",
    "FamilyF_NotebookSnapshot",
    "FamilyG_Lifecycle",
    "HandshakeAuth",
    "HandshakeRequestPayload",
    "HandshakeResponsePayload",
    "HandshakeRequest",
    "HandshakeResponse",
    "Envelope",
]

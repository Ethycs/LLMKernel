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
# Handshake shapes (referenced by wire-handshake protocol; implemented in
# S5.0.3d -- these are forward-declared stubs so families.py is importable
# without import errors in slice a).
# ---------------------------------------------------------------------------

class HandshakeRequest(TypedDict, total=False):
    """Client -> Kernel handshake envelope (V1.5+, PLAN-S5.0.3d).

    Declared here so ``wire/__init__.py`` can re-export the symbol per
    PLAN §4.1; full implementation lands in slice S5.0.3d.
    """
    type: Literal["kernel.handshake"]
    payload: dict[str, Any]


class HandshakeResponse(TypedDict, total=False):
    """Kernel -> Client handshake response (V1.5+, PLAN-S5.0.3d)."""
    type: Literal["kernel.handshake"]
    payload: dict[str, Any]


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
    "HandshakeRequest",
    "HandshakeResponse",
    "Envelope",
]

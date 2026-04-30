"""Public wire API for LLMKernel (S5.0.3a).

Envelope schemas (Family A/B/C/F/G), version constants, tool validators,
and JSON-Schema exports.  This is the *only* surface external drivers
(llm_client, future Rust/Go clients) may import from llm_kernel.

Imported by both the kernel (llm_kernel.*) and external clients.
Has no imports of other llm_kernel modules -- only stdlib and this package.
"""

from .version import WIRE_VERSION, WIRE_MAJOR, WIRE_MINOR, WIRE_PATCH
from .families import (
    Envelope,
    FamilyA_OperatorAction,
    FamilyB_LayoutEdit,
    FamilyC_AgentGraphCommand,
    FamilyF_NotebookSnapshot,
    FamilyG_Lifecycle,
    HandshakeAuth,
    HandshakeRequest,
    HandshakeRequestPayload,
    HandshakeResponse,
    HandshakeResponsePayload,
)
from .tools import TOOL_CATALOG, validate_tool_input, validate_tool_output

__all__ = [
    # Version constants
    "WIRE_VERSION",
    "WIRE_MAJOR",
    "WIRE_MINOR",
    "WIRE_PATCH",
    # Envelope families
    "Envelope",
    "FamilyA_OperatorAction",
    "FamilyB_LayoutEdit",
    "FamilyC_AgentGraphCommand",
    "FamilyF_NotebookSnapshot",
    "FamilyG_Lifecycle",
    "HandshakeAuth",
    "HandshakeRequest",
    "HandshakeRequestPayload",
    "HandshakeResponse",
    "HandshakeResponsePayload",
    # Tool catalog + validators
    "TOOL_CATALOG",
    "validate_tool_input",
    "validate_tool_output",
]

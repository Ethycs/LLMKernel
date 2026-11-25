"""Public wire API for LLMKernel (S5.0.3a).

Envelope schemas (Family A/B/C/F/G), version constants, tool validators,
and JSON-Schema exports.  This is the *only* surface external drivers
(llm_client, future Rust/Go clients) may import from llm_kernel.

Imported by both the kernel (llm_kernel.*) and external clients.
Has no imports of other llm_kernel modules -- only stdlib and this package.

Version constants are *forwarded*, not copied: ``llm_kernel.wire.X`` for
``X`` in :data:`_FORWARDED` is resolved on every access (PEP 562) from
the single source :mod:`llm_kernel.wire.versions`, so a runtime
override of that module propagates to every reader that goes through
this package at call time.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from llm_kernel.wire import versions
from llm_kernel.wire import version as _version_mod
from llm_kernel.wire.version import wire_version_tuple
from llm_kernel.wire.families import (
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
from llm_kernel.wire.tools import TOOL_CATALOG, validate_tool_input, validate_tool_output

_FORWARDED: Dict[str, Callable[[], Any]] = {
    # Wire version (derived components live in wire.version)
    "WIRE_VERSION": lambda: _version_mod.WIRE_VERSION,
    "WIRE_MAJOR": lambda: _version_mod.WIRE_MAJOR,
    "WIRE_MINOR": lambda: _version_mod.WIRE_MINOR,
    "WIRE_PATCH": lambda: _version_mod.WIRE_PATCH,
    # Everything else in the single source
    "KERNEL_VERSION": lambda: versions.KERNEL_VERSION,
    "SCHEMA_VERSION": lambda: versions.SCHEMA_VERSION,
    "RFC_VERSIONS": lambda: versions.RFC_VERSIONS,
    "KERNEL_CAPABILITIES": lambda: versions.KERNEL_CAPABILITIES,
    "COMM_TARGET": lambda: versions.COMM_TARGET,
    "PACKAGE_VERSION": lambda: versions.PACKAGE_VERSION,
    "CLIENT_VERSION": lambda: versions.CLIENT_VERSION,
}


def __getattr__(name: str) -> Any:
    try:
        getter = _FORWARDED[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getter()


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_FORWARDED))


__all__ = [
    # Version constants (forwarded from llm_kernel.wire.versions)
    "WIRE_VERSION",
    "WIRE_MAJOR",
    "WIRE_MINOR",
    "WIRE_PATCH",
    "KERNEL_VERSION",
    "SCHEMA_VERSION",
    "RFC_VERSIONS",
    "KERNEL_CAPABILITIES",
    "COMM_TARGET",
    "PACKAGE_VERSION",
    "CLIENT_VERSION",
    "versions",
    "wire_version_tuple",
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

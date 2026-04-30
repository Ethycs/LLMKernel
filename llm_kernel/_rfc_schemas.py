"""Deprecated: prefer ``llm_kernel.wire``.

Aliases retained for in-repo back-compat (S5.0.3a).  Every existing
import of this module continues to work unchanged.  New code MUST use
``llm_kernel.wire.tools`` directly.

Ship note: promoted to public API in S5.0.3a — see
``llm_kernel/wire/`` package.  Commit pin: <TBD-after-commit>.
"""

from __future__ import annotations

# Re-export everything the old module exposed.  The star import covers
# all public names; the explicit list covers the underscore-prefixed
# private helpers that callers may have imported directly.
from llm_kernel.wire.tools import *  # noqa: F401, F403
from llm_kernel.wire.tools import (  # noqa: F401
    # Private module helpers (re-exported for back-compat)
    _DRAFT,
    _VER,
    _RID,
    _VS,
    _obj,
    _ack,
    _ACK,
    _TYPE_PY,
    _hand_validate,
    # Public catalog + validators
    TOOL_CATALOG,
    validate_tool_input,
    validate_tool_output,
    # K-class registry
    K_CLASS_REGISTRY,
    k_class_info,
    # K-code string constants (new in S5.0.3a; harmless additions)
    K30_MULTIPLE_KINDS,
    K31_UNKNOWN_CELL_MAGIC,
    K32_RESERVED_MAGIC_NAME,
    K33_MAGIC_HASH_MISMATCH,
    K34_INCOMPATIBLE_KIND_CHANGE,
    K35_PLAIN_MAGIC_IN_HASH_MODE,
    K36_HASHED_MAGIC_EMISSION_BLOCKED,
    K3C_RUNNING_CELL_STRUCTURAL_OP_BLOCKED,
    K3D_RUNNING_CELL_KIND_CHANGE_BLOCKED,
    K3E_CONTAMINATED_CELL_STRUCTURAL_OP_BLOCKED,
    K3F_RUNNING_CELL_EDIT_TEXT_ONLY_PATH,
    K3G_OPERATOR_ACCEPTED_INJECTION_PERSISTED,
    K3H_AGENT_EMITTED_GENERATOR_MAGIC_BLOCKED,
    K3I_GENERATOR_HANDLER_PRODUCED_INVALID_HASH,
    K3J_GENERATOR_PROVENANCE_MISSING,
    # Individual tool schemas
    ASK_INPUT, ASK_OUTPUT,
    CLARIFY_INPUT, CLARIFY_OUTPUT,
    PROPOSE_INPUT, PROPOSE_OUTPUT,
    REQUEST_APPROVAL_INPUT, REQUEST_APPROVAL_OUTPUT,
    REPORT_PROGRESS_INPUT, REPORT_PROGRESS_OUTPUT,
    REPORT_COMPLETION_INPUT, REPORT_COMPLETION_OUTPUT,
    REPORT_PROBLEM_INPUT, REPORT_PROBLEM_OUTPUT,
    PRESENT_INPUT, PRESENT_OUTPUT,
    NOTIFY_INPUT, NOTIFY_OUTPUT,
    ESCALATE_INPUT, ESCALATE_OUTPUT,
    READ_FILE_INPUT, READ_FILE_OUTPUT,
    WRITE_FILE_INPUT, WRITE_FILE_OUTPUT,
    RUN_COMMAND_INPUT, RUN_COMMAND_OUTPUT,
    NATIVE_TOOLS,
    PROXIED_TOOLS,
    JSONSchema,
)

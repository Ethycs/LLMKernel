"""RFC-001 v1.0.0 tool input/output JSON Schemas.

JSON Schemas (Draft 2020-12) for the thirteen RFC-001 tools — ten native
operator-interaction primitives and three proxied system tools — extracted
verbatim from ``docs/rfcs/RFC-001-mcp-tool-taxonomy.md``. The MCP server
in :mod:`llm_kernel.mcp_server` registers these as the ``inputSchema`` /
``outputSchema`` for each tool.

Per RFC-001 common conventions, every schema sets
``$schema = "https://json-schema.org/draft/2020-12/schema"``,
``additionalProperties = False``, and includes the optional
``_rfc_version`` field defaulting to ``"1.0.0"``. Section anchors below
(``# RFC-001 §...``) point at the headings in the RFC where each schema
was lifted from.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

JSONSchema = Dict[str, Any]

_DRAFT: str = "https://json-schema.org/draft/2020-12/schema"
_VER: Dict[str, Any] = {"type": "string", "default": "1.0.0"}
_RID: Dict[str, str] = {"type": "string", "format": "uuid"}
_VS: Dict[str, str] = {"type": "string"}


def _obj(required: list, props: Dict[str, Any]) -> JSONSchema:
    """Return a Draft-2020-12 object schema with strict additionalProperties."""
    return {"$schema": _DRAFT, "type": "object", "required": required,
            "additionalProperties": False, "properties": props}


# Shared output shape — RFC-001 reuses the acknowledged/run_id pair across
# report_progress, report_completion, report_problem, and notify.
_ACK: JSONSchema = _obj(["acknowledged", "run_id"], {
    "_rfc_version": _VS, "run_id": _RID,
    "acknowledged": {"type": "boolean", "const": True}})


def _ack() -> JSONSchema:
    """Fresh copy of the shared acknowledged/run_id output schema."""
    return copy.deepcopy(_ACK)


# ---- Native tools --------------------------------------------------------

# RFC-001 §ask
ASK_INPUT: JSONSchema = _obj(["question"], {
    "_rfc_version": _VER,
    "question": {"type": "string", "minLength": 1},
    "context": _VS,
    "options": _obj([], {
        "timeout_ms": {"type": "integer", "minimum": 0, "default": 600000},
        "allow_followup": {"type": "boolean", "default": True}})})
ASK_OUTPUT: JSONSchema = _obj(["answer", "run_id"], {
    "_rfc_version": _VS, "run_id": _RID, "answer": _VS,
    "answered_at": {"type": "string", "format": "date-time"}})

# RFC-001 §clarify
CLARIFY_INPUT: JSONSchema = _obj(["question", "options"], {
    "_rfc_version": _VER,
    "question": {"type": "string", "minLength": 1},
    "options": {"type": "array", "minItems": 2, "items": _obj(["id", "label"], {
        "id": {"type": "string", "pattern": "^[a-z0-9_]+$"},
        "label": _VS, "description": _VS})},
    "default_id": _VS,
    "timeout_ms": {"type": "integer", "minimum": 0, "default": 600000}})
CLARIFY_OUTPUT: JSONSchema = _obj(["selected_id", "run_id"], {
    "_rfc_version": _VS, "run_id": _RID,
    "selected_id": _VS, "free_text": _VS})

# RFC-001 §propose
PROPOSE_INPUT: JSONSchema = _obj(["action", "rationale"], {
    "_rfc_version": _VER,
    "action": {"type": "string", "minLength": 1},
    "rationale": {"type": "string", "minLength": 1},
    "preview": _obj([], {
        "kind": {"type": "string", "enum": ["text", "diff", "plan", "code", "json"]},
        "body": _VS}),
    "scope": {"type": "string",
              "enum": ["one_shot", "this_file", "this_zone", "session"],
              "default": "one_shot"},
    "timeout_ms": {"type": "integer", "minimum": 0, "default": 1800000}})
PROPOSE_OUTPUT: JSONSchema = _obj(["decision", "run_id"], {
    "_rfc_version": _VS, "run_id": _RID,
    "decision": {"type": "string", "enum": ["accept", "reject", "modify", "defer"]},
    "modification": _VS,
    "scope_granted": {"type": "string",
                      "enum": ["one_shot", "this_file", "this_zone", "session"]}})

# RFC-001 §request_approval
REQUEST_APPROVAL_INPUT: JSONSchema = _obj(["action", "diff_preview", "risk_level"], {
    "_rfc_version": _VER,
    "action": {"type": "string", "minLength": 1},
    "diff_preview": _obj(["kind", "body"], {
        "kind": {"type": "string", "enum": ["unified_diff", "text", "code", "command"]},
        "body": _VS, "file_a": _VS, "file_b": _VS}),
    "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
    "alternatives": {"type": "array", "items": _obj(["label", "description"], {
        "label": _VS, "description": _VS})},
    "timeout_ms": {"type": "integer", "minimum": 0, "default": 1800000}})
REQUEST_APPROVAL_OUTPUT: JSONSchema = _obj(["decision", "run_id"], {
    "_rfc_version": _VS, "run_id": _RID,
    "decision": {"type": "string",
                 "enum": ["approve", "approve_with_modification", "deny", "defer"]},
    "modification": _VS, "alternative_label": _VS})

# RFC-001 §report_progress
REPORT_PROGRESS_INPUT: JSONSchema = _obj(["status"], {
    "_rfc_version": _VER,
    "status": {"type": "string", "minLength": 1},
    "percent": {"type": "number", "minimum": 0, "maximum": 100},
    "blockers": {"type": "array", "items": _VS},
    "display_id": _VS})
REPORT_PROGRESS_OUTPUT: JSONSchema = _ack()

# RFC-001 §report_completion
REPORT_COMPLETION_INPUT: JSONSchema = _obj(["summary"], {
    "_rfc_version": _VER,
    "summary": {"type": "string", "minLength": 1},
    "artifacts": {"type": "array", "items": _obj(["uri", "kind"], {
        "uri": _VS,
        "kind": {"type": "string", "enum": ["file", "diff", "plan", "url", "log"]},
        "title": _VS})},
    "outcome": {"type": "string",
                "enum": ["success", "partial", "aborted"], "default": "success"}})
REPORT_COMPLETION_OUTPUT: JSONSchema = _ack()

# RFC-001 §report_problem
REPORT_PROBLEM_INPUT: JSONSchema = _obj(["severity", "description"], {
    "_rfc_version": _VER,
    "severity": {"type": "string", "enum": ["info", "warning", "error", "fatal"]},
    "description": {"type": "string", "minLength": 1},
    "suggested_remediation": _VS,
    "related_artifacts": {"type": "array", "items": _VS}})
REPORT_PROBLEM_OUTPUT: JSONSchema = _ack()

# RFC-001 §present
PRESENT_INPUT: JSONSchema = _obj(["artifact", "kind", "summary"], {
    "_rfc_version": _VER,
    "artifact": _obj(["body"], {
        "body": _VS, "uri": _VS, "language": _VS,
        "encoding": {"type": "string", "enum": ["utf-8", "base64"], "default": "utf-8"}}),
    "kind": {"type": "string", "enum": ["code", "plan", "diff", "doc", "json", "image"]},
    "summary": {"type": "string", "minLength": 1}})
PRESENT_OUTPUT: JSONSchema = _obj(["artifact_id", "run_id"], {
    "_rfc_version": _VS, "run_id": _RID, "artifact_id": _VS})

# RFC-001 §notify
NOTIFY_INPUT: JSONSchema = _obj(["observation", "importance"], {
    "_rfc_version": _VER,
    "observation": {"type": "string", "minLength": 1},
    "importance": {"type": "string", "enum": ["trace", "info", "warn"]},
    "tags": {"type": "array", "items": _VS}})
NOTIFY_OUTPUT: JSONSchema = _ack()

# RFC-001 §escalate
ESCALATE_INPUT: JSONSchema = _obj(["reason", "severity"], {
    "_rfc_version": _VER,
    "reason": {"type": "string", "minLength": 1},
    "severity": {"type": "string", "enum": ["medium", "high", "critical"]},
    "context": _VS,
    "timeout_ms": {"type": "integer", "minimum": 0, "default": 300000}})
ESCALATE_OUTPUT: JSONSchema = _obj(["acknowledged", "run_id"], {
    "_rfc_version": _VS, "run_id": _RID,
    "acknowledged": {"type": "boolean", "const": True},
    "operator_response": _VS})

# ---- Proxied tools -------------------------------------------------------

# RFC-001 §read_file
READ_FILE_INPUT: JSONSchema = _obj(["path"], {
    "_rfc_version": _VER,
    "path": {"type": "string", "minLength": 1},
    "encoding": {"type": "string", "enum": ["utf-8", "base64"], "default": "utf-8"},
    "max_bytes": {"type": "integer", "minimum": 1, "default": 1048576}})
READ_FILE_OUTPUT: JSONSchema = _obj(["content", "encoding", "run_id"], {
    "_rfc_version": _VS, "run_id": _RID, "content": _VS,
    "encoding": {"type": "string", "enum": ["utf-8", "base64"]},
    "truncated": {"type": "boolean"},
    "size_bytes": {"type": "integer", "minimum": 0}})

# RFC-001 §write_file
WRITE_FILE_INPUT: JSONSchema = _obj(["path", "content"], {
    "_rfc_version": _VER,
    "path": {"type": "string", "minLength": 1},
    "content": _VS,
    "encoding": {"type": "string", "enum": ["utf-8", "base64"], "default": "utf-8"},
    "mode": {"type": "string",
             "enum": ["create", "overwrite", "append"], "default": "overwrite"}})
WRITE_FILE_OUTPUT: JSONSchema = _obj(["bytes_written", "run_id"], {
    "_rfc_version": _VS, "run_id": _RID,
    "bytes_written": {"type": "integer", "minimum": 0},
    "created": {"type": "boolean"}})

# RFC-001 §run_command
RUN_COMMAND_INPUT: JSONSchema = _obj(["command"], {
    "_rfc_version": _VER,
    "command": {"type": "string", "minLength": 1},
    "args": {"type": "array", "items": _VS},
    "cwd": _VS,
    "timeout_ms": {"type": "integer", "minimum": 1, "default": 60000},
    "env": {"type": "object", "additionalProperties": _VS}})
RUN_COMMAND_OUTPUT: JSONSchema = _obj(["exit_code", "stdout", "stderr", "run_id"], {
    "_rfc_version": _VS, "run_id": _RID,
    "exit_code": {"type": "integer"},
    "stdout": _VS, "stderr": _VS,
    "timed_out": {"type": "boolean"},
    "duration_ms": {"type": "integer", "minimum": 0}})


# Public catalog: tool name -> (input_schema, output_schema, description).
# Order matches RFC-002's mcpServers.allowedTools array exactly.
TOOL_CATALOG: Dict[str, Tuple[JSONSchema, JSONSchema, str]] = {
    "ask": (ASK_INPUT, ASK_OUTPUT, "Operator-targeted free-form question."),
    "clarify": (CLARIFY_INPUT, CLARIFY_OUTPUT, "Typed clarification with a discrete option set."),
    "propose": (PROPOSE_INPUT, PROPOSE_OUTPUT, "Proposed action with rationale, optional preview, and scope."),
    "request_approval": (REQUEST_APPROVAL_INPUT, REQUEST_APPROVAL_OUTPUT, "Hard gate before performing an executable operation."),
    "report_progress": (REPORT_PROGRESS_INPUT, REPORT_PROGRESS_OUTPUT, "Status update during long-running work; non-blocking."),
    "report_completion": (REPORT_COMPLETION_INPUT, REPORT_COMPLETION_OUTPUT, "Final completion signal for a unit of agent work."),
    "report_problem": (REPORT_PROBLEM_INPUT, REPORT_PROBLEM_OUTPUT, "Blocking issue the agent encountered."),
    "present": (PRESENT_INPUT, PRESENT_OUTPUT, "Generated content lifted to the artifacts surface."),
    "notify": (NOTIFY_INPUT, NOTIFY_OUTPUT, "Fire-and-forget annotation."),
    "escalate": (ESCALATE_INPUT, ESCALATE_OUTPUT, "Demands operator attention urgently."),
    "read_file": (READ_FILE_INPUT, READ_FILE_OUTPUT, "Proxied: returns file contents from the workspace."),
    "write_file": (WRITE_FILE_INPUT, WRITE_FILE_OUTPUT, "Proxied: writes file contents inside the workspace."),
    "run_command": (RUN_COMMAND_INPUT, RUN_COMMAND_OUTPUT, "Proxied: executes a shell command in the zone workspace."),
}

# Tools whose handlers are real (B1-stub) implementations.
NATIVE_TOOLS: Tuple[str, ...] = (
    "ask", "clarify", "propose", "request_approval",
    "report_progress", "report_completion", "report_problem",
    "present", "notify", "escalate")

# Tools that are proxied; B1 leaves them unimplemented and raises NotImplementedError.
PROXIED_TOOLS: Tuple[str, ...] = ("read_file", "write_file", "run_command")

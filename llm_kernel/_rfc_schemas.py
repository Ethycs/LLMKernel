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
from typing import Any, Dict, Optional, Tuple

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
# ``task_id`` is additive (V1 mega-round): the kernel uses it to enforce
# RFC-001 §report_completion's "exactly one per task" invariant.
REPORT_COMPLETION_INPUT: JSONSchema = _obj(["summary"], {
    "_rfc_version": _VER,
    "summary": {"type": "string", "minLength": 1},
    "artifacts": {"type": "array", "items": _obj(["uri", "kind"], {
        "uri": _VS,
        "kind": {"type": "string", "enum": ["file", "diff", "plan", "url", "log"]},
        "title": _VS})},
    "outcome": {"type": "string",
                "enum": ["success", "partial", "aborted"], "default": "success"},
    "task_id": _VS})
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
# ``artifact_id`` is additive (V1 mega-round): when supplied, the
# kernel returns the cached response for that id instead of minting a
# fresh one (RFC-001 §present idempotency).
PRESENT_INPUT: JSONSchema = _obj(["artifact", "kind", "summary"], {
    "_rfc_version": _VER,
    "artifact": _obj(["body"], {
        "body": _VS, "uri": _VS, "language": _VS,
        "encoding": {"type": "string", "enum": ["utf-8", "base64"], "default": "utf-8"}}),
    "kind": {"type": "string", "enum": ["code", "plan", "diff", "doc", "json", "image"]},
    "summary": {"type": "string", "minLength": 1},
    "artifact_id": _VS})
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


# ---- Input validation -----------------------------------------------------

# JSON Schema -> Python type-check map.  Used by the hand-rolled fallback
# checker if ``jsonschema`` is unavailable.  ``"integer"`` accepts ``bool``
# rejection because Python booleans subclass ``int``.
_TYPE_PY: Dict[str, Tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def _hand_validate(schema: JSONSchema, value: Any, path: str = "$") -> Optional[str]:
    """Tiny hand-rolled checker covering the subset RFC-001 schemas use.

    Validates: ``type``, ``required``, ``additionalProperties`` (bool),
    ``minLength``, ``minItems``, ``minimum``, ``maximum``, ``enum``,
    ``items``, nested ``properties``.  Returns ``None`` on success or a
    human-readable error string on first failure.  The structure matches
    Draft-2020-12's behavior closely enough for the RFC-001 schemas, but
    falls short of full conformance — which is why ``validate_tool_input``
    prefers ``jsonschema`` when available.
    """
    expected = schema.get("type")
    if expected:
        py_types = _TYPE_PY.get(expected)
        if py_types is not None:
            # ``bool`` is a subclass of ``int``; reject it for "integer"
            # / "number" to match JSON Schema semantics.
            if expected in ("integer", "number") and isinstance(value, bool):
                return f"{path}: expected {expected}, got boolean"
            if not isinstance(value, py_types):
                got = type(value).__name__
                return f"{path}: expected {expected}, got {got}"

    if expected == "object" and isinstance(value, dict):
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                return f"{path}: missing required property {key!r}"
        props: Dict[str, Any] = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    return f"{path}: unexpected property {key!r}"
        for key, sub_value in value.items():
            sub_schema = props.get(key)
            if sub_schema is None:
                continue
            err = _hand_validate(sub_schema, sub_value, f"{path}.{key}")
            if err is not None:
                return err

    if expected == "array" and isinstance(value, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            return f"{path}: expected at least {min_items} items, got {len(value)}"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                err = _hand_validate(item_schema, item, f"{path}[{idx}]")
                if err is not None:
                    return err

    if expected == "string" and isinstance(value, str):
        min_len = schema.get("minLength")
        if min_len is not None and len(value) < min_len:
            return f"{path}: string shorter than minLength={min_len}"

    if expected in ("integer", "number") and isinstance(value, (int, float)) \
            and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            return f"{path}: {value} < minimum {minimum}"
        maximum = schema.get("maximum")
        if maximum is not None and value > maximum:
            return f"{path}: {value} > maximum {maximum}"

    enum = schema.get("enum")
    if enum is not None and value not in enum:
        return f"{path}: {value!r} not in enum {enum}"

    return None


def validate_tool_input(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    """Validate ``args`` against the RFC-001 input schema for ``tool_name``.

    Returns ``None`` if validation passes (or the tool is unknown — the
    caller is responsible for routing unknown tools).  Returns a
    human-readable error string when validation fails: the JSON Pointer
    of the failing field plus the expected vs received shape.

    Prefers the standard-library-grade ``jsonschema`` validator when
    available; falls back to :func:`_hand_validate`, a small
    hand-rolled subset that covers ``type`` / ``required`` /
    ``additionalProperties`` / ``minLength`` / ``minItems`` /
    ``minimum`` / ``maximum`` / ``enum`` / ``items`` / nested
    ``properties`` — sufficient for the RFC-001 schemas this module
    publishes.
    """
    entry = TOOL_CATALOG.get(tool_name)
    if entry is None:
        return None  # Unknown tool: routed elsewhere (-32601).
    input_schema, _output_schema, _description = entry
    if not isinstance(args, dict):
        return f"$: expected object, got {type(args).__name__}"

    try:
        import jsonschema  # type: ignore[import-untyped]
    except ImportError:
        return _hand_validate(input_schema, args)

    try:
        jsonschema.validate(instance=args, schema=input_schema)
    except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
        path_parts = list(exc.absolute_path)
        path = "$" + "".join(
            f"[{p}]" if isinstance(p, int) else f".{p}" for p in path_parts
        )
        return f"{path}: {exc.message}"
    except jsonschema.SchemaError as exc:  # type: ignore[attr-defined]
        # The schema itself is malformed -- this is a kernel bug, not a
        # client error.  Surface it but don't crash.
        return f"$: schema error ({exc.message})"
    return None

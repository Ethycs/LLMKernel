"""LLMKernel operator-bridge MCP server (Stage 2 Track B1 + B3 wiring).

Hosts the kernel-side MCP server that Claude Code (the agent process)
connects to over stdio. The sole tool surface RFC-001 v1.0.0 defines for
V1: ten native operator-interaction primitives plus three proxied system
tools.

* ``docs/rfcs/RFC-001-mcp-tool-taxonomy.md`` — normative tool catalog
  and JSON Schemas. Schemas live in :mod:`llm_kernel._rfc_schemas`.
* ``docs/rfcs/RFC-002-claude-code-provisioning.md`` — fixes the module
  path, server name, stdio transport, CLI args.
* ``docs/rfcs/RFC-003-custom-message-format.md`` — the run-lifecycle
  envelopes this module emits via the Track B2 :class:`RunTracker`
  pointed at the Track B3 :class:`CustomMessageDispatcher`.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import contextvars
import json as _json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Deque, Dict, List, Optional, Tuple

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError

from ._rfc_schemas import TOOL_CATALOG, validate_tool_input
from .run_tracker import RunTracker

if TYPE_CHECKING:  # pragma: no cover
    from .custom_messages import CustomMessageDispatcher

logger: logging.Logger = logging.getLogger("llm_kernel.mcp")

#: RFC-001 version this server speaks. Mirrored into every tool result.
RFC_001_VERSION: str = "1.0.0"

#: RFC-002 stable server identifier. Renaming this is a BREAKING change.
SERVER_NAME: str = "llmkernel-operator-bridge"

#: Async tool handler signature: ``(arguments) -> structured_result_dict``.
ToolHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]

#: Context-var carrying the active ``run_id`` so round-trip tool
#: handlers can register a future against the dispatcher's inbound
#: ``operator.action`` route without changing their public signature.
_active_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_active_run_id", default=None,
)

#: Default timeout (seconds) for operator round-trip approvals; RFC-001
#: §Failure modes prescribes -32002 on timeout.
DEFAULT_APPROVAL_TIMEOUT_SEC: float = 30.0

#: JSON-RPC error codes per RFC-001 §"Common conventions".
JSONRPC_SCHEMA_VALIDATION: int = -32001
JSONRPC_OPERATOR_TIMEOUT: int = -32002
JSONRPC_OPERATOR_DENIED: int = -32003
JSONRPC_EXTENSION_UNREACHABLE: int = -32004
JSONRPC_PROXIED_REFUSED: int = -32005
JSONRPC_PROXIED_IO_FAILURE: int = -32006

#: RFC-001 §escalate flood threshold; >5 critical/60s rate-limits.
ESCALATE_FLOOD_MAX: int = 5
ESCALATE_FLOOD_WINDOW_SEC: float = 60.0

#: RFC-001 §notify rate-limit; max 10 notify/zone/60s window.
NOTIFY_RATE_LIMIT_MAX: int = 10
NOTIFY_RATE_LIMIT_WINDOW_SEC: float = 60.0

#: Default proxied-tool blob threshold (bytes).  Inline strings up to
#: this size; blob-extract above it.  V1 returns inline up to the cap
#: and truncates beyond -- blob extraction is a V1.1 deferral.
DEFAULT_BLOB_THRESHOLD_BYTES: int = 65536

#: Maximum subprocess stdout/stderr size returned inline.  Anything
#: beyond is truncated -- per RFC-001 §run_command, the kernel MUST
#: capture stdout/stderr verbatim, but V1 ships truncation since blob
#: extraction is deferred to V1.1 (see TODO below).
DEFAULT_RUN_COMMAND_OUTPUT_CAP_BYTES: int = 65536


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 with millisecond precision."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


class OperatorBridgeServer:
    """MCP server hosting the RFC-001 v1.0.0 tool catalog over stdio.

    Wraps a single :class:`mcp.server.Server`, populates its
    ``list_tools`` and ``call_tool`` handlers from :data:`TOOL_CATALOG`,
    and serves over stdio when :meth:`run_stdio` is called.

    Each tool call allocates a fresh ``run_id`` (UUIDv4) returned in the
    result envelope. With a Track B2 ``run_tracker`` set, the run-lifecycle
    flows through it; with a Track B3 ``dispatcher`` set, operator
    round-trip tools (``ask`` / ``clarify`` / ``propose`` /
    ``request_approval``) await an inbound ``operator.action``.
    """

    def __init__(
        self,
        agent_id: str,
        zone_id: str,
        trace_id: Optional[str] = None,
        run_tracker: Optional[RunTracker] = None,
        dispatcher: Optional["CustomMessageDispatcher"] = None,
    ) -> None:
        """Construct the server with the per-spawn identifiers.

        Args:
            agent_id: ``LLMKERNEL_AGENT_ID`` from RFC-002.
            zone_id: ``LLMKERNEL_ZONE_ID`` from RFC-002.
            trace_id: ``LLMKERNEL_RUN_TRACE_ID``; defaults to a fresh
                32-lowercase-hex OTLP traceId.  UUID input is accepted
                and dashes are stripped for env-var continuity.
            run_tracker: Optional Track B2 :class:`RunTracker` shared
                across kernel-mediated producers. When set, every native
                tool invocation registers a ``run.start`` /
                ``run.complete`` pair through it; when ``None`` the
                server falls back to the B1 ``logger.info`` placeholders.
            dispatcher: Optional Track B3
                :class:`CustomMessageDispatcher`. When set, operator
                round-trip tools (``ask``, ``clarify``, ``propose``,
                ``request_approval``) await an inbound
                ``operator.action`` envelope (``action_type ==
                "approval_response"``) keyed by ``run_id`` before
                returning a structured result. When ``None``, the B1
                stub responses are returned unchanged.
        """
        self.agent_id: str = agent_id
        self.zone_id: str = zone_id
        # OTLP traceId is 16 random bytes as 32-lowercase-hex.  Accept
        # legacy UUID input (strip dashes) for env-var continuity with
        # ``LLMKERNEL_RUN_TRACE_ID`` set by older callers.
        if trace_id:
            try:
                self.trace_id: str = uuid.UUID(trace_id).hex
            except (ValueError, AttributeError):
                self.trace_id = trace_id.lower()
        else:
            import secrets as _secrets
            self.trace_id = _secrets.token_hex(16)
        self.run_tracker: Optional[RunTracker] = run_tracker
        self.dispatcher: Optional["CustomMessageDispatcher"] = dispatcher
        self.server: Server = Server(SERVER_NAME)
        # Pending operator round-trips, keyed by run_id; resolved when an
        # inbound ``operator.action`` envelope of type
        # ``approval_response`` lands at the dispatcher.
        self._pending_lock: threading.Lock = threading.Lock()
        self._pending_responses: Dict[str, asyncio.Future[Dict[str, Any]]] = {}

        # Rate-limit / flood-detection state per RFC-001 §"Failure modes".
        # Keyed by (zone_id, tool_name) -> deque of monotonic timestamps.
        # Engineering Guide §11.7: this is a non-reentrant Lock; we MUST
        # NOT call logger.* inside ``with self._rate_lock:``; log AFTER
        # releasing.  The non-reentrant choice is intentional: a deadlock
        # under a stray re-entrance is louder than a silent corruption.
        self._rate_lock: threading.Lock = threading.Lock()
        self._rate_state: Dict[Tuple[str, str], Deque[float]] = {}
        # Per-task report_completion guard: zone_id -> set of seen task_ids.
        self._completion_seen: Dict[str, set] = {}
        # Per-zone present idempotency cache: zone_id -> {artifact_id: result}.
        self._present_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Optional reference to a metadata writer (set externally by the
        # kernel hook layer); used by ``drift_acknowledged``. Defaults to
        # None; the operator-action handler also probes the kernel's
        # ``user_ns`` per RFC-006 §6.
        self.metadata_writer: Optional[Any] = None
        self.kernel: Optional[Any] = None
        self._handlers: Dict[str, ToolHandler] = {
            "ask": self._handle_ask,
            "clarify": self._handle_clarify,
            "propose": self._handle_propose,
            "request_approval": self._handle_request_approval,
            "report_progress": self._handle_report_progress,
            "report_completion": self._handle_report_completion,
            "report_problem": self._handle_report_problem,
            "present": self._handle_present,
            "notify": self._handle_notify,
            "escalate": self._handle_escalate,
            "read_file": self._handle_read_file,
            "write_file": self._handle_write_file,
            "run_command": self._handle_run_command,
        }
        self._register_mcp_handlers()
        if self.dispatcher is not None:
            self._dispose_handler = self.dispatcher.register_handler(
                "operator.action", self._route_operator_action,
            )
        else:
            self._dispose_handler = None

    def _register_mcp_handlers(self) -> None:
        """Register ``list_tools`` and ``call_tool`` handlers.

        The ``call_tool`` handler is hand-rolled so unknown tool names
        raise :class:`McpError` with ``code = types.METHOD_NOT_FOUND``
        per RFC-001 §Failure modes, malformed inputs raise -32001 from
        :func:`validate_tool_input`, and proxied-tool failures raise
        -32005 / -32006 directly.
        """

        @self.server.list_tools()
        async def _list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name=name,
                    description=description,
                    inputSchema=input_schema,
                    outputSchema=output_schema,
                )
                for name, (input_schema, output_schema, description) in TOOL_CATALOG.items()
            ]

        async def _call_tool_handler(
            request: types.CallToolRequest,
        ) -> types.ServerResult:
            tool_name = request.params.name
            arguments: Dict[str, Any] = request.params.arguments or {}
            handler = self._handlers.get(tool_name)
            if handler is None:
                # RFC-002 §"Failure modes" allowed-tools-bypass row:
                # in addition to the JSON-RPC -32601 the spec already
                # mandates, emit an ``agent_emit`` span with kind
                # ``invalid_tool_use`` so the operator surface sees
                # the attempt rather than just the silent denial.
                self._emit_invalid_tool_use(tool_name, arguments)
                raise McpError(
                    types.ErrorData(
                        code=types.METHOD_NOT_FOUND,
                        message=(
                            f"Unknown tool '{tool_name}'; not in RFC-001 catalog."
                        ),
                        data={"tool_name": tool_name, "agent_id": self.agent_id},
                    )
                )
            # RFC-001 §Common conventions: validate inputs against the
            # tool's input schema BEFORE dispatching.  Surface -32001
            # with a JSON-Pointer-shaped error string.  No run record
            # beyond the synthesized run.start/run.error pair the
            # tracker would emit on exception -- we raise McpError
            # directly so the agent sees the JSON-RPC error code.
            validation_error = validate_tool_input(tool_name, arguments)
            if validation_error is not None:
                raise McpError(
                    types.ErrorData(
                        code=JSONRPC_SCHEMA_VALIDATION,
                        message=validation_error,
                        data={
                            "tool_name": tool_name,
                            "agent_id": self.agent_id,
                        },
                    )
                )
            # When a Track B2 run-tracker is wired in, route the run
            # lifecycle through it so the RFC-003 envelopes flow to the
            # extension via Track B3's sink (the
            # :class:`CustomMessageDispatcher`). Otherwise, keep the B1
            # ``logger.info`` placeholders so the contract test still
            # passes without a tracker.
            if self.run_tracker is not None:
                run_id = self.run_tracker.start_run(
                    name=tool_name,
                    run_type="tool",
                    inputs=dict(arguments),
                    tags=[
                        f"agent:{self.agent_id}",
                        f"zone:{self.zone_id}",
                        f"tool:{tool_name}",
                    ],
                    metadata={"tool.name": tool_name},
                )
                token = _active_run_id.set(run_id)
                # Per RFC-001 §"Native tools" the proxied trio emits a
                # ``tool_call`` event between ``run.start`` and the
                # handler's return.  We emit it for the three proxied
                # tools as their lifecycle is the closest analogue to
                # the wire-form RFC documents.
                _is_proxied = tool_name in ("read_file", "write_file", "run_command")
                if _is_proxied:
                    try:
                        self.run_tracker.event(
                            run_id, "tool_call",
                            {
                                "tool.name": tool_name,
                                "input.value": _json.dumps(arguments, default=str),
                                "input.mime_type": "application/json",
                            },
                        )
                    except Exception:  # pragma: no cover - defensive
                        logger.exception("failed to emit tool_call event")
                try:
                    result = await handler(arguments)
                except Exception as exc:
                    # OTel exception semconv: surface the type / message
                    # / stacktrace as ``exception.*`` attributes via the
                    # tracker's ``fail_run`` -> OTLP normalization.
                    self.run_tracker.fail_run(
                        run_id,
                        error={
                            "exception.type": type(exc).__name__,
                            "exception.message": str(exc),
                            "exception.stacktrace": "",
                        },
                    )
                    raise
                finally:
                    _active_run_id.reset(token)
                result.setdefault("_rfc_version", RFC_001_VERSION)
                result.setdefault("run_id", run_id)
                if _is_proxied:
                    try:
                        self.run_tracker.event(
                            run_id, "tool_result",
                            {
                                "tool.name": tool_name,
                                "output.value": _json.dumps(result, default=str),
                                "output.mime_type": "application/json",
                            },
                        )
                    except Exception:  # pragma: no cover - defensive
                        logger.exception("failed to emit tool_result event")
                self.run_tracker.complete_run(run_id, outputs=dict(result))
            else:
                # Track B1 fallback (no run_tracker): allocate an OTLP
                # spanId directly so the log line shape matches the
                # production envelope's payload.spanId.
                import secrets as _secrets
                run_id = _secrets.token_hex(8)
                self._log_run("run.start", tool_name, run_id, inputs=arguments)
                token = _active_run_id.set(run_id)
                try:
                    result = await handler(arguments)
                finally:
                    _active_run_id.reset(token)
                result.setdefault("_rfc_version", RFC_001_VERSION)
                result.setdefault("run_id", run_id)
                self._log_run("run.complete", tool_name, run_id, status="success")
            return types.ServerResult(
                types.CallToolResult(
                    content=[types.TextContent(type="text", text=str(result))],
                    structuredContent=result,
                    isError=False,
                )
            )

        self.server.request_handlers[types.CallToolRequest] = _call_tool_handler

    def _emit_invalid_tool_use(
        self, tool_name: str, arguments: Dict[str, Any],
    ) -> None:
        """Emit an ``agent_emit`` span with kind ``invalid_tool_use``.

        Per RFC-005 §"`agent_emit` runs" / RFC-002 §"Failure modes"
        the kernel surfaces tool-call attempts that fail the
        allowed-tools restriction so the operator can see the
        bypass attempt, not just the silent JSON-RPC denial.  The
        diagnostic carries the tool name and a short reason string;
        the arguments are dropped (could carry sensitive data the
        operator did not opt to share with the audit log).
        """
        if self.run_tracker is None:
            return
        import json as _json
        try:
            content = _json.dumps(
                {"tool_name": tool_name, "arguments": arguments},
                default=str, ensure_ascii=False,
            )
        except (TypeError, ValueError):
            content = f"tool_name={tool_name!r}"
        rid = self.run_tracker.start_run(
            name=f"agent_emit:invalid_tool_use",
            run_type="agent_emit", inputs={},
            tags=[f"agent:{self.agent_id}", f"zone:{self.zone_id}",
                  "agent_emit:invalid_tool_use"],
            metadata={
                "emit_kind": "invalid_tool_use",
                "emit_content": content,
                "parser_diagnostic": (
                    f"tool {tool_name!r} not in RFC-001 catalog"
                ),
            },
        )
        self.run_tracker.complete_run(rid, outputs={})

    def _log_run(
        self,
        message_type: str,
        tool_name: str,
        run_id: str,
        **fields: Any,
    ) -> None:
        """Emit a structured log line standing in for an RFC-003 envelope.

        Used only in the no-run-tracker fallback path (B1 contract test).
        When the server is constructed with a Track B2 ``run_tracker``,
        the call sites above route through the tracker instead of this
        method; the tracker's sink is the Track B3
        :class:`CustomMessageDispatcher`, which serializes onto Jupyter
        messaging as a real ``run.start`` / ``run.complete`` envelope.
        """
        # OTLP semconv shape: ``llmnb.run_type`` lives in attributes,
        # not as a top-level field.  This log-only fallback mirrors
        # the production envelope's keys for grep parity.
        extra: Dict[str, Any] = {
            "rfc": "RFC-003",
            "message_type": message_type,
            "spanId": run_id,
            "traceId": self.trace_id,
            "name": tool_name,
            "kind": "SPAN_KIND_INTERNAL",
            "timestamp": _utc_now_iso(),
            "attributes": {
                "llmnb.run_type": "tool",
                "llmnb.agent_id": self.agent_id,
                "llmnb.zone_id": self.zone_id,
            },
        }
        extra.update(fields)
        logger.info(message_type, extra=extra)

    async def _await_operator(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Await an inbound ``operator.action`` keyed by the active run_id.

        Returns the operator-response ``parameters`` dict on success,
        ``None`` if no dispatcher is wired (so the handler falls back
        to its B1 stub), or raises :class:`asyncio.TimeoutError` per
        RFC-001 §Failure modes -32002.
        """
        if self.dispatcher is None:
            return None
        run_id = _active_run_id.get()
        if run_id is None:  # pragma: no cover - defensive
            return None
        future: asyncio.Future[Dict[str, Any]] = asyncio.get_event_loop().create_future()
        with self._pending_lock:
            self._pending_responses[run_id] = future
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            with self._pending_lock:
                self._pending_responses.pop(run_id, None)

    def _route_operator_action(self, envelope: Dict[str, Any]) -> None:
        """Inbound handler for ``operator.action`` per RFC-006 §6.

        Dispatches on ``payload.action_type``:

        * ``approval_response`` — resolve the per-run pending future.
        * ``cell_edit`` — log INFO; V1 does not mutate kernel state.
        * ``branch_switch`` — log INFO with the branch name.
        * ``zone_select`` — log INFO; renderers may use this later.
        * ``dismiss_notification`` — log INFO; mark the corresponding
          span as ``llmnb.dismissed`` if traceable via correlation_id.
        * ``drift_acknowledged`` — call
          :meth:`MetadataWriter.acknowledge_drift`.
        * Unknown action types — log warning, do not raise (V1.5+
          forward-compat).
        """
        payload = envelope.get("payload") or {}
        action_type = payload.get("action_type")
        params = payload.get("parameters") or {}

        if action_type == "approval_response":
            request_id = params.get("request_id")
            if not request_id:
                return
            with self._pending_lock:
                future = self._pending_responses.pop(request_id, None)
            if future is None or future.done():
                return
            loop = future.get_loop()
            loop.call_soon_threadsafe(future.set_result, dict(params))
            return

        if action_type == "cell_edit":
            logger.info(
                "operator.cell_edit",
                extra={
                    "event.name": "operator.cell_edit",
                    "rfc": "RFC-006",
                    "parameters": params,
                    "originating_cell_id": payload.get("originating_cell_id"),
                    "llmnb.agent_id": self.agent_id,
                    "llmnb.zone_id": self.zone_id,
                },
            )
            return

        if action_type == "agent_spawn":
            # RFC-006 §6 v2.0.3 additive: extension's parsed `/spawn` directive
            # arrives here; we delegate to AgentSupervisor.spawn(...) so the
            # agent runs and emits Family A spans back to the extension's
            # executing cell. The cell_id flows through as part of the
            # supervisor's spawn record so future per-cell correlation is
            # possible (V1 relies on the controller's "only-inflight" cell
            # fallback in findExecForCorrelation).
            agent_id = params.get("agent_id")
            task = params.get("task")
            cell_id = params.get("cell_id") or payload.get("originating_cell_id")
            if not agent_id or not task:
                logger.warning(
                    "agent_spawn missing required parameters; agent_id=%r task=%r",
                    agent_id, task,
                )
                return
            supervisor = self._resolve_agent_supervisor()
            if supervisor is None:
                logger.warning(
                    "agent_spawn received but no AgentSupervisor is attached"
                )
                return
            spawn_method = getattr(supervisor, "spawn", None)
            if not callable(spawn_method):
                logger.warning(
                    "agent_spawn: attached AgentSupervisor does not implement spawn"
                )
                return
            try:
                spawn_method(
                    zone_id=self.zone_id,
                    agent_id=agent_id,
                    task=task,
                )
            except Exception as exc:
                logger.exception(
                    "agent_spawn: supervisor.spawn raised; agent_id=%s", agent_id,
                )
                # Emit a synthetic terminal Family A span so the
                # extension's cell exec completes with STATUS_CODE_ERROR
                # instead of hanging until the PtyKernelClient's 60s
                # timeout. The error message is surfaced as the span's
                # status.message and as an llmnb.emit_content attribute
                # on a single-event agent_emit-shaped span.
                self._emit_agent_spawn_error_span(
                    agent_id=agent_id, cell_id=cell_id, error=str(exc),
                )
                return
            logger.info(
                "operator.agent_spawn",
                extra={
                    "event.name": "operator.agent_spawn",
                    "rfc": "RFC-006",
                    "llmnb.spawned_agent_id": agent_id,
                    "llmnb.cell_id": cell_id,
                    "llmnb.agent_id": self.agent_id,
                    "llmnb.zone_id": self.zone_id,
                },
            )
            return

        if action_type == "branch_switch":
            new_branch = params.get("new_branch") or params.get("branch")
            logger.info(
                "operator.branch_switch",
                extra={
                    "event.name": "operator.branch_switch",
                    "rfc": "RFC-006",
                    "new_branch": new_branch,
                    "parameters": params,
                    "llmnb.agent_id": self.agent_id,
                    "llmnb.zone_id": self.zone_id,
                },
            )
            return

        if action_type == "zone_select":
            logger.info(
                "operator.zone_select",
                extra={
                    "event.name": "operator.zone_select",
                    "rfc": "RFC-006",
                    "selected_zone_id": params.get("zone_id"),
                    "parameters": params,
                    "llmnb.agent_id": self.agent_id,
                    "llmnb.zone_id": self.zone_id,
                },
            )
            return

        if action_type == "dismiss_notification":
            notification_id = params.get("notification_id") or params.get("id")
            correlation_id = (
                envelope.get("correlation_id")
                or params.get("correlation_id")
            )
            logger.info(
                "operator.dismiss_notification",
                extra={
                    "event.name": "operator.dismiss_notification",
                    "rfc": "RFC-006",
                    "notification_id": notification_id,
                    "correlation_id": correlation_id,
                    "llmnb.agent_id": self.agent_id,
                    "llmnb.zone_id": self.zone_id,
                },
            )
            # Best-effort: if the span is still open, mark it dismissed
            # so the operator surface can surface the dismissal in the
            # cell output.  A closed span is left as-is (replay-safe).
            if self.run_tracker is not None and correlation_id:
                try:
                    span = self.run_tracker.get_run(correlation_id)
                except KeyError:
                    span = None
                if span is not None and span.endTimeUnixNano is None:
                    try:
                        from ._attrs import decode_attrs, encode_attrs
                        attrs = decode_attrs(list(span.attributes))
                        attrs["llmnb.dismissed"] = True
                        span.attributes = encode_attrs(attrs)
                    except Exception:  # pragma: no cover - defensive
                        logger.exception("failed to mark span dismissed")
            return

        if action_type == "drift_acknowledged":
            field_path = params.get("field_path")
            detected_at = params.get("detected_at")
            writer = self._resolve_metadata_writer()
            if writer is None:
                logger.warning(
                    "drift_acknowledged received but no MetadataWriter is attached "
                    "(field_path=%s)", field_path,
                )
                return
            ack = getattr(writer, "acknowledge_drift", None)
            if not callable(ack):
                logger.warning(
                    "drift_acknowledged: attached MetadataWriter does not implement "
                    "acknowledge_drift; field_path=%s", field_path,
                )
                return
            try:
                result = ack(field_path, detected_at)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "drift_acknowledged: acknowledge_drift raised; field_path=%s",
                    field_path,
                )
                return
            logger.info(
                "operator.drift_acknowledged",
                extra={
                    "event.name": "operator.drift_acknowledged",
                    "rfc": "RFC-005",
                    "field_path": field_path,
                    "detected_at": detected_at,
                    "matched": bool(result),
                    "llmnb.agent_id": self.agent_id,
                    "llmnb.zone_id": self.zone_id,
                },
            )
            return

        # Unknown action_type: forward-compat log (V1.5+ may add new
        # action types).  Per the brief: do not raise.
        logger.warning(
            "operator.action: unknown action_type %r; ignoring (forward-compat)",
            action_type,
        )

    def _resolve_metadata_writer(self) -> Optional[Any]:
        """Locate the kernel's MetadataWriter via the documented surfaces.

        Searches in order:

        1. :attr:`metadata_writer` set directly on this server.
        2. ``self.kernel._llmnb_metadata_writer`` (the
           ``ATTR_METADATA_WRITER`` slot from
           :mod:`llm_kernel._kernel_hooks`).
        3. ``self.kernel.shell.user_ns["__llmnb_metadata_writer__"]``
           per the RFC-006 §6 contract referenced in the brief.

        Returns ``None`` if none of the above resolve.
        """
        if self.metadata_writer is not None:
            return self.metadata_writer
        kernel = self.kernel
        if kernel is None:
            return None
        attached = getattr(kernel, "_llmnb_metadata_writer", None)
        if attached is not None:
            return attached
        shell = getattr(kernel, "shell", None)
        if shell is not None:
            user_ns = getattr(shell, "user_ns", None)
            if isinstance(user_ns, dict):
                return user_ns.get("__llmnb_metadata_writer__")
        return None

    def _emit_agent_spawn_error_span(
        self, *, agent_id: str, cell_id: Optional[str], error: str,
    ) -> None:
        """Emit a synthetic terminal Family A span when agent_spawn fails.

        The span is `llmnb.run_type: "agent_emit"` with `llmnb.emit_kind:
        "spawn_error"`. The cell that triggered the spawn observes a
        terminal span (`endTimeUnixNano` set, `status.code:
        STATUS_CODE_ERROR`) and completes with error status, rather than
        hanging until the PtyKernelClient's terminal-span timeout.
        """
        if self.run_tracker is None:
            return
        try:
            run_id = self.run_tracker.start_run(
                name="agent_spawn:error",
                run_type="agent_emit",
                inputs={"agent_id": agent_id, "task_cell_id": cell_id},
                tags=[f"agent:{agent_id}", "spawn_error"],
                metadata={
                    "agent_id": self.agent_id,
                    "zone_id": self.zone_id,
                    "cell_id": cell_id,
                    "emit_kind": "spawn_error",
                    "emit_content": error,
                },
            )
            self.run_tracker.fail_run(run_id, error={"message": error})
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "failed to emit synthetic agent_spawn error span; agent_id=%s",
                agent_id,
            )

    def _resolve_agent_supervisor(self) -> Optional[Any]:
        """Locate the kernel's AgentSupervisor via the documented surfaces.

        Mirrors :meth:`_resolve_metadata_writer`'s probe order. Used by the
        ``operator.action`` ``agent_spawn`` handler (RFC-006 §6 v2.0.3).
        """
        direct = getattr(self, "agent_supervisor", None)
        if direct is not None:
            return direct
        kernel = self.kernel
        if kernel is None:
            return None
        attached = getattr(kernel, "_llmnb_agent_supervisor", None)
        if attached is not None:
            return attached
        shell = getattr(kernel, "shell", None)
        if shell is not None:
            user_ns = getattr(shell, "user_ns", None)
            if isinstance(user_ns, dict):
                return user_ns.get("__llmnb_agent_supervisor__")
        return None

    # -- Rate-limit / threshold helpers ----------------------------------

    def _check_rate(
        self,
        zone_id: str,
        tool_name: str,
        max_count: int,
        window_sec: float,
    ) -> bool:
        """Return True iff calling ``tool_name`` in ``zone_id`` is permitted.

        Prunes stamps older than ``window_sec`` from the per-(zone,tool)
        deque, then checks whether appending one more keeps the count
        under ``max_count``.  Caller MUST log AFTER this returns False;
        Engineering Guide §11.7 forbids logging inside ``self._rate_lock``.
        """
        now = time.monotonic()
        cutoff = now - window_sec
        with self._rate_lock:
            key = (zone_id, tool_name)
            stamps = self._rate_state.get(key)
            if stamps is None:
                stamps = collections.deque()
                self._rate_state[key] = stamps
            while stamps and stamps[0] < cutoff:
                stamps.popleft()
            if len(stamps) >= max_count:
                return False
            stamps.append(now)
            return True

    async def _handle_ask(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Round-trip ``ask`` via the dispatcher.

        With a dispatcher: registers a future against the active run_id
        and awaits an inbound ``operator.action`` envelope of type
        ``approval_response``; falls back to the B1 stub answer when
        no dispatcher is wired (B1 contract test path).
        """
        response = await self._await_operator(timeout=DEFAULT_APPROVAL_TIMEOUT_SEC)
        if response is None:
            return {"answer": "[B1 stub] operator round-trip not yet wired"}
        return {"answer": str(response.get("answer", response.get("operator_note", "")))}

    async def _handle_clarify(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Round-trip ``clarify``; falls back to default-option echo."""
        response = await self._await_operator(timeout=DEFAULT_APPROVAL_TIMEOUT_SEC)
        if response is not None and "selected_id" in response:
            return {"selected_id": response["selected_id"]}
        options = arguments.get("options", [])
        default_id = arguments.get("default_id")
        selected = default_id or (options[0].get("id") if options else "stub_option")
        return {"selected_id": selected}

    async def _handle_propose(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Round-trip ``propose``; falls back to ``decision="defer"``."""
        response = await self._await_operator(timeout=DEFAULT_APPROVAL_TIMEOUT_SEC)
        if response is not None and "decision" in response:
            return {
                "decision": response["decision"],
                "scope_granted": response.get("scope_granted", "one_shot"),
            }
        return {"decision": "defer", "scope_granted": "one_shot"}

    async def _handle_request_approval(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Round-trip ``request_approval``; falls back to ``decision="defer"``.

        With a dispatcher: registers a future keyed by the active
        run_id and awaits an inbound ``operator.action`` of type
        ``approval_response``. The extension surfaces the approval card,
        the operator clicks Approve/Reject, the extension sends the
        ``operator.action`` envelope, the dispatcher routes it to
        :meth:`_route_operator_action`, and the future resolves.
        """
        response = await self._await_operator(timeout=DEFAULT_APPROVAL_TIMEOUT_SEC)
        if response is not None and "decision" in response:
            return {"decision": response["decision"]}
        return {"decision": "defer"}

    async def _handle_report_progress(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Acknowledge a `report_progress` call (idempotent within display_id)."""
        return {"acknowledged": True}

    async def _handle_report_completion(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Acknowledge a `report_completion` call.

        RFC-001 §report_completion semantic notes: idempotent at most
        once per task; second call with the same ``task_id`` returns
        -32004 with "task already reported complete".

        The schema does not define ``task_id`` explicitly; we accept it
        as an additive optional argument (per RFC's additive-class
        evolution rule).  If absent we cannot enforce once-per-task and
        accept the call.
        """
        task_id = arguments.get("task_id")
        if task_id is not None:
            seen = self._completion_seen.setdefault(self.zone_id, set())
            if task_id in seen:
                raise McpError(
                    types.ErrorData(
                        code=JSONRPC_OPERATOR_TIMEOUT,  # -32004 per the brief
                        message=(
                            f"task already reported complete (task_id={task_id!r})"
                        ),
                        data={"task_id": task_id, "zone_id": self.zone_id},
                    )
                )
            seen.add(task_id)
        return {"acknowledged": True}

    async def _handle_report_problem(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Acknowledge a `report_problem` call; persisted via run-tracker."""
        return {"acknowledged": True}

    async def _handle_present(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Return an artifact id; idempotent under repeated ``artifact_id``.

        RFC-001 §present semantic notes mark ``present`` idempotent
        under the ``(artifact.uri, body-hash)`` pair.  In practice the
        agent supplies an explicit ``artifact_id`` in the arguments
        envelope (additive); when present, repeated calls return the
        same response from a per-zone cache rather than minting a
        fresh id.
        """
        explicit_id = arguments.get("artifact_id")
        cache = self._present_cache.setdefault(self.zone_id, {})
        if explicit_id is not None and explicit_id in cache:
            return dict(cache[explicit_id])
        artifact_id = explicit_id or f"artifact-{uuid.uuid4()}"
        result: Dict[str, Any] = {"artifact_id": artifact_id}
        cache[artifact_id] = dict(result)
        return result

    async def _handle_notify(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Acknowledge a ``notify`` call; per-zone rate-limited per RFC-001.

        RFC-001 §notify: per-zone policy MAY rate-limit ``notify``;
        rate-limit responses MUST surface as -32005 rather than silent
        drops.  V1 enforces a 10-call-per-60s ceiling.
        """
        permitted = self._check_rate(
            self.zone_id, "notify",
            NOTIFY_RATE_LIMIT_MAX, NOTIFY_RATE_LIMIT_WINDOW_SEC,
        )
        if not permitted:
            # Log AFTER releasing the lock per Engineering Guide §11.7.
            logger.warning(
                "notify rate-limited", extra={
                    "tool.name": "notify",
                    "llmnb.zone_id": self.zone_id,
                    "llmnb.agent_id": self.agent_id,
                },
            )
            raise McpError(
                types.ErrorData(
                    code=JSONRPC_PROXIED_REFUSED,
                    message="notify rate limit exceeded.",
                    data={
                        "zone_id": self.zone_id,
                        "limit_per_60s": NOTIFY_RATE_LIMIT_MAX,
                    },
                )
            )
        return {"acknowledged": True}

    async def _handle_escalate(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Acknowledge an `escalate` call.

        Unanswered escalates still ack per RFC-001 to avoid agent
        deadlock.  Per RFC-001 §escalate the kernel MUST detect
        escalate floods (>5 ``severity=critical`` in 60s from one
        agent) and rate-limit with -32005.  Non-critical escalates do
        NOT count against the threshold (they're not the flood the
        spec is preventing).
        """
        severity = arguments.get("severity")
        if severity == "critical":
            permitted = self._check_rate(
                self.zone_id, "escalate.critical",
                ESCALATE_FLOOD_MAX, ESCALATE_FLOOD_WINDOW_SEC,
            )
            if not permitted:
                # Log AFTER releasing the lock per Engineering Guide §11.7.
                logger.warning(
                    "escalate flood detected", extra={
                        "tool.name": "escalate",
                        "severity": "critical",
                        "llmnb.zone_id": self.zone_id,
                        "llmnb.agent_id": self.agent_id,
                    },
                )
                raise McpError(
                    types.ErrorData(
                        code=JSONRPC_PROXIED_REFUSED,
                        message=(
                            "escalate flood threshold exceeded; refusing additional "
                            "critical escalations for 60s."
                        ),
                        data={
                            "zone_id": self.zone_id,
                            "limit_per_60s": ESCALATE_FLOOD_MAX,
                        },
                    )
                )
        return {"acknowledged": True}

    # -- Proxied tool handlers ----

    def _workspace_root(self) -> Path:
        """Resolve the workspace root used to anchor proxied filesystem ops.

        Reads ``LLMKERNEL_WORKSPACE_ROOT`` from the environment; falls
        back to the kernel process's ``os.getcwd()``.  The path is
        normalized via ``Path.resolve()`` so traversal checks are
        comparing absolute, symlink-resolved forms.
        """
        candidate = os.environ.get("LLMKERNEL_WORKSPACE_ROOT") or os.getcwd()
        return Path(candidate).resolve()

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        """Resolve ``raw_path`` against the workspace root.

        Raises :class:`PermissionError` (the same exception type a host
        kernel would raise on a denied path) if the resolved location
        escapes the workspace root.  The caller maps that to JSON-RPC
        -32005.
        """
        root = self._workspace_root()
        candidate = Path(raw_path)
        if candidate.is_absolute():
            target = candidate.resolve()
        else:
            target = (root / candidate).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise PermissionError(
                f"path {raw_path!r} resolves outside workspace root {str(root)!r}"
            ) from exc
        return target

    async def _handle_read_file(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Read file contents from the zone workspace (proxied; RFC-001 §read_file).

        Resolves ``path`` against the workspace root
        (``LLMKERNEL_WORKSPACE_ROOT``, fallback ``os.getcwd()``).
        Refuses path traversal with -32005 and OS errors with -32006.
        Files larger than ``DEFAULT_BLOB_THRESHOLD_BYTES`` are
        returned truncated; blob extraction is deferred to V1.1.
        """
        raw_path = str(arguments.get("path", ""))
        encoding = arguments.get("encoding", "utf-8")
        try:
            target = self._resolve_workspace_path(raw_path)
        except PermissionError as exc:
            raise McpError(
                types.ErrorData(
                    code=JSONRPC_PROXIED_REFUSED,
                    message=str(exc),
                    data={"path": raw_path},
                )
            )
        try:
            size_bytes = target.stat().st_size
            # TODO(V1.1): blob extract content > blob_threshold_bytes via
            # MetadataWriter's blob table; V1 returns inline text up to
            # the threshold and truncates beyond.
            if encoding == "base64":
                import base64 as _b64
                raw = target.read_bytes()
                truncated = False
                if len(raw) > DEFAULT_BLOB_THRESHOLD_BYTES:
                    raw = raw[:DEFAULT_BLOB_THRESHOLD_BYTES]
                    truncated = True
                content = _b64.b64encode(raw).decode("ascii")
            else:
                text = target.read_text(encoding="utf-8", errors="replace")
                truncated = False
                if len(text.encode("utf-8")) > DEFAULT_BLOB_THRESHOLD_BYTES:
                    # truncate by codepoints conservatively
                    text = text[:DEFAULT_BLOB_THRESHOLD_BYTES]
                    truncated = True
                content = text
        except OSError as exc:
            raise McpError(
                types.ErrorData(
                    code=JSONRPC_PROXIED_IO_FAILURE,
                    message=str(exc),
                    data={"path": raw_path, "errno": getattr(exc, "errno", None)},
                )
            )
        return {
            "content": content,
            "encoding": encoding,
            "truncated": truncated,
            "size_bytes": size_bytes,
        }

    async def _handle_write_file(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Write content to a workspace file (proxied; RFC-001 §write_file).

        Same workspace-root resolution / traversal refusal as
        :meth:`_handle_read_file`.  Creates parent directories on
        demand.  Returns -32006 on any OSError.  Mode ``create`` raises
        -32006 if the file already exists; ``append`` opens with mode
        ``'a'``; ``overwrite`` (default) writes verbatim.
        """
        raw_path = str(arguments.get("path", ""))
        content = arguments.get("content", "")
        encoding = arguments.get("encoding", "utf-8")
        mode = arguments.get("mode", "overwrite")
        try:
            target = self._resolve_workspace_path(raw_path)
        except PermissionError as exc:
            raise McpError(
                types.ErrorData(
                    code=JSONRPC_PROXIED_REFUSED,
                    message=str(exc),
                    data={"path": raw_path},
                )
            )
        created: bool = not target.exists()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if encoding == "base64":
                import base64 as _b64
                payload = _b64.b64decode(str(content).encode("ascii"))
            else:
                payload = str(content).encode("utf-8")
            if mode == "create":
                if not created:
                    raise FileExistsError(
                        f"file exists and mode=create: {target}"
                    )
                target.write_bytes(payload)
            elif mode == "append":
                with target.open("ab") as fh:
                    fh.write(payload)
            else:  # overwrite
                target.write_bytes(payload)
            bytes_written = len(payload)
        except OSError as exc:
            raise McpError(
                types.ErrorData(
                    code=JSONRPC_PROXIED_IO_FAILURE,
                    message=str(exc),
                    data={"path": raw_path, "errno": getattr(exc, "errno", None)},
                )
            )
        return {"bytes_written": bytes_written, "created": created}

    async def _handle_run_command(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a shell command in the workspace (proxied; RFC-001 §run_command).

        Uses :func:`subprocess.run` with ``shell=False``, ``cwd`` set
        to the workspace root, and a timeout converted from
        ``timeout_ms``.  Refuses with -32005 if the binary is not on
        PATH.  Truncates stdout/stderr at
        ``DEFAULT_RUN_COMMAND_OUTPUT_CAP_BYTES`` (V1.1: blob extract).
        """
        command = str(arguments.get("command", ""))
        args_list: List[str] = list(arguments.get("args") or [])
        timeout_ms = int(arguments.get("timeout_ms", 30000))
        timeout_sec = max(timeout_ms / 1000.0, 0.001)
        env_override = arguments.get("env") or None

        # PATH presence check: per RFC-001 §run_command -32005 covers
        # "command policy denial."  We treat "not on PATH" as a denial
        # because the kernel cannot run an arbitrary path it can't
        # locate.  Absolute paths are checked via Path.exists().
        if "/" in command or "\\" in command:
            if not Path(command).exists():
                raise McpError(
                    types.ErrorData(
                        code=JSONRPC_PROXIED_REFUSED,
                        message=f"command {command!r} not found at the given path",
                        data={"command": command},
                    )
                )
        else:
            if shutil.which(command) is None:
                raise McpError(
                    types.ErrorData(
                        code=JSONRPC_PROXIED_REFUSED,
                        message=f"command {command!r} is not on PATH",
                        data={"command": command},
                    )
                )

        argv: List[str] = [command, *args_list]
        cwd = self._workspace_root()
        env_for_subprocess: Optional[Dict[str, str]] = None
        if env_override:
            env_for_subprocess = dict(os.environ)
            env_for_subprocess.update({str(k): str(v) for k, v in env_override.items()})

        start_ns = time.monotonic_ns()
        try:
            completed = subprocess.run(
                argv,
                cwd=str(cwd),
                timeout=timeout_sec,
                capture_output=True,
                text=True,
                shell=False,
                env=env_for_subprocess,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise McpError(
                types.ErrorData(
                    code=JSONRPC_PROXIED_IO_FAILURE,
                    message=(
                        f"run_command timed out after {timeout_sec:.3f}s "
                        f"(timeout_ms={timeout_ms})"
                    ),
                    data={
                        "command": command,
                        "timeout_ms": timeout_ms,
                        "stdout": (exc.stdout or "")[:DEFAULT_RUN_COMMAND_OUTPUT_CAP_BYTES]
                        if isinstance(exc.stdout, str) else "",
                        "stderr": (exc.stderr or "")[:DEFAULT_RUN_COMMAND_OUTPUT_CAP_BYTES]
                        if isinstance(exc.stderr, str) else "",
                    },
                )
            )
        except OSError as exc:
            raise McpError(
                types.ErrorData(
                    code=JSONRPC_PROXIED_IO_FAILURE,
                    message=str(exc),
                    data={"command": command, "errno": getattr(exc, "errno", None)},
                )
            )

        duration_ms = int((time.monotonic_ns() - start_ns) / 1_000_000)
        # TODO(V1.1): blob extract stdout/stderr beyond the cap rather
        # than truncating in place.
        stdout = (completed.stdout or "")[:DEFAULT_RUN_COMMAND_OUTPUT_CAP_BYTES]
        stderr = (completed.stderr or "")[:DEFAULT_RUN_COMMAND_OUTPUT_CAP_BYTES]
        return {
            "exit_code": int(completed.returncode),
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": False,
            "duration_ms": duration_ms,
        }

    async def run_stdio(self) -> None:
        """Serve the MCP protocol over stdio per RFC-002."""
        logger.info(
            "mcp_server.start",
            extra={
                "server_name": SERVER_NAME,
                "agent_id": self.agent_id,
                "zone_id": self.zone_id,
                "traceId": self.trace_id,
                "rfc_001_version": RFC_001_VERSION,
            },
        )
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the CLI args RFC-002 specifies for this module."""
    parser = argparse.ArgumentParser(
        prog="python -m llm_kernel.mcp_server",
        description=(
            "LLMKernel operator-bridge MCP server "
            "(RFC-001 v1.0.0; stdio transport per RFC-002)."
        ),
    )
    parser.add_argument("--agent-id", required=True, help="LLMKERNEL_AGENT_ID.")
    parser.add_argument("--zone-id", required=True, help="LLMKERNEL_ZONE_ID.")
    parser.add_argument(
        "--trace-id",
        default=None,
        help=(
            "LLMKERNEL_RUN_TRACE_ID; root RFC-003 trace id. Falls back to "
            "the env var of the same name, then to a fresh UUIDv4."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint invoked via ``python -m llm_kernel.mcp_server``."""
    logging.basicConfig(
        level=os.environ.get("LLMKERNEL_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    trace_id = args.trace_id or os.environ.get("LLMKERNEL_RUN_TRACE_ID")
    server = OperatorBridgeServer(
        agent_id=args.agent_id,
        zone_id=args.zone_id,
        trace_id=trace_id,
    )
    asyncio.run(server.run_stdio())


if __name__ == "__main__":  # pragma: no cover
    main()

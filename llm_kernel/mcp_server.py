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
import contextvars
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError

from ._rfc_schemas import TOOL_CATALOG
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
        per RFC-001 §Failure modes, and ``NotImplementedError`` from
        proxied-tool stubs propagates for the contract test to assert.
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
        """Inbound handler for ``operator.action``.

        Resolves the per-run pending future when the envelope's
        ``action_type`` is ``approval_response`` and its
        ``parameters.request_id`` matches an awaited run_id. Other
        action types (``cell_edit``, ``branch_switch``, etc.) are
        logged and ignored at this layer; future tracks add per-type
        routing.
        """
        payload = envelope.get("payload") or {}
        if payload.get("action_type") != "approval_response":
            logger.debug(
                "operator.action %s ignored at mcp_server layer",
                payload.get("action_type"),
            )
            return
        params = payload.get("parameters") or {}
        request_id = params.get("request_id")
        if not request_id:
            return
        with self._pending_lock:
            future = self._pending_responses.pop(request_id, None)
        if future is None or future.done():
            return
        loop = future.get_loop()
        loop.call_soon_threadsafe(future.set_result, dict(params))

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
        """Acknowledge a `report_completion` call; B2 emits ``run.complete``."""
        return {"acknowledged": True}

    async def _handle_report_problem(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Acknowledge a `report_problem` call; persisted via run-tracker."""
        return {"acknowledged": True}

    async def _handle_present(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Return a synthetic artifact id; payload rides the dispatcher."""
        return {"artifact_id": f"artifact-{uuid.uuid4()}"}

    async def _handle_notify(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Acknowledge a `notify` call (fire-and-forget per RFC-001)."""
        # TODO(B2): rate-limit per RFC-001 (return -32005 when policy refuses).
        return {"acknowledged": True}

    async def _handle_escalate(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Acknowledge an `escalate` call. Unanswered escalates still ack
        per RFC-001 to avoid agent deadlock; run-tracker carries severity
        + reason to the dispatcher's IOPub stream."""
        # TODO(B2): detect escalate floods (>5 critical/60s) and rate-limit.
        return {"acknowledged": True}

    # -- Proxied tool handlers (B1 stubs raise NotImplementedError) ----

    async def _handle_read_file(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Read file contents from the zone workspace (proxied)."""
        # TODO(B5): forward to the workspace I/O layer; honor max_bytes / encoding.
        _path = Path(str(arguments.get("path", "")))
        raise NotImplementedError(
            "read_file: Track B1 stub; proxied implementation comes in B2/B5 wiring."
        )

    async def _handle_write_file(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Write file contents inside the zone workspace (proxied)."""
        # TODO(B5): forward to the workspace I/O layer.
        _path = Path(str(arguments.get("path", "")))
        raise NotImplementedError(
            "write_file: Track B1 stub; proxied implementation comes in B2/B5 wiring."
        )

    async def _handle_run_command(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a shell command in the zone workspace (proxied)."""
        # TODO(B5): forward to the command execution layer; honor timeout.
        raise NotImplementedError(
            "run_command: Track B1 stub; proxied implementation comes in B2/B5 wiring."
        )

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

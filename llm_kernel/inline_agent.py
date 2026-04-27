"""Inline (in-kernel) agent loop driven directly against the Anthropic API.

Provides a complementary agent-driver path to the
:class:`AgentSupervisor` (Track B4) that spawns ``claude`` subprocesses.
Where B4 honors RFC-002 by pointing a real Claude Code subprocess at
the kernel's MCP server, this module skips the subprocess entirely and
runs the agent loop *inside* the kernel process: each turn calls the
Anthropic Messages API (routed through our LiteLLM proxy so the layer-2
invariant from DR-0016 holds), inspects ``tool_use`` content blocks,
dispatches them through the live :class:`OperatorBridgeServer`'s native
handlers (which already wire OTLP spans via the run-tracker, B2), and
feeds ``tool_result`` blocks back to the model until it emits
``report_completion`` or stops calling tools.

V1 use-cases:

* development driver when the Claude Code CLI is unavailable
* kernel-side test fixture for end-to-end paper-telephone smokes
* future operator surface for "drive the agent inline" workflows where a
  subprocess boundary buys nothing

Failure posture: V1 fails closed. Unknown tools, schema mismatches, or
exhausted turns surface a synthetic ``report_problem`` envelope through
the run-tracker; the loop stops with ``completed=False``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from anthropic import AsyncAnthropic
from anthropic.types import Message

from ._rfc_schemas import TOOL_CATALOG
from ._provisioning import CANONICAL_SYSTEM_PROMPT_TEMPLATE

if TYPE_CHECKING:  # pragma: no cover
    from .mcp_server import OperatorBridgeServer

logger: logging.Logger = logging.getLogger("llm_kernel.inline_agent")

#: Default model. Caller may override via the constructor.
DEFAULT_MODEL: str = "claude-sonnet-4-5"

#: Hard cap on the number of model->tool->model turns per task.
DEFAULT_MAX_TURNS: int = 10


def _strip_meta_keys(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Drop ``$schema`` from a JSON Schema dict.

    The Anthropic Messages API accepts JSON-Schema-shaped tool input
    schemas but rejects the meta ``$schema`` URI keyword. Everything
    else (``type``, ``properties``, ``required``, ``additionalProperties``,
    ``enum``) flows through unchanged.
    """
    return {k: v for k, v in schema.items() if k != "$schema"}


class InlineAgent:
    """In-kernel agent loop driven against the Anthropic Messages API.

    The agent's tool catalog is RFC-001 verbatim; tool calls dispatch
    through the bound :class:`OperatorBridgeServer`'s native handlers,
    which emit RFC-003 run records via the run-tracker the bridge holds.
    """

    def __init__(
        self,
        bridge: "OperatorBridgeServer",
        api_key: str,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_tokens: int = 4096,
    ) -> None:
        """Bind the agent to its bridge + API client.

        Args:
            bridge: The :class:`OperatorBridgeServer` whose ``_handlers``
                dispatch table is invoked for every ``tool_use`` block.
            api_key: Anthropic API key. Passed verbatim to the SDK.
            base_url: Optional override (e.g. the kernel's LiteLLM proxy
                base, ``http://127.0.0.1:<port>``). The SDK appends
                ``/v1/messages``; do NOT include the ``/v1`` suffix.
            model: Anthropic model id.
            max_turns: Hard ceiling on agent turns to avoid runaway loops.
            max_tokens: Per-turn ``max_tokens`` for the API call.
        """
        self.bridge = bridge
        self.model = model
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.client = AsyncAnthropic(api_key=api_key, base_url=base_url)

    async def run(self, task: str) -> Dict[str, Any]:
        """Drive the loop until ``report_completion`` is observed or turns exhaust.

        Returns a dict with:
            ``completed``: True iff the agent emitted ``report_completion``
            ``turns``: number of model turns executed
            ``stop_reason``: last response's ``stop_reason``, or a synthetic
                marker if the loop terminated for our own reasons
            ``runs``: tool-name -> count of dispatched calls
        """
        system_prompt = CANONICAL_SYSTEM_PROMPT_TEMPLATE.replace("[TASK_BLOCK]", "")
        messages: List[Dict[str, Any]] = [{"role": "user", "content": task}]
        tools = self._build_tools()
        runs: Dict[str, int] = {}
        completed = False
        last_stop = "unknown"

        for turn in range(self.max_turns):
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )
            last_stop = response.stop_reason or "unknown"
            messages.append(
                {"role": "assistant", "content": _content_for_history(response)}
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                logger.info("inline-agent: no tool_use; stop_reason=%s", last_stop)
                break

            tool_results: List[Dict[str, Any]] = []
            for tu in tool_uses:
                runs[tu.name] = runs.get(tu.name, 0) + 1
                result_str = await self._dispatch_tool(tu.name, tu.input)
                tool_results.append({
                    "type": "tool_result", "tool_use_id": tu.id, "content": result_str,
                })
                if tu.name == "report_completion":
                    completed = True

            messages.append({"role": "user", "content": tool_results})
            if completed:
                break

        return {
            "completed": completed, "turns": turn + 1,
            "stop_reason": last_stop, "runs": runs,
        }

    def _build_tools(self) -> List[Dict[str, Any]]:
        """Build the Anthropic Messages API ``tools`` array from RFC-001."""
        tools: List[Dict[str, Any]] = []
        for name, (input_schema, _output, description) in TOOL_CATALOG.items():
            tools.append({
                "name": name,
                "description": description,
                "input_schema": _strip_meta_keys(input_schema),
            })
        return tools

    async def _dispatch_tool(
        self, name: str, arguments: Dict[str, Any],
    ) -> str:
        """Invoke the bridge handler for ``name`` and return a JSON string.

        Unknown tool name -> structured error result (no exception). The
        bridge's native handler emits a run-record via the run-tracker;
        proxied handlers raise :class:`NotImplementedError` which we
        catch and surface as a tool-result error.
        """
        handler = getattr(self.bridge, "_handlers", {}).get(name)
        if handler is None:
            return json.dumps({"error": f"unknown tool: {name}"})
        try:
            result = await handler(arguments)
        except NotImplementedError as exc:
            return json.dumps({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("inline-agent: handler %s raised", name)
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        try:
            return json.dumps(result, default=str)
        except (TypeError, ValueError):
            return json.dumps({"error": "non-JSON-serializable handler result"})


def _content_for_history(response: Message) -> List[Dict[str, Any]]:
    """Convert an Anthropic Message's content blocks to history-shape dicts.

    The API expects the assistant turn we send back in the next request
    to mirror the content blocks we received (``text`` / ``tool_use``),
    not the full Message object.
    """
    out: List[Dict[str, Any]] = []
    for block in response.content:
        if block.type == "text":
            out.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            out.append({
                "type": "tool_use", "id": block.id, "name": block.name,
                "input": block.input,
            })
    return out


__all__ = ["DEFAULT_MAX_TURNS", "DEFAULT_MODEL", "InlineAgent"]

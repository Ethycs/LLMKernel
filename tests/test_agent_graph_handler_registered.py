"""K-CM Family C: ``agent_graph.query`` inbound handler registration.

Per RFC-006 §5 the kernel responds to ``agent_graph.query`` with an
``agent_graph.response`` whose ``correlation_id`` matches the query's.
The dispatcher consumes K-MW's
:meth:`MetadataWriter.apply_agent_graph_command(command, parameters)`
to compute the response payload.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict
from unittest.mock import MagicMock

from llm_kernel.custom_messages import DEFAULT_COMM_TARGET, CustomMessageDispatcher


class _StubCommManager:
    def __init__(self) -> None:
        self.targets: Dict[str, Callable[..., None]] = {}

    def register_target(self, name: str, callback: Callable[..., None]) -> None:
        self.targets[name] = callback

    def unregister_target(self, name: str, callback: Callable[..., None]) -> None:
        self.targets.pop(name, None)


class _StubKernel:
    def __init__(self) -> None:
        self.session = MagicMock()
        self.iopub_socket = MagicMock(name="iopub_socket")
        self.shell = MagicMock()
        self.shell.comm_manager = _StubCommManager()
        self._parent_header: Dict[str, Any] = {}


def _new() -> tuple[CustomMessageDispatcher, _StubKernel]:
    kernel = _StubKernel()
    return CustomMessageDispatcher(kernel, heartbeat_interval_sec=300.0), kernel


def test_agent_graph_query_calls_writer_and_emits_response() -> None:
    """agent_graph.query -> apply_agent_graph_command -> agent_graph.response."""
    dispatcher, kernel = _new()
    writer = MagicMock()
    response_payload = {
        "nodes": [{"id": "agent:alpha", "type": "agent"}],
        "edges": [],
        "truncated": False,
    }
    writer.apply_agent_graph_command.return_value = response_payload
    dispatcher.set_metadata_writer(writer)

    dispatcher.start()
    try:
        comm = MagicMock(name="comm")
        kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
            comm, {"content": {}}
        )
        cid = str(uuid.uuid4())
        dispatcher._on_comm_msg(  # noqa: SLF001
            {"content": {"data": {
                "type": "agent_graph.query",
                "correlation_id": cid,
                "payload": {
                    "query_type": "neighbors",
                    "node_id": "agent:alpha",
                    "hops": 1,
                },
            }}}
        )

        writer.apply_agent_graph_command.assert_called_once()
        kwargs = writer.apply_agent_graph_command.call_args.kwargs
        assert kwargs["command"] == "neighbors"
        # The full payload (which includes query_type / node_id / hops)
        # is forwarded as ``parameters``.
        assert kwargs["parameters"]["query_type"] == "neighbors"
        assert kwargs["parameters"]["node_id"] == "agent:alpha"

        # Echo: agent_graph.response with the same correlation_id.
        sent_responses = [
            args[0][0] for args in comm.send.call_args_list
            if isinstance(args[0][0], dict)
            and args[0][0].get("type") == "agent_graph.response"
        ]
        assert len(sent_responses) == 1
        assert sent_responses[0]["correlation_id"] == cid
        assert sent_responses[0]["payload"] == response_payload
    finally:
        dispatcher.stop()


def test_agent_graph_query_handler_is_registered_via_start() -> None:
    """``start()`` MUST register the agent_graph.query handler."""
    dispatcher, _ = _new()
    dispatcher.start()
    try:
        with dispatcher._lock:  # noqa: SLF001
            handlers = dispatcher._handlers.get("agent_graph.query", [])  # noqa: SLF001
        assert handlers, "agent_graph.query handler not auto-registered"
    finally:
        dispatcher.stop()

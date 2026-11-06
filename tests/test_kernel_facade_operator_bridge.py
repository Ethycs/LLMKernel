"""The ``Kernel`` facade owns the kernel's operator-action router.

Every facade start path (serve, pty-mode, the app, library callers) must
route inbound ``operator.action`` envelopes to one
:class:`OperatorBridgeServer` wired to the facade's own supervisor and
metadata writer, exactly as ``attach_kernel_subsystems`` does for the
IPython kernel. The bridge is started and stopped with the facade, one per
kernel, and leaves nothing registered (or pending) after shutdown.

Tests wrap the async surface in ``asyncio.run`` (no pytest-asyncio).
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from llm_kernel.kernel_facade import Kernel
from llm_kernel.mcp.mcp_server import OperatorBridgeServer
from llm_kernel.wire import versions as _versions


@pytest.fixture(autouse=True)
def _dead_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLMKERNEL_LITELLM_ENDPOINT_URL", "http://127.0.0.1:1/v1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)


def _action(action_type: str, params: Dict[str, Any], cell_id: str) -> Dict[str, Any]:
    return {
        "type": "operator.action",
        "payload": {
            "action_type": action_type,
            "parameters": {**params, "cell_id": cell_id},
            "originating_cell_id": cell_id,
        },
    }


def _deliver(k: Kernel, envelope: Dict[str, Any]) -> None:
    """Inbound frame through the embedded comm, as a transport would."""
    k._embedded_kernel.shell.comm_manager.deliver(
        _versions.COMM_TARGET, {"content": {"data": envelope}},
    )


def _operator_handlers(k: Kernel) -> List[Any]:
    return list(k._dispatcher._handlers.get("operator.action", []))


def _k_replies(frames: List[Dict[str, Any]], cell_id: str) -> List[Dict[str, Any]]:
    return [
        f for f in frames
        if f.get("type") == "operator.action"
        and (f.get("payload") or {}).get("status") == "error"
        and (f.get("payload") or {}).get("cell_id") == cell_id
    ]


def test_start_builds_one_cross_wired_bridge(tmp_path: Path) -> None:
    async def main() -> None:
        k = Kernel(workdir=tmp_path)
        await k.start()
        try:
            bridge = k._operator_bridge
            assert isinstance(bridge, OperatorBridgeServer)
            assert bridge.dispatcher is k.dispatcher
            assert bridge.run_tracker is k.run_tracker
            assert bridge.agent_supervisor is k.supervisor
            assert bridge.metadata_writer is k.metadata_writer
            assert bridge.kernel is k._embedded_kernel
            handlers = _operator_handlers(k)
            assert len(handlers) == 1
            assert handlers[0] == bridge._route_operator_action
            await k.start()  # idempotent: no second router
            assert k._operator_bridge is bridge
            assert len(_operator_handlers(k)) == 1
        finally:
            await k.shutdown()

    asyncio.run(main())


def test_stop_unknown_agent_answers_k20_on_the_event_log(tmp_path: Path) -> None:
    async def main() -> None:
        async with Kernel(workdir=tmp_path) as k:
            _deliver(k, _action("agent_stop", {"agent_id": "ghost"}, "c-stop"))
            _deliver(k, _action(
                "agent_revert", {"agent_id": "ghost", "target_turn_id": "t1"}, "c-rev",
            ))
            frames = k.read_event_log()
        stop = _k_replies(frames, "c-stop")
        rev = _k_replies(frames, "c-rev")
        assert [f["payload"]["k_code"] for f in stop] == ["K20"]
        assert [f["payload"]["k_code"] for f in rev] == ["K20"]
        assert stop[0]["payload"]["action_type"] == "agent_stop"

    asyncio.run(main())


def test_failed_spawn_emits_terminal_error_span(tmp_path: Path) -> None:
    async def main() -> None:
        async with Kernel(workdir=tmp_path) as k:
            _deliver(k, _action("agent_spawn", {"agent_id": "alpha", "task": "t"}, "c-sp"))
            frames = k.read_event_log()
        ends = [
            f for f in frames
            if f.get("endTimeUnixNano")
            and (f.get("status") or {}).get("code") == "STATUS_CODE_ERROR"
        ]
        assert ends, frames

    asyncio.run(main())


def test_approval_response_resolves_a_pending_round_trip(tmp_path: Path) -> None:
    async def main() -> None:
        async with Kernel(workdir=tmp_path) as k:
            bridge = k._operator_bridge
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            with bridge._pending_lock:
                bridge._pending_responses["run-1"] = fut
            _deliver(k, {"type": "operator.action", "payload": {
                "action_type": "approval_response",
                "parameters": {"request_id": "run-1", "run_id": "run-1", "decision": "deny"},
            }})
            result = await asyncio.wait_for(fut, timeout=5.0)
            assert result["decision"] == "deny"

    asyncio.run(main())


def test_shutdown_detaches_bridge_and_cancels_pending(tmp_path: Path) -> None:
    async def main() -> None:
        k = Kernel(workdir=tmp_path)
        await k.start()
        dispatcher = k.dispatcher
        bridge = k._operator_bridge
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        with bridge._pending_lock:
            bridge._pending_responses["run-2"] = fut
        await k.shutdown()
        await asyncio.sleep(0)
        assert fut.cancelled()
        assert dispatcher._handlers.get("operator.action", []) == []
        assert k._operator_bridge is None
        bridge.close()  # idempotent

    asyncio.run(main())


def test_restart_builds_a_fresh_single_router(tmp_path: Path) -> None:
    async def main() -> None:
        k = Kernel(workdir=tmp_path)
        seen = []
        for _ in range(2):
            await k.start()
            seen.append(k._operator_bridge)
            assert len(_operator_handlers(k)) == 1
            await k.shutdown()
        assert seen[0] is not seen[1]

    before = {t.ident for t in threading.enumerate()}
    asyncio.run(main())
    leaked = [
        t.name for t in threading.enumerate()
        if t.ident not in before and t.is_alive()
    ]
    assert leaked == []


def test_failed_start_detaches_the_bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_kernel import kernel_facade as facade_mod
    from llm_kernel.kernel_facade import KernelStartError

    built: List[OperatorBridgeServer] = []
    real = facade_mod.OperatorBridgeServer

    def _spy(*args: Any, **kwargs: Any) -> OperatorBridgeServer:
        bridge = real(*args, **kwargs)
        built.append(bridge)
        return bridge

    monkeypatch.setattr(facade_mod, "OperatorBridgeServer", _spy)
    monkeypatch.setattr(
        facade_mod, "wire_magic_defense", MagicMock(side_effect=RuntimeError("boom")),
    )

    async def main() -> None:
        k = Kernel(workdir=tmp_path)
        with pytest.raises(KernelStartError):
            await k.start()
        assert len(built) == 1
        assert built[0]._dispose_handler is None
        assert built[0].dispatcher._handlers.get("operator.action", []) == []

    asyncio.run(main())


def test_legacy_ipython_hooks_still_register_exactly_one_router() -> None:
    from llm_kernel._kernel_hooks import attach_kernel_subsystems

    kernel = MagicMock()
    for attr in (
        "_llmnb_dispatcher", "_llmnb_run_tracker", "_llmnb_operator_bridge",
        "_llmnb_agent_supervisor", "_llmnb_metadata_writer",
    ):
        setattr(kernel, attr, None)
    dispatcher, _tracker, bridge, _supervisor, writer = attach_kernel_subsystems(kernel)
    try:
        handlers = dispatcher._handlers.get("operator.action", [])
        assert handlers == [bridge._route_operator_action]
    finally:
        writer.stop(emit_final=False)
        dispatcher.stop()

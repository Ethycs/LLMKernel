"""Facade commit 2 — ``Kernel`` lifecycle (start / shutdown / context mgr).

Covers PLAN-kernel-facade.md commit 2: the class skeleton and its
lifecycle. Public delegators and event-log accessors are later commits and
are not exercised here.

Tests wrap the async surface in ``asyncio.run`` so no pytest-asyncio plugin
is required. Each test fully starts+shuts down so no autosave / heartbeat
threads leak across tests.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_kernel.kernel_facade import Kernel, KernelStartError


def test_kernel_importable_from_package_root() -> None:
    """The headline API `from llm_kernel import Kernel` must resolve."""
    from llm_kernel import Kernel as ExportedKernel
    from llm_kernel import KernelStartError as ExportedError

    assert ExportedKernel is Kernel
    assert ExportedError is KernelStartError


def test_kernel_lifecycle_start_shutdown(tmp_path: Path) -> None:
    """start() wires the subsystems; shutdown() tears them down."""
    async def main() -> None:
        k = Kernel(workdir=tmp_path)
        assert not k._started
        await k.start()
        assert k._started
        # Subsystems wired (private refs — asserted here, never public API).
        assert k._dispatcher is not None
        assert k._run_tracker is not None
        assert k._metadata_writer is not None
        assert k._supervisor is not None
        assert k._trace_id is not None
        await k.shutdown()
        assert not k._started
        # Refs cleared on shutdown.
        assert k._dispatcher is None
        assert k._supervisor is None

    asyncio.run(main())


def test_kernel_lifecycle_idempotent(tmp_path: Path) -> None:
    """Double-start is a no-op (same subsystems); double-shutdown is safe."""
    async def main() -> None:
        k = Kernel(workdir=tmp_path)
        await k.start()
        dispatcher_first = k._dispatcher
        supervisor_first = k._supervisor
        await k.start()  # second start must NOT reconstruct
        assert k._dispatcher is dispatcher_first
        assert k._supervisor is supervisor_first
        assert k._started

        await k.shutdown()
        assert not k._started
        await k.shutdown()  # second shutdown must not raise
        assert not k._started

    asyncio.run(main())


def test_kernel_context_manager(tmp_path: Path) -> None:
    """`async with Kernel(...)` starts on enter and shuts down on exit."""
    async def main() -> None:
        k = Kernel(workdir=tmp_path)
        async with k as entered:
            assert entered is k
            assert k._started
            assert k._dispatcher is not None
        assert not k._started

    asyncio.run(main())


def test_kernel_context_manager_shutdown_on_exception(tmp_path: Path) -> None:
    """An exception inside the `async with` body still triggers shutdown."""
    class _Boom(Exception):
        pass

    async def main() -> None:
        k = Kernel(workdir=tmp_path)
        with pytest.raises(_Boom):
            async with k:
                assert k._started
                raise _Boom()
        # Shutdown ran despite the exception — no leaked threads.
        assert not k._started
        assert k._dispatcher is None

    asyncio.run(main())


def test_kernel_start_failure_raises_kernel_start_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A construction failure surfaces as KernelStartError (never os._exit),
    and leaves no partial subsystems started."""
    import llm_kernel.kernel_facade as facade

    class _BoomSupervisor:
        def __init__(self, *a: object, **k: object) -> None:
            raise RuntimeError("supervisor boom")

    monkeypatch.setattr(facade, "AgentSupervisor", _BoomSupervisor)

    async def main() -> None:
        k = Kernel(workdir=tmp_path)
        with pytest.raises(KernelStartError):
            await k.start()
        assert not k._started

    asyncio.run(main())


# ---------------------------------------------------------------------------
# Facade commit 3 — public delegators.
# ---------------------------------------------------------------------------


def test_kernel_delegators_require_started(tmp_path: Path) -> None:
    """Delegators raise RuntimeError before start()."""
    k = Kernel(workdir=tmp_path)
    with pytest.raises(RuntimeError):
        k.list_agents()
    with pytest.raises(RuntimeError):
        k.spawn_agent(zone_id="z", agent_id="a", task="t", work_dir=tmp_path)


def test_kernel_delegators_call_supervisor(tmp_path: Path) -> None:
    """Each delegator calls through to the supervisor with matching args."""
    async def main() -> None:
        async with Kernel(workdir=tmp_path) as k:
            sup = MagicMock()
            sup.spawn.return_value = MagicMock(agent_id="alpha")
            sup.send_user_turn.return_value = {"agent_id": "alpha", "status": "sent"}
            sup.list_agents.return_value = [
                {"agent_id": "alpha", "zone_id": "z1"},
                {"agent_id": "beta", "zone_id": "z1"},
                {"agent_id": "gamma", "zone_id": "z2"},
            ]
            k._supervisor = sup  # swap in a spy after start

            assert k.spawn_agent(
                zone_id="z1", agent_id="alpha", task="t", work_dir=tmp_path,
                model="haiku",
            ) == "alpha"
            sup.spawn.assert_called_once_with(
                zone_id="z1", agent_id="alpha", task="t",
                work_dir=tmp_path, model="haiku",
            )

            k.send_user_turn("alpha", "hi", cell_id="c1")
            sup.send_user_turn.assert_called_once_with(
                "alpha", "hi", cell_id="c1", inject_notebook_prefix=True,
            )

            k.revert("alpha", "T5")
            sup.revert.assert_called_once_with("alpha", "T5")

            k.stop("alpha")
            sup.stop.assert_called_once_with("alpha")

            assert len(k.list_agents()) == 3
            zones = k.list_zones()
            by_id = {z["zone_id"]: z for z in zones}
            assert set(by_id) == {"z1", "z2"}
            assert by_id["z1"]["agent_ids"] == ["alpha", "beta"]
            assert by_id["z2"]["agent_ids"] == ["gamma"]

    asyncio.run(main())


def test_kernel_list_agents_empty_on_fresh_kernel(tmp_path: Path) -> None:
    """A started kernel with no spawns lists no agents or zones."""
    async def main() -> None:
        async with Kernel(workdir=tmp_path) as k:
            assert k.list_agents() == []
            assert k.list_zones() == []

    asyncio.run(main())


def test_supervisor_list_agents_summary(tmp_path: Path) -> None:
    """AgentSupervisor.list_agents summarizes live handles; the facade's
    list_zones groups them."""
    from llm_kernel.supervision.agent_supervisor import AgentHandle

    async def main() -> None:
        async with Kernel(workdir=tmp_path) as k:
            popen = MagicMock()
            popen.poll = MagicMock(return_value=None)  # alive
            handle = AgentHandle(
                agent_id="alpha", zone_id="z1", popen=popen,
                started_at=1.0, work_dir=tmp_path,
                stdout_thread=threading.Thread(),
                stderr_thread=threading.Thread(),
                claude_session_id="sess-1",
            )
            k._supervisor._agents["alpha"] = handle
            try:
                agents = k.list_agents()
                assert len(agents) == 1
                a = agents[0]
                assert a["agent_id"] == "alpha"
                assert a["zone_id"] == "z1"
                assert a["claude_session_id"] == "sess-1"
                assert a["alive"] is True
                assert k.list_zones() == [
                    {"zone_id": "z1", "agent_ids": ["alpha"]}
                ]
            finally:
                # Drop the stub so shutdown's terminate_all has nothing live.
                k._supervisor._agents.clear()

    asyncio.run(main())


if __name__ == "__main__":  # pragma: no cover - library-embedding smoke
    async def _smoke() -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            async with Kernel(workdir=Path(d)) as k:
                assert k._started
        print("facade lifecycle smoke OK")

    asyncio.run(_smoke())

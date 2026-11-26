"""The ``Kernel`` facade honours its documented contract.

Sources: ``docs/04 - Reference/kernel/architecture.md`` (Lifecycle,
Construction / Shutdown sequence, Public surface, Transport protocol),
``docs/04 - Reference/kernel/embedding.md`` (async-only, event-loop
ownership, threadsafety, reading vs subscribing) and
``docs/06 - Roadmaps/PLAN-kernel-facade.md`` (public-API surface, tests).
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

import llm_kernel
from llm_kernel import kernel_facade
from llm_kernel.capture.custom_messages import CustomMessageDispatcher
from llm_kernel.capture.metadata_writer import MetadataWriter
from llm_kernel.supervision.agent_supervisor import AgentSupervisor

from tests.conformance._support import (
    RecordingShell,
    run,
    thread_names_started_since,
    wait_until,
)

Kernel = kernel_facade.Kernel
KernelStartError = kernel_facade.KernelStartError


def _metadata_triggers(frames: List[Dict[str, Any]]) -> List[str]:
    return [
        f["payload"].get("trigger") for f in frames
        if isinstance(f, dict) and f.get("type") == "notebook.metadata"
    ]


# -- import surface ---------------------------------------------------------


def test_kernel_is_importable_from_the_package_root() -> None:
    """embedding.md: ``from llm_kernel import Kernel``."""
    from llm_kernel import Kernel as RootKernel

    assert RootKernel is kernel_facade.Kernel
    assert "Kernel" in kernel_facade.__all__
    assert "KernelStartError" in kernel_facade.__all__
    assert llm_kernel.Kernel is RootKernel


def test_kernel_start_error_carries_rc() -> None:
    """architecture.md §KernelStartError: a RuntimeError with ``rc``
    (None unless one of the known boot-failure codes)."""
    err = KernelStartError("boom", rc=12)
    assert isinstance(err, RuntimeError)
    assert err.rc == 12
    assert "boom" in str(err)
    assert KernelStartError("no code").rc is None


# -- lifecycle --------------------------------------------------------------


def test_worked_example_library_smoke(tmp_path: Path) -> None:
    """embedding.md worked example / PLAN verification smoke: a started
    kernel with no transport and no agents is usable as a library."""

    async def main() -> List[Dict[str, Any]]:
        async with Kernel(workdir=tmp_path) as k:
            assert k.list_agents() == []
            assert k.list_zones() == []
            assert isinstance(k.read_event_log(), list)
            assert k.read_event_log() == k.read_event_log()  # stable snapshot
        return k.read_event_log()

    frames = run(main())
    # Leaving the block calls shutdown(): metadata snapshotted.
    assert _metadata_triggers(frames)[-1:] == ["shutdown"]


def test_context_manager_returns_self_and_shuts_down(tmp_path: Path) -> None:
    """architecture.md Lifecycle: ``__aenter__`` calls start() and returns
    self; ``__aexit__`` calls shutdown()."""
    seen: Dict[str, Any] = {}

    async def main() -> None:
        k = Kernel(workdir=tmp_path)
        async with k as entered:
            seen["same"] = entered is k
            seen["agents"] = k.list_agents()
        with pytest.raises(RuntimeError):
            k.list_agents()

    run(main())
    assert seen == {"same": True, "agents": []}


def test_context_manager_shuts_down_when_the_body_raises(tmp_path: Path) -> None:
    """PLAN tests: exception during the body still triggers shutdown."""
    holder: Dict[str, Any] = {}

    class _Boom(Exception):
        pass

    async def main() -> None:
        with pytest.raises(_Boom):
            async with Kernel(workdir=tmp_path) as k:
                holder["k"] = k
                raise _Boom()

    run(main())
    k = holder["k"]
    with pytest.raises(RuntimeError):
        k.list_agents()
    assert _metadata_triggers(k.read_event_log())[-1:] == ["shutdown"]


def test_start_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """architecture.md: a second start() while started returns immediately
    (nothing is wired twice)."""
    constructed: List[Any] = []
    original = CustomMessageDispatcher.__init__

    def _counting_init(self: Any, *args: Any, **kwargs: Any) -> None:
        constructed.append(self)
        original(self, *args, **kwargs)

    monkeypatch.setattr(CustomMessageDispatcher, "__init__", _counting_init)

    async def main() -> None:
        k = Kernel(workdir=tmp_path)
        await k.start()
        try:
            await k.start()
            assert k.list_agents() == []
        finally:
            await k.shutdown()

    run(main())
    assert len(constructed) == 1


def test_shutdown_is_idempotent_and_a_noop_when_not_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """architecture.md: shutdown() is idempotent and a no-op if not started."""
    terminations: List[float] = []
    monkeypatch.setattr(
        AgentSupervisor, "terminate_all",
        lambda self, grace_seconds=10.0: terminations.append(grace_seconds),
    )

    async def main() -> None:
        k = Kernel(workdir=tmp_path)
        await k.shutdown()  # never started
        assert terminations == []
        await k.start()
        await k.shutdown()
        await k.shutdown()

    run(main())
    assert len(terminations) == 1


def test_concurrent_shutdowns_tear_down_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """architecture.md: shutdown flips ``_started`` under the lifecycle
    lock, so a concurrent shutdown sees it False and no-ops."""
    terminations: List[float] = []
    monkeypatch.setattr(
        AgentSupervisor, "terminate_all",
        lambda self, grace_seconds=10.0: terminations.append(grace_seconds),
    )

    async def main() -> None:
        k = Kernel(workdir=tmp_path, shutdown_grace_seconds=0.5)
        await k.start()
        await asyncio.gather(k.shutdown(), k.shutdown(), k.shutdown())

    run(main())
    assert terminations == [0.5]


def test_delegators_refuse_unless_started(tmp_path: Path) -> None:
    """architecture.md Public surface: the sync delegators route through
    ``_require_started`` and raise RuntimeError when not started."""
    calls = {
        "spawn_agent": lambda k: k.spawn_agent(
            zone_id="z", agent_id="a", task="t", work_dir=tmp_path),
        "send_user_turn": lambda k: k.send_user_turn("a", "hi"),
        "revert": lambda k: k.revert("a", "t1"),
        "stop": lambda k: k.stop("a"),
        "list_agents": lambda k: k.list_agents(),
        "list_zones": lambda k: k.list_zones(),
    }

    async def main() -> None:
        k = Kernel(workdir=tmp_path)
        for name, call in calls.items():
            with pytest.raises(RuntimeError):
                call(k)
        with pytest.raises(RuntimeError):
            await k.bind_transport(object())
        with pytest.raises(RuntimeError):
            await k.run_forever()
        with pytest.raises(RuntimeError):
            async for _ in k.subscribe_events():
                pass
        await k.start()
        await k.shutdown()
        for name, call in calls.items():
            with pytest.raises(RuntimeError):
                call(k)

    run(main())


def test_delegators_forward_to_the_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PLAN tests: each public delegator calls through to the supervisor
    with matching arguments; spawn_agent returns the handle's agent_id."""
    calls: List[Any] = []

    class _Handle:
        agent_id = "alpha-from-handle"

    def _spawn(self: Any, zone_id: str, agent_id: str, task: str, work_dir: Path, **kw: Any) -> Any:
        calls.append(("spawn", zone_id, agent_id, task, work_dir, kw))
        return _Handle()

    def _send(self: Any, agent_id: str, text: str, cell_id: Any = None,
              inject_notebook_prefix: bool = True, **kw: Any) -> Dict[str, Any]:
        calls.append(("send_user_turn", agent_id, text, cell_id, inject_notebook_prefix))
        return {"ok": True}

    monkeypatch.setattr(AgentSupervisor, "spawn", _spawn)
    monkeypatch.setattr(AgentSupervisor, "send_user_turn", _send)
    monkeypatch.setattr(AgentSupervisor, "revert",
                        lambda self, a, t: calls.append(("revert", a, t)))
    monkeypatch.setattr(AgentSupervisor, "stop",
                        lambda self, a: calls.append(("stop", a)))

    async def main() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        async with Kernel(workdir=tmp_path) as k:
            out["spawn"] = k.spawn_agent(
                zone_id="zone-1", agent_id="alpha", task="do x",
                work_dir=tmp_path, model="m-1",
            )
            out["send"] = k.send_user_turn(
                "alpha", "next", cell_id="c-1", inject_notebook_prefix=False,
            )
            out["revert"] = k.revert("alpha", "turn-3")
            out["stop"] = k.stop("alpha")
        return out

    out = run(main())
    assert out == {"spawn": "alpha-from-handle", "send": {"ok": True},
                   "revert": None, "stop": None}
    assert calls == [
        ("spawn", "zone-1", "alpha", "do x", tmp_path, {"model": "m-1"}),
        ("send_user_turn", "alpha", "next", "c-1", False),
        ("revert", "alpha", "turn-3"),
        ("stop", "alpha"),
    ]


def test_list_zones_is_the_live_view_derived_from_list_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """architecture.md: ``list_zones`` is one dict per distinct zone_id
    with its live agent_ids, derived from ``list_agents``."""
    agents = [
        {"agent_id": "a1", "zone_id": "red"},
        {"agent_id": "b1", "zone_id": "blue"},
        {"agent_id": "a2", "zone_id": "red"},
    ]
    monkeypatch.setattr(AgentSupervisor, "list_agents", lambda self: [dict(a) for a in agents])

    async def main() -> Any:
        async with Kernel(workdir=tmp_path) as k:
            return k.list_agents(), k.list_zones()

    listed, zones = run(main())
    assert listed == agents
    assert sorted(zones, key=lambda z: z["zone_id"]) == [
        {"zone_id": "blue", "agent_ids": ["b1"]},
        {"zone_id": "red", "agent_ids": ["a1", "a2"]},
    ]


# -- event log --------------------------------------------------------------


def test_read_event_log_is_an_emission_ordered_slice_readable_after_shutdown(
    tmp_path: Path,
) -> None:
    """embedding.md: ``read_event_log(since, until)`` returns the
    ``[since:until]`` slice and is safe after shutdown."""

    async def main() -> Kernel:
        async with Kernel(workdir=tmp_path) as k:
            pass
        return k

    k = run(main())
    full = k.read_event_log()
    assert full, "shutdown must have captured at least the final snapshot"
    assert k.read_event_log(since=0) == full
    assert k.read_event_log(since=1) == full[1:]
    assert k.read_event_log(since=0, until=1) == full[:1]
    assert k.read_event_log(since=len(full)) == []
    assert k.read_event_log(since=-5) == full  # negative clamps to 0
    final = full[-1]
    assert final["type"] == "notebook.metadata"
    assert final["payload"]["trigger"] == "shutdown"


def test_subscribe_events_replays_then_ends_cleanly_on_shutdown(tmp_path: Path) -> None:
    """embedding.md: the subscription replays from ``since``, yields new
    frames, and ends (not cancelled) when shutdown closes the sink."""

    async def main() -> Any:
        k = Kernel(workdir=tmp_path)
        await k.start()
        seen: List[Dict[str, Any]] = []

        async def drain() -> str:
            async for frame in k.subscribe_events(since=0):
                seen.append(frame)
            return "ended"

        task = asyncio.create_task(drain())
        await asyncio.sleep(0.2)
        await k.shutdown()
        result = await asyncio.wait_for(task, timeout=10.0)
        return result, seen, k.read_event_log()

    result, seen, log = run(main())
    assert result == "ended"
    assert seen == log
    assert _metadata_triggers(seen)[-1:] == ["shutdown"]


def test_subscribe_since_skips_earlier_frames(tmp_path: Path) -> None:
    """embedding.md: ``subscribe_events(since=N)`` replays from index N."""

    async def main() -> Any:
        k = Kernel(workdir=tmp_path)
        await k.start()
        k.metadata_writer.snapshot(trigger="save")
        k.metadata_writer.snapshot(trigger="save")
        assert wait_until(lambda: len(k.read_event_log()) >= 2)
        frames: List[Dict[str, Any]] = []

        async def drain() -> None:
            async for frame in k.subscribe_events(since=1):
                frames.append(frame)

        task = asyncio.create_task(drain())
        await asyncio.sleep(0.2)
        await k.shutdown()
        await asyncio.wait_for(task, timeout=10.0)
        return frames, k.read_event_log()

    frames, log = run(main())
    assert len(log) >= 3
    assert frames == log[1:]
    assert _metadata_triggers(frames)[-1:] == ["shutdown"]


def test_restart_collects_into_a_fresh_event_log(tmp_path: Path) -> None:
    """architecture.md: start() builds a fresh sink on every call, so a
    start-after-shutdown collects into a live buffer."""

    async def main() -> Any:
        k = Kernel(workdir=tmp_path)
        await k.start()
        await k.shutdown()
        first = k.read_event_log()
        await k.start()
        second_live = k.read_event_log()
        assert k.list_agents() == []
        await k.shutdown()
        return first, second_live, k.read_event_log()

    first, second_live, second = run(main())
    assert _metadata_triggers(first)[-1:] == ["shutdown"]
    assert "shutdown" not in _metadata_triggers(second_live)
    assert _metadata_triggers(second)[-1:] == ["shutdown"]
    assert _metadata_triggers(second).count("shutdown") == 1


def test_transport_supplied_egress_keeps_the_embedded_log_empty(tmp_path: Path) -> None:
    """kernel_facade: with ``kernel_shell=`` the dispatcher emits to the
    transport, so the in-memory accessors see nothing by design."""
    shell = RecordingShell()

    async def main() -> Kernel:
        async with Kernel(workdir=tmp_path, kernel_shell=shell) as k:
            pass
        return k

    k = run(main())
    assert k.read_event_log() == []
    assert _metadata_triggers(shell.snapshot())[-1:] == ["shutdown"]


# -- transports + shutdown ordering ----------------------------------------


def test_bind_transport_then_shutdown_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """architecture.md Shutdown sequence: drain transports (await stop())
    -> terminate agents -> metadata snapshot(trigger="shutdown") + stop ->
    dispatcher stop -> close the sink (ends subscribers). ``bind(kernel)``
    is awaited immediately with the kernel itself."""
    order: List[str] = []

    monkeypatch.setattr(AgentSupervisor, "terminate_all",
                        lambda self, grace_seconds=10.0: order.append("terminate_all"))
    original_snapshot = MetadataWriter.snapshot
    original_mw_stop = MetadataWriter.stop
    original_dispatcher_stop = CustomMessageDispatcher.stop

    def _snapshot(self: Any, trigger: str = "save") -> Any:
        if trigger == "shutdown":
            order.append("snapshot:shutdown")
        return original_snapshot(self, trigger=trigger)

    def _mw_stop(self: Any, *args: Any, **kwargs: Any) -> Any:
        order.append("metadata_writer.stop")
        return original_mw_stop(self, *args, **kwargs)

    def _dispatcher_stop(self: Any, *args: Any, **kwargs: Any) -> Any:
        order.append("dispatcher.stop")
        return original_dispatcher_stop(self, *args, **kwargs)

    monkeypatch.setattr(MetadataWriter, "snapshot", _snapshot)
    monkeypatch.setattr(MetadataWriter, "stop", _mw_stop)
    monkeypatch.setattr(CustomMessageDispatcher, "stop", _dispatcher_stop)

    class _Transport:
        def __init__(self, label: str) -> None:
            self.label = label
            self.bound_to: Any = None

        async def bind(self, kernel: Any) -> None:
            self.bound_to = kernel
            order.append(f"bind:{self.label}")

        async def stop(self) -> None:
            order.append(f"stop:{self.label}")

    one, two = _Transport("one"), _Transport("two")

    async def main() -> Any:
        k = Kernel(workdir=tmp_path)
        await k.start()
        ended = asyncio.Event()

        async def drain() -> None:
            async for _ in k.subscribe_events():
                pass
            order.append("subscriber-ended")
            ended.set()

        task = asyncio.create_task(drain())
        await k.bind_transport(one)
        assert one.bound_to is k and order == ["bind:one"]
        await k.bind_transport(two)
        await asyncio.sleep(0.1)
        await k.shutdown()
        await asyncio.wait_for(task, timeout=10.0)
        return k

    k = run(main())
    assert two.bound_to is k
    stops = [o for o in order if o.startswith("stop:")]
    assert sorted(stops) == ["stop:one", "stop:two"]
    first_non_transport = next(
        i for i, o in enumerate(order)
        if not o.startswith(("bind:", "stop:"))
    )
    assert all(o.startswith(("bind:", "stop:")) for o in order[:first_non_transport])
    tail = order[first_non_transport:]
    assert tail == [
        "terminate_all", "snapshot:shutdown", "metadata_writer.stop",
        "dispatcher.stop", "subscriber-ended",
    ], order


def test_run_forever_blocks_until_cancelled(tmp_path: Path) -> None:
    """architecture.md: ``run_forever`` blocks until cancelled."""

    async def main() -> bool:
        async with Kernel(workdir=tmp_path) as k:
            task = asyncio.create_task(k.run_forever())
            await asyncio.sleep(0.2)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return k.list_agents() == []

    assert run(main()) is True


# -- failure ----------------------------------------------------------------


def test_failed_start_raises_kernel_start_error_and_tears_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """architecture.md: a failing start() runs a best-effort partial
    teardown (no autosave / heartbeat threads left) and re-raises as
    KernelStartError -- never ``os._exit``."""

    def _explode(self: Any, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("supervisor refused to construct")

    monkeypatch.setattr(AgentSupervisor, "__init__", _explode)
    before = [t.ident for t in threading.enumerate()]

    async def main() -> Kernel:
        k = Kernel(workdir=tmp_path)
        with pytest.raises(KernelStartError) as info:
            await k.start()
        assert "supervisor refused to construct" in str(info.value)
        assert isinstance(info.value.__cause__, RuntimeError)
        return k

    k = run(main())
    with pytest.raises(RuntimeError):
        k.list_agents()
    leftover = wait_until(lambda: not thread_names_started_since(before), timeout=5.0)
    assert leftover, thread_names_started_since(before)


def test_shutdown_leaves_no_subsystem_threads(tmp_path: Path) -> None:
    before = [t.ident for t in threading.enumerate()]

    async def main() -> None:
        async with Kernel(workdir=tmp_path):
            await asyncio.sleep(0.05)

    run(main())
    assert wait_until(lambda: not thread_names_started_since(before), timeout=5.0), (
        thread_names_started_since(before)
    )


# -- threadsafety -----------------------------------------------------------


def test_public_methods_are_callable_from_plain_threads(tmp_path: Path) -> None:
    """embedding.md Threadsafety: the sync delegators and read_event_log
    are threadsafe; read_event_log is safe from anywhere."""
    errors: List[BaseException] = []
    results: List[Any] = []

    async def main() -> None:
        async with Kernel(workdir=tmp_path) as k:
            def worker() -> None:
                try:
                    for _ in range(25):
                        results.append((k.list_agents(), k.list_zones(),
                                        type(k.read_event_log())))
                except BaseException as exc:  # pragma: no cover - reported below
                    errors.append(exc)

            threads = [threading.Thread(target=worker, daemon=True) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                await asyncio.get_running_loop().run_in_executor(None, t.join, 10.0)

    run(main())
    assert errors == []
    assert len(results) == 200
    assert all(r == ([], [], list) for r in results)

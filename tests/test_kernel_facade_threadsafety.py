"""Facade commit 9 — threadsafety of the Kernel facade's public surface.

The facade delegators are thin pass-throughs; their thread-safety is the
subsystems' thread-safety plus the facade holding no unsafe shared state.
The load-bearing check is that the SE-01 per-handle stdin lock still
composes when send_user_turn is driven concurrently *through the facade*.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

from llm_kernel.kernel_facade import Kernel
from llm_kernel.supervision.agent_supervisor import AgentHandle


class _SplittingStdin:
    """Records each write as two fragments with a gap between, widening the
    window for an unsynchronized second writer to interleave (see SE-01)."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.flushed: int = 0

    def write(self, data: str) -> int:
        mid = max(1, len(data) // 2)
        self.events.append(data[:mid])
        import time

        time.sleep(0.01)
        self.events.append(data[mid:])
        return len(data)

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:  # pragma: no cover - teardown only
        pass


def test_concurrent_send_user_turn_no_interleave_via_facade(tmp_path: Path) -> None:
    """Two threads calling Kernel.send_user_turn on the same agent must not
    interleave the lines of their turns — the SE-01 per-handle lock composes
    across the facade → supervisor call path."""
    async def main() -> None:
        async with Kernel(workdir=tmp_path) as k:
            split = _SplittingStdin()
            popen = MagicMock()
            popen.poll = MagicMock(return_value=None)  # alive
            popen.stdin = split
            handle = AgentHandle(
                agent_id="alpha", zone_id="z1", popen=popen,
                started_at=0.0, work_dir=tmp_path,
                stdout_thread=threading.Thread(),
                stderr_thread=threading.Thread(),
            )
            k._supervisor._agents["alpha"] = handle

            errors: list[BaseException] = []

            def send(text: str) -> None:
                try:
                    k.send_user_turn(
                        "alpha", text, cell_id=None,
                        inject_notebook_prefix=False,
                    )
                except BaseException as exc:  # pragma: no cover - surfaced below
                    errors.append(exc)

            t1 = threading.Thread(target=send, args=("AAAA-first-turn",))
            t2 = threading.Thread(target=send, args=("BBBB-second-turn",))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            try:
                assert not errors, errors
                full = "".join(split.events)
                lines = [ln for ln in full.split("\n") if ln]
                assert len(lines) == 2, f"interleaved via facade: {full!r}"
                contents = sorted(
                    json.loads(ln)["message"]["content"] for ln in lines
                )
                assert contents == sorted(["AAAA-first-turn", "BBBB-second-turn"])
            finally:
                k._supervisor._agents.clear()

    asyncio.run(main())


def test_concurrent_delegator_calls_are_threadsafe(tmp_path: Path) -> None:
    """Hammering the facade delegators from many threads reaches the
    supervisor for every call with no exceptions (the facade holds no
    unsafe shared state of its own)."""
    async def main() -> None:
        async with Kernel(workdir=tmp_path) as k:
            sup = MagicMock()
            sup.list_agents.return_value = []
            k._supervisor = sup

            errors: list[BaseException] = []
            n = 40

            def worker(i: int) -> None:
                try:
                    k.list_agents()
                    k.send_user_turn(f"a{i}", "hi")
                    k.revert(f"a{i}", "T1")
                    k.stop(f"a{i}")
                except BaseException as exc:  # pragma: no cover
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, errors
            assert sup.list_agents.call_count == n
            assert sup.send_user_turn.call_count == n
            assert sup.revert.call_count == n
            assert sup.stop.call_count == n

    asyncio.run(main())

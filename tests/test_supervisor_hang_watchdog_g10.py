"""K-AS-B / G10 — 120s stdout-silence hang watchdog (acceptance tests).

The G10 audit found the hang-watchdog is already implemented end-to-end
in :mod:`llm_kernel.agent_supervisor` (see ``_silence_watchdog``,
``DEFAULT_AGENT_SILENCE_THRESHOLD_SEC = 120.0``, ``_last_stdout_ts``
bumped per stdout chunk in ``_read_stdout``, ``_hang_terminated`` flag
threading the regular crash-restart watchdog). Existing unit tests in
``test_hang_watchdog.py`` cover the SIGTERM + log-event paths.

These acceptance tests re-pin the contract surface that the
``contracts/agent-supervisor.md`` "Silence watchdog" invariant promises:

* No kill while stdout is active.
* Kill after the configured silence threshold elapses.
* Hang marker observable on the supervisor's data plane (``ERROR``-
  level ``agent.hang_detected`` with ``llmnb.silence_seconds``).
* Watchdog is a no-op for an already-exited (idle) agent — does NOT
  attempt SIGTERM after ``popen.returncode`` is set.

Acceptance tests use a small threshold (and granularity) to exercise
the contract in milliseconds. Threading discipline (Engineering Guide
§9): every helper Popen yields control to ``threading.Event`` so the
test never blocks on a real process.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from llm_kernel.agent_supervisor import AgentSupervisor


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StallPopen:
    """Popen that produces zero stdout and never exits until terminated."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self.returncode: Optional[int] = None
        self.stdout = self._stdout_iter()
        self.stderr = iter([""])
        self._terminated_at: Optional[float] = None
        self.pid: int = 80001

    def _stdout_iter(self):  # type: ignore[no-untyped-def]
        self._stop.wait()
        if False:  # pragma: no cover
            yield ""
        return

    def poll(self) -> Optional[int]:
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        self._stop.wait(timeout=timeout)
        if self.returncode is None:
            self.returncode = -15
        return self.returncode

    def terminate(self) -> None:
        self._terminated_at = time.monotonic()
        self.returncode = -15
        self._stop.set()

    def kill(self) -> None:  # pragma: no cover
        self.returncode = -9
        self._stop.set()


class _ChattyPopen:
    """Popen whose stdout emits a JSON line every ``emit_interval`` seconds."""

    def __init__(self, emit_interval: float, total_emits: int = 50) -> None:
        self._stop = threading.Event()
        self.returncode: Optional[int] = None
        self._emit_interval = emit_interval
        self._total_emits = total_emits
        self.stdout = self._stdout_iter()
        self.stderr = iter([""])
        self._terminated_at: Optional[float] = None
        self.pid: int = 80002

    def _stdout_iter(self):  # type: ignore[no-untyped-def]
        for i in range(self._total_emits):
            if self._stop.wait(timeout=self._emit_interval):
                return
            # A non-empty line resets the silence timer; the contents
            # are not parsed (DR-0010 violation path is fine, since
            # `_read_stdout` bumps `_last_stdout_ts` BEFORE the parse).
            yield f'{{"type":"ping","seq":{i}}}\n'

    def poll(self) -> Optional[int]:
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        self._stop.wait(timeout=timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self._terminated_at = time.monotonic()
        self.returncode = -15
        self._stop.set()

    def kill(self) -> None:  # pragma: no cover
        self.returncode = -9
        self._stop.set()


class _ExitedPopen:
    """Popen that has already exited cleanly before the watchdog wakes."""

    def __init__(self) -> None:
        self.returncode: int = 0
        self.stdout = iter([""])
        self.stderr = iter([""])
        self._terminated_at: Optional[float] = None
        self.pid: int = 80003

    def poll(self) -> Optional[int]:
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        return self.returncode

    def terminate(self) -> None:
        self._terminated_at = time.monotonic()

    def kill(self) -> None:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_health(status_code: int = 200):
    fake = MagicMock()
    fake.status_code = status_code
    return patch("llm_kernel._provisioning.httpx.head", return_value=fake)


def _make_supervisor(threshold: float, granularity: float) -> AgentSupervisor:
    from llm_kernel.run_tracker import RunTracker

    class _ListSink:
        def __init__(self) -> None:
            self.envelopes: List[Dict[str, Any]] = []

        def emit(self, env: Dict[str, Any]) -> None:
            self.envelopes.append(env)

    tracker = RunTracker(
        trace_id=str(uuid.uuid4()), sink=_ListSink(),
        agent_id="alpha", zone_id="z1",
    )
    dispatcher = MagicMock()
    return AgentSupervisor(
        run_tracker=tracker, dispatcher=dispatcher,
        litellm_endpoint_url="http://127.0.0.1:9999/v1",
        agent_silence_threshold_seconds=threshold,
        silence_watchdog_granularity_seconds=granularity,
    )


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------


def test_no_kill_when_stdout_active(tmp_path: Path) -> None:
    """An agent emitting stdout faster than the silence threshold MUST live.

    Contract: the silence timer is reset by every non-empty stdout
    chunk (``_read_stdout`` bumps ``handle._last_stdout_ts`` BEFORE
    parser dispatch, per RFC-002 §"Failure modes" Hang row). With
    threshold=300ms and emit_interval=50ms, the watchdog must never
    observe silence > threshold.
    """
    sup = _make_supervisor(threshold=0.3, granularity=0.05)
    fake = _ChattyPopen(emit_interval=0.05, total_emits=30)
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="x",
            work_dir=tmp_path, api_key="sk-x",
        )
    # Run for well over the threshold; chatty popen keeps resetting.
    time.sleep(0.6)
    assert fake._terminated_at is None, (
        "silence watchdog SIGTERMed an actively-emitting agent"
    )
    assert not handle._hang_terminated
    fake._stop.set()
    handle.terminate(grace_seconds=0.1)


def test_kill_after_120s_silence(tmp_path: Path) -> None:
    """Silent agent past the threshold MUST receive SIGTERM.

    Contract: ``_silence_watchdog`` calls ``handle.popen.terminate()``
    after ``time.monotonic() - _last_stdout_ts > threshold`` AND the
    process is still running. We use threshold=200ms as a stand-in for
    the 120s constant; the production default
    ``DEFAULT_AGENT_SILENCE_THRESHOLD_SEC = 120.0`` is asserted at the
    module level.
    """
    from llm_kernel.agent_supervisor import DEFAULT_AGENT_SILENCE_THRESHOLD_SEC

    # Pin the canonical 120s default so a future RFC bump triggers the
    # contract test rather than silently shipping.
    assert DEFAULT_AGENT_SILENCE_THRESHOLD_SEC == 120.0

    sup = _make_supervisor(threshold=0.2, granularity=0.05)
    fake = _StallPopen()
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="x",
            work_dir=tmp_path, api_key="sk-x",
        )
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            if fake._terminated_at is not None:
                break
            time.sleep(0.025)
        assert fake._terminated_at is not None, (
            "silence watchdog did not SIGTERM past the silence threshold"
        )
        # Stop the regular watchdog from re-entering Popen during teardown.
        handle._stop_event.set()
        handle.terminate(grace_seconds=0.1)


def test_marker_emitted_on_hang_kill(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Hang kill MUST emit the ``agent.hang_detected`` marker (ERROR).

    Contract (RFC-002 §"Failure modes" Hang row + the
    ``agent-supervisor.md`` invariant): one ERROR-level log record
    with ``event.name=agent.hang_detected``, attached
    ``llmnb.agent_id`` and ``llmnb.silence_seconds``. Until a K-class
    code is allocated in BSP-002 §7 for hang-restart bookkeeping, the
    log signature is the durable contract surface.
    """
    sup = _make_supervisor(threshold=0.2, granularity=0.05)
    fake = _StallPopen()
    with caplog.at_level("ERROR", logger="llm_kernel.agent_supervisor"):
        with _patch_health(), patch("subprocess.Popen", return_value=fake):
            handle = sup.spawn(
                zone_id="z1", agent_id="alpha", task="x",
                work_dir=tmp_path, api_key="sk-x",
            )
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                if fake._terminated_at is not None:
                    break
                time.sleep(0.025)
            handle._stop_event.set()
            handle.terminate(grace_seconds=0.1)
    matches = [
        r for r in caplog.records
        if "agent.hang_detected" in r.getMessage() and r.levelname == "ERROR"
    ]
    assert matches, (
        f"agent.hang_detected ERROR record not emitted; "
        f"got {[r.getMessage() for r in caplog.records]!r}"
    )
    rec = matches[0]
    assert getattr(rec, "llmnb.agent_id", None) == "alpha"
    silence_attr = getattr(rec, "llmnb.silence_seconds", None)
    assert silence_attr is not None and silence_attr >= 0.2


def test_watchdog_no_op_for_idle_agent() -> None:
    """If popen has already exited, the watchdog MUST NOT call terminate.

    Contract: ``_silence_watchdog`` checks ``handle.popen.returncode is
    not None`` at the top of every loop iteration and returns. An idle
    agent (clean exit; awaiting --resume) MUST NOT receive a stray
    SIGTERM that would race the regular crash-restart bookkeeping.

    We exercise the watchdog directly (no spawn(): an exited popen
    short-circuits the whole spawn pipeline). With ``returncode=0`` set
    BEFORE the loop wakes, the very first iteration must return.
    """
    sup = _make_supervisor(threshold=10.0, granularity=0.05)
    fake = _ExitedPopen()

    from llm_kernel.agent_supervisor import AgentHandle

    handle = AgentHandle(
        agent_id="alpha", zone_id="z1", popen=fake,  # type: ignore[arg-type]
        started_at=time.monotonic(),
        work_dir=Path("/tmp/x"),
        stdout_thread=threading.Thread(),
        stderr_thread=threading.Thread(),
        _last_stdout_ts=0.0,  # ancient → would trip silence if alive
    )
    # Run the watchdog in a thread so a regression (terminate() called)
    # is observable without hanging the test.
    t = threading.Thread(
        target=sup._silence_watchdog, args=(handle,),
        name="test-silence-noop", daemon=True,
    )
    t.start()
    t.join(timeout=0.5)
    assert not t.is_alive(), "silence watchdog did not return for an exited popen"
    assert fake._terminated_at is None, (
        "silence watchdog SIGTERMed an already-exited agent"
    )
    assert not handle._hang_terminated

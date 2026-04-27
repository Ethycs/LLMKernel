"""K-AS G10 — 120s hang watchdog (RFC-002 §"Failure modes" Hang row).

The supervisor's silence watchdog SIGTERMs an agent whose stdout has
been silent for longer than ``agent_silence_threshold_seconds``. These
tests use a small threshold (and small granularity) to exercise the
behavior in milliseconds rather than minutes.

Threading discipline (Engineering Guide §9): the watchdog is daemon=True
so test processes never hang on it; tests join the thread with a short
timeout to keep failures visible rather than silent.
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


class _StallPopen:
    """Popen substitute that produces zero stdout and never exits.

    The stdout iterator blocks on an Event so the reader thread is
    parked in :meth:`AgentSupervisor._read_stdout` waiting forever.
    The silence watchdog SHOULD observe stdout silence and call
    :meth:`terminate`, which wakes the iterator and lets the regular
    crash-restart watchdog observe a non-zero exit.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self.returncode: Optional[int] = None
        self.stdout = self._stdout_iter()
        self.stderr = iter([""])
        self._terminated_at: Optional[float] = None

    def _stdout_iter(self):  # type: ignore[no-untyped-def]
        # Block until terminate() unblocks us, then return EOF.
        self._stop.wait()
        if False:  # pragma: no cover - ensures generator semantics
            yield ""
        return

    def poll(self) -> Optional[int]:
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        self._stop.wait(timeout=timeout)
        if self.returncode is None:
            self.returncode = -15  # SIGTERM-ish
        return self.returncode

    def terminate(self) -> None:
        self._terminated_at = time.monotonic()
        self.returncode = -15
        self._stop.set()

    def kill(self) -> None:  # pragma: no cover - not exercised
        self.returncode = -9
        self._stop.set()


def _patch_health(status_code: int = 200):
    fake = MagicMock()
    fake.status_code = status_code
    return patch("llm_kernel._provisioning.httpx.head", return_value=fake)


def _make_supervisor(threshold: float, granularity: float) -> AgentSupervisor:
    """Build a supervisor with stub run-tracker + dispatcher."""
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


def test_silence_watchdog_terminates_after_threshold(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Stalled agent (no stdout) MUST be SIGTERMed past the silence threshold."""
    sup = _make_supervisor(threshold=0.2, granularity=0.05)
    fake = _StallPopen()
    # Keep the Popen patch alive for the duration of the test so the
    # crash-restart watchdog (which fires after the silence watchdog
    # SIGTERMs) does not re-enter real subprocess.Popen.
    with caplog.at_level("ERROR", logger="llm_kernel.agent_supervisor"):
        with _patch_health(), patch("subprocess.Popen", return_value=fake):
            handle = sup.spawn(
                zone_id="z1", agent_id="alpha", task="x",
                work_dir=tmp_path, api_key="sk-x",
            )
            # Wait for the silence watchdog to fire. Threshold 200ms;
            # allow 1.5s.
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                if fake._terminated_at is not None:
                    break
                time.sleep(0.025)
            assert fake._terminated_at is not None, (
                "silence watchdog did not SIGTERM"
            )
            # Stop the watchdog from chasing this handle into restart
            # logic so the test does not race with respawn.
            handle._stop_event.set()
            handle.terminate(grace_seconds=0.1)
    # Hang event MUST be logged at ERROR severity. The transient
    # ``_hang_terminated`` flag is cleared by the crash-restart watchdog
    # before this assertion can race; assert the durable log instead.
    matches = [r for r in caplog.records if "agent.hang_detected" in r.getMessage()]
    assert matches, [r.getMessage() for r in caplog.records]


def test_silence_watchdog_does_not_terminate_active_agent(tmp_path: Path) -> None:
    """Agent emitting stdout regularly MUST NOT be terminated."""
    # Build a popen whose stdout produces a line then waits past the
    # threshold has elapsed — the line should reset _last_stdout_ts.
    class _ActivePopen:
        def __init__(self) -> None:
            self._stop = threading.Event()
            self.returncode: Optional[int] = None
            self._lines = iter(['{"type":"system","subtype":"init"}\n', ''])
            self.stdout = self._lines
            self.stderr = iter([""])
            self._terminated_at: Optional[float] = None

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

        def kill(self) -> None:
            self.returncode = -9
            self._stop.set()

    sup = _make_supervisor(threshold=2.0, granularity=0.05)
    fake = _ActivePopen()
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="x",
            work_dir=tmp_path, api_key="sk-x",
        )
    # Wait briefly — way under the threshold — then check.
    time.sleep(0.3)
    assert fake._terminated_at is None
    assert not handle._hang_terminated
    fake._stop.set()
    handle.terminate(grace_seconds=0.1)


def test_silence_watchdog_logs_event_with_attributes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Hang detection MUST log ``agent.hang_detected`` at ERROR level."""
    sup = _make_supervisor(threshold=0.2, granularity=0.05)
    fake = _StallPopen()
    with caplog.at_level("ERROR", logger="llm_kernel.agent_supervisor"):
        with _patch_health(), patch("subprocess.Popen", return_value=fake):
            handle = sup.spawn(
                zone_id="z1", agent_id="alpha", task="x",
                work_dir=tmp_path, api_key="sk-x",
            )
            # Wait for fire.
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                if fake._terminated_at is not None:
                    break
                time.sleep(0.025)
            handle._stop_event.set()
            handle.terminate(grace_seconds=0.1)
    # Assert one ERROR record with the hang-detected log signature.
    matches = [r for r in caplog.records if "agent.hang_detected" in r.getMessage()]
    assert matches, [r.getMessage() for r in caplog.records]
    rec = matches[0]
    assert rec.levelname == "ERROR"
    # Structured attributes attached via ``extra``.
    assert getattr(rec, "llmnb.agent_id", None) == "alpha"
    silence = getattr(rec, "llmnb.silence_seconds", None)
    assert silence is not None and silence >= 0.2


def test_silence_watchdog_thread_is_daemon(tmp_path: Path) -> None:
    """Per Engineering Guide §9, the watchdog MUST be daemon=True."""
    sup = _make_supervisor(threshold=10.0, granularity=0.5)
    fake = _StallPopen()
    with _patch_health(), patch("subprocess.Popen", return_value=fake):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="x",
            work_dir=tmp_path, api_key="sk-x",
        )
    assert handle.silence_watchdog_thread is not None
    assert handle.silence_watchdog_thread.daemon is True
    fake.terminate()
    handle.terminate(grace_seconds=0.1)

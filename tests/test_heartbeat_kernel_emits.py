"""K-CM Family E: ``heartbeat.kernel`` emitter (RFC-006 §7 v2.0.2).

The kernel MUST emit ``heartbeat.kernel`` every 5 seconds to keep the
operator-facing kernel-state badge fresh.  Tests use a small cadence
override (100 ms) via the dispatcher constructor so we can verify ≥2
heartbeats in well under a second instead of 11 s.

The heartbeat thread is daemon=True; tests still call
``dispatcher.stop()`` to keep logger / handler state clean across tests.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock

from llm_kernel.custom_messages import (
    DEFAULT_COMM_TARGET, DEFAULT_HEARTBEAT_INTERVAL_SEC,
    CustomMessageDispatcher,
)


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


def test_default_cadence_is_five_seconds() -> None:
    """RFC-006 §7 v2.0.2: production cadence is 5s."""
    assert DEFAULT_HEARTBEAT_INTERVAL_SEC == 5.0


def test_heartbeat_loop_emits_at_least_twice() -> None:
    """≥2 heartbeats in ~3 cadences (300 ms here)."""
    kernel = _StubKernel()
    dispatcher = CustomMessageDispatcher(
        kernel, heartbeat_interval_sec=0.1,
    )
    sent: List[Dict[str, Any]] = []
    sent_lock = threading.Lock()

    comm = MagicMock(name="comm")

    def _capture(thin: Dict[str, Any]) -> None:
        with sent_lock:
            sent.append(thin)
    comm.send.side_effect = _capture

    dispatcher.start()
    try:
        # Attach the Comm so heartbeat envelopes flow through the live
        # send path rather than the pre-attach buffer.
        kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
            comm, {"content": {}}
        )
        # 5 cadences -> 5 heartbeats expected; assert ≥2.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with sent_lock:
                if (
                    sum(1 for s in sent if s.get("type") == "heartbeat.kernel") >= 2
                ):
                    break
            time.sleep(0.02)
        with sent_lock:
            heartbeats = [
                s for s in sent if s.get("type") == "heartbeat.kernel"
            ]
        assert len(heartbeats) >= 2, f"expected ≥2 heartbeats, got {len(heartbeats)}"
    finally:
        dispatcher.stop()


def test_heartbeat_payload_carries_kernel_state_uptime_and_last_run() -> None:
    """Every heartbeat carries kernel_state, uptime_seconds, last_run_timestamp."""
    kernel = _StubKernel()
    dispatcher = CustomMessageDispatcher(
        kernel, heartbeat_interval_sec=0.05,
    )
    sent: List[Dict[str, Any]] = []
    sent_lock = threading.Lock()

    comm = MagicMock(name="comm")

    def _capture(thin: Dict[str, Any]) -> None:
        with sent_lock:
            sent.append(thin)
    comm.send.side_effect = _capture

    dispatcher.start()
    try:
        kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
            comm, {"content": {}}
        )
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline:
            with sent_lock:
                if any(s.get("type") == "heartbeat.kernel" for s in sent):
                    break
            time.sleep(0.02)
        with sent_lock:
            heartbeat = next(
                (s for s in sent if s.get("type") == "heartbeat.kernel"),
                None,
            )
        assert heartbeat is not None, "no heartbeat envelope captured"
        payload = heartbeat["payload"]
        assert payload["kernel_state"] in {"ok", "starting", "shutting_down"}
        # Uptime is a non-negative float.
        assert isinstance(payload["uptime_seconds"], (int, float))
        assert payload["uptime_seconds"] >= 0
        # last_run_timestamp is None until a run.complete is emitted.
        assert payload["last_run_timestamp"] is None
    finally:
        dispatcher.stop()


def test_heartbeat_thread_exits_on_stop() -> None:
    """``stop()`` MUST join the heartbeat thread."""
    kernel = _StubKernel()
    dispatcher = CustomMessageDispatcher(
        kernel, heartbeat_interval_sec=0.05,
    )
    dispatcher.start()
    # Capture the thread reference before stop nulls it.
    thread = dispatcher._heartbeat_thread  # noqa: SLF001
    assert thread is not None and thread.is_alive()
    dispatcher.stop()
    # Joined; the thread is no longer alive.
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "heartbeat thread did not exit on stop()"


def test_kernel_state_reflects_shutdown_event() -> None:
    """``shutdown_event`` set -> kernel_state == "shutting_down"."""
    kernel = _StubKernel()
    dispatcher = CustomMessageDispatcher(
        kernel, heartbeat_interval_sec=0.05,
    )
    sent: List[Dict[str, Any]] = []
    sent_lock = threading.Lock()
    comm = MagicMock(name="comm")

    def _capture(thin: Dict[str, Any]) -> None:
        with sent_lock:
            sent.append(thin)
    comm.send.side_effect = _capture

    shutdown_event = threading.Event()
    dispatcher.set_shutdown_event(shutdown_event)
    dispatcher.start()
    try:
        kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
            comm, {"content": {}}
        )
        shutdown_event.set()
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline:
            with sent_lock:
                shutting = [
                    s for s in sent
                    if s.get("type") == "heartbeat.kernel"
                    and s.get("payload", {}).get("kernel_state") == "shutting_down"
                ]
                if shutting:
                    break
            time.sleep(0.02)
        with sent_lock:
            shutting = [
                s for s in sent
                if s.get("type") == "heartbeat.kernel"
                and s.get("payload", {}).get("kernel_state") == "shutting_down"
            ]
        assert shutting, "expected at least one shutting_down heartbeat"
    finally:
        # Reset so cleanup is graceful.
        dispatcher.stop()

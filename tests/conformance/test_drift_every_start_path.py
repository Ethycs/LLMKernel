"""Drift detection is alive in every start path, measured against one source.

Sources: RFC-005 §"Resume-time RFC version check" (same major + minor
differs -> drift entry; different major -> drift entry, error), RFC-005
§"metadata.rts.drift_log" (event schema; events are never removed),
``docs/04 - Reference/atoms/contracts/drift-detector.md`` (severity
classification, invariants, hydrate caller), RFC-006 §8/§9 (hydrate ->
``mode:"snapshot"`` confirmation with ``trigger:"hydrate_complete"``;
hydrate exclusivity) and ``llm_kernel/wire/versions.py`` (every start
path -- pty-mode, serve, the app, the Kernel facade, the legacy IPython
kernel -- measures drift against ``versions.current_volatile``, read at
call time).

Each start path gets ONE hydrate carrying a persisted notebook whose
``rfc_006_version`` has a different major and whose ``rfc_001_version``
has a different minor than the running kernel; every other key matches.
"""

from __future__ import annotations

import asyncio
import copy
import signal
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from llm_kernel.capture import drift_detector as dd
from llm_kernel.kernel_facade import Kernel
from llm_kernel.wire import versions as _versions

from tests.conformance._support import (
    WAIT_SECONDS,
    RecordingShell,
    WirePeer,
    bump,
    connect_with_retry,
    current_kernel_block,
    free_port,
    handshake_request,
    hydrate_complete_snapshot,
    hydrate_envelope,
    is_hydrate_complete,
    kernel_drift,
    kernel_subprocess_env,
    run,
    spawn_kernel,
    stop_process,
    wait_until,
)


def _persisted_block() -> Dict[str, str]:
    block = current_kernel_block()
    block["rfc_006_version"] = bump(_versions.RFC_VERSIONS["rfc_006_version"], "major")
    block["rfc_001_version"] = bump(_versions.RFC_VERSIONS["rfc_001_version"], "minor")
    return block


def _assert_rfc_drift(snapshot: Optional[Dict[str, Any]], persisted: Dict[str, str]) -> None:
    assert snapshot is not None, "no hydrate_complete confirmation was emitted"
    drift = kernel_drift(snapshot)
    assert set(drift) == {"rfc_006_version", "rfc_001_version"}, drift
    major, minor = drift["rfc_006_version"], drift["rfc_001_version"]
    assert major["severity"] == dd.SEVERITY_ERROR
    assert major["previous_value"] == persisted["rfc_006_version"]
    assert major["current_value"] == _versions.RFC_VERSIONS["rfc_006_version"]
    assert minor["severity"] == dd.SEVERITY_WARN
    assert minor["previous_value"] == persisted["rfc_001_version"]
    assert minor["current_value"] == _versions.RFC_VERSIONS["rfc_001_version"]
    for event in (major, minor):
        # RFC-005 drift event schema.
        assert {"detected_at", "field_path", "previous_value", "current_value",
                "severity", "operator_acknowledged"} <= set(event)
        assert event["operator_acknowledged"] is False


def _facade_hydrate(tmp_path: Path, frame: Dict[str, Any], *,
                    shell: Optional[RecordingShell] = None) -> Optional[Dict[str, Any]]:
    async def main() -> Optional[Dict[str, Any]]:
        async with Kernel(workdir=tmp_path, kernel_shell=shell) as k:
            if shell is None:
                k.dispatcher._on_comm_msg({"content": {"data": frame}})  # noqa: SLF001
                source: Callable[[], List[Dict[str, Any]]] = k.read_event_log
            else:
                shell.deliver(frame)
                source = shell.snapshot
            return wait_until(lambda: hydrate_complete_snapshot(source()))

    return run(main())


# -- one hydrate per start path --------------------------------------------


def test_facade_embedded(tmp_path: Path) -> None:
    persisted = _persisted_block()
    _assert_rfc_drift(_facade_hydrate(tmp_path, hydrate_envelope(persisted)), persisted)


def test_facade_with_transport_egress(tmp_path: Path) -> None:
    persisted = _persisted_block()
    shell = RecordingShell()
    snap = _facade_hydrate(tmp_path, hydrate_envelope(persisted), shell=shell)
    _assert_rfc_drift(snap, persisted)


def test_legacy_ipython_hooks(tmp_path: Path) -> None:
    """The legacy IPython kernel's ``attach_kernel_subsystems`` path."""
    from llm_kernel._kernel_hooks import attach_kernel_subsystems

    shell = RecordingShell()
    dispatcher, _tracker, _bridge, _supervisor, writer = attach_kernel_subsystems(shell)
    try:
        persisted = _persisted_block()
        shell.deliver(hydrate_envelope(persisted))
        snap = wait_until(lambda: hydrate_complete_snapshot(shell.snapshot()))
    finally:
        writer.stop(emit_final=False)
        dispatcher.stop()
    _assert_rfc_drift(snap, persisted)


def test_pty_mode_main_in_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from llm_kernel.transports import pty_mode

    monkeypatch.chdir(tmp_path)
    peer = WirePeer()
    monkeypatch.setenv(pty_mode.ENV_IPC_SOCKET, peer.address)
    rc: Dict[str, int] = {}
    runner = threading.Thread(target=lambda: rc.setdefault("rc", pty_mode.main([])),
                              name="conformance-pty-main", daemon=True)
    runner.start()
    persisted = _persisted_block()
    try:
        peer.send(hydrate_envelope(persisted))
        confirmation = peer.wait_for(is_hydrate_complete)
        peer.send({"type": "kernel.shutdown_request", "payload": {}})
        assert peer.eof.wait(WAIT_SECONDS)
        runner.join(WAIT_SECONDS)
    finally:
        peer.close()
    assert rc.get("rc") == 0
    _assert_rfc_drift(confirmation and confirmation["payload"]["snapshot"], persisted)


def test_pty_boot_kernel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from llm_kernel.transports import pty_mode

    peer = WirePeer()
    monkeypatch.setenv(pty_mode.ENV_IPC_SOCKET, peer.address)
    state = pty_mode.boot_kernel(workdir=tmp_path)
    persisted = _persisted_block()
    try:
        peer.send(hydrate_envelope(persisted))
        confirmation = peer.wait_for(is_hydrate_complete)
    finally:
        pty_mode.shutdown_kernel(state)
        peer.close()
    _assert_rfc_drift(confirmation and confirmation["payload"]["snapshot"], persisted)


def test_serve_boot_helper(tmp_path: Path) -> None:
    from llm_kernel.transports import serve_mode

    state = serve_mode._boot_kernel_for_serve("none", workdir=tmp_path)  # noqa: SLF001
    persisted = _persisted_block()
    emitted: List[Dict[str, Any]] = []
    dispatcher = state["dispatcher"]
    original_emit = dispatcher.emit

    def _tap(envelope: Dict[str, Any]) -> None:
        emitted.append(copy.deepcopy(envelope))
        original_emit(envelope)

    dispatcher.emit = _tap  # the same tap serve's connection handler installs
    try:
        dispatcher._on_comm_msg({"content": {"data": hydrate_envelope(persisted)}})  # noqa: SLF001
        snap = wait_until(lambda: hydrate_complete_snapshot(list(emitted)))
    finally:
        dispatcher.emit = original_emit
        serve_mode._teardown_serve_kernel(state)  # noqa: SLF001
    _assert_rfc_drift(snap, persisted)


def test_app_lifespan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    from llm_kernel import app as app_module

    monkeypatch.chdir(tmp_path)
    peer = WirePeer()
    monkeypatch.setenv("LLMKERNEL_IPC_SOCKET", peer.address)
    persisted = _persisted_block()
    try:
        with TestClient(app_module.app):
            peer.send(hydrate_envelope(persisted))
            confirmation = peer.wait_for(is_hydrate_complete)
    finally:
        peer.close()
    _assert_rfc_drift(confirmation and confirmation["payload"]["snapshot"], persisted)


def test_pty_mode_subprocess(tmp_path: Path) -> None:
    """Black box: ``python -m llm_kernel pty-mode`` (the extension's spawn)."""
    peer = WirePeer()
    proc = spawn_kernel(["pty-mode"], kernel_subprocess_env(
        LLMKERNEL_IPC_SOCKET=peer.address, LLMKERNEL_SESSION_ID="conformance-sub",
    ), cwd=str(tmp_path))
    persisted = _persisted_block()
    try:
        assert peer.connected.wait(60), "pty-mode subprocess never connected"
        peer.send(hydrate_envelope(persisted))
        confirmation = peer.wait_for(is_hydrate_complete, timeout=30)
        peer.send({"type": "kernel.shutdown_request", "payload": {}})
        assert peer.eof.wait(30)
    finally:
        rc, _out, err = stop_process(proc, timeout=30)
        peer.close()
    assert rc == 0, err.decode("utf-8", "replace")[-2000:]
    _assert_rfc_drift(confirmation and confirmation["payload"]["snapshot"], persisted)


def test_serve_subprocess(tmp_path: Path) -> None:
    """Black box: ``python -m llm_kernel serve --transport tcp``."""
    port = free_port()
    proc = spawn_kernel(
        ["serve", "--transport", "tcp", "--bind", f"127.0.0.1:{port}",
         "--auth-token-env", "CONFORMANCE_TOKEN"],
        kernel_subprocess_env(CONFORMANCE_TOKEN="conformance-token"),
        cwd=str(tmp_path),
    )
    persisted = _persisted_block()
    confirmation: Optional[Dict[str, Any]] = None
    try:
        conn = connect_with_retry("127.0.0.1", port, timeout=60)
        try:
            conn.send(handshake_request("conformance-token"))
            response = conn.recv(timeout=30)
            assert response and "error" not in response["payload"], response
            conn.send(hydrate_envelope(persisted))
            confirmation = conn.recv_until(is_hydrate_complete, timeout=30)
        finally:
            conn.close()
    finally:
        rc, _out, err = stop_process(proc, signal.SIGTERM, timeout=30)
    assert rc == 0, err.decode("utf-8", "replace")[-2000:]
    _assert_rfc_drift(confirmation and confirmation["payload"]["snapshot"], persisted)


# -- what drift is measured against ----------------------------------------


def test_notebook_saved_by_this_kernel_hydrates_drift_free(tmp_path: Path) -> None:
    """A notebook that persisted exactly what the ready frame announced
    compares drift-free (versions.kernel_volatile docstring)."""
    snap = _facade_hydrate(tmp_path, hydrate_envelope(current_kernel_block()))
    assert snap is not None
    assert kernel_drift(snap) == {}


def test_every_rfc_key_of_the_source_is_compared(tmp_path: Path) -> None:
    """No hand-kept subset: a major mismatch on ANY RFC the source defines
    is an error-severity drift event."""
    block = current_kernel_block()
    for key, value in _versions.RFC_VERSIONS.items():
        block[key] = bump(value, "major")
    drift = kernel_drift(_facade_hydrate(tmp_path, hydrate_envelope(block)))
    assert set(drift) == set(_versions.RFC_VERSIONS)
    assert {e["severity"] for e in drift.values()} == {dd.SEVERITY_ERROR}


def test_patch_only_difference_is_recorded(tmp_path: Path) -> None:
    """RFC-005: any recorded/current difference is a drift entry; same
    major is not resume-blocking."""
    block = current_kernel_block()
    block["rfc_003_version"] = bump(_versions.RFC_VERSIONS["rfc_003_version"], "patch")
    drift = kernel_drift(_facade_hydrate(tmp_path, hydrate_envelope(block)))
    assert set(drift) == {"rfc_003_version"}
    assert drift["rfc_003_version"]["severity"] != dd.SEVERITY_ERROR


def test_runtime_override_of_the_source_is_what_drift_measures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """versions.py: callers read the source at call time, so a runtime
    override is observed by the drift provider of a started kernel."""
    overridden = bump(_versions.RFC_VERSIONS["rfc_004_version"], "minor")
    persisted = current_kernel_block()  # recorded before the override
    monkeypatch.setitem(_versions.RFC_VERSIONS, "rfc_004_version", overridden)
    drift = kernel_drift(_facade_hydrate(tmp_path, hydrate_envelope(persisted)))
    assert set(drift) == {"rfc_004_version"}
    assert drift["rfc_004_version"]["current_value"] == overridden
    assert drift["rfc_004_version"]["previous_value"] == persisted["rfc_004_version"]
    assert drift["rfc_004_version"]["severity"] == dd.SEVERITY_WARN


def test_runtime_override_reaches_the_pty_start_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from llm_kernel.transports import pty_mode

    overridden = bump(_versions.RFC_VERSIONS["rfc_002_version"], "major")
    persisted = current_kernel_block()
    monkeypatch.setitem(_versions.RFC_VERSIONS, "rfc_002_version", overridden)
    peer = WirePeer()
    monkeypatch.setenv(pty_mode.ENV_IPC_SOCKET, peer.address)
    state = pty_mode.boot_kernel(workdir=tmp_path)
    try:
        peer.send(hydrate_envelope(persisted))
        confirmation = peer.wait_for(is_hydrate_complete)
    finally:
        pty_mode.shutdown_kernel(state)
        peer.close()
    assert confirmation is not None
    drift = kernel_drift(confirmation["payload"]["snapshot"])
    assert set(drift) == {"rfc_002_version"}
    assert drift["rfc_002_version"]["current_value"] == overridden
    assert drift["rfc_002_version"]["severity"] == dd.SEVERITY_ERROR


def test_rfc_added_to_the_source_at_runtime_is_compared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(_versions.RFC_VERSIONS, "rfc_009_version", "2.0.0")
    persisted = current_kernel_block()
    persisted["rfc_009_version"] = "1.0.0"
    drift = kernel_drift(_facade_hydrate(tmp_path, hydrate_envelope(persisted)))
    assert set(drift) == {"rfc_009_version"}
    assert drift["rfc_009_version"]["severity"] == dd.SEVERITY_ERROR


def test_current_volatile_is_the_versions_shape() -> None:
    assert _versions.current_volatile() == {"kernel": current_kernel_block()}
    assert _versions.kernel_volatile() == current_kernel_block()
    # Fresh copies: mutating a result never mutates the source.
    _versions.kernel_volatile()["rfc_001_version"] = "9.9.9"
    assert _versions.RFC_VERSIONS["rfc_001_version"] != "9.9.9"


# -- drift_log semantics on hydrate ----------------------------------------


def test_persisted_drift_events_are_never_removed(tmp_path: Path) -> None:
    """RFC-005: the drift log is append-only; prior events survive load."""
    old = {
        "detected_at": "2026-01-01T00:00:00.000Z",
        "field_path": "config.volatile.agents[0].model",
        "previous_value": "m-1", "current_value": "m-2",
        "severity": "warn", "operator_acknowledged": True,
    }
    frame = hydrate_envelope(_persisted_block())
    frame["payload"]["snapshot"]["drift_log"] = [dict(old)]
    snap = _facade_hydrate(tmp_path, frame)
    assert snap is not None
    log = snap["drift_log"]
    assert log[0] == old
    assert set(kernel_drift(snap)) == {"rfc_006_version", "rfc_001_version"}


def test_hydrate_is_single_shot_per_session(tmp_path: Path) -> None:
    """RFC-006 §9 hydrate exclusivity: a second hydrate is rejected."""
    shell = RecordingShell()

    async def main() -> int:
        async with Kernel(workdir=tmp_path, kernel_shell=shell) as k:
            shell.deliver(hydrate_envelope(current_kernel_block()))
            assert wait_until(lambda: hydrate_complete_snapshot(shell.snapshot()))
            shell.deliver(hydrate_envelope(_persisted_block()))
            await asyncio.sleep(0.3)
            return sum(1 for f in shell.snapshot()
                       if isinstance(f, dict) and f.get("type") == "notebook.metadata"
                       and f["payload"].get("trigger") == "hydrate_complete")

    assert run(main()) == 1


# -- the detector's own contract ------------------------------------------


class TestDriftDetectorContract:
    """docs/04 - Reference/atoms/contracts/drift-detector.md."""

    def _persisted(self, **kernel: str) -> Dict[str, Any]:
        return {"config": {"volatile": {"kernel": dict(kernel)}}}

    def test_severity_classification(self) -> None:
        det = dd.DriftDetector()
        persisted = self._persisted(rfc_001_version="1.0.0", rfc_002_version="1.0.0")
        events = det.compare(persisted, current_kernel={
            "rfc_001_version": "1.2.0", "rfc_002_version": "2.0.0",
        })
        by_path = {e["field_path"]: e["severity"] for e in events}
        assert by_path == {
            "config.volatile.kernel.rfc_001_version": dd.SEVERITY_WARN,
            "config.volatile.kernel.rfc_002_version": dd.SEVERITY_ERROR,
        }
        assert (dd.SEVERITY_INFO, dd.SEVERITY_WARN, dd.SEVERITY_ERROR) == ("info", "warn", "error")

    def test_accepts_both_current_shapes(self) -> None:
        """The provider returns ``{"kernel": {...}}``; flat dicts also work."""
        det = dd.DriftDetector()
        persisted = self._persisted(rfc_005_version="0.9.0")
        flat = det.compare(persisted, current_kernel={"rfc_005_version": "1.0.0"})
        nested = det.compare(persisted, current_kernel={"kernel": {"rfc_005_version": "1.0.0"}})
        strip = lambda evs: [{k: v for k, v in e.items() if k != "detected_at"} for e in evs]  # noqa: E731
        assert strip(flat) == strip(nested) and len(flat) == 1

    def test_none_skips_the_category(self) -> None:
        det = dd.DriftDetector()
        persisted = self._persisted(rfc_001_version="0.1.0")
        assert det.compare(persisted, current_kernel=None) == []

    def test_stateless_and_non_mutating(self) -> None:
        det = dd.DriftDetector()
        persisted = self._persisted(rfc_001_version="2.0.0")
        before = copy.deepcopy(persisted)
        first = det.compare(persisted, current_kernel=_versions.current_volatile())
        second = det.compare(persisted, current_kernel=_versions.current_volatile())
        assert persisted == before
        assert len(first) == len(second) == 1
        assert det.compare(self._persisted(**current_kernel_block()),
                           current_kernel=_versions.current_volatile()) == []

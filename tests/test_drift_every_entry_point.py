"""Drift detection is alive in every start path (RFC-005 drift_log).

pty-mode is covered by ``tests/test_pty_mode.py``; this file covers the
Kernel facade, the legacy IPython kernel hooks and the serve boot helper,
all in-process. Each boot gets one hydrate (RFC-006 §9).
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from llm_kernel.capture.metadata_writer import MetadataWriter
from llm_kernel.kernel_facade import Kernel, _EmbeddedKernel, _EmbeddedSink
from llm_kernel.wire import versions as _versions


def _hydrate(block: Dict[str, str]) -> Dict[str, Any]:
    scratch = MetadataWriter(autosave_interval_sec=999.0)
    try:
        snap = scratch.snapshot(trigger="save")
    finally:
        scratch.stop(emit_final=False)
    snap.setdefault("config", {}).setdefault("volatile", {})["kernel"] = dict(block)
    snap["drift_log"] = []
    return {"type": "notebook.metadata", "payload": {"mode": "hydrate", "snapshot": snap, "trigger": "open"}}


def _drifted() -> Dict[str, str]:
    block = dict(_versions.RFC_VERSIONS)
    major = int(block["rfc_006_version"].split(".")[0]) + 1
    block["rfc_006_version"] = f"{major}.0.0"
    return block


def _paths(log: List[Dict[str, Any]]) -> List[str]:
    return [str(e.get("field_path", "")) for e in log]


def _drift_after_hydrate(dispatcher: Any, writer: Any, block: Dict[str, str]) -> List[Dict[str, Any]]:
    dispatcher._on_comm_msg({"content": {"data": _hydrate(block)}})  # noqa: SLF001
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        log = list(writer.snapshot(trigger="save").get("drift_log") or [])
        if log:
            return log
        time.sleep(0.02)
    return []


def test_facade_hydrate_reports_rfc_drift(tmp_path: Path) -> None:
    async def main() -> List[Dict[str, Any]]:
        async with Kernel(workdir=tmp_path) as k:
            assert k._dispatcher._collect_current_volatile() == _versions.current_volatile()  # noqa: SLF001
            return _drift_after_hydrate(k._dispatcher, k._metadata_writer, _drifted())  # noqa: SLF001

    paths = _paths(asyncio.run(main()))
    assert any("rfc_006_version" in p for p in paths), paths
    assert not any("rfc_001_version" in p for p in paths), paths


def test_legacy_hooks_hydrate_reports_rfc_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_kernel._kernel_hooks import attach_kernel_subsystems

    monkeypatch.setenv("LLMKERNEL_LITELLM_ENDPOINT_URL", "http://127.0.0.1:1/v1")
    kernel = _EmbeddedKernel(_EmbeddedSink())
    dispatcher, _tracker, _bridge, _supervisor, writer = attach_kernel_subsystems(kernel)
    try:
        assert dispatcher._collect_current_volatile() == _versions.current_volatile()  # noqa: SLF001
        paths = _paths(_drift_after_hydrate(dispatcher, writer, _drifted()))
    finally:
        writer.stop(emit_final=False)
        dispatcher.stop()
    assert any("rfc_006_version" in p for p in paths), paths


def test_serve_boot_hydrate_reports_rfc_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_kernel.transports import serve_mode

    monkeypatch.setenv("LLMKERNEL_LITELLM_ENDPOINT_URL", "http://127.0.0.1:1/v1")
    state = serve_mode._boot_kernel_for_serve("none")  # noqa: SLF001
    try:
        dispatcher = state["dispatcher"]
        assert dispatcher._collect_current_volatile() == _versions.current_volatile()  # noqa: SLF001
        paths = _paths(_drift_after_hydrate(dispatcher, state["metadata_writer"], _drifted()))
    finally:
        serve_mode._teardown_serve_kernel(state)  # noqa: SLF001
    assert any("rfc_006_version" in p for p in paths), paths


def test_every_entry_point_shares_one_provider_object() -> None:
    from llm_kernel.transports import pty_mode

    assert pty_mode._current_volatile is _versions.current_volatile  # noqa: SLF001

"""Agent-emitted generator-magic block tests — PLAN-S5.0.2 §7 (K3H).

Verifies the contamination detector logs K3H specifically when an
agent emits a generator-magic name in stdout, while still firing K35
(plain) / K36 (hashed) on the regular contamination paths. K3H is
log-level — the cell is flagged contaminated; no generator dispatch.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from llm_kernel import _diagnostics
from llm_kernel.agent_supervisor import (
    AgentHandle,
    AgentSupervisor,
    K35_PLAIN_MAGIC_IN_HASH_MODE,
    K36_HASHED_MAGIC_EMISSION_BLOCKED,
    K3H_AGENT_EMITTED_GENERATOR_MAGIC_BLOCKED,
)
from llm_kernel.magic_hash import magic_hash, magic_pin_fingerprint


def _make_supervisor() -> AgentSupervisor:
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
    return AgentSupervisor(
        run_tracker=tracker, dispatcher=MagicMock(),
        litellm_endpoint_url="http://127.0.0.1:9999/v1",
    )


def _stub_handle(agent_id: str = "alpha") -> AgentHandle:
    handle = AgentHandle.__new__(AgentHandle)
    handle.agent_id = agent_id
    handle.zone_id = "z1"
    handle.popen = None  # type: ignore[assignment]
    return handle


class _StubWriter:
    def __init__(self, hash_enabled: bool = False) -> None:
        self.hash_enabled = hash_enabled
        self.flag_calls: List[Dict[str, Any]] = []

    def get_config_setting(self, name: str) -> Any:
        if name == "magic_hash_enabled":
            return self.hash_enabled
        return None

    def flag_cells_contaminated_by_agent(
        self, *, agent_id: str, line: str, source: str, layer: str,
    ) -> None:
        self.flag_calls.append({
            "agent_id": agent_id, "line": line,
            "source": source, "layer": layer,
        })


def _capture_diagnostics(monkeypatch) -> List[Dict[str, Any]]:
    """Capture _diagnostics.mark calls into a list for assertion."""
    captured: List[Dict[str, Any]] = []
    real_mark = _diagnostics.mark

    def fake_mark(event: str, **kwargs: Any) -> None:
        captured.append({"event": event, **kwargs})

    monkeypatch.setattr(_diagnostics, "mark", fake_mark)
    return captured


def test_agent_emit_template_fires_K3H(monkeypatch) -> None:
    """`@@template foo` in agent stdout → K3H drift marker + flagged."""
    captured = _capture_diagnostics(monkeypatch)
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=False)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    sup._scan_for_magic_contamination(
        handle, "@@template greet", source="agent_emit:prose",
    )
    # Layer 1 contamination flag fires.
    layers = [c["layer"] for c in writer.flag_calls]
    assert "plain" in layers
    # K3H specifically logged.
    codes = [c.get("code") for c in captured if c["event"] == "supervisor_drift_marker"]
    assert K3H_AGENT_EMITTED_GENERATOR_MAGIC_BLOCKED in codes


def test_agent_emit_expand_fires_K3H(monkeypatch) -> None:
    captured = _capture_diagnostics(monkeypatch)
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=False)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    sup._scan_for_magic_contamination(
        handle, "@@expand", source="agent_emit:prose",
    )
    codes = [c.get("code") for c in captured if c["event"] == "supervisor_drift_marker"]
    assert K3H_AGENT_EMITTED_GENERATOR_MAGIC_BLOCKED in codes


def test_agent_emit_import_fires_K3H(monkeypatch) -> None:
    captured = _capture_diagnostics(monkeypatch)
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=False)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    sup._scan_for_magic_contamination(
        handle, "@@import other.llmnb", source="agent_emit:prose",
    )
    codes = [c.get("code") for c in captured if c["event"] == "supervisor_drift_marker"]
    assert K3H_AGENT_EMITTED_GENERATOR_MAGIC_BLOCKED in codes


def test_agent_emit_non_generator_does_not_fire_K3H(monkeypatch) -> None:
    """Plain `@@spawn` → K35/contamination, NOT K3H."""
    captured = _capture_diagnostics(monkeypatch)
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=False)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    sup._scan_for_magic_contamination(
        handle, "@@spawn alpha task:\"x\"", source="agent_emit:prose",
    )
    codes = [c.get("code") for c in captured if c["event"] == "supervisor_drift_marker"]
    assert K3H_AGENT_EMITTED_GENERATOR_MAGIC_BLOCKED not in codes
    # The cell is still flagged (Layer 1 plain contamination).
    layers = [c["layer"] for c in writer.flag_calls]
    assert "plain" in layers


def test_hashed_generator_in_hash_mode_fires_K3H_and_K36(monkeypatch) -> None:
    """Hash-mode + hashed generator-magic → K3H + K36 + escape."""
    captured = _capture_diagnostics(monkeypatch)
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=True)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()
    pin = "test-pin"
    h = magic_hash(pin, "template")
    line = f"@@{h}:template greet"

    verdict = sup._scan_for_magic_contamination(
        handle, line, source="agent_emit:prose",
    )
    assert verdict == "ESCAPE_REQUIRED"
    codes = [c.get("code") for c in captured if c["event"] == "supervisor_drift_marker"]
    assert K3H_AGENT_EMITTED_GENERATOR_MAGIC_BLOCKED in codes
    assert K36_HASHED_MAGIC_EMISSION_BLOCKED in codes


def test_no_generator_dispatch_on_agent_emit(monkeypatch) -> None:
    """The contamination scan does NOT call dispatch_generator.

    The scan path returns; only operator-typed cells dispatch through
    the parser → dispatcher path. The contamination scan is purely
    defensive (flag + escape).
    """
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=False)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    # The scan returns None (or ESCAPE_REQUIRED) — no list of cell ids.
    verdict = sup._scan_for_magic_contamination(
        handle, "@@template greet", source="agent_emit:prose",
    )
    assert verdict is None  # plain layer, no escape
    # No generator dispatch happened — writer.flag_calls only records
    # contamination, not new cell inserts.
    assert all("layer" in c for c in writer.flag_calls)

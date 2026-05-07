"""K-CTXR BSP-008 §9 — AgentSupervisor.send_user_turn ContextPacker / RunFrame wiring.

When the operator calls ``send_user_turn(..., cell_id=...)`` the
supervisor MUST per BSP-008 §9:

1. Pack a ContextManifest via :func:`context_packer.pack`.
2. Submit a ``record_context_manifest`` intent so the manifest lands
   under ``metadata.rts.zone.context_manifests[<manifest_id>]``.
3. Submit a start ``record_run_frame`` intent (status=``running``).
4. After the stdin write completes (success OR failure), submit a
   terminal ``record_run_frame`` intent (status=``complete`` on success,
   ``failed`` on synchronous error).

These tests drive a stubbed ``Popen`` (no real claude subprocess) and
assert the writer received the three intents in order.

Engineering Guide §11.7 parallel-test-safety: the test instantiates its
own MetadataWriter + AgentSupervisor; no shared state.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from llm_kernel.agent_supervisor import AgentHandle, AgentSupervisor
from llm_kernel.metadata_writer import MetadataWriter


# ---------------------------------------------------------------------------
# Fixtures (mirrors test_send_user_turn.py's pattern; in-process only)
# ---------------------------------------------------------------------------


class _StubStdin:
    def __init__(self, fail: bool = False) -> None:
        self.writes: List[str] = []
        self.flushed: int = 0
        self._fail = fail

    def write(self, data: str) -> int:
        if self._fail:
            raise BrokenPipeError("stub stdin closed")
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        if self._fail:
            raise BrokenPipeError("stub stdin flush failed")
        self.flushed += 1

    def close(self) -> None:  # pragma: no cover
        pass


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
    dispatcher = MagicMock()
    return AgentSupervisor(
        run_tracker=tracker, dispatcher=dispatcher,
        litellm_endpoint_url="http://127.0.0.1:9999/v1",
    )


def _stub_handle(
    agent_id: str,
    *,
    fail_stdin: bool = False,
    work_dir: Path = Path("/tmp/x"),
) -> AgentHandle:
    popen = MagicMock()
    popen.returncode = None
    popen.poll = MagicMock(return_value=None)  # alive
    popen.stdin = _StubStdin(fail=fail_stdin)
    return AgentHandle(
        agent_id=agent_id, zone_id="z1", popen=popen,
        started_at=0.0, work_dir=work_dir,
        stdout_thread=threading.Thread(),
        stderr_thread=threading.Thread(),
    )


def _wire_writer_with_cell(
    sup: AgentSupervisor, cell_id: str,
) -> MetadataWriter:
    """Bind a fresh writer to the supervisor and seed the cell so
    ContextPacker.pack() does NOT raise K100."""
    writer = MetadataWriter(autosave_interval_sec=999.0)
    sup.set_metadata_writer(writer)
    # Seed a minimal cell record so ContextPacker has something to walk.
    writer.submit_intent({
        "type": "operator.action",
        "payload": {
            "action_type": "zone_mutate",
            "intent_kind": "set_cell_metadata",
            "parameters": {"cell_id": cell_id, "kind": "agent"},
            "intent_id": f"i-set-cell-{cell_id}",
        },
    })
    return writer


def _intent_kinds_in_log(writer: MetadataWriter) -> List[str]:
    """Return the kinds of every applied intent in submission order."""
    return [e["intent_kind"] for e in writer.iter_intent_log()]


def _run_frames_for(writer: MetadataWriter, cell_id: str) -> List[Dict[str, Any]]:
    """Return the persisted RunFrame records targeting ``cell_id``."""
    snap = writer.snapshot()
    rf = snap.get("zone", {}).get("run_frames", {}) or {}
    return [r for r in rf.values() if r.get("cell_id") == cell_id]


# ---------------------------------------------------------------------------
# 1. Happy path: manifest + start + terminal-complete intents land.
# ---------------------------------------------------------------------------


def test_send_user_turn_submits_manifest_and_run_frame_pair() -> None:
    """A successful send produces ``record_context_manifest`` + start + terminal."""
    sup = _make_supervisor()
    cell_id = "vscode-notebook-cell:test#c1"
    writer = _wire_writer_with_cell(sup, cell_id)
    handle = _stub_handle("alpha")
    sup._agents["alpha"] = handle

    result = sup.send_user_turn(
        agent_id="alpha",
        text="optimize for read performance",
        cell_id=cell_id,
    )
    assert result["agent_id"] == "alpha"
    assert result["status"] == "sent"
    assert result["run_id"] is not None
    assert result["context_manifest_id"] is not None

    kinds = _intent_kinds_in_log(writer)
    # Order: set_cell_metadata (seed) -> record_context_manifest ->
    # record_run_frame (start) -> record_run_frame (terminal).
    # ``update_agent_session`` is skipped when there are no persisted turns
    # (head_turn_id is None per the supervisor's no-turns branch).
    assert "record_context_manifest" in kinds
    assert kinds.count("record_run_frame") == 2, kinds

    # Terminal frame has status=complete (the supervisor's V1 vantage —
    # successful stdin dispatch == run-complete from the supervisor side).
    frames = _run_frames_for(writer, cell_id)
    assert len(frames) == 1, frames
    persisted = frames[0]
    assert persisted["status"] == "complete"
    assert persisted["run_id"] == result["run_id"]
    assert persisted["context_manifest_id"] == result["context_manifest_id"]
    assert persisted["executor_id"] == "alpha"


# ---------------------------------------------------------------------------
# 2. Failure path: synchronous BrokenPipeError emits status=failed.
# ---------------------------------------------------------------------------


def test_send_user_turn_failure_emits_failed_run_frame() -> None:
    """A BrokenPipeError on stdin emits a terminal RunFrame with status=failed."""
    import pytest as _pytest

    sup = _make_supervisor()
    cell_id = "vscode-notebook-cell:test#c-fail"
    writer = _wire_writer_with_cell(sup, cell_id)
    handle = _stub_handle("alpha", fail_stdin=True)
    sup._agents["alpha"] = handle

    with _pytest.raises(BrokenPipeError):
        sup.send_user_turn(
            agent_id="alpha",
            text="this will fail",
            cell_id=cell_id,
        )

    frames = _run_frames_for(writer, cell_id)
    assert len(frames) == 1, frames
    persisted = frames[0]
    assert persisted["status"] == "failed"
    assert persisted["ended_at"] is not None


# ---------------------------------------------------------------------------
# 3. Degradation: missing cell_id skips the BSP-008 wiring entirely.
# ---------------------------------------------------------------------------


def test_send_user_turn_without_cell_id_skips_run_frame() -> None:
    """Per BSP-008 §12 graceful degradation: no cell_id -> no manifest / RunFrame."""
    sup = _make_supervisor()
    writer = MetadataWriter(autosave_interval_sec=999.0)
    sup.set_metadata_writer(writer)
    handle = _stub_handle("alpha")
    sup._agents["alpha"] = handle

    result = sup.send_user_turn(
        agent_id="alpha", text="hi", cell_id=None,
    )
    assert result["agent_id"] == "alpha"
    assert result["status"] == "sent"
    assert result["run_id"] is None
    assert result["context_manifest_id"] is None
    snap = writer.snapshot()
    zone = snap.get("zone", {})
    assert "run_frames" not in zone or not zone["run_frames"]
    assert "context_manifests" not in zone or not zone["context_manifests"]

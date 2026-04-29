"""K-MW V1 Kernel Gap Closure G13 -- intent-queue overflow disk fallback.

Per ``docs/notebook/PLAN-substrate-gap-closure.md`` G13 + RFC-005
§F13: when the buffered intent queue exceeds the threshold the writer
spills the buffered envelopes to a JSON-line file under
``<workspace_root>/.llmnb-intent-queue/`` and records a checkpoint
marker at ``metadata.rts.queues[<queue>].overflow``.  On the next
flush, the writer drains the disk-spilled intents IN ORDER, then
resumes in-memory queueing.

Threshold note: RFC-005 §F13 caps the *event-log* queue at 10 000;
the brief asks for a separate threshold on the *intent* queue.  The
writer ships :data:`DEFAULT_INTENT_QUEUE_OVERFLOW_THRESHOLD = 1 000`
as the chosen default (flagged in the slice report); tests use a
small threshold for speed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from llm_kernel.metadata_writer import MetadataWriter


def _envelope(intent_id: str) -> Dict[str, Any]:
    """Build a minimal-but-valid layout-edit intent envelope."""
    return {
        "type": "operator.action",
        "payload": {
            "action_type": "zone_mutate",
            "intent_kind": "apply_layout_edit",
            "parameters": {
                "operation": "add_zone",
                "parameters": {
                    "node_spec": {"id": f"zone-{intent_id}", "type": "zone"},
                },
            },
            "intent_id": intent_id,
        },
    }


def test_queue_below_threshold_uses_memory(tmp_path: Path) -> None:
    """Below-threshold ``enqueue_intent`` keeps everything in RAM.

    No disk spill, no marker on the snapshot, ``flush_pending_intents``
    drains in arrival order.
    """
    threshold = 5
    writer = MetadataWriter(
        autosave_interval_sec=999.0,
        workspace_root=tmp_path,
        intent_queue_overflow_threshold=threshold,
    )
    for i in range(threshold):  # exactly threshold => still in-memory
        status = writer.enqueue_intent(_envelope(f"i-{i:03d}"))
        assert status["overflow"] is False, status
    # Spill directory should NOT have been created (we never wrote).
    spill_dir = tmp_path / ".llmnb-intent-queue"
    assert not spill_dir.exists(), "below-threshold path wrote to disk"
    # No queue overflow marker.
    assert writer.get_queue_overflow_marker("intents") is None
    # Drain works.
    results = writer.flush_pending_intents()
    assert len(results) == threshold
    assert all(r["applied"] for r in results), results


def test_queue_overflow_writes_checkpoint_marker(tmp_path: Path) -> None:
    """Overflow records ``metadata.rts.queues[intents].overflow`` marker.

    The marker carries ``checkpoint_id`` + ``disk_path``; the snapshot
    body includes the queue record so the next file-open resumes the
    drain.
    """
    threshold = 3
    writer = MetadataWriter(
        autosave_interval_sec=999.0,
        workspace_root=tmp_path,
        intent_queue_overflow_threshold=threshold,
    )
    # Push threshold + 1 to trigger one overflow.
    statuses = [
        writer.enqueue_intent(_envelope(f"i-{i:03d}"))
        for i in range(threshold + 1)
    ]
    # Last enqueue triggers the overflow.
    assert any(s["overflow"] for s in statuses), statuses
    marker = writer.get_queue_overflow_marker("intents")
    assert marker is not None, "overflow marker not recorded"
    assert isinstance(marker["checkpoint_id"], str) and marker["checkpoint_id"]
    assert isinstance(marker["disk_path"], str) and marker["disk_path"]
    # Snapshot also exposes the queue record under metadata.rts.queues.
    snap = writer.snapshot()
    queues = snap.get("queues", {})
    assert "intents" in queues
    assert "overflow" in queues["intents"]
    assert (
        queues["intents"]["overflow"]["checkpoint_id"]
        == marker["checkpoint_id"]
    )


def test_queue_overflow_spills_to_disk(tmp_path: Path) -> None:
    """The overflow writes one JSON-line per buffered intent to disk.

    Semantics: the buffer overflows the moment its size exceeds the
    configured threshold; all currently-buffered intents are spilled
    in one atomic write, and the buffer is drained.  Subsequent
    enqueues land in the freshly-empty buffer (and trigger a SECOND
    overflow only after they exceed the threshold again).
    """
    threshold = 2
    writer = MetadataWriter(
        autosave_interval_sec=999.0,
        workspace_root=tmp_path,
        intent_queue_overflow_threshold=threshold,
    )
    # Push exactly threshold+1 intents -> single overflow, all spill.
    spilled_ids = [f"i-{i:03d}" for i in range(threshold + 1)]
    for iid in spilled_ids:
        writer.enqueue_intent(_envelope(iid))
    marker = writer.get_queue_overflow_marker("intents")
    assert marker is not None
    spill_path = tmp_path / marker["disk_path"]
    assert spill_path.exists(), f"spill file missing: {spill_path}"
    lines = [
        line for line in spill_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Every buffered intent ended up on disk in arrival order.
    assert len(lines) == len(spilled_ids)
    decoded = [json.loads(line) for line in lines]
    assert [
        env["payload"]["intent_id"] for env in decoded
    ] == spilled_ids


def test_queue_overflow_drained_on_next_flush_in_order(tmp_path: Path) -> None:
    """On flush the disk-spilled intents drain BEFORE in-memory entries, in order."""
    threshold = 2
    writer = MetadataWriter(
        autosave_interval_sec=999.0,
        workspace_root=tmp_path,
        intent_queue_overflow_threshold=threshold,
    )
    spilled_ids = [f"spill-{i}" for i in range(threshold + 1)]
    for iid in spilled_ids:
        writer.enqueue_intent(_envelope(iid))
    # Confirm the buffer drained to disk (overflow happened).
    assert writer.get_queue_overflow_marker("intents") is not None
    # Now enqueue more intents that go into the freshly-empty in-memory buffer.
    in_memory_ids = ["mem-0", "mem-1"]
    for iid in in_memory_ids:
        writer.enqueue_intent(_envelope(iid))
    # Flush drains disk first, then in-memory.
    results = writer.flush_pending_intents()
    applied_ids: List[str] = []
    for env_result in results:
        # submit_intent's response carries the intent_id; reconstruct.
        applied_ids.append(env_result["intent_id"])
    assert applied_ids == spilled_ids + in_memory_ids, applied_ids
    # Marker cleared after successful drain.
    assert writer.get_queue_overflow_marker("intents") is None
    # Spill file removed.
    spill_dir = tmp_path / ".llmnb-intent-queue"
    leftover = list(spill_dir.glob("*.jsonl")) if spill_dir.exists() else []
    assert leftover == [], f"spill file left behind: {leftover}"


def test_queue_overflow_disk_path_uses_workspace_relative_dir(
    tmp_path: Path,
) -> None:
    """Spill files live at ``<workspace_root>/.llmnb-intent-queue/<id>.jsonl``.

    The brief asks for the workspace-relative convention (so the
    marker's ``disk_path`` survives moving the workspace).  This test
    verifies the directory layout AND that the marker's ``disk_path``
    is a relative POSIX-style path under that directory.
    """
    threshold = 1
    writer = MetadataWriter(
        autosave_interval_sec=999.0,
        workspace_root=tmp_path,
        intent_queue_overflow_threshold=threshold,
    )
    writer.enqueue_intent(_envelope("i-0"))
    writer.enqueue_intent(_envelope("i-1"))  # threshold+1 -> overflow
    marker = writer.get_queue_overflow_marker("intents")
    assert marker is not None
    disk_path = marker["disk_path"]
    # Relative, POSIX-style.
    assert not Path(disk_path).is_absolute()
    assert disk_path.startswith(".llmnb-intent-queue/")
    assert disk_path.endswith(".jsonl")
    # Resolved file exists under workspace_root.
    resolved = tmp_path / disk_path
    assert resolved.exists()



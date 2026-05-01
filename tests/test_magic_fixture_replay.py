"""PLAN-S6.0 §5 magic-fixture replay pipeline.

Drives the magic-text -> envelope-stream -> event-log -> projection
chain end to end, asserting:

* parsing the magic produces the expected operator-action sequence;
* the resulting ``zone.event_log[]`` round-trips through
  :class:`EventLogReplayer` deterministically (5 consecutive replays
  yield byte-identical projections).

The fixture is local to the submodule (the outer-repo
``tests/fixtures/spawn-and-notify.magic`` is unreachable from the
submodule's pytest invocation; PLAN-S6.0 §5 doesn't require shared
fixtures).
"""

from __future__ import annotations

from typing import Any, Dict, List

from llm_kernel.cell_text import parse_cell, split_at_breaks
from llm_kernel.event_log import EventLogReplayer
from llm_kernel.metadata_writer import MetadataWriter
from llm_kernel.run_envelope import (
    DIRECTION_EXTENSION_TO_KERNEL, DIRECTION_KERNEL_TO_EXTENSION, make_envelope,
)


_MAGIC_FIXTURE: str = (
    '@@spawn alpha task:"hello world"\n'
    '@@break\n'
    '@@scratch\n'
    'print("hello from stub")\n'
)


def _operator_action_envelope(
    seq: int, kind: str, args: Dict[str, Any],
) -> Dict[str, Any]:
    """Helper: build an operator.action envelope shaped like the live path."""
    return make_envelope(
        "operator.action",
        {"action_type": "magic", "kind": kind, "args": args, "seq": seq},
        correlation_id=f"00000000-0000-0000-0000-{seq:012d}",
        direction=DIRECTION_EXTENSION_TO_KERNEL,
        timestamp="2026-04-30T00:00:00.000Z",
    )


def _checkpoint_snapshot() -> Dict[str, Any]:
    return make_envelope(
        "notebook.metadata",
        {
            "mode": "snapshot", "snapshot_version": 1,
            "snapshot": {
                "schema_version": "1.0.0",
                "session_id": "magic-fixture",
                "snapshot_version": 1,
                "fixture": "spawn-and-notify",
            },
            "trigger": "save",
        },
        correlation_id="magic-fixture:1",
        direction=DIRECTION_KERNEL_TO_EXTENSION,
        timestamp="2026-04-30T00:00:00.000Z",
    )


def _drive_fixture(writer: MetadataWriter) -> List[Dict[str, Any]]:
    """Parse the fixture, capture the resulting envelope sequence.

    Returns the captured ``zone.event_log[]`` for replay assertions.
    """
    # Initial checkpoint so the replayer has a snapshot to anchor on.
    writer.capture_envelope(_checkpoint_snapshot())

    parsed_actions: List[Dict[str, Any]] = []
    for idx, cell_src in enumerate(split_at_breaks(_MAGIC_FIXTURE)):
        parsed = parse_cell(cell_src)
        parsed_actions.append({"kind": parsed.kind, "args": parsed.args})
        envelope = _operator_action_envelope(
            seq=idx + 1, kind=parsed.kind or "unknown", args=parsed.args,
        )
        writer.capture_envelope(envelope)

    snapshot = writer.snapshot()
    return snapshot["zone"]["event_log"]


class _NullDispatcher:
    def _on_comm_msg(self, msg: Dict[str, Any]) -> None:
        return None


def test_magic_to_events_to_tree() -> None:
    """Parse the .magic fixture, capture envelopes, replay to projection.

    The resulting tree (``working_rts``) must carry the snapshot
    fixture marker (the checkpoint) -- the dispatcher in this test is
    a no-op, so the tail does not mutate the projection beyond the
    snapshot anchor.  This proves the pipeline closes: text in,
    deterministic projection out.
    """
    writer = MetadataWriter(autosave_interval_sec=999.0)
    log = _drive_fixture(writer)

    # Two cells in the fixture (split by @@break) -> two operator-action
    # envelopes plus the checkpoint snapshot envelope.
    assert len(log) == 3
    assert log[0]["payload"]["mode"] == "snapshot"
    assert log[1]["payload"]["kind"] == "spawn"
    assert log[1]["payload"]["args"]["agent_id"] == "alpha"
    assert log[2]["payload"]["kind"] == "scratch"

    replayer = EventLogReplayer(log)
    projected = replayer.project_state(dispatcher=_NullDispatcher())
    assert projected["fixture"] == "spawn-and-notify"


def test_magic_fixture_round_trip_deterministic() -> None:
    """5 consecutive replays of the same fixture log produce byte-identical trees."""
    writer = MetadataWriter(autosave_interval_sec=999.0)
    log = _drive_fixture(writer)

    dispatcher = _NullDispatcher()
    projections = [
        EventLogReplayer(log).project_state(dispatcher=dispatcher)
        for _ in range(5)
    ]
    base = projections[0]
    for p in projections[1:]:
        assert p == base, "replay produced divergent projection"

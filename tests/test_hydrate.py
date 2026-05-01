"""K-MW :meth:`MetadataWriter.hydrate` (RFC-005 / RFC-006 §8 mode:hydrate).

Verifies idempotency, schema_version validation, forbidden-secret
rejection, and that hydrating from a known snapshot lands the writer
in a state that re-emits an equivalent snapshot.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from llm_kernel.metadata_writer import (
    SCHEMA_VERSION,
    MetadataWriter,
)


def _baseline_snapshot() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_uri": "https://llmnb.dev/llmnb/v1/schema.json",
        "session_id": "9c1a3b2d-4e5f-4061-a072-8d9e3f4a5b6c",
        "created_at": "2026-04-26T12:14:08.221Z",
        "snapshot_version": 17,
        "layout": {
            "version": 1,
            "tree": {
                "id": "root", "type": "workspace",
                "render_hints": {"label": "monorepo"},
                "children": [
                    {"id": "zone-a", "type": "zone",
                     "render_hints": {}, "children": []},
                ],
            },
        },
        "agents": {
            "version": 1,
            "nodes": [
                {"id": "agent:alpha", "type": "agent",
                 "properties": {"status": "idle"}},
            ],
            "edges": [],
        },
        "config": {
            "version": 1,
            "recoverable": {
                "kernel": {"blob_threshold_bytes": 65536},
                "agents": [
                    {"agent_id": "alpha", "zone_id": "refactor",
                     "tools_allowed": ["notify"]},
                ],
                "mcp_servers": [],
            },
            "volatile": {
                "kernel": {
                    "model_default": "claude-sonnet-4-5",
                    "rfc_001_version": "1.0.0",
                },
                "agents": [],
                "mcp_servers": [],
            },
        },
        "event_log": {
            "version": 1,
            "runs": [
                {
                    "traceId": "5d27f5dd26ce4d619dbb9fbf36d2fe2b",
                    "spanId": "8a3c1a2e9d774f0a",
                    "parentSpanId": None,
                    "name": "notify", "kind": "SPAN_KIND_INTERNAL",
                    "startTimeUnixNano": "1745588938412000000",
                    "endTimeUnixNano": "1745588938611000000",
                    "status": {"code": "STATUS_CODE_OK", "message": ""},
                    "attributes": [
                        {"key": "llmnb.run_type",
                         "value": {"stringValue": "tool"}},
                    ],
                    "events": [], "links": [],
                },
            ],
        },
        "blobs": {},
        "drift_log": [
            {
                "detected_at": "2026-04-26T13:22:45.000Z",
                "field_path": "config.volatile.kernel.model_default",
                "previous_value": "claude-sonnet-4-5",
                "current_value": "claude-sonnet-4-6",
                "severity": "warn",
                "operator_acknowledged": False,
            },
        ],
    }


def test_hydrate_round_trip_yields_equivalent_state() -> None:
    """A hydrated writer's next snapshot mirrors the persisted shape."""
    snap = _baseline_snapshot()
    writer = MetadataWriter(autosave_interval_sec=999.0)
    writer.hydrate(snap)

    out = writer.snapshot()
    # Snapshot version increments on emission per the writer's
    # contract: the persisted version was 17 and the next emission is
    # 18 (RFC-006 §"hydrate request/response semantics").
    assert out["snapshot_version"] == snap["snapshot_version"] + 1
    assert out["session_id"] == snap["session_id"]
    assert out["created_at"] == snap["created_at"]
    assert out["layout"] == snap["layout"]
    assert out["agents"] == snap["agents"]
    # event_log runs are preserved (deduped by spanId).
    assert len(out["event_log"]["runs"]) == 1
    assert out["event_log"]["runs"][0]["spanId"] == "8a3c1a2e9d774f0a"
    assert out["drift_log"] == snap["drift_log"]


def test_hydrate_is_idempotent() -> None:
    """Hydrating with the same snapshot twice leaves observable state equal."""
    snap = _baseline_snapshot()
    a = MetadataWriter(autosave_interval_sec=999.0)
    a.hydrate(snap)
    a.hydrate(snap)

    b = MetadataWriter(autosave_interval_sec=999.0)
    b.hydrate(snap)

    # Compare by extracting state-relevant private fields.
    assert a._snapshot_version == b._snapshot_version  # noqa: SLF001
    assert a._layout == b._layout  # noqa: SLF001
    assert a._agent_graph == b._agent_graph  # noqa: SLF001
    assert a._drift_log == b._drift_log  # noqa: SLF001
    assert a._extra_runs == b._extra_runs  # noqa: SLF001
    assert a._blobs == b._blobs  # noqa: SLF001
    assert a._config == b._config  # noqa: SLF001


def test_hydrate_rejects_schema_version_major_mismatch() -> None:
    """Major-version mismatch raises :class:`ValueError`."""
    snap = _baseline_snapshot()
    snap["schema_version"] = "2.0.0"
    writer = MetadataWriter(autosave_interval_sec=999.0)
    with pytest.raises(ValueError):
        writer.hydrate(snap)


def test_hydrate_rejects_forbidden_secret_in_config() -> None:
    """A forbidden field anywhere in config raises ``ValueError``."""
    snap = _baseline_snapshot()
    snap["config"]["volatile"]["kernel"]["api_key"] = "sk-DO_NOT_LOG"
    writer = MetadataWriter(autosave_interval_sec=999.0)
    with pytest.raises(ValueError) as ei:
        writer.hydrate(snap)
    assert "forbidden secret in config" in str(ei.value)
    # The offending value MUST NOT appear in the exception text.
    assert "DO_NOT_LOG" not in str(ei.value)


# ---------------------------------------------------------------------------
# PLAN-S4: hydrate + handoff tests
# ---------------------------------------------------------------------------

import io as _io
import threading as _threading
import uuid as _uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_kernel.agent_supervisor import AgentSupervisor


class _FakeStdinH:
    """Captures write/flush calls for hydrate handoff tests."""

    def __init__(self) -> None:
        self._buf = _io.StringIO()

    def write(self, s: str) -> int:
        return self._buf.write(s)

    def flush(self) -> None:
        pass

    def lines(self) -> list:
        return [ln for ln in self._buf.getvalue().splitlines() if ln]


class _FakePopenH:
    """Minimal Popen double with writable stdin."""

    def __init__(self) -> None:
        self.stdout = iter([""])
        self.stderr = iter([""])
        self.returncode: Any = None
        self._exited = _threading.Event()
        self.pid = 99999
        self.stdin = _FakeStdinH()

    def poll(self) -> Any:
        return self.returncode

    def wait(self, timeout: Any = None) -> int:
        self._exited.wait(timeout=timeout or 0.2)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0
        self._exited.set()

    def kill(self) -> None:
        self.returncode = -9
        self._exited.set()


def _make_sup_h() -> AgentSupervisor:
    """Build a hydrate-test supervisor with a real writer wired (PLAN-S4.1)."""
    from llm_kernel.run_tracker import RunTracker

    class _Sink:
        def emit(self, e: Any) -> None:
            pass

    tracker = RunTracker(
        trace_id=str(_uuid.uuid4()), sink=_Sink(),
        agent_id="alpha", zone_id="z1",
    )
    sup = AgentSupervisor(
        run_tracker=tracker,
        dispatcher=MagicMock(),
        litellm_endpoint_url="http://127.0.0.1:9999/v1",
    )
    writer = MetadataWriter(autosave_interval_sec=999.0)
    sup.set_metadata_writer(writer)
    return sup


def _seed_turn_h(
    sup: AgentSupervisor, turn_id: str, agent_id: str, role: str,
    content: str, parent_id: Any = None,
) -> None:
    """PLAN-S4.1 replacement for the deleted ``record_turn`` test seam."""
    norm_role = {"assistant": "agent", "user": "operator"}.get(role, role)
    sup._metadata_writer.submit_intent({  # type: ignore[union-attr]
        "payload": {
            "action_type": "zone_mutate",
            "intent_kind": "append_turn",
            "parameters": {
                "id": turn_id, "agent_id": agent_id,
                "role": norm_role, "body": content,
                "parent_id": parent_id,
            },
            "intent_id": f"seed-{turn_id}-{_uuid.uuid4().hex[:8]}",
        },
    })


def _patch_health_h(status_code: int = 200):
    fake = MagicMock()
    fake.status_code = status_code
    return patch("llm_kernel._provisioning.httpx.head", return_value=fake)


def test_hydrate_restores_last_seen_turn_id_per_agent(tmp_path: Path) -> None:
    """respawn_from_config restores last_seen_turn_id from the config entry."""
    sup = _make_sup_h()
    fake = _FakePopenH()

    entry = {
        "agent_id": "alpha",
        "zone_id": "z1",
        "task": "hello",
        "work_dir": str(tmp_path),
        "api_key": "sk-x",
        "last_seen_turn_id": "t_hydrated_99",
    }

    with _patch_health_h(), patch("subprocess.Popen", return_value=fake):
        results = sup.respawn_from_config([entry])

    assert results.get("alpha") == "spawned"
    with sup._lock:
        handle = sup._agents.get("alpha")
    assert handle is not None
    # The handle's last_seen_turn_id should be restored from the config entry.
    assert handle.last_seen_turn_id == "t_hydrated_99"
    fake.terminate()


def test_handoff_after_hydrate_replays_correctly(tmp_path: Path) -> None:
    """After hydrate with last_seen_turn_id set, send_user_turn injects only newer turns."""
    sup = _make_sup_h()
    fake_alpha = _FakePopenH()
    fake_beta = _FakePopenH()

    popped_h: list = [fake_alpha, fake_beta]

    def _factory_h(*a: Any, **kw: Any) -> Any:
        return popped_h.pop(0)

    entry = {
        "agent_id": "alpha",
        "zone_id": "z1",
        "task": "hello",
        "work_dir": str(tmp_path),
        "api_key": "sk-x",
        "last_seen_turn_id": "t_old",
    }

    with _patch_health_h(), patch("subprocess.Popen", side_effect=_factory_h):
        sup.respawn_from_config([entry])
        # Also spawn beta directly.
        sup.spawn(
            zone_id="z1", agent_id="beta", task="t",
            work_dir=tmp_path, api_key="sk-x",
        )

    # Record turns: t_old (already seen) and t_new (missed).
    _seed_turn_h(sup,"t_old", "beta", "assistant", "old beta msg", parent_id=None)
    _seed_turn_h(sup,"t_new", "beta", "assistant", "new beta msg", parent_id="t_old")

    # send_user_turn for alpha should inject only t_new (t_old was last_seen).
    result = sup.send_user_turn("alpha", "after hydrate")
    lines = fake_alpha.stdin.lines()
    # Expect: 1 prefix (t_new only) + 1 operator = 2 lines.
    assert len(lines) == 2, lines
    prefix_content = json.loads(lines[0])["message"]["content"]
    assert "new beta msg" in prefix_content, prefix_content
    assert "old beta msg" not in prefix_content, prefix_content
    assert result["handoff_prefix_count"] == 1
    fake_alpha.terminate()
    fake_beta.terminate()


def test_hydrate_replaces_state_not_merges() -> None:
    """Hydrate replaces -- pre-existing layout edits are wiped."""
    snap = _baseline_snapshot()
    writer = MetadataWriter(autosave_interval_sec=999.0)
    # Seed pre-hydrate state that the hydrate MUST clear.
    writer.apply_layout_edit(
        operation="add_zone",
        parameters={"node_spec": {"id": "pre-existing", "type": "zone"}},
    )
    writer.append_drift_event(
        field_path="x.y", previous_value=1, current_value=2, severity="info",
    )
    writer.hydrate(snap)
    out = writer.snapshot()
    ids = {c["id"] for c in out["layout"]["tree"]["children"]}
    assert "pre-existing" not in ids
    assert "zone-a" in ids
    # drift_log was reset to the hydrated shape (1 entry, not 2).
    assert len(out["drift_log"]) == 1


def test_hydrate_serializes_to_baseline_via_serialize_snapshot() -> None:
    """The post-hydrate snapshot is JSON-equivalent to the baseline.

    The next :meth:`snapshot` increments ``snapshot_version`` so the
    direct equality fails by one field; we drop that field and the
    round-trip is exact for the recoverable shape.
    """
    snap = _baseline_snapshot()
    writer = MetadataWriter(autosave_interval_sec=999.0)
    writer.hydrate(snap)
    out = writer.snapshot()
    # Strip the bumped version for the comparison.
    out_copy = dict(out)
    base_copy = dict(snap)
    out_copy["snapshot_version"] = 0
    base_copy["snapshot_version"] = 0
    assert json.dumps(out_copy, sort_keys=True) == json.dumps(
        base_copy, sort_keys=True,
    )


# ---------------------------------------------------------------------------
# PLAN-S4.1 §5: hydrate-rebuild test (was deferred in PLAN-S4 §9).
# ---------------------------------------------------------------------------


def test_handoff_after_hydrate_walks_persisted_turns(tmp_path: Path) -> None:
    """Close → reopen → addressed-agent replay reads from persisted turns[].

    PLAN-S4.1: post-migration the supervisor's ``_missed_turns`` reads
    directly from ``metadata.rts.zone.agents.<*>.turns[]``.  Even after
    a snapshot round-trip (close → reopen) the persisted graph is the
    source of truth — no in-memory cache to rebuild.
    """
    # Step 1: build state in writer-A and snapshot it.
    sup_a = _make_sup_h()
    fake_a_alpha = _FakePopenH()
    fake_a_beta = _FakePopenH()
    queue_a = [fake_a_alpha, fake_a_beta]

    def _factory_a(*a: Any, **k: Any) -> Any:
        return queue_a.pop(0)
    with _patch_health_h(), patch("subprocess.Popen", side_effect=_factory_a):
        sup_a.spawn(zone_id="z1", agent_id="alpha", task="t",
                    work_dir=tmp_path, api_key="sk-x")
        sup_a.spawn(zone_id="z1", agent_id="beta", task="t",
                    work_dir=tmp_path, api_key="sk-x")
    _seed_turn_h(sup_a, "t_pre1", "beta", "assistant", "pre-1", parent_id=None)
    _seed_turn_h(sup_a, "t_pre2", "beta", "assistant", "pre-2", parent_id="t_pre1")
    snap_a = sup_a._metadata_writer.snapshot()  # type: ignore[union-attr]
    fake_a_alpha.terminate()
    fake_a_beta.terminate()

    # Step 2: open a fresh supervisor + writer; hydrate from snap_a.
    sup_b = _make_sup_h()
    sup_b._metadata_writer.hydrate(snap_a)  # type: ignore[union-attr]
    fake_b_alpha = _FakePopenH()
    fake_b_beta = _FakePopenH()
    queue_b = [fake_b_alpha, fake_b_beta]

    def _factory_b(*a: Any, **k: Any) -> Any:
        return queue_b.pop(0)
    entry = {"agent_id": "alpha", "zone_id": "z1", "task": "t",
             "work_dir": str(tmp_path), "api_key": "sk-x",
             "last_seen_turn_id": None}
    with _patch_health_h(), patch("subprocess.Popen", side_effect=_factory_b):
        sup_b.respawn_from_config([entry])
        sup_b.spawn(zone_id="z1", agent_id="beta", task="t",
                    work_dir=tmp_path, api_key="sk-x")
    head = sup_b._notebook_head_turn_id()
    assert head == "t_pre2"
    missed = sup_b._missed_turns("alpha", head)
    assert [t["id"] for t in missed] == ["t_pre1", "t_pre2"]
    fake_b_alpha.terminate()
    fake_b_beta.terminate()

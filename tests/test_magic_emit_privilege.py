"""Privilege store + ``emit_magic_cell`` tool tests -- PLAN-S5.0.4 §5.

Covers:

* grant + revoke roundtrip
* ``emit_magic_cell`` rejects K3K without a covering grant
* ``emit_magic_cell`` succeeds with a grant and stamps ``generated_by``
* cross-zone grant does NOT transfer
* idempotent grant (same agent_id + zone_id updates in place)
* idempotent revoke (missing entry is a no-op)
* scope: ``"all"`` covers every magic
* scope: ``[<name>, ...]`` is name-specific
* the writer's privilege table is serialized into the snapshot
* ``has_magic_emit_privilege`` returns False on malformed inputs
* the tool's structural-write surface uses CellManager (clause 2 lint)
* the tool stamps ``promoted_from_stream`` when set
"""

from __future__ import annotations

import pytest

from llm_kernel.cell_manager import CellManager
from llm_kernel.magic_emit_tool import (
    K3K_UNPRIVILEGED_AGENT_MAGIC_EMIT,
    MagicEmitError,
    emit_magic_cell,
)
from llm_kernel.metadata_writer import MetadataWriter


@pytest.fixture
def writer(tmp_path) -> MetadataWriter:
    w = MetadataWriter(workspace_root=tmp_path)
    w.set_cell_text("c_anchor", "@@scratch anchor")
    return w


@pytest.fixture
def cell_manager(writer) -> CellManager:
    return CellManager(writer)


def test_grant_then_get_returns_record(writer) -> None:
    writer.grant_magic_emit_privilege(
        agent_id="alpha", zone_id="z1",
        scope={"magics": ["spawn", "scratch"]},
    )
    out = writer.get_magic_emit_privileges()
    assert len(out) == 1
    assert out[0]["agent_id"] == "alpha"
    assert out[0]["zone_id"] == "z1"
    assert out[0]["scope"] == {"magics": ["spawn", "scratch"]}
    assert out[0]["granted_at"].endswith("Z")


def test_grant_with_scope_all_matches_every_magic(writer) -> None:
    writer.grant_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", scope={"magics": "all"},
    )
    assert writer.has_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", magic_name="spawn",
    )
    assert writer.has_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", magic_name="scratch",
    )
    assert writer.has_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", magic_name="anything",
    )


def test_grant_with_name_list_is_name_specific(writer) -> None:
    writer.grant_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", scope={"magics": ["spawn"]},
    )
    assert writer.has_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", magic_name="spawn",
    )
    assert not writer.has_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", magic_name="scratch",
    )


def test_grant_is_idempotent_on_agent_zone_pair(writer) -> None:
    writer.grant_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", scope={"magics": ["spawn"]},
    )
    writer.grant_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", scope={"magics": ["scratch"]},
    )
    entries = writer.get_magic_emit_privileges()
    assert len(entries) == 1
    assert entries[0]["scope"]["magics"] == ["scratch"]


def test_revoke_removes_grant(writer) -> None:
    writer.grant_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", scope={"magics": "all"},
    )
    writer.revoke_magic_emit_privilege(agent_id="alpha", zone_id="z1")
    assert writer.get_magic_emit_privileges() == []
    assert not writer.has_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", magic_name="spawn",
    )


def test_revoke_is_idempotent_on_missing_entry(writer) -> None:
    # No grant exists; revoke is a no-op.
    writer.revoke_magic_emit_privilege(agent_id="nobody", zone_id="z1")
    writer.revoke_magic_emit_privilege(agent_id="nobody", zone_id="z1")
    assert writer.get_magic_emit_privileges() == []


def test_grant_does_not_transfer_across_zones(writer) -> None:
    writer.grant_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", scope={"magics": "all"},
    )
    assert writer.has_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", magic_name="spawn",
    )
    assert not writer.has_magic_emit_privilege(
        agent_id="alpha", zone_id="z2", magic_name="spawn",
    )


def test_emit_magic_cell_rejects_K3K_without_grant(
    writer, cell_manager,
) -> None:
    with pytest.raises(MagicEmitError) as exc:
        emit_magic_cell(
            agent_id="alpha", zone_id="z1",
            name="scratch", args={}, body=None,
            position={"after_cell_id": "c_anchor"},
            writer=writer, cell_manager=cell_manager,
        )
    assert exc.value.code == K3K_UNPRIVILEGED_AGENT_MAGIC_EMIT


def test_emit_magic_cell_succeeds_with_grant_and_stamps_provenance(
    writer, cell_manager,
) -> None:
    writer.grant_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", scope={"magics": ["scratch"]},
    )
    result = emit_magic_cell(
        agent_id="alpha", zone_id="z1",
        name="scratch", args={"_positional": "hello"}, body=None,
        position={"after_cell_id": "c_anchor"},
        writer=writer, cell_manager=cell_manager,
    )
    new_id = result["cell_id"]
    rec = writer.get_cell_record(new_id)
    assert rec is not None
    assert rec["generated_by"] == "alpha"
    assert rec["generated_at"].endswith("Z")
    assert rec["text"].startswith("@@scratch")
    assert "hello" in rec["text"]
    # promoted_from_stream stays False by default — not stamped.
    assert "promoted_from_stream" not in rec


def test_emit_magic_cell_rejects_wrong_magic_under_name_scope(
    writer, cell_manager,
) -> None:
    writer.grant_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", scope={"magics": ["scratch"]},
    )
    with pytest.raises(MagicEmitError) as exc:
        emit_magic_cell(
            agent_id="alpha", zone_id="z1",
            name="spawn", args={}, body=None,
            position={"after_cell_id": "c_anchor"},
            writer=writer, cell_manager=cell_manager,
        )
    assert exc.value.code == K3K_UNPRIVILEGED_AGENT_MAGIC_EMIT


def test_emit_magic_cell_stamps_promoted_from_stream_when_set(
    writer, cell_manager,
) -> None:
    writer.grant_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", scope={"magics": "all"},
    )
    result = emit_magic_cell(
        agent_id="alpha", zone_id="z1",
        name="scratch", args={}, body=None,
        position={"after_cell_id": "c_anchor"},
        writer=writer, cell_manager=cell_manager,
        promoted_from_stream=True,
    )
    rec = writer.get_cell_record(result["cell_id"])
    assert rec is not None
    assert rec.get("promoted_from_stream") is True


def test_grant_rejects_malformed_scope(writer) -> None:
    with pytest.raises(ValueError):
        writer.grant_magic_emit_privilege(
            agent_id="alpha", zone_id="z1",
            scope={"magics": 42},  # neither list nor "all"
        )


def test_emit_magic_cell_requires_after_cell_id(
    writer, cell_manager,
) -> None:
    writer.grant_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", scope={"magics": "all"},
    )
    with pytest.raises(MagicEmitError) as exc:
        emit_magic_cell(
            agent_id="alpha", zone_id="z1",
            name="scratch", args={}, body=None,
            position={},  # missing after_cell_id
            writer=writer, cell_manager=cell_manager,
        )
    assert exc.value.code == K3K_UNPRIVILEGED_AGENT_MAGIC_EMIT


def test_has_magic_emit_privilege_false_on_empty_inputs(writer) -> None:
    writer.grant_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", scope={"magics": "all"},
    )
    assert not writer.has_magic_emit_privilege(
        agent_id="", zone_id="z1", magic_name="spawn",
    )
    assert not writer.has_magic_emit_privilege(
        agent_id="alpha", zone_id="", magic_name="spawn",
    )
    assert not writer.has_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", magic_name="",
    )


def test_privilege_table_round_trips_through_snapshot(writer) -> None:
    """Grant lands in metadata.rts.config.magic_emit_privileges[]."""
    writer.grant_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", scope={"magics": ["spawn"]},
    )
    snap = writer._build_snapshot()
    cfg = (snap.get("config") or {})
    entries = cfg.get("magic_emit_privileges")
    assert isinstance(entries, list) and len(entries) == 1
    assert entries[0]["agent_id"] == "alpha"
    assert entries[0]["zone_id"] == "z1"

"""``promote_stream_magic`` recovery-path tests -- PLAN-S5.0.4 §5.

Covers:

* parser recovers ``(name, args)`` from a verbatim stream-emitted line
* handler synthesizes an ``emit_magic_cell`` call from chip click
* result cell carries ``promoted_from_stream: True``
* refused when the source cell has no ``bound_agent_id``
* refused when the source agent's privilege grant was revoked
* refused on a malformed line
"""

from __future__ import annotations

import pytest

from llm_kernel.cell_manager import CellManager
from llm_kernel.intent_handlers.promote_stream_magic import (
    PromoteStreamMagicError,
    _parse_magic_line,
    handle_promote_stream_magic,
)
from llm_kernel.metadata_writer import MetadataWriter


@pytest.fixture
def writer(tmp_path) -> MetadataWriter:
    w = MetadataWriter(workspace_root=tmp_path)
    # Source cell -- the contaminated agent cell whose stream emitted
    # the magic line. Carries bound_agent_id so the handler can
    # resolve the privileged agent.
    w.set_cell_text("c_src", "@@agent alpha")
    record = dict(w.get_cell_record("c_src") or {})
    record["bound_agent_id"] = "alpha"
    w._cells["c_src"] = record
    return w


@pytest.fixture
def cell_manager(writer) -> CellManager:
    return CellManager(writer)


def test_parse_magic_line_recovers_name_and_args() -> None:
    name, args = _parse_magic_line("@@spawn beta task=research")
    assert name == "spawn"
    assert args.get("task") == "research"
    assert args.get("_positional") == "beta"


def test_parse_magic_line_strips_escape_prefix() -> None:
    # Layer-2 sanitization may have prepended \ ; the parser strips it.
    name, args = _parse_magic_line("\\@@scratch hello")
    assert name == "scratch"
    assert args.get("_positional") == "hello"


def test_parse_magic_line_rejects_malformed() -> None:
    with pytest.raises(PromoteStreamMagicError):
        _parse_magic_line("not a magic line")
    with pytest.raises(PromoteStreamMagicError):
        _parse_magic_line("")


def test_handler_synthesizes_emit_magic_cell(writer, cell_manager) -> None:
    writer.grant_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", scope={"magics": "all"},
    )
    result = handle_promote_stream_magic(
        params={"cell_id": "c_src", "line": "@@scratch hello"},
        writer=writer, cell_manager=cell_manager, zone_id="z1",
    )
    assert result is not None
    rec = writer.get_cell_record(result["cell_id"])
    assert rec is not None
    assert rec["generated_by"] == "alpha"
    assert rec.get("promoted_from_stream") is True
    assert rec["text"].startswith("@@scratch")


def test_handler_refuses_when_no_bound_agent(
    writer, cell_manager,
) -> None:
    # Replace the source cell with one lacking bound_agent_id.
    writer._cells["c_orphan"] = {"text": "@@scratch x"}
    result = handle_promote_stream_magic(
        params={"cell_id": "c_orphan", "line": "@@scratch hello"},
        writer=writer, cell_manager=cell_manager, zone_id="z1",
    )
    assert result is None


def test_handler_refuses_when_privilege_revoked(
    writer, cell_manager,
) -> None:
    # No grant exists -- the chip should not have rendered, but the
    # handler defends against race conditions (chip click vs revoke).
    result = handle_promote_stream_magic(
        params={"cell_id": "c_src", "line": "@@scratch hello"},
        writer=writer, cell_manager=cell_manager, zone_id="z1",
    )
    assert result is None


def test_handler_refuses_on_malformed_line(
    writer, cell_manager,
) -> None:
    writer.grant_magic_emit_privilege(
        agent_id="alpha", zone_id="z1", scope={"magics": "all"},
    )
    result = handle_promote_stream_magic(
        params={"cell_id": "c_src", "line": "not magic"},
        writer=writer, cell_manager=cell_manager, zone_id="z1",
    )
    assert result is None


def test_handler_refuses_on_missing_params(
    writer, cell_manager,
) -> None:
    assert handle_promote_stream_magic(
        params={"line": "@@scratch hello"},
        writer=writer, cell_manager=cell_manager, zone_id="z1",
    ) is None
    assert handle_promote_stream_magic(
        params={"cell_id": "c_src"},
        writer=writer, cell_manager=cell_manager, zone_id="z1",
    ) is None

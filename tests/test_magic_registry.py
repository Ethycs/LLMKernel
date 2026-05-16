"""Tests for ``llm_kernel.magic_registry`` — BSP-005 S5.0."""

from __future__ import annotations

import pytest

from llm_kernel import magic_registry as MR
from llm_kernel.cell_text import parse_cell


# --- registry shape --------------------------------------------------


def test_cell_magics_registered() -> None:
    """All V1 cell magics enumerated in PLAN §3.3 are present."""
    expected_v1 = {
        "agent", "spawn", "markdown", "scratch", "checkpoint",
        "endpoint", "compare", "section", "tool", "artifact", "native",
        "break",
    }
    missing = expected_v1 - set(MR.CELL_MAGICS.keys())
    assert not missing, f"missing CELL_MAGICS entries: {missing}"


def test_line_magics_registered() -> None:
    """All V1 line magics enumerated in PLAN §3.4 are present."""
    expected_v1 = {
        "pin", "unpin", "exclude", "include", "mark",
        "affinity", "handoff", "status",
        "revert", "stop", "branch",
    }
    missing = expected_v1 - set(MR.LINE_MAGICS.keys())
    assert not missing, f"missing LINE_MAGICS entries: {missing}"


def test_reserved_names_union() -> None:
    """RESERVED_NAMES is the union of cell + line magic keys."""
    assert MR.RESERVED_NAMES >= set(MR.CELL_MAGICS.keys())
    assert MR.RESERVED_NAMES >= set(MR.LINE_MAGICS.keys())


def test_is_reserved_name_llmnb_prefix() -> None:
    """The ``llmnb_*`` future-reservation prefix is reserved."""
    assert MR.is_reserved_name("llmnb_internal")
    assert MR.is_reserved_name("llmnb_x")
    # A normal-looking id is not reserved.
    assert not MR.is_reserved_name("alpha")
    assert not MR.is_reserved_name("zone-1/beta")


def test_k32_on_reserved_agent_id_via_supervisor() -> None:
    """AgentSupervisor.spawn raises K32 when agent_id is reserved."""
    from llm_kernel._provisioning import PreSpawnValidationError
    from llm_kernel.agent_supervisor import AgentSupervisor

    # Build a minimal supervisor — we only need spawn() to reach the
    # K32 guard before any real work happens.
    sup = AgentSupervisor.__new__(AgentSupervisor)
    # The K32 check is the FIRST thing spawn does after _diagnostics —
    # we don't need full ctor wiring.

    with pytest.raises(PreSpawnValidationError) as ei:
        sup.spawn(
            zone_id="z1", agent_id="pin", task="x",
            work_dir=__import__("pathlib").Path.cwd(),
        )
    msg = str(ei.value)
    assert "K32" in msg
    assert ei.value.log_signature == "reserved_magic_name_as_agent_id"


def test_k32_on_break_agent_id() -> None:
    """``@@spawn break`` would shadow the splitter — rejected."""
    from llm_kernel._provisioning import PreSpawnValidationError
    from llm_kernel.agent_supervisor import AgentSupervisor

    sup = AgentSupervisor.__new__(AgentSupervisor)
    with pytest.raises(PreSpawnValidationError) as ei:
        sup.spawn(
            zone_id="z1", agent_id="break", task="x",
            work_dir=__import__("pathlib").Path.cwd(),
        )
    assert "K32" in str(ei.value)


# --- handler effects -------------------------------------------------


def test_spawn_cell_magic_handler_extracts_args() -> None:
    """``@@spawn alpha endpoint:cheap task:"X"`` extracts the named args."""
    cell = parse_cell('@@spawn alpha endpoint:cheap task:"X"')
    assert cell.kind == "spawn"
    assert cell.args.get("agent_id") == "alpha"
    assert cell.args.get("endpoint") == "cheap"
    assert cell.args.get("task") == "X"


def test_endpoint_cell_magic_handler_extracts_args() -> None:
    """``@@endpoint cheap provider:openai model:gpt-4o-mini`` parses."""
    cell = parse_cell(
        "@@endpoint cheap provider:openai model:gpt-4o-mini"
    )
    assert cell.kind == "endpoint"
    assert cell.args.get("endpoint_name") == "cheap"
    assert cell.args.get("provider") == "openai"
    assert cell.args.get("model") == "gpt-4o-mini"


def test_stub_cell_magic_marks_pending() -> None:
    """A stub cell magic (``@@compare``) marks the cell pending."""
    cell = parse_cell("@@compare endpoints:a,b\nshared body")
    assert cell.kind == "compare"
    assert cell.args.get("_pending") is True
    assert cell.args.get("endpoints") == ["a", "b"]


def test_mark_line_magic_flips_kind() -> None:
    """``@mark scratch`` flips kind from agent to scratch."""
    cell = parse_cell("@@agent alpha\n@mark scratch\nbody")
    assert cell.kind == "scratch"


def test_mark_line_magic_unknown_kind_raises_k34() -> None:
    """``@mark xyzzy`` raises K34 (incompatible kind change)."""
    from llm_kernel.cell_text import (
        CellParseError,
        K34_INCOMPATIBLE_KIND_CHANGE,
    )

    with pytest.raises(CellParseError) as ei:
        parse_cell("@@agent alpha\n@mark xyzzy\nbody")
    assert ei.value.code == K34_INCOMPATIBLE_KIND_CHANGE


# --- S5b: @revert line-magic is active -----------------------------------


def test_revert_line_magic_dispatches_agent_revert_envelope() -> None:
    """``@revert alpha to t_2`` is active: recorded to cell.line_magics."""
    # @revert is now active (not stub); parse_cell on a cell containing
    # @revert should record ("revert", "alpha to t_2") in cell.line_magics
    # so the kernel routing layer can ship an agent_revert envelope.
    cell = parse_cell("@@agent alpha\n@revert alpha to t_2\nbody text")
    assert ("revert", "alpha to t_2") in cell.line_magics, cell.line_magics
    # Confirm status is active (no _pending flag injected by the handler).
    # The handler does not set _pending when status == "active".
    assert MR.LINE_MAGICS["revert"].status == "active"
    assert not getattr(MR.LINE_MAGICS["revert"], "pending_slice", None)


def test_stop_line_magic_dispatches_agent_stop_envelope() -> None:
    """``@stop alpha`` is active: recorded to cell.line_magics."""
    # @stop is now active (not stub); parse_cell on a cell containing
    # @stop should record ("stop", "alpha") in cell.line_magics so the
    # kernel routing layer can ship an agent_stop envelope.
    cell = parse_cell("@@agent alpha\n@stop alpha\nbody text")
    assert ("stop", "alpha") in cell.line_magics, cell.line_magics
    # Confirm status is active (no _pending flag injected by the handler).
    assert MR.LINE_MAGICS["stop"].status == "active"
    assert not getattr(MR.LINE_MAGICS["stop"], "pending_slice", None)


def test_branch_line_magic_dispatches_agent_branch_envelope() -> None:
    """``@branch alpha at t_2 as beta`` is active: recorded to cell.line_magics."""
    # @branch is now active (not stub); parse_cell on a cell containing
    # @branch should record ("branch", "alpha at t_2 as beta") in
    # cell.line_magics so the kernel routing layer can ship an
    # agent_branch envelope {source_agent, at_turn_id, new_agent_id, cell_id}.
    cell = parse_cell("@@agent alpha\n@branch alpha at t_2 as beta\nbody text")
    assert ("branch", "alpha at t_2 as beta") in cell.line_magics, cell.line_magics
    # Confirm status is active (no _pending flag injected by the handler).
    assert MR.LINE_MAGICS["branch"].status == "active"
    assert not getattr(MR.LINE_MAGICS["branch"], "pending_slice", None)


# PLAN-S5.5 Phase 4 — @@section cell magic.


def test_section_magic_active_post_phase4() -> None:
    """``@@section`` is no longer status="stub" — Phase 4 flipped it
    active and the parser routes ``@@section <title>`` to typed args."""
    assert MR.CELL_MAGICS["section"].status == "active"
    assert not getattr(MR.CELL_MAGICS["section"], "pending_slice", "")


def test_section_magic_extracts_title_from_positional() -> None:
    cell = parse_cell("@@section Architecture\nnotes about the section")
    assert cell.kind == "section"
    assert cell.args.get("title") == "Architecture"
    # Back-compat alias is preserved for pre-Phase-4 callers.
    assert cell.args.get("section_name") == "Architecture"


def test_section_magic_extracts_quoted_title() -> None:
    """Quoted titles survive shlex tokenization."""
    cell = parse_cell('@@section "Runtime Concerns"\nnotes')
    assert cell.kind == "section"
    assert cell.args.get("title") == "Runtime Concerns"


def test_section_magic_named_title_kwarg() -> None:
    """``title:`` named arg works in place of positional."""
    cell = parse_cell('@@section title:"Architecture"\nnotes')
    assert cell.kind == "section"
    assert cell.args.get("title") == "Architecture"


def test_section_magic_extracts_explicit_id() -> None:
    """``id:"sec_xxx"`` named arg pins the section_id."""
    cell = parse_cell('@@section Tests id:"sec_tests_42"\nbody')
    assert cell.kind == "section"
    assert cell.args.get("title") == "Tests"
    assert cell.args.get("section_id") == "sec_tests_42"


def test_section_magic_no_title_still_classifies_as_section() -> None:
    """A bare ``@@section`` cell parses as kind=section but lacks title.
    Runtime dispatch is responsible for failing the create gracefully."""
    cell = parse_cell("@@section\nbody")
    assert cell.kind == "section"
    assert "title" not in cell.args
    assert "section_id" not in cell.args

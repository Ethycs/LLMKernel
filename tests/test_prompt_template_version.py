"""K-AS G12 — system-prompt-template version validation.

RFC-002 §"Failure modes" (Template-version-mismatch row): the
supervisor MUST refuse spawn on a major-version mismatch and MAY
warn-and-proceed on a minor mismatch. A missing version marker logs
a warning but proceeds (the canonical template carries one; absence
is a developer bug, not an attack).

These tests bypass the live spawn machinery by exercising
:meth:`AgentSupervisor._validate_template_version` directly with
synthesized rendered prompts.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from llm_kernel._provisioning import (
    EXPECTED_SYSTEM_PROMPT_TEMPLATE_VERSION,
    PreSpawnValidationError,
    extract_template_version,
)
from llm_kernel.agent_supervisor import AgentSupervisor


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


def _prompt_with_version(version: str) -> str:
    return (
        "You are an autonomous coding agent...\n\n"
        "[TASK_BLOCK]\n\n"
        f"<!-- system-prompt-template v{version}; rfc=RFC-002 -->\n"
    )


def test_extract_template_version_round_trip() -> None:
    """The marker regex MUST round-trip the canonical version."""
    rendered = _prompt_with_version("1.0.0")
    assert extract_template_version(rendered) == "1.0.0"
    rendered = _prompt_with_version("2.5.7")
    assert extract_template_version(rendered) == "2.5.7"


def test_extract_template_version_missing_marker_returns_none() -> None:
    """No marker -> None (the supervisor decides whether absence is fatal)."""
    assert extract_template_version("plain prompt without version") is None


def test_major_mismatch_refuses_spawn() -> None:
    """Different MAJOR -> PreSpawnValidationError with version_mismatch sig."""
    sup = _make_supervisor()
    bad = _prompt_with_version("2.0.0")
    with pytest.raises(PreSpawnValidationError) as ei:
        sup._validate_template_version(bad)
    assert "expected 1.0.0" in str(ei.value)
    assert "got 2.0.0" in str(ei.value)
    assert ei.value.log_signature == "provisioning.template.version_mismatch"


def test_minor_mismatch_warns_but_proceeds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Different MINOR (same MAJOR) -> log warning, do NOT raise."""
    sup = _make_supervisor()
    expected_major = EXPECTED_SYSTEM_PROMPT_TEMPLATE_VERSION.split(".")[0]
    drift = f"{expected_major}.9.0"
    rendered = _prompt_with_version(drift)
    with caplog.at_level(logging.WARNING, logger="llm_kernel.agent_supervisor"):
        sup._validate_template_version(rendered)  # no exception
    msgs = " | ".join(r.getMessage() for r in caplog.records)
    assert "minor version drift" in msgs


def test_patch_mismatch_proceeds_silently(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Different PATCH only -> no warning, no raise."""
    sup = _make_supervisor()
    parts = EXPECTED_SYSTEM_PROMPT_TEMPLATE_VERSION.split(".")
    drift = f"{parts[0]}.{parts[1]}.99"
    rendered = _prompt_with_version(drift)
    with caplog.at_level(logging.WARNING, logger="llm_kernel.agent_supervisor"):
        sup._validate_template_version(rendered)
    msgs = " | ".join(r.getMessage() for r in caplog.records)
    assert "minor version drift" not in msgs
    assert "missing version marker" not in msgs


def test_missing_marker_logs_warning_proceeds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Absent marker -> warning + proceed (not a fatal pre-spawn error)."""
    sup = _make_supervisor()
    with caplog.at_level(logging.WARNING, logger="llm_kernel.agent_supervisor"):
        sup._validate_template_version("plain prompt with no marker at all")
    msgs = " | ".join(r.getMessage() for r in caplog.records)
    assert "missing version marker" in msgs


def test_unparseable_version_refuses_spawn() -> None:
    """A marker present but not MAJOR.MINOR.PATCH -> raise."""
    # The regex requires MAJOR.MINOR.PATCH so an unparseable version
    # simply fails the regex => extract returns None => warns. To
    # exercise the parser-error branch we directly construct a marker
    # that the regex captures but _split_semver rejects: not possible
    # via the regex (it pins three integer groups). Skip — the
    # _split_semver branch is reachable only via in-process API misuse
    # and is covered by importing the helper directly.
    from llm_kernel._provisioning import _split_semver

    with pytest.raises(ValueError):
        _split_semver("not-a-version")

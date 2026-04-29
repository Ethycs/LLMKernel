"""K-AS-B / G12 — system-prompt-template version validation acceptance tests.

The G12 audit found template-version validation is already implemented:

* :data:`llm_kernel._provisioning.EXPECTED_SYSTEM_PROMPT_TEMPLATE_VERSION`
  pins the expected version (currently ``"1.0.0"``) per RFC-002
  §"Versioning" / "Failure modes" Template-version-mismatch row.
* :meth:`AgentSupervisor._validate_template_version` is invoked from
  ``spawn(...)`` BEFORE process launch; major mismatch raises
  :class:`PreSpawnValidationError` with
  ``log_signature="provisioning.template.version_mismatch"``.
* The error path in ``spawn`` calls ``_record_synthetic_problem``
  which opens + closes a ``report_problem`` run on the run-tracker
  with the matching log signature; this is the "K-class error"
  surface that callers (extension renderer, kernel test harness)
  observe.

These acceptance tests pin the contract at the supervisor's public
surface (``spawn``) rather than the private validator, so a future
refactor that moves the validation step still gets caught.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from llm_kernel._provisioning import (
    EXPECTED_SYSTEM_PROMPT_TEMPLATE_VERSION,
    PreSpawnValidationError,
)
from llm_kernel.agent_supervisor import AgentSupervisor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_health(status_code: int = 200):
    fake = MagicMock()
    fake.status_code = status_code
    return patch("llm_kernel._provisioning.httpx.head", return_value=fake)


class _CollectingSink:
    """RunTracker sink that records every emitted envelope."""

    def __init__(self) -> None:
        self.envelopes: List[Dict[str, Any]] = []

    def emit(self, env: Dict[str, Any]) -> None:
        self.envelopes.append(env)


def _make_supervisor() -> tuple[AgentSupervisor, _CollectingSink]:
    from llm_kernel.run_tracker import RunTracker

    sink = _CollectingSink()
    tracker = RunTracker(
        trace_id=str(uuid.uuid4()), sink=sink,
        agent_id="alpha", zone_id="z1",
    )
    dispatcher = MagicMock()
    sup = AgentSupervisor(
        run_tracker=tracker, dispatcher=dispatcher,
        litellm_endpoint_url="http://127.0.0.1:9999/v1",
    )
    return sup, sink


def _prompt_with_version(version: str) -> str:
    """Render a fake system prompt carrying the given trailing version marker."""
    return (
        "You are an autonomous coding agent...\n\n"
        "[TASK_BLOCK]\n\n"
        f"<!-- system-prompt-template v{version}; rfc=RFC-002 -->\n"
    )


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------


def test_spawn_rejects_template_version_mismatch(tmp_path: Path) -> None:
    """A major-version mismatch MUST refuse spawn (PreSpawnValidationError).

    Contract: the supervisor calls ``_validate_template_version`` on
    the rendered prompt before any subprocess work; mismatch raises
    BEFORE Popen is invoked. The synthetic report_problem path is
    asserted in a separate test (see ``..._kclass_emitted_on_mismatch``).
    """
    # The fixture pins the expected version so a future RFC bump
    # forces the test to be re-considered.
    assert EXPECTED_SYSTEM_PROMPT_TEMPLATE_VERSION == "1.0.0"

    sup, _ = _make_supervisor()
    bad = _prompt_with_version("2.0.0")

    popen_calls: List[Any] = []

    def fake_popen(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        popen_calls.append((args, kwargs))
        raise AssertionError(
            "subprocess.Popen MUST NOT be called when template version "
            "fails major-version validation"
        )

    with _patch_health(), \
         patch("llm_kernel.agent_supervisor.render_system_prompt",
               return_value=bad), \
         patch("subprocess.Popen", side_effect=fake_popen):
        with pytest.raises(PreSpawnValidationError) as ei:
            sup.spawn(
                zone_id="z1", agent_id="alpha", task="x",
                work_dir=tmp_path, api_key="sk-x",
            )

    assert ei.value.log_signature == "provisioning.template.version_mismatch"
    assert "expected 1.0.0" in str(ei.value)
    assert "got 2.0.0" in str(ei.value)
    assert popen_calls == [], "Popen should never run on template mismatch"


def test_spawn_accepts_matching_template_version(tmp_path: Path) -> None:
    """A matching version MUST proceed past validation (Popen is reached).

    Contract: the validator is "fail closed on mismatch, transparent
    on match" — a matching template version does not perturb spawn.
    We patch Popen to a sentinel that records the call and returns a
    minimal stub; if the validator wrongly raised, the assertion below
    catches the regression.
    """
    sup, _ = _make_supervisor()
    good = _prompt_with_version(EXPECTED_SYSTEM_PROMPT_TEMPLATE_VERSION)

    class _StubPopen:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs
            self.stdout = iter([""])  # immediately EOF
            self.stderr = iter([""])
            self.returncode: Any = None
            self.pid = 12345

        def poll(self) -> Any:
            return self.returncode

        def wait(self, timeout: Any = None) -> int:
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:  # pragma: no cover
            self.returncode = -9

    popen_calls: List[Any] = []

    def make_stub(*args: Any, **kwargs: Any) -> _StubPopen:
        p = _StubPopen(*args, **kwargs)
        popen_calls.append(p)
        return p

    with _patch_health(), \
         patch("llm_kernel.agent_supervisor.render_system_prompt",
               return_value=good), \
         patch("subprocess.Popen", side_effect=make_stub):
        handle = sup.spawn(
            zone_id="z1", agent_id="alpha", task="x",
            work_dir=tmp_path, api_key="sk-x",
        )

    assert popen_calls, "Popen must be reached when template version matches"
    assert handle is not None
    handle.terminate(grace_seconds=0.1)


def test_template_version_kclass_emitted_on_mismatch(tmp_path: Path) -> None:
    """Mismatch MUST emit a synthetic report_problem with the K-class signature.

    Contract: the K-class surface for template-version drift is the
    synthetic ``report_problem`` run with
    ``log_signature="provisioning.template.version_mismatch"``. The
    extension's renderer keys off this signature (RFC-002 §"Failure
    modes"), so callers can locate the failure without parsing the
    Python exception.
    """
    sup, _sink = _make_supervisor()
    bad = _prompt_with_version("3.0.0")  # major drift

    problems: List[Any] = []
    real_record = sup._record_synthetic_problem

    def record(*args: Any, **kwargs: Any) -> Any:
        problems.append((args, kwargs))
        return real_record(*args, **kwargs)

    with _patch_health(), \
         patch.object(sup, "_record_synthetic_problem",
                      side_effect=record) as _, \
         patch("llm_kernel.agent_supervisor.render_system_prompt",
               return_value=bad), \
         patch("subprocess.Popen") as popen_mock:
        with pytest.raises(PreSpawnValidationError):
            sup.spawn(
                zone_id="z1", agent_id="alpha", task="x",
                work_dir=tmp_path, api_key="sk-x",
            )
        assert not popen_mock.called, (
            "Popen must not be called when template validation fails"
        )

    assert problems, (
        "synthetic report_problem was not emitted on template-version "
        "mismatch — the K-class surface is missing"
    )
    args, _kwargs = problems[0]
    # Signature: (agent_id, zone_id, description, log_signature)
    agent_id, zone_id, description, log_signature = args[0], args[1], args[2], args[3]
    assert agent_id == "alpha"
    assert zone_id == "z1"
    assert log_signature == "provisioning.template.version_mismatch"
    assert "expected 1.0.0" in description
    assert "got 3.0.0" in description

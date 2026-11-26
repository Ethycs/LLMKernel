"""Fixtures private to the conformance suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _conformance_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    """Hermetic environment for every conformance test.

    * External proxy lifecycle -- no kernel-owned LiteLLM / passthrough
      process is started by an entry point under test.
    * No operator pin leaks in from the developer's shell: hash-mode
      tests set ``LLMNB_OPERATOR_PIN`` themselves.
    * Run-identity overrides from the shell must not leak into the facade.
    """
    monkeypatch.setenv("LLMKERNEL_LITELLM_ENDPOINT_URL", "http://127.0.0.1:1/v1")
    # Lifecycle diagnostics markers go to the test's tmp dir, not the repo.
    monkeypatch.setenv("LLMNB_E2E_MARKER_FILE", str(tmp_path / "kernel-markers.jsonl"))
    for name in (
        "LLMNB_OPERATOR_PIN", "LLMKERNEL_RUN_TRACE_ID", "LLMKERNEL_AGENT_ID",
        "LLMKERNEL_ZONE_ID", "LLMKERNEL_USE_PASSTHROUGH", "LLMKERNEL_PTY_MODE",
        "LLMKERNEL_IPC_SOCKET", "LLMKERNEL_SESSION_ID",
    ):
        monkeypatch.delenv(name, raising=False)

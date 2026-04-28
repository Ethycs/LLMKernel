"""RFC-009 §9 tests — zone_control resolution and precedence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest

from llm_kernel import zone_control


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip all RFC-009 env vars so tests start from a known baseline."""
    for var in (
        zone_control.ENV_CLAUDE_BIN,
        zone_control.ENV_USE_BARE,
        zone_control.ENV_USE_PASSTHROUGH,
        zone_control.ENV_MODEL_OVERRIDE,
        zone_control.ENV_ANTHROPIC_API_KEY,
        zone_control.ENV_MARKER_FILE,
    ):
        monkeypatch.delenv(var, raising=False)
    yield


# ---------------------------------------------------------------------------
# resolve_zone_config — happy path with only defaults
# ---------------------------------------------------------------------------


def test_resolve_with_only_default(clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No env vars + no claude on PATH + cwd outside any pixi env →
    ZoneConfig carries defaults (None / False / empty string)."""
    # cwd to tmp_path so the pixi probe finds nothing.
    monkeypatch.chdir(tmp_path)
    # Strip PATH so shutil.which("claude") returns None.
    monkeypatch.setenv("PATH", "")
    cfg = zone_control.resolve_zone_config()
    assert cfg.claude_bin is None
    assert cfg.use_bare is False
    assert cfg.use_passthrough is False
    assert cfg.model_override is None
    assert cfg.anthropic_api_key == ""
    assert cfg.marker_file == ""


# ---------------------------------------------------------------------------
# Booleans — env beats default (RFC-009 §4.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_value,expected", [
    ("1", True),
    ("true", True),
    ("True", True),
    ("yes", True),
    ("on", True),
    ("0", False),
    ("false", False),
    ("", False),
])
def test_env_var_beats_default_use_bare(
    clean_env: None, monkeypatch: pytest.MonkeyPatch,
    env_value: str, expected: bool,
) -> None:
    """LLMKERNEL_USE_BARE truthy values flip use_bare to True per §4.1."""
    monkeypatch.setenv(zone_control.ENV_USE_BARE, env_value)
    assert zone_control.effective_use_bare() is expected


def test_env_var_beats_default_use_passthrough(
    clean_env: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(zone_control.ENV_USE_PASSTHROUGH, "1")
    assert zone_control.effective_use_passthrough() is True


def test_env_var_beats_default_model_override(
    clean_env: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(zone_control.ENV_MODEL_OVERRIDE, "claude-haiku-4-5")
    assert zone_control.effective_model_override() == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# locate_claude_bin — env > PATH > pixi probe (RFC-009 §4.2)
# ---------------------------------------------------------------------------


def test_locate_claude_env_var_beats_path(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """LLMNB_CLAUDE_BIN pointing at a real file beats PATH."""
    fake_claude = tmp_path / "claude.cmd"
    fake_claude.write_text("@echo fake claude")
    monkeypatch.setenv(zone_control.ENV_CLAUDE_BIN, str(fake_claude))
    # Even if PATH had a real claude, env should win — we verify by
    # asserting the returned path equals the env var's value.
    result = zone_control.locate_claude_bin()
    assert result == str(fake_claude)


def test_locate_claude_env_var_invalid_falls_through(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """LLMNB_CLAUDE_BIN pointing at a non-existent path falls through to
    PATH lookup (with a diagnostic mark per RFC-009 §8 K81 logic)."""
    monkeypatch.setenv(zone_control.ENV_CLAUDE_BIN, "/no/such/claude")
    monkeypatch.setenv("PATH", "")  # also kill PATH
    monkeypatch.chdir(tmp_path)  # no pixi env in tmp
    # Should fall through; tmp has no pixi env; result is None.
    assert zone_control.locate_claude_bin() is None


def test_locate_claude_path_then_pixi(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """When PATH hits, return the PATH value without probing pixi."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    is_win = os.name == "nt"
    claude_path = bin_dir / ("claude.cmd" if is_win else "claude")
    claude_path.write_text("@echo path-claude" if is_win else "#!/bin/sh")
    if not is_win:
        os.chmod(claude_path, 0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    result = zone_control.locate_claude_bin()
    assert result is not None
    assert Path(result).resolve() == claude_path.resolve()


def test_locate_claude_pixi_probe_walks_up(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """No env, no PATH, but a pixi env exists 2 levels up → probe finds it."""
    is_win = os.name == "nt"
    repo_root = tmp_path / "repo"
    pixi_env = repo_root / Path(*zone_control.PIXI_ENV_RELATIVE)
    bin_dir = pixi_env if is_win else pixi_env / "bin"
    bin_dir.mkdir(parents=True)
    claude_path = bin_dir / ("claude.cmd" if is_win else "claude")
    claude_path.write_text("fake")
    if not is_win:
        os.chmod(claude_path, 0o755)
    # cwd two levels deep within the repo
    deep = repo_root / "extension" / "test"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    monkeypatch.setenv("PATH", "")
    result = zone_control.locate_claude_bin()
    assert result is not None
    assert Path(result).resolve() == claude_path.resolve()
    # Side effect: env var was set, PATH was prepended.
    assert os.environ.get(zone_control.ENV_CLAUDE_BIN) == result
    assert str(bin_dir) in os.environ.get("PATH", "").split(
        ";" if is_win else ":"
    )


def test_locate_claude_no_source_returns_none(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Exhaustive miss → None; caller surfaces K83."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    assert zone_control.locate_claude_bin() is None


# ---------------------------------------------------------------------------
# Credentials — env-only (RFC-009 §4.4)
# ---------------------------------------------------------------------------


def test_anthropic_api_key_via_env(
    clean_env: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(zone_control.ENV_ANTHROPIC_API_KEY, "sk-test-123")
    assert zone_control.effective_anthropic_api_key() == "sk-test-123"


def test_anthropic_api_key_unset_returns_empty(clean_env: None) -> None:
    assert zone_control.effective_anthropic_api_key() == ""


# ---------------------------------------------------------------------------
# Diagnostic markers (RFC-009 §6 #4)
# ---------------------------------------------------------------------------


def test_diagnostic_marker_per_resolution(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Every resolver call emits a zone_control.resolved marker."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv(zone_control.ENV_USE_BARE, "1")
    captured: list = []

    def _capture(stage: str, **kwargs):
        captured.append((stage, kwargs))

    with patch("llm_kernel.zone_control._diagnostics") as mock_diag:
        mock_diag.mark = _capture
        zone_control.effective_use_bare()
        zone_control.locate_claude_bin()

    resolved_marks = [c for c in captured if c[0] == "zone_control.resolved"]
    assert len(resolved_marks) >= 2  # use_bare + claude_bin
    settings_recorded = {c[1]["setting"] for c in resolved_marks}
    assert "use_bare" in settings_recorded
    assert "claude_bin" in settings_recorded


def test_secrets_value_redacted_in_marker(
    clean_env: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credentials' VALUE must not appear in marker file (only "<set>" /
    "<unset>") per RFC-009 §6 #4 / §4.4."""
    monkeypatch.setenv(zone_control.ENV_ANTHROPIC_API_KEY, "sk-secret-do-not-leak")
    captured: list = []

    def _capture(stage: str, **kwargs):
        captured.append((stage, kwargs))

    with patch("llm_kernel.zone_control._diagnostics") as mock_diag:
        mock_diag.mark = _capture
        zone_control.effective_anthropic_api_key()

    relevant = [c for c in captured if c[0] == "zone_control.resolved"
                and c[1].get("setting") == "anthropic_api_key"]
    assert len(relevant) == 1
    assert relevant[0][1]["value"] == "<set>"
    # Belt-and-braces: the literal secret must not appear anywhere in
    # the marker payload.
    assert "sk-secret-do-not-leak" not in str(relevant[0][1])


# ---------------------------------------------------------------------------
# resolve_zone_config — full snapshot
# ---------------------------------------------------------------------------


def test_resolve_zone_config_full_snapshot(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """resolve_zone_config builds a frozen ZoneConfig with every field."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv(zone_control.ENV_USE_BARE, "1")
    monkeypatch.setenv(zone_control.ENV_MODEL_OVERRIDE, "haiku-4-5")
    monkeypatch.setenv(zone_control.ENV_ANTHROPIC_API_KEY, "sk-x")

    cfg = zone_control.resolve_zone_config()

    assert cfg.claude_bin is None  # no PATH, no pixi env
    assert cfg.use_bare is True
    assert cfg.use_passthrough is False
    assert cfg.model_override == "haiku-4-5"
    assert cfg.anthropic_api_key == "sk-x"
    assert cfg.marker_file == ""

    # Frozen — assigning to a field should raise.
    with pytest.raises((AttributeError, TypeError)):
        cfg.use_bare = False  # type: ignore[misc]

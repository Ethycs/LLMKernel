"""Zone control — RFC-009 §6 implementation (kernel side).

Single entry point for resolving any kernel-side setting per the
RFC-009 precedence rules:

  1. CLI arg                    (kernel only; per-spawn override)
  2. Process env var
  3. VS Code workspace settings (passed through env from extension)
  4. VS Code user settings      (passed through env from extension)
  5. metadata.rts.config        (kernel only)
  6. package.json default       (passed through env from extension)
  7. Module-level constant      (defined here)
  8. Discovery probe            (filesystem walk; last resort)

The extension-side counterpart is the preflight + activation glue that
exports settings into env vars before spawning the kernel
(``node-pty.spawn(... env ...)``). That contract is mirrored in
extension/test/util/preflight.ts so both production and test paths
follow the same rules.

Each resolution emits a ``zone_control.resolved`` diagnostic mark with
``{setting, value, source}`` so an operator can trace where any
effective value came from by reading the marker file. This is the
single most useful debugging affordance for "why is this setting not
taking effect?" — instead of guessing, read the marker.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import _diagnostics


# ---------------------------------------------------------------------------
# Naming conventions (RFC-009 §5)
# ---------------------------------------------------------------------------

# Env var names for kernel-side settings. These follow the LLMNB_* prefix
# for cross-cutting (extension AND kernel both read or write) and
# LLMKERNEL_* for kernel-only. The prefix split is intentional: an
# operator can grep their env once and tell which subsystem reads what.
ENV_CLAUDE_BIN = "LLMNB_CLAUDE_BIN"
ENV_PYTHON_BIN = "LLMNB_PYTHON_BIN"
ENV_USE_BARE = "LLMKERNEL_USE_BARE"
ENV_USE_PASSTHROUGH = "LLMKERNEL_USE_PASSTHROUGH"
ENV_MODEL_OVERRIDE = "LLMKERNEL_MODEL"
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_MARKER_FILE = "LLMNB_MARKER_FILE"

# Pixi env discovery probe configuration (RFC-009 §4.2 step 6).
PIXI_ENV_RELATIVE = (".pixi", "envs", "kernel")
PIXI_PROBE_MAX_LEVELS = 6


# ---------------------------------------------------------------------------
# Source attribution
# ---------------------------------------------------------------------------

#: Symbolic source labels emitted into the ``zone_control.resolved`` marker.
#: These match RFC-009 §3's priority list (lower number = higher priority).
SOURCE_CLI = "cli_arg"
SOURCE_ENV = "env"
SOURCE_VSCODE_WORKSPACE = "vscode_workspace_setting"
SOURCE_VSCODE_USER = "vscode_user_setting"
SOURCE_RTS_CONFIG = "metadata_rts_config"
SOURCE_PACKAGE_DEFAULT = "package_default"
SOURCE_MODULE_CONSTANT = "module_constant"
SOURCE_PROBE = "discovery_probe"
SOURCE_UNRESOLVED = "unresolved"


def _record(setting: str, value: Any, source: str) -> None:
    """Emit a zone_control.resolved diagnostic mark."""
    # Avoid logging secret-class settings' VALUES — only that they were set.
    is_secret = setting.endswith("api_key") or setting.endswith("token")
    if is_secret:
        _diagnostics.mark(
            "zone_control.resolved",
            setting=setting,
            value=("<set>" if value else "<unset>"),
            source=source,
        )
    else:
        _diagnostics.mark(
            "zone_control.resolved",
            setting=setting,
            value=value,
            source=source,
        )


# ---------------------------------------------------------------------------
# ZoneConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZoneConfig:
    """Effective zone configuration after RFC-009 precedence resolution.

    Frozen because: settings are read once at the relevant effective time
    (process boot or agent spawn) and re-resolution requires explicit
    invalidation (V1 doesn't support that — restart kernel to change).
    """

    #: Absolute path to claude executable, or None if unresolved.
    #: When None, ``AgentSupervisor.spawn`` raises K83 instead of trying to
    #: spawn an unfound binary.
    claude_bin: Optional[str]
    #: --bare flag for claude (forces ANTHROPIC_API_KEY auth path).
    use_bare: bool
    #: Use the kernel-owned passthrough proxy (BSP-001) for claude's
    #: model calls. ``False`` means use litellm proxy. Set via
    #: ``LLMKERNEL_USE_PASSTHROUGH=1``.
    use_passthrough: bool
    #: Optional model override; overrides the default Sonnet-4-5 the CLI
    #: picks. Used by tests pinning a small model.
    model_override: Optional[str]
    #: Anthropic API key. NEVER persisted to settings or metadata.rts.config
    #: (RFC-009 §4.4). Empty string when unset.
    anthropic_api_key: str
    #: Path to the diagnostic marker file. Used by tests + Pillar A
    #: typed waits. Empty string disables marker writing.
    marker_file: str


# ---------------------------------------------------------------------------
# locate_claude_bin — RFC-009 §4.2
# ---------------------------------------------------------------------------


def _probe_pixi_env_for(binary_name: str) -> Tuple[Optional[str], List[str]]:
    """Walk up from cwd looking for ``.pixi/envs/kernel/<binary>``.

    Returns ``(absolute_path, searched_paths)``. RFC-009 §4.2 step 6.
    """
    searched: List[str] = []
    cwd = Path.cwd()
    is_win = os.name == "nt"
    candidate_names = (
        [f"{binary_name}.cmd", f"{binary_name}.exe", binary_name]
        if is_win else [binary_name]
    )
    bin_subdir = "" if is_win else "bin"
    for level in range(PIXI_PROBE_MAX_LEVELS):
        env_dir = cwd / Path(*PIXI_ENV_RELATIVE)
        if bin_subdir:
            env_dir = env_dir / bin_subdir
        searched.append(str(env_dir))
        if env_dir.is_dir():
            for name in candidate_names:
                candidate = env_dir / name
                if candidate.is_file():
                    return (str(candidate), searched)
        parent = cwd.parent
        if parent == cwd:
            break
        cwd = parent
    return (None, searched)


def locate_claude_bin() -> Optional[str]:
    """Resolve the absolute path to the claude executable per RFC-009 §4.2.

    Priority: ``LLMNB_CLAUDE_BIN`` env > PATH lookup > pixi env probe.

    On a successful pixi-probe match, sets ``LLMNB_CLAUDE_BIN`` in the
    current process env so subsequent ``subprocess.Popen("claude", ...)``
    calls (which inherit env) don't need to re-probe. This is the
    side-effect pinned in RFC-009 §4.2's probe rules.
    """
    # 1) env var
    explicit = os.environ.get(ENV_CLAUDE_BIN, "").strip()
    if explicit:
        if Path(explicit).is_file():
            _record("claude_bin", explicit, SOURCE_ENV)
            return explicit
        # The env var was set but doesn't point at a real file. Don't
        # silently fall through — the operator's intent was explicit.
        # Falling through would mask their mistake.
        _diagnostics.mark(
            "zone_control_invalid_value",
            setting="claude_bin",
            value=explicit,
            reason="LLMNB_CLAUDE_BIN points at a non-file path",
        )
        # Per RFC-009 §8 K81 — but for V1 we treat "explicit-but-broken"
        # as fall-through with a warning. Production K83 handling at the
        # caller catches the ultimate "no claude" case.
    # 2) PATH lookup
    via_path = shutil.which("claude")
    if via_path:
        _record("claude_bin", via_path, SOURCE_PROBE)  # PATH is technically a probe too
        return via_path
    # 3) Pixi env probe
    probed, searched = _probe_pixi_env_for("claude")
    if probed:
        # Side effect: prepend the probed dir to PATH AND set the
        # LLMNB_CLAUDE_BIN env var so subprocess.Popen inheritance works
        # without each call site re-probing.
        env_dir = str(Path(probed).parent)
        sep = ";" if os.name == "nt" else ":"
        current_path = os.environ.get("PATH", "")
        if env_dir not in current_path.split(sep):
            os.environ["PATH"] = f"{env_dir}{sep}{current_path}"
        os.environ[ENV_CLAUDE_BIN] = probed
        _record("claude_bin", probed, SOURCE_PROBE)
        return probed
    _diagnostics.mark(
        "zone_control_discovery_failed",
        binary="claude",
        searched_paths=searched,
    )
    _record("claude_bin", None, SOURCE_UNRESOLVED)
    return None


# ---------------------------------------------------------------------------
# Boolean / mode-switch resolvers — RFC-009 §4.1
# ---------------------------------------------------------------------------


def _bool_from_env(name: str, default: bool = False) -> Tuple[bool, str]:
    """Resolve a boolean from env var with ``"1"`` / ``"true"`` truthy.

    Returns ``(value, source)``.
    """
    raw = os.environ.get(name)
    if raw is not None:
        truthy = raw.strip().lower() in ("1", "true", "yes", "on")
        return (truthy, SOURCE_ENV)
    return (default, SOURCE_PACKAGE_DEFAULT)


def effective_use_bare() -> bool:
    value, source = _bool_from_env(ENV_USE_BARE, default=False)
    _record("use_bare", value, source)
    return value


def effective_use_passthrough() -> bool:
    value, source = _bool_from_env(ENV_USE_PASSTHROUGH, default=False)
    _record("use_passthrough", value, source)
    return value


# ---------------------------------------------------------------------------
# String resolvers — RFC-009 §4.3
# ---------------------------------------------------------------------------


def effective_model_override() -> Optional[str]:
    raw = os.environ.get(ENV_MODEL_OVERRIDE, "").strip()
    if raw:
        _record("model_override", raw, SOURCE_ENV)
        return raw
    _record("model_override", None, SOURCE_PACKAGE_DEFAULT)
    return None


# ---------------------------------------------------------------------------
# Credential resolver — RFC-009 §4.4 (env-only)
# ---------------------------------------------------------------------------


def effective_anthropic_api_key() -> str:
    raw = os.environ.get(ENV_ANTHROPIC_API_KEY, "")
    _record("anthropic_api_key", raw, SOURCE_ENV)
    return raw


# ---------------------------------------------------------------------------
# Diagnostic resolver — RFC-009 §4.5 (env-only)
# ---------------------------------------------------------------------------


def effective_marker_file() -> str:
    raw = os.environ.get(ENV_MARKER_FILE, "")
    _record("marker_file", raw, SOURCE_ENV)
    return raw


# ---------------------------------------------------------------------------
# Top-level factory — RFC-009 §6 #2
# ---------------------------------------------------------------------------


def resolve_zone_config() -> ZoneConfig:
    """Build a :class:`ZoneConfig` snapshot from the current environment.

    Reads each setting through its dedicated resolver, which emits the
    ``zone_control.resolved`` diagnostic marker. The returned dataclass
    is frozen — re-resolution requires a new call.
    """
    return ZoneConfig(
        claude_bin=locate_claude_bin(),
        use_bare=effective_use_bare(),
        use_passthrough=effective_use_passthrough(),
        model_override=effective_model_override(),
        anthropic_api_key=effective_anthropic_api_key(),
        marker_file=effective_marker_file(),
    )


__all__ = [
    "ENV_CLAUDE_BIN",
    "ENV_USE_BARE",
    "ENV_USE_PASSTHROUGH",
    "ENV_MODEL_OVERRIDE",
    "ENV_ANTHROPIC_API_KEY",
    "ENV_MARKER_FILE",
    "PIXI_ENV_RELATIVE",
    "ZoneConfig",
    "SOURCE_CLI",
    "SOURCE_ENV",
    "SOURCE_VSCODE_WORKSPACE",
    "SOURCE_VSCODE_USER",
    "SOURCE_PACKAGE_DEFAULT",
    "SOURCE_PROBE",
    "SOURCE_UNRESOLVED",
    "locate_claude_bin",
    "effective_use_bare",
    "effective_use_passthrough",
    "effective_model_override",
    "effective_anthropic_api_key",
    "effective_marker_file",
    "resolve_zone_config",
]

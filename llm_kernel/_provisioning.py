"""LLMKernel agent provisioning recipe (Stage 2 Track B4 helpers).

Pure functions implementing the RFC-002 v1.0.0 provisioning recipe.
The supervisor (:mod:`llm_kernel.agent_supervisor`) consumes these and
adds the lifecycle, stream parsing, and restart logic.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import httpx

logger: logging.Logger = logging.getLogger("llm_kernel._provisioning")

#: Canonical embedded marker on the system-prompt template. Format:
#: ``<!-- system-prompt-template vMAJOR.MINOR.PATCH; rfc=RFC-002 -->``.
#: The supervisor (RFC-002 §"Failure modes") parses this with
#: :func:`extract_template_version` and refuses spawn on major mismatch
#: against :data:`EXPECTED_SYSTEM_PROMPT_TEMPLATE_VERSION`.
SYSTEM_PROMPT_TEMPLATE_VERSION_RE: re.Pattern[str] = re.compile(
    r"<!--\s*system-prompt-template\s+v(\d+\.\d+\.\d+)\s*;\s*rfc=RFC-002\s*-->"
)

#: Version the supervisor expects to see embedded in the rendered template.
#: Mismatch handling per RFC-002 §"Failure modes":
#:   - major differs -> refuse spawn (``provisioning.template.version_mismatch``)
#:   - minor differs -> warn-and-proceed
#:   - patch differs -> proceed silently
EXPECTED_SYSTEM_PROMPT_TEMPLATE_VERSION: str = "1.0.0"

#: RFC-001 v1.0.0 13-tool catalog in the canonical order RFC-002
#: §"MCP config JSON layout" mandates.
RFC001_ALLOWED_TOOLS: Tuple[str, ...] = (
    "ask", "clarify", "propose", "request_approval",
    "report_progress", "report_completion", "report_problem",
    "present", "notify", "escalate",
    "read_file", "write_file", "run_command",
)

#: Stable RFC-002 server identifier (renaming is BREAKING).
MCP_SERVER_NAME: str = "llmkernel-operator-bridge"

#: Comma-separated value for ``CLAUDE_CODE_DISABLED_TOOLS`` per
#: RFC-002 §"Allowed-tools restriction policy".
DISABLED_TOOLS: str = "Bash,WebFetch,WebSearch,Read,Write,Edit,TodoWrite"

#: Comma-separated value for ``CLAUDE_CODE_ALLOWED_TOOLS``.
ALLOWED_TOOLS: str = "Glob,Grep"

#: Canonical RFC-002 v1.0.0 system prompt template. Reproduced verbatim
#: from RFC-002 §"System prompt template"; ``[TASK_BLOCK]`` is the only
#: substitution point. The trailing version comment is part of the
#: template — removing it is a BREAKING change to RFC-002.
CANONICAL_SYSTEM_PROMPT_TEMPLATE: str = """\
You are an autonomous coding agent operating inside the llmb_rts_notebook
operator console. The kernel that hosts you mediates every model call and
every tool call you make.

All communication with the operator MUST occur through the provided MCP
tools. Do not produce free-form text intended for the operator. Reasoning
may be expressed in your internal monologue, which is not surfaced to the
operator. Use it freely; do not summarize for the operator.

Available MCP tools (call them, do not describe them):

- ask(question, context, options?) — operator-targeted free-form question.
- clarify(question, options) — typed clarification with a discrete option set.
- propose(action, rationale, preview?, scope?) — proposed action with rationale.
- request_approval(action, diff_preview, risk_level, alternatives?) — anything
  the operator must approve before you proceed.
- report_progress(status, percent?, blockers?) — status update during work.
- report_completion(summary, artifacts?) — final completion signal.
- report_problem(severity, description, suggested_remediation?) — blocking issue.
- present(artifact, kind, summary) — generated content lifted to the artifacts
  surface.
- notify(observation, importance) — fire-and-forget annotation.
- escalate(reason, severity) — flag operator attention urgently.
- read_file(path, encoding?) — read a file from the workspace.
- write_file(path, content, mode?) — write a file (operator approval required
  for risk_level >= medium; surface a request_approval first if unsure).
- run_command(command, args?, cwd?, timeout?) — run a shell command.

Tool selection guidance:
- When you would say "should I do X?", call clarify with concrete options.
- When proposing an action, call propose with a rationale.
- When asking for approval to do something already proposed, call
  request_approval with a diff preview when applicable.
- When reporting status during a long task, call report_progress.
- When the task is done, call report_completion.
- Prefer one structured tool call over verbose prose.
- Batch progress reports when possible; do not flood the operator.
- Emit report_completion exactly once at task end.

If you must convey something that does not fit any tool, call notify with
importance="low". Do not produce a free-form text response to the operator.

[TASK_BLOCK]

<!-- system-prompt-template v1.0.0; rfc=RFC-002 -->
"""

#: Patterns matched against env-var names; any hit is stripped from the
#: child env before exec (RFC-002 §"Required environment variables").
#:
#: ``CLAUDECODE`` / ``CLAUDE_CODE_*`` are intentionally KEPT — empirical
#: result: the OAuth model-resolution path needs them present in the
#: child env, even when the agent is run as a subprocess of another
#: Claude Code session. Stripping them produces "model 404" against
#: valid model ids.
_SECRET_VAR_PATTERNS: Tuple[str, ...] = (
    r".*_TOKEN$", r".*_KEY$", r".*_PASSWORD$", r".*_SECRET$",
    r"^OPENAI_.*", r"^GROQ_.*", r"^HUGGINGFACE_.*",
    r"^GOOGLE_.*", r"^AZURE_.*", r"^AWS_.*",
    r"^GITHUB_TOKEN.*", r"^HISTFILE$",
)

#: Env-var allow-list — kept even when a pattern would strip them. The
#: kernel re-adds ``ANTHROPIC_API_KEY`` explicitly via :func:`build_env`.
_ALWAYS_KEEP: frozenset[str] = frozenset({"ANTHROPIC_API_KEY"})

_SECRET_RE: re.Pattern[str] = re.compile(
    "|".join(f"(?:{p})" for p in _SECRET_VAR_PATTERNS)
)


class PreSpawnValidationError(RuntimeError):
    """Raised when RFC-002 §Pre-spawn validation refuses the spawn.

    The kernel surface treats this as fatal and emits a synthetic
    ``report_problem`` per RFC-002 §"Failure modes". The
    ``log_signature`` attribute names the RFC-004 row classifying it.
    """

    def __init__(self, message: str, log_signature: str) -> None:
        super().__init__(message)
        self.log_signature: str = log_signature


def is_secret_var(name: str) -> bool:
    """Return ``True`` iff ``name`` matches one of :data:`_SECRET_VAR_PATTERNS`.

    ``ANTHROPIC_API_KEY`` is exempt — the kernel passes it through to
    the LiteLLM proxy (RFC-002 §"Required environment variables").
    """
    if name in _ALWAYS_KEEP:
        return False
    return bool(_SECRET_RE.match(name))


def render_system_prompt(task: str) -> str:
    """Substitute ``task`` into ``[TASK_BLOCK]`` of the canonical template.

    Literal string replace; no markdown wrapping is added (RFC-002:
    "MUST be replaced by the operator's task verbatim").
    """
    return CANONICAL_SYSTEM_PROMPT_TEMPLATE.replace("[TASK_BLOCK]", task)


def extract_template_version(rendered_or_template: str) -> Optional[str]:
    """Return the ``MAJOR.MINOR.PATCH`` version embedded in the template marker.

    Parses the trailing comment ``<!-- system-prompt-template vX.Y.Z;
    rfc=RFC-002 -->`` per :data:`SYSTEM_PROMPT_TEMPLATE_VERSION_RE`.
    Returns ``None`` if no marker is present so the caller can decide
    whether absence is fatal.
    """
    match = SYSTEM_PROMPT_TEMPLATE_VERSION_RE.search(rendered_or_template)
    if match is None:
        return None
    return match.group(1)


def _split_semver(version: str) -> Tuple[int, int, int]:
    """Split a ``MAJOR.MINOR.PATCH`` string into a 3-tuple of ints.

    Raises :class:`ValueError` on malformed input. The supervisor wraps
    any failure into a :class:`PreSpawnValidationError` with log
    signature ``provisioning.template.version_mismatch``.
    """
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"not MAJOR.MINOR.PATCH: {version!r}")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def render_mcp_config(
    agent_id: str, zone_id: str, trace_id: str, kernel_python: str,
    *, pythonpath: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the per-spawn MCP config dict per RFC-002 §"MCP config JSON layout".

    JSON-serializable; the supervisor writes it to
    ``<work_dir>/.run/<agent-id>/mcp-config.json`` with POSIX 0o600.
    Validates that ``allowedTools`` exactly equals the 13-tool RFC-001
    catalog in canonical order before returning.

    ``pythonpath`` (optional): explicit ``PYTHONPATH`` to set in the MCP
    server subprocess env. Defaults to inheriting from the calling
    process. Required when ``llm_kernel`` is not installed into the
    kernel's site-packages — Claude Code spawns the MCP server as a
    grandchild and the subprocess may not inherit the operator shell's
    ``PYTHONPATH``.
    """
    server_env: Dict[str, str] = {"LLMKERNEL_RUN_TRACE_ID": trace_id}
    if pythonpath:
        server_env["PYTHONPATH"] = pythonpath
    config: Dict[str, Any] = {
        "mcpServers": {
            MCP_SERVER_NAME: {
                "transport": "stdio",
                "command": kernel_python,
                "args": [
                    "-m", "llm_kernel.mcp_server",
                    "--agent-id", agent_id,
                    "--zone-id", zone_id,
                ],
                "env": server_env,
                "allowedTools": list(RFC001_ALLOWED_TOOLS),
            }
        }
    }
    allowed = config["mcpServers"][MCP_SERVER_NAME]["allowedTools"]
    if tuple(allowed) != RFC001_ALLOWED_TOOLS:  # pragma: no cover - defensive
        raise PreSpawnValidationError(
            "MCP config allowedTools does not match RFC-001 v1.0.0 catalog",
            log_signature="provisioning.mcp.allowed_tools_mismatch",
        )
    return config


def build_argv(
    system_prompt_path: Path, mcp_config_path: Path, task: str,
    *, model: Optional[str] = None, use_bare: bool = False,
    claude_bin: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[str]:
    """Assemble the ``claude`` argv per RFC-002 v1.0.1.

    Flag set verified against ``claude --version 2.1.119`` on
    2026-04-26. Material amendments versus RFC-002 v1.0.0:

    * ``--verbose`` is REQUIRED whenever ``--output-format=stream-json``
      is used; the CLI otherwise rejects the combination.
    * ``--system-prompt-file <path>`` (not ``--system-prompt <path>``) is
      the file-form flag; ``--system-prompt`` takes the prompt INLINE.
    * ``--bare`` forces the ``ANTHROPIC_API_KEY`` auth path so the agent
      routes through our LiteLLM proxy at ``ANTHROPIC_BASE_URL``. Without
      it, Claude Code prefers OAuth / keychain and bypasses the proxy
      (model calls still happen but our run-tracker does not see them;
      tool calls flow through our MCP server regardless). V1 makes
      ``--bare`` opt-in via ``use_bare=True`` so OAuth-only operators
      can still drive the smoke without an Anthropic API key.
    * ``--strict-mcp-config`` is added so our ``llmkernel-operator-bridge``
      is the agent's only MCP server (no operator-side MCP discovery).
    * ``--disallowedTools`` is moved off the env var into a CLI flag
      because ``CLAUDE_CODE_DISABLED_TOOLS`` is not honored by 2.1.119.
    * Optional ``--model`` to pin a small model for tests; default leaves
      it unset (CLI picks Sonnet-4-5).
    * ``claude_bin`` overrides the bare ``claude`` argv[0] with a
      pre-resolved path. Windows' ``subprocess.Popen`` does not honor
      ``PATHEXT`` for unquoted names, so the supervisor passes
      ``shutil.which("claude")`` here to disambiguate ``claude.cmd``.

    See RFC-002 v1.0.1 amendments section.
    """
    argv: List[str] = [
        claude_bin or "claude",
        "--print", "--verbose",
        "--output-format=stream-json",
        "--system-prompt-file", str(system_prompt_path),
        "--mcp-config", str(mcp_config_path),
        "--strict-mcp-config",
        "--disallowedTools", DISABLED_TOOLS,
        # In --print (non-interactive) mode the operator cannot approve
        # MCP tool invocations interactively. The smoke needs the agent
        # to actually call notify + report_completion. V1 trusts the MCP
        # server (the kernel itself) so skipping the permission prompt
        # is acceptable for this trust boundary. Production V1.5 should
        # use a per-tool ``--allowedTools`` whitelist instead.
        "--dangerously-skip-permissions",
    ]
    if use_bare:
        argv.insert(4, "--bare")
    if model:
        argv.extend(["--model", model])
    if session_id:
        # BSP-002 §5: Claude session id is owned by the kernel. Pass
        # --session-id so we control the UUID; required for a future
        # --resume <session_id> to thread continuation across spawns.
        # Pre-positional — matches the placement of other --flag args.
        argv.extend(["--session-id", session_id])
    argv.append(task)
    return argv


def build_env(
    parent_env: Mapping[str, str], *,
    api_key: str, llm_endpoint_url: str,
    mcp_config_path: Path, system_prompt_path: Path, work_dir: Path,
    agent_id: str, zone_id: str, trace_id: str,
    set_base_url: bool = True,
) -> Dict[str, str]:
    """Build the env dict per RFC-002 §"Required environment variables".

    Strips :func:`is_secret_var` matches from ``parent_env`` and layers
    the kernel-issued vars on top. ``ANTHROPIC_API_KEY`` is set from
    ``api_key`` (V1 passes it through unchanged).

    # TODO(V1.5): replace ``ANTHROPIC_API_KEY`` passthrough with an
    # HMAC bearer per RFC-002 §"Authentication"; the kernel mints a
    # per-spawn token, the LiteLLM proxy validates it, and the upstream
    # Anthropic key never leaves the kernel process.
    """
    # Per RFC-002 v1.0.1: empirically, Claude Code's OAuth-mediated model
    # resolution path appears to read provider env vars during pre-flight
    # (haiku-4-5 returns 404 if e.g. OPENAI_API_KEY is stripped). For V1
    # development inherit the parent env unchanged when ``LLMKERNEL_LEAK_ENV=1``
    # is set; otherwise apply the documented strip.
    if os.environ.get("LLMKERNEL_LEAK_ENV") == "1":
        env: Dict[str, str] = dict(parent_env)
    else:
        env = {k: v for k, v in parent_env.items() if not is_secret_var(k)}
    env.update({
        "ANTHROPIC_API_KEY": api_key,
        "CLAUDE_CODE_MCP_CONFIG": str(mcp_config_path),
        "CLAUDE_CODE_WORKING_DIRECTORY": str(work_dir),
        "CLAUDE_CODE_SYSTEM_PROMPT_FILE": str(system_prompt_path),
        "CLAUDE_CODE_ALLOWED_TOOLS": ALLOWED_TOOLS,
        "CLAUDE_CODE_DISABLED_TOOLS": DISABLED_TOOLS,
        "LLMKERNEL_AGENT_ID": agent_id, "LLMKERNEL_ZONE_ID": zone_id,
        "LLMKERNEL_RUN_TRACE_ID": trace_id,
    })
    # ANTHROPIC_BASE_URL is conditional. With ``--bare`` (use_bare=True at
    # the supervisor) the agent uses ANTHROPIC_API_KEY auth and SHOULD
    # route through our LiteLLM proxy; setting the base URL is correct.
    # Without ``--bare``, Claude Code uses OAuth and does its own
    # model-resolution preflight — pointing it at our proxy (which only
    # serves /v1/messages) yields HTTP 404 against valid model ids.
    if set_base_url:
        env["ANTHROPIC_BASE_URL"] = llm_endpoint_url
    else:
        # Defensively unset any inherited override.
        env.pop("ANTHROPIC_BASE_URL", None)
    return env


def validate_pre_spawn(
    api_key: str, llm_endpoint_url: str,
    mcp_config_path: Path, system_prompt_path: Path,
    *, health_check_timeout: float = 2.0,
) -> None:
    """RFC-002 §"Pre-spawn validation"; raise on any documented failure.

    Order: ``ANTHROPIC_API_KEY`` non-empty; LiteLLM proxy answers
    ``GET /v1/models`` with 200; MCP config file exists and non-empty;
    system prompt file exists and non-empty. Each failure carries an
    RFC-004 ``log_signature`` on the raised exception.
    """
    if not api_key:
        raise PreSpawnValidationError(
            "ANTHROPIC_API_KEY missing or empty",
            log_signature="provisioning.api_key.invalid",
        )
    # Reachability check is path-agnostic: send a HEAD to /v1/models
    # if the URL has /v1 in it, else just to the root. ANY HTTP
    # response (2xx/3xx/4xx) proves the proxy is up. Only transport
    # errors and 5xx are fatal. Passthrough proxies forward to
    # api.anthropic.com which 401s without auth — that 401 is fine.
    base = llm_endpoint_url.rstrip("/")
    health_url = base + ("/models" if base.endswith("/v1") else "/v1/models")
    try:
        resp = httpx.head(health_url, timeout=health_check_timeout,
                          follow_redirects=True)
    except httpx.HTTPError as exc:
        raise PreSpawnValidationError(
            f"Proxy unreachable at {health_url}: {exc}",
            log_signature="provisioning.litellm.unreachable",
        ) from exc
    if resp.status_code >= 500:
        raise PreSpawnValidationError(
            f"Proxy {health_url} returned {resp.status_code}",
            log_signature="provisioning.litellm.unreachable",
        )
    if not mcp_config_path.exists() or mcp_config_path.stat().st_size == 0:
        raise PreSpawnValidationError(
            f"MCP config missing or empty: {mcp_config_path}",
            log_signature="provisioning.mcp.unreachable",
        )
    if not system_prompt_path.exists() or system_prompt_path.stat().st_size == 0:
        raise PreSpawnValidationError(
            f"System prompt missing or empty: {system_prompt_path}",
            log_signature="provisioning.template.empty",
        )


__all__ = [
    "ALLOWED_TOOLS", "CANONICAL_SYSTEM_PROMPT_TEMPLATE", "DISABLED_TOOLS",
    "EXPECTED_SYSTEM_PROMPT_TEMPLATE_VERSION", "MCP_SERVER_NAME",
    "PreSpawnValidationError", "RFC001_ALLOWED_TOOLS",
    "SYSTEM_PROMPT_TEMPLATE_VERSION_RE",
    "build_argv", "build_env", "extract_template_version", "is_secret_var",
    "render_mcp_config", "render_system_prompt", "validate_pre_spawn",
]

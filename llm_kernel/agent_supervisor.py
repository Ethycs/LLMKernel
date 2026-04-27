"""LLMKernel agent-process supervisor (Stage 2 Track B4).

Spawns Claude Code subprocesses per RFC-002 v1.0.0, captures their MCP
tool calls, enforces DR-0010 (suppressed text channel), and feeds
run-lifecycle records into the Track B2 :class:`RunTracker` (which in
turn flows them through Track B3 :class:`CustomMessageDispatcher`).

The supervisor is sync (threads, not asyncio). Each agent gets two
daemon reader threads (stdout, stderr) and a watchdog. Per RFC-004 V1
fails closed: at most three restart attempts; on exhaustion a synthetic
``report_problem`` lands and the supervisor halts that agent.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Deque, Dict, Optional

from ._provisioning import (
    PreSpawnValidationError, build_argv, build_env,
    render_mcp_config, render_system_prompt, validate_pre_spawn,
)

if TYPE_CHECKING:  # pragma: no cover
    from .custom_messages import CustomMessageDispatcher
    from .run_tracker import RunTracker

logger: logging.Logger = logging.getLogger("llm_kernel.agent_supervisor")

#: Restart-policy backoffs (seconds) per RFC-002 §"Process lifecycle" 5.
_RESTART_BACKOFFS_SEC: tuple[float, ...] = (0.0, 5.0, 25.0)
#: DR-0010 prose-violation flood threshold: >5 within 60s -> escalate.
_VIOLATION_FLOOD_COUNT: int = 5
_VIOLATION_FLOOD_WINDOW_SEC: float = 60.0

AgentState = str  # starting | running | restarting | failed | terminated


@dataclass
class AgentHandle:
    """Per-spawn handle returned by :meth:`AgentSupervisor.spawn`.

    Carries the live :class:`subprocess.Popen`, both reader threads, the
    watchdog, and a coarse :data:`AgentState`. Use :meth:`terminate`
    for graceful shutdown.
    """

    agent_id: str
    zone_id: str
    popen: subprocess.Popen
    started_at: float
    work_dir: Path
    stdout_thread: threading.Thread
    stderr_thread: threading.Thread
    watchdog_thread: Optional[threading.Thread] = None
    state: AgentState = "starting"
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _violation_times: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    _restart_attempts: int = 0

    def poll(self) -> Optional[int]:
        """Return the process exit code if it has exited, else ``None``."""
        return self.popen.poll()

    def wait(self, timeout: Optional[float] = None) -> int:
        """Block until the process exits and return its exit code."""
        return self.popen.wait(timeout=timeout)

    def terminate(self, grace_seconds: float = 10.0) -> None:
        """SIGTERM, wait ``grace_seconds``, then SIGKILL (RFC-002 §6)."""
        self._stop_event.set()
        if self.popen.poll() is not None:
            self.state = "terminated"
            return
        try:
            self.popen.terminate()
        except (OSError, ProcessLookupError):  # pragma: no cover
            pass
        try:
            self.popen.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            logger.warning(
                "agent %s did not exit within %.1fs; SIGKILL",
                self.agent_id, grace_seconds,
            )
            try:
                self.popen.kill()
            except (OSError, ProcessLookupError):  # pragma: no cover
                pass
            try:
                self.popen.wait(timeout=2.0)
            except subprocess.TimeoutExpired:  # pragma: no cover
                logger.error("agent %s SIGKILL did not reap", self.agent_id)
        self.state = "terminated"


class AgentSupervisor:
    """Spawn and supervise Claude Code agents per RFC-002 v1.0.0.

    Wires the supervisor to the kernel's run-tracker (B2) and
    custom-message dispatcher (B3). Spawned agents inherit
    ``LLMKERNEL_RUN_TRACE_ID`` so their MCP-server tool runs share a
    trace with the supervisor's synthetic runs. Thread-safe.
    """

    def __init__(
        self, run_tracker: "RunTracker",
        dispatcher: "CustomMessageDispatcher",
        litellm_endpoint_url: str,
        kernel_python: str = sys.executable,
    ) -> None:
        """Bind the supervisor to its B2/B3 collaborators.

        ``litellm_endpoint_url`` is ``ANTHROPIC_BASE_URL`` for spawned
        agents and the URL the pre-spawn health check probes.
        ``kernel_python`` is the absolute path to the kernel's Python;
        the per-spawn MCP config's ``command`` field uses this.
        """
        self._run_tracker = run_tracker
        self._dispatcher = dispatcher
        self._litellm_endpoint_url: str = litellm_endpoint_url
        self._kernel_python: str = kernel_python
        self._lock: threading.RLock = threading.RLock()
        self._agents: Dict[str, AgentHandle] = {}

    def spawn(
        self, zone_id: str, agent_id: str, task: str, work_dir: Path,
        api_key: Optional[str] = None, model: Optional[str] = None,
        use_bare: bool = False, set_base_url: Optional[bool] = None,
    ) -> AgentHandle:
        """Spawn a Claude Code subprocess wired into paper-telephone topology.

        Implements RFC-002 §"Process lifecycle": pre-spawn validation,
        artifact rendering (POSIX 0o600), env build, ``Popen``, two
        reader threads, watchdog. Raises
        :class:`PreSpawnValidationError` on validation failure (also
        emits a synthetic ``report_problem``).
        """
        api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        trace_id = self._run_tracker.trace_id
        # Resolve to absolute paths up-front. The child process is launched
        # with ``cwd=work_dir``, so passing relative paths to Claude's CLI
        # would double-prefix the work_dir (cwd + relative arg = doubled).
        work_dir = Path(work_dir).resolve()
        spawn_dir = (work_dir / ".run" / agent_id).resolve()
        spawn_dir.mkdir(parents=True, exist_ok=True)
        mcp_config_path = spawn_dir / "mcp-config.json"
        system_prompt_path = spawn_dir / "system-prompt.txt"

        # Propagate PYTHONPATH explicitly into the MCP server's env so
        # ``python -m llm_kernel.mcp_server`` resolves the package even
        # when llm_kernel is not installed into the kernel's site-packages
        # (e.g. development runs via PYTHONPATH=vendor/LLMKernel).
        # CRITICAL: convert each entry to an absolute path before passing
        # it to the MCP-server subprocess. The supervisor sets cwd=work_dir
        # for the agent, and Claude Code spawns the MCP server with the
        # SAME cwd. Relative PYTHONPATH entries (e.g. "vendor/LLMKernel")
        # would resolve against work_dir, not the operator's project root.
        raw_pp = os.environ.get("PYTHONPATH") or ""
        abs_pp = (
            os.pathsep.join(
                str(Path(p).resolve()) for p in raw_pp.split(os.pathsep) if p
            )
            or None
        )
        config = render_mcp_config(
            agent_id=agent_id, zone_id=zone_id,
            trace_id=trace_id, kernel_python=self._kernel_python,
            pythonpath=abs_pp,
        )
        mcp_config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        system_prompt_path.write_text(render_system_prompt(task), encoding="utf-8")
        for p in (mcp_config_path, system_prompt_path):
            try:
                p.chmod(0o600)
            except OSError:  # pragma: no cover - non-POSIX
                pass

        try:
            validate_pre_spawn(
                api_key=api_key, llm_endpoint_url=self._litellm_endpoint_url,
                mcp_config_path=mcp_config_path,
                system_prompt_path=system_prompt_path,
            )
        except PreSpawnValidationError as exc:
            self._record_synthetic_problem(
                agent_id, zone_id, str(exc), exc.log_signature,
            )
            raise

        # Default: set ANTHROPIC_BASE_URL only under --bare, because the
        # LiteLLM proxy only handles /v1/messages and breaks OAuth's
        # model-resolution preflight. The transparent
        # AnthropicPassthroughServer handles ALL of /v1/* so it CAN be
        # used under OAuth — pass set_base_url=True explicitly when the
        # supervisor's endpoint is a passthrough.
        effective_set_base_url = (
            set_base_url if set_base_url is not None else use_bare
        )
        env = build_env(
            os.environ, api_key=api_key,
            llm_endpoint_url=self._litellm_endpoint_url,
            mcp_config_path=mcp_config_path,
            system_prompt_path=system_prompt_path,
            work_dir=work_dir, agent_id=agent_id, zone_id=zone_id,
            trace_id=trace_id,
            set_base_url=effective_set_base_url,
        )
        # Windows: subprocess.Popen does not honor PATHEXT for unquoted
        # names, so resolve "claude" to its actual file (claude.cmd /
        # claude.exe / claude bash script). Lookup uses the parent env's
        # PATH because the child env's PATH is identical (we only filter
        # secrets, not add path entries).
        claude_bin = shutil.which("claude") or "claude"
        argv = build_argv(
            system_prompt_path, mcp_config_path, task,
            model=model, use_bare=use_bare, claude_bin=claude_bin,
        )
        logger.info(
            "spawning agent %s (zone=%s) model=%s bare=%s claude=%s",
            agent_id, zone_id, model, use_bare, claude_bin,
        )
        popen = subprocess.Popen(
            argv, env=env, cwd=str(work_dir),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        handle = self._build_handle(popen, agent_id, zone_id, work_dir, task)
        with self._lock:
            self._agents[agent_id] = handle
        return handle

    def get(self, agent_id: str) -> Optional[AgentHandle]:
        """Return the live handle for ``agent_id``, or ``None`` if unknown."""
        with self._lock:
            return self._agents.get(agent_id)

    def terminate_all(self, grace_seconds: float = 10.0) -> None:
        """Graceful shutdown of every active agent (RFC-002 §6)."""
        with self._lock:
            handles = list(self._agents.values())
        for handle in handles:
            try:
                handle.terminate(grace_seconds=grace_seconds)
            except Exception:  # pragma: no cover - defensive
                logger.exception("terminate_all: %s raised", handle.agent_id)

    # -- Thread workers ----------------------------------------------

    def _build_handle(
        self, popen: subprocess.Popen, agent_id: str, zone_id: str,
        work_dir: Path, task: str,
    ) -> AgentHandle:
        """Allocate the handle + start reader/watchdog threads."""
        spawn_dir = work_dir / ".run" / agent_id
        stderr_log = spawn_dir / f"kernel.stderr.{agent_id}.log"
        handle = AgentHandle(
            agent_id=agent_id, zone_id=zone_id, popen=popen,
            started_at=time.monotonic(), work_dir=work_dir,
            stdout_thread=threading.Thread(),  # placeholder
            stderr_thread=threading.Thread(),  # placeholder
        )
        handle.stdout_thread = threading.Thread(
            target=self._read_stdout, args=(handle,),
            name=f"agent-stdout-{agent_id}", daemon=True,
        )
        handle.stderr_thread = threading.Thread(
            target=self._read_stderr, args=(handle, stderr_log),
            name=f"agent-stderr-{agent_id}", daemon=True,
        )
        handle.stdout_thread.start()
        handle.stderr_thread.start()
        handle.watchdog_thread = threading.Thread(
            target=self._watchdog, args=(handle, task),
            name=f"agent-watchdog-{agent_id}", daemon=True,
        )
        handle.watchdog_thread.start()
        handle.state = "running"
        return handle

    def _read_stdout(self, handle: AgentHandle) -> None:
        """Parse each stdout line as stream-json or MCP JSON-RPC.

        Lines that fail both parsers are DR-0010 violations; logged via
        a synthetic ``notify`` and counted toward the flood threshold.
        Lines that parse as MCP JSON-RPC ``tools/call`` frames are
        recorded as a courtesy run.

        When ``LLMKERNEL_DEBUG_STDOUT=1`` is set in the supervisor's env,
        every raw stdout line is also appended to
        ``<spawn_dir>/agent.stdout.<id>.log`` for post-mortem inspection,
        and a pretty-printed copy of each parseable JSON event is
        appended to ``<spawn_dir>/agent.stdout.<id>.pretty.log``.
        """
        assert handle.popen.stdout is not None
        debug = os.environ.get("LLMKERNEL_DEBUG_STDOUT") == "1"
        debug_path: Optional[Path] = None
        pretty_path: Optional[Path] = None
        if debug:
            debug_dir = handle.work_dir / ".run" / handle.agent_id
            debug_path = debug_dir / f"agent.stdout.{handle.agent_id}.log"
            pretty_path = debug_dir / f"agent.stdout.{handle.agent_id}.pretty.log"
            debug_dir.mkdir(parents=True, exist_ok=True)
        for raw in handle.popen.stdout:
            if handle._stop_event.is_set():
                break
            line = raw.rstrip("\r\n")
            if debug_path is not None:
                with debug_path.open("a", encoding="utf-8") as fh:
                    fh.write(raw)
            if not line:
                continue
            parsed = self._try_parse_json(line)
            if pretty_path is not None and parsed is not None:
                _append_pretty(pretty_path, parsed)
            if parsed is None:
                self._handle_violation(handle, line)
                continue
            if self._is_mcp_jsonrpc(parsed):
                self._record_jsonrpc(handle, parsed)
                continue
            if self._is_stream_json(parsed):
                self._record_stream_json(handle, parsed)
                continue
            self._handle_violation(handle, line)

    def _read_stderr(self, handle: AgentHandle, log_path: Path) -> None:
        """Append stderr to a per-agent log; never operator-surface (RFC-002 §4)."""
        assert handle.popen.stderr is not None
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                for raw in handle.popen.stderr:
                    if handle._stop_event.is_set():
                        break
                    fh.write(raw)
                    fh.flush()
        except OSError:  # pragma: no cover - defensive
            logger.exception("agent %s stderr log write failed", handle.agent_id)

    def _watchdog(self, handle: AgentHandle, task: str) -> None:
        """Monitor process exit; restart per RFC-002 §"Process lifecycle" 5.

        Up to three restarts with the documented backoff; after
        exhaustion emits a synthetic ``report_problem`` and leaves
        ``state=failed``.
        """
        exit_code = handle.popen.wait()
        if handle._stop_event.is_set() or exit_code == 0:
            handle.state = "terminated"
            return
        logger.warning(
            "agent %s exited code=%s (attempt %d)",
            handle.agent_id, exit_code, handle._restart_attempts + 1,
        )
        if handle._restart_attempts >= len(_RESTART_BACKOFFS_SEC) - 1:
            self._record_synthetic_problem(
                handle.agent_id, handle.zone_id,
                f"agent process unrestartable after "
                f"{handle._restart_attempts + 1} attempts (last exit={exit_code})",
                "agent.unrestartable",
            )
            handle.state = "failed"
            return
        backoff = _RESTART_BACKOFFS_SEC[handle._restart_attempts + 1]
        handle._restart_attempts += 1
        handle.state = "restarting"
        time.sleep(backoff)
        if handle._stop_event.is_set():
            return
        try:
            self._respawn_in_place(handle, task)
        except PreSpawnValidationError:
            handle.state = "failed"
            return
        self._watchdog(handle, task)

    def _respawn_in_place(self, handle: AgentHandle, task: str) -> None:
        """Spawn a new ``claude`` process replacing the dead one in ``handle``."""
        spawn_dir = handle.work_dir / ".run" / handle.agent_id
        mcp_config_path = spawn_dir / "mcp-config.json"
        system_prompt_path = spawn_dir / "system-prompt.txt"
        argv = build_argv(system_prompt_path, mcp_config_path, task)
        env = build_env(
            os.environ, api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            llm_endpoint_url=self._litellm_endpoint_url,
            mcp_config_path=mcp_config_path,
            system_prompt_path=system_prompt_path,
            work_dir=handle.work_dir, agent_id=handle.agent_id,
            zone_id=handle.zone_id, trace_id=self._run_tracker.trace_id,
        )
        handle.popen = subprocess.Popen(
            argv, env=env, cwd=str(handle.work_dir),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        handle.stdout_thread = threading.Thread(
            target=self._read_stdout, args=(handle,),
            name=f"agent-stdout-{handle.agent_id}", daemon=True,
        )
        handle.stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(handle, spawn_dir / f"kernel.stderr.{handle.agent_id}.log"),
            name=f"agent-stderr-{handle.agent_id}", daemon=True,
        )
        handle.stdout_thread.start()
        handle.stderr_thread.start()
        handle.state = "running"

    # -- Synthetic-run helpers ---------------------------------------

    def _handle_violation(self, handle: AgentHandle, line: str) -> None:
        """DR-0010 prose violation: synthetic ``notify`` + flood-check."""
        truncated = line if len(line) <= 240 else line[:237] + "..."
        rid = self._run_tracker.start_run(
            name="notify", run_type="tool",
            inputs={"observation": f"agent emitted prose: {truncated}",
                    "importance": "info"},
            tags=[f"agent:{handle.agent_id}", f"zone:{handle.zone_id}",
                  "tool:notify", "synthetic:dr0010_violation"],
            metadata={"log_signature": "dr0010.violation"},
        )
        self._run_tracker.complete_run(rid, outputs={"acknowledged": True})
        now = time.monotonic()
        handle._violation_times.append(now)
        cutoff = now - _VIOLATION_FLOOD_WINDOW_SEC
        recent = [t for t in handle._violation_times if t >= cutoff]
        if len(recent) > _VIOLATION_FLOOD_COUNT:
            self._record_synthetic_escalate(
                handle.agent_id, handle.zone_id, "DR-0010 flood", "medium",
            )
            handle._violation_times.clear()

    def _record_jsonrpc(self, handle: AgentHandle, parsed: Dict[str, Any]) -> None:
        """Record a tool-call JSON-RPC frame as a courtesy run."""
        if parsed.get("method") != "tools/call":
            return
        params = parsed.get("params") or {}
        tool_name = params.get("name", "<unknown>")
        rid = self._run_tracker.start_run(
            name=f"agent_emit:{tool_name}", run_type="tool",
            inputs={"arguments": params.get("arguments", {})},
            tags=[f"agent:{handle.agent_id}", f"zone:{handle.zone_id}",
                  f"tool:{tool_name}", "synthetic:agent_emit"],
            metadata={"jsonrpc_id": parsed.get("id")},
        )
        self._run_tracker.complete_run(rid, outputs={"acknowledged": True})

    def _record_stream_json(self, handle: AgentHandle, parsed: Dict[str, Any]) -> None:
        """Surface tool_use blocks from a Claude stream-json assistant event.

        Claude Code's ``--output-format=stream-json`` emits assistant
        messages with content arrays; ``tool_use`` blocks indicate the
        agent invoked a tool. We strip the MCP-namespaced prefix
        ``mcp__llmkernel-operator-bridge__`` (Claude Code's convention
        for MCP tools) and emit a synthetic run record so the operator
        surface sees ``notify`` etc. by their RFC-001 names. The actual
        tool result is executed by the spawned MCP-server subprocess and
        is not visible from the supervisor's run-tracker — V1 records
        the *attempt* here, not the result.
        """
        if parsed.get("type") != "assistant":
            return
        message = parsed.get("message") or {}
        content = message.get("content") or []
        if not isinstance(content, list):
            return
        prefix = "mcp__llmkernel-operator-bridge__"
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            raw_name = str(block.get("name", "<unknown>"))
            tool_name = raw_name[len(prefix):] if raw_name.startswith(prefix) else raw_name
            rid = self._run_tracker.start_run(
                name=tool_name, run_type="tool",
                inputs=block.get("input") or {},
                tags=[f"agent:{handle.agent_id}", f"zone:{handle.zone_id}",
                      f"tool:{tool_name}", "via:stream-json"],
                metadata={"tool_use_id": block.get("id"),
                          "raw_name": raw_name},
            )
            # The supervisor cannot observe the MCP-server subprocess's
            # run completion; mark this as success so I1 (every start has
            # a complete) holds. RFC-002 v1.0.1 amendment captures this.
            self._run_tracker.complete_run(rid, outputs={"observed_via": "stream-json"})

    def _record_synthetic_problem(
        self, agent_id: str, zone_id: str, description: str, log_signature: str,
    ) -> None:
        """Open + close a synthetic ``report_problem`` run."""
        rid = self._run_tracker.start_run(
            name="report_problem", run_type="tool",
            inputs={"severity": "error", "description": description},
            tags=[f"agent:{agent_id}", f"zone:{zone_id}",
                  "tool:report_problem", "synthetic:supervisor"],
            metadata={"log_signature": log_signature},
        )
        self._run_tracker.complete_run(
            rid, outputs={"acknowledged": True}, status="error",
        )

    def _record_synthetic_escalate(
        self, agent_id: str, zone_id: str, reason: str, severity: str,
    ) -> None:
        """Open + close a synthetic ``escalate`` run (DR-0010 flood)."""
        rid = self._run_tracker.start_run(
            name="escalate", run_type="tool",
            inputs={"reason": reason, "severity": severity},
            tags=[f"agent:{agent_id}", f"zone:{zone_id}",
                  "tool:escalate", "synthetic:dr0010_flood"],
            metadata={"log_signature": "dr0010.flood"},
        )
        self._run_tracker.complete_run(rid, outputs={"acknowledged": True})

    @staticmethod
    def _try_parse_json(line: str) -> Optional[Dict[str, Any]]:
        """Return the parsed JSON object or ``None`` for non-JSON lines."""
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    @staticmethod
    def _is_mcp_jsonrpc(obj: Dict[str, Any]) -> bool:
        """Return True for an MCP JSON-RPC frame."""
        return obj.get("jsonrpc") == "2.0" and (
            "method" in obj or "result" in obj or "error" in obj
        )

    @staticmethod
    def _is_stream_json(obj: Dict[str, Any]) -> bool:
        """Return True for a Claude stream-json record (``type``/``event``)."""
        return "type" in obj or "event" in obj


def _append_pretty(path: Path, parsed: Dict[str, Any]) -> None:
    """Append a pretty-printed JSON event with a one-line header.

    The header summarizes the event's ``type`` / ``subtype`` and the
    most informative inner field (model name for assistant messages,
    error code for results, mcp_servers status for system init). The
    payload follows as 2-space-indented JSON. Used only when
    ``LLMKERNEL_DEBUG_STDOUT=1``.
    """
    kind = str(parsed.get("type") or parsed.get("event") or "?")
    subtype = parsed.get("subtype")
    summary_parts = [kind]
    if subtype:
        summary_parts.append(f"subtype={subtype}")
    if kind == "system" and subtype == "init":
        servers = parsed.get("mcp_servers") or []
        statuses = ",".join(
            f"{s.get('name')}={s.get('status')}" for s in servers
        ) or "(none)"
        summary_parts.append(f"mcp_servers={statuses}")
    if kind == "assistant":
        msg = parsed.get("message") or {}
        content = msg.get("content") or []
        block_kinds = ",".join(
            str(b.get("type")) for b in content if isinstance(b, dict)
        )
        summary_parts.append(f"blocks=[{block_kinds}]")
        if parsed.get("error"):
            summary_parts.append(f"error={parsed.get('error')}")
    if kind == "result":
        if parsed.get("is_error"):
            summary_parts.append(
                f"is_error api_status={parsed.get('api_error_status')}"
            )
        summary_parts.append(f"stop_reason={parsed.get('stop_reason')}")
        cost = parsed.get("total_cost_usd")
        if cost:
            summary_parts.append(f"cost=${cost:.4f}")
    header = " ".join(summary_parts)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"=== {header} ===\n")
        fh.write(json.dumps(parsed, indent=2, default=str))
        fh.write("\n\n")


__all__ = ["AgentHandle", "AgentState", "AgentSupervisor"]

"""Agent-process supervisor.

Spawns Claude Code subprocesses, captures their MCP tool calls,
suppresses their free-text channel, and feeds run-lifecycle records
into the run tracker.

Runs on threads, not asyncio. Each agent gets two daemon reader
threads (stdout, stderr) and a watchdog. Fails closed: at most three
restart attempts, after which the agent is halted with a problem
report.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Tuple

from llm_kernel.capture import context_packer
from llm_kernel.supervision._provisioning import (
    EXPECTED_SYSTEM_PROMPT_TEMPLATE_VERSION,
    PreSpawnValidationError, build_argv, build_env,
    extract_template_version, render_mcp_config, render_system_prompt,
    _split_semver, validate_pre_spawn,
)

if TYPE_CHECKING:  # pragma: no cover
    from llm_kernel.capture.custom_messages import CustomMessageDispatcher
    from llm_kernel.capture.run_tracker import RunTracker

logger: logging.Logger = logging.getLogger("llm_kernel.supervision.agent_supervisor")

# Delay before each restart attempt, in seconds.
_RESTART_BACKOFFS_SEC: tuple[float, ...] = (0.0, 5.0, 25.0)
# Escalate when more than 5 prose violations arrive within 60 seconds.
_VIOLATION_FLOOD_COUNT: int = 5
_VIOLATION_FLOOD_WINDOW_SEC: float = 60.0
# How long an agent may emit no stdout before it counts as hung,
# in seconds. Overridable via the constructor.
DEFAULT_AGENT_SILENCE_THRESHOLD_SEC: float = 120.0
# How often the silence watchdog wakes to check the timer, in seconds.
_SILENCE_WATCHDOG_GRANULARITY_SEC: float = 5.0
# Restart rate limit: at most 3 attempts within any 5-minute window.
# Tracked as a sliding window of timestamps; further restarts are
# refused once the limit is reached.
_RESTART_WINDOW_SEC: float = 300.0
_RESTART_WINDOW_MAX: int = 3

# Error codes flagged when an agent's stream mishandles magic lines.
K35_PLAIN_MAGIC_IN_HASH_MODE: str = "K35"
K36_HASHED_MAGIC_EMISSION_BLOCKED: str = "K36"
K3H_AGENT_EMITTED_GENERATOR_MAGIC_BLOCKED: str = "K3H"
# Emitted when a privileged agent's stream contains a magic-shaped
# line. The line is still sanitized; this marks the cell so the
# operator can choose to promote it.
K3L_PRIVILEGED_AGENT_STREAM_MAGIC: str = "K3L"

AgentState = str  # starting | running | restarting | failed | terminated


class _HandoffCycleError(Exception):
    """Raised when the turn-DAG walker detects a cycle.

    Signals that the turn chain exceeded the max-depth guard or that a
    parent_id loop was found. The supervisor turns this into a runtime
    error surfaced as a problem report.
    """


@dataclass
class AgentHandle:
    """Per-spawn handle returned by spawn().

    Carries the live subprocess, both reader threads, the watchdog, and
    a coarse state string. Use terminate() for graceful shutdown.
    """

    agent_id: str
    zone_id: str
    popen: subprocess.Popen
    started_at: float
    work_dir: Path
    stdout_thread: threading.Thread
    stderr_thread: threading.Thread
    watchdog_thread: Optional[threading.Thread] = None
    silence_watchdog_thread: Optional[threading.Thread] = None
    state: AgentState = "starting"
    _stop_event: threading.Event = field(default_factory=threading.Event)
    # Serializes the multi-line stdin write+flush in send_user_turn so
    # concurrent callers (external MCP, PBX, library-embedded use) cannot
    # interleave the lines of two user turns on the same pipe. Per-handle,
    # not the supervisor lock, so a slow write to one agent never blocks
    # dispatch to another. SE-01.
    _stdin_lock: threading.Lock = field(default_factory=threading.Lock)
    _violation_times: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    _restart_attempts: int = 0
    # Timestamps of recent restart attempts. Pruned to the restart
    # window; once it reaches the max, further restarts are refused.
    _restart_history: Deque[float] = field(default_factory=deque)
    # Timestamp of the most recent stdout byte. Updated whenever a
    # reader thread sees a non-empty line, and read by the silence
    # watchdog. Defaults to spawn time so a fresh agent gets the full
    # silence budget for its first byte.
    _last_stdout_ts: float = 0.0
    #: Set whenever the silence watchdog has SIGTERMed the process so
    #: the regular watchdog (which observes the resulting non-zero
    #: exit) does not double-log; tag is plumbed into the restart
    #: bookkeeping.
    _hang_terminated: bool = False
    # Claude session id (a UUID). Assigned at spawn time and passed as
    # --session-id. Survives across spawn/idle/respawn of the same
    # agent so a later --resume can pick the conversation back up.
    # Empty string when unset.
    claude_session_id: str = ""
    # The most recent turn-id this agent has been fed. None for fresh
    # spawns. After a successful send_user_turn it advances to the
    # notebook head so later calls can compute which turns were missed.
    last_seen_turn_id: Optional[str] = None

    def poll(self) -> Optional[int]:
        """Return the process exit code if it has exited, else None."""
        return self.popen.poll()

    def wait(self, timeout: Optional[float] = None) -> int:
        """Block until the process exits and return its exit code."""
        return self.popen.wait(timeout=timeout)

    def terminate(self, grace_seconds: float = 10.0) -> None:
        """SIGTERM, wait grace_seconds, then SIGKILL."""
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
    """Spawn and supervise Claude Code agents.

    Wires the supervisor to the kernel's run-tracker and custom-message
    dispatcher. Spawned agents inherit the run trace id so their MCP
    tool runs share a trace with the supervisor's own runs. Thread-safe.
    """

    def __init__(
        self, run_tracker: "RunTracker",
        dispatcher: "CustomMessageDispatcher",
        litellm_endpoint_url: str,
        kernel_python: str = sys.executable,
        agent_silence_threshold_seconds: float = DEFAULT_AGENT_SILENCE_THRESHOLD_SEC,
        silence_watchdog_granularity_seconds: float = _SILENCE_WATCHDOG_GRANULARITY_SEC,
    ) -> None:
        """Bind the supervisor to its collaborators.

        litellm_endpoint_url is the ANTHROPIC_BASE_URL for spawned
        agents and the URL the pre-spawn health check probes.
        kernel_python is the absolute path to the kernel's Python, used
        as the command in each agent's MCP config.

        When an agent emits no stdout for agent_silence_threshold_seconds,
        the silence watchdog kills the process and the crash-restart
        machinery picks up the resulting exit.
        silence_watchdog_granularity_seconds controls how often the
        watchdog wakes to check that timer.
        """
        self._run_tracker = run_tracker
        self._dispatcher = dispatcher
        self._litellm_endpoint_url: str = litellm_endpoint_url
        self._kernel_python: str = kernel_python
        self._silence_threshold_sec: float = agent_silence_threshold_seconds
        self._silence_granularity_sec: float = silence_watchdog_granularity_seconds
        # Reentrant because it is held on the data path (spawn / restart
        # bookkeeping) while code inside the critical section may log,
        # and the logging path can re-enter this lock.
        self._lock: threading.RLock = threading.RLock()
        self._agents: Dict[str, AgentHandle] = {}
        # Contamination-detector hook. The kernel wires its metadata
        # writer in via set_metadata_writer so the supervisor can flag
        # receiving cells when an agent emit carries a magic-shaped
        # line. When None the detector still runs and logs, so the
        # audit trail is preserved without a wired writer.
        self._metadata_writer: Optional[Any] = None
        # The turn graph is read from the metadata writer's snapshot,
        # not cached in memory. Callers seed turns by submitting
        # append_turn intents.

        # Counts how many runs are currently active in each section so
        # a section flips from in-progress back to open only when its
        # last run terminates. Lock-protected because run start and
        # terminal hooks fire from arbitrary worker threads.
        self._in_progress_run_count: Dict[str, int] = {}
        self._auto_flip_lock: threading.Lock = threading.Lock()

    # Callers submit append_turn intents to the writer; the missed-turn
    # walk reads the persisted turn arrays directly.

    def _read_persisted_turns(self) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
        """Return (turn_index, notebook_head_id) from the writer snapshot.

        turn_index is keyed on turn-id and carries each persisted turn
        dict, with a content field mirrored from body.

        The notebook head is resolved as follows:

        1. If any agent has a session head turn set, pick the latest of
           those by (created_at, id).
        2. Otherwise fall back to the most recently created turn across
           all agents (id breaks ties).

        Returns ({}, None) when no writer is wired or no turns exist.
        """
        if self._metadata_writer is None:
            return {}, None
        try:
            snap = self._metadata_writer.snapshot(trigger="save")
        except Exception:  # pragma: no cover - defensive
            return {}, None
        zone = snap.get("zone") or {}
        agents = zone.get("agents") or {}
        index: Dict[str, Dict[str, Any]] = {}
        session_heads: List[str] = []
        for agent_state in agents.values():
            if not isinstance(agent_state, dict):
                continue
            session = agent_state.get("session") or {}
            sh = session.get("head_turn_id")
            if isinstance(sh, str) and sh:
                session_heads.append(sh)
            for t in agent_state.get("turns", []) or []:
                if not isinstance(t, dict):
                    continue
                tid = t.get("id")
                if not isinstance(tid, str):
                    continue
                # Callers expect a content field; the persisted shape
                # uses body. Mirror body -> content so the missed-turn
                # and synthesize paths work without per-call adaptation.
                norm = dict(t)
                if "content" not in norm and "body" in norm:
                    norm["content"] = norm["body"]
                index[tid] = norm

        def _key_for(tid: str) -> Tuple[str, str]:
            t = index.get(tid)
            return (str(t.get("created_at") or "") if t else "", tid)

        head_id: Optional[str] = None
        # Prefer agent session heads when present.
        candidates = [h for h in session_heads if h in index]
        if candidates:
            head_id = max(candidates, key=_key_for)
        else:
            # Fall back to leaf turns (no other turn names them as
            # parent) — the chain tips. Among leaves, pick the most
            # recently created (id breaks ties). If there are no leaves
            # (a cycle), use the global most-recent.
            referenced_parents: set = set()
            for t in index.values():
                pid = t.get("parent_id")
                if isinstance(pid, str):
                    referenced_parents.add(pid)
            leaves = [tid for tid in index if tid not in referenced_parents]
            pool = leaves if leaves else list(index.keys())
            head_key: Optional[Tuple[str, str]] = None
            for tid in pool:
                key = _key_for(tid)
                if head_key is None or key > head_key:
                    head_key = key
                    head_id = tid
        return index, head_id

    def _notebook_head_turn_id(self) -> Optional[str]:
        """Return the current notebook-head turn-id.

        Computed from the persisted turns as the most recently created
        turn (id breaks ties). Returns None when no turns are persisted.
        """
        _, head_id = self._read_persisted_turns()
        return head_id

    def spawn(
        self, zone_id: str, agent_id: str, task: str, work_dir: Path,
        api_key: Optional[str] = None, model: Optional[str] = None,
        use_bare: bool = False, set_base_url: Optional[bool] = None,
        resume_claude_session_id: Optional[str] = None,
    ) -> AgentHandle:
        """Spawn a Claude Code subprocess and wire it into the topology.

        Runs pre-spawn validation, renders the spawn artifacts (mode
        0o600), builds the env, starts the process, and launches two
        reader threads and a watchdog. Raises PreSpawnValidationError on
        validation failure (and emits a synthetic problem report).

        When resume_claude_session_id is set, re-attach to that existing
        claude session instead of minting a new UUID: argv carries
        --resume <id> rather than --session-id <uuid>, and the handle's
        claude_session_id matches the resumed value so the operator-
        visible identity is unchanged. Used when respawning an agent
        whose process has gone away.
        """
        from llm_kernel._internal import _diagnostics
        _diagnostics.mark("supervisor_spawn_entry", agent_id=agent_id, zone_id=zone_id)

        # Reject agent_ids that collide with a reserved magic name, so
        # an operator can't shadow a cell or line magic (pin, agent,
        # break, and the like) by spawning an agent with that name.
        from llm_kernel.text.magic_registry import is_reserved_name, K32_RESERVED_MAGIC_NAME
        if is_reserved_name(agent_id):
            _diagnostics.mark(
                "supervisor_spawn_reserved_name_rejected",
                agent_id=agent_id, k_class=K32_RESERVED_MAGIC_NAME,
            )
            raise PreSpawnValidationError(
                f"{K32_RESERVED_MAGIC_NAME}: agent_id {agent_id!r} collides "
                f"with a reserved magic name (cell or line magic, or the "
                f"`llmnb_*` future-reservation prefix). Pick a non-reserved id.",
                log_signature="reserved_magic_name_as_agent_id",
            )

        # Idempotency: a spawn for an agent whose process is still alive
        # returns the existing handle instead of double-spawning.
        # Successive cells against the same agent are continuations, not
        # fresh spawns, so the kernel dedups here.
        with self._lock:
            existing = self._agents.get(agent_id)
        if existing is not None and existing.popen.poll() is None:
            _diagnostics.mark(
                "supervisor_spawn_idempotent_alive",
                agent_id=agent_id,
                claude_session_id=existing.claude_session_id,
            )
            logger.info(
                "spawn(%s): agent already alive (session=%s); returning existing handle",
                agent_id, existing.claude_session_id,
            )
            return existing

        api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        trace_id = self._run_tracker.trace_id
        # Kernel-owned claude_session_id. On a fresh spawn, mint a UUID
        # and pass it via --session-id so the conversation is tagged
        # with a stable id we control. On resume, reuse the stored id
        # and emit --resume instead, so claude re-attaches to the
        # existing conversation cache.
        if resume_claude_session_id:
            claude_session_id = resume_claude_session_id
        else:
            claude_session_id = str(uuid.uuid4())
        # Resolve to absolute paths up-front. The child runs with
        # cwd=work_dir, so relative paths passed to Claude's CLI would
        # be doubled (cwd + relative arg).
        work_dir = Path(work_dir).resolve()
        spawn_dir = (work_dir / ".run" / agent_id).resolve()
        spawn_dir.mkdir(parents=True, exist_ok=True)
        _diagnostics.mark("supervisor_spawn_dirs_ready", agent_id=agent_id, spawn_dir=str(spawn_dir))
        mcp_config_path = spawn_dir / "mcp-config.json"
        system_prompt_path = spawn_dir / "system-prompt.txt"

        # Propagate PYTHONPATH explicitly into the MCP server's env so
        # python -m llm_kernel.mcp.mcp_server resolves the package even
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
        _diagnostics.mark("supervisor_spawn_mcp_config_written", agent_id=agent_id)
        rendered_prompt = render_system_prompt(task)
        # Refuse spawn on a system-prompt template major-version
        # mismatch; warn and proceed on a minor mismatch.
        try:
            self._validate_template_version(rendered_prompt)
        except PreSpawnValidationError as exc:
            _diagnostics.mark(
                "supervisor_spawn_template_version_failed",
                agent_id=agent_id, error=str(exc),
            )
            self._record_synthetic_problem(
                agent_id, zone_id, str(exc), exc.log_signature,
            )
            raise
        _diagnostics.mark("supervisor_spawn_template_validated", agent_id=agent_id)
        system_prompt_path.write_text(rendered_prompt, encoding="utf-8")
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
            _diagnostics.mark(
                "supervisor_spawn_preflight_failed",
                agent_id=agent_id, log_signature=exc.log_signature, error=str(exc),
            )
            self._record_synthetic_problem(
                agent_id, zone_id, str(exc), exc.log_signature,
            )
            raise
        _diagnostics.mark("supervisor_spawn_preflight_passed", agent_id=agent_id)

        # By default set ANTHROPIC_BASE_URL only under --bare: the
        # LiteLLM proxy only handles /v1/messages and breaks OAuth's
        # model-resolution preflight. A transparent passthrough handles
        # all of /v1/*, so pass set_base_url=True explicitly when the
        # endpoint is a passthrough.
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
        # Resolve the claude binary through zone_control so the
        # precedence rules (env > PATH > pixi-env probe) apply
        # consistently. Returns None if nothing resolves; degrade to a
        # bare "claude" so a failed launch gives a useful ENOENT.
        from llm_kernel.notebook import zone_control
        claude_bin = zone_control.locate_claude_bin() or "claude"
        _diagnostics.mark(
            "supervisor_spawn_claude_resolved",
            agent_id=agent_id, claude_bin=claude_bin,
        )
        # Mutually-exclusive flags: --session-id <uuid> for fresh
        # spawns vs. --resume <id> for re-attach. Decide here which
        # branch this spawn takes.
        if resume_claude_session_id:
            argv = build_argv(
                system_prompt_path, mcp_config_path, task,
                model=model, use_bare=use_bare, claude_bin=claude_bin,
                resume_session_id=claude_session_id,
            )
        else:
            argv = build_argv(
                system_prompt_path, mcp_config_path, task,
                model=model, use_bare=use_bare, claude_bin=claude_bin,
                session_id=claude_session_id,
            )
        logger.info(
            "spawning agent %s (zone=%s) model=%s bare=%s claude=%s",
            agent_id, zone_id, model, use_bare, claude_bin,
        )
        try:
            # stdin is a JSON-line channel so send_user_turn can write
            # user-message frames to continue an existing conversation.
            # The agent process stays alive between turns; the stdout
            # reader threads pick up its response spans.
            popen = subprocess.Popen(
                argv, env=env, cwd=str(work_dir),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
            )
        except OSError as exc:
            _diagnostics.mark(
                "supervisor_spawn_popen_failed",
                agent_id=agent_id, error_type=type(exc).__name__, error=str(exc),
            )
            raise
        _diagnostics.mark(
            "supervisor_spawn_popen_started",
            agent_id=agent_id, pid=popen.pid,
        )
        handle = self._build_handle(
            popen, agent_id, zone_id, work_dir, task,
            claude_session_id=claude_session_id,
        )
        with self._lock:
            self._agents[agent_id] = handle
        _diagnostics.mark("supervisor_spawn_handle_built", agent_id=agent_id)
        return handle

    def get(self, agent_id: str) -> Optional[AgentHandle]:
        """Return the live handle for agent_id, or None if unknown."""
        with self._lock:
            return self._agents.get(agent_id)

    def list_agents(self) -> List[Dict[str, Any]]:
        """Return a read-only summary of every live (spawned) agent.

        One dict per handle currently tracked, with agent_id, zone_id,
        coarse state, spawn time, claude session id, last-seen turn, and
        whether the process is still alive (``poll()`` is None). The
        handle dict is snapshotted under the lock; ``poll()`` is called
        outside it so a slow reap never blocks other callers. Complements
        :meth:`get` (single lookup) with an enumeration the facade and
        other consumers can use.
        """
        with self._lock:
            handles = list(self._agents.values())
        return [
            {
                "agent_id": h.agent_id,
                "zone_id": h.zone_id,
                "state": h.state,
                "started_at": h.started_at,
                "claude_session_id": h.claude_session_id,
                "last_seen_turn_id": h.last_seen_turn_id,
                "alive": h.poll() is None,
            }
            for h in handles
        ]

    # Multi-turn continuation. Writes one stream-json user turn to the
    # live agent's stdin; if the agent has gone idle/exited, resume it
    # first.
    #
    # Returned status:
    #   "sent"              -- alive agent; one stdin write only.
    #   "resumed_then_sent" -- idle/exited agent resumed via --resume,
    #                          then stdin write.
    #   "spawned_fresh"     -- resume failed; a fresh session was minted,
    #                          the new handle is alive, and the stdin
    #                          write went through.
    def _resume_if_needed(
        self, handle: "AgentHandle", agent_id: str, fresh_task_text: str,
    ) -> Tuple["AgentHandle", str]:
        """Resume/respawn the agent if its process has reaped.

        Returns ``(live_handle, status)`` where status is ``"sent"`` (the
        handle was already alive), ``"resumed_then_sent"`` (re-attached to
        the preserved claude session), or ``"spawned_fresh"`` (no session
        to resume, or the resume attempt itself reaped).

        Extracted from send_user_turn (SE-04) so any writer path -- the
        external MCP transport, PBX, a library caller -- reuses the exact
        resume ladder instead of copy-pasting it. Idempotent when the
        handle is already live (returns it unchanged as ``"sent"``).
        """
        if handle.popen.poll() is None:
            return handle, "sent"
        from llm_kernel._internal import _diagnostics
        prior_session_id = handle.claude_session_id
        zone_id = handle.zone_id
        work_dir = handle.work_dir
        # Drop the dead handle before respawning so the live-agent
        # idempotency check in spawn does not return the now-reaped handle.
        with self._lock:
            if self._agents.get(agent_id) is handle:
                del self._agents[agent_id]
        _diagnostics.mark(
            "send_user_turn_resume_attempt",
            agent_id=agent_id, prior_session_id=prior_session_id,
        )
        if not prior_session_id:
            # No prior session id -> fresh spawn straight away.
            return (
                self.spawn(
                    zone_id=zone_id, agent_id=agent_id,
                    task=fresh_task_text, work_dir=work_dir,
                ),
                "spawned_fresh",
            )
        resumed = self.spawn(
            zone_id=zone_id, agent_id=agent_id,
            task=fresh_task_text, work_dir=work_dir,
            resume_claude_session_id=prior_session_id,
        )
        if not self._resume_failed(resumed):
            return resumed, "resumed_then_sent"
        self._record_resume_failure_k24(
            agent_id=agent_id, zone_id=zone_id,
            attempted_session_id=prior_session_id,
            exit_code=resumed.popen.poll(),
        )
        with self._lock:
            if self._agents.get(agent_id) is resumed:
                del self._agents[agent_id]
        return (
            self.spawn(
                zone_id=zone_id, agent_id=agent_id,
                task=fresh_task_text, work_dir=work_dir,
            ),
            "spawned_fresh",
        )

    def send_user_turn(
        self,
        agent_id: str,
        text: str,
        cell_id: Optional[str] = None,
        *,
        inject_notebook_prefix: bool = True,
    ) -> Dict[str, Any]:
        """Continue an existing agent's conversation with one user turn.

        Writes a stream-json user-message line to the agent's stdin. The
        agent's response streams back through the existing stdout reader
        thread.

        If the agent has gone idle or exited (its process has reaped),
        resume it first by respawning with its preserved
        claude_session_id, then write stdin.

        Args:
            agent_id: The agent to continue. Unknown ids raise KeyError.
            text: The operator's message body. Empty or whitespace-only
                input is rejected with ValueError.
            cell_id: Optional originating cell URI, threaded through for
                diagnostics and cell-to-turn bookkeeping.

        Returns:
            A dict with agent_id, status ("sent", "resumed_then_sent",
            or "spawned_fresh"), and cell_id.
        """
        from llm_kernel._internal import _diagnostics
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"send_user_turn: text must be a non-empty string "
                f"(agent_id={agent_id!r})"
            )
        with self._lock:
            handle = self._agents.get(agent_id)
        if handle is None:
            raise KeyError(agent_id)

        # Resume if needed: a reaped process is respawned/resumed here.
        # SE-04 extracted the ladder to _resume_if_needed so other writer
        # paths reuse it; it returns the live handle plus a status string.
        handle, status = self._resume_if_needed(handle, agent_id, text)

        # Cross-agent context handoff: walk the turn DAG between this
        # agent's last-seen turn and the current notebook head,
        # synthesize prefix lines for the missed sibling turns, and
        # inject them before the operator's message.
        # SE-03: a non-notebook caller (external MCP / PBX / library-
        # embedded use) passes inject_notebook_prefix=False to skip the
        # missed-turn walk entirely and leave the notebook cursor
        # untouched. Default True preserves the operator path.
        head_turn_id = self._notebook_head_turn_id()
        prefix_lines: List[str] = []
        if inject_notebook_prefix:
            try:
                missed = self._missed_turns(agent_id, head_turn_id)
                prefix_lines = self._synthesize_handoff_prefix(missed)
            except _HandoffCycleError as exc:
                from llm_kernel.wire.tools import K26_CROSS_AGENT_HANDOFF_FAILED
                _diagnostics.mark(
                    "send_user_turn_handoff_cycle_detected",
                    agent_id=agent_id, k_class=K26_CROSS_AGENT_HANDOFF_FAILED,
                    reason="cycle_detected",
                )
                raise RuntimeError(
                    f"{K26_CROSS_AGENT_HANDOFF_FAILED}: cross-agent handoff "
                    f"failed (reason: cycle_detected) for agent {agent_id!r}: "
                    f"{exc}"
                ) from exc

        # ContextPacker + RunFrame wiring. When the operator passes a
        # cell_id, pack the manifest, record it, and submit a start run
        # frame (status=running). On stdin-write failure submit a
        # terminal frame with status=failed; on success submit one with
        # status=complete. Without a cell_id the run still works, it
        # just produces no Inspect-mode trail.
        #
        # The terminal frame is submitted when the user turn is
        # delivered to the agent, not when the agent finishes replying
        # (the reply streams in asynchronously), so run-frame durations
        # read as roughly zero. The writer is idempotent on run_id, so a
        # later per-turn completion detector could resubmit with
        # accurate timing.
        run_id: Optional[str] = None
        manifest_id: Optional[str] = None
        if (
            cell_id is not None
            and isinstance(cell_id, str)
            and cell_id
            and self._metadata_writer is not None
        ):
            run_id, manifest_id = self._submit_context_pack_and_start_run_frame(
                agent_id=agent_id,
                cell_id=cell_id,
                turn_head_before=head_turn_id,
            )

        # Write the stream-json user turn. claude reads a user-message
        # frame from stdin when launched with --input-format=stream-json;
        # without that flag it treats stdin as the initial prompt only.
        operator_line = json.dumps({
            "type": "user",
            "message": {"role": "user", "content": text},
        }) + "\n"
        if handle.popen.stdin is None:
            # Defensive: spawn opens stdin=PIPE; a handle built without
            # stdin lands here.
            raise RuntimeError(
                f"send_user_turn: agent {agent_id!r} has no writable "
                f"stdin (was the popen built with stdin=PIPE?)"
            )
        try:
            # Serialize the whole prefix+operator write+flush per handle so
            # a concurrent caller cannot interleave its lines between ours
            # (SE-01). Single-writer paths are unaffected.
            with handle._stdin_lock:
                # Write prefix lines (handoff context) before operator message.
                for prefix_line in prefix_lines:
                    handle.popen.stdin.write(prefix_line + "\n")
                handle.popen.stdin.write(operator_line)
                handle.popen.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            from llm_kernel.wire.tools import K26_CROSS_AGENT_HANDOFF_FAILED
            _diagnostics.mark(
                "send_user_turn_stdin_write_failed",
                agent_id=agent_id, error_type=type(exc).__name__,
                k_class=K26_CROSS_AGENT_HANDOFF_FAILED if prefix_lines else None,
                reason="stdin_write_failed" if prefix_lines else None,
            )
            # Emit a terminal run frame with status=failed so Inspect
            # mode records the synchronous failure.
            if run_id is not None and cell_id is not None and manifest_id is not None:
                self._submit_terminal_run_frame(
                    run_id=run_id,
                    cell_id=cell_id,
                    executor_id=agent_id,
                    context_manifest_id=manifest_id,
                    turn_head_before=head_turn_id,
                    turn_head_after=head_turn_id,
                    status="failed",
                )
            raise
        # Advance last_seen_turn_id to the notebook head so the next call
        # computes the correct missed-turn delta. SE-03: only when this
        # turn participated in the notebook handoff -- a non-notebook
        # caller must not move the agent's notebook cursor.
        if inject_notebook_prefix:
            handle.last_seen_turn_id = head_turn_id
            # Persist via update_agent_session intent if a writer is wired.
            if self._metadata_writer is not None and head_turn_id is not None:
                try:
                    self._metadata_writer.submit_intent({
                        "payload": {
                            "action_type": "zone_mutate",
                            "intent_kind": "update_agent_session",
                            "parameters": {
                                "agent_id": agent_id,
                                "last_seen_turn_id": head_turn_id,
                            },
                            "intent_id": f"sut-uas-{agent_id}-{uuid.uuid4().hex[:8]}",
                        },
                    })
                except Exception:  # pragma: no cover — writer errors are best-effort
                    _diagnostics.mark(
                        "send_user_turn_update_session_intent_failed",
                        agent_id=agent_id,
                    )
        # Emit a terminal run frame with status=complete on a successful
        # stdin dispatch. "complete" here means the user turn was
        # delivered to the agent, not that the agent finished replying.
        # The writer is idempotent on run_id, so a later completion
        # detector can re-emit.
        if run_id is not None and cell_id is not None and manifest_id is not None:
            # Re-read the head turn id after the write. There's no
            # atomic turn-commit yet, so head_turn_after may equal
            # head_turn_before; the field is recorded either way.
            head_after = self._notebook_head_turn_id()
            self._submit_terminal_run_frame(
                run_id=run_id,
                cell_id=cell_id,
                executor_id=agent_id,
                context_manifest_id=manifest_id,
                turn_head_before=head_turn_id,
                turn_head_after=head_after,
                status="complete",
            )
        _diagnostics.mark(
            "send_user_turn_stdin_written",
            agent_id=agent_id, status=status, cell_id=cell_id,
            bytes_written=len(operator_line),
            handoff_prefix_count=len(prefix_lines),
        )
        return {
            "agent_id": agent_id,
            "status": status,
            "cell_id": cell_id,
            "handoff_prefix_count": len(prefix_lines),
            "run_id": run_id,
            "context_manifest_id": manifest_id,
        }

    # ------------------------------------------------------------------
    # ContextPacker + RunFrame integration helpers
    # ------------------------------------------------------------------

    def _runframe_now_iso(self) -> str:
        """Return the current UTC timestamp in ISO 8601, millisecond precision.

        Matches the manifest's generated_at format so the start and
        terminal run-frame timestamps line up with it.
        """
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    def _build_pack_snapshot(self) -> Dict[str, Any]:
        """Build the minimal overlay snapshot that pack() needs.

        The walker reads a cells map (with per-cell pinned / section_id
        / sub_turns / scratch / excluded / obsolete flags) and an
        ordering list (document order). Ordering may be absent, in which
        case the packer falls back to insertion order.

        Returns an empty dict when the writer has no cells map; pack()
        then raises an orphan-cell error that the caller catches and
        skips.
        """
        writer = self._metadata_writer
        if writer is None:
            return {}
        try:
            snap = writer.snapshot(trigger="save")
        except Exception:  # pragma: no cover — defensive
            return {}
        out: Dict[str, Any] = {}
        cells = snap.get("cells")
        if isinstance(cells, dict):
            out["cells"] = cells
        ordering = snap.get("ordering")
        if isinstance(ordering, list):
            out["ordering"] = ordering
        return out

    def _submit_context_pack_and_start_run_frame(
        self,
        *,
        agent_id: str,
        cell_id: str,
        turn_head_before: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Pack the manifest, persist it, and submit the start run frame.

        Wraps the pack call in a record_context_manifest intent and
        emits a start run frame (status=running, ended_at=null) before
        dispatching the user turn.

        All failure modes are caught and logged so a packer or writer
        error never blocks the operator's run; the operator just sees a
        degraded Inspect-mode trail. Returns (run_id, manifest_id) on
        success, (None, None) on degradation.
        """
        from llm_kernel._internal import _diagnostics
        writer = self._metadata_writer
        if writer is None:
            return None, None
        snapshot = self._build_pack_snapshot()
        try:
            manifest = context_packer.pack(cell_id, snapshot)
        except context_packer.K100OrphanCellError as exc:
            _diagnostics.mark(
                "contextpacker_orphan_cell",
                agent_id=agent_id, cell_id=cell_id,
                k_class="K100", reason=str(exc),
            )
            return None, None
        except context_packer.ContextPackerError as exc:
            _diagnostics.mark(
                "contextpacker_pack_failed",
                agent_id=agent_id, cell_id=cell_id, reason=str(exc),
            )
            return None, None

        manifest_id = manifest.get("manifest_id")
        if not isinstance(manifest_id, str) or not manifest_id:
            _diagnostics.mark(
                "contextpacker_missing_manifest_id",
                agent_id=agent_id, cell_id=cell_id,
            )
            return None, None

        try:
            writer.submit_intent({
                "payload": {
                    "action_type": "zone_mutate",
                    "intent_kind": "record_context_manifest",
                    "parameters": {"manifest": manifest},
                    "intent_id": f"sut-rcm-{manifest_id[:12]}",
                },
            })
        except Exception:  # pragma: no cover — writer errors are best-effort
            _diagnostics.mark(
                "send_user_turn_record_manifest_failed",
                agent_id=agent_id, cell_id=cell_id,
                manifest_id=manifest_id,
            )
            return None, None

        run_id = uuid.uuid4().hex
        started_at = self._runframe_now_iso()
        try:
            writer.submit_intent({
                "payload": {
                    "action_type": "zone_mutate",
                    "intent_kind": "record_run_frame",
                    "parameters": {
                        "run_frame": {
                            "run_id": run_id,
                            "cell_id": cell_id,
                            "executor_id": agent_id,
                            "turn_head_before": turn_head_before,
                            "turn_head_after": None,
                            "context_manifest_id": manifest_id,
                            "status": "running",
                            "started_at": started_at,
                            "ended_at": None,
                        },
                    },
                    "intent_id": f"sut-rrf-start-{run_id[:12]}",
                },
            })
        except Exception:  # pragma: no cover — writer errors are best-effort
            _diagnostics.mark(
                "send_user_turn_start_run_frame_failed",
                agent_id=agent_id, cell_id=cell_id, run_id=run_id,
            )
            return None, None

        _diagnostics.mark(
            "send_user_turn_run_frame_started",
            agent_id=agent_id, cell_id=cell_id, run_id=run_id,
            manifest_id=manifest_id,
        )
        # Increment the run count for the section containing this cell.
        # May flip the section to in-progress.
        self._maybe_flip_section_on_run_start(cell_id)
        return run_id, manifest_id

    def _submit_terminal_run_frame(
        self,
        *,
        run_id: str,
        cell_id: str,
        executor_id: str,
        context_manifest_id: str,
        turn_head_before: Optional[str],
        turn_head_after: Optional[str],
        status: str,
    ) -> None:
        """Submit the terminal run-frame intent (complete, failed, or interrupted).

        The writer is idempotent on run_id for same-cell resubmissions,
        so this may be called more than once for a single run (e.g. the
        synchronous failure path plus a later completion detector). All
        failure modes are best-effort: a writer outage is logged but
        never blocks the caller.
        """
        from llm_kernel._internal import _diagnostics
        writer = self._metadata_writer
        if writer is None:
            return
        ended_at = self._runframe_now_iso()
        try:
            writer.submit_intent({
                "payload": {
                    "action_type": "zone_mutate",
                    "intent_kind": "record_run_frame",
                    "parameters": {
                        "run_frame": {
                            "run_id": run_id,
                            "cell_id": cell_id,
                            "executor_id": executor_id,
                            "turn_head_before": turn_head_before,
                            "turn_head_after": turn_head_after,
                            "context_manifest_id": context_manifest_id,
                            "status": status,
                            # started_at is required, but we don't carry
                            # the start time across the call, so we
                            # re-stamp with the terminal time here.
                            "started_at": ended_at,
                            "ended_at": ended_at,
                        },
                    },
                    "intent_id": f"sut-rrf-end-{run_id[:12]}-{status}",
                },
            })
        except Exception:  # pragma: no cover — writer errors are best-effort
            _diagnostics.mark(
                "send_user_turn_terminal_run_frame_failed",
                run_id=run_id, cell_id=cell_id, status=status,
            )
            return
        _diagnostics.mark(
            "send_user_turn_run_frame_terminal",
            run_id=run_id, cell_id=cell_id, status=status,
        )
        # Decrement the run count for the section containing this cell.
        # May flip the section back to open.
        self._maybe_flip_section_on_run_end(cell_id)

    # ------------------------------------------------------------------
    # Section status auto-flip
    # ------------------------------------------------------------------
    #
    # When an agent run starts in a cell that belongs to a section, the
    # section flips from open to in-progress so the UI shows it is being
    # worked on and structural ops on its cells are blocked while the
    # run is in flight. Reference-counted: only flip on the first run
    # (0->1) and the last run (1->0) so concurrent runs in the same
    # section don't thrash the status. The flip is a best-effort intent;
    # if the section is frozen the writer declines it and the run still
    # proceeds (auto-flip never overrides operator intent). Failures are
    # logged and never propagate.

    def _maybe_flip_section_on_run_start(self, cell_id: str) -> None:
        """Increment the run count for cell_id's section; flip it from
        open to in-progress if this is the first active run.

        No-op when the cell has no section or when the writer is not
        wired. Safe to call repeatedly; the ref-count math handles it.
        Thread-safe.
        """
        from llm_kernel._internal import _diagnostics
        writer = self._metadata_writer
        if writer is None:
            return
        rec = writer.get_cell_record(cell_id)
        if not isinstance(rec, dict):
            return
        section_id = rec.get("section_id")
        if not isinstance(section_id, str) or not section_id:
            return
        with self._auto_flip_lock:
            count = self._in_progress_run_count.get(section_id, 0)
            self._in_progress_run_count[section_id] = count + 1
            should_flip = count == 0
        if not should_flip:
            return
        self._submit_section_status_flip(
            section_id=section_id,
            new_status="in_progress",
            cell_id=cell_id,
            reason_marker="auto_flip_section_in_progress",
        )

    def _maybe_flip_section_on_run_end(self, cell_id: str) -> None:
        """Decrement the run count for cell_id's section; flip it from
        in-progress to open if this was the last active run.

        Pairs with the on-run-start method. Extra decrements clamp at
        zero rather than going negative.
        """
        from llm_kernel._internal import _diagnostics
        writer = self._metadata_writer
        if writer is None:
            return
        rec = writer.get_cell_record(cell_id)
        if not isinstance(rec, dict):
            return
        section_id = rec.get("section_id")
        if not isinstance(section_id, str) or not section_id:
            return
        with self._auto_flip_lock:
            count = self._in_progress_run_count.get(section_id, 0)
            new_count = count - 1 if count > 0 else 0
            self._in_progress_run_count[section_id] = new_count
            should_flip = count > 0 and new_count == 0
        if not should_flip:
            return
        self._submit_section_status_flip(
            section_id=section_id,
            new_status="open",
            cell_id=cell_id,
            reason_marker="auto_flip_section_open",
        )

    def _submit_section_status_flip(
        self,
        *,
        section_id: str,
        new_status: str,
        cell_id: str,
        reason_marker: str,
    ) -> None:
        """Submit one set_section_status overlay intent.

        Failures (transport error, unknown section, frozen section) are
        logged but never raised — a section bookkeeping error must not
        block the supervisor's run path. A frozen section is expected;
        it gets a distinct marker so the audit trail tells "frozen,
        auto-flip declined" apart from "writer down, auto-flip lost".
        """
        from llm_kernel._internal import _diagnostics
        writer = self._metadata_writer
        if writer is None:
            return
        intent_id = f"af-{new_status[:3]}-{section_id[:8]}-{uuid.uuid4().hex[:8]}"
        try:
            result = writer.submit_intent({
                "payload": {
                    "action_type": "zone_mutate",
                    "intent_kind": "apply_overlay_commit",
                    "parameters": {
                        "operations": [{
                            "kind": "set_section_status",
                            "section_id": section_id,
                            "new_status": new_status,
                        }],
                        "message": (
                            f"auto-flip: section→{new_status} "
                            f"(cell {cell_id})"
                        ),
                    },
                    "intent_id": intent_id,
                },
            })
        except Exception:
            _diagnostics.mark(
                f"{reason_marker}_failed",
                cell_id=cell_id, section_id=section_id,
                reason="writer_raised",
            )
            return
        if not result.get("applied", False):
            # Unknown section, forbidden transition, or frozen section —
            # all expected on degraded paths. Mark for the audit trail;
            # the run continues unaffected.
            details = (result.get("response") or {}).get("details") or {}
            _diagnostics.mark(
                f"{reason_marker}_declined",
                cell_id=cell_id, section_id=section_id,
                error_code=result.get("error_code"),
                reason=details.get("reason"),
            )
            return
        _diagnostics.mark(
            reason_marker,
            cell_id=cell_id, section_id=section_id,
            new_status=new_status,
        )

    # ------------------------------------------------------------------
    # Cross-agent context handoff helpers
    # ------------------------------------------------------------------

    # Maximum turn-chain depth the missed-turn walker traverses before
    # treating the chain as a cycle. A single chain longer than this is
    # pathological; 1000 is generous.
    _HANDOFF_MAX_DEPTH: int = 1_000

    def _missed_turns(
        self,
        agent_id: str,
        head_turn_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Walk the turn DAG from head_turn_id back to the agent's
        last-seen turn.

        Returns the chain in chronological (root to head) order,
        filtered to turns not authored by agent_id. Returns an empty
        list when there are no turns, or when the agent's last-seen turn
        already equals head_turn_id.

        Raises _HandoffCycleError when the chain exceeds the max depth.
        """
        with self._lock:
            handle = self._agents.get(agent_id)
            last_seen = handle.last_seen_turn_id if handle is not None else None
        turns_snapshot, _ = self._read_persisted_turns()

        if head_turn_id is None:
            # No turns in the zone yet.
            return []
        if head_turn_id == last_seen:
            # Agent is already up to date.
            return []

        # Walk backward from head to last_seen (exclusive).
        chain: List[Dict[str, Any]] = []
        current_id: Optional[str] = head_turn_id
        visited: set = set()
        depth = 0
        while current_id is not None and current_id != last_seen:
            if depth > self._HANDOFF_MAX_DEPTH:
                raise _HandoffCycleError(
                    f"DAG walk exceeded max depth {self._HANDOFF_MAX_DEPTH} "
                    f"starting from {head_turn_id!r}; possible cycle."
                )
            if current_id in visited:
                raise _HandoffCycleError(
                    f"Cycle detected at turn {current_id!r} while walking "
                    f"DAG from {head_turn_id!r}."
                )
            visited.add(current_id)
            turn = turns_snapshot.get(current_id)
            if turn is None:
                # Turn referenced but not stored; stop here.
                break
            chain.append(turn)
            current_id = turn.get("parent_id")
            depth += 1

        # Reverse so the list is chronological (root → head).
        chain.reverse()
        # Filter out turns authored by this agent.
        return [t for t in chain if t.get("agent_id") != agent_id]

    def _synthesize_handoff_prefix(
        self,
        turns: List[Dict[str, Any]],
    ) -> List[str]:
        """Build stream-json prefix lines for the given turns.

        One user-message line per turn, whose content reads
        "<role> <agent_id> said: <body>". The body is stripped of magic
        hashes before wrapping so agents never observe hashed magic
        patterns.

        Returns a list of JSON strings without trailing newlines; the
        caller appends the newline when writing to stdin.
        """
        from llm_kernel.text.magic_hash import strip_hashes_from_text
        from llm_kernel.text.magic_registry import CELL_MAGICS, LINE_MAGICS
        known_names: set = set(CELL_MAGICS) | set(LINE_MAGICS)

        result: List[str] = []
        for turn in turns:
            role = turn.get("role", "assistant")
            author = turn.get("agent_id", "unknown")
            raw_body = turn.get("content", "")
            stripped_body = strip_hashes_from_text(raw_body, known_names)
            content = f"{role} {author} said: {stripped_body}"
            line = json.dumps({
                "type": "user",
                "message": {"role": "user", "content": content},
            })
            result.append(line)
        return result

    # Interrupt method paired with the cell-toolbar interrupt button:
    # the extension sends an agent_interrupt action, which the
    # dispatcher routes here.
    def interrupt(self, agent_id: str) -> Dict[str, Any]:
        """Send SIGINT to agent_id's live process.

        How this differs from stop:

        * stop is a clean shutdown -- SIGTERM, status goes idle, and the
          conversation is resumable on the next turn.
        * interrupt is an in-flight cancellation -- SIGINT so claude
          aborts its current generation while the process stays alive
          for the next turn. No resume cycle.

        Returns:
            ``{"agent_id": ..., "status": "interrupted" | "not_running"
            | "unknown"}``

            * ``"interrupted"`` -- SIGINT was delivered to a live PID.
            * ``"not_running"`` -- the agent is registered with this
              supervisor but its process has reaped (idle / exited
              runtime status); SIGINT would target a stale or recycled
              PID so we refuse.
            * ``"unknown"`` -- ``agent_id`` is not registered with this
              supervisor (no spawn record).

        Best-effort: a ``ProcessLookupError`` on the kill (race against
        the process reaping between ``poll()`` and ``os.kill()``)
        downgrades to ``"not_running"`` rather than raising.
        """
        from llm_kernel._internal import _diagnostics
        with self._lock:
            handle = self._agents.get(agent_id)
        if handle is None:
            _diagnostics.mark(
                "supervisor_interrupt_unknown_agent",
                agent_id=agent_id,
            )
            return {"agent_id": agent_id, "status": "unknown"}
        if handle.popen is None or handle.popen.poll() is not None:
            _diagnostics.mark(
                "supervisor_interrupt_not_running",
                agent_id=agent_id,
                state=handle.state,
            )
            return {"agent_id": agent_id, "status": "not_running"}
        pid = getattr(handle.popen, "pid", None)
        if pid is None:
            _diagnostics.mark(
                "supervisor_interrupt_no_pid",
                agent_id=agent_id,
            )
            return {"agent_id": agent_id, "status": "not_running"}
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            _diagnostics.mark(
                "supervisor_interrupt_pid_gone",
                agent_id=agent_id, pid=pid,
            )
            return {"agent_id": agent_id, "status": "not_running"}
        _diagnostics.mark(
            "supervisor_interrupt_sigint_sent",
            agent_id=agent_id, pid=pid,
        )
        logger.info(
            "agent.interrupted agent_id=%s pid=%s",
            agent_id, pid,
            extra={
                "event.name": "agent.interrupted",
                "llmnb.agent_id": agent_id,
                "llmnb.pid": pid,
            },
        )
        return {"agent_id": agent_id, "status": "interrupted"}

    # ------------------------------------------------------------------
    # Revert operation
    # ------------------------------------------------------------------

    # Error code for a revert target not in the agent's ancestry.
    K22_INVALID_REVERT_TARGET: str = "K22"

    def revert(self, agent_id: str, target_turn_id: str) -> None:
        """Move agent_id's head backward to target_turn_id.

        Steps:

        1. Validate the agent exists.
        2. Validate target_turn_id is in the agent's ancestry by walking
           parent_id from the current head.
        3. SIGTERM the live process if it has not yet exited. The
           watchdog handles cleanup; exit code 0 is expected.
        4. Submit a move_agent_head intent to the writer, setting the
           head and last-seen turns to target_turn_id.
        5. Submit a record_event intent for the head move.
        6. Mint a fresh claude_session_id for the next continue; the old
           one stays attached to the historical turns.

        Args:
            agent_id: The agent whose head to move.
            target_turn_id: The turn to revert to; must be in the
                agent's ancestry.

        Raises:
            RuntimeError: if the agent is unknown, or if target_turn_id
                is not in the agent's ancestry.
        """
        from llm_kernel._internal import _diagnostics

        with self._lock:
            handle = self._agents.get(agent_id)
            if handle is None:
                raise RuntimeError(
                    f"K20: agent {agent_id!r} not found in supervisor. "
                    "Spawn or respawn the agent before calling revert."
                )

            # --- ancestry walk -----------------------------------------
            # Read the persisted turn graph and walk parent_id from the
            # notebook head back through the union of all agents' chains
            # looking for target_turn_id.
            turn_index, current_head = self._read_persisted_turns()
            found = False
            visited: set = set()
            cursor = current_head
            depth = 0
            while cursor is not None and depth < self._HANDOFF_MAX_DEPTH:
                if cursor in visited:
                    break  # cycle guard
                visited.add(cursor)
                if cursor == target_turn_id:
                    found = True
                    break
                turn_rec = turn_index.get(cursor)
                if turn_rec is None:
                    break
                cursor = turn_rec.get("parent_id")
                depth += 1

            if not found:
                raise RuntimeError(
                    f"{self.K22_INVALID_REVERT_TARGET}: "
                    f"turn {target_turn_id!r} is not in agent "
                    f"{agent_id!r}'s ancestry.  Use @branch to reach a "
                    "turn outside this agent's lineage."
                )

            prior_head = current_head

            # --- SIGTERM live process -----------------------------------
            if handle.popen.poll() is None:
                try:
                    handle.popen.terminate()
                except (OSError, ProcessLookupError):  # pragma: no cover
                    pass
                _diagnostics.mark(
                    "revert_sigterm_sent",
                    agent_id=agent_id, target_turn_id=target_turn_id,
                )

            # --- in-memory head update ---------------------------------
            handle.last_seen_turn_id = target_turn_id
            # The notebook head is computed from the persisted turns on
            # demand; the move_agent_head intent submitted below records
            # the new head as the canonical post-revert position.

            # --- mint fresh claude_session_id for next continue --------
            handle.claude_session_id = str(uuid.uuid4())

        # --- writer intents (outside lock) -----------------------------
        if self._metadata_writer is not None:
            # move_agent_head mutates the agent's session head and
            # last-seen turns and bumps the snapshot version.
            self._metadata_writer.submit_intent({
                "payload": {
                    "action_type": "zone_mutate",
                    "intent_kind": "move_agent_head",
                    "parameters": {
                        "agent_id": agent_id,
                        "head_turn_id": target_turn_id,
                        "last_seen_turn_id": target_turn_id,
                    },
                    "intent_id": f"revert-mah-{agent_id}-{target_turn_id}-{uuid.uuid4().hex[:8]}",
                },
            })

            # Log the head move as a record_event with
            # parameters.kind = 'agent_ref_move' (not as an intent_kind
            # of that name, which would be rejected).
            self._metadata_writer.submit_intent({
                "payload": {
                    "action_type": "zone_mutate",
                    "intent_kind": "record_event",
                    "parameters": {
                        "kind": "agent_ref_move",
                        "reason": "operator_revert",
                        "agent_id": agent_id,
                        "from_turn_id": prior_head,
                        "to_turn_id": target_turn_id,
                    },
                    "intent_id": f"revert-arm-{agent_id}-{target_turn_id}-{uuid.uuid4().hex[:8]}",
                },
            })

        _diagnostics.mark(
            "revert_complete",
            agent_id=agent_id,
            target_turn_id=target_turn_id,
        )

    # ------------------------------------------------------------------
    # Stop operation
    # ------------------------------------------------------------------

    def stop(self, agent_id: str) -> None:
        """Cleanly SIGTERM agent_id's runtime process.

        Steps:

        1. Validate the agent exists.
        2. If the process has already exited, log a diagnostic and
           return — stop is idempotent, so an already-idle agent is fine.
        3. SIGTERM the live process. The watchdog observes the exit;
           exit code 0 is expected.
        4. Mark the agent idle.
        5. Submit an update_agent_session intent with the new status
           (best-effort).

        What stop does NOT do (unlike revert):
        - It does not mint a new claude_session_id — stop preserves the
          session so the next continue can --resume.
        - It does not touch the head or last-seen turns — stop does not
          move the head; revert does.

        Args:
            agent_id: The agent to stop.

        Raises:
            RuntimeError: if the agent is unknown.
        """
        from llm_kernel._internal import _diagnostics

        with self._lock:
            handle = self._agents.get(agent_id)
            if handle is None:
                raise RuntimeError(
                    f"K20: agent {agent_id!r} not found in supervisor. "
                    "Spawn or respawn the agent before calling stop."
                )

            # --- idempotency guard: already idle / exited ---------------
            if handle.popen.poll() is not None:
                # Process has already exited; nothing to do. Log the
                # no-op so the operator sees it.
                _diagnostics.mark(
                    "stop_noop_already_idle",
                    agent_id=agent_id,
                )
                return

            # --- SIGTERM live process -----------------------------------
            try:
                handle.popen.terminate()
            except (OSError, ProcessLookupError):  # pragma: no cover
                pass
            _diagnostics.mark(
                "stop_sigterm_sent",
                agent_id=agent_id,
            )

            # --- in-memory status update --------------------------------
            # pid lives on handle.popen, not a separate field; the agent
            # schema exposes pid: null when idle. The writer intent below
            # carries the authoritative state.

        # --- writer intent (outside lock; best-effort) ------------------
        if self._metadata_writer is not None:
            try:
                self._metadata_writer.submit_intent({
                    "payload": {
                        "action_type": "zone_mutate",
                        "intent_kind": "update_agent_session",
                        "parameters": {
                            "agent_id": agent_id,
                            "runtime_status": "idle",
                            "pid": None,
                        },
                        "intent_id": f"stop-uas-{agent_id}-{uuid.uuid4().hex[:8]}",
                    },
                })
            except Exception:  # pragma: no cover — best-effort; writer errors are non-fatal
                _diagnostics.mark(
                    "stop_update_agent_session_failed",
                    agent_id=agent_id,
                )

        _diagnostics.mark(
            "stop_complete",
            agent_id=agent_id,
        )

    # ------------------------------------------------------------------
    # Branch / fork operation
    # ------------------------------------------------------------------

    # Error code for an invalid branch target (new_agent_id already exists).
    K21_INVALID_BRANCH_TARGET: str = "K21"

    def fork(
        self,
        source_agent_id: str,
        at_turn_id: Optional[str],
        new_agent_id: str,
    ) -> "AgentHandle":
        """Create a new agent branching from source_agent_id.

        Steps:

        1. Validate source_agent_id exists.
        2. Validate new_agent_id does not already exist.
        3. Case A (at_turn_id is None or equals the source's head):
           - The new agent's head and last-seen turns are the source's
             head.
           - A fresh claude_session_id is minted (the source session is
             not shared).
           - The new agent starts idle, with no live process; it spawns
             on the first continue-turn.
        4. Case B (at_turn_id is an ancestor):
           - Validate at_turn_id is in the source head's ancestry by
             walking parent_id.
           - The new agent's head and last-seen turns are at_turn_id.
           - A fresh claude_session_id; the agent starts idle.
           - Replaying the transcript up to at_turn_id happens lazily on
             the first continue-turn, not here.
        5. Insert the new handle into the agent map.
        6. Submit a fork_agent intent to the writer (best-effort).
        7. Submit a record_event intent for the branch.
        8. Return the new handle.

        Args:
            source_agent_id: The agent to branch from.
            at_turn_id: The turn to branch at. None or the source's head
                triggers Case A; any ancestor turn triggers Case B.
            new_agent_id: The id for the new agent; must be unique.

        Returns:
            The new AgentHandle (idle; no live process yet).

        Raises:
            RuntimeError: if source_agent_id is unknown, if new_agent_id
                already exists, or if at_turn_id is not in the source
                agent's ancestry.
        """
        from llm_kernel._internal import _diagnostics

        with self._lock:
            # --- validate source exists --------------------------------
            source_handle = self._agents.get(source_agent_id)
            if source_handle is None:
                raise RuntimeError(
                    f"K20: source agent {source_agent_id!r} not found in "
                    "supervisor.  Spawn the source agent before branching."
                )

            # --- validate new_agent_id is unique -----------------------
            if new_agent_id in self._agents:
                raise RuntimeError(
                    f"{self.K21_INVALID_BRANCH_TARGET}: "
                    f"new_agent_id {new_agent_id!r} already exists in "
                    "supervisor.  Choose a different new_agent_id."
                )

            # --- resolve effective branch point ------------------------
            turn_index, source_head = self._read_persisted_turns()

            # Determine Case A vs Case B.
            # Case A: at_turn_id is None OR equals the source's current head.
            if at_turn_id is None or at_turn_id == source_head:
                branch_turn_id = source_head
                case = "A"
            else:
                # Case B: at_turn_id must be reachable via parent_id walk.
                found = False
                visited: set = set()
                cursor = source_head
                depth = 0
                while cursor is not None and depth < self._HANDOFF_MAX_DEPTH:
                    if cursor in visited:
                        break  # cycle guard
                    visited.add(cursor)
                    if cursor == at_turn_id:
                        found = True
                        break
                    turn_rec = turn_index.get(cursor)
                    if turn_rec is None:
                        break
                    cursor = turn_rec.get("parent_id")
                    depth += 1

                if not found:
                    raise RuntimeError(
                        f"{self.K22_INVALID_REVERT_TARGET}: "
                        f"turn {at_turn_id!r} is not in agent "
                        f"{source_agent_id!r}'s ancestry.  Use a turn "
                        "reachable via parent_id walk from the source head."
                    )
                branch_turn_id = at_turn_id
                case = "B"

            # --- mint fresh claude_session_id for the new agent --------
            new_session_id = str(uuid.uuid4())

            # --- build the new AgentHandle (metadata-only; no process) -
            # Use a sentinel Popen-like object whose poll() returns a
            # non-None code, so callers treat the agent as already
            # exited (idle).
            import types
            idle_popen: Any = types.SimpleNamespace(
                poll=lambda: 0,
                returncode=0,
                pid=None,
                terminate=lambda: None,
                kill=lambda: None,
                wait=lambda **_: 0,
                stdin=None,
                stdout=iter([]),
                stderr=iter([]),
            )
            now = time.monotonic()
            new_handle = AgentHandle(
                agent_id=new_agent_id,
                zone_id=source_handle.zone_id,
                popen=idle_popen,
                started_at=now,
                work_dir=source_handle.work_dir,
                stdout_thread=threading.Thread(),
                stderr_thread=threading.Thread(),
                claude_session_id=new_session_id,
                last_seen_turn_id=branch_turn_id,
            )
            new_handle.state = "terminated"  # idle; no live process

            self._agents[new_agent_id] = new_handle

        # --- writer intents (outside lock) -----------------------------
        if self._metadata_writer is not None:
            # fork_agent persists the new agent into the zone's agent map.
            self._metadata_writer.submit_intent({
                "payload": {
                    "action_type": "zone_mutate",
                    "intent_kind": "fork_agent",
                    "parameters": {
                        "source_agent_id": source_agent_id,
                        "new_agent_id": new_agent_id,
                        "at_turn_id": branch_turn_id,
                        "case": case,
                        "claude_session_id": new_session_id,
                    },
                    "intent_id": f"fork-{source_agent_id}-{new_agent_id}-{uuid.uuid4().hex[:8]}",
                },
            })

            # Log the branch as an agent_ref_move event.
            self._metadata_writer.submit_intent({
                "payload": {
                    "action_type": "zone_mutate",
                    "intent_kind": "record_event",
                    "parameters": {
                        "kind": "agent_ref_move",
                        "reason": "operator_branch",
                        "agent_id": new_agent_id,
                        "from_turn_id": source_head,
                        "to_turn_id": branch_turn_id,
                    },
                    "intent_id": f"fork-arm-{new_agent_id}-{uuid.uuid4().hex[:8]}",
                },
            })

        _diagnostics.mark(
            "fork_complete",
            source_agent_id=source_agent_id,
            new_agent_id=new_agent_id,
            at_turn_id=branch_turn_id,
            case=case,
        )
        return new_handle

    def terminate_all(self, grace_seconds: float = 10.0) -> None:
        """Graceful shutdown of every active agent."""
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
        *, claude_session_id: str = "",
    ) -> AgentHandle:
        """Allocate the handle and start the reader/watchdog threads.

        claude_session_id is the kernel-owned UUID passed to claude via
        --session-id; it persists on the handle so a future --resume can
        continue across spawn cycles.
        """
        spawn_dir = work_dir / ".run" / agent_id
        stderr_log = spawn_dir / f"kernel.stderr.{agent_id}.log"
        now = time.monotonic()
        handle = AgentHandle(
            agent_id=agent_id, zone_id=zone_id, popen=popen,
            started_at=now, work_dir=work_dir,
            stdout_thread=threading.Thread(),  # placeholder
            stderr_thread=threading.Thread(),  # placeholder
            _last_stdout_ts=now,
            claude_session_id=claude_session_id,
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
        handle.silence_watchdog_thread = threading.Thread(
            target=self._silence_watchdog, args=(handle,),
            name=f"agent-silence-{agent_id}", daemon=True,
        )
        handle.silence_watchdog_thread.start()
        handle.state = "running"
        return handle

    def _read_stdout(self, handle: AgentHandle) -> None:
        """Parse each stdout line as stream-json or MCP JSON-RPC.

        Lines that fail both parsers are prose violations; each is logged
        via a synthetic notify and counted toward the flood threshold.
        Lines that parse as MCP tools/call frames are recorded as a
        courtesy run.

        When LLMKERNEL_DEBUG_STDOUT=1 is set, every raw stdout line is
        also appended to a per-agent log for post-mortem inspection, and
        a pretty-printed copy of each parseable JSON event is appended to
        a companion .pretty.log.
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
            # Hang detection: every byte on stdout (even an empty line)
            # resets the silence timer, so the watchdog only fires on
            # true silence, not on parser-rejection silence.
            handle._last_stdout_ts = time.monotonic()
            line = raw.rstrip("\r\n")
            if debug_path is not None:
                with debug_path.open("a", encoding="utf-8") as fh:
                    fh.write(raw)
            if not line:
                continue
            parsed, parse_error = self._try_parse_json_with_error(line)
            if pretty_path is not None and parsed is not None:
                _append_pretty(pretty_path, parsed)
            if parsed is None:
                # Lines that fail both parsers become agent_emit spans
                # of kind malformed_json, carrying the short parser
                # error string.
                self._emit_agent_emit(
                    handle, emit_kind="malformed_json",
                    emit_content=line, parser_diagnostic=parse_error,
                )
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
        """Append stderr to a per-agent log and emit an agent_emit span per line.

        stderr is captured line by line and emitted as agent_emit spans
        of kind "stderr"; the raw stream is also written to a per-agent
        log file for debugging. Renderers may collapse stderr by default,
        but the data must reach the operator surface — no silent drops.
        """
        assert handle.popen.stderr is not None
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                for raw in handle.popen.stderr:
                    if handle._stop_event.is_set():
                        break
                    fh.write(raw)
                    fh.flush()
                    line = raw.rstrip("\r\n")
                    if not line:
                        continue
                    self._emit_agent_emit(
                        handle, emit_kind="stderr", emit_content=line,
                    )
        except OSError:  # pragma: no cover - defensive
            logger.exception("agent %s stderr log write failed", handle.agent_id)

    def _watchdog(self, handle: AgentHandle, task: str) -> None:
        """Monitor process exit and restart the agent.

        Up to three restarts in a 5-minute sliding window per agent;
        after exhaustion emits a synthetic problem report and leaves the
        agent in the failed state. Per-agent, not zone-wide.
        """
        exit_code = handle.popen.wait()
        if handle._stop_event.is_set() or (
            exit_code == 0 and not handle._hang_terminated
        ):
            handle.state = "terminated"
            return
        if handle._hang_terminated:
            # The silence watchdog has already logged the hang event and
            # SIGTERMed the process. Reset the flag so the next
            # crash-restart iteration can run cleanly.
            handle._hang_terminated = False
            logger.warning(
                "agent %s SIGTERMed by silence watchdog; restart attempt %d",
                handle.agent_id, len(handle._restart_history) + 1,
            )
        else:
            logger.warning(
                "agent %s exited code=%s (attempt %d)",
                handle.agent_id, exit_code,
                len(handle._restart_history) + 1,
            )
        # Sliding-window check: prune entries older than the restart
        # window; if we're at or past the max, refuse another restart.
        now = time.monotonic()
        cutoff = now - _RESTART_WINDOW_SEC
        with self._lock:
            while handle._restart_history and handle._restart_history[0] < cutoff:
                handle._restart_history.popleft()
            if len(handle._restart_history) >= _RESTART_WINDOW_MAX:
                self._record_synthetic_problem(
                    handle.agent_id, handle.zone_id,
                    "agent unrestartable: 3 restart attempts in 5 minutes",
                    "agent.unrestartable",
                )
                handle.state = "failed"
                return
            handle._restart_history.append(now)
            attempt_idx = min(
                len(handle._restart_history) - 1,
                len(_RESTART_BACKOFFS_SEC) - 1,
            )
        backoff = _RESTART_BACKOFFS_SEC[attempt_idx]
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
        """Spawn a new claude process replacing the dead one in the handle."""
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
        # New popen -> reset silence timer so the just-spawned process
        # has the full silence budget to produce output.
        handle._last_stdout_ts = time.monotonic()
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
        # Restart the silence watchdog if the previous one exited.
        existing_silence = handle.silence_watchdog_thread
        if existing_silence is None or not existing_silence.is_alive():
            handle.silence_watchdog_thread = threading.Thread(
                target=self._silence_watchdog, args=(handle,),
                name=f"agent-silence-{handle.agent_id}", daemon=True,
            )
            handle.silence_watchdog_thread.start()
        handle.state = "running"

    def _silence_watchdog(self, handle: AgentHandle) -> None:
        """Hang detection: SIGTERM if stdout is silent past the threshold.

        When the agent emits no stdout for longer than the threshold and
        the process is still running, SIGTERM it and let the crash-
        restart machinery in the watchdog pick up the resulting non-zero
        exit (with the hang-terminated flag set so the log line
        attributes the cause correctly).

        Loops at the configured granularity and exits when the process
        has reaped or the stop event is set. A daemon thread, so it
        never prevents process teardown.
        """
        threshold = self._silence_threshold_sec
        granularity = self._silence_granularity_sec
        # Defensive lower bound — granularity 0 would burn the CPU.
        if granularity <= 0:
            granularity = _SILENCE_WATCHDOG_GRANULARITY_SEC
        while not handle._stop_event.is_set():
            if handle._stop_event.wait(timeout=granularity):
                return
            if handle.popen.returncode is not None:
                return
            silence = time.monotonic() - handle._last_stdout_ts
            if silence <= threshold:
                continue
            # Mark first so the watchdog sees the flag even if the
            # SIGTERM races. Emit one error-level log event carrying the
            # silence duration.
            handle._hang_terminated = True
            logger.error(
                "agent.hang_detected agent_id=%s silence_seconds=%.1f",
                handle.agent_id, silence,
                extra={
                    "event.name": "agent.hang_detected",
                    "llmnb.agent_id": handle.agent_id,
                    "llmnb.silence_seconds": silence,
                },
            )
            try:
                handle.popen.terminate()
            except (OSError, ProcessLookupError):  # pragma: no cover - defensive
                pass
            return

    def _validate_template_version(self, rendered_prompt: str) -> None:
        """Refuse spawn on a system-prompt-template major-version mismatch.

        Major mismatch -> raise PreSpawnValidationError. Minor mismatch
        -> log a warning and proceed. Patch differences are silent. A
        missing version marker -> warn (we just rendered the canonical
        template, so it's a developer bug, not an attack).
        """
        seen = extract_template_version(rendered_prompt)
        expected = EXPECTED_SYSTEM_PROMPT_TEMPLATE_VERSION
        if seen is None:
            logger.warning(
                "system prompt template missing version marker; "
                "expected v%s embedded in trailing comment",
                expected,
            )
            return
        try:
            seen_major, seen_minor, _ = _split_semver(seen)
            exp_major, exp_minor, _ = _split_semver(expected)
        except ValueError as exc:
            raise PreSpawnValidationError(
                f"system prompt template version unparseable: "
                f"expected {expected}, got {seen!r} ({exc})",
                log_signature="provisioning.template.version_mismatch",
            ) from exc
        if seen_major != exp_major:
            raise PreSpawnValidationError(
                f"system prompt template version mismatch: "
                f"expected {expected}, got {seen}",
                log_signature="provisioning.template.version_mismatch",
            )
        if seen_minor != exp_minor:
            logger.warning(
                "system prompt template minor version drift: "
                "expected %s, got %s; proceeding",
                expected, seen,
            )

    # -- Config-driven respawn (consumed by the hydrate handler) -----

    def respawn_from_config(
        self, config_recoverable_agents: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Spawn (or rebind) every agent in a recoverable-agents list.

        The hydrate handler calls this with the recoverable agent list;
        each entry carries agent_id, zone_id, tools_allowed, and
        optionally model, system_prompt_template_id, system_prompt_hash,
        and a task string. Returns {agent_id: status} where status is
        "spawned", "failed", or "skipped".

        Idempotency: agents already running are skipped. One bad entry
        must not block the rest; each spawn is wrapped in its own
        try-block.

        The recoverable schema does not carry the task string; callers
        pass it as a synthetic "task" key inside each entry. A missing
        task fails that entry only.
        """
        results: Dict[str, str] = {}
        for entry in config_recoverable_agents:
            agent_id = entry.get("agent_id") if isinstance(entry, dict) else None
            if not isinstance(agent_id, str) or not agent_id:
                # Cannot key by agent_id; surface a sentinel so the
                # caller can correlate by index.
                key = f"<malformed-{len(results)}>"
                results[key] = "failed"
                logger.error("respawn_from_config: malformed entry %r", entry)
                continue
            try:
                with self._lock:
                    already = self._agents.get(agent_id)
                if already is not None and already.popen.poll() is None:
                    results[agent_id] = "skipped"
                    continue
                self._spawn_from_config_entry(entry)
                results[agent_id] = "spawned"
            except Exception:  # noqa: BLE001 — isolate per-agent failures
                logger.exception(
                    "respawn_from_config: spawn failed for %s", agent_id,
                )
                results[agent_id] = "failed"
        return results

    # Statuses that signal a resumable claude session. "alive" is
    # included because a snapshot taken before clean shutdown may still
    # carry that status even though the process has since gone away;
    # that case is treated as idle.
    _RESUMABLE_RUNTIME_STATUSES: frozenset[str] = frozenset(
        {"alive", "idle", "exited"}
    )
    # How long to wait for a resume-spawn's process to confirm liveness
    # before declaring the resume failed. Short by design: claude exits
    # quickly when the session id is unknown or its cache has expired.
    _RESUME_VERIFY_TIMEOUT_SEC: float = 0.5

    def _spawn_from_config_entry(self, entry: Dict[str, Any]) -> AgentHandle:
        """Resolve a recoverable config entry to a spawn() call.

        Required keys: agent_id, zone_id, task, work_dir. Optional keys:
        model, claude_session_id, runtime_status. Raises ValueError on a
        missing required key; the caller translates that to "failed".

        Resume branch: when the entry carries a claude_session_id and a
        resumable runtime_status, spawn with resume_claude_session_id so
        claude re-attaches via --resume instead of minting a new UUID.

        Fallback: if the resume-spawn's process exits non-zero before
        the verify timeout, emit a problem report, drop the failed
        handle, and retry as a fresh spawn. The agent's task then
        replays on the next continuation.
        """
        agent_id = entry["agent_id"]
        zone_id = entry["zone_id"]
        task = entry.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError(
                f"respawn entry for {agent_id!r} missing required 'task'"
            )
        work_dir_raw = entry.get("work_dir")
        if not work_dir_raw:
            raise ValueError(
                f"respawn entry for {agent_id!r} missing 'work_dir'"
            )
        work_dir = Path(work_dir_raw)
        model = entry.get("model")
        api_key = entry.get("api_key")
        use_bare = bool(entry.get("use_bare", False))

        # Decide whether this entry is a fresh spawn or a resume.
        prior_session_id = entry.get("claude_session_id")
        runtime_status = entry.get("runtime_status")
        is_resume = bool(
            prior_session_id
            and isinstance(prior_session_id, str)
            and isinstance(runtime_status, str)
            and runtime_status in self._RESUMABLE_RUNTIME_STATUSES
        )
        # Restore last_seen_turn_id from the config entry so the handoff
        # walker knows where to start after a notebook reopen.
        last_seen_turn_id: Optional[str] = entry.get("last_seen_turn_id")

        if not is_resume:
            handle = self.spawn(
                zone_id=zone_id, agent_id=agent_id, task=task,
                work_dir=work_dir, api_key=api_key,
                model=model, use_bare=use_bare,
            )
            if last_seen_turn_id:
                handle.last_seen_turn_id = last_seen_turn_id
            return handle

        handle = self.spawn(
            zone_id=zone_id, agent_id=agent_id, task=task,
            work_dir=work_dir, api_key=api_key,
            model=model, use_bare=use_bare,
            resume_claude_session_id=prior_session_id,
        )
        if self._resume_failed(handle):
            self._record_resume_failure_k24(
                agent_id=agent_id, zone_id=zone_id,
                attempted_session_id=prior_session_id,
                exit_code=handle.popen.poll(),
            )
            # Drop the failed handle so the fresh spawn does not hit
            # the live-agent idempotency check in spawn.
            with self._lock:
                if self._agents.get(agent_id) is handle:
                    del self._agents[agent_id]
            handle = self.spawn(
                zone_id=zone_id, agent_id=agent_id, task=task,
                work_dir=work_dir, api_key=api_key,
                model=model, use_bare=use_bare,
            )
        if last_seen_turn_id:
            handle.last_seen_turn_id = last_seen_turn_id
        return handle

    def _resume_failed(self, handle: AgentHandle) -> bool:
        """Return True if the just-spawned resume process has already
        exited with a non-zero status.

        Polls the process for at most the verify timeout at a fine
        granularity. A still-alive process (poll() is None) and a clean
        exit (poll() == 0) both count as success — the fallback path
        triggers only on an observable non-zero termination.
        """
        deadline = time.monotonic() + self._RESUME_VERIFY_TIMEOUT_SEC
        granularity = 0.05
        while time.monotonic() < deadline:
            rc = handle.popen.poll()
            if rc is None:
                time.sleep(granularity)
                continue
            return rc != 0
        return False

    def _record_resume_failure_k24(
        self,
        *,
        agent_id: str,
        zone_id: str,
        attempted_session_id: str,
        exit_code: Optional[int],
    ) -> None:
        """Emit the resume-failed marker and a synthetic problem report.

        --resume <session_id> failed, most likely because the local
        claude cache expired the session. The kernel then mints a new
        session and retries as a fresh spawn; the operator surface sees
        a problem run flagging the lossy transition.
        """
        from llm_kernel._internal import _diagnostics
        _diagnostics.mark(
            "supervisor_resume_failed_k24",
            agent_id=agent_id, zone_id=zone_id,
            attempted_session_id=attempted_session_id,
            exit_code=exit_code,
        )
        self._record_synthetic_problem(
            agent_id, zone_id,
            (
                f"claude --resume <{attempted_session_id}> failed "
                f"(exit_code={exit_code}); falling back to fresh spawn"
            ),
            "agent.resume_failed.k24",
        )

    # -- Synthetic-run helpers ---------------------------------------

    def _handle_violation(self, handle: AgentHandle, line: str) -> None:
        """Record a prose violation as an agent_emit span and flood-check.

        Free-form prose emitted despite the suppression prompt becomes
        an agent_emit span of kind "prose". On a flood (more than five a
        minute) the supervisor also emits a synthetic escalate run for
        operator attention.
        """
        self._emit_agent_emit(
            handle, emit_kind="prose", emit_content=line,
        )
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
            metadata={"tool.name": tool_name,
                      "jsonrpc_id": parsed.get("id")},
        )
        self._run_tracker.complete_run(rid, outputs={"acknowledged": True})

    def _record_stream_json(self, handle: AgentHandle, parsed: Dict[str, Any]) -> None:
        """Dispatch a Claude stream-json record.

        Record types and their handling:

        * assistant -- iterate the content array. tool_use blocks become
          tool spans. text blocks become agent_emit spans of kind
          "reasoning" (text preceding a tool call) or "prose" (text in
          an assistant message with no companion tool_use).
        * system -- agent_emit of kind system_message.
        * result -- agent_emit of kind result.
        * error -- agent_emit of kind error.

        The MCP tool-name prefix that Claude Code adds is stripped so the
        operator surface sees tools by their plain names. The supervisor
        cannot observe the MCP server completing the tool run, so it
        records the attempt here, not the result.
        """
        record_type = parsed.get("type")
        if record_type == "system":
            self._emit_agent_emit(
                handle, emit_kind="system_message",
                emit_content=_stringify_payload(parsed),
            )
            return
        if record_type == "result":
            self._emit_agent_emit(
                handle, emit_kind="result",
                emit_content=_stringify_payload(parsed),
            )
            return
        if record_type == "error":
            self._emit_agent_emit(
                handle, emit_kind="error",
                emit_content=_stringify_payload(parsed),
            )
            return
        if record_type != "assistant":
            return
        message = parsed.get("message") or {}
        content = message.get("content") or []
        if not isinstance(content, list):
            return
        prefix = "mcp__llmkernel-operator-bridge__"
        # Pre-scan to decide whether text blocks are reasoning (paired
        # with a tool_use later in the array) or stand-alone prose.
        has_tool_use = any(
            isinstance(b, dict) and b.get("type") == "tool_use"
            for b in content
        )
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_value = str(block.get("text", ""))
                if not text_value:
                    continue
                # Reasoning if this assistant message also contains a
                # tool_use block (the text precedes the tool call);
                # prose otherwise (a text-only assistant message, in
                # spite of the suppression prompt).
                emit_kind = "reasoning" if has_tool_use else "prose"
                self._emit_agent_emit(
                    handle, emit_kind=emit_kind, emit_content=text_value,
                )
                if emit_kind == "prose":
                    # Track flood threshold for prose-only assistant
                    # messages too, not just unparseable lines.
                    self._track_prose_flood(handle)
                continue
            if block_type != "tool_use":
                continue
            raw_name = str(block.get("name", "<unknown>"))
            tool_name = raw_name[len(prefix):] if raw_name.startswith(prefix) else raw_name
            rid = self._run_tracker.start_run(
                name=tool_name, run_type="tool",
                inputs=block.get("input") or {},
                tags=[f"agent:{handle.agent_id}", f"zone:{handle.zone_id}",
                      f"tool:{tool_name}", "via:stream-json"],
                metadata={"tool.name": tool_name,
                          "tool_use_id": block.get("id"),
                          "raw_name": raw_name},
            )
            self._run_tracker.complete_run(rid, outputs={"observed_via": "stream-json"})

    def _record_synthetic_problem(
        self, agent_id: str, zone_id: str, description: str, log_signature: str,
    ) -> None:
        """Open and close a synthetic report_problem run."""
        rid = self._run_tracker.start_run(
            name="report_problem", run_type="tool",
            inputs={"severity": "error", "description": description},
            tags=[f"agent:{agent_id}", f"zone:{zone_id}",
                  "tool:report_problem", "synthetic:supervisor"],
            metadata={"tool.name": "report_problem",
                      "log_signature": log_signature},
        )
        self._run_tracker.complete_run(
            rid, outputs={"acknowledged": True}, status="error",
        )

    def _record_synthetic_escalate(
        self, agent_id: str, zone_id: str, reason: str, severity: str,
    ) -> None:
        """Open and close a synthetic escalate run (prose flood)."""
        rid = self._run_tracker.start_run(
            name="escalate", run_type="tool",
            inputs={"reason": reason, "severity": severity},
            tags=[f"agent:{agent_id}", f"zone:{zone_id}",
                  "tool:escalate", "synthetic:dr0010_flood"],
            metadata={"tool.name": "escalate",
                      "log_signature": "dr0010.flood"},
        )
        self._run_tracker.complete_run(rid, outputs={"acknowledged": True})

    @staticmethod
    def _try_parse_json(line: str) -> Optional[Dict[str, Any]]:
        """Return the parsed JSON object or None for non-JSON lines."""
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    @staticmethod
    def _try_parse_json_with_error(
        line: str,
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Like _try_parse_json but also surfaces the decode error.

        Returns (parsed_obj_or_None, short_error_string_or_None). The
        short error string captures the decoder's message plus position
        so the operator can locate the failing byte.
        """
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            return None, f"{exc.msg} at position {exc.pos}"
        if not isinstance(obj, dict):
            return None, "top-level JSON is not an object"
        return obj, None

    def _scan_and_rewrite_emit_content(
        self, handle: AgentHandle, content: str, *, source: str,
    ) -> str:
        """Walk ``content`` line-by-line, flag contamination, escape on emission ban.

        Returns the content with any hash-mode-banned lines escaped
        (``@`` → ``\\@``) per :func:`magic_hash.escape_leading_at`.
        Plain magic lines are flagged but **not** rewritten — the
        emission ban is a hash-mode property; plain magics are
        operator-typed text that the renderer surfaces as warning.
        """
        if not isinstance(content, str) or not content:
            return content
        # Fast path: skip the splitlines work when there's no @ anywhere
        # in the content — covers most system_message / result spans.
        if "@" not in content:
            return content
        from llm_kernel.text.magic_hash import escape_leading_at

        out_lines: List[str] = []
        any_rewritten = False
        for line in content.splitlines():
            verdict = self._scan_for_magic_contamination(
                handle, line, source=source,
            )
            if verdict == "ESCAPE_REQUIRED":
                out_lines.append(escape_leading_at(line))
                any_rewritten = True
            else:
                out_lines.append(line)
        if not any_rewritten:
            return content
        # Preserve the trailing newline if the original had one.
        suffix = "\n" if content.endswith("\n") else ""
        return "\n".join(out_lines) + suffix

    def set_metadata_writer(self, writer: Any) -> None:
        """Wire a metadata writer for contamination flagging.

        The always-on contamination detector marks any cell as
        contaminated when its bound agent emits a magic-shaped line. The
        writer is supplied after construction. Wiring is optional: when
        None, the detector still runs and logs, so the audit trail
        survives.
        """
        self._metadata_writer = writer

    def _scan_for_magic_contamination(
        self, handle: AgentHandle, line: str, *, source: str,
    ) -> Optional[str]:
        """Scan one agent-emitted line for cell-magic injection patterns.

        Two layers:

        * Always-on plain detection: a leading @ or @@ followed by a
          known magic name. Flags any cell bound to this agent as
          contaminated and logs it with layer "plain".
        * Hash-mode emission ban: when hash mode is enabled and the line
          matches the hashed-magic shape, flag the cell, log it with
          layer "hashed_emission_ban", and signal the caller that the
          line must be escaped before write (returns "ESCAPE_REQUIRED").

        Hash mode is read from the writer's config; when the writer is
        unwired or the setting is absent, the hash-mode branch is
        dormant.

        Returns:
            "ESCAPE_REQUIRED" when the line matches the hashed-magic
            shape under hash mode (the caller must escape the leading @
            before pushing to outputs); None otherwise.
        """
        # Lazy import: this module otherwise has no dependency on the
        # magic registry.
        from llm_kernel.text.magic_hash import (
            HASHED_MAGIC_LINE,
            PLAIN_MAGIC_LINE,
            looks_like_hashed_magic,
            looks_like_plain_magic,
        )
        from llm_kernel.text.magic_registry import RESERVED_NAMES, is_generator

        if not isinstance(line, str) or not line:
            return None

        # Extract the magic name regardless of shape so we can tag
        # generator-magic emissions specifically. Generator names are a
        # strict subset of the reserved names.
        emitted_name: Optional[str] = None
        plain_match = PLAIN_MAGIC_LINE.match(line) if line else None
        hashed_match = HASHED_MAGIC_LINE.match(line) if line else None
        if hashed_match is not None:
            emitted_name = hashed_match.group(2)
        elif plain_match is not None:
            emitted_name = plain_match.group(1)
        emitted_is_generator = bool(
            isinstance(emitted_name, str) and is_generator(emitted_name)
        )

        # Layer 1: plain magic detection (always on).
        if looks_like_plain_magic(line, RESERVED_NAMES):
            self._flag_contaminated(
                handle.agent_id, line=line, source=source, layer="plain",
            )
            if emitted_is_generator:
                # Log generator names specifically so analytics can
                # split generator-class injection attempts from generic
                # plain-magic ones. This marker is log-level only; the
                # contamination flag and Layer-2 escape are the actual
                # defenses.
                self._emit_drift_marker(
                    handle.agent_id,
                    code=K3H_AGENT_EMITTED_GENERATOR_MAGIC_BLOCKED,
                    reason=(
                        f"agent emitted generator magic "
                        f"@@{emitted_name}"
                    ),
                    line=line,
                )
            # When the source agent holds an active magic-emit privilege
            # for the detected name, emit an informational marker. The
            # line is still sanitized; this just offers the operator a
            # promotion chip.
            if self._privileged_for_magic(
                handle, emitted_name,
            ):
                self._emit_drift_marker(
                    handle.agent_id,
                    code=K3L_PRIVILEGED_AGENT_STREAM_MAGIC,
                    reason=(
                        f"privileged agent stream-emitted "
                        f"@@{emitted_name}; promotion chip available"
                    ),
                    line=line,
                )

        # Layer 2: hashed magic emission ban (hash-mode-only).
        if self._magic_hash_enabled() and looks_like_hashed_magic(line):
            self._flag_contaminated(
                handle.agent_id, line=line, source=source,
                layer="hashed_emission_ban",
            )
            # When the recovered name is a generator, tag it in addition
            # to the emission-ban marker — analytics index on both.
            if emitted_is_generator:
                self._emit_drift_marker(
                    handle.agent_id,
                    code=K3H_AGENT_EMITTED_GENERATOR_MAGIC_BLOCKED,
                    reason=(
                        f"agent emitted hashed generator magic "
                        f"{emitted_name}"
                    ),
                    line=line,
                )
            self._emit_drift_marker(
                handle.agent_id,
                code=K36_HASHED_MAGIC_EMISSION_BLOCKED,
                reason="hashed-magic pattern in agent output",
                line=line,
            )
            return "ESCAPE_REQUIRED"
        return None

    def _privileged_for_magic(
        self, handle: "AgentHandle", magic_name: Optional[str],
    ) -> bool:
        """Return True if the agent holds an active grant for magic_name.

        Consults the writer's has_magic_emit_privilege if available;
        returns False when the writer isn't wired or the method is
        absent.
        """
        if not isinstance(magic_name, str) or not magic_name:
            return False
        writer = self._metadata_writer
        if writer is None:
            return False
        checker = getattr(writer, "has_magic_emit_privilege", None)
        if not callable(checker):
            return False
        try:
            return bool(checker(
                agent_id=handle.agent_id,
                zone_id=handle.zone_id,
                magic_name=magic_name,
            ))
        except Exception:  # pragma: no cover - defensive
            return False

    def _magic_hash_enabled(self) -> bool:
        """Best-effort lookup of the magic_hash_enabled config setting.

        Defaults to False when the writer is unwired or the config field
        is absent.
        """
        writer = self._metadata_writer
        if writer is None:
            return False
        try:
            getter = getattr(writer, "get_config_setting", None)
            if callable(getter):
                value = getter("magic_hash_enabled")
                return bool(value)
            # Fallback: probe a _config dict if the writer exposes one.
            config = getattr(writer, "_config", None)
            if isinstance(config, dict):
                return bool(config.get("magic_hash_enabled", False))
        except Exception:  # pragma: no cover - defensive
            return False
        return False

    def _flag_contaminated(
        self,
        agent_id: str,
        *,
        line: str,
        source: str,
        layer: str,
    ) -> None:
        """Mark every cell bound to agent_id as contaminated.

        Sets a contaminated flag and appends a record to an append-only
        contamination log. The line is truncated to 256 chars before
        storage so a flood can't grow the notebook without bound.
        """
        from llm_kernel._internal import _diagnostics

        truncated = line[:256] if isinstance(line, str) else ""
        _diagnostics.mark(
            "supervisor_magic_contamination_detected",
            agent_id=agent_id, source=source, layer=layer,
            line_prefix=truncated[:64],
        )
        writer = self._metadata_writer
        if writer is None:
            return
        # Duck-type the writer's flag method; when it's missing the
        # diagnostic mark above is the audit trail.
        flagger = getattr(writer, "flag_cells_contaminated_by_agent", None)
        if not callable(flagger):
            return
        try:
            flagger(
                agent_id=agent_id, line=truncated,
                source=source, layer=layer,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "supervisor: contamination flagger raised "
                "(agent=%s, layer=%s)", agent_id, layer,
            )

    def _emit_drift_marker(
        self, agent_id: str, *, code: str, reason: str, line: str,
    ) -> None:
        """Emit a drift-log marker.

        Downstream consumers index on the code field.
        """
        from llm_kernel._internal import _diagnostics

        _diagnostics.mark(
            "supervisor_drift_marker",
            agent_id=agent_id, code=code, reason=reason,
            line_prefix=(line or "")[:64],
        )

    def _emit_agent_emit(
        self,
        handle: AgentHandle,
        *,
        emit_kind: str,
        emit_content: str,
        parser_diagnostic: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> str:
        """Emit one agent_emit span.

        Records the run type, agent id, emit kind (the category the
        renderer dispatches on), and the verbatim agent output. Content
        above the blob threshold is blob-extracted on persistence; the
        wire carries the full content for in-cell rendering.

        Every emitted line is scanned for cell-magic injection patterns.
        Plain magic lines flag bound cells as contaminated; hashed-magic
        lines (when hash mode is on) trigger the emission-ban escape so
        the leading @ is escaped before the span lands in cell outputs.
        """
        # Contamination scan + emission-ban escape.
        # The scan operates per line on emit_content (which may be a
        # multi-line string for stream-json result/error records); we
        # walk lines, flag on detection, and rebuild the content with
        # escaped leading @ characters where the emission ban triggered.
        scanned_content = self._scan_and_rewrite_emit_content(
            handle, emit_content, source=f"agent_emit:{emit_kind}",
        )
        emit_content = scanned_content
        metadata: Dict[str, Any] = {
            "emit_kind": emit_kind,
            "emit_content": emit_content,
        }
        if parser_diagnostic:
            metadata["parser_diagnostic"] = parser_diagnostic
        rid = self._run_tracker.start_run(
            name=f"agent_emit:{emit_kind}", run_type="agent_emit",
            inputs={},
            parent_run_id=parent_span_id,
            tags=[f"agent:{handle.agent_id}", f"zone:{handle.zone_id}",
                  f"agent_emit:{emit_kind}"],
            metadata=metadata,
        )
        # Open and immediately close: an agent_emit captures point-in-
        # time output, not a duration. The renderer keys on the closed
        # span; receivers still see a run.start then a run.complete.
        self._run_tracker.complete_run(rid, outputs={})
        return rid

    def _track_prose_flood(self, handle: AgentHandle) -> None:
        """Track the prose-flood threshold (more than 5 in 60s escalates).

        Used by the assistant-content text path so flood detection also
        covers stream-json prose, not just unparseable lines.
        """
        now = time.monotonic()
        handle._violation_times.append(now)
        cutoff = now - _VIOLATION_FLOOD_WINDOW_SEC
        recent = [t for t in handle._violation_times if t >= cutoff]
        if len(recent) > _VIOLATION_FLOOD_COUNT:
            self._record_synthetic_escalate(
                handle.agent_id, handle.zone_id, "DR-0010 flood", "medium",
            )
            handle._violation_times.clear()

    @staticmethod
    def _is_mcp_jsonrpc(obj: Dict[str, Any]) -> bool:
        """Return True for an MCP JSON-RPC frame."""
        return obj.get("jsonrpc") == "2.0" and (
            "method" in obj or "result" in obj or "error" in obj
        )

    @staticmethod
    def _is_stream_json(obj: Dict[str, Any]) -> bool:
        """Return True for a Claude stream-json record (has type or event)."""
        return "type" in obj or "event" in obj


def _stringify_payload(parsed: Dict[str, Any]) -> str:
    """Serialize a stream-json record as a single-line JSON string.

    Used as the content of agent_emit spans of kind system_message,
    result, or error. The full record is preserved verbatim so the
    operator surface can show the raw stream-json; the metadata writer
    blob-extracts it on persistence if it exceeds the threshold.
    """
    try:
        return json.dumps(parsed, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(parsed)


def _append_pretty(path: Path, parsed: Dict[str, Any]) -> None:
    """Append a pretty-printed JSON event with a one-line header.

    The header summarizes the event's type and subtype plus the most
    informative inner field (model name for assistant messages, error
    code for results, mcp_servers status for system init). The payload
    follows as 2-space-indented JSON. Used only when
    LLMKERNEL_DEBUG_STDOUT=1.
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

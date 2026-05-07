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

from . import context_packer
from ._provisioning import (
    EXPECTED_SYSTEM_PROMPT_TEMPLATE_VERSION,
    PreSpawnValidationError, build_argv, build_env,
    extract_template_version, render_mcp_config, render_system_prompt,
    _split_semver, validate_pre_spawn,
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
#: Default stdout-silence threshold (RFC-002 §"Failure modes" — Hang).
#: Configurable per-supervisor via the ``agent_silence_threshold_seconds``
#: ctor kwarg (kernel config maps onto it via ``config.kernel``).
DEFAULT_AGENT_SILENCE_THRESHOLD_SEC: float = 120.0
#: Granularity at which the silence watchdog wakes to check the timer.
_SILENCE_WATCHDOG_GRANULARITY_SEC: float = 5.0
#: Per-agent restart-window per RFC-002 §"Process lifecycle" 5
#: ("≤3 attempts in 5 minutes"). Sliding window of monotonic
#: timestamps; on overflow the supervisor refuses further restarts.
_RESTART_WINDOW_SEC: float = 300.0
_RESTART_WINDOW_MAX: int = 3

#: PLAN-S5.0.1 §3.9 K-class identifiers wired in slice 5.0.1a.
#: The remaining K3C..K3G land with the precondition-gates +
#: acceptance-flag slice (5.0.1c). K35 itself fires from the
#: hash-aware parser in 5.0.1b — the constant ships here so the
#: drift-marker emit path can reference the canonical name.
K35_PLAIN_MAGIC_IN_HASH_MODE: str = "K35"
K36_HASHED_MAGIC_EMISSION_BLOCKED: str = "K36"
K3H_AGENT_EMITTED_GENERATOR_MAGIC_BLOCKED: str = "K3H"

AgentState = str  # starting | running | restarting | failed | terminated


class _HandoffCycleError(Exception):
    """Raised by ``_missed_turns`` when the DAG walker detects a cycle.

    PLAN-S4 K26 ``cycle_detected``: the turn chain exceeded the max-depth
    guard or a ``parent_id`` loop was detected.  The supervisor converts
    this into a K26 RuntimeError surfaced as a ``report_problem`` span.
    """


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
    silence_watchdog_thread: Optional[threading.Thread] = None
    state: AgentState = "starting"
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _violation_times: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    _restart_attempts: int = 0
    #: Sliding-window timestamps of restart attempts per RFC-002 §6.
    #: Pruned to ``_RESTART_WINDOW_SEC``; >= ``_RESTART_WINDOW_MAX`` =>
    #: refuse further restarts.
    _restart_history: Deque[float] = field(default_factory=deque)
    #: Monotonic timestamp of the most recent stdout byte. Updated by
    #: every reader-thread iteration that observes a non-empty line;
    #: read by :meth:`AgentSupervisor._silence_watchdog`. Defaults to
    #: spawn time so a freshly-spawned agent has the full silence
    #: budget to produce its first byte.
    _last_stdout_ts: float = 0.0
    #: Set whenever the silence watchdog has SIGTERMed the process so
    #: the regular watchdog (which observes the resulting non-zero
    #: exit) does not double-log; tag is plumbed into the restart
    #: bookkeeping.
    _hang_terminated: bool = False
    #: Claude session id (UUID) per BSP-002 §5. Assigned at spawn time
    #: via ``--session-id=<uuid>``. Survives across spawn/idle/respawn
    #: of the same agent_id so a future ``--resume <claude_session_id>``
    #: can thread the conversation back to where the prior process left
    #: off. Empty string when unset (legacy callers, fixtures).
    claude_session_id: str = ""
    #: PLAN-S4: the most recent turn-id this agent's claude session has
    #: been fed.  ``None`` for fresh spawns (agent has seen no turns
    #: yet).  After a successful ``send_user_turn`` this advances to
    #: the notebook head turn-id so subsequent calls can compute the
    #: missed-turn delta.  Mirrored into
    #: ``metadata.rts.zone.agents.<id>.session.last_seen_turn_id`` via
    #: the ``update_agent_session`` intent (BSP-002 §4.6 / agent.md).
    last_seen_turn_id: Optional[str] = None

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
        agent_silence_threshold_seconds: float = DEFAULT_AGENT_SILENCE_THRESHOLD_SEC,
        silence_watchdog_granularity_seconds: float = _SILENCE_WATCHDOG_GRANULARITY_SEC,
    ) -> None:
        """Bind the supervisor to its B2/B3 collaborators.

        ``litellm_endpoint_url`` is ``ANTHROPIC_BASE_URL`` for spawned
        agents and the URL the pre-spawn health check probes.
        ``kernel_python`` is the absolute path to the kernel's Python;
        the per-spawn MCP config's ``command`` field uses this.

        ``agent_silence_threshold_seconds`` (RFC-002 §"Failure modes"
        Hang row): when the agent emits no stdout for this many
        seconds, the silence watchdog SIGTERMs the process and the
        regular crash-restart machinery picks up the resulting exit.
        Default 120s; tests pass a smaller value via the ctor.
        ``silence_watchdog_granularity_seconds`` controls how often
        the watchdog wakes to check the silence timer (default 5s);
        tests reduce it for fast iteration.
        """
        self._run_tracker = run_tracker
        self._dispatcher = dispatcher
        self._litellm_endpoint_url: str = litellm_endpoint_url
        self._kernel_python: str = kernel_python
        self._silence_threshold_sec: float = agent_silence_threshold_seconds
        self._silence_granularity_sec: float = silence_watchdog_granularity_seconds
        # Per Engineering Guide §11.7: this lock is acquired on the
        # data path (spawn / restart bookkeeping) AND we may log inside
        # the critical section; downstream loggers route through
        # OtlpDataPlaneHandler -> SocketWriter (now an RLock) so the
        # supervisor's own lock must also be reentrant.
        self._lock: threading.RLock = threading.RLock()
        self._agents: Dict[str, AgentHandle] = {}
        # PLAN-S5.0.1 §3.2 — contamination detector hook. The kernel
        # wires its MetadataWriter in via ``set_metadata_writer`` so
        # the supervisor can flag receiving cells when an agent emit
        # carries a magic-shaped line. Optional: when None the
        # detector still runs (logging via ``_diagnostics.mark``)
        # so the audit trail is preserved without a wired writer.
        self._metadata_writer: Optional[Any] = None
        # PLAN-S4.1: in-memory ``_turns`` cache REMOVED.  The turn graph
        # is now read from ``metadata.rts.zone.agents.<id>.turns[]`` via
        # the ``MetadataWriter``.  Tests and callers seed turns with
        # ``submit_intent({intent_kind: "append_turn", ...})``.

    # PLAN-S4.1: ``record_turn`` REMOVED.  Callers submit ``append_turn``
    # intents to the writer; ``_missed_turns`` reads the persisted
    # ``metadata.rts.zone.agents.<id>.turns[]`` arrays directly.

    def _read_persisted_turns(self) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
        """Return ``(turn_index, notebook_head_id)`` from the writer snapshot.

        ``turn_index`` is keyed on turn-id and carries each persisted
        turn dict (with normalized ``content`` field aliasing ``body``
        for back-compat with the prior in-memory shape).

        ``notebook_head_id`` resolution (PLAN-S4.1 §3.C.3):

        1. If any agent has a ``session.head_turn_id`` set (e.g., by
           ``move_agent_head`` after revert), pick the max across
           agents by (created_at, id) of the corresponding turn.
        2. Otherwise, fall back to the most-recently-``created_at`` turn
           across all agents (lex tiebreak on id).

        Returns ``({}, None)`` when no writer is wired or no turns exist.
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
                # Normalize: callers historically used ``content``; the
                # persisted shape uses ``body``.  Mirror ``body`` ->
                # ``content`` so the _missed_turns / synthesize path
                # keeps working without per-call adaptation.
                norm = dict(t)
                if "content" not in norm and "body" in norm:
                    norm["content"] = norm["body"]
                index[tid] = norm

        def _key_for(tid: str) -> Tuple[str, str]:
            t = index.get(tid)
            return (str(t.get("created_at") or "") if t else "", tid)

        head_id: Optional[str] = None
        # Prefer agent session heads when present (post-revert / post-fork).
        candidates = [h for h in session_heads if h in index]
        if candidates:
            head_id = max(candidates, key=_key_for)
        else:
            # Fall back to "leaf" turns (no other turn references them as
            # parent) — these are the chain tips.  Among leaves, pick the
            # most-recent created_at (lex tiebreak on id).  Falls through
            # to the global most-recent if no leaves found (cycle).
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
        """Return the current notebook-head turn-id (PLAN-S4.1 §3.C).

        Computed from ``metadata.rts.zone.agents.<*>.turns[]`` as the
        most-recently-``created_at`` turn (lexicographic tiebreak on
        ``id``).  Returns ``None`` when no turns are persisted.
        """
        _, head_id = self._read_persisted_turns()
        return head_id

    def spawn(
        self, zone_id: str, agent_id: str, task: str, work_dir: Path,
        api_key: Optional[str] = None, model: Optional[str] = None,
        use_bare: bool = False, set_base_url: Optional[bool] = None,
        resume_claude_session_id: Optional[str] = None,
    ) -> AgentHandle:
        """Spawn a Claude Code subprocess wired into paper-telephone topology.

        Implements RFC-002 §"Process lifecycle": pre-spawn validation,
        artifact rendering (POSIX 0o600), env build, ``Popen``, two
        reader threads, watchdog. Raises
        :class:`PreSpawnValidationError` on validation failure (also
        emits a synthetic ``report_problem``).

        BSP-002 §4.3 resume branch — when ``resume_claude_session_id`` is
        set, the supervisor re-attaches to that existing claude session
        instead of minting a new UUID. argv carries ``--resume <id>``
        rather than ``--session-id <uuid>``; the resulting AgentHandle's
        ``claude_session_id`` matches the resumed value so the operator-
        visible identity is unchanged. Used by ``respawn_from_config``
        when the snapshot's ``runtime_status`` is ``idle`` / ``exited``
        / ``alive`` (per ``decisions/no-rebind-popen``: alive-in-snapshot
        but process-gone is treated as idle).
        """
        from . import _diagnostics
        _diagnostics.mark("supervisor_spawn_entry", agent_id=agent_id, zone_id=zone_id)

        # BSP-005 S5.0 K32 — reject agent_ids that collide with the
        # cell-magic registry. PLAN-S5.0 §4 reserves the union of
        # CELL_MAGICS + LINE_MAGICS keys (plus the ``llmnb_*`` future-
        # reservation prefix) so an operator can't shadow a magic by
        # spawning ``pin`` / ``agent`` / ``break`` / etc.
        from .magic_registry import is_reserved_name, K32_RESERVED_MAGIC_NAME
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

        # BSP-002 Phase 1 idempotency: a /spawn for an agent_id whose
        # process is still alive returns the existing handle instead of
        # double-spawning. The conversation graph (BSP-002 §4.2) treats
        # successive cells against the same agent_id as continuations,
        # not fresh spawns; the wire layer doesn't yet distinguish, so
        # the kernel takes responsibility for the dedup. A future
        # AgentSupervisor.resume(agent_id, task) covers the dead-agent
        # case via ``--resume <claude_session_id>`` per BSP-002 §4.3.
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
        # BSP-002 §5: kernel-owned claude_session_id. Assigned at fresh
        # spawn time and passed via --session-id so the runtime conversation
        # is tagged with a stable id we control. Persists on AgentHandle
        # for a future --resume path.
        #
        # Resume branch (BSP-002 §4.3 / decisions/no-rebind-popen): when
        # ``resume_claude_session_id`` is provided, do NOT mint a new UUID
        # — re-use the stored id and emit ``--resume`` instead of
        # ``--session-id`` so claude re-attaches to the existing
        # conversation cache.
        if resume_claude_session_id:
            claude_session_id = resume_claude_session_id
        else:
            claude_session_id = str(uuid.uuid4())
        # Resolve to absolute paths up-front. The child process is launched
        # with ``cwd=work_dir``, so passing relative paths to Claude's CLI
        # would double-prefix the work_dir (cwd + relative arg = doubled).
        work_dir = Path(work_dir).resolve()
        spawn_dir = (work_dir / ".run" / agent_id).resolve()
        spawn_dir.mkdir(parents=True, exist_ok=True)
        _diagnostics.mark("supervisor_spawn_dirs_ready", agent_id=agent_id, spawn_dir=str(spawn_dir))
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
        _diagnostics.mark("supervisor_spawn_mcp_config_written", agent_id=agent_id)
        rendered_prompt = render_system_prompt(task)
        # RFC-002 §"Failure modes": refuse spawn on system-prompt
        # template major-version mismatch; warn-and-proceed on minor.
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
        # RFC-009 §4.2 — resolve claude binary through zone_control so
        # the precedence rules (env > PATH > pixi-env probe) are applied
        # consistently and the diagnostic ``zone_control.resolved`` marker
        # tells the operator which source supplied the value. Returns
        # None if no source resolves; we degrade to bare "claude" so
        # downstream Popen errors carry a useful ENOENT instead of None.
        from . import zone_control
        claude_bin = zone_control.locate_claude_bin() or "claude"
        _diagnostics.mark(
            "supervisor_spawn_claude_resolved",
            agent_id=agent_id, claude_bin=claude_bin,
        )
        # Mutually-exclusive flags per `_provisioning.build_argv`:
        # ``--session-id <uuid>`` for fresh spawns vs. ``--resume <id>``
        # for re-attach. The supervisor decides at spawn-time which
        # branch the entry belongs to.
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
            # BSP-002 §4.1 / §4.2: stdin is a JSON-line channel so
            # ``send_user_turn`` can write ``{"type":"user","message":...}``
            # frames to continue an existing conversation. The agent
            # process stays alive between turns; reader threads on
            # stdout pick up the agent's response spans. Tests stub
            # subprocess.Popen so the PIPE here is a MagicMock and the
            # supervisor never actually writes to a real pipe.
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
        """Return the live handle for ``agent_id``, or ``None`` if unknown."""
        with self._lock:
            return self._agents.get(agent_id)

    # BSP-002 §4.2 / atoms/operations/continue-turn.md — multi-turn
    # continuation. Writes one stream-json user turn to the live agent's
    # stdin; if the agent has gone idle/exited the supervisor resumes via
    # the S2 ``resume_claude_session_id`` path on ``spawn`` first.
    #
    # Returned status discriminator:
    #   "sent"              -- alive agent; one stdin write only.
    #   "resumed_then_sent" -- idle/exited agent resumed via --resume,
    #                          then stdin write.
    #   "spawned_fresh"     -- resume failed (K24 fallback ran inside
    #                          ``_spawn_from_config_entry`` semantics);
    #                          a fresh session was minted, the new
    #                          handle is alive, the stdin write went
    #                          through. The transcript replay BSP-002
    #                          §4.4 Case-B is left to a higher slice.
    def send_user_turn(
        self,
        agent_id: str,
        text: str,
        cell_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Continue an existing agent's conversation with one user turn.

        Per ``atoms/operations/continue-turn.md`` and
        ``atoms/contracts/agent-supervisor.md``: write a stream-json
        ``{"type":"user","message":{"role":"user","content":text}}``
        line to the agent's stdin. The agent's response streams back via
        the existing stdout reader thread (Family A spans).

        Resume semantics (S2 path): if the agent's ``runtime_status`` is
        idle/exited (i.e., the popen has reaped), the supervisor calls
        :meth:`spawn` with ``resume_claude_session_id`` set to the
        handle's preserved ``claude_session_id`` before writing stdin.
        BSP-002 §4.6 cross-agent context handoff is NOT implemented in
        this slice; that lands in S4.

        Args:
            agent_id: The agent to continue. Unknown ids raise
                :class:`KeyError`; the dispatcher translates that to
                **K23** (``agent_continue_unknown_agent``) per
                BSP-002 §7. (Note: the atom's K20 ``cell_directive_unknown_agent``
                covers the wire-side directive; the supervisor's
                contract uses K23 because the runtime miss is the
                supervisor's failure mode.)
            text: The operator's message body. Empty / whitespace-only
                input is rejected with :class:`ValueError`; the
                dispatcher translates that to BSP-003 K42
                (validator-rejected) per the intent envelope contract.
            cell_id: Optional originating cell URI. Threaded through
                for diagnostics and future cell→turn bookkeeping
                (BSP-005 S6).

        Returns:
            ``{"agent_id": ..., "status": "sent" | "resumed_then_sent"
            | "spawned_fresh", "cell_id": ...}``
        """
        from . import _diagnostics
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"send_user_turn: text must be a non-empty string "
                f"(agent_id={agent_id!r})"
            )
        with self._lock:
            handle = self._agents.get(agent_id)
        if handle is None:
            raise KeyError(agent_id)

        status: str = "sent"
        # Resume-if-needed: poll() returning a non-None code means the
        # claude process has reaped. The S2 plumbing on ``spawn``
        # accepts ``resume_claude_session_id`` so claude re-attaches
        # via ``--resume`` instead of minting a fresh session id.
        if handle.popen.poll() is not None:
            prior_session_id = handle.claude_session_id
            zone_id = handle.zone_id
            work_dir = handle.work_dir
            # Drop the dead handle BEFORE re-spawning so the live-agent
            # idempotency short-circuit in ``spawn`` does not return
            # the (now reaped) handle. The fresh-spawn fallback below
            # mirrors the K24 path in ``_spawn_from_config_entry``.
            with self._lock:
                if self._agents.get(agent_id) is handle:
                    del self._agents[agent_id]
            _diagnostics.mark(
                "send_user_turn_resume_attempt",
                agent_id=agent_id, prior_session_id=prior_session_id,
            )
            if prior_session_id:
                resumed = self.spawn(
                    zone_id=zone_id, agent_id=agent_id,
                    task=text, work_dir=work_dir,
                    resume_claude_session_id=prior_session_id,
                )
                if self._resume_failed(resumed):
                    self._record_resume_failure_k24(
                        agent_id=agent_id, zone_id=zone_id,
                        attempted_session_id=prior_session_id,
                        exit_code=resumed.popen.poll(),
                    )
                    with self._lock:
                        if self._agents.get(agent_id) is resumed:
                            del self._agents[agent_id]
                    handle = self.spawn(
                        zone_id=zone_id, agent_id=agent_id,
                        task=text, work_dir=work_dir,
                    )
                    status = "spawned_fresh"
                else:
                    handle = resumed
                    status = "resumed_then_sent"
            else:
                # No prior session id -> fresh spawn straight away.
                handle = self.spawn(
                    zone_id=zone_id, agent_id=agent_id,
                    task=text, work_dir=work_dir,
                )
                status = "spawned_fresh"

        # PLAN-S4: cross-agent context handoff — walk the turn DAG
        # between handle.last_seen_turn_id and the current notebook head,
        # synthesize prefix lines for missed sibling turns, and inject
        # them before the operator's message.
        head_turn_id = self._notebook_head_turn_id()
        try:
            missed = self._missed_turns(agent_id, head_turn_id)
            prefix_lines = self._synthesize_handoff_prefix(missed)
        except _HandoffCycleError as exc:
            from .wire.tools import K26_CROSS_AGENT_HANDOFF_FAILED
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

        # BSP-008 §9 — ContextPacker + RunFrame wiring (K-CTXR slice).
        # When the operator passes a ``cell_id``, pack the manifest, submit
        # ``record_context_manifest``, and submit a start ``record_run_frame``
        # (status=running). On synchronous send failure we submit a terminal
        # frame with status=failed; on successful stdin write we submit
        # a terminal frame with status=complete. A missing ``cell_id`` is
        # the BSP-008 §12 graceful-degradation path: the run still works,
        # just produces no Inspect-mode trail.
        #
        # FLAGGED: the spec describes the "after completion" terminal frame
        # as triggered when the agent finishes its reply. send_user_turn
        # returns immediately after stdin write — the agent's response
        # streams asynchronously through the stdout reader. For V1 we treat
        # a successful stdin dispatch as run-complete from the supervisor's
        # vantage and emit the terminal frame inline. A future slice with a
        # per-turn completion detector (e.g. observing the agent's
        # ``stop_reason`` envelope) can resubmit the terminal frame; the
        # writer is idempotent-on-run_id for same-cell updates.
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

        # Write the stream-json user turn. BSP-002 §4.1: claude reads
        # ``{"type":"user","message":{"role":"user","content":<text>}}``
        # from stdin when the process was launched with
        # ``--input-format=stream-json``. Drift flag: ``build_argv`` in
        # _provisioning.py does not yet emit that flag (atom drift
        # noted in S3 report); without it claude treats stdin as the
        # initial prompt only. The wire-side write is correct per the
        # atom; the argv-side completion is a separate amendment.
        operator_line = json.dumps({
            "type": "user",
            "message": {"role": "user", "content": text},
        }) + "\n"
        if handle.popen.stdin is None:
            # Defensive: spawn now opens stdin=PIPE; legacy fixtures
            # that pre-build a handle without stdin land here.
            raise RuntimeError(
                f"send_user_turn: agent {agent_id!r} has no writable "
                f"stdin (was the popen built with stdin=PIPE?)"
            )
        try:
            # Write prefix lines (handoff context) before operator message.
            for prefix_line in prefix_lines:
                handle.popen.stdin.write(prefix_line + "\n")
            handle.popen.stdin.write(operator_line)
            handle.popen.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            from .wire.tools import K26_CROSS_AGENT_HANDOFF_FAILED
            _diagnostics.mark(
                "send_user_turn_stdin_write_failed",
                agent_id=agent_id, error_type=type(exc).__name__,
                k_class=K26_CROSS_AGENT_HANDOFF_FAILED if prefix_lines else None,
                reason="stdin_write_failed" if prefix_lines else None,
            )
            # BSP-008 §9 — emit terminal RunFrame with status=failed so
            # Inspect mode records the synchronous failure.
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
        # PLAN-S4: advance last_seen_turn_id to the notebook head so the
        # next call computes the correct missed-turn delta.
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
        # BSP-008 §9 — emit terminal RunFrame with status=complete on
        # successful stdin dispatch. See FLAGGED note above the start
        # frame for the V1 semantics: "complete" here reflects the
        # supervisor's vantage (user-turn delivered to the agent), not
        # the agent's reply lifecycle. The writer is idempotent on
        # run_id for same-cell updates so a future per-turn completion
        # detector can re-emit without K102.
        if run_id is not None and cell_id is not None and manifest_id is not None:
            # Re-read the head turn id post-write — for V1 ContextPacker
            # there's no atomic turn-commit yet, so head_turn_after may
            # equal head_turn_before; the field is still recorded.
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
    # BSP-008 §9 — ContextPacker + RunFrame integration helpers
    # ------------------------------------------------------------------

    def _runframe_now_iso(self) -> str:
        """Return the current UTC timestamp in ISO 8601 with millisecond precision.

        Mirrors ``context_packer._utc_now_iso`` so the start/terminal
        RunFrame timestamps are byte-aligned with the manifest's
        ``generated_at`` stamp.
        """
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    def _build_pack_snapshot(self) -> Dict[str, Any]:
        """Build the minimal overlay snapshot ContextPacker.pack() needs.

        Per BSP-008 §3 the V1 walker reads ``cells`` (with per-cell
        ``pinned`` / ``section_id`` / ``sub_turns`` / ``scratch`` /
        ``excluded`` / ``obsolete`` flags) and ``ordering`` (document
        order). The K-MW writer carries the cells map at
        ``self._metadata_writer._cells`` and the layout-derived ordering
        is not yet wired through; for V1 we read what the writer's full
        snapshot exposes (``cells`` field) and let ContextPacker fall
        back to insertion order when ``ordering`` is absent.

        Returns an empty dict when the writer has no cells map (the
        V1 graceful-degradation path — pack() will raise K100 if
        cell_id is genuinely orphan; the caller catches and skips).
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
        """Pack the manifest, persist it, and submit the start RunFrame.

        Per BSP-008 §9 the AgentSupervisor wraps the ContextPacker call
        in a ``record_context_manifest`` intent and emits a start
        ``record_run_frame`` (status=running, ended_at=null) before
        dispatching the user turn.

        All failure modes are caught and logged so a packer / writer
        error never blocks the operator's run; the operator just sees a
        degraded Inspect-mode trail. Returns ``(run_id, manifest_id)``
        on success, ``(None, None)`` on degradation.
        """
        from . import _diagnostics
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
        """Submit the terminal RunFrame intent (status=complete|failed|interrupted).

        Per BSP-008 §8 the writer is idempotent on ``run_id`` for
        same-``cell_id`` resubmissions, so this method may be called
        more than once across the lifetime of a single run (e.g.,
        synchronous failure path + future stdout-based completion
        detector). All failure modes are best-effort: a writer outage
        produces a ``_diagnostics.mark`` but never blocks the caller.
        """
        from . import _diagnostics
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
                            # ``started_at`` is required; we don't carry
                            # it across the call so we re-stamp with the
                            # terminal time. The persisted record's
                            # ``started_at`` will reflect the terminal
                            # submission's stamp — V2 may carry the start
                            # time forward if precise duration matters.
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

    # ------------------------------------------------------------------
    # PLAN-S4: cross-agent context handoff helpers
    # ------------------------------------------------------------------

    #: Maximum DAG depth the missed-turn walker will traverse before
    #: raising K26 ``cycle_detected``.  A zone with more turns than this
    #: limit in a single chain is pathological; 1 000 is generous.
    _HANDOFF_MAX_DEPTH: int = 1_000

    def _missed_turns(
        self,
        agent_id: str,
        head_turn_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Walk the turn DAG from ``head_turn_id`` back to ``agent.last_seen_turn_id``.

        Returns the chain in chronological (root → head) order, filtered
        to turns NOT authored by ``agent_id``.  Returns an empty list when
        there are no turns, or when ``agent.last_seen_turn_id`` already
        equals ``head_turn_id``.

        Raises :class:`_HandoffCycleError` (K26 ``cycle_detected``) when
        the chain exceeds :attr:`_HANDOFF_MAX_DEPTH` steps.
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
                # Turn referenced but not in store; stop here (partial DAG).
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
        """Build stream-json prefix lines for ``turns``.

        One line per turn in the format::

            {"type":"user","message":{"role":"user","content":"<role> <agent_id> said: <body>"}}

        Body content passes through ``magic_hash.strip_hashes_from_text``
        before the JSON wrap so agents never observe ``@@<hash>:<name>``
        patterns.

        Returns a list of JSON strings (WITHOUT trailing newlines; the
        caller appends ``"\\n"`` when writing to stdin).
        """
        from .magic_hash import strip_hashes_from_text
        from .magic_registry import CELL_MAGICS, LINE_MAGICS
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

    # BSP-005 S9 — interrupt method paired with the X-EXT cell-toolbar
    # interrupt button (commit 5de3401 sends ``{action_type:
    # "agent_interrupt", agent_id}``). The dispatcher in mcp_server.py
    # routes that envelope here.
    def interrupt(self, agent_id: str) -> Dict[str, Any]:
        """Send SIGINT to ``agent_id``'s live process.

        Pairs with the X-EXT cell-toolbar interrupt button (commit
        5de3401): the extension ships an ``operator.action`` envelope
        with ``action_type: "agent_interrupt"`` whose dispatcher routes
        through here. The semantic distinction vs ``/stop`` (per
        ``atoms/operations/stop-agent.md``):

        * ``/stop`` is a clean shutdown -- SIGTERM, ``runtime_status:
          idle``, conversation resumable on the next turn.
        * ``interrupt`` is an in-flight cancellation -- SIGINT to the
          live process so claude aborts its current generation but the
          process stays alive for the next turn. No resume cycle.

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
        from . import _diagnostics
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
    # PLAN-S5b: revert operation
    # ------------------------------------------------------------------

    #: K-class for revert target not in agent ancestry (BSP-002 §7).
    K22_INVALID_REVERT_TARGET: str = "K22"

    def revert(self, agent_id: str, target_turn_id: str) -> None:
        """Move ``agent_id``'s HEAD backward to ``target_turn_id``.

        PLAN-S5b / `revert-agent.md`:

        1. Validate agent exists (K20 if not).
        2. Validate ``target_turn_id`` is in the agent's ancestry by
           walking ``parent_id`` from the current head through
           ``self._turns`` (K22 if not found).
        3. SIGTERM the live process if ``runtime_status == "alive"``
           (i.e. the process has not yet exited).  Watchdog handles
           cleanup; exit code 0 treated as expected.
        4. Submit ``move_agent_head`` intent to writer:
           ``agent.head_turn_id = target_turn_id``,
           ``agent.last_seen_turn_id = target_turn_id``.
           ``move_agent_head`` is in ``_PENDING_SLICE``; the writer
           raises ``ValueError`` for it, so we log a no-op diagnostic
           (same pattern as S4's ``update_agent_session`` best-effort
           path) — the in-memory state is still updated.
        5. Submit ``record_event`` intent with ``kind: "agent_ref_move"``,
           ``reason: "operator_revert"``.
        6. Mint a fresh ``claude_session_id`` for next continue; the old
           one stays attached to historical turns.

        Args:
            agent_id: The agent whose HEAD to move.
            target_turn_id: The turn to revert to; MUST be in the
                agent's ancestry (reachable via ``parent_id`` walk).

        Raises:
            RuntimeError: K20 if the agent is unknown; K22 if
                ``target_turn_id`` is not in the agent's ancestry.
        """
        from . import _diagnostics

        with self._lock:
            handle = self._agents.get(agent_id)
            if handle is None:
                raise RuntimeError(
                    f"K20: agent {agent_id!r} not found in supervisor. "
                    "Spawn or respawn the agent before calling revert."
                )

            # --- ancestry walk (PLAN-S4.1) -----------------------------
            # Read the persisted turn graph from
            # ``metadata.rts.zone.agents.<*>.turns[]`` and walk parent_id
            # from the notebook head (most-recent turn) back through the
            # union of all agents' chains looking for ``target_turn_id``.
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
            # PLAN-S4.1: notebook head is computed from
            # ``metadata.rts.zone.agents.<*>.turns[]`` on demand; the
            # ``move_agent_head`` writer intent submitted below records
            # the new head as the canonical post-revert position.

            # --- mint fresh claude_session_id for next continue --------
            handle.claude_session_id = str(uuid.uuid4())

        # --- writer intents (outside lock; PLAN-S4.1 active) -----------
        if self._metadata_writer is not None:
            # ``move_agent_head`` is now active per PLAN-S4.1 §3.A.  The
            # writer mutates ``zone.agents.<id>.session.{head_turn_id,
            # last_seen_turn_id}`` and bumps snapshot_version.
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

            # agent_ref_move event log — PLAN-S4.1 §3.B.  Submitted as
            # ``record_event`` with ``parameters.kind = 'agent_ref_move'``
            # (NOT ``intent_kind: 'agent_ref_move'`` — that K40s).
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
    # PLAN-S5c: stop operation
    # ------------------------------------------------------------------

    def stop(self, agent_id: str) -> None:
        """Clean SIGTERM of ``agent_id``'s runtime process.

        PLAN-S5c / `stop-agent.md`:

        1. Validate agent exists (K20 if not).
        2. If ``runtime_status != "alive"`` (i.e. the process has already
           exited), log a diagnostic and return — the operation is
           idempotent; the operator may stop an already-idle agent.
        3. SIGTERM the live process.  Watchdog observes exit; exit code 0
           is treated as expected.
        4. Set ``handle.runtime_status = "idle"``, ``handle.pid = None``.
        5. Submit ``update_agent_session`` intent with the new
           ``runtime_status`` (best-effort per S4 pattern; the writer
           handler is active for ``update_agent_session``).

        NOT done by stop (contrast with revert):
        - Do NOT mint a new ``claude_session_id`` — stop preserves the
          session so the next continue can ``--resume``.
        - Do NOT touch ``head_turn_id`` or ``last_seen_turn_id`` — stop
          does not move HEAD; revert does.

        Args:
            agent_id: The agent to stop.

        Raises:
            RuntimeError: K20 if the agent is unknown.
        """
        from . import _diagnostics

        with self._lock:
            handle = self._agents.get(agent_id)
            if handle is None:
                raise RuntimeError(
                    f"K20: agent {agent_id!r} not found in supervisor. "
                    "Spawn or respawn the agent before calling stop."
                )

            # --- idempotency guard: already idle / exited ---------------
            if handle.popen.poll() is not None:
                # Process has already exited; runtime_status is already
                # idle or exited — nothing to do.  Log as diagnostic so
                # the operator sees the no-op.
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
            # pid is represented on handle.popen, not a separate field, but
            # the agent schema exposes pid: null on idle.  We set the
            # canonical runtime_status flag on the handle so callers can
            # inspect it; the writer intent carries the authoritative state.

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
    # PLAN-S5a: branch / fork operation
    # ------------------------------------------------------------------

    #: K-class for invalid branch target (new_agent_id already exists).
    K21_INVALID_BRANCH_TARGET: str = "K21"

    def fork(
        self,
        source_agent_id: str,
        at_turn_id: Optional[str],
        new_agent_id: str,
    ) -> "AgentHandle":
        """Create a new agent ref branching from ``source_agent_id``.

        PLAN-S5a / `branch-agent.md`:

        1. Validate ``source_agent_id`` exists (K20 if not).
        2. Validate ``new_agent_id`` does NOT already exist (K21 if it does).
        3. **Case A** (``at_turn_id`` is ``None`` OR equals
           ``source.head_turn_id``):
           - New agent's ``head_turn_id = source.head_turn_id``.
           - New agent's ``last_seen_turn_id = source.head_turn_id``.
           - Fresh ``claude_session_id`` minted (source session NOT shared).
           - ``runtime_status = "idle"`` — no live process; spawn on first
             continue-turn via existing resume + replay mechanics.
        4. **Case B** (``at_turn_id`` is an ancestor):
           - Validate ``at_turn_id`` is in ``source.head_turn_id``'s ancestry
             by walking ``parent_id`` (K22 if not found).
           - New agent's ``head_turn_id = at_turn_id``.
           - New agent's ``last_seen_turn_id = at_turn_id``.
           - Fresh ``claude_session_id``; ``runtime_status = "idle"``.
           - Transcript replay (synthesizing a claude session at ``at_turn_id``)
             is NOT this slice's job — it happens lazily on first continue-turn
             via existing resume + replay mechanics.
        5. Insert the new AgentHandle into ``self._agents``.
        6. Submit ``fork_agent`` intent to writer (best-effort; logs diagnostic
           if the handler is ``_PENDING_SLICE``).
        7. Submit ``record_event`` intent with ``kind: "agent_ref_move"``,
           ``reason: "operator_branch"``.
        8. Return the new AgentHandle.

        Args:
            source_agent_id: The agent to branch from.
            at_turn_id: The turn to branch at.  ``None`` or equal to
                ``source.head_turn_id`` triggers Case A (branch at head).
                Any ancestor turn triggers Case B.
            new_agent_id: The ID for the new agent.  MUST be unique.

        Returns:
            The newly-created :class:`AgentHandle` (``runtime_status``
            is ``"idle"``; no live process yet).

        Raises:
            RuntimeError: K20 if ``source_agent_id`` is unknown; K21 if
                ``new_agent_id`` already exists; K22 if ``at_turn_id``
                is not in the source agent's ancestry.
        """
        from . import _diagnostics

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

            # --- resolve effective branch point (PLAN-S4.1) ------------
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
            # We construct a minimal handle using a sentinel Popen-like
            # object: the handle's popen.poll() returns non-None (already
            # "exited") so runtime_status is treated as idle by callers.
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

        # --- writer intents (outside lock; PLAN-S4.1 active) -----------
        if self._metadata_writer is not None:
            # fork_agent intent: persists the new agent ref into
            # ``zone.agents.<new_id>`` per PLAN-S4.1 §3.A.
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

            # agent_ref_move event — PLAN-S4.1 §3.B.
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
        *, claude_session_id: str = "",
    ) -> AgentHandle:
        """Allocate the handle + start reader/watchdog threads.

        ``claude_session_id`` is the BSP-002 §5 kernel-owned UUID passed
        to claude via ``--session-id``; persists on the handle so a future
        ``--resume`` can thread continuation across spawn cycles.
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
            # RFC-002 §"Failure modes" hang detection: every byte
            # observed on stdout (even an empty line) resets the
            # silence timer so the watchdog only fires on TRUE
            # silence, not parser-rejection silence.
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
                # RFC-002 §"Process lifecycle" 3: lines that fail both
                # parsers become ``agent_emit`` spans of kind
                # ``malformed_json`` with ``llmnb.parser_diagnostic``
                # carrying the short error string.
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
        """Append stderr to a per-agent log AND emit ``agent_emit`` spans per line.

        Per RFC-002 §"Process lifecycle" 4: stderr is captured line by
        line and emitted as ``agent_emit`` spans with
        ``llmnb.emit_kind: "stderr"``; the kernel additionally writes
        the raw stream to a per-agent log file (indexed by agent_id)
        for debugging.  Renderers MAY collapse stderr by default but
        the data MUST reach the operator surface -- silent drops are
        forbidden.
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
        """Monitor process exit; restart per RFC-002 §"Process lifecycle" 5.

        Up to three restarts in a 5-minute sliding window per agent;
        after exhaustion emits a synthetic ``report_problem`` and
        leaves ``state=failed``. Per-agent (not zone-wide) — see RFC
        ambiguity note in the implementation report.
        """
        exit_code = handle.popen.wait()
        if handle._stop_event.is_set() or (
            exit_code == 0 and not handle._hang_terminated
        ):
            handle.state = "terminated"
            return
        if handle._hang_terminated:
            # The silence watchdog has already logged + recorded the
            # hang event and SIGTERMed the process. Reset the flag so
            # the next iteration of crash-restart can run cleanly.
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
        # RFC-002 §6 sliding-window check: prune entries older than
        # _RESTART_WINDOW_SEC; if >= _RESTART_WINDOW_MAX, refuse.
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

        RFC-002 §"Failure modes" Hang row: when the agent emits no
        stdout for >threshold seconds AND the process is still running,
        the supervisor SIGTERMs and lets the regular crash-restart
        machinery in :meth:`_watchdog` pick up the resulting non-zero
        exit (with ``handle._hang_terminated`` set so the log line
        attributes the cause correctly).

        Loops at ``self._silence_granularity_sec`` granularity and
        exits when the popen has reaped (returncode is not None) or
        the stop-event is set. Daemon thread: never prevents process
        teardown.
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
            # Mark first so the regular watchdog sees the flag even if
            # SIGTERM races. RFC-002 mandates one ERROR-level log
            # event with the silence_seconds attribute.
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
        """Refuse spawn on system-prompt-template major-version mismatch.

        RFC-002 §"Failure modes" Template-version-mismatch row.
        Major mismatch -> :class:`PreSpawnValidationError` with
        ``provisioning.template.version_mismatch`` log signature.
        Minor mismatch -> log a warning and proceed. Patch differs
        silently. Missing marker -> warn (we just rendered the
        canonical template, so it is a developer bug not an attack).
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

    # -- Config-driven respawn (consumed by K-CM hydrate handler) ----

    def respawn_from_config(
        self, config_recoverable_agents: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Spawn (or rebind) every agent in ``config.recoverable.agents[]``.

        Round-1 G1 contract: K-CM's ``notebook.metadata mode:hydrate``
        handler calls this with the recoverable agent list (each entry
        carries ``agent_id``, ``zone_id``, ``tools_allowed`` and
        optionally ``volatile`` keys ``model``,
        ``system_prompt_template_id``, ``system_prompt_hash``, plus a
        ``task`` string). Returns ``{agent_id: status}`` where status
        is ``"spawned"`` | ``"failed"`` | ``"skipped"``.

        Idempotency: agents with an ``agent_id`` already running are
        skipped (no respawn). One bad entry MUST NOT block the rest;
        each spawn is wrapped in its own try-block.

        The recoverable schema (RFC-005 §"`metadata.rts.config.recoverable.agents`")
        does NOT carry the task string. V1 callers pass the task as a
        synthetic key inside each entry (``"task"``) until RFC-005 v2
        decides whether ``task`` is recoverable; missing ``task`` is
        an error for that entry only.
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
            except Exception:  # noqa: BLE001 — RFC-002 mandates per-agent
                logger.exception(
                    "respawn_from_config: spawn failed for %s", agent_id,
                )
                results[agent_id] = "failed"
        return results

    #: Statuses that signal a resumable claude session per
    #: ``decisions/no-rebind-popen``. ``alive`` is included because a
    #: snapshot taken before clean shutdown may carry the prior
    #: ``runtime_status`` even though the process has since gone away —
    #: per the decision atom that case is treated as ``idle``.
    _RESUMABLE_RUNTIME_STATUSES: frozenset[str] = frozenset(
        {"alive", "idle", "exited"}
    )
    #: How long the supervisor waits for a resume-spawn's popen to
    #: confirm liveness before declaring the resume failed (K24). Short
    #: by design — claude exits quickly when the session id is unknown
    #: or the cache has expired. Tests pass a smaller value via
    #: ``_resume_verify_timeout_sec``.
    _RESUME_VERIFY_TIMEOUT_SEC: float = 0.5

    def _spawn_from_config_entry(self, entry: Dict[str, Any]) -> AgentHandle:
        """Resolve a recoverable+volatile config entry to a spawn(...) call.

        Required keys: ``agent_id``, ``zone_id``, ``task``,
        ``work_dir``. Optional volatile keys (per RFC-005
        ``config.volatile.agents[]``): ``model``,
        ``claude_session_id``, ``runtime_status``. Raises ValueError on
        missing required key; the caller in
        :meth:`respawn_from_config` translates that to ``'failed'``.

        Resume branch (BSP-002 §4.3 / decisions/no-rebind-popen): when
        the entry carries ``claude_session_id`` AND ``runtime_status``
        is in :data:`_RESUMABLE_RUNTIME_STATUSES`, the supervisor calls
        :meth:`spawn` with ``resume_claude_session_id=...`` so claude
        re-attaches via ``--resume <id>`` instead of minting a new UUID.

        K24 fallback (BSP-002 §7): if the resume-spawn's claude process
        exits non-zero before the verify timeout elapses, the
        supervisor emits a K24 ``report_problem``, drops the failed
        handle, and retries as a FRESH spawn (new UUID). The agent's
        task will then replay via the BSP-002 §4.4 Case-B mechanism on
        the next continuation; for V1 the K24 marker + fresh spawn is
        the documented fallback path.
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
        # PLAN-S4: restore last_seen_turn_id from the config entry so the
        # handoff walker knows where to start after a notebook reopen.
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
            # the live-agent idempotency short-circuit in :meth:`spawn`.
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
        """Return True if the just-spawned resume process has already exited
        with a non-zero status.

        Polls ``popen.poll()`` for at most ``_RESUME_VERIFY_TIMEOUT_SEC``
        at a fine granularity. A still-alive popen (``poll() is None``)
        and a clean exit (``poll() == 0``) both count as success — the
        K24 path triggers only on observable non-zero termination.
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
        """Emit the K24 fallback marker + synthetic ``report_problem``.

        Per BSP-002 §7 K24: ``--resume <session_id>`` failed (the local
        claude cache likely expired the session per
        ``concepts/agent.md`` ``runtime_status: "exited"`` semantics).
        The kernel then mints a new session and retries as a fresh
        spawn; the operator surface sees a K24 problem run flagging
        the lossy transition.
        """
        from . import _diagnostics
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
        """DR-0010 prose violation: ``agent_emit`` (prose) + flood-check.

        Per RFC-002 §"Process lifecycle" 3 / RFC-005 §"`agent_emit`
        runs", free-form prose despite the suppression prompt becomes
        an ``agent_emit`` span with ``llmnb.emit_kind: "prose"``
        rather than a synthetic ``notify`` (which conflated agent
        output with kernel-issued tool calls).  On flood (>5/min),
        the supervisor additionally emits a synthetic ``escalate``
        run for operator attention.
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
        """Dispatch a Claude stream-json record per RFC-002 §"Process lifecycle" 3.

        Stream-json record types and their handling:

        * ``assistant`` -- iterate the content array.  ``tool_use``
          blocks become tool spans (existing behavior).  ``text``
          blocks become ``agent_emit`` spans with
          ``llmnb.emit_kind: "reasoning"`` (text preceding a tool
          call) or ``llmnb.emit_kind: "prose"`` (text in an
          assistant message with no companion tool_use).
        * ``system`` -- ``agent_emit`` with kind ``system_message``.
        * ``result`` -- ``agent_emit`` with kind ``result``.
        * ``error`` -- ``agent_emit`` with kind ``error``.

        The MCP-namespaced prefix ``mcp__llmkernel-operator-bridge__``
        (Claude Code's convention for MCP tools) is stripped from
        tool names so the operator surface sees them by their RFC-001
        names.  The supervisor cannot observe the MCP server's
        completion of the tool run -- V1 records the ATTEMPT here,
        not the result; RFC-002 v1.0.1 amendment captures this.
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
                # prose otherwise (text-only assistant message in spite
                # of the suppression prompt -- DR-0010 violation).
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
        """Open + close a synthetic ``report_problem`` run."""
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
        """Open + close a synthetic ``escalate`` run (DR-0010 flood)."""
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
        """Return the parsed JSON object or ``None`` for non-JSON lines."""
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    @staticmethod
    def _try_parse_json_with_error(
        line: str,
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Like :meth:`_try_parse_json` but also surfaces the decode error.

        Returns ``(parsed_obj_or_None, short_error_string_or_None)``.
        The short error string is suitable for
        ``llmnb.parser_diagnostic`` per RFC-005 §"`agent_emit` runs"
        on a malformed-json span; it captures the decoder's message
        plus position so the operator can locate the failing byte.
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
        # Fast path: avoid splitlines work when content is not even
        # multiline AND has no leading ``@`` anywhere — covers the
        # majority of agent stream-json system_message / result spans.
        if "@" not in content:
            return content
        from .magic_hash import escape_leading_at

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

        PLAN-S5.0.1 §3.2 — the always-on contamination detector flags
        ``cells[<id>].contaminated = True`` on any cell whose
        ``bound_agent_id`` matches the agent that just emitted a
        magic-shaped line. The writer is supplied post-construction
        (mirroring the dispatcher's ``set_metadata_writer`` plumbing).
        Wiring is optional: when ``None``, the detector still runs
        and emits a ``_diagnostics.mark`` so the audit trail survives.
        """
        self._metadata_writer = writer

    def _scan_for_magic_contamination(
        self, handle: AgentHandle, line: str, *, source: str,
    ) -> Optional[str]:
        """Scan one agent-emitted line for cell-magic injection patterns.

        PLAN-S5.0.1 §3.2 — two layers:

        * **Always-on plain detection**: ``^@@?<known_name>(\\s|:|$)``.
          Sets ``contaminated`` on any cell bound to ``handle.agent_id``;
          appends to ``contamination_log`` with ``layer="plain"``.
        * **Hash-mode emission ban**: when
          ``magic_hash_enabled`` is True AND the line matches the
          canonical hashed-magic shape, flag the cell, append a
          ``layer="hashed_emission_ban"`` log entry, AND signal the
          caller that the line MUST be escaped before write
          (``return "ESCAPE_REQUIRED"``).

        Hash-mode is read from the writer's
        ``metadata.rts.config.magic_hash_enabled`` setting. The schema
        addition lands in slice 5.0.1b; until then the lookup returns
        None and the hash-mode branch is dormant by design.

        Returns:
            ``"ESCAPE_REQUIRED"`` when the line matches the hashed-
            magic shape under hash mode (caller must escape leading
            ``@`` before push to outputs); ``None`` otherwise.
        """
        # Lazy import: this module otherwise has no dependency on the
        # registry (preserves the pre-S5.0.1 import topology).
        from .magic_hash import (
            HASHED_MAGIC_LINE,
            PLAIN_MAGIC_LINE,
            looks_like_hashed_magic,
            looks_like_plain_magic,
        )
        from .magic_registry import RESERVED_NAMES, is_generator

        if not isinstance(line, str) or not line:
            return None

        # PLAN-S5.0.2 §7 — extract the magic name regardless of shape
        # so we can upgrade the contamination tag to K3H when the agent
        # emitted a generator-magic call. Generator names are a strict
        # subset of RESERVED_NAMES so we only do this when a layer-1
        # match happens.
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
                # PLAN-S5.0.2 §7 — log K3H specifically for generator
                # names so analytics can split generator-class injection
                # attempts from generic plain-magic ones. K3H is log-
                # level (the contamination flag + Layer-2 escape are
                # the actual defenses).
                self._emit_drift_marker(
                    handle.agent_id,
                    code=K3H_AGENT_EMITTED_GENERATOR_MAGIC_BLOCKED,
                    reason=(
                        f"agent emitted generator magic "
                        f"@@{emitted_name}"
                    ),
                    line=line,
                )

        # Layer 2: hashed magic emission ban (hash-mode-only).
        if self._magic_hash_enabled() and looks_like_hashed_magic(line):
            self._flag_contaminated(
                handle.agent_id, line=line, source=source,
                layer="hashed_emission_ban",
            )
            # When the recovered name is a generator we tag K3H *in
            # addition to* K36 — analytics consumers index on both.
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

    def _magic_hash_enabled(self) -> bool:
        """Best-effort lookup of ``metadata.rts.config.magic_hash_enabled``.

        Defaults to ``False`` when the writer is unwired or the
        config field is absent. The schema field itself lands in
        slice 5.0.1b; this slice's reader is forward-compatible.
        """
        writer = self._metadata_writer
        if writer is None:
            return False
        try:
            getter = getattr(writer, "get_config_setting", None)
            if callable(getter):
                value = getter("magic_hash_enabled")
                return bool(value)
            # Fallback: probe a public ``_config`` dict if exposed.
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
        """Mark every cell bound to ``agent_id`` as contaminated.

        PLAN-S5.0.1 §3.6 schema — ``cells[<id>].contaminated`` and an
        append-only ``contamination_log`` of ``{detected_at, line,
        reason, layer}`` records. The line is truncated to a sane
        bound (256 chars) before storage so a flooded contamination
        path can't unbounded-grow the notebook.
        """
        from . import _diagnostics

        truncated = line[:256] if isinstance(line, str) else ""
        _diagnostics.mark(
            "supervisor_magic_contamination_detected",
            agent_id=agent_id, source=source, layer=layer,
            line_prefix=truncated[:64],
        )
        writer = self._metadata_writer
        if writer is None:
            return
        # The writer-side flag method is added in slice 5.0.1c; for
        # this foundation slice we duck-type the call. When the
        # method is missing the diagnostics-mark above is the audit
        # trail.
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
        """Emit a K3x drift-log marker via ``_diagnostics.mark``.

        Slice 5.0.1a wires K35/K36 only (the others — K3C..K3G —
        belong to the precondition-gates / acceptance-flag slices).
        The drift-detector path is fed by ``_diagnostics.mark``;
        downstream consumers index on the ``code`` field.
        """
        from . import _diagnostics

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
        """Emit one ``agent_emit`` span per RFC-005 §"`agent_emit` runs".

        Records ``llmnb.run_type: "agent_emit"``, ``llmnb.agent_id``,
        ``llmnb.emit_kind`` (the categorical attribute the renderer
        dispatches on), and ``llmnb.emit_content`` (the verbatim
        agent output).  Wire emission is verbatim per the RFC's
        guidance -- contents above the blob threshold are blob-
        extracted by ``metadata_writer.py`` on persistence; the wire
        carries the full content for in-cell rendering.

        PLAN-S5.0.1 §3.2 — every emitted line is scanned for
        cell-magic injection patterns. Plain ``@@<known>`` lines flag
        bound cells as contaminated; hashed-magic-shaped lines (when
        hash mode is on) trigger the emission-ban escape so the
        offending leading ``@`` is replaced with ``\\@`` before the
        span lands in cell outputs.
        """
        # PLAN-S5.0.1 §3.2 contamination scan + emission-ban escape.
        # The scan operates per-LINE on the emit_content (which may
        # be a multi-line string for stream-json result/error
        # records); we walk lines, flag on detection, and rebuild
        # the content with escaped leading ``@`` characters where
        # the hash-mode emission ban triggered.
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
        # Open + immediately close: agent_emit captures point-in-time
        # output, not a duration.  The renderer keys on the closed
        # span; receivers that need the open/closed lifecycle still
        # see ``run.start`` then ``run.complete`` for I1 compliance.
        self._run_tracker.complete_run(rid, outputs={})
        return rid

    def _track_prose_flood(self, handle: AgentHandle) -> None:
        """Track DR-0010 prose-flood threshold (>5 / 60s -> escalate).

        Used by the assistant-content text path so flood detection
        also covers stream-json prose, not just unparseable lines.
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
        """Return True for a Claude stream-json record (``type``/``event``)."""
        return "type" in obj or "event" in obj


def _stringify_payload(parsed: Dict[str, Any]) -> str:
    """Serialize a stream-json record as a single-line JSON string.

    Used as the ``llmnb.emit_content`` value for ``agent_emit`` spans
    of kind ``system_message`` / ``result`` / ``error``.  The full
    record is preserved verbatim so the operator surface can show
    the raw stream-json content; the metadata writer applies the
    blob-extraction pass on persistence if it exceeds the threshold.
    """
    try:
        return json.dumps(parsed, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(parsed)


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

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
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional

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
        rendered_prompt = render_system_prompt(task)
        # RFC-002 §"Failure modes": refuse spawn on system-prompt
        # template major-version mismatch; warn-and-proceed on minor.
        try:
            self._validate_template_version(rendered_prompt)
        except PreSpawnValidationError as exc:
            self._record_synthetic_problem(
                agent_id, zone_id, str(exc), exc.log_signature,
            )
            raise
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
        now = time.monotonic()
        handle = AgentHandle(
            agent_id=agent_id, zone_id=zone_id, popen=popen,
            started_at=now, work_dir=work_dir,
            stdout_thread=threading.Thread(),  # placeholder
            stderr_thread=threading.Thread(),  # placeholder
            _last_stdout_ts=now,
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

    def _spawn_from_config_entry(self, entry: Dict[str, Any]) -> AgentHandle:
        """Resolve a recoverable+volatile config entry to a spawn(...) call.

        Required keys: ``agent_id``, ``zone_id``, ``task``,
        ``work_dir``. Optional volatile keys (per RFC-005
        ``config.volatile.agents[]``): ``model``. Raises ValueError on
        missing required key; the caller in
        :meth:`respawn_from_config` translates that to ``'failed'``.
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
        return self.spawn(
            zone_id=zone_id, agent_id=agent_id, task=task,
            work_dir=work_dir, api_key=api_key,
            model=model, use_bare=use_bare,
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
        """
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

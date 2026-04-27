"""LLMKernel ↔ subsystems bootstrap (Stage 2 Track B3 + B4 wiring).

Additive helpers integrating Track B2 :class:`RunTracker`, Track B3
:class:`CustomMessageDispatcher`, Track B1
:class:`OperatorBridgeServer`, and Track B4
:class:`AgentSupervisor` into the kernel lifecycle without modifying
``kernel.py`` invasively.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import TYPE_CHECKING, Optional, Tuple

from .agent_supervisor import AgentSupervisor
from .custom_messages import CustomMessageDispatcher
from .mcp_server import OperatorBridgeServer
from .metadata_writer import MetadataWriter
from .run_tracker import RunTracker

if TYPE_CHECKING:  # pragma: no cover
    from ipykernel.ipkernel import IPythonKernel

logger: logging.Logger = logging.getLogger("llm_kernel._kernel_hooks")

#: Kernel attribute names used to stash the wired-up subsystems.
ATTR_DISPATCHER: str = "_llmnb_dispatcher"
ATTR_RUN_TRACKER: str = "_llmnb_run_tracker"
ATTR_OPERATOR_BRIDGE: str = "_llmnb_operator_bridge"
ATTR_AGENT_SUPERVISOR: str = "_llmnb_agent_supervisor"
ATTR_METADATA_WRITER: str = "_llmnb_metadata_writer"

#: Default URL the supervisor's spawned agents point ANTHROPIC_BASE_URL at
#: when no ``LLMKERNEL_LITELLM_ENDPOINT_URL`` env override is set. The
#: operator runs the LiteLLM proxy via ``python -m llm_kernel litellm-proxy``
#: or via ``attach_kernel_subsystems`` once the kernel feature is fully
#: wired; the supervisor's pre-spawn health check probes this URL.
_DEFAULT_LITELLM_ENDPOINT: str = "http://127.0.0.1:8000/v1"


def attach_dispatcher(kernel: "IPythonKernel") -> CustomMessageDispatcher:
    """Instantiate and start a :class:`CustomMessageDispatcher`.

    Stashes on ``kernel._llmnb_dispatcher``. Idempotent.
    """
    existing = getattr(kernel, ATTR_DISPATCHER, None)
    if existing is not None:
        return existing  # type: ignore[no-any-return]
    dispatcher = CustomMessageDispatcher(kernel)
    dispatcher.start()
    setattr(kernel, ATTR_DISPATCHER, dispatcher)
    logger.info("attached CustomMessageDispatcher to kernel")
    return dispatcher


def attach_run_tracker(
    kernel: "IPythonKernel", trace_id: str,
    agent_id: Optional[str] = None, zone_id: Optional[str] = None,
) -> RunTracker:
    """Instantiate a :class:`RunTracker` whose sink is the dispatcher.

    Calls :func:`attach_dispatcher` if needed. Stashes on
    ``kernel._llmnb_run_tracker``. Idempotent.
    """
    dispatcher = attach_dispatcher(kernel)
    existing = getattr(kernel, ATTR_RUN_TRACKER, None)
    if existing is not None:
        return existing  # type: ignore[no-any-return]
    tracker = RunTracker(
        trace_id=trace_id, sink=dispatcher,
        agent_id=agent_id, zone_id=zone_id,
    )
    setattr(kernel, ATTR_RUN_TRACKER, tracker)
    logger.info(
        "attached RunTracker; trace_id=%s agent=%s zone=%s",
        trace_id, agent_id, zone_id,
    )
    return tracker


def attach_agent_supervisor(
    kernel: "IPythonKernel",
    run_tracker: RunTracker, dispatcher: CustomMessageDispatcher,
    litellm_endpoint_url: Optional[str] = None,
) -> AgentSupervisor:
    """Instantiate an :class:`AgentSupervisor` wired to B2 + B3 collaborators.

    Reads ``LLMKERNEL_LITELLM_ENDPOINT_URL`` from the environment if
    ``litellm_endpoint_url`` is None. Stashes on
    ``kernel._llmnb_agent_supervisor``. Idempotent.
    """
    existing = getattr(kernel, ATTR_AGENT_SUPERVISOR, None)
    if existing is not None:
        return existing  # type: ignore[no-any-return]
    url = (litellm_endpoint_url
           or os.environ.get("LLMKERNEL_LITELLM_ENDPOINT_URL")
           or _DEFAULT_LITELLM_ENDPOINT)
    supervisor = AgentSupervisor(
        run_tracker=run_tracker, dispatcher=dispatcher,
        litellm_endpoint_url=url,
    )
    setattr(kernel, ATTR_AGENT_SUPERVISOR, supervisor)
    logger.info("attached AgentSupervisor; litellm_endpoint=%s", url)
    return supervisor


def attach_metadata_writer(
    kernel: "IPythonKernel",
    dispatcher: CustomMessageDispatcher,
    run_tracker: RunTracker,
) -> MetadataWriter:
    """Instantiate a :class:`MetadataWriter` wired to dispatcher + tracker.

    Stashes on ``kernel._llmnb_metadata_writer``. Idempotent. Starts
    the 30-second autosave timer per RFC-005 §"Snapshot triggers"
    case 3.
    """
    existing = getattr(kernel, ATTR_METADATA_WRITER, None)
    if existing is not None:
        return existing  # type: ignore[no-any-return]
    writer = MetadataWriter(
        dispatcher=dispatcher, run_tracker=run_tracker,
    )
    writer.start()
    setattr(kernel, ATTR_METADATA_WRITER, writer)
    logger.info("attached MetadataWriter; 30s autosave running")
    return writer


def attach_kernel_subsystems(
    kernel: "IPythonKernel",
) -> Tuple[
    CustomMessageDispatcher, RunTracker, OperatorBridgeServer,
    AgentSupervisor, MetadataWriter,
]:
    """One-call paper-telephone bootstrap.

    Reads ``LLMKERNEL_AGENT_ID`` / ``LLMKERNEL_ZONE_ID`` /
    ``LLMKERNEL_RUN_TRACE_ID`` / ``LLMKERNEL_LITELLM_ENDPOINT_URL`` from
    the environment with defaults; wires the dispatcher, run-tracker,
    an :class:`OperatorBridgeServer` pointed at the same run-tracker,
    the :class:`AgentSupervisor` (Track B4), and the new
    :class:`MetadataWriter` (RFC-005). Returns the five-tuple.
    """
    agent_id = os.environ.get("LLMKERNEL_AGENT_ID", "kernel")
    zone_id = os.environ.get("LLMKERNEL_ZONE_ID", "default")
    trace_id = os.environ.get("LLMKERNEL_RUN_TRACE_ID") or str(uuid.uuid4())
    dispatcher = attach_dispatcher(kernel)
    tracker = attach_run_tracker(kernel, trace_id, agent_id, zone_id)
    bridge = getattr(kernel, ATTR_OPERATOR_BRIDGE, None)
    if bridge is None:
        bridge = OperatorBridgeServer(
            agent_id=agent_id, zone_id=zone_id,
            trace_id=trace_id, run_tracker=tracker, dispatcher=dispatcher,
        )
        setattr(kernel, ATTR_OPERATOR_BRIDGE, bridge)
    supervisor = attach_agent_supervisor(kernel, tracker, dispatcher)
    metadata_writer = attach_metadata_writer(kernel, dispatcher, tracker)
    return dispatcher, tracker, bridge, supervisor, metadata_writer


__all__ = [
    "ATTR_AGENT_SUPERVISOR", "ATTR_DISPATCHER", "ATTR_METADATA_WRITER",
    "ATTR_OPERATOR_BRIDGE", "ATTR_RUN_TRACKER",
    "attach_agent_supervisor", "attach_dispatcher",
    "attach_kernel_subsystems", "attach_metadata_writer",
    "attach_run_tracker",
]

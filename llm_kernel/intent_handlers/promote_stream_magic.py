"""``promote_stream_magic`` operator-action handler -- PLAN-S5.0.4 §3.3.

Per [discipline/certified-magic-emitter](../../docs/atoms/discipline/certified-magic-emitter.md)'s
"Stream emissions by agents (banned but observed)" section: when a
privileged agent forgets to invoke the ``emit_magic_cell`` MCP tool
and instead types magic in its prose, Layer-1 contamination detection
flags the cell AND emits K3L (informational marker). The extension
renders a one-click promotion chip on the contaminated cell. Clicking
the chip emits an ``operator.action`` envelope with
``action_type: "promote_stream_magic"`` and parameters
``{cell_id: <id>, line: <verbatim>}``. The mcp_server router calls
this handler, which:

1. Resolves the source cell's ``bound_agent_id`` (the privileged
   agent whose stream contained the magic).
2. Parses the verbatim line to recover ``name`` + ``args`` + (no body
   -- stream-emitted single lines never carry @@expand bodies).
3. Synthesizes an :func:`emit_magic_cell` call on the operator's
   behalf -- clause 1 is satisfied by the click, NOT the stream.
4. Stamps ``promoted_from_stream: True`` on the resulting cell's
   provenance record so the renderer chip can render
   "Promoted from stream emission".

Without the operator click, nothing dispatches -- the sanitized
line stays in outputs and never reaches a parser. This handler is
the operator's escape hatch from forgetfulness, not a widening of
the emission ban.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

logger: logging.Logger = logging.getLogger(
    "llm_kernel.intent_handlers.promote_stream_magic"
)


# Match plain ``@@<name>`` / ``@<name>`` followed by optional args;
# rejects hashed shapes (those should never reach the promotion
# chip -- the hash-mode emission ban already strips them at the
# socket_writer boundary).
_PROMOTE_LINE_PATTERN = re.compile(
    r"^@@?(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)(?:\s+(?P<tail>.*))?$"
)


class PromoteStreamMagicError(ValueError):
    """Raised on malformed inputs (missing cell_id, unparseable line)."""


def _parse_magic_line(line: str) -> Tuple[str, Dict[str, str]]:
    """Recover ``(name, args)`` from a verbatim stream-emitted magic line.

    Accepts plain magic shape only. Args are recovered as a single
    positional string under key ``"_positional"`` AND parsed for
    ``k=v`` tokens which land as named entries. Body text from
    multi-line emissions is dropped -- the promotion chip targets
    single-line emissions only.
    """
    if not isinstance(line, str) or not line.strip():
        raise PromoteStreamMagicError("promote_stream_magic: empty line")
    # Take only the first non-blank line -- the chip is per-line, the
    # operator selected one. Strip any escaped leading-at (Layer-2
    # sanitization may have prepended ``\``).
    candidate = line.strip()
    if candidate.startswith("\\@"):
        candidate = candidate[1:]
    m = _PROMOTE_LINE_PATTERN.match(candidate)
    if m is None:
        raise PromoteStreamMagicError(
            f"promote_stream_magic: line does not match plain magic shape: "
            f"{candidate[:64]!r}"
        )
    name = m.group("name")
    tail = (m.group("tail") or "").strip()
    args: Dict[str, str] = {}
    if tail:
        # Parse ``k=v`` tokens. Tokens without ``=`` are gathered as
        # positional under ``_positional`` (space-joined to preserve
        # operator intent). This mirrors generators' arg parsing
        # convention.
        positional_parts = []
        for token in tail.split():
            if "=" in token:
                key, _, value = token.partition("=")
                if key:
                    args[key] = value
            else:
                positional_parts.append(token)
        if positional_parts:
            args["_positional"] = " ".join(positional_parts)
    return name, args


def handle_promote_stream_magic(
    *,
    params: Dict[str, Any],
    writer: Any,
    cell_manager: Any,
    zone_id: str,
) -> Optional[Dict[str, str]]:
    """Synthesize an :func:`emit_magic_cell` call from a chip click.

    PLAN-S5.0.4 §3.3 entry point. Parameters from the
    ``operator.action`` envelope:

    * ``cell_id``: the contaminated cell whose ``contamination_log``
      carries the stream-emitted magic.
    * ``line``: the verbatim line the operator chose to promote.

    Returns ``{"cell_id": <new_id>}`` on success, ``None`` when the
    handler refuses (logged at warning level so the operator's click
    is visible in the audit trail).
    """
    cell_id = params.get("cell_id")
    line = params.get("line")
    if not isinstance(cell_id, str) or not cell_id:
        logger.warning(
            "promote_stream_magic: missing/empty cell_id; params=%r",
            params,
        )
        return None
    if not isinstance(line, str) or not line.strip():
        logger.warning(
            "promote_stream_magic: missing/empty line for cell_id=%s",
            cell_id,
        )
        return None
    # Recover the source cell's bound_agent_id -- this is the
    # privileged agent the magic was emitted *by*. The cell record
    # carries ``bound_agent_id`` per the cell-kinds schema.
    get_record = getattr(writer, "get_cell_record", None)
    if not callable(get_record):
        logger.warning(
            "promote_stream_magic: writer does not expose get_cell_record"
        )
        return None
    record = get_record(cell_id)
    if not isinstance(record, dict):
        logger.warning(
            "promote_stream_magic: cell_id=%s not found", cell_id,
        )
        return None
    agent_id = record.get("bound_agent_id")
    if not isinstance(agent_id, str) or not agent_id:
        logger.warning(
            "promote_stream_magic: cell_id=%s has no bound_agent_id; "
            "cannot resolve source agent for promotion",
            cell_id,
        )
        return None
    # Parse the line.
    try:
        name, args = _parse_magic_line(line)
    except PromoteStreamMagicError as exc:
        logger.warning("promote_stream_magic: %s", exc)
        return None
    # Synthesize the emit_magic_cell call. ``promoted_from_stream``
    # tags the resulting cell's provenance per PLAN §3.3.
    from .. import magic_emit_tool

    try:
        result = magic_emit_tool.emit_magic_cell(
            agent_id=agent_id,
            zone_id=zone_id,
            name=name,
            args=args,
            body=None,
            position={"after_cell_id": cell_id},
            writer=writer,
            cell_manager=cell_manager,
            promoted_from_stream=True,
        )
    except magic_emit_tool.MagicEmitError as exc:
        # The privileged-emit handler rejects K3K when the agent has
        # no covering grant. Surface as a warning -- the chip should
        # only have rendered for privileged agents, but a revoke
        # raced with the click. The audit trail records both events.
        logger.warning(
            "promote_stream_magic: emit_magic_cell rejected "
            "(agent_id=%s, name=%s): %s",
            agent_id, name, exc,
        )
        return None
    return result

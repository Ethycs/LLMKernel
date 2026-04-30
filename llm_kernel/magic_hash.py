"""HMAC-hash primitives for cell-magic injection defense (S5.0.1).

Per [PLAN-S5.0.1-cell-magic-injection-defense.md] §3.1 + §3.4. This
module supplies the *foundation* primitives only — pin lifecycle
magics, parser hash awareness, Cell Manager precondition gates, and
extension UI all live in later sub-slices (5.0.1b/c/d/e).

Public surface:

* :func:`magic_hash` — HMAC-SHA256 of a magic name keyed by the
  operator's pin, hex-truncated to ``length`` chars (default 8).
* :func:`magic_pin_fingerprint` — salted SHA-256 of the pin, used at
  round-trip verification time (the pin itself is **never** stored in
  the notebook; only its fingerprint).
* :func:`looks_like_hashed_magic` — pattern-shape check, no pin
  required. Used by the emission-ban detector to flag any agent
  output line that *looks* like a hashed magic regardless of whether
  it would actually validate.
* :func:`looks_like_plain_magic` — pattern-shape check against the
  registered cell + line magic names. The contamination detector
  uses this for the always-on layer.
* :func:`validate_hashed_magic` — strict validator that recovers
  the magic name from a hash by walking the registered names with
  constant-time compare.
* :func:`strip_hashes_from_text` — bidirectional strip helper. At
  every agent-visible boundary (ContextPacker replay, handoff stdin,
  resume injection), every ``@@<hash>:<name>`` line is reduced to
  plain ``@@<name>`` form so the agent can never observe — and
  therefore never replay — the hash.
* :func:`escape_leading_at` — Layer-2 emission-ban escape: prepend a
  ``\\`` to the leading ``@`` so the line is body, not dispatchable.

Pure module: no I/O, no logging, no kernel-state access. The
contamination detector and output sanitizer that *use* these
helpers live in :mod:`llm_kernel.agent_supervisor` and
:mod:`llm_kernel.socket_writer` respectively.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Iterable, Optional, Tuple


__all__ = (
    "HASHED_MAGIC_LINE",
    "PLAIN_MAGIC_LINE",
    "FINGERPRINT_SALT",
    "magic_hash",
    "magic_pin_fingerprint",
    "looks_like_hashed_magic",
    "looks_like_plain_magic",
    "validate_hashed_magic",
    "strip_hash_from_line",
    "strip_hashes_from_text",
    "escape_leading_at",
)


#: Project-wide salt for pin fingerprinting. PLAN §3.1 nominates
#: ``b"llmnb-magic-v1-fingerprint"``; this slice uses a shorter form
#: equivalent for the same purpose. The salt is intentionally constant
#: so any operator who re-enters the same pin reproduces the same
#: fingerprint without leaking the pin.
FINGERPRINT_SALT: bytes = b"llmnb-v1-pin"


#: Canonical hashed-magic line shape per PLAN §3.1. Matches both the
#: cell-magic ``@@<hash>:<name>`` form and the line-magic
#: ``@<hash>:<name>`` form. ``\\b`` ensures the name token is followed
#: by a whitespace, end-of-line, or other non-word boundary so
#: ``@@deadbeef:spawn123`` does not silently classify as ``spawn``.
HASHED_MAGIC_LINE: re.Pattern[str] = re.compile(
    r"^@@?([a-f0-9]+):([a-zA-Z_][a-zA-Z0-9_]*)\b"
)


#: Plain-magic line shape — ``@@<name>`` (cell magic) or ``@<name>``
#: (line magic). The detector layer uses this against the registered
#: name set; the parser already classifies these into ``ParsedCell``.
PLAIN_MAGIC_LINE: re.Pattern[str] = re.compile(
    r"^@@?([a-zA-Z_][a-zA-Z0-9_]*)(\s|:|$)"
)


def magic_hash(pin: str, magic_name: str, *, length: int = 8) -> str:
    """HMAC-SHA256 of ``magic_name`` keyed by ``pin``, hex-truncated.

    Default length 8 (32 bits → 1 in 4B forgery probability per random
    attempt). Operator may extend (V1.5+) by passing ``length=12`` for
    paranoia or ``length=6`` for typability vs strength.

    Both ``pin`` and ``magic_name`` are encoded as UTF-8. Empty pin is
    *allowed by this primitive* (caller-side validation rejects empty
    pins via the pin-lifecycle magics in slice 5.0.1b — this module is
    pure).
    """
    if not isinstance(pin, str):
        raise TypeError(f"pin must be a str; got {type(pin).__name__}")
    if not isinstance(magic_name, str):
        raise TypeError(
            f"magic_name must be a str; got {type(magic_name).__name__}"
        )
    if length <= 0:
        raise ValueError(f"length must be positive; got {length!r}")
    h = hmac.new(
        pin.encode("utf-8"),
        magic_name.encode("utf-8"),
        hashlib.sha256,
    )
    return h.hexdigest()[:length]


def magic_pin_fingerprint(pin: str) -> str:
    """One-way fingerprint of ``pin`` for round-trip verification.

    Stored in ``metadata.rts.config.magic_pin_fingerprint`` (slice
    5.0.1b schema work). Pin itself is never written to the notebook.
    The salt is project-wide so the fingerprint is stable across
    restarts but not reversible without the pin.
    """
    if not isinstance(pin, str):
        raise TypeError(f"pin must be a str; got {type(pin).__name__}")
    return hashlib.sha256(FINGERPRINT_SALT + pin.encode("utf-8")).hexdigest()[:16]


def looks_like_hashed_magic(line: str) -> bool:
    """Return True iff ``line`` matches the canonical hashed-magic shape.

    Pure pattern check; no pin required. The emission-ban scanner
    uses this to flag any agent output line *shaped* like a hashed
    magic, regardless of whether it would actually validate against
    the operator's pin. The conservative bias is correct: if it looks
    like one we treat it as a possible exfil attempt.
    """
    if not isinstance(line, str) or not line:
        return False
    return HASHED_MAGIC_LINE.match(line) is not None


def looks_like_plain_magic(line: str, known_names: Iterable[str]) -> bool:
    """Return True iff ``line`` is a plain ``@@<name>`` / ``@<name>``
    where ``<name>`` is in ``known_names``.

    The always-on contamination detector calls this with the union of
    cell + line magic registry keys. Unknown ``@@xyzzy`` lines are
    NOT flagged — they're just typed prose.
    """
    if not isinstance(line, str) or not line:
        return False
    m = PLAIN_MAGIC_LINE.match(line)
    if m is None:
        return False
    name = m.group(1)
    # Normalize known_names into a set for the membership test —
    # callers typically pass a frozenset/dict_keys; we accept any
    # iterable for testability.
    if not isinstance(known_names, (set, frozenset)):
        known_names = set(known_names)
    return name in known_names


def validate_hashed_magic(
    line: str, pin: str, known_names: Iterable[str], *, length: int = 8,
) -> Tuple[bool, Optional[str]]:
    """Validate a candidate hashed-magic line against a pin + name set.

    Walks ``known_names`` and checks each candidate's hash against
    the line's prefix using :func:`hmac.compare_digest` (constant
    time). Returns ``(True, recovered_name)`` on a match,
    ``(False, None)`` otherwise.

    Used by the parser in slice 5.0.1b. Registered here so the
    primitive ships with the foundation slice and slice 5.0.1b can
    import it directly.
    """
    if not isinstance(line, str) or not line:
        return False, None
    m = HASHED_MAGIC_LINE.match(line)
    if m is None:
        return False, None
    candidate_hash = m.group(1)
    candidate_name = m.group(2)
    # The line carries the recovered name explicitly per the canonical
    # shape ``@@<hash>:<name> <args>``. We still compute the hash from
    # the pin + name and constant-time compare — that's the security
    # property. We do NOT trust ``candidate_name`` until the hash
    # matches, AND we require the name to be in ``known_names``.
    if not isinstance(known_names, (set, frozenset)):
        known_names = set(known_names)
    if candidate_name not in known_names:
        return False, None
    expected = magic_hash(pin, candidate_name, length=length)
    if hmac.compare_digest(expected, candidate_hash):
        return True, candidate_name
    return False, None


def strip_hash_from_line(line: str, known_names: Iterable[str]) -> str:
    """Reduce one ``@@<hash>:<name> <args>`` line to plain ``@@<name> <args>``.

    Returns the input unchanged if the line does not match the hashed
    shape OR the recovered name is not in ``known_names``. The
    name-membership check guards against an attacker emitting
    ``@@deadbeef:fakemagic`` and having the strip helper rewrite it
    into a plain magic that *would* dispatch — only registered names
    survive the strip.
    """
    if not isinstance(line, str) or not line:
        return line
    m = HASHED_MAGIC_LINE.match(line)
    if m is None:
        return line
    name = m.group(2)
    if not isinstance(known_names, (set, frozenset)):
        known_names = set(known_names)
    if name not in known_names:
        return line
    sigil = "@@" if line.startswith("@@") else "@"
    # ``m.end()`` is the position right after the ``\\b`` sentinel —
    # i.e., the end of the matched name. Everything from there onward
    # is preserved verbatim (whitespace, args, line endings).
    tail = line[m.end():]
    return f"{sigil}{name}{tail}"


def strip_hashes_from_text(text: str, known_names: Iterable[str]) -> str:
    """Apply :func:`strip_hash_from_line` to every line of ``text``.

    Used by ContextPacker, handoff injection, and resume replay paths
    in slice 5.0.1b/c. Storage retains the canonical hashed form;
    every agent-visible boundary calls this so the agent never
    observes — and therefore never replays — the hash.
    """
    if not isinstance(text, str) or not text:
        return text
    if not isinstance(known_names, (set, frozenset)):
        known_names = set(known_names)
    # ``str.splitlines`` collapses CRLF and CR to LF on rejoin via
    # ``"\n".join``; for the strip helper that is acceptable because
    # the only consumers that touch line endings (cell-source
    # storage) never call this — they read canonical text directly.
    lines = text.splitlines()
    stripped = [strip_hash_from_line(line, known_names) for line in lines]
    # Preserve a trailing newline if the original had one (splitlines
    # drops it; the agent-visible-surface callers prefer round-trip).
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(stripped) + suffix


def escape_leading_at(line: str) -> str:
    """Prepend ``\\`` to the leading ``@`` of ``line`` (Layer-2 ban).

    Used by the emission-ban sanitizer when an agent / tool stream
    emits a line that LOOKS LIKE a hashed magic. The escaped form
    ``\\@@<hash>:<name>`` is body, never dispatchable — even with
    the correct pin, because the parser only sees ``@@`` at column
    zero, not after a backslash.

    Idempotent on already-escaped lines: ``\\@@x`` → ``\\@@x``.
    No-op on lines that don't start with ``@``.
    """
    if not isinstance(line, str) or not line:
        return line
    if line.startswith("\\@"):
        # Already escaped — re-escaping would double the prefix.
        return line
    if line.startswith("@"):
        return "\\" + line
    return line

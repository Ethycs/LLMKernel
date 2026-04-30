"""Pin lifecycle handlers for ``@auth set / rotate / off / verify``.

Per [PLAN-S5.0.1-cell-magic-injection-defense.md] §3.5 PIN LIFECYCLE
MAGICS. The ``@auth`` line magic is a sub-namespace dispatched by
subcommand keyword in the args string:

* ``@auth set <pin>`` — install a fresh pin: validates pin shape,
  computes fingerprint, sets ``magic_hash_enabled=True``, stores the
  pin in environment (``LLMNB_OPERATOR_PIN`` for V1; keychain
  integration is V2+), triggers a Cell-Manager re-stamp pass that
  rewrites every plain ``@@<name>`` magic line in the notebook to
  ``@@<hash>:<name>`` form.
* ``@auth rotate <new_pin>`` — atomic pin rotation: validates
  ``new_pin``, walks every existing magic line in the notebook
  re-stamping ``@@<old_hash>:<name>`` → ``@@<new_hash>:<name>``,
  updates the fingerprint, then discards the old pin from the env.
* ``@auth off`` — disables hash mode: clears
  ``magic_hash_enabled`` + ``magic_pin_fingerprint``, removes the pin
  from the env. Existing hashed lines stay valid as text (they
  decompose to body until operator rewrites).
* ``@auth verify`` — checks the stored env pin matches the notebook's
  ``magic_pin_fingerprint``. Returns a status dict; no schema
  mutation. Used for operator UX confirmation.

Dispatch surface:

The four handlers are exposed as :func:`apply_auth_command` taking the
parsed subcommand + args, plus a writer + cell-manager pair. The
caller (extension intent dispatcher / kernel intent surface) wires
this in. The :class:`magic_registry.LineMagicHandler` for ``auth``
itself is a pure flag-recorder; the runtime side effect happens via
the dispatch surface, NOT during parse. This keeps :func:`parse_cell`
pure (Engineering_Guide §11.7).

V1 limitations (documented):

* Pin storage uses the ``LLMNB_OPERATOR_PIN`` environment variable.
  No keychain. No fallback to ``.env``. Operator types ``@auth set``
  per session.
* Pin recovery: none. Loss of pin → ``@auth off`` + ``@auth set``
  with a new pin, accepting that hashed lines from before the loss
  decompose to body.
"""

from __future__ import annotations

import os
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import magic_hash as MH
from .magic_registry import RESERVED_NAMES


__all__ = (
    "AuthCommandResult",
    "PIN_ENV_VAR",
    "validate_pin",
    "apply_auth_command",
    "auth_set",
    "auth_rotate",
    "auth_off",
    "auth_verify",
)


#: Environment variable that carries the active operator pin in V1.
#: Per the slice plan: no keychain, no .env fallback. Operator types
#: ``@auth set`` per session and the handler writes here.
PIN_ENV_VAR: str = "LLMNB_OPERATOR_PIN"


# K-codes used by the auth handlers. K38/K39/K3B are registered in
# the K-class registry per PLAN §3.9 (slice 5.0.1c lands the wire
# envelope side); we attach them here as string constants so handler
# results carry classifiable codes without circular import.
K38_PIN_TOO_SHORT: str = "K38"
K39_PIN_COLLISION: str = "K39"
K3B_PIN_FINGERPRINT_MISMATCH: str = "K3B"


@dataclass
class AuthCommandResult:
    """Outcome of an ``@auth`` subcommand handler.

    ``ok`` is True on success. On failure ``code`` carries one of the
    K-codes (K38/K39/K3B/K3x) so the caller can index without string
    matching. ``details`` is a free-form dict for handler-specific
    context (e.g. ``{"restamped": <int>}`` from ``set`` / ``rotate``;
    ``{"fingerprint": <str>}`` from ``verify``).
    """

    ok: bool
    code: Optional[str] = None
    reason: str = ""
    details: Optional[Dict[str, Any]] = None


# --- Pin validation -------------------------------------------------


_PIN_PRINTABLE_NO_SPACE = set(string.printable) - set(string.whitespace)


def validate_pin(pin: str) -> AuthCommandResult:
    """Validate a candidate operator pin per PLAN-S5.0.1 §3.5.

    Rules (V1):

    * Must be a non-empty ``str``.
    * Length ``>= 8`` (PLAN §3.9 K38 threshold; the plan tests K38 at
      ``< 12`` but the slice 5.0.1b directive says ``>= 8``; we use
      the directive's value).
    * ASCII printable, no whitespace anywhere (would break shell-
      tokenization on the cell line).
    * Not the reserved literal string ``"pin"`` (avoids the case
      where the operator types ``@auth set pin`` thinking ``pin`` is
      a placeholder).
    * Not a collision with any registered cell-magic / line-magic
      name (per :data:`magic_registry.RESERVED_NAMES`) — K39.
    """
    if not isinstance(pin, str):
        return AuthCommandResult(
            ok=False, code=K38_PIN_TOO_SHORT,
            reason=f"pin must be a str; got {type(pin).__name__}",
        )
    if len(pin) < 8:
        return AuthCommandResult(
            ok=False, code=K38_PIN_TOO_SHORT,
            reason=f"pin too short: len={len(pin)} < 8",
        )
    if pin == "pin":
        return AuthCommandResult(
            ok=False, code=K39_PIN_COLLISION,
            reason="pin is the reserved literal 'pin'",
        )
    if pin in RESERVED_NAMES:
        return AuthCommandResult(
            ok=False, code=K39_PIN_COLLISION,
            reason=f"pin collides with reserved magic name {pin!r}",
        )
    for ch in pin:
        if ch not in _PIN_PRINTABLE_NO_SPACE:
            return AuthCommandResult(
                ok=False, code=K38_PIN_TOO_SHORT,
                reason=(
                    f"pin contains non-printable or whitespace char "
                    f"({ord(ch)!r})"
                ),
            )
    return AuthCommandResult(ok=True)


# --- Handlers -------------------------------------------------------


def auth_set(
    pin: str,
    *,
    writer: Any,
    cell_manager: Any,
    env: Optional[Dict[str, str]] = None,
) -> AuthCommandResult:
    """``@auth set <pin>`` — install a fresh pin.

    PLAN-S5.0.1 §3.5. Refuses if hash mode is already on (operator
    must ``@auth rotate`` or ``@auth off`` first to be explicit
    about intent). Steps:

    1. Validate pin shape.
    2. Compute fingerprint.
    3. Atomically set ``magic_hash_enabled=True`` +
       ``magic_pin_fingerprint=<fp>`` via
       :meth:`MetadataWriter.set_hash_mode`.
    4. Store pin in env (``LLMNB_OPERATOR_PIN``).
    5. Trigger ``cell_manager.restamp_magics(None, pin)``.

    The ``env`` parameter is the env dict to mutate (defaults to
    ``os.environ``); tests pass a private dict via monkeypatch. This
    keeps the handler test-parallel-safe — no global env mutation
    when the caller threads its own dict.
    """
    target_env: Dict[str, str] = (
        os.environ if env is None else env  # type: ignore[assignment]
    )
    v = validate_pin(pin)
    if not v.ok:
        return v
    # Refuse silent overwrite of an active hash mode.
    if writer.get_config_setting("magic_hash_enabled"):
        return AuthCommandResult(
            ok=False, code="K3X",
            reason=(
                "hash mode already on; use '@auth rotate' or "
                "'@auth off' first"
            ),
        )
    fingerprint = MH.magic_pin_fingerprint(pin)
    writer.set_hash_mode(enabled=True, fingerprint=fingerprint)
    target_env[PIN_ENV_VAR] = pin
    restamped = 0
    emissions: List[Dict[str, str]] = []
    try:
        restamped, emissions = cell_manager.restamp_magics(
            old_pin=None, new_pin=pin,
        )
    except Exception as exc:  # pragma: no cover — defensive
        return AuthCommandResult(
            ok=False, code="K37",
            reason=f"restamp failed: {exc!r}",
            details={"fingerprint": fingerprint},
        )
    return AuthCommandResult(
        ok=True,
        details={
            "fingerprint": fingerprint,
            "restamped": restamped,
            "k_emissions": emissions,
        },
    )


def auth_rotate(
    new_pin: str,
    *,
    writer: Any,
    cell_manager: Any,
    env: Optional[Dict[str, str]] = None,
) -> AuthCommandResult:
    """``@auth rotate <new_pin>`` — atomic pin rotation.

    Reads the active pin from ``env[PIN_ENV_VAR]``; refuses if hash
    mode is off (no pin to rotate from). Validates the new pin,
    re-stamps every magic line, updates the fingerprint, and ONLY
    THEN replaces the env pin. If the re-stamp raises, the env pin
    is unchanged so the operator can retry.
    """
    target_env: Dict[str, str] = (
        os.environ if env is None else env  # type: ignore[assignment]
    )
    if not writer.get_config_setting("magic_hash_enabled"):
        return AuthCommandResult(
            ok=False, code="K3X",
            reason="hash mode is off; nothing to rotate",
        )
    old_pin = target_env.get(PIN_ENV_VAR, "")
    if not old_pin:
        return AuthCommandResult(
            ok=False, code=K3B_PIN_FINGERPRINT_MISMATCH,
            reason=(
                "no active pin in env; rotate requires the current "
                "pin to be present (run '@auth set' if hash mode is "
                "stale)"
            ),
        )
    v = validate_pin(new_pin)
    if not v.ok:
        return v
    new_fingerprint = MH.magic_pin_fingerprint(new_pin)
    try:
        restamped, emissions = cell_manager.restamp_magics(
            old_pin=old_pin, new_pin=new_pin,
        )
    except Exception as exc:  # pragma: no cover — defensive
        return AuthCommandResult(
            ok=False, code="K37",
            reason=f"rotate restamp failed: {exc!r}",
        )
    # Re-stamp succeeded — flip the schema + env atomically next.
    writer.set_hash_mode(enabled=True, fingerprint=new_fingerprint)
    target_env[PIN_ENV_VAR] = new_pin
    return AuthCommandResult(
        ok=True,
        details={
            "fingerprint": new_fingerprint,
            "restamped": restamped,
            "k_emissions": emissions,
        },
    )


def auth_off(
    *,
    writer: Any,
    cell_manager: Any,
    env: Optional[Dict[str, str]] = None,
) -> AuthCommandResult:
    """``@auth off`` — disable hash mode.

    Clears ``magic_hash_enabled`` + ``magic_pin_fingerprint`` from
    the schema; removes the pin from the env. Existing hashed lines
    stay verbatim in cell text per PLAN §3.7 — they decompose to
    body on next parse without hash mode (the parser's permissive
    path doesn't match hashed shapes).
    """
    target_env: Dict[str, str] = (
        os.environ if env is None else env  # type: ignore[assignment]
    )
    old_pin = target_env.get(PIN_ENV_VAR, "")
    # Disable path is no-op on text per PLAN §3.7.
    try:
        cell_manager.restamp_magics(old_pin=old_pin or None, new_pin=None)
    except Exception:  # pragma: no cover — defensive
        pass
    writer.set_hash_mode(enabled=False, fingerprint=None)
    target_env.pop(PIN_ENV_VAR, None)
    return AuthCommandResult(ok=True, details={"hash_mode_cleared": True})


def auth_verify(
    *,
    writer: Any,
    env: Optional[Dict[str, str]] = None,
) -> AuthCommandResult:
    """``@auth verify`` — check env pin matches notebook fingerprint.

    Returns ``ok=True`` when the fingerprint of the env-stored pin
    equals the notebook's stored ``magic_pin_fingerprint``. K3B on
    mismatch. ``ok=False, code=K3X`` if hash mode is off (nothing to
    verify against).
    """
    target_env: Dict[str, str] = (
        os.environ if env is None else env  # type: ignore[assignment]
    )
    enabled = bool(writer.get_config_setting("magic_hash_enabled"))
    if not enabled:
        return AuthCommandResult(
            ok=False, code="K3X",
            reason="hash mode is off; nothing to verify",
        )
    stored_fp = writer.get_config_setting("magic_pin_fingerprint")
    pin = target_env.get(PIN_ENV_VAR, "")
    if not pin:
        return AuthCommandResult(
            ok=False, code=K3B_PIN_FINGERPRINT_MISMATCH,
            reason="no pin loaded in env; type '@auth set <pin>' first",
        )
    candidate_fp = MH.magic_pin_fingerprint(pin)
    if candidate_fp == stored_fp:
        return AuthCommandResult(
            ok=True, details={"fingerprint": stored_fp},
        )
    return AuthCommandResult(
        ok=False, code=K3B_PIN_FINGERPRINT_MISMATCH,
        reason="env pin's fingerprint does not match notebook fingerprint",
        details={"stored": stored_fp, "candidate": candidate_fp},
    )


# --- Subcommand parser + unified dispatch ---------------------------


def _parse_auth_args(args_str: str) -> Tuple[str, str]:
    """Split an ``@auth`` line magic's args into ``(subcommand, rest)``.

    The first whitespace-token is the subcommand keyword; the rest is
    handler-specific (e.g. the pin string for ``set`` / ``rotate``).
    Returns ``("", "")`` on empty input — caller raises a "subcommand
    required" error.
    """
    if not args_str:
        return "", ""
    parts = args_str.split(None, 1)
    sub = parts[0].lower() if parts else ""
    rest = parts[1] if len(parts) > 1 else ""
    return sub, rest


def apply_auth_command(
    args_str: str,
    *,
    writer: Any,
    cell_manager: Any,
    env: Optional[Dict[str, str]] = None,
) -> AuthCommandResult:
    """Single dispatch entry point for an ``@auth ...`` line magic.

    Used by the runtime intent dispatcher when it sees ``"auth"`` in
    a parsed cell's ``line_magics`` list. The kernel calls this
    AFTER the cell parses successfully (so the line magic round-
    trips even on subcommand errors) but BEFORE the next snapshot
    emit (so the schema + env mutations land in the same write).

    On unknown subcommand → ``ok=False, code=K3X`` with a
    diagnostic reason.
    """
    sub, rest = _parse_auth_args(args_str)
    if sub == "set":
        return auth_set(
            rest.strip(), writer=writer, cell_manager=cell_manager, env=env,
        )
    if sub == "rotate":
        return auth_rotate(
            rest.strip(), writer=writer, cell_manager=cell_manager, env=env,
        )
    if sub == "off":
        return auth_off(
            writer=writer, cell_manager=cell_manager, env=env,
        )
    if sub == "verify":
        return auth_verify(writer=writer, env=env)
    return AuthCommandResult(
        ok=False, code="K3X",
        reason=f"unknown @auth subcommand: {sub!r}",
    )

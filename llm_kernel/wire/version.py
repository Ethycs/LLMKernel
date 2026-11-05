"""Wire envelope version (RFC-006), forwarded from :mod:`llm_kernel.wire.versions`.

``WIRE_VERSION`` and the ``WIRE_MAJOR`` / ``WIRE_MINOR`` / ``WIRE_PATCH``
components are *not* stored here.  They are resolved on every attribute
access (PEP 562 module ``__getattr__``) from the single source in
:mod:`llm_kernel.wire.versions`, so a runtime override of
``versions.WIRE_VERSION`` is observed by anyone reading
``llm_kernel.wire.version.WIRE_MAJOR`` at call time and the three
components can never disagree with the string.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from llm_kernel.wire import versions as _versions


def wire_version_tuple() -> Tuple[int, int, int]:
    """Return (WIRE_MAJOR, WIRE_MINOR, WIRE_PATCH) for handshake / version-skew checks."""
    return _versions.wire_version_tuple()


_FORWARDED: Dict[str, Callable[[], Any]] = {
    "WIRE_VERSION": lambda: _versions.WIRE_VERSION,
    "WIRE_MAJOR": lambda: _versions.wire_version_tuple()[0],
    "WIRE_MINOR": lambda: _versions.wire_version_tuple()[1],
    "WIRE_PATCH": lambda: _versions.wire_version_tuple()[2],
}


def __getattr__(name: str) -> Any:
    try:
        getter = _FORWARDED[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getter()


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_FORWARDED))


__all__ = [
    "WIRE_VERSION",
    "WIRE_MAJOR",
    "WIRE_MINOR",
    "WIRE_PATCH",
    "wire_version_tuple",
]

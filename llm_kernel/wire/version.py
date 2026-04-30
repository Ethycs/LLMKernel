"""Wire envelope version. Bumped by RFC-006 amendments."""

WIRE_VERSION: str = "1.0.0"
WIRE_MAJOR: int = 1
WIRE_MINOR: int = 0
WIRE_PATCH: int = 0


def wire_version_tuple() -> tuple[int, int, int]:
    """Return (WIRE_MAJOR, WIRE_MINOR, WIRE_PATCH) for handshake / version-skew checks."""
    return (WIRE_MAJOR, WIRE_MINOR, WIRE_PATCH)


__all__ = [
    "WIRE_VERSION",
    "WIRE_MAJOR",
    "WIRE_MINOR",
    "WIRE_PATCH",
    "wire_version_tuple",
]

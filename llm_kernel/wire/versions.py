"""Single source of truth for every version LLMKernel puts on the wire.

This module is the ONE place where the kernel version, the RFC
implementation versions (RFC-001 .. RFC-008), the RFC-006 wire version,
the RFC-005 ``metadata.rts`` schema version, and the capability set the
kernel advertises are defined.  Every other site -- the pty-mode ready
frame, the serve-mode handshake payloads and validation, the volatile
provider that feeds :class:`~llm_kernel.capture.drift_detector.DriftDetector`
on hydrate, the metadata writer's schema stamp and hydrate check, the
notebook-format default, the health endpoint, the exported JSON schemas,
and the driver's TCP connect defaults -- reads *through* this module at
call time rather than copying values into module-level constants.

Rules for this module:

* stdlib-only, no imports of other ``llm_kernel`` modules.  It lives
  under :mod:`llm_kernel.wire` because that package is the only kernel
  surface external drivers may import (``tests/test_lint_boundary.py``
  in the notebook repo).
* The values here are the canonical ones: what the ready frame emits.
  Bumping a version is a one-line edit here and shows up everywhere.
* Callers that need the *live* value (so a runtime override of this
  module propagates) must read ``versions.NAME`` at call time, or use
  the accessor functions below -- never ``from versions import NAME``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

__all__ = [
    # Canonical values
    "KERNEL_VERSION",
    "WIRE_VERSION",
    "SCHEMA_VERSION",
    "RFC_VERSIONS",
    "KERNEL_CAPABILITIES",
    "COMM_TARGET",
    "PACKAGE_VERSION",
    "CLIENT_VERSION",
    "SYSTEM_PROMPT_TEMPLATE_VERSION",
    # Live accessors
    "semver_tuple",
    "semver_major",
    "wire_version_tuple",
    "rfc_key",
    "rfc_version",
    "rfc_versions",
    "kernel_volatile",
    "current_volatile",
    "ready_frame_attrs",
    "kernel_capabilities",
]

_SEMVER_RE: "re.Pattern[str]" = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# ---------------------------------------------------------------------------
# Canonical values.  These are what goes over the wire today; the ready
# frame set is the truth (RFC-008 §4).
# ---------------------------------------------------------------------------

#: Kernel version emitted in the pty-mode ready frame
#: (``llmnb.kernel.version``), the serve-mode handshake response
#: (``kernel_version``), the ``/health`` route and the boot banners.
KERNEL_VERSION: str = "1.0.0"

#: RFC-006 envelope / handshake wire version.  Bumped by RFC-006
#: amendments.  ``WIRE_MAJOR`` / ``WIRE_MINOR`` / ``WIRE_PATCH`` in
#: :mod:`llm_kernel.wire.version` are derived from this string.
WIRE_VERSION: str = "1.0.0"

#: RFC-005 ``metadata.rts`` schema version the metadata writer stamps on
#: every snapshot and checks (major) on hydrate.
SCHEMA_VERSION: str = "1.0.0"

#: RFC implementation versions, keyed exactly as they appear in the
#: ready frame's ``llmnb.kernel.rfc_NNN_version`` attributes and in the
#: persisted ``config.volatile.kernel.*`` block (RFC-005).
RFC_VERSIONS: Dict[str, str] = {
    "rfc_001_version": "1.0.0",
    "rfc_002_version": "1.0.0",
    "rfc_003_version": "1.0.0",
    "rfc_004_version": "1.0.0",
    "rfc_005_version": "1.0.0",
    "rfc_006_version": "2.0.0",
    "rfc_007_version": "1.0.0",
    "rfc_008_version": "1.0.0",
}

#: Capabilities the kernel advertises in handshake responses and the
#: driver requests by default on connect.  The kernel must echo every
#: capability the driver advertises (V1), so both sides read this list.
KERNEL_CAPABILITIES: List[str] = [
    "family_a", "family_b", "family_c", "family_f", "family_g",
]

#: RFC-006 §2 Comm target name.  Mismatched majors fail at Comm-open
#: time; that is the version-rejection mechanism of the v2 supersession.
COMM_TARGET: str = "llmnb.rts.v2"

#: Distribution version of the ``llm_kernel`` package (``__version__``
#: and the Jupyter ``implementation_version``).  Distinct from
#: :data:`KERNEL_VERSION`, which is the protocol-level kernel version.
PACKAGE_VERSION: str = "0.1.0"

#: Default ``client_version`` the headless driver sends in its handshake.
CLIENT_VERSION: str = "0.1.0"

#: RFC-002 system-prompt template version the agent supervisor expects
#: to find embedded in the rendered template.  Not a wire version; it is
#: co-located here so that no semantic-version literal lives outside
#: this module.
SYSTEM_PROMPT_TEMPLATE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Live accessors.  All of these read the module globals at call time so a
# runtime override (``versions.KERNEL_VERSION = ...`` or an in-place
# ``versions.RFC_VERSIONS[...] = ...``) is observed by every consumer.
# ---------------------------------------------------------------------------


def semver_tuple(version: str) -> Tuple[int, int, int]:
    """Parse ``"MAJOR.MINOR.PATCH"`` into an int 3-tuple.

    Raises :class:`ValueError` on anything that is not strict semver
    (three dot-separated non-negative integers).
    """
    match = _SEMVER_RE.match(version) if isinstance(version, str) else None
    if match is None:
        raise ValueError(f"not a MAJOR.MINOR.PATCH version string: {version!r}")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def semver_major(version: str) -> int:
    """Return the MAJOR component of a semver string."""
    return semver_tuple(version)[0]


def wire_version_tuple() -> Tuple[int, int, int]:
    """Return ``(WIRE_MAJOR, WIRE_MINOR, WIRE_PATCH)`` derived from :data:`WIRE_VERSION`."""
    return semver_tuple(WIRE_VERSION)


def rfc_key(number: int) -> str:
    """Return the :data:`RFC_VERSIONS` key for RFC ``number`` (``rfc_003_version``)."""
    return f"rfc_{int(number):03d}_version"


def rfc_version(number: int) -> str:
    """Return the implementation version of RFC ``number`` (e.g. ``rfc_version(3)``)."""
    return RFC_VERSIONS[rfc_key(number)]


def rfc_versions() -> Dict[str, str]:
    """Return a fresh copy of :data:`RFC_VERSIONS` in definition order."""
    return dict(RFC_VERSIONS)


def kernel_volatile() -> Dict[str, str]:
    """Return the kernel's current ``config.volatile.kernel`` version fields.

    This is the current-environment side of the RFC-005 drift
    comparison: every RFC key this module defines plus
    ``kernel_version``.  A notebook that persisted exactly what the
    ready frame announced must compare drift-free against this.
    """
    out: Dict[str, str] = dict(RFC_VERSIONS)
    out["kernel_version"] = KERNEL_VERSION
    return out


def current_volatile() -> Dict[str, Any]:
    """Return the ``config.volatile`` shape the dispatcher hands the drift
    detector on hydrate: ``{"kernel": kernel_volatile()}``.

    Every start path (pty-mode, serve, the app, the Kernel facade, the
    legacy IPython kernel) registers this one function as the
    dispatcher's current-volatile provider, so what drift is measured
    against is the same object everywhere and is read at call time.
    """
    return {"kernel": kernel_volatile()}


def ready_frame_attrs() -> Dict[str, str]:
    """Return the version attributes of the RFC-008 §4 ready handshake.

    ``llmnb.kernel.version`` plus one ``llmnb.kernel.rfc_NNN_version``
    per entry in :data:`RFC_VERSIONS`.
    """
    attrs: Dict[str, str] = {"llmnb.kernel.version": KERNEL_VERSION}
    for key, value in RFC_VERSIONS.items():
        attrs[f"llmnb.kernel.{key}"] = value
    return attrs


def kernel_capabilities() -> List[str]:
    """Return a fresh copy of :data:`KERNEL_CAPABILITIES`."""
    return list(KERNEL_CAPABILITIES)

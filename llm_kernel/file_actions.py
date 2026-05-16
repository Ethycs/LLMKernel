"""llm_kernel.file_actions — kernel-side file I/O for cell-magic dispatch.

PLAN-S5.0.5 §3.1, §4.1, §7. The encode-side handler for the cell-magic
file encode/decode vocabulary. Pairs with the multi-format @@import
support already wired into ``magic_generators._handle_import``.

Per the kernel-speaks-magic design (PLAN-S5.0.5 §1 paragraph 3), file
I/O runs in-kernel — there is no operator-action envelope round-trip
to a driver-side handler. Every driver (extension, ``llmnb`` CLI,
future nvim plugin) gets ``@@export`` for free because the kernel
handles it.

Public API:

* :class:`ExportOutcome` — structured result; carries success metadata
  or K-class failure with a ``cause`` sub-code.
* :func:`apply_export` — validate path, infer/check format, atomic
  write via tmp + replace. Returns ``ExportOutcome``; never raises for
  expected error modes (returns ``status="error"`` instead) so the
  calling dispatcher can convert directly into a ``run.complete``
  envelope.

K-class surface (registered in ``wire/tools.py``):

* **K3M** ``notebook_export_path_outside_workspace`` — ``path`` resolves
  outside ``workspace_root`` (after ``..`` normalization and symlink
  resolution).
* **K3N** ``notebook_export_refused_overwrite`` — target exists and the
  operator did not pass ``overwrite:true``.
* **K3O** ``notebook_io_failed`` — write or serialize failure;
  ``cause`` sub-code disambiguates (``permission_denied``,
  ``disk_full``, ``unsupported_format``, ``parse_failed``,
  ``encoding_error``, ``invalid_input``, ``io_failed``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional

from . import notebook_format
from .wire.tools import (
    K3M_NOTEBOOK_EXPORT_PATH_OUTSIDE_WORKSPACE,
    K3N_NOTEBOOK_EXPORT_REFUSED_OVERWRITE,
    K3O_NOTEBOOK_IO_FAILED,
)


__all__ = [
    "ExportOutcome",
    "apply_export",
]


_SUPPORTED_FORMATS: tuple[str, ...] = ("llmnb", "magic", "ipynb")


@dataclass
class ExportOutcome:
    """Result of :func:`apply_export`.

    On ``status="ok"``: ``path`` is the resolved absolute target,
    ``format`` is the format actually written, ``cells_written`` counts
    the cells in the source notebook, ``warnings`` collects lossy-format
    advisories.

    On ``status="error"``: ``k_code`` is one of K3M / K3N / K3O;
    ``cause`` is the sub-code that disambiguates K3O; ``message`` is
    operator-readable.
    """

    status: Literal["ok", "error"]
    path: Optional[Path] = None
    format: Optional[str] = None
    cells_written: int = 0
    warnings: List[str] = field(default_factory=list)
    k_code: Optional[str] = None
    message: Optional[str] = None
    cause: Optional[str] = None


def _infer_format_from_extension(suffix: str) -> Optional[str]:
    """Map a path suffix to a format name. ``None`` if unrecognized."""
    suffix = suffix.lower()
    if suffix == ".llmnb":
        return "llmnb"
    if suffix == ".ipynb":
        return "ipynb"
    if suffix in (".magic", ".txt"):
        return "magic"
    return None


def _serialize(
    notebook_state: dict,
    format: str,
) -> tuple[str, List[str]]:
    """Render ``notebook_state`` (the ``metadata.rts`` snapshot) into
    on-disk content for ``format``. Returns ``(content, warnings)``.

    Raises any exception unchanged; the caller catches and converts to
    a K3O envelope. Warnings are advisory (lossy-format reminders).
    """
    warnings: List[str] = []

    # The format converters expect a top-level dict with
    # ``metadata.rts``; we always wrap the snapshot accordingly so all
    # three formats see a consistent input shape.
    wrapped = {
        "cells": [],
        "metadata": {"rts": notebook_state},
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    if format == "llmnb":
        return (
            json.dumps(wrapped, indent=2, ensure_ascii=False),
            warnings,
        )
    if format == "magic":
        warnings.append(
            "outputs dropped (magic format is operator-edit form; "
            "does not carry cell outputs)"
        )
        return notebook_format.llmnb_to_magic(wrapped), warnings
    if format == "ipynb":
        warnings.append(
            "provenance dropped (generated_by/generated_at are not part "
            "of the Jupyter ipynb schema)"
        )
        ipynb_data = notebook_format.llmnb_to_ipynb(wrapped)
        return (
            json.dumps(ipynb_data, indent=2, ensure_ascii=False),
            warnings,
        )
    raise ValueError(f"unsupported_format: {format!r}")


def _io_error_cause(exc: OSError) -> str:
    """Map an OSError to a K3O cause sub-code."""
    msg = str(exc).lower()
    if "permission" in msg or "denied" in msg:
        return "permission_denied"
    if "no space" in msg or "disk full" in msg or "ENOSPC" in str(exc):
        return "disk_full"
    return "io_failed"


def apply_export(
    cell_id: str,
    path: str,
    format: Optional[str],
    overwrite: bool,
    notebook_state: dict,
    workspace_root: Path,
) -> ExportOutcome:
    """Serialize ``notebook_state`` to ``path`` under ``workspace_root``.

    PLAN-S5.0.5 §4.1, §5.1, §7.

    Parameters
    ----------
    cell_id:
        The originating ``@@export`` cell id. Surfaced in the outcome
        for run.complete envelope correlation (callers stamp it).
    path:
        Operator-typed relative path; resolved against ``workspace_root``.
    format:
        Explicit format (``"llmnb" | "magic" | "ipynb"``); ``None``
        means "infer from extension".
    overwrite:
        When ``False`` (default) and ``path`` exists, fail with K3N.
    notebook_state:
        The current ``metadata.rts`` snapshot dict (the same shape a
        Family F snapshot carries).
    workspace_root:
        Resolved absolute path. Files must land at or beneath this
        directory; ``..`` traversal and symlink escapes return K3M.

    Returns
    -------
    ExportOutcome
        ``status="ok"`` on success with ``path`` / ``format`` /
        ``cells_written`` / ``warnings`` filled. ``status="error"``
        with ``k_code`` and ``message`` on any failure; never raises
        for documented error modes so the dispatcher can convert
        directly into a ``run.complete`` envelope.
    """
    # ---- Input validation ----
    if format is not None and format not in _SUPPORTED_FORMATS:
        return ExportOutcome(
            status="error",
            k_code=K3O_NOTEBOOK_IO_FAILED,
            message=(
                f"export_unsupported_format: {format!r} "
                f"(must be one of {', '.join(_SUPPORTED_FORMATS)})"
            ),
            cause="unsupported_format",
        )
    if not isinstance(path, str) or not path.strip():
        return ExportOutcome(
            status="error",
            k_code=K3O_NOTEBOOK_IO_FAILED,
            message="export_requires_path",
            cause="invalid_input",
        )
    if not isinstance(notebook_state, dict):
        return ExportOutcome(
            status="error",
            k_code=K3O_NOTEBOOK_IO_FAILED,
            message="export_notebook_state_not_dict",
            cause="invalid_input",
        )

    # ---- Path resolution + workspace containment (K3M) ----
    try:
        ws_root = Path(workspace_root).resolve()
    except OSError as exc:
        return ExportOutcome(
            status="error",
            k_code=K3O_NOTEBOOK_IO_FAILED,
            message=f"export_workspace_root_unresolvable: {exc}",
            cause=_io_error_cause(exc),
        )
    candidate = (ws_root / path).resolve()
    try:
        candidate.relative_to(ws_root)
    except ValueError:
        return ExportOutcome(
            status="error",
            k_code=K3M_NOTEBOOK_EXPORT_PATH_OUTSIDE_WORKSPACE,
            message=f"export_path_escapes_workspace: {path!r}",
            cause="path_outside_workspace",
        )

    # ---- Format inference ----
    if format is None:
        inferred = _infer_format_from_extension(candidate.suffix)
        if inferred is None:
            return ExportOutcome(
                status="error",
                k_code=K3O_NOTEBOOK_IO_FAILED,
                message=(
                    f"export_unknown_format_for_extension: "
                    f"{candidate.suffix!r} (pass format:"
                    "\"llmnb\"|\"magic\"|\"ipynb\")"
                ),
                cause="unsupported_format",
            )
        format = inferred

    # ---- Overwrite check (K3N) ----
    if candidate.exists() and not overwrite:
        return ExportOutcome(
            status="error",
            k_code=K3N_NOTEBOOK_EXPORT_REFUSED_OVERWRITE,
            message=(
                f"export_target_exists: {path!r} "
                "(pass overwrite:true to replace)"
            ),
            cause="exists_no_overwrite",
        )

    # ---- Serialize ----
    try:
        content, warnings = _serialize(notebook_state, format)
    except Exception as exc:  # noqa: BLE001 — surface as K3O
        return ExportOutcome(
            status="error",
            k_code=K3O_NOTEBOOK_IO_FAILED,
            message=f"export_serialize_failed: {exc}",
            cause="parse_failed",
        )

    # ---- Atomic write (tmp + replace) ----
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return ExportOutcome(
            status="error",
            k_code=K3O_NOTEBOOK_IO_FAILED,
            message=f"export_mkdir_failed: {exc}",
            cause=_io_error_cause(exc),
        )
    tmp_path = candidate.with_name(candidate.name + ".tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, candidate)
    except OSError as exc:
        # Best-effort cleanup of stray tmp file. Swallow any cleanup
        # error — the original write failure is what the operator
        # needs to see.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return ExportOutcome(
            status="error",
            k_code=K3O_NOTEBOOK_IO_FAILED,
            message=f"export_write_failed: {exc}",
            cause=_io_error_cause(exc),
        )
    except UnicodeEncodeError as exc:
        return ExportOutcome(
            status="error",
            k_code=K3O_NOTEBOOK_IO_FAILED,
            message=f"export_encoding_failed: {exc}",
            cause="encoding_error",
        )

    # ---- Count cells written (best-effort) ----
    cells = notebook_state.get("cells")
    cells_written = len(cells) if isinstance(cells, dict) else 0

    return ExportOutcome(
        status="ok",
        path=candidate,
        format=format,
        cells_written=cells_written,
        warnings=warnings,
    )

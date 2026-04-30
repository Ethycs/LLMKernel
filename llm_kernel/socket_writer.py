"""Thread-safe data-plane socket writer (RFC-008 §2).

Emits newline-delimited JSON frames over a Unix domain socket, a
Windows named pipe (deferred, see :func:`parse_address`), or a loopback
TCP socket. The single underlying byte stream gives us frame ordering
for free; the lock here exists so multi-threaded producers (run_tracker,
dispatcher, agent supervisor, the OTLP logging handler) never interleave
their JSON bodies mid-line.

Thread safety: :meth:`SocketWriter.write_frame`, :meth:`SocketWriter.connect`,
and :meth:`SocketWriter.close` are safe to call from any thread; they
serialize on a single :class:`threading.Lock`.

Address scheme (RFC-008 §2):

* ``unix:<path>`` -- POSIX UDS at ``<path>``. The ``unix:`` prefix is
  optional; an unprefixed value is treated as UDS.
* ``pipe:<name>`` -- Windows named pipe (full path, e.g.
  ``\\\\.\\pipe\\llmnb-<session>``). V1 returns ``NotImplementedError``;
  cross-platform support lands in I-T-W (see ``TODO`` below).
* ``tcp:<host>:<port>`` -- loopback TCP fallback. ``<host>`` MUST be
  ``127.0.0.1`` or ``::1``; the parser does not enforce that, but the
  extension MUST only listen on loopback per RFC-008 §2.

Framing: one JSON object per line, terminated by a single ``\\n``. The
framing test in ``tests/test_socket_writer.py`` exercises 1000 frames
across N producer threads to verify the receiver-side splitter recovers
the same N-frame multiset the producers emitted.
"""

from __future__ import annotations

import json
import logging
import socket
import sys
import threading
from typing import Any, Callable, Dict, Optional, Tuple

logger: logging.Logger = logging.getLogger("llm_kernel.socket_writer")


#: PLAN-S5.0.1 §3.3 — string fields inside outbound frames whose
#: payload may flow into a cell's outputs and therefore must be
#: scanned for the canonical hashed-magic line shape when hash mode
#: is enabled. The list is kept short (only fields the kernel emits
#: into cell-visible spans) so the per-frame walk is O(small) — not
#: a regex over every JSON property.
_OUTPUT_TEXT_FIELDS: Tuple[str, ...] = (
    "emit_content",
    "text",
    "observation",
    "preview",
    "message",
)

#: Address-prefix tokens that disambiguate the transport. Unprefixed
#: addresses are treated as UDS on POSIX and as ``pipe:`` on win32 only
#: when the value LOOKS like a Windows pipe path; otherwise UDS.
_PREFIX_UNIX: str = "unix:"
_PREFIX_PIPE: str = "pipe:"
_PREFIX_TCP: str = "tcp:"


class SocketWriter:
    """Newline-delimited JSON writer with a producer-side serialization lock.

    A single instance is shared across the kernel's threads. Construct
    once, :meth:`connect`, then call :meth:`write_frame` from any thread.
    :meth:`close` is idempotent.

    The writer never raises on a broken socket; instead it logs once and
    drops the frame. Callers MAY check :meth:`is_connected` to detect a
    dead transport, but the kernel main loop normally treats socket EOF
    as the shutdown signal (RFC-008 §4 step 6).
    """

    def __init__(self) -> None:
        """Construct an unconnected writer.

        Uses :class:`threading.RLock` because :meth:`write_frame`'s error
        path may invoke :func:`logging.warning`, which — when the root
        logger carries an :class:`OtlpDataPlaneHandler` whose backing
        writer is the same instance — re-enters this lock to emit the
        OTLP log record. A non-reentrant Lock would deadlock that path.
        """
        self._lock: threading.RLock = threading.RLock()
        self._sock: Optional[socket.socket] = None
        self._closed: bool = False
        self._address: Optional[str] = None
        # PLAN-S5.0.1 §3.3 — defense-in-depth output sanitizer. When
        # hash mode is on, every outbound frame's text fields are
        # scanned for the canonical hashed-magic line shape; matches
        # are escaped before transmission so the kernel can never
        # accidentally emit a valid hashed-magic into a cell's
        # outputs through any path.
        #
        # ``_hash_mode_provider`` is a zero-arg callable returning a
        # bool. When None (the V1 default until slice 5.0.1b wires
        # the writer's config field), the sanitizer no-ops — fully
        # backwards-compatible with pre-S5.0.1 callers.
        self._hash_mode_provider: Optional[Callable[[], bool]] = None

    # -- Connection lifecycle ---------------------------------------

    def connect(self, address: str) -> None:
        """Connect to ``address`` per RFC-008 §2.

        Idempotent on already-connected: subsequent ``connect`` calls
        with the same address are no-ops; mismatched addresses raise
        :class:`RuntimeError`.
        """
        with self._lock:
            if self._sock is not None:
                if self._address == address:
                    return
                raise RuntimeError(
                    f"SocketWriter already connected to {self._address!r}; "
                    f"cannot reconnect to {address!r}"
                )
            family, target = parse_address(address)
            sock = socket.socket(family, socket.SOCK_STREAM)
            # On Windows, socket.socket() returns an inheritable handle by
            # default. When the kernel later spawns Claude via subprocess.Popen
            # (AgentSupervisor.spawn), the child inherits a copy of this fd
            # despite close_fds=True (Windows close_fds covers regular handles
            # but socket fds need explicit set_inheritable(False)). The
            # inherited copy keeps the OS-level reference alive past the
            # child's startup, which can EOF the parent's recv when the child
            # does its own handle cleanup. Set non-inheritable here so the
            # child gets nothing.
            sock.set_inheritable(False)
            sock.connect(target)
            self._sock = sock
            self._address = address
            self._closed = False
            logger.debug("SocketWriter connected to %s", address)

    def close(self) -> None:
        """Close the socket cleanly. Idempotent."""
        with self._lock:
            if self._sock is None:
                self._closed = True
                return
            sock = self._sock
            self._sock = None
            self._closed = True
        try:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                # Already half-closed by the peer; not fatal.
                pass
            sock.close()
        except OSError:
            logger.debug("SocketWriter.close: OSError on shutdown/close", exc_info=True)
        else:
            logger.debug("SocketWriter closed cleanly")

    def is_connected(self) -> bool:
        """Return True iff the socket is open."""
        with self._lock:
            return self._sock is not None and not self._closed

    # -- Frame I/O ---------------------------------------------------

    def set_hash_mode_provider(
        self, provider: Optional[Callable[[], bool]],
    ) -> None:
        """Wire the hash-mode lookup for outbound sanitization.

        PLAN-S5.0.1 §3.3 — when hash mode is on, every outbound
        frame's text fields are scanned for the canonical
        hashed-magic line shape and escaped before transmission.
        The provider is a zero-arg callable so the writer doesn't
        depend on the metadata-writer module directly (avoids an
        import cycle); the kernel binds the callable at startup.

        Set to ``None`` to disable sanitization (test isolation).
        """
        self._hash_mode_provider = provider

    def _hash_mode_enabled(self) -> bool:
        """Best-effort hash-mode lookup. Defaults False on any error."""
        provider = self._hash_mode_provider
        if provider is None:
            return False
        try:
            return bool(provider())
        except Exception:  # pragma: no cover - defensive
            return False

    def write_frame(self, record: Dict[str, Any]) -> None:
        """Encode ``record`` as one newline-terminated JSON line and send it.

        The encode happens INSIDE the lock so two producers can't
        interleave bytes mid-frame. JSON encoding raises ``TypeError``
        on un-serializable values; we let that propagate so producers
        learn about bad payloads at emission time, not on the wire.

        PLAN-S5.0.1 §3.3 — defense-in-depth sanitization: when hash
        mode is on, the record's known text fields (per
        :data:`_OUTPUT_TEXT_FIELDS`) are scanned for the canonical
        hashed-magic line shape; matches are escaped (``@`` → ``\\@``)
        before encoding so the wire never carries a valid hashed-
        magic line into a cell's outputs through any kernel-emitted
        path.
        """
        if self._hash_mode_enabled():
            record = sanitize_outbound_record(record)
        # Encode first so a producer-side ``TypeError`` doesn't leave
        # the socket holding a partial frame. The encode itself is
        # CPU-bound and lock-free.
        try:
            line = json.dumps(record, separators=(",", ":"), default=_json_default)
        except (TypeError, ValueError):
            logger.exception(
                "SocketWriter.write_frame: JSON encode failed; frame dropped"
            )
            return
        payload = (line + "\n").encode("utf-8")
        with self._lock:
            sock = self._sock
            if sock is None:
                logger.debug("SocketWriter.write_frame: not connected; dropped")
                return
            try:
                sock.sendall(payload)
            except OSError as exc:
                logger.warning(
                    "SocketWriter.write_frame: sendall failed (%s); frame dropped",
                    exc,
                )

    # -- Context manager sugar --------------------------------------

    def __enter__(self) -> "SocketWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Address parsing
# ---------------------------------------------------------------------------


def parse_address(address: str) -> Tuple[int, Any]:
    """Parse an RFC-008 §2 address into ``(family, connect_target)``.

    Returns a tuple suitable for ``socket.socket(family).connect(target)``.

    * ``unix:<path>`` / unprefixed POSIX path -> ``(AF_UNIX, "<path>")``.
    * ``tcp:<host>:<port>`` -> ``(AF_INET, ("<host>", port))``.
    * ``pipe:<name>`` on Windows -> raises ``NotImplementedError`` in V1
      (TODO(I-T-W): Windows named pipes need ``win32pipe`` / ``pywin32``;
      see RFC-008 §2 platform binding table).

    Raises :class:`ValueError` on malformed addresses.
    """
    if not isinstance(address, str) or not address:
        raise ValueError(f"address must be a non-empty string; got {address!r}")

    if address.startswith(_PREFIX_TCP):
        host_port = address[len(_PREFIX_TCP):]
        if ":" not in host_port:
            raise ValueError(f"tcp address missing :port -- {address!r}")
        host, _, port_str = host_port.rpartition(":")
        try:
            port = int(port_str)
        except ValueError as exc:
            raise ValueError(
                f"tcp address has non-integer port -- {address!r}"
            ) from exc
        return socket.AF_INET, (host, port)

    if address.startswith(_PREFIX_PIPE):
        # TODO(I-T-W): Windows named pipes. The kernel's V1 dev surface
        # uses UDS on POSIX; cross-platform CI is queued. Surfacing the
        # branch lets callers exercise address parsing without a Windows
        # IPC backend installed.
        raise NotImplementedError(
            "Windows named-pipe transport not implemented in V1; "
            "use UDS or tcp:127.0.0.1:<port> for cross-platform dev"
        )

    # Plain UDS path with the optional ``unix:`` prefix.
    path = address[len(_PREFIX_UNIX):] if address.startswith(_PREFIX_UNIX) else address
    if not hasattr(socket, "AF_UNIX"):
        # Windows lacking AF_UNIX is the loud failure operators want.
        # The TCP fallback exists for exactly this case.
        raise NotImplementedError(
            f"AF_UNIX not available on platform {sys.platform!r}; "
            f"use tcp:127.0.0.1:<port> instead"
        )
    return socket.AF_UNIX, path


def _json_default(value: Any) -> Any:
    """Fallback for non-JSON-native types so encoding stays total.

    Bytes round-trip through ``latin-1`` (raw byte preservation); other
    types fall back to ``repr`` so the receiver still gets a string.
    """
    if isinstance(value, bytes):
        return value.decode("latin-1")
    return repr(value)


# ---------------------------------------------------------------------------
# PLAN-S5.0.1 §3.3 — outbound sanitizer
# ---------------------------------------------------------------------------


def sanitize_outbound_line(line: str) -> str:
    """Escape a single line if it matches the canonical hashed-magic shape.

    Helper for callers that already operate at the line level
    (renderers, ContextPacker, kernel-emitted notify spans). Idempotent
    on already-escaped or non-matching lines.

    See :func:`magic_hash.escape_leading_at` for the escape semantics.
    """
    # Lazy import: socket_writer otherwise has no magic_hash dep.
    from .magic_hash import escape_leading_at, looks_like_hashed_magic

    if not isinstance(line, str) or not line:
        return line
    if looks_like_hashed_magic(line):
        return escape_leading_at(line)
    return line


def sanitize_outbound_text(text: str) -> str:
    """Apply :func:`sanitize_outbound_line` to every line of ``text``.

    Preserves the trailing newline if the original had one.
    Multiline-safe; idempotent.
    """
    if not isinstance(text, str) or not text:
        return text
    if "@" not in text:
        return text  # fast path — no candidate lines
    lines = text.splitlines()
    rewritten = [sanitize_outbound_line(L) for L in lines]
    if rewritten == lines:
        return text  # identity short-circuit
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(rewritten) + suffix


def sanitize_outbound_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Walk a frame ``record`` and sanitize known text fields.

    Walks the record dict (and nested ``metadata`` / ``attributes`` /
    ``data`` sub-dicts when present) looking for keys in
    :data:`_OUTPUT_TEXT_FIELDS`. String values are run through
    :func:`sanitize_outbound_text`; non-string values are passed
    through unchanged.

    Returns a new dict on rewrite, or the input dict on identity
    short-circuit. The function is shallow-copy at the top level
    when a rewrite happens, deep enough to not mutate the caller's
    data.
    """
    if not isinstance(record, dict):
        return record
    out = record  # may rebind to a shallow copy below
    rewrote = False
    for key, value in record.items():
        if key in _OUTPUT_TEXT_FIELDS and isinstance(value, str):
            new_value = sanitize_outbound_text(value)
            if new_value is not value:
                if not rewrote:
                    out = dict(record)
                    rewrote = True
                out[key] = new_value
        elif isinstance(value, dict):
            new_value = sanitize_outbound_record(value)
            if new_value is not value:
                if not rewrote:
                    out = dict(record)
                    rewrote = True
                out[key] = new_value
    return out


__all__ = [
    "SocketWriter", "parse_address",
    "sanitize_outbound_line", "sanitize_outbound_text",
    "sanitize_outbound_record",
]

"""Cross-machine path resolution helpers (PR 2B coherence-pass 2e).

Real labs share batch sessions across machines: a researcher saves a
session on their lab desktop, emails the JSON to a collaborator, and
the collaborator opens it on their own computer. The session JSON's
absolute paths (``D:\\labs\\study1\\videos\\...``) won't exist on the
collaborator's machine, so every video shows up missing.

This module fixes that by introducing a **project root** convention:

* When a session is saved, paths that sit *under* the same parent
  folder as the session file are stored **relative** to that folder.
* When the session is loaded, those relative paths are resolved
  against the session file's location on the new machine.

A session JSON plus its referenced data folder, kept together, becomes
portable. Paths that point *outside* the session folder (e.g., a model
file kept in ``~/models/``) stay absolute — same behavior as before.

Public API:

    to_storage_path(absolute_path, project_root) -> str
    resolve_storage_path(stored_value, project_root) -> str | None
    is_unc_path(path) -> bool
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _normalise_string_path(value: str) -> str:
    """Best-effort tidy of a path string (no I/O)."""
    return str(value or "").strip()


def to_storage_path(absolute_path: str, project_root: Path) -> str:
    """Convert an absolute path into the form to store in a session JSON.

    Returns:
        A POSIX-style relative path string when ``absolute_path`` is under
        ``project_root``; the original absolute path string otherwise. Empty
        / unset inputs round-trip as empty strings.

    The relative form uses POSIX separators (``"/"``) so the JSON is
    portable across Windows and POSIX without escaping.
    """
    text = _normalise_string_path(absolute_path)
    if not text:
        return ""
    try:
        candidate = Path(text)
    except (TypeError, ValueError):
        return text
    # Cannot resolve unresolvable inputs — bail out, store as-is.
    try:
        candidate_resolved = candidate.resolve(strict=False)
        root_resolved = project_root.resolve(strict=False)
    except Exception:
        return text
    try:
        relative = candidate_resolved.relative_to(root_resolved)
    except ValueError:
        # Not under the project root → keep absolute.
        return str(candidate_resolved)
    # POSIX-style separators for cross-platform portability.
    return relative.as_posix()


def resolve_storage_path(stored_value: str, project_root: Path) -> str:
    """Convert a stored path back to an absolute path string.

    Args:
        stored_value: The string read from the session JSON. May be empty,
            an absolute path, or a relative path (in which case it is
            anchored to ``project_root``).
        project_root: Folder on the *current* machine that should anchor
            relative paths — typically ``Path(session_file).parent``.

    Returns:
        An absolute path string. Empty input → empty output.

    Notes:
        * The returned string is *not* guaranteed to exist. Callers are
          responsible for missing-file checks; this helper only resolves
          the textual form.
        * UNC paths and absolute paths pass through unchanged.
    """
    text = _normalise_string_path(stored_value)
    if not text:
        return ""
    try:
        candidate = Path(text)
    except (TypeError, ValueError):
        return text
    if candidate.is_absolute():
        # Already absolute — normalise but don't re-anchor.
        try:
            return str(candidate.resolve(strict=False))
        except Exception:
            return text
    # Relative: anchor to project root.
    try:
        anchored = (project_root / candidate).resolve(strict=False)
        return str(anchored)
    except Exception:
        return str(project_root / candidate)


def is_unc_path(path: str) -> bool:
    """True if ``path`` is a Windows UNC path (``\\server\\share\\...``).

    Used by the wizard to surface a slow-network warning at run start.
    Non-Windows callers can still call this safely — POSIX paths return
    ``False``.
    """
    text = _normalise_string_path(path)
    if not text:
        return False
    if os.name != "nt":
        return False
    # Windows UNC paths start with two backslashes (or two forward
    # slashes after pathlib normalisation).
    return text.startswith("\\\\") or text.startswith("//")


def find_missing_paths(
    candidates: dict[str, str],
    *,
    project_root: Optional[Path] = None,
) -> list[tuple[str, str]]:
    """Filter ``candidates`` to entries whose *resolved* path does not exist.

    Args:
        candidates: ``{field_name: stored_path_value}`` — typically a
            subset of the session payload's path-bearing fields.
        project_root: When provided, relative paths are anchored here
            before the existence check. ``None`` means treat the values
            as already absolute.

    Returns:
        A list of ``(field_name, resolved_absolute_path)`` for entries
        that would not be findable on disk. Empty values in ``candidates``
        are skipped (they're not "missing" in the actionable sense).
    """
    out: list[tuple[str, str]] = []
    for name, value in candidates.items():
        text = _normalise_string_path(value)
        if not text:
            continue
        if project_root is not None:
            resolved = resolve_storage_path(text, project_root)
        else:
            resolved = text
        try:
            exists = Path(resolved).exists() if resolved else False
        except Exception:
            exists = False
        if not exists:
            out.append((name, resolved))
    return out


__all__ = [
    "to_storage_path",
    "resolve_storage_path",
    "is_unc_path",
    "find_missing_paths",
]

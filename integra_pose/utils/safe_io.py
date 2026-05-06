"""Safe I/O helpers for project / session / manifest files.

Two fragility classes that recur across the codebase:

1. **Encoding.** ``open(path, 'r')`` on Windows uses the locale codepage
   (often CP1252). A project file containing non-ASCII characters
   (German umlauts, accented filenames) round-trips wrong across machines.
   These helpers always pass ``encoding="utf-8"``.

2. **Atomicity.** ``Path.write_text`` and ``json.dump`` truncate-and-write the
   destination directly. A crash, GPU driver kill, or laptop sleep mid-write
   leaves a truncated file that fails to parse on next load — destroying the
   user's project / batch session / repro manifest. These helpers write to a
   sibling ``<name>.tmp`` and then ``os.replace`` it onto the target, which is
   atomic on POSIX and on Windows for same-volume replaces.

Use ``safe_write_json``, ``safe_read_json``, ``safe_write_text``,
``safe_read_text`` everywhere we previously used the raw stdlib calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def _resolve(path: PathLike) -> Path:
    return Path(path).expanduser()


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    # On Windows, an existing tmp file from a previous crashed write would
    # block creation. Remove it best-effort before opening fresh.
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except (OSError, AttributeError):
            # fsync is best-effort; some filesystems / Windows corner cases
            # don't support it. The os.replace below is still atomic.
            pass
    os.replace(tmp, target)


def safe_write_text(path: PathLike, text: str) -> Path:
    """Atomically write ``text`` to ``path`` as UTF-8."""
    target = _resolve(path)
    _atomic_write_bytes(target, text.encode("utf-8"))
    return target


def safe_read_text(path: PathLike) -> str:
    """Read ``path`` as UTF-8 text."""
    return _resolve(path).read_text(encoding="utf-8")


def safe_write_json(path: PathLike, data: Any, *, indent: int = 2, sort_keys: bool = False) -> Path:
    """Atomically write ``data`` as JSON to ``path`` (UTF-8, indented)."""
    payload = json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
    return safe_write_text(path, payload)


def safe_read_json(path: PathLike) -> Any:
    """Read ``path`` as UTF-8 JSON and return the parsed payload."""
    return json.loads(safe_read_text(path))

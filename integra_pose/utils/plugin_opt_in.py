"""Persistent opt-in/enablement state for optional plugins.

IntegraPose treats plugins as opt-in features. This module stores whether the user has
explicitly enabled a plugin and ties that consent to a lightweight fingerprint so
updated plugin code requires re-consent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_OPT_IN_STORE_PATH = Path.home() / ".integrapose" / "plugins" / "enabled_plugins.json"
_FINGERPRINT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml"}
_FINGERPRINT_SKIP_DIRS = {"__pycache__", ".git", ".hg", ".svn", ".pytest_cache", ".mypy_cache"}
_MAX_FINGERPRINT_FILE_BYTES = 512 * 1024


@dataclass(frozen=True)
class PluginOptInRecord:
    state: str  # "enabled" | "disabled"
    fingerprint: str
    name: str
    version: str
    source: str
    updated_at: str


def plugin_key(plugin_path: Path) -> str:
    return str(plugin_path.resolve())


def compute_fingerprint(plugin_path: Path) -> Optional[str]:
    """Hash the plugin's importable/config surface.

    The goal is to require re-opt-in when plugin code or runtime configuration
    changes, without scanning large binary artifacts such as datasets or models.
    """
    hasher = hashlib.sha256()
    included = 0
    for path in _iter_fingerprint_files(plugin_path):
        try:
            rel = path.relative_to(plugin_path).as_posix()
            hasher.update(rel.encode("utf-8"))
            hasher.update(b"\x00")
            hasher.update(path.read_bytes())
            hasher.update(b"\x00")
            included += 1
        except Exception:
            continue
    return hasher.hexdigest() if included else None


def _iter_fingerprint_files(plugin_path: Path) -> Iterable[Path]:
    candidates: list[Path] = []
    try:
        iterator = plugin_path.rglob("*")
    except Exception:
        return candidates

    for path in iterator:
        try:
            if not path.is_file():
                continue
            if any(part in _FINGERPRINT_SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() not in _FINGERPRINT_SUFFIXES:
                continue
            if path.stat().st_size > _MAX_FINGERPRINT_FILE_BYTES:
                continue
        except Exception:
            continue
        candidates.append(path)

    candidates.sort(key=lambda candidate: candidate.relative_to(plugin_path).as_posix().lower())
    return candidates


def is_plugin_enabled(plugin_path: Path) -> bool:
    record = get_record(plugin_path)
    if record is None:
        return False
    if record.state != "enabled":
        return False
    current = compute_fingerprint(plugin_path)
    return bool(current) and current == record.fingerprint


def get_record(plugin_path: Path) -> Optional[PluginOptInRecord]:
    store = _load_store()
    key = plugin_key(plugin_path)
    raw = store.get(key)
    if not isinstance(raw, dict):
        return None

    state = str(raw.get("state", "disabled"))
    fingerprint = str(raw.get("fingerprint", ""))
    name = str(raw.get("name", plugin_path.name))
    version = str(raw.get("version", ""))
    source = str(raw.get("source", ""))
    updated_at = str(raw.get("updated_at", ""))

    if state not in {"enabled", "disabled"}:
        return None
    return PluginOptInRecord(
        state=state,
        fingerprint=fingerprint,
        name=name,
        version=version,
        source=source,
        updated_at=updated_at,
    )


def set_plugin_enabled(
    plugin_path: Path,
    enabled: bool,
    *,
    name: Optional[str] = None,
    version: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    fp = compute_fingerprint(plugin_path) or ""
    timestamp = datetime.now(timezone.utc).isoformat()

    store = _load_store()
    key = plugin_key(plugin_path)
    previous = store.get(key) if isinstance(store.get(key), dict) else {}

    record: Dict[str, Any] = {
        "state": "enabled" if enabled else "disabled",
        "fingerprint": fp or str(previous.get("fingerprint", "")),
        "name": name or str(previous.get("name", plugin_path.name)),
        "version": version or str(previous.get("version", "")),
        "source": source or str(previous.get("source", "")),
        "updated_at": timestamp,
    }

    store[key] = record
    _save_store(store)


def remove_plugin_record(plugin_path: Path) -> None:
    store = _load_store()
    key = plugin_key(plugin_path)
    if key in store:
        store.pop(key, None)
        _save_store(store)


def _load_store() -> Dict[str, Dict[str, Any]]:
    try:
        if _OPT_IN_STORE_PATH.is_file():
            raw = _OPT_IN_STORE_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                out: Dict[str, Dict[str, Any]] = {}
                for k, v in data.items():
                    if isinstance(k, str) and isinstance(v, dict):
                        out[k] = dict(v)
                return out
    except Exception:
        return {}
    return {}


def _save_store(data: Dict[str, Dict[str, Any]]) -> None:
    try:
        _OPT_IN_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _OPT_IN_STORE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        # Avoid blocking GUI if the filesystem is read-only or unavailable.
        pass

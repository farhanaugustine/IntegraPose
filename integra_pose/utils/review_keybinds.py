"""Keybind profile helpers for batch bout review."""

from __future__ import annotations

from pathlib import Path
import json


DEFAULT_KEYBINDS = {
    "confirm_bout": "a",
    "reject_bout": "r",
    "next_bout": "k",
    "prev_bout": "j",
    "next_unreviewed": "]",
    "prev_unreviewed": "[",
    "toggle_play_pause": "space",
    "step_frame_forward": ".",
    "step_frame_back": ",",
    "save_review_state": "Control-s",
}

SCHEMA_VERSION = 1


def normalize_keybind_token(token: str) -> str:
    raw = str(token or "").strip()
    if not raw:
        return ""
    alias = {
        "ctrl+s": "Control-s",
        "ctrl-a": "Control-a",
        "ctrl+r": "Control-r",
        "spacebar": "space",
        "space": "space",
    }
    key = raw.lower()
    if key in alias:
        return alias[key]
    return raw


def validate_keybinds(bindings: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    resolved = dict(DEFAULT_KEYBINDS)
    warnings: list[str] = []
    if isinstance(bindings, dict):
        for action, token in bindings.items():
            action_name = str(action or "").strip()
            if not action_name:
                continue
            normalized = normalize_keybind_token(str(token or ""))
            if not normalized:
                continue
            resolved[action_name] = normalized

    seen: dict[str, str] = {}
    for action, token in resolved.items():
        prev = seen.get(token)
        if prev is not None and prev != action:
            warnings.append(f"Key '{token}' is assigned to both '{prev}' and '{action}'.")
        else:
            seen[token] = action

    return resolved, warnings


def load_keybind_profile(path: str | Path) -> tuple[dict[str, str], list[str], str]:
    profile_path = Path(path).expanduser().resolve()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Keybind profile must be a JSON object.")

    schema_version = int(payload.get("schema_version", SCHEMA_VERSION) or SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported keybind schema version: {schema_version}")

    profile_name = str(payload.get("profile_name", profile_path.stem)).strip() or profile_path.stem
    bindings_raw = payload.get("bindings") if isinstance(payload.get("bindings"), dict) else {}
    resolved, warnings = validate_keybinds(bindings_raw)
    return resolved, warnings, profile_name


def save_keybind_profile(path: str | Path, profile_name: str, bindings: dict[str, str]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved, _warnings = validate_keybinds(bindings)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "profile_name": str(profile_name or "").strip() or target.stem,
        "bindings": resolved,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


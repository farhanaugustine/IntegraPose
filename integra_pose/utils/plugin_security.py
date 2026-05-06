"""Utility helpers for enforcing safer plugin discovery."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

_TRUST_STORE_PATH = Path.home() / ".integrapose" / "security" / "trusted_plugins.json"
_BUNDLED_PLUGINS_ROOT = Path(__file__).resolve().parents[1] / "plugins"
# Built-in plugins that ship with IntegraPose and are trusted implicitly.
_DEFAULT_TRUSTED = {
    "plugin_autolabel_forge",
    "plugin_dataset_augmentor_lab",
    "plugin_eda",
    "plugin_gait_kinematics",
    "plugin_tandem_yolo_toolkit",
    "plugin_behavior_scope",
    "plugin_zone_counter",
}
_MANIFEST_FILENAME = "plugin.json"


@dataclass(frozen=True)
class PluginMetadata:
    """Metadata surfaced to the trust prompt."""

    name: str
    version: Optional[str]
    description: Optional[str]
    author: Optional[str]
    path: Path


@dataclass(frozen=True)
class PluginTrustDecision:
    """Represents the outcome of a trust evaluation."""

    allowed: bool
    level: str
    message: str


PromptFn = Callable[[PluginMetadata], bool]

def is_bundled_plugin(plugin_name: str) -> bool:
    """Return True when a plugin ships with IntegraPose and is trusted implicitly."""
    return plugin_name in _DEFAULT_TRUSTED


def is_bundled_plugin_path(plugin_path: Path) -> bool:
    """Return True only for plugins that resolve inside the installed bundled plugins directory."""

    if plugin_path.name not in _DEFAULT_TRUSTED:
        return False
    try:
        resolved_path = plugin_path.resolve()
        bundled_root = _BUNDLED_PLUGINS_ROOT.resolve()
    except Exception:
        return False
    return resolved_path.parent == bundled_root


def get_trust_state(plugin_path: Path) -> Optional[str]:
    """Return the persisted trust decision for a plugin path.

    Returns:
        "allow", "deny", or None when no decision is stored.
    """
    if is_bundled_plugin_path(plugin_path):
        return "allow"
    trust_store = _load_trust_store()
    return trust_store.get(str(plugin_path.resolve()))


def clear_trust_state(plugin_path: Path) -> bool:
    """Remove a stored allow/deny decision for a third-party plugin."""
    if is_bundled_plugin_path(plugin_path):
        return False
    trust_store = _load_trust_store()
    key = str(plugin_path.resolve())
    if key not in trust_store:
        return False
    del trust_store[key]
    _save_trust_store(trust_store)
    return True


def get_plugin_metadata(plugin_path: Path) -> PluginMetadata:
    """Load metadata from plugin.json when available, otherwise fall back to defaults."""
    manifest, _manifest_path = _load_manifest(plugin_path)
    if manifest is not None:
        return _manifest_to_metadata(plugin_path, manifest)
    return PluginMetadata(
        name=plugin_path.name,
        version=None,
        description=None,
        author=None,
        path=plugin_path,
    )


def ensure_plugin_trusted(plugin_path: Path, prompt: Optional[PromptFn]) -> PluginTrustDecision:
    """
    Decide whether a plugin directory should be imported.

    Args:
        plugin_path: Directory that potentially contains a plugin.
        prompt: Callable used to confirm trust for third-party plugins. If None and
            trust has not already been granted, the plugin will be skipped.
    """
    plugin_name = plugin_path.name
    if is_bundled_plugin_path(plugin_path):
        return PluginTrustDecision(
            allowed=True,
            level="DEBUG",
            message=f"Trusted bundled plugin detected: {plugin_name}",
        )

    manifest, manifest_path = _load_manifest(plugin_path)
    if manifest is None:
        return PluginTrustDecision(
            allowed=False,
            level="WARNING",
            message=(
                f"Skipping plugin '{plugin_name}': "
                f"{'invalid' if manifest_path is not None else 'missing'} {_MANIFEST_FILENAME}. "
                "Third-party plugins must declare metadata before they can be enabled."
            ),
        )

    metadata = _manifest_to_metadata(plugin_path, manifest)
    trust_store = _load_trust_store()
    key = str(plugin_path.resolve())
    existing = trust_store.get(key)
    reprompting_after_denial = existing == "deny"
    if existing == "allow":
        return PluginTrustDecision(
            allowed=True,
            level="DEBUG",
            message=f"Previously trusted plugin '{metadata.name}' at {plugin_path}",
        )
    if existing == "deny" and prompt is None:
        return PluginTrustDecision(
            allowed=False,
            level="WARNING",
            message=f"Skipping plugin '{metadata.name}' (path: {plugin_path}); trust was denied earlier.",
        )

    if prompt is None:
        return PluginTrustDecision(
            allowed=False,
            level="WARNING",
            message=(
                f"Skipping plugin '{metadata.name}' at {plugin_path} because interactive trust "
                "confirmation is unavailable in this session."
            ),
        )

    try:
        should_trust = prompt(metadata)
    except Exception as exc:  # pragma: no cover - Tk callbacks
        return PluginTrustDecision(
            allowed=False,
            level="ERROR",
            message=f"Trust prompt failed for plugin '{metadata.name}': {exc}",
        )

    _update_trust_store(key, should_trust)
    if should_trust:
        message = f"Plugin '{metadata.name}' trusted by user confirmation."
        if reprompting_after_denial:
            message = f"Plugin '{metadata.name}' trust denial was replaced by user confirmation."
        return PluginTrustDecision(
            allowed=True,
            level="INFO",
            message=message,
        )
    return PluginTrustDecision(
        allowed=False,
        level="WARNING",
        message=f"Plugin '{metadata.name}' was declined by the user and will not be loaded.",
    )


def _load_manifest(plugin_path: Path) -> tuple[Optional[Dict[str, object]], Optional[Path]]:
    manifest_path = plugin_path / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None, None

    try:
        raw = manifest_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Manifest must be a JSON object.")
        return data, manifest_path
    except Exception:
        return None, manifest_path


def _manifest_to_metadata(plugin_path: Path, manifest: Dict[str, object]) -> PluginMetadata:
    def _safe_get(key: str) -> Optional[str]:
        value = manifest.get(key)
        return str(value) if isinstance(value, (str, int, float)) else None

    return PluginMetadata(
        name=_safe_get("name") or plugin_path.name,
        version=_safe_get("version"),
        description=_safe_get("description"),
        author=_safe_get("author"),
        path=plugin_path,
    )


def _load_trust_store() -> Dict[str, str]:
    try:
        if _TRUST_STORE_PATH.is_file():
            raw = _TRUST_STORE_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items() if v in {"allow", "deny"}}
    except Exception:
        return {}
    return {}


def _update_trust_store(plugin_key: str, allowed: bool) -> None:
    trust_store = _load_trust_store()
    trust_store[plugin_key] = "allow" if allowed else "deny"
    _save_trust_store(trust_store)


def _save_trust_store(data: Dict[str, str]) -> None:
    try:
        _TRUST_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TRUST_STORE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        # Last-resort fallback: silently ignore persistence issues to avoid blocking the UI.
        pass

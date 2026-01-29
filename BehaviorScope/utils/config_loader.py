from __future__ import annotations

import argparse
import yaml
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens a nested config (e.g. {'training': {'epochs': 50}}) into 
    a single level dict ({'epochs': 50}) for argparse defaults.
    """
    flat_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat_config.update(value)
        else:
            flat_config[key] = value
    return flat_config

def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parser_actions(parser: argparse.ArgumentParser) -> Dict[str, argparse.Action]:
    actions: Dict[str, argparse.Action] = {}
    for action in parser._actions:
        actions[action.dest] = action
    return actions


def _coerce_action_value(action: argparse.Action, value: object) -> object:
    if value is None:
        return None

    # Flags (store_true/store_false) have no explicit type; accept bool-ish inputs.
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):  # type: ignore[attr-defined]
        return _coerce_bool(value)

    action_type = getattr(action, "type", None)
    if action_type is None:
        return value

    if isinstance(value, list) and getattr(action, "nargs", None) not in (None, 0, 1):
        return [action_type(v) for v in value]

    return action_type(value)


def apply_config_as_parser_defaults(
    parser: argparse.ArgumentParser, config: Mapping[str, Any]
) -> None:
    """
    Applies config values as parser defaults, coercing types via argparse Actions.

    Only keys that correspond to parser destinations are applied; other config
    fields are ignored so multi-tool configs (GUI/preprocessing/inference) can
    coexist without polluting unrelated CLIs.
    """
    flat_config = flatten_config(dict(config))
    # Backward/GUI-friendly aliases.
    if "no_pretrained_backbone" not in flat_config:
        if "use_pretrained" in flat_config:
            try:
                flat_config["no_pretrained_backbone"] = not _coerce_bool(
                    flat_config["use_pretrained"]
                )
            except Exception:
                pass
        elif "pretrained_backbone" in flat_config:
            try:
                flat_config["no_pretrained_backbone"] = not _coerce_bool(
                    flat_config["pretrained_backbone"]
                )
            except Exception:
                pass
    actions = _parser_actions(parser)

    typed_defaults: Dict[str, object] = {}
    for key, raw_value in flat_config.items():
        action = actions.get(key)
        if action is None:
            continue
        try:
            typed_defaults[key] = _coerce_action_value(action, raw_value)
        except Exception as exc:
            raise ValueError(f"Invalid config value for '{key}': {raw_value!r}") from exc

    parser.set_defaults(**typed_defaults)

def merge_args_with_config(args: Any, config: Dict[str, Any]) -> Any:
    """
    Updates argparse Namespace with values from the config dictionary.
    Warning: This overwrites args with config values regardless of whether
    the arg was explicitly set on CLI. Prefer using parser.set_defaults()
    strategy instead.
    """
    flat_config = flatten_config(config)

    # Update args
    for key, value in flat_config.items():
        if hasattr(args, key):
            setattr(args, key, value)
            
    return args

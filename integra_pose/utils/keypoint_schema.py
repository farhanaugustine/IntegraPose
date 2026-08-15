"""Resolve and validate pose-keypoint names for inference outputs.

Ultralytics checkpoints reliably expose the number of keypoints, but older
checkpoints do not always retain their names.  This module keeps schema
resolution deterministic: embedded metadata wins, then an explicitly
referenced dataset YAML, then ``dataset.yaml``/``data.yaml`` beside the
checkpoint.  A candidate is accepted only when it exactly matches the model's
keypoint count and produces unique CSV column tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(slots=True)
class KeypointSchemaResolution:
    names: list[str] = field(default_factory=list)
    source: str = "unresolved"
    source_path: str = ""
    warnings: list[str] = field(default_factory=list)


def safe_keypoint_token(value: str) -> str:
    """Return the token used for keypoint columns in ``labels.csv``."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    token = "".join(ch if ch in allowed else "_" for ch in str(value))
    return token.strip("_") or "kp"


def coerce_keypoint_names(value: Any) -> list[str]:
    """Normalize common Ultralytics/YAML keypoint-name containers."""
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, Mapping):
        return []

    # Single-class pose datasets may store ``{0: [nose, ...]}``.
    nested = [item for item in value.values() if isinstance(item, (list, tuple))]
    if nested:
        first = coerce_keypoint_names(nested[0])
        if first and all(coerce_keypoint_names(item) == first for item in nested[1:]):
            return first
        return []

    ordered: list[tuple[int | None, str]] = []
    for raw_index, raw_name in value.items():
        try:
            index: int | None = int(raw_index)
        except (TypeError, ValueError):
            index = None
        name = str(raw_name).strip()
        if name:
            ordered.append((index, name))
    ordered.sort(key=lambda row: (row[0] is None, row[0] if row[0] is not None else 0))
    return [name for _index, name in ordered]


def validate_keypoint_names(names: Iterable[str], expected_count: int) -> tuple[list[str], str]:
    """Validate an ordered schema, returning ``(clean_names, error)``."""
    clean = [str(name).strip() for name in names if str(name).strip()]
    expected = max(0, int(expected_count or 0))
    if len(clean) != expected:
        return clean, f"expected {expected} names, found {len(clean)}"
    folded = [name.casefold() for name in clean]
    if len(set(folded)) != len(folded):
        return clean, "keypoint names are not unique"
    tokens = [safe_keypoint_token(name).casefold() for name in clean]
    if len(set(tokens)) != len(tokens):
        return clean, "keypoint names collide after conversion to CSV column tokens"
    return clean, ""


def _mapping_candidates(model: Any) -> list[tuple[str, Mapping[str, Any]]]:
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for label, value in (
        ("model_overrides", getattr(model, "overrides", None)),
        ("model_yaml", getattr(getattr(model, "model", None), "yaml", None)),
    ):
        if isinstance(value, Mapping):
            candidates.append((label, value))
    checkpoint = getattr(model, "ckpt", None)
    if isinstance(checkpoint, Mapping):
        candidates.append(("checkpoint", checkpoint))
        train_args = checkpoint.get("train_args")
        if isinstance(train_args, Mapping):
            candidates.append(("checkpoint_train_args", train_args))
    return candidates


def _names_from_mapping(mapping: Mapping[str, Any]) -> list[str]:
    for key in ("kpt_names", "keypoint_names"):
        names = coerce_keypoint_names(mapping.get(key))
        if names:
            return names
    return []


def _candidate_dataset_paths(model: Any, model_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for _label, mapping in _mapping_candidates(model):
        raw_data = mapping.get("data")
        if not isinstance(raw_data, (str, Path)) or not str(raw_data).strip():
            continue
        candidate = Path(str(raw_data)).expanduser()
        if not candidate.is_absolute():
            candidate = model_path.parent / candidate
        candidates.append(candidate)
    candidates.extend((model_path.parent / "dataset.yaml", model_path.parent / "data.yaml"))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        token = str(resolved).casefold()
        if token not in seen:
            seen.add(token)
            unique.append(resolved)
    return unique


def resolve_model_keypoint_schema(
    model: Any,
    model_path: str | Path,
    expected_count: int,
) -> KeypointSchemaResolution:
    """Resolve the ordered keypoint schema associated with a pose model."""
    expected = max(0, int(expected_count or 0))
    if expected <= 0:
        return KeypointSchemaResolution()

    warnings: list[str] = []
    embedded_candidates: list[tuple[str, Any]] = [
        ("model_attribute", getattr(model, "kpt_names", None)),
        ("model_core_attribute", getattr(getattr(model, "model", None), "kpt_names", None)),
    ]
    embedded_candidates.extend(
        (label, _names_from_mapping(mapping)) for label, mapping in _mapping_candidates(model)
    )
    for source, value in embedded_candidates:
        names = coerce_keypoint_names(value)
        if not names:
            continue
        clean, error = validate_keypoint_names(names, expected)
        if not error:
            return KeypointSchemaResolution(names=clean, source=source, warnings=warnings)
        warnings.append(f"Ignored {source} keypoint names: {error}.")

    checkpoint_path = Path(model_path).expanduser().resolve()
    for candidate in _candidate_dataset_paths(model, checkpoint_path):
        if not candidate.is_file():
            continue
        try:
            import yaml

            payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            warnings.append(f"Could not read keypoint schema from {candidate}: {exc}")
            continue
        if not isinstance(payload, Mapping):
            warnings.append(f"Ignored {candidate}: YAML root is not a mapping.")
            continue
        names = _names_from_mapping(payload)
        if not names:
            continue
        clean, error = validate_keypoint_names(names, expected)
        if error:
            warnings.append(f"Ignored keypoint names in {candidate}: {error}.")
            continue
        return KeypointSchemaResolution(
            names=clean,
            source="dataset_yaml",
            source_path=str(candidate),
            warnings=warnings,
        )

    warnings.append(
        f"The model emits {expected} keypoints, but no validated keypoint-name schema was found. "
        "Generic kp0... names will be used unless the GUI contains exactly the same number of names."
    )
    return KeypointSchemaResolution(warnings=warnings)


__all__ = [
    "KeypointSchemaResolution",
    "coerce_keypoint_names",
    "resolve_model_keypoint_schema",
    "safe_keypoint_token",
    "validate_keypoint_names",
]

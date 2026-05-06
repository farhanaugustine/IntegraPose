"""Helpers for loading model artifacts with secure defaults."""

from __future__ import annotations

import inspect
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

UNSAFE_MODEL_LOAD_ENV = "INTEGRAPOSE_ALLOW_UNSAFE_MODEL_LOAD"


def _unsafe_model_load_allowed() -> bool:
    raw = os.environ.get(UNSAFE_MODEL_LOAD_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _torch_load_supports_weights_only() -> bool:
    try:
        params = inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        return False
    return "weights_only" in params


def load_torch_artifact(
    path: str | Path,
    *,
    map_location: Any = None,
    description: str = "PyTorch artifact",
) -> Any:
    """Load a torch artifact using weights-only mode unless the user opts out."""

    target = Path(path).expanduser()
    if not target.is_file():
        raise FileNotFoundError(f"{description} not found at '{target}'")
    if _torch_load_supports_weights_only():
        try:
            return torch.load(target, map_location=map_location, weights_only=True)
        except Exception as exc:
            if not _unsafe_model_load_allowed():
                raise RuntimeError(
                    f"Could not safely load {description} from '{target}'. "
                    f"The file may be legacy or contain unsupported objects. "
                    f"Set {UNSAFE_MODEL_LOAD_ENV}=1 only if you trust the source."
                ) from exc
    elif not _unsafe_model_load_allowed():
        raise RuntimeError(
            f"This PyTorch build cannot safely load {description} from '{target}' "
            f"because weights-only loading is unavailable. Upgrade PyTorch or set "
            f"{UNSAFE_MODEL_LOAD_ENV}=1 only if you trust the source."
        )

    return torch.load(target, map_location=map_location)


def save_normalization_stats(
    path: str | Path,
    *,
    mean: np.ndarray | list[float],
    std: np.ndarray | list[float],
) -> str:
    """Persist normalization stats in a non-executable npz container."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        mean=np.asarray(mean, dtype=float),
        std=np.asarray(std, dtype=float),
    )
    return str(target)


def load_normalization_stats(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load normalization stats from a safe npz file, with gated legacy fallback."""

    target = Path(path).expanduser()
    safe_path = target if target.suffix.lower() == ".npz" else target.with_suffix(".npz")
    legacy_path = safe_path.with_suffix(".pkl")

    if safe_path.is_file():
        with np.load(safe_path, allow_pickle=False) as payload:
            try:
                mean = np.asarray(payload["mean"], dtype=float)
                std = np.asarray(payload["std"], dtype=float)
            except KeyError as exc:
                raise ValueError(f"Normalization stats file is missing '{exc.args[0]}' data: {safe_path}") from exc
        return mean, std

    if legacy_path.is_file():
        if not _unsafe_model_load_allowed():
            raise RuntimeError(
                f"Legacy normalization stats at '{legacy_path}' use pickle and are blocked by default. "
                f"Regenerate them or set {UNSAFE_MODEL_LOAD_ENV}=1 only if you trust the source."
            )
        with legacy_path.open("rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict):
            raise ValueError(f"Legacy normalization stats are invalid: {legacy_path}")
        try:
            mean = np.asarray(payload["mean"], dtype=float)
            std = np.asarray(payload["std"], dtype=float)
        except KeyError as exc:
            raise ValueError(
                f"Legacy normalization stats file is missing '{exc.args[0]}' data: {legacy_path}"
            ) from exc
        save_normalization_stats(safe_path, mean=mean, std=std)
        return mean, std

    raise FileNotFoundError(f"Normalization stats not found at '{safe_path}' or '{legacy_path}'")

"""Secure checkpoint loading for package and standalone TandemYTC runs."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

import torch

UNSAFE_MODEL_LOAD_ENV = "INTEGRAPOSE_ALLOW_UNSAFE_MODEL_LOAD"

try:  # Preferred when TandemYTC runs from the installed IntegraPose package.
    from integra_pose.utils.safe_model_io import load_torch_artifact as _core_loader
except ImportError:  # Standalone script execution may not have the package root on sys.path.
    _core_loader = None


def _unsafe_load_allowed() -> bool:
    return os.environ.get(UNSAFE_MODEL_LOAD_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _supports_weights_only() -> bool:
    try:
        return "weights_only" in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        return False


def safe_torch_load(path: Path | str, *, map_location: Any = None) -> Any:
    """Load a checkpoint without executable pickle fallback by default."""

    target = Path(path).expanduser()
    if _core_loader is not None:
        return _core_loader(
            target,
            map_location=map_location,
            description="TandemYTC checkpoint",
        )
    if not target.is_file():
        raise FileNotFoundError(f"TandemYTC checkpoint not found at '{target}'")

    if _supports_weights_only():
        try:
            return torch.load(str(target), map_location=map_location, weights_only=True)
        except Exception as exc:
            if not _unsafe_load_allowed():
                raise RuntimeError(
                    "The TandemYTC checkpoint could not be loaded in weights-only mode. "
                    f"Set {UNSAFE_MODEL_LOAD_ENV}=1 only if you trust its source."
                ) from exc
    elif not _unsafe_load_allowed():
        raise RuntimeError(
            "This PyTorch version does not support weights-only checkpoint loading. "
            f"Upgrade PyTorch or set {UNSAFE_MODEL_LOAD_ENV}=1 only if you trust the checkpoint."
        )

    return torch.load(str(target), map_location=map_location)

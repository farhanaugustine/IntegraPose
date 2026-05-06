"""Project-wide random seed management.

A single named seed per workflow stage (split, VAE init, dataloader, HMM,
HDBSCAN, augmentation). Two responsibilities:

1. **Apply** the seeds to every random-number source the pipeline uses:
   ``random``, ``numpy``, and ``torch`` (with cudnn deterministic flags), plus
   any caller-side libraries that need their own ``random_state`` argument.
2. **Persist** the resolved seeds into the run's metadata so a researcher who
   loads the output six months later can re-run the exact same analysis.

This module deliberately knows nothing about Tk; ``ConfigManager`` owns the
GUI side and converts to/from this dict-shaped representation.
"""

from __future__ import annotations

import os
import random as py_random
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional


DEFAULT_SEED = 42


@dataclass(slots=True)
class SeedsConfig:
    """Named seeds per workflow stage. All default to ``DEFAULT_SEED``.

    A single value per stage is intentional — researchers can re-run only the
    stage they changed without disturbing earlier stages. ``global`` is the
    base seed applied at process start; the others are layered on top of
    whatever is set globally at the call site.
    """

    global_seed: int = DEFAULT_SEED
    split_seed: int = DEFAULT_SEED
    vae_init_seed: int = DEFAULT_SEED
    vae_dataloader_seed: int = DEFAULT_SEED
    hmm_init_seed: int = DEFAULT_SEED
    hdbscan_seed: int = DEFAULT_SEED
    augmentation_seed: int = DEFAULT_SEED
    cudnn_deterministic: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Any) -> "SeedsConfig":
        if not isinstance(payload, dict):
            return cls()
        kwargs: dict[str, Any] = {}
        valid = {f.name for f in fields(cls)}
        for key, value in payload.items():
            if key not in valid:
                continue
            if key == "cudnn_deterministic":
                kwargs[key] = bool(value)
                continue
            try:
                kwargs[key] = int(value)
            except (TypeError, ValueError):
                # Leave the field at its default if the saved value is junk.
                continue
        return cls(**kwargs)


def apply_global_seed(seeds: SeedsConfig, *, log_fn: Optional[Any] = None) -> None:
    """Apply ``global_seed`` (and cudnn determinism flag) to every shared RNG.

    Called once at the start of any pipeline that consumes randomness. Does
    not raise if torch is unavailable; libraries that aren't installed are
    simply skipped.
    """
    seed = int(seeds.global_seed)
    py_random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if seeds.cudnn_deterministic:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except AttributeError:
                pass
    except ImportError:
        pass

    if log_fn is not None:
        try:
            log_fn(
                f"Seeds applied: global={seed}, cudnn_deterministic="
                f"{bool(seeds.cudnn_deterministic)}.",
                "INFO",
            )
        except Exception:
            pass


def coerce_seed(value: Any, *, default: int = DEFAULT_SEED) -> int:
    """Best-effort parse of a Tk StringVar / GUI input back to an int."""
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def snapshot(seeds: SeedsConfig) -> dict[str, Any]:
    """JSON-friendly dict of the seeds, for inclusion in inference / repro metadata."""
    return seeds.to_dict()

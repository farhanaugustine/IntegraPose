"""Runtime version inventory for the sanity check.

Importable without side effects. Each "probe" runs in isolation and
reports either the resolved version string or a brief failure reason
— the runner uses this to render a green/red strip per dependency.
"""

from __future__ import annotations

import importlib
import platform
import sys
from typing import Dict


# Modules whose presence is *required* for IntegraPose's core workflows.
# Order matters for display: most-foundational first.
_REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "scipy",
    "cv2",
    "PIL",
    "yaml",
    "torch",
    "ultralytics",
    "supervision",
    "tkinter",
    "sklearn",
    "matplotlib",
)

# Modules used by specific plugins. Missing → warn, don't fail; the
# user only needs them when they reach for the relevant plugin.
_OPTIONAL_MODULES = (
    "hmmlearn",
    "umap",
    "hdbscan",
    "albumentations",
    "statsmodels",
    "scikit_posthocs",
)


def _module_version(module_name: str) -> str:
    """Return the version string for ``module_name``, or a failure reason.

    Tries ``__version__``, then ``importlib.metadata.version``, then a
    sentinel "(version unknown)" if the module imports but exposes no
    version. Imports happen in this function so a missing module
    surfaces here, not at sanity-check module-import time.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return f"(missing — {exc.__class__.__name__})"
    except Exception as exc:  # pragma: no cover - defensive
        return f"(import error — {exc.__class__.__name__}: {exc})"
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    # importlib.metadata is more authoritative for installed packages.
    try:
        from importlib.metadata import PackageNotFoundError, version as _pkg_version

        return str(_pkg_version(module_name))
    except (PackageNotFoundError, Exception):  # pragma: no cover
        pass
    return "(version unknown)"


def collect_runtime_info() -> Dict[str, Dict[str, str]]:
    """Snapshot the runtime: python, OS, required + optional modules.

    Returns a dict of two dicts so the UI can render them in two
    visually-distinct sections::

        {
            "host":     {"python": "...", "platform": "...", ...},
            "required": {"numpy": "...", "torch": "...", ...},
            "optional": {"hmmlearn": "...", ...},
        }
    """
    host = {
        "python": sys.version.split(maxsplit=1)[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    required = {name: _module_version(name) for name in _REQUIRED_MODULES}
    optional = {name: _module_version(name) for name in _OPTIONAL_MODULES}

    # CUDA availability is a torch sub-fact worth reporting separately
    # so users with a GPU box can see "torch is installed but CUDA isn't
    # wired up" cleanly. Failure is non-fatal.
    try:
        import torch  # noqa: PLC0415

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            try:
                count = int(torch.cuda.device_count())
                first = torch.cuda.get_device_name(0) if count else "(none)"
                required["torch.cuda"] = f"available ({count} device(s); first: {first})"
            except Exception:
                required["torch.cuda"] = "available"
        else:
            required["torch.cuda"] = "not available (CPU-only)"
    except Exception:
        # Already surfaced as a missing-torch row above.
        pass

    return {"host": host, "required": required, "optional": optional}


def is_failure(version_string: str) -> bool:
    """True if a row produced by :func:`collect_runtime_info` indicates a hard failure."""
    return version_string.startswith("(missing") or version_string.startswith("(import error")


__all__ = ["collect_runtime_info", "is_failure"]

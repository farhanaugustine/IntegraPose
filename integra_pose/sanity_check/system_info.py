"""Runtime version inventory for the sanity check.

Importable without side effects. Each "probe" runs in isolation and
reports either the resolved version string or a brief failure reason
— the runner uses this to render a green/red strip per dependency.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import platform
import sys
from typing import Dict

from integra_pose.utils.torch_backend import detect_torch_backend


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

_OPTIONAL_DISTRIBUTIONS = {
    "hmmlearn": ("hmmlearn",),
    "umap": ("umap-learn",),
    "hdbscan": ("hdbscan",),
    "albumentations": ("albumentations",),
    "statsmodels": ("statsmodels",),
    "scikit_posthocs": ("scikit-posthocs",),
}


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


def _optional_package_version(module_name: str) -> str:
    """Return optional-plugin availability without importing heavy modules.

    UMAP imports Numba and may trigger expensive first-run initialization.
    The sanity dialog only needs an inventory for optional plugins, so package
    metadata is the correct side-effect-free probe. The plugin validates its
    executable imports when the user launches it.
    """
    candidates = _OPTIONAL_DISTRIBUTIONS.get(module_name, (module_name,))
    for distribution_name in candidates:
        try:
            return str(importlib.metadata.version(distribution_name))
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception as exc:  # pragma: no cover - defensive metadata failure
            return f"(metadata error - {exc.__class__.__name__}: {exc})"

    try:
        if importlib.util.find_spec(module_name) is not None:
            return "(version unknown)"
    except Exception as exc:  # pragma: no cover - defensive finder failure
        return f"(metadata error - {exc.__class__.__name__}: {exc})"
    return "(missing - package not installed)"


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
    optional = {name: _optional_package_version(name) for name in _OPTIONAL_MODULES}

    # GPU backend availability is a torch sub-fact worth reporting separately
    # so users can see whether PyTorch resolved CPU, NVIDIA CUDA, or AMD ROCm.
    try:
        backend = detect_torch_backend("auto")
        required["torch.backend"] = backend.backend
        if backend.supports_cuda_api:
            label = "ROCm/HIP via torch.cuda" if backend.is_rocm else "NVIDIA CUDA"
            required["torch.cuda_api"] = (
                f"available ({label}; {backend.device_count} device(s); "
                f"first: {backend.display_name})"
            )
        else:
            required["torch.cuda_api"] = "not available (CPU-only)"
    except Exception:
        # Already surfaced as a missing-torch row above.
        pass

    return {"host": host, "required": required, "optional": optional}


def is_failure(version_string: str) -> bool:
    """True if a row produced by :func:`collect_runtime_info` indicates a hard failure."""
    return version_string.startswith("(missing") or version_string.startswith("(import error")


__all__ = ["collect_runtime_info", "is_failure"]

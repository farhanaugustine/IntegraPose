"""Reproducibility metadata helpers.

Functions here build the ``inference_metadata.json`` payload that pins each
batch / inference run to the exact model, library versions, and tool versions
that produced it. Researchers who load these outputs months later can verify
the model checkpoint hasn't been swapped out from under them and trace the
exact ``ultralytics`` / ``IntegraPose`` versions used.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

PathLike = Union[str, Path]


def file_sha256(path: PathLike, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 of ``path``, or an empty string if unreadable.

    Reads in 1 MiB chunks so very large model files don't blow up memory.
    """
    target = Path(path).expanduser()
    if not target.is_file():
        return ""
    try:
        digest = hashlib.sha256()
        with open(target, "rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def _package_version(name: str) -> str:
    """Best-effort lookup of an installed package's version string.

    Returns "" instead of raising when the input is empty/None or the
    package isn't installed. Defensive against the (always-a-bug) case
    where a caller passes an empty string — Python 3.13's
    ``importlib.metadata.version("")`` raises ``ValueError`` rather
    than ``PackageNotFoundError``, so we short-circuit here."""
    if not name or not str(name).strip():
        return ""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except (PackageNotFoundError, ValueError):
            return ""
    except ImportError:  # pragma: no cover - py<3.8 fallback
        try:
            from importlib_metadata import PackageNotFoundError, version  # type: ignore

            try:
                return version(name)
            except (PackageNotFoundError, ValueError):
                return ""
        except ImportError:
            return ""


def _integrapose_version() -> str:
    try:
        from integra_pose import __version__

        return str(__version__)
    except Exception:
        return ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_inference_metadata(
    *,
    source_video: PathLike,
    frame_width: int,
    frame_height: int,
    fps: float,
    fps_source: str,
    device_requested: str,
    device_used: str,
    single_animal_mode: bool,
    model_path: PathLike,
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    use_tracker: bool = False,
    tracker_config: Optional[PathLike] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble the dict written to ``inference_metadata.json``.

    Captures *what* was processed (video, fps), *how* (model + thresholds +
    tracker), and *with what versions* (IntegraPose / ultralytics / pytorch).
    Includes a SHA-256 of the model file so a swapped checkpoint is detectable.
    """
    model_path_resolved = Path(str(model_path)).expanduser() if model_path else Path()
    payload: dict[str, Any] = {
        "schema_version": 2,
        "recorded_at_utc": _utc_now_iso(),
        "source_video": str(source_video),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "fps": float(fps),
        "fps_source": str(fps_source or ""),
        "device_requested": str(device_requested or ""),
        "device_used": str(device_used or ""),
        "single_animal_mode": bool(single_animal_mode),
        "model_path": str(model_path_resolved) if model_path else "",
        "model_sha256": file_sha256(model_path_resolved) if model_path else "",
        "use_tracker": bool(use_tracker),
        "tracker_config": str(tracker_config) if tracker_config else "",
        "integrapose_version": _integrapose_version(),
        "ultralytics_version": _package_version("ultralytics"),
        "torch_version": _package_version("torch"),
        # python_version is filled in immediately below from sys.version_info;
        # keeping the key here so the schema order is stable.
        "python_version": "",
    }
    # python_version via importlib.metadata isn't appropriate (Python isn't
    # a regular pip-installed distribution), so pull from sys instead.
    try:
        import sys

        payload["python_version"] = ".".join(str(p) for p in sys.version_info[:3])
    except Exception:
        payload["python_version"] = ""
    if conf_threshold is not None:
        payload["conf_threshold"] = float(conf_threshold)
    if iou_threshold is not None:
        payload["iou_threshold"] = float(iou_threshold)
    if extra:
        payload.update(extra)
    return payload

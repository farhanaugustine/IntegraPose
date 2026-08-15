from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .models import (
    BEHAVIOR,
    OBJECT_INTERACTION,
    ROI_CONCURRENT,
    ROI_EXCLUSIVE,
)


SPATIAL_MODE = "spatial"
BEHAVIOR_MODE = "behavior"
VALID_MODES = {SPATIAL_MODE, BEHAVIOR_MODE}
VALID_EVENT_KINDS = {
    ROI_CONCURRENT,
    ROI_EXCLUSIVE,
    OBJECT_INTERACTION,
    BEHAVIOR,
}


class ReviewLaunchError(RuntimeError):
    """Raised when the integrated review process cannot be started safely."""


@dataclass(frozen=True)
class ReviewLaunch:
    process: subprocess.Popen[Any]
    manifest_path: Path
    mode: str
    event_kind: str | None
    launched_at: float


_ACTIVE: dict[str, ReviewLaunch] = {}


def _manifest_path(raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "run_manifest.json"
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ReviewLaunchError(
            f"IntegraPose run_manifest.json was not found: {candidate}"
        ) from exc
    if not resolved.is_file() or resolved.name.casefold() != "run_manifest.json":
        raise ReviewLaunchError(
            f"Expected an IntegraPose run_manifest.json, received: {resolved}"
        )
    return resolved


def _read_status(manifest_path: Path, launched_at: float) -> dict[str, Any]:
    status_path = (
        manifest_path.parent
        / "bout_review_workspace"
        / "last_review_status.json"
    )
    try:
        if not status_path.is_file() or status_path.stat().st_mtime < launched_at:
            return {}
        value = json.loads(status_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def launch_reviewer(
    parent: Any,
    manifest_path: str | Path,
    *,
    mode: str,
    event_kind: str | None = None,
    source_video_root: str | Path | None = None,
    wait: bool = False,
    on_exit: Callable[[int, dict[str, Any]], None] | None = None,
) -> ReviewLaunch:
    """Launch the PySide6 reviewer without mixing Qt and Tk event loops."""

    if importlib.util.find_spec("PySide6") is None:
        raise ReviewLaunchError(
            "PySide6 is not installed in the Python environment running "
            "IntegraPose. Install the project requirements before using the "
            "integrated reviewer."
        )
    normalized_mode = str(mode or "").strip().casefold()
    if normalized_mode not in VALID_MODES:
        raise ReviewLaunchError(f"Unknown reviewer mode: {mode!r}")
    normalized_kind = str(event_kind or "").strip() or None
    if normalized_kind is not None and normalized_kind not in VALID_EVENT_KINDS:
        raise ReviewLaunchError(f"Unknown reviewer event kind: {event_kind!r}")

    manifest = _manifest_path(manifest_path)
    key = str(manifest).casefold()
    existing = _ACTIVE.get(key)
    if existing is not None and existing.process.poll() is None:
        raise ReviewLaunchError(
            "This analytics run is already open in the review workspace. "
            "Close that reviewer before launching a second editor for the "
            "same review database."
        )
    _ACTIVE.pop(key, None)

    command = [
        sys.executable,
        "-m",
        "integra_pose.bout_reviewer",
        "--root",
        str(manifest),
        "--mode",
        normalized_mode,
    ]
    if normalized_kind:
        command.extend(["--event-kind", normalized_kind])
    if source_video_root:
        command.extend(["--source-video-root", str(source_video_root)])

    package_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    existing_python_path = str(environment.get("PYTHONPATH") or "").strip()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(package_root), existing_python_path)
        if value
    )
    try:
        process = subprocess.Popen(
            command,
            shell=False,
            cwd=str(package_root),
            env=environment,
        )
    except OSError as exc:
        raise ReviewLaunchError(
            f"Could not start the IntegraPose review workspace: {exc}"
        ) from exc
    launch = ReviewLaunch(
        process=process,
        manifest_path=manifest,
        mode=normalized_mode,
        event_kind=normalized_kind,
        launched_at=time.time(),
    )
    _ACTIVE[key] = launch

    done_variable = None
    if wait:
        try:
            import tkinter as tk

            done_variable = tk.BooleanVar(master=parent, value=False)
        except Exception:
            done_variable = None

    def poll() -> None:
        return_code = process.poll()
        if return_code is None:
            try:
                parent.after(250, poll)
            except Exception:
                pass
            return
        _ACTIVE.pop(key, None)
        status = _read_status(manifest, launch.launched_at)
        if on_exit is not None:
            try:
                on_exit(int(return_code), status)
            except Exception:
                pass
        if done_variable is not None:
            try:
                done_variable.set(True)
            except Exception:
                pass

    try:
        parent.after(100, poll)
    except Exception:
        pass
    if done_variable is not None:
        parent.wait_variable(done_variable)
    return launch

"""FPS resolution helper.

Each workflow (Tab 6, Batch Wizard, plugins) holds its own user-visible FPS
field. This module centralises only the *policy* for combining a user-entered
value with a probed value — never the storage of the value itself.

Resolution priority:
    1. ``user_value`` — whatever the user typed in the workflow's GUI field.
       Wins if it parses to a positive float. Codecs sometimes report bogus FPS
       (0, 29.97 inside true-30 files, etc.), so the user's explicit choice is
       authoritative.
    2. Probed value from ``cv2.CAP_PROP_FPS`` for the given video. Used when
       ``user_value`` is blank/invalid.
    3. Otherwise: raise ``FpsUnavailableError``. There is no silent fallback to
       30.0 — callers that previously relied on that fallback should surface
       the failure to the user.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import cv2


class FpsUnavailableError(ValueError):
    """Raised when neither the user-entered FPS nor the probed FPS is usable."""


_LogFn = Optional[Callable[[str, str], None]]


def _coerce_user_value(user_value) -> Optional[float]:
    if user_value is None:
        return None
    if isinstance(user_value, (int, float)):
        value = float(user_value)
    else:
        text = str(user_value).strip()
        if not text:
            return None
        try:
            value = float(text)
        except (TypeError, ValueError):
            return None
    if not (value > 0) or value != value:  # rejects 0, negative, NaN
        return None
    return value


def _probe_video_fps(video_path: Union[str, Path, None]) -> Optional[float]:
    if not video_path:
        return None
    cap = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap or not cap.isOpened():
            return None
        raw = cap.get(cv2.CAP_PROP_FPS)
    except Exception:
        return None
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not (value > 0) or value != value:
        return None
    return value


def resolve_fps(
    video_path: Union[str, Path, None],
    user_value=None,
    *,
    log_fn: _LogFn = None,
    workflow_label: str = "this workflow",
) -> Tuple[float, str]:
    """Return ``(fps, source)`` where source is ``"user"`` or ``"probe"``.

    User-entered value wins when it parses to a positive float. Probed video
    FPS is used only when no user value is set. Both failing raises
    ``FpsUnavailableError`` — there is no silent ``30.0`` fallback.

    ``log_fn`` (optional) is called as ``log_fn(message, level)`` with levels
    ``"INFO"`` or ``"WARNING"`` so callers can surface the choice in the app
    log without printing.
    """

    user_fps = _coerce_user_value(user_value)
    probed_fps = _probe_video_fps(video_path)

    if user_fps is not None:
        if log_fn is not None:
            if probed_fps is not None and abs(probed_fps - user_fps) > 0.5:
                log_fn(
                    f"Using user-entered FPS {user_fps:g} for {video_path} "
                    f"(probe reported {probed_fps:g}; user value wins).",
                    "INFO",
                )
            else:
                log_fn(f"Using user-entered FPS {user_fps:g} for {video_path}.", "INFO")
        return user_fps, "user"

    if probed_fps is not None:
        if log_fn is not None:
            log_fn(f"Using probed FPS {probed_fps:g} for {video_path}.", "INFO")
        return probed_fps, "probe"

    raise FpsUnavailableError(
        f"Could not determine video FPS for {video_path}. "
        f"Set the Video FPS field in {workflow_label} and try again."
    )


def detect_video_fps(video_path: Union[str, Path, None]) -> Optional[float]:
    """Convenience helper for GUIs that want to display the probed FPS as a hint."""
    return _probe_video_fps(video_path)

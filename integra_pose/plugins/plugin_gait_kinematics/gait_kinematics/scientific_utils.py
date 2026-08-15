"""Scientific contracts shared by the gait analysis implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd


GAIT_RESULT_COLUMNS = [
    "track_id",
    "paw",
    "start_frame",
    "end_frame",
    "stride_duration_frames",
    "stride_duration_s",
    "stride_length",
    "stride_speed",
    "stride_speed_px_per_frame",
    "stride_speed_px_per_s",
    "step_length",
    "step_width",
    "video_fps",
]


def validated_fps(fps):
    if fps is None:
        return None
    try:
        value = float(fps)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0 else None


def contiguous_difference(df, column):
    """Return one-frame differences; gaps never become derivatives."""
    frame_delta = df.groupby("track_id", sort=False)["frame"].diff()
    values = df.groupby("track_id", sort=False)[column].diff()
    return values.where(frame_delta.eq(1))


def frames_are_consecutive(frame_series):
    frames = pd.to_numeric(frame_series, errors="coerce").to_numpy(dtype=float)
    return len(frames) >= 2 and np.isfinite(frames).all() and np.all(np.diff(frames) == 1)


def empty_gait_result():
    return pd.DataFrame(columns=GAIT_RESULT_COLUMNS)

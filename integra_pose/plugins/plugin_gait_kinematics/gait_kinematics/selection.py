"""Select one subject without joining different identities across frames."""
import numpy as np


def select_subject(data, config, total_frames):
    general = config["GENERAL_PARAMS"]
    data = data.copy()
    if not np.isfinite(data["frame"]).all() or not data["frame"].between(0, total_frames - 1).all():
        raise ValueError("Label frame numbers exceed the selected video's frame range. Check video/label pairing.")
    start, end = general["START_FRAME"], general["END_FRAME"]
    if (start is not None and start >= total_frames) or (end is not None and end >= total_frames):
        raise ValueError("Selected frame range exceeds the video length (frames are zero-based).")
    if start is not None:
        data = data[data.frame >= start]
    if end is not None:
        data = data[data.frame <= end]
    counts = data.groupby("track_id").size()
    target = general["TARGET_TRACK_ID"]
    if target is None:
        if len(counts) != 1:
            available = ", ".join(f"{tid}: {count} detections" for tid, count in counts.items()) or "none"
            raise ValueError("Choose a target track ID; available tracks in the selected range: " + available)
        target = counts.index[0]
    data = data[data.track_id == target].copy()
    if data.empty:
        raise ValueError(f"No detections for target track {target} in the selected frame range.")
    if data.frame.duplicated().any():
        raise ValueError("The selected track has multiple detections in a frame. Resolve duplicate track IDs first.")
    threshold = general["KEYPOINT_CONF_THRESHOLD"]
    for point in config["DATASET"]["KEYPOINT_ORDER"]:
        xy = [f"{point}_x", f"{point}_y"]
        confidence = data[f"{point}_conf"]
        invalid = ~np.isfinite(confidence) | (confidence < threshold)
        invalid |= ~np.isfinite(data[xy]).all(axis=1)
        data.loc[invalid, xy] = np.nan
    return data.sort_values("frame").reset_index(drop=True)

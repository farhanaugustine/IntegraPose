"""Batch crop and clean videos using a reusable ROI."""

from __future__ import annotations

import cv2
import glob
import json
import os
import subprocess
from datetime import datetime
from typing import Callable, Iterable

VIDEO_EXTENSIONS = ["*.mp4", "*.mov", "*.avi", "*.mkv"]


def find_videos(video_dir: str) -> list[str]:
    files = []
    search_dir = video_dir if video_dir else "."
    for ext in VIDEO_EXTENSIONS:
        files.extend(glob.glob(os.path.join(search_dir, ext)))
    return sorted(files)


def pick_roi_on_first_frame(video_path: str) -> tuple[int, int, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read first frame of {video_path}")

    cv2.namedWindow("Select ROI (ENTER/SPACE to confirm, ESC to cancel)", cv2.WINDOW_NORMAL)
    x, y, w, h = cv2.selectROI("Select ROI (ENTER/SPACE to confirm, ESC to cancel)", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        raise RuntimeError("No ROI selected.")
    return int(x), int(y), int(w), int(h)


def save_roi_config(path: str, roi: tuple[int, int, int, int], reference_video_path: str) -> None:
    x, y, w, h = roi
    config = {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "reference_video": os.path.abspath(reference_video_path),
        "reference_frame_index": 0,
        "created_at": datetime.now().isoformat(),
        "notes": "Crop region for arena; coordinates in original video pixel space.",
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_roi_config(path: str) -> tuple[int, int, int, int] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    required_keys = ["x", "y", "width", "height"]
    if all(k in config for k in required_keys):
        return int(config["x"]), int(config["y"]), int(config["width"]), int(config["height"])
    return None


def build_ffmpeg_cmd(
    ffmpeg_bin: str,
    input_path: str,
    output_path: str,
    crop: tuple[int, int, int, int],
    *,
    use_cuda: bool = True,
) -> list[str]:
    x, y, w, h = crop
    vf_filter = f"crop={w}:{h}:{x}:{y},deflicker,unsharp=5:5:1.0:5:5:0.0"
    if use_cuda:
        return [
            ffmpeg_bin,
            "-y",
            "-hwaccel",
            "cuda",
            "-i",
            input_path,
            "-vf",
            vf_filter,
            "-c:v",
            "h264_nvenc",
            "-preset",
            "fast",
            "-cq",
            "23",
            "-c:a",
            "copy",
            output_path,
        ]
    return [
        ffmpeg_bin,
        "-y",
        "-i",
        input_path,
        "-vf",
        vf_filter,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "copy",
        output_path,
    ]


def crop_videos(
    video_dir: str,
    *,
    output_subdir: str = "cropped",
    use_cuda: bool = True,
    force_new_roi: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    roi_config_path: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[str]:
    """
    Crop all videos in a folder using a shared ROI (selected once or loaded).

    Returns a list of output video paths.
    """
    videos = find_videos(video_dir)
    if not videos:
        raise FileNotFoundError("No videos found in the selected folder.")

    roi_store = roi_config_path or os.path.join(video_dir, "crop_roi.json")
    roi = None if force_new_roi else load_roi_config(roi_store)

    if roi is None:
        roi = pick_roi_on_first_frame(videos[0])
        save_roi_config(roi_store, roi, videos[0])
        if on_progress:
            on_progress(f"ROI saved to {roi_store}")
    else:
        if on_progress:
            on_progress(f"Loaded ROI from {roi_store}")

    out_dir = os.path.join(video_dir, output_subdir or "cropped")
    os.makedirs(out_dir, exist_ok=True)

    produced = []
    for vpath in videos:
        base_name = os.path.splitext(os.path.basename(vpath))[0]
        out_path = os.path.join(out_dir, base_name + "_cropped.mp4")
        cmd = build_ffmpeg_cmd(ffmpeg_bin, vpath, out_path, roi, use_cuda=use_cuda)
        if on_progress:
            on_progress("Running: " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {vpath}: {proc.stderr}")
        produced.append(out_path)
        if on_progress:
            on_progress(f"Wrote {out_path}")

    if on_progress:
        on_progress(f"Completed crop for {len(produced)} video(s).")
    return produced

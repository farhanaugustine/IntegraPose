"""Frame extraction helpers reused by the GUI."""

from __future__ import annotations

import os
import random
import re
import cv2
from pathlib import Path
from typing import Callable, Iterable

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def count_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total and total > 0:
        cap.release()
        return total
    total = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        total += 1
    cap.release()
    return total


def sanitize_video_stem(video_path: str | Path, *, fallback: str = "video") -> str:
    stem = Path(video_path).stem.strip() if str(video_path).strip() else ""
    if not stem:
        stem = fallback
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or fallback


def frame_filename(idx: int, *, video_path: str | Path | None = None, prefix: str | None = None, ext: str = ".jpg") -> str:
    stem = str(prefix).strip() if prefix is not None else ""
    if not stem and video_path is not None:
        stem = sanitize_video_stem(video_path)
    suffix = ext if str(ext).startswith(".") else f".{ext}"
    if not stem:
        return f"frame_{int(idx):06d}{suffix}"
    return f"{stem}__frame_{int(idx):06d}{suffix}"


def frame_name(out_dir: str, idx: int, *, video_path: str | Path | None = None, prefix: str | None = None, ext: str = ".jpg") -> str:
    return os.path.join(out_dir, frame_filename(idx, video_path=video_path, prefix=prefix, ext=ext))


def save_frame(image, out_dir: str, idx: int, *, video_path: str | Path | None = None, prefix: str | None = None, ext: str = ".jpg"):
    ensure_dir(out_dir)
    cv2.imwrite(frame_name(out_dir, idx, video_path=video_path, prefix=prefix, ext=ext), image)


def interactive_extractor(video_path: str, output_dir: str):
    """
    Shows frames; press:
      - 's' to save the current frame
      - 'q' or ESC to quit
      - any other key to advance
    """
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = 0
    frame_idx = 0
    window_title = f"Interactive: {os.path.basename(video_path)} (s=save, q=quit)"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow(window_title, frame)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                save_frame(frame, output_dir, frame_idx, video_path=video_path)
                saved += 1
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return {"saved": saved, "total_seen": frame_idx}


def stride_extractor(video_path: str, output_dir: str, stride: int, total_to_save: int):
    if stride <= 0:
        stride = 1

    ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = 0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % stride == 0:
                save_frame(frame, output_dir, frame_idx, video_path=video_path)
                saved += 1
                if total_to_save > 0 and saved >= total_to_save:
                    break
            frame_idx += 1
    finally:
        cap.release()

    return {"saved": saved, "last_frame_index": frame_idx}


def random_extractor(video_path: str, output_dir: str, total_to_save: int):
    ensure_dir(output_dir)
    n_frames = count_frames(video_path)
    if n_frames <= 0:
        raise RuntimeError(f"Could not read frame count for: {video_path}")

    if total_to_save <= 0:
        total_to_save = min(100, n_frames)

    total_to_save = min(total_to_save, n_frames)
    sample_indices = sorted(random.sample(range(n_frames), total_to_save))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = 0
    current_target_i = 0
    target_idx = sample_indices[current_target_i]

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx == target_idx:
                save_frame(frame, output_dir, frame_idx, video_path=video_path)
                saved += 1
                current_target_i += 1
                if current_target_i >= len(sample_indices):
                    break
                target_idx = sample_indices[current_target_i]

            frame_idx += 1
    finally:
        cap.release()

    return {"saved": saved, "selected_indices": sample_indices}


def extract_frames(
    video_path: str,
    output_dir: str,
    *,
    mode: str = "stride",
    stride: int = 5,
    total_to_save: int = 100,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """
    Extract frames from a single video using the selected mode.
    """
    if mode == "interactive":
        result = interactive_extractor(video_path, output_dir)
    elif mode == "random":
        result = random_extractor(video_path, output_dir, total_to_save)
    else:
        result = stride_extractor(video_path, output_dir, stride, total_to_save)

    if on_progress:
        on_progress(f"Extraction complete: {result}")
    return result

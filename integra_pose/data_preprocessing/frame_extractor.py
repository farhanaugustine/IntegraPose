"""Frame extraction helpers reused by the GUI."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import random
import re
from typing import Callable

import cv2
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
PATH_LENGTH_WARN_AT = 220
PATH_LENGTH_BLOCK_AT = 255


@dataclass(slots=True)
class ExtractedFrameRecord:
    filename: str
    path: str
    source_video: str
    frame_index: int
    timestamp_s: float
    mode: str
    score: float = 0.0
    reason: str = ""
    path_length: int = 0


def ensure_dir(path: str) -> None:
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


def frame_filename(
    idx: int,
    *,
    video_path: str | Path | None = None,
    prefix: str | None = None,
    ext: str = ".jpg",
) -> str:
    stem = str(prefix).strip() if prefix is not None else ""
    if not stem and video_path is not None:
        stem = sanitize_video_stem(video_path)
    suffix = ext if str(ext).startswith(".") else f".{ext}"
    if not stem:
        return f"frame_{int(idx):06d}{suffix}"
    return f"{stem}__frame_{int(idx):06d}{suffix}"


def frame_name(
    out_dir: str,
    idx: int,
    *,
    video_path: str | Path | None = None,
    prefix: str | None = None,
    ext: str = ".jpg",
) -> str:
    return os.path.join(out_dir, frame_filename(idx, video_path=video_path, prefix=prefix, ext=ext))


def save_frame(
    image,
    out_dir: str,
    idx: int,
    *,
    video_path: str | Path | None = None,
    prefix: str | None = None,
    ext: str = ".jpg",
) -> str:
    ensure_dir(out_dir)
    out_path = frame_name(out_dir, idx, video_path=video_path, prefix=prefix, ext=ext)
    _check_output_path(out_path)
    ok = cv2.imwrite(out_path, image)
    if not ok:
        raise RuntimeError(f"Failed to write frame image: {out_path}")
    return out_path


def interactive_extractor(video_path: str, output_dir: str) -> dict:
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

    records: list[ExtractedFrameRecord] = []
    frame_idx = 0
    fps = _video_fps(cap)
    window_title = f"Interactive: {os.path.basename(video_path)} (s=save, q=quit)"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow(window_title, frame)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out_path = save_frame(frame, output_dir, frame_idx, video_path=video_path)
                records.append(_record(out_path, video_path, frame_idx, fps, "interactive", reason="manual"))
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    manifest = write_extraction_manifest(output_dir, records, mode="interactive", source_video=video_path)
    return {"saved": len(records), "total_seen": frame_idx, "manifest_path": manifest}


def stride_extractor(
    video_path: str,
    output_dir: str,
    stride: int,
    total_to_save: int,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    if stride <= 0:
        stride = 1
    selected = _stride_indices(video_path, stride, total_to_save)
    return _save_selected_frames(
        video_path,
        output_dir,
        selected,
        mode="stride",
        on_progress=on_progress,
    )


def random_extractor(
    video_path: str,
    output_dir: str,
    total_to_save: int,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    n_frames = count_frames(video_path)
    if n_frames <= 0:
        raise RuntimeError(f"Could not read frame count for: {video_path}")
    if total_to_save <= 0:
        total_to_save = min(100, n_frames)
    selected = sorted(random.sample(range(n_frames), min(total_to_save, n_frames)))
    selected_map = {idx: {"score": 0.0, "reason": "random"} for idx in selected}
    return _save_selected_frames(video_path, output_dir, selected_map, mode="random", on_progress=on_progress)


def time_balanced_extractor(
    video_path: str,
    output_dir: str,
    total_to_save: int,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    n_frames = count_frames(video_path)
    if n_frames <= 0:
        raise RuntimeError(f"Could not read frame count for: {video_path}")
    if total_to_save <= 0:
        total_to_save = min(100, n_frames)
    total_to_save = min(total_to_save, n_frames)
    selected: list[int] = []
    for bin_idx in range(total_to_save):
        start = int(round(bin_idx * n_frames / total_to_save))
        end = int(round((bin_idx + 1) * n_frames / total_to_save))
        if end <= start:
            end = min(n_frames, start + 1)
        selected.append(random.randrange(start, end))
    selected = sorted(set(selected))
    selected_map = {idx: {"score": 0.0, "reason": "time_balanced"} for idx in selected}
    return _save_selected_frames(video_path, output_dir, selected_map, mode="time_balanced", on_progress=on_progress)


def motion_rich_extractor(
    video_path: str,
    output_dir: str,
    total_to_save: int,
    *,
    scan_stride: int = 3,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    if total_to_save <= 0:
        total_to_save = 100
    scores = _motion_scores(video_path, scan_stride=scan_stride, on_progress=on_progress)
    if not scores:
        return random_extractor(video_path, output_dir, total_to_save, on_progress=on_progress)
    top = sorted(scores, key=lambda item: item[1], reverse=True)[:total_to_save]
    selected_map = {idx: {"score": score, "reason": "motion"} for idx, score in sorted(top)}
    return _save_selected_frames(video_path, output_dir, selected_map, mode="motion_rich", on_progress=on_progress)


def hybrid_extractor(
    video_path: str,
    output_dir: str,
    total_to_save: int,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    n_frames = count_frames(video_path)
    if n_frames <= 0:
        raise RuntimeError(f"Could not read frame count for: {video_path}")
    if total_to_save <= 0:
        total_to_save = min(100, n_frames)
    total_to_save = min(total_to_save, n_frames)

    n_time = max(1, int(round(total_to_save * 0.4)))
    n_motion = max(1, int(round(total_to_save * 0.4)))
    n_random = max(0, total_to_save - n_time - n_motion)

    selected_map: dict[int, dict] = {}
    time_indices = _time_balanced_indices(n_frames, n_time)
    for idx in time_indices:
        selected_map[idx] = {"score": 0.0, "reason": "time_balanced"}

    scores = _motion_scores(video_path, scan_stride=3, on_progress=on_progress)
    for idx, score in sorted(scores, key=lambda item: item[1], reverse=True):
        if len([v for v in selected_map.values() if v["reason"] == "motion"]) >= n_motion:
            break
        if idx not in selected_map:
            selected_map[idx] = {"score": score, "reason": "motion"}

    remaining = [idx for idx in range(n_frames) if idx not in selected_map]
    if n_random > 0 and remaining:
        for idx in random.sample(remaining, min(n_random, len(remaining))):
            selected_map[idx] = {"score": 0.0, "reason": "random"}

    while len(selected_map) < total_to_save:
        remaining = [idx for idx in range(n_frames) if idx not in selected_map]
        if not remaining:
            break
        idx = remaining[len(remaining) // 2]
        selected_map[idx] = {"score": 0.0, "reason": "fill"}

    return _save_selected_frames(video_path, output_dir, selected_map, mode="hybrid", on_progress=on_progress)


def extract_frames(
    video_path: str,
    output_dir: str,
    *,
    mode: str = "stride",
    stride: int = 5,
    total_to_save: int = 100,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Extract frames from a single video using the selected mode."""

    mode_key = str(mode or "stride").strip().lower().replace("-", "_").replace(" ", "_")
    if mode_key == "interactive":
        result = interactive_extractor(video_path, output_dir)
    elif mode_key == "random":
        result = random_extractor(video_path, output_dir, total_to_save, on_progress=on_progress)
    elif mode_key in {"time_balanced", "time_balanced_random"}:
        result = time_balanced_extractor(video_path, output_dir, total_to_save, on_progress=on_progress)
    elif mode_key in {"motion_rich", "motion"}:
        result = motion_rich_extractor(video_path, output_dir, total_to_save, on_progress=on_progress)
    elif mode_key in {"hybrid", "hybrid_recommended"}:
        result = hybrid_extractor(video_path, output_dir, total_to_save, on_progress=on_progress)
    else:
        result = stride_extractor(video_path, output_dir, stride, total_to_save, on_progress=on_progress)

    if on_progress:
        on_progress(f"Extraction complete: {result}")
    return result


def write_extraction_manifest(
    output_dir: str,
    records: list[ExtractedFrameRecord],
    *,
    mode: str,
    source_video: str,
) -> str:
    ensure_dir(output_dir)
    path = Path(output_dir) / "frame_extraction_manifest.csv"
    fields = [
        "filename",
        "path",
        "source_video",
        "frame_index",
        "timestamp_s",
        "mode",
        "score",
        "reason",
        "path_length",
        "created_at_utc",
    ]
    created_at = datetime.now(timezone.utc).isoformat()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "filename": record.filename,
                    "path": record.path,
                    "source_video": record.source_video,
                    "frame_index": int(record.frame_index),
                    "timestamp_s": f"{float(record.timestamp_s):.6f}",
                    "mode": mode,
                    "score": f"{float(record.score):.6f}",
                    "reason": record.reason,
                    "path_length": int(record.path_length),
                    "created_at_utc": created_at,
                }
            )
    return str(path)


def _save_selected_frames(
    video_path: str,
    output_dir: str,
    selected: dict[int, dict],
    *,
    mode: str,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = _video_fps(cap)
    target_indices = sorted(int(idx) for idx in selected)
    target_set = set(target_indices)
    long_path_count = _preflight_paths(video_path, output_dir, target_indices)
    if on_progress and long_path_count:
        on_progress(
            f"Warning: {long_path_count} extracted frame path(s) are long. "
            "Use a shorter output root if extraction is slow."
        )
    records: list[ExtractedFrameRecord] = []
    saved = 0
    total = len(target_indices)
    frame_idx = 0

    try:
        while target_set:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in target_set:
                out_path = save_frame(frame, output_dir, frame_idx, video_path=video_path)
                payload = selected.get(frame_idx, {})
                records.append(
                    _record(
                        out_path,
                        video_path,
                        frame_idx,
                        fps,
                        mode,
                        score=float(payload.get("score", 0.0) or 0.0),
                        reason=str(payload.get("reason", mode)),
                    )
                )
                target_set.remove(frame_idx)
                saved += 1
                if on_progress and (saved <= 5 or saved == total or saved % 25 == 0):
                    on_progress(f"Saved frame {saved}/{total}: source frame {frame_idx}")
            frame_idx += 1
    finally:
        cap.release()

    manifest = write_extraction_manifest(output_dir, records, mode=mode, source_video=video_path)
    return {
        "saved": saved,
        "selected_indices": target_indices,
        "manifest_path": manifest,
        "last_frame_index": frame_idx,
    }


def _stride_indices(video_path: str, stride: int, total_to_save: int) -> dict[int, dict]:
    n_frames = count_frames(video_path)
    if n_frames <= 0:
        raise RuntimeError(f"Could not read frame count for: {video_path}")
    indices = list(range(0, n_frames, max(1, int(stride))))
    if total_to_save > 0:
        indices = indices[:total_to_save]
    return {idx: {"score": 0.0, "reason": "stride"} for idx in indices}


def _time_balanced_indices(n_frames: int, total_to_save: int) -> list[int]:
    total_to_save = min(max(1, int(total_to_save)), max(1, n_frames))
    out: list[int] = []
    for bin_idx in range(total_to_save):
        start = int(round(bin_idx * n_frames / total_to_save))
        end = int(round((bin_idx + 1) * n_frames / total_to_save))
        if end <= start:
            end = min(n_frames, start + 1)
        out.append((start + end - 1) // 2)
    return sorted(set(out))


def _motion_scores(
    video_path: str,
    *,
    scan_stride: int,
    on_progress: Callable[[str], None] | None = None,
) -> list[tuple[int, float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    scores: list[tuple[int, float]] = []
    prev_gray = None
    frame_idx = 0
    scan_stride = max(1, int(scan_stride))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % scan_stride != 0:
                frame_idx += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            if prev_gray is not None:
                score = float(cv2.absdiff(gray_small, prev_gray).mean())
                scores.append((frame_idx, score))
            prev_gray = gray_small
            if on_progress and frame_idx and frame_idx % 500 == 0:
                on_progress(f"Scanning motion: source frame {frame_idx}")
            frame_idx += 1
    finally:
        cap.release()
    return scores


def _record(
    out_path: str,
    video_path: str,
    frame_idx: int,
    fps: float,
    mode: str,
    *,
    score: float = 0.0,
    reason: str = "",
) -> ExtractedFrameRecord:
    return ExtractedFrameRecord(
        filename=Path(out_path).name,
        path=str(out_path),
        source_video=str(video_path),
        frame_index=int(frame_idx),
        timestamp_s=float(frame_idx) / max(float(fps), 1e-6),
        mode=mode,
        score=float(score),
        reason=reason,
        path_length=len(str(Path(out_path).absolute())),
    )


def _video_fps(cap: cv2.VideoCapture) -> float:
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    except Exception:
        fps = 0.0
    return fps if fps > 0 else 30.0


def _preflight_paths(video_path: str, output_dir: str, indices: list[int]) -> int:
    blocked: list[str] = []
    warned = 0
    for idx in indices:
        out_path = frame_name(output_dir, idx, video_path=video_path)
        length = len(str(Path(out_path).absolute()))
        if length >= PATH_LENGTH_BLOCK_AT:
            blocked.append(out_path)
        elif length >= PATH_LENGTH_WARN_AT:
            warned += 1
    if blocked:
        first = blocked[0]
        raise RuntimeError(
            f"{len(blocked)} extracted frame path(s) would be too long for safe Windows file operations. "
            f"Choose a shorter output root. First blocked path: {first}"
        )
    return warned


def _check_output_path(out_path: str) -> None:
    length = len(str(Path(out_path).absolute()))
    if length >= PATH_LENGTH_BLOCK_AT:
        raise RuntimeError(
            f"Output path is too long for safe Windows file operations ({length} chars): {out_path}"
        )

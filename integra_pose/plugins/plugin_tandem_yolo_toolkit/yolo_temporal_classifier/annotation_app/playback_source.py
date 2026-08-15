"""Resolve a *playable* video source for the annotation player.

Why this exists
---------------
NorPix ``.seq`` files carry no frame-rate metadata that a standard media
backend understands. When such a file is handed to Qt's ``QMediaPlayer`` the
underlying decoder falls back to a default **25 fps**, even though MARS ``.seq``
recordings are **30 fps**. The player therefore presents frame *N* at wall-clock
``N / 25`` instead of ``N / 30`` and the video runs 1.2x slow. Because the
annotation timeline positions behavior spans from the (correct) millisecond
values in the database, the spans end up looking time-shifted against the video
-- e.g. a mount at a true 66 s is shown at ``66 * 30/25 = 79.2 s``. The
annotations are fine; only playback is wrong.

This module resolves a ``.seq`` path to a real 30 fps MP4 before it reaches the
player:

1. If the path is already a normal video container, use it unchanged.
2. If a sibling MP4 (a prior transcode of the same recording) exists and its
   frame count matches the database, play that -- instant, no work.
3. Otherwise transcode the ``.seq`` to a cached MP4 proxy at the true fps using
   the bundled :class:`SeqReader`, then play the proxy.

The transcode mirrors ``prepare_full_video_npz.convert_seq_to_mp4`` (SeqReader
-> OpenCV ``VideoWriter``) but is dependency-light so importing it from the GUI
does not pull in the training stack.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional, Tuple

SEQ_SUFFIXES = {".seq"}
PLAYABLE_SUFFIXES = {".mp4", ".m4v", ".avi", ".mov", ".mkv"}

ProgressCallback = Callable[[int, int], None]


def _proxy_cache_dir() -> Path:
    return Path.home() / ".integrapose" / "tandemytc" / "playback_proxies"


def _frame_count(path: Path) -> Optional[int]:
    """Frame count via OpenCV, or ``None`` if it cannot be determined."""
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return count or None
    finally:
        cap.release()


def _aligned(candidate_frames: Optional[int], total_frames: Optional[int]) -> bool:
    """True unless the candidate is *positively* known to be misaligned."""
    if not total_frames or total_frames <= 0:
        return True
    if not candidate_frames or candidate_frames <= 0:
        return True  # cannot validate -> trust the sibling rather than transcode
    return abs(candidate_frames - total_frames) <= max(2, int(total_frames * 0.01))


def find_sibling_playable(seq_path: Path, total_frames: Optional[int] = None) -> Optional[Path]:
    """Return a sibling playable video for ``seq_path``.

    Prefers a candidate whose frame count matches ``total_frames``. If none can
    be probed (e.g. OpenCV unavailable) the first candidate is returned so a
    working sibling still beats the unplayable ``.seq``.
    """
    directory = seq_path.parent
    if not directory.is_dir():
        return None
    candidates: list[Path] = []
    for suffix in (".mp4", ".m4v", ".mov", ".avi", ".mkv"):
        candidates.extend(sorted(directory.glob(f"*{suffix}")))
    if not candidates:
        return None
    fallback: Optional[Path] = None
    for candidate in candidates:
        frames = _frame_count(candidate)
        if frames is None:
            fallback = fallback or candidate
            continue
        if _aligned(frames, total_frames):
            return candidate
    # No positively-aligned candidate: use a non-probeable one if we have it,
    # otherwise None so the caller transcodes a guaranteed-aligned proxy.
    return fallback


def transcode_seq_to_proxy(
    seq_path: Path,
    out_path: Path,
    *,
    fps: float,
    progress: Optional[ProgressCallback] = None,
) -> bool:
    """Transcode a ``.seq`` to a 30 fps MP4 proxy. Returns True on success."""
    try:
        import cv2
    except Exception:
        return False
    try:
        from ..utils.seq_reader import SeqReader  # type: ignore
    except Exception:
        try:
            from integra_pose.plugins.plugin_tandem_yolo_toolkit.yolo_temporal_classifier.utils.seq_reader import (  # type: ignore
                SeqReader,
            )
        except Exception:
            return False

    seq_path = Path(seq_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(f".{out_path.stem}.partial{out_path.suffix}")
    try:
        with SeqReader(str(seq_path)) as reader:
            reader.build_seek_table()
            n_frames = len(reader.seek_table or []) or int(reader.num_frames)
            if n_frames <= 0:
                return False
            width, height = int(reader.width), int(reader.height)
            writer = cv2.VideoWriter(
                str(temp_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps),
                (width, height),
            )
            if not writer.isOpened():
                return False
            try:
                for idx in range(n_frames):
                    frame = reader.read_frame(idx)
                    if frame.ndim == 2:
                        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(bgr)
                    if progress is not None and idx % 250 == 0:
                        progress(idx, n_frames)
            finally:
                writer.release()
        if not temp_path.is_file() or temp_path.stat().st_size <= 0:
            return False
        temp_path.replace(out_path)
        if progress is not None:
            progress(n_frames, n_frames)
        return True
    except Exception:
        return False
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _proxy_path_for(seq_path: Path, fps: float) -> Path:
    try:
        stat = seq_path.stat()
        stamp = f"{seq_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{float(fps):.6f}"
    except OSError:
        stamp = f"{seq_path}|{float(fps):.6f}"
    key = hashlib.sha1(stamp.encode("utf-8", "ignore")).hexdigest()[:16]
    return _proxy_cache_dir() / f"{seq_path.stem}.{key}.mp4"


def resolve_playback_source(
    video_path: str,
    *,
    fps: float,
    total_frames: Optional[int] = None,
    allow_transcode: bool = True,
    progress: Optional[ProgressCallback] = None,
) -> Tuple[str, Optional[str]]:
    """Resolve ``video_path`` to a player-friendly file.

    Returns ``(playable_path, note)`` where ``note`` is a human-readable warning
    or ``None``. Never raises for expected filesystem/decoder problems -- it
    degrades to the original path with an explanatory note so callers can always
    fall back to the previous behavior.
    """
    try:
        path = Path(str(video_path))
        if path.suffix.lower() not in SEQ_SUFFIXES:
            return str(video_path), None

        sibling = find_sibling_playable(path, total_frames)
        if sibling is not None:
            return str(sibling), None

        proxy = _proxy_path_for(path, fps)
        if proxy.is_file() and proxy.stat().st_size > 0:
            return str(proxy), None

        if not allow_transcode:
            return str(video_path), (
                f"'{path.name}' is a NorPix .seq with no MP4 available yet; "
                "playback rate may be wrong until a proxy is built."
            )

        if transcode_seq_to_proxy(path, proxy, fps=fps, progress=progress):
            return str(proxy), None

        return str(video_path), (
            f"Could not build a playable proxy for '{path.name}'. Playback may "
            "run at the wrong frame rate, making behaviors look time-shifted."
        )
    except Exception as exc:  # never block video loading on resolution issues
        return str(video_path), f"Playback source resolution failed: {exc}"

"""ADP-4 Commit G — Frame extraction for cluster naming.

Pulls representative frames from a source video given a list of bouts.
Default policy: for each bout, return three frames — start, middle,
end — so the user sees a "before/during/after" triptych. Optionally
configurable via ``n_per_bout``.

Why a separate module:

  - cv2 is a heavy import; isolating the import here keeps the rest
    of the toolkit cheap to load.
  - The extraction logic is pure given (video_path, frame_indices) —
    easy to unit-test against a synthesized video without wiring up
    Tk.
  - Reused by the naming dialog (Commit G) AND the bout scrubber
    (Commit H) once that lands.

Output format: list of PIL Image objects (RGB), thumbnailed to
``thumbnail_size`` so the naming dialog's 3×3 grid stays manageable.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


# A 3×3 grid of triptychs (3 frames each) → 9 thumbnails per row × 3 rows
# = up to 27 PIL images live at once. 160px wide keeps memory low and
# the layout readable on a 1080p screen.
DEFAULT_THUMBNAIL_SIZE = (160, 160)
DEFAULT_FRAMES_PER_BOUT = 3


def _frame_indices_for_bout(start_frame: int, end_frame: int, n: int) -> list:
    """Pick ``n`` evenly-spaced frame indices spanning ``[start, end]``.

    For ``n=3`` (the default) this is [start, midpoint, end]. For
    ``n=1`` it's just the midpoint. Larger n still works for the
    bout scrubber's use case.

    Frames are 1-indexed in the IntegraPose convention (matches
    ``read_detections`` and ``aggregate_states_into_bouts``); the
    cv2 seek call below adjusts to 0-indexed.
    """
    if n <= 0:
        return []
    if n == 1:
        return [int((start_frame + end_frame) // 2)]
    if start_frame == end_frame or n >= (end_frame - start_frame + 1):
        # Bout is too short for n distinct frames — repeat the same.
        return [int(start_frame)] * n
    step = (end_frame - start_frame) / (n - 1)
    return [int(round(start_frame + step * i)) for i in range(n)]


def extract_bout_thumbnails(
    bouts: Sequence[dict],
    *,
    video_path_map: dict,
    n_per_bout: int = DEFAULT_FRAMES_PER_BOUT,
    thumbnail_size: tuple = DEFAULT_THUMBNAIL_SIZE,
) -> list:
    """Read N frames per bout from each bout's source video.

    Args:
        bouts: List of bout dicts from ``aggregate_states_into_bouts``.
            Each bout must have ``directory``, ``start_frame``,
            ``end_frame``. Bouts without a known video for their
            directory are skipped (warning logged).
        video_path_map: ``{pose_dir: video_path}`` lookup. Same shape
            as Tab 7's cached ``video_path_map``.
        n_per_bout: Frames per bout. Default 3 (start/middle/end).
        thumbnail_size: ``(w, h)`` for the returned PIL Images. Pillow
            preserves aspect ratio internally, so the result fits
            within this box but isn't necessarily exactly this size.

    Returns:
        List of dicts, one per input bout, each with:
        ``{"bout_index": int, "frames": [PIL.Image, ...] or []}``.
        Bouts that couldn't be read produce an empty ``frames`` list
        rather than missing entries — that way the caller can render
        a "missing" placeholder cell instead of skipping silently.

    Raises:
        ImportError: If cv2 or Pillow isn't installed. The naming
            dialog should disable itself when this happens.
    """
    try:
        import cv2  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except Exception as exc:
        raise ImportError(
            "extract_bout_thumbnails: opencv-python and Pillow are required. "
            "`pip install opencv-python pillow`."
        ) from exc

    out: list = []

    # Group bouts by directory so we open each video only once and seek
    # forward through it. cv2.VideoCapture.set(POS_FRAMES) is allowed
    # but expensive on some codecs — sequential reads are friendlier.
    bouts_by_dir: dict = {}
    for idx, bout in enumerate(bouts):
        directory = bout.get("directory")
        if not directory:
            out.append({"bout_index": idx, "frames": []})
            continue
        bouts_by_dir.setdefault(directory, []).append((idx, bout))

    # Initialize the output list with the same length / order.
    out = [{"bout_index": i, "frames": []} for i in range(len(bouts))]

    for directory, items in bouts_by_dir.items():
        video_path = video_path_map.get(directory)
        if not video_path or not os.path.isfile(video_path):
            logger.warning(
                "extract_bout_thumbnails: no readable video for directory %r; "
                "skipping %d bout(s).",
                directory, len(items),
            )
            continue
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(
                "extract_bout_thumbnails: cv2 failed to open %r; skipping.",
                video_path,
            )
            continue

        try:
            for orig_index, bout in items:
                try:
                    start = int(bout.get("start_frame", 0))
                    end = int(bout.get("end_frame", start))
                except Exception:
                    out[orig_index]["frames"] = []
                    continue
                indices = _frame_indices_for_bout(start, end, n_per_bout)
                frames = _read_frames_at(cap, indices, thumbnail_size, Image, cv2)
                out[orig_index]["frames"] = frames
        finally:
            cap.release()

    return out


def _read_frames_at(cap, indices, thumbnail_size, Image, cv2):
    """Seek + read each frame; return PIL Images thumbnailed."""
    frames = []
    for fidx in indices:
        # IntegraPose convention: 1-indexed frame numbers.
        zero_idx = max(0, int(fidx) - 1)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, zero_idx)
            ret, frame = cap.read()
        except Exception:
            ret = False
            frame = None
        if not ret or frame is None:
            continue
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail(thumbnail_size)
            frames.append(img)
        except Exception:
            continue
    return frames


def select_top_n_bouts(
    bouts: Sequence[dict],
    *,
    n: int = 9,
    target_state: Any = None,
    sort_key: str = "duration_frames",
) -> list:
    """Pick the strongest ``n`` bouts for one sub-cluster.

    Default: longest bouts. Empty result if ``bouts`` is empty or no
    bouts match ``target_state``.

    Args:
        bouts: Aggregated bouts.
        n: Maximum number to return. Default 9 (per ADP-4 user signoff).
        target_state: When supplied, only bouts with ``state ==
            target_state`` are considered. Compares as string or int —
            the namespaced cluster_label or raw HDBSCAN id both work.
        sort_key: Bout-dict key whose value is sorted descending. For
            ADP-4 we use ``duration_frames`` (longest first).
    """
    if target_state is not None:
        candidates = [
            b for b in bouts
            if str(b.get("state")) == str(target_state)
        ]
    else:
        candidates = list(bouts)
    candidates.sort(key=lambda b: b.get(sort_key, 0), reverse=True)
    return candidates[:n]


__all__ = [
    "extract_bout_thumbnails",
    "select_top_n_bouts",
    "DEFAULT_FRAMES_PER_BOUT",
    "DEFAULT_THUMBNAIL_SIZE",
]

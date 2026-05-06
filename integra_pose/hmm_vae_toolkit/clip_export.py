"""ADP-4 Tab 7 — sub-cluster clip export (Commit I).

Two modes, sharing helpers:

  * ``export_bout_clips`` (recommended) — one .mp4 per bout, contiguous
    frames, organized into folders by behavior name. Output is shaped
    for downstream classifier training (BehaviorScope-style: each
    behavior's folder is a class). Default since 2026-04-26.

  * ``export_sub_cluster_clips`` (legacy) — one collated .mp4 per
    (class_id, sub_id) per source video, contains every frame matching
    that label end-to-end. Useful for visual review of a sub-cluster's
    full content; less useful as classifier training input because the
    frames aren't contiguous in source time.

Both modes:

  - Late-import cv2. The module imports cleanly without OpenCV.
  - Accept the namespaced label format from ``per_class_clustering``
    (``"0:0"``, ``"1:2"``, etc.).
  - Use ``state_names`` (Commit G) for human-readable folder/file
    names when supplied; fall back to ``class_name__subcluster_N``.
  - Are **always optional** — Tab 7's UI calls them only when the
    user clicks the explicit Export button.

Output layout (bout mode, recommended):

    output_dir/
      forward_locomotion/        # state_name from naming dialog
        video1__t0__f12-87.mp4
        video2__t0__f30-115.mp4
      walking__subcluster_2/     # not yet named
        video1__t0__f600-650.mp4
      clip_manifest.csv          # one row per clip, with metadata

The manifest is the bridge into supervised training: a follow-up
"Build BehaviorScope dataset" button can read it directly.

Why a separate module:

  - cv2 is a heavy import we don't want to drag into the rest of the
    toolkit.
  - Pure side-effect operations (write files), easy to skip in
    headless runs.
  - Clean module boundary keeps future BehaviorScope manifest export
    a sibling function rather than an inline branch.
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# Sentinel that means "no sub-cluster" — frames in this state are
# skipped during export so we don't waste video on noise.
_NOISE_LABEL = "-1"

# Filename of the manifest written next to the per-behavior folders.
# Schema described in ``export_bout_clips``'s docstring.
CLIP_MANIFEST_FILENAME = "clip_manifest.csv"


def _safe_filename_part(s: str) -> str:
    """Sanitize a class/cluster name for use in a filename.

    Strips characters that Windows / POSIX hate, collapses whitespace.
    """
    s = re.sub(r"[^\w\-_.]+", "_", str(s)).strip("_")
    return s or "unnamed"


def _parse_namespaced_label(label) -> Optional[tuple[int, int]]:
    """Parse the per_class_clustering namespaced label.

    Returns ``(class_id, sub_id)`` or ``None`` for noise / malformed.
    """
    if label is None:
        return None
    s = str(label)
    if s == _NOISE_LABEL:
        return None
    if ":" not in s:
        return None
    class_part, sub_part = s.split(":", 1)
    try:
        return int(class_part), int(sub_part)
    except ValueError:
        return None


def _behavior_folder_name(
    class_id: int,
    sub_id: int,
    *,
    class_names: Optional[dict] = None,
    state_names: Optional[dict] = None,
) -> str:
    """Pick the per-behavior folder name for a (class_id, sub_id) pair.

    Precedence:
      1. ``state_names["class_id:sub_id"]`` (user-supplied human name)
      2. ``{class_name}__subcluster_{sub_id}`` (class label + integer)
      3. ``class_{class_id}__subcluster_{sub_id}`` (no class name known)

    Sanitized to be filesystem-safe.
    """
    state_names = state_names or {}
    class_names = class_names or {}
    ns_key = f"{class_id}:{sub_id}"
    raw = state_names.get(ns_key)
    if raw and str(raw).strip():
        return _safe_filename_part(raw)
    class_part = _safe_filename_part(
        class_names.get(class_id) or f"class_{class_id}"
    )
    return f"{class_part}__subcluster_{sub_id}"


def export_sub_cluster_clips(
    detections_df,
    *,
    video_path: str,
    output_dir: str,
    class_names: Optional[dict] = None,
    state_names: Optional[dict] = None,
    label_column: str = "cluster_label",
    conf_threshold: float = 0.3,
) -> dict:
    """Write one .mp4 per (class_id, sub-cluster) pair from a source video.

    The export is **always optional** — the call site decides when to
    run it. A lab that only wants the sub-cluster bouts table can
    skip clip generation entirely.

    Args:
        detections_df: Frame-level dataframe with columns ``frame``,
            ``bbox``, ``keypoints``, and ``label_column``. Frames with
            label ``"-1"`` are skipped.
        video_path: Source video to extract frames from. Must exist.
        output_dir: Destination folder; created if missing.
        class_names: Optional ``{class_id: human_label}`` mapping.
            Used in filenames; falls back to the integer class id.
        state_names: Optional mapping of namespaced labels → user
            names assigned via the cluster-naming UI (Commit G). Keys
            are the same strings used in ``label_column`` (e.g.
            ``"0:1"``). When supplied, the file name uses the human
            name; otherwise it uses ``subcluster_0:1``.
        label_column: Column carrying the namespaced cluster labels.
            Default ``"cluster_label"``.
        conf_threshold: Keypoint confidence below which a keypoint is
            not drawn on the overlay. Default 0.3.

    Returns:
        Dict ``{(class_id, sub_id): file_path}`` of the .mp4 files
        actually written. Empty if nothing exported (e.g., all frames
        were noise).

    Raises:
        ImportError: If cv2 is missing. Tab 7 should hide / disable
            the export button when cv2 isn't importable, but the
            error is raised here too as a defense-in-depth.
        FileNotFoundError: If the source video can't be opened.
        ValueError: If required columns are missing.
    """
    # Late imports keep this module cheap.
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:
        raise ImportError(
            "export_sub_cluster_clips: opencv-python (cv2) is required to "
            "write video clips. `pip install opencv-python`."
        ) from exc

    for col in ("frame", "bbox", "keypoints", label_column):
        if col not in detections_df.columns:
            raise ValueError(
                f"export_sub_cluster_clips: detections_df missing required "
                f"column {col!r}."
            )
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"export_sub_cluster_clips: source video not found: {video_path!r}"
        )

    os.makedirs(output_dir, exist_ok=True)
    class_names = class_names or {}
    state_names = state_names or {}

    # Build frame_idx → list of (key, detection_dict) for quick lookup
    # during the single pass over the video.
    frame_map: dict[int, list] = defaultdict(list)
    sub_clusters_seen: set = set()
    for _, row in detections_df.iterrows():
        parsed = _parse_namespaced_label(row[label_column])
        if parsed is None:
            continue
        class_id, sub_id = parsed
        sub_clusters_seen.add((class_id, sub_id))
        try:
            frame_idx = int(row["frame"])
        except Exception:
            continue
        frame_map[frame_idx].append(
            (
                (class_id, sub_id),
                {
                    "bbox": row["bbox"],
                    "keypoints": row["keypoints"],
                },
            )
        )

    if not sub_clusters_seen:
        logger.warning(
            "export_sub_cluster_clips: no non-noise frames to export; "
            "nothing written."
        )
        return {}

    # Open the source video once; we'll iterate frame-by-frame and write
    # to the appropriate per-cluster writer.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(
            f"export_sub_cluster_clips: cv2 could not open video: {video_path}"
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writers: dict = {}
    written_paths: dict = {}

    def _build_filename(class_id: int, sub_id: int) -> str:
        ns_key = f"{class_id}:{sub_id}"
        # Prefer a human name from cluster-naming UI; fall back to a
        # generic "subcluster_X:Y" form.
        if ns_key in state_names and str(state_names[ns_key]).strip():
            human = _safe_filename_part(state_names[ns_key])
            return f"{human}.mp4"
        class_part = _safe_filename_part(class_names.get(class_id) or f"class_{class_id}")
        return f"{class_part}__subcluster_{sub_id}.mp4"

    for class_id, sub_id in sorted(sub_clusters_seen):
        out_path = os.path.join(output_dir, _build_filename(class_id, sub_id))
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if writer.isOpened():
            writers[(class_id, sub_id)] = writer
            written_paths[(class_id, sub_id)] = out_path
        else:
            logger.error(
                "export_sub_cluster_clips: cv2.VideoWriter failed to open: %s",
                out_path,
            )

    if not writers:
        cap.release()
        logger.warning(
            "export_sub_cluster_clips: no valid VideoWriters could be created."
        )
        return {}

    try:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1  # detections are 1-indexed by convention
            if frame_idx not in frame_map:
                continue
            for key, det in frame_map[frame_idx]:
                if key not in writers:
                    continue
                annotated = _annotate_frame(
                    frame.copy(),
                    det,
                    class_id=key[0],
                    sub_id=key[1],
                    frame_idx=frame_idx,
                    class_names=class_names,
                    state_names=state_names,
                    conf_threshold=conf_threshold,
                    cv2=cv2,
                )
                writers[key].write(annotated)
    finally:
        cap.release()
        for w in writers.values():
            try:
                w.release()
            except Exception:
                pass

    logger.info(
        "export_sub_cluster_clips: wrote %d clip(s) to %s.",
        len(written_paths), output_dir,
    )
    return written_paths


def _annotate_frame(
    frame,
    det: dict,
    *,
    class_id: int,
    sub_id: int,
    frame_idx: int,
    class_names: dict,
    state_names: dict,
    conf_threshold: float,
    cv2,
):
    """Draw bbox + keypoints + a label strip on a frame.

    Kept private because the exact rendering is a UI choice that may
    change. The signature is a kwargs grab-bag intentionally — call
    sites are localized and refactor risk is low.
    """
    bbox = det.get("bbox")
    keypoints = det.get("keypoints") or {}

    if bbox is not None:
        try:
            cx, cy, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            tl = (int(cx - w / 2), int(cy - h / 2))
            br = (int(cx + w / 2), int(cy + h / 2))
            cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
        except Exception:
            pass

    for _kp_name, payload in keypoints.items():
        try:
            x, y, conf = payload
            if conf >= conf_threshold:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        except Exception:
            continue

    ns_key = f"{class_id}:{sub_id}"
    human = state_names.get(ns_key) if state_names else None
    class_label = class_names.get(class_id) or f"class_{class_id}"
    text = (
        f"{human} | frame {frame_idx}"
        if human
        else f"{class_label} | sub {sub_id} | frame {frame_idx}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
    cv2.rectangle(frame, (10, 10), (20 + tw, 30 + th), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 30 + th - 5), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def export_bout_clips(
    bouts: Iterable[dict],
    *,
    video_path_map: dict,
    output_dir: str,
    class_names: Optional[dict] = None,
    state_names: Optional[dict] = None,
    overlay: bool = True,
    conf_threshold: float = 0.3,
    detections_df=None,
    progress_callback=None,
    write_manifest: bool = True,
    run_id: str = "",
) -> list[dict]:
    """Write one .mp4 per bout, organized by behavior name.

    This is the **recommended** export shape for downstream classifier
    training (e.g., BehaviorScope): each behavior name owns a folder,
    and inside each folder are the bout-clips that belong to that
    behavior. A clip is a contiguous slice of the source video covering
    ``[start_frame, end_frame]`` for one bout — distinct from the legacy
    ``export_sub_cluster_clips`` collated mode.

    Output layout::

        output_dir/
          forward_locomotion/
            video1__t0__b0001__f12-87.mp4
            video2__t0__b0023__f120-178.mp4
          lateral_creep/
            video1__t0__b0007__f200-260.mp4
          walking__subcluster_2/      # not yet named
            video1__t0__b0011__f600-650.mp4
          clip_manifest.csv

    Args:
        bouts: Iterable of bout dicts produced by
            ``aggregate_states_into_bouts`` (must carry ``directory``,
            ``start_frame``, ``end_frame``, ``state``, ``track_id``).
            Bouts whose ``state`` parses as noise (``"-1"``) are skipped.
        video_path_map: ``{pose_directory: video_file_path}`` lookup.
            Bouts whose directory has no entry are recorded in the
            manifest as skipped (no .mp4 written).
        output_dir: Destination root — per-behavior folders are created
            inside it; the manifest CSV lands at the root.
        class_names: ``{class_id: class_label}`` for fallback folder
            names when no human label was assigned via the naming dialog.
        state_names: ``{namespaced_label: human_name}`` from
            ``state_names.json``. When present, folders use the human
            names instead of ``class_X__subcluster_Y``.
        overlay: When True, draw bbox + keypoints + a label strip on
            each frame (uses ``detections_df`` if supplied). When
            False, write the raw video frames untouched. Default True.
        conf_threshold: Keypoint confidence floor for the overlay.
        detections_df: Frame-level dataframe (same one passed to the
            clusterer). Required when ``overlay=True`` so we can
            look up bbox/keypoints per frame. Optional otherwise.
        progress_callback: Optional ``callable(done, total, message)``
            that the GUI uses to drive its progress bar.
        write_manifest: When True (default), write
            ``clip_manifest.csv`` at ``output_dir`` listing every clip
            attempted. Disable for ad-hoc test runs.
        run_id: Optional UUID stamped into the manifest CSV so a later
            session can tell which discovery run a clip came from.

    Returns:
        List of dicts — one per bout — with at minimum::

            {
              "behavior_folder": "forward_locomotion",
              "namespaced_label": "0:1",
              "class_id": 0, "sub_id": 1,
              "video_basename": "video1",
              "start_frame": 12, "end_frame": 87,
              "duration_frames": 76,
              "track_id": 0,
              "clip_path": "<...>.mp4" or None,
              "status": "written" | "skipped" | "failed",
              "reason": "" or human-readable reason,
            }

    Raises:
        ImportError: If cv2 is missing.
        ValueError: If ``output_dir`` is empty.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:
        raise ImportError(
            "export_bout_clips: opencv-python (cv2) is required. "
            "`pip install opencv-python`."
        ) from exc

    if not output_dir:
        raise ValueError("export_bout_clips: output_dir is required.")

    os.makedirs(output_dir, exist_ok=True)
    class_names = class_names or {}
    state_names = state_names or {}
    video_path_map = video_path_map or {}

    bouts_list = list(bouts or [])
    total = len(bouts_list)
    manifest_rows: list[dict] = []

    if total == 0:
        logger.info("export_bout_clips: no bouts supplied; nothing to do.")
        if write_manifest:
            _write_manifest(output_dir, manifest_rows, run_id=run_id)
        return manifest_rows

    # Group bouts by source directory so we open each video at most
    # once. Within a directory, sort by start_frame for sequential reads.
    bouts_by_dir: dict[str, list[dict]] = defaultdict(list)
    for idx, bout in enumerate(bouts_list):
        # Stamp with a stable original index so the manifest preserves
        # input order even after we group/sort.
        bout = dict(bout)
        bout["_input_index"] = idx
        bouts_by_dir[str(bout.get("directory") or "")].append(bout)

    # Pre-build the frame index for overlay lookup. Indexed by
    # (directory, frame, track_id) so a single bout can render all its
    # frames without scanning the whole df each time.
    overlay_index = None
    if overlay and detections_df is not None and len(detections_df) > 0:
        overlay_index = _build_overlay_index(detections_df)

    completed = 0
    for directory, dir_bouts in bouts_by_dir.items():
        video_path = video_path_map.get(directory)
        if not video_path or not os.path.isfile(video_path):
            for bout in dir_bouts:
                manifest_rows.append(_skip_row(
                    bout, class_names, state_names,
                    reason=f"no source video for directory {directory!r}",
                ))
                completed += 1
                _report(progress_callback, completed, total, "Skipped (no video)")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            for bout in dir_bouts:
                manifest_rows.append(_skip_row(
                    bout, class_names, state_names,
                    reason=f"cv2 could not open {os.path.basename(video_path)}",
                ))
                completed += 1
                _report(progress_callback, completed, total, "Skipped (cv2 open failed)")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_basename = _safe_filename_part(
            os.path.splitext(os.path.basename(video_path))[0]
        )

        try:
            # Keep bouts in start_frame order so we can seek-once-and-read.
            dir_bouts.sort(key=lambda b: int(b.get("start_frame", 0)))
            for bout in dir_bouts:
                row = _write_one_bout(
                    cap, bout,
                    video_basename=video_basename,
                    output_dir=output_dir,
                    class_names=class_names,
                    state_names=state_names,
                    fourcc=fourcc, fps=fps, width=width, height=height,
                    overlay=overlay,
                    overlay_index=overlay_index,
                    directory=directory,
                    conf_threshold=conf_threshold,
                    cv2=cv2,
                )
                manifest_rows.append(row)
                completed += 1
                _report(
                    progress_callback, completed, total,
                    f"Wrote {row.get('status', 'unknown')}: "
                    f"{row.get('behavior_folder', '?')}/"
                    f"{os.path.basename(row.get('clip_path') or '')}",
                )
        finally:
            cap.release()

    # Restore input order in the manifest so users can correlate with
    # the bouts CSV index.
    manifest_rows.sort(key=lambda r: r.get("_input_index", 0))
    for r in manifest_rows:
        r.pop("_input_index", None)

    if write_manifest:
        _write_manifest(output_dir, manifest_rows, run_id=run_id)
    logger.info(
        "export_bout_clips: %d bout(s) processed; %d written, %d skipped, %d failed.",
        total,
        sum(1 for r in manifest_rows if r["status"] == "written"),
        sum(1 for r in manifest_rows if r["status"] == "skipped"),
        sum(1 for r in manifest_rows if r["status"] == "failed"),
    )
    return manifest_rows


def _build_overlay_index(detections_df) -> dict:
    """Build a ``{(directory, frame, track_id): row_dict}`` for overlay lookups.

    track_id may be missing in older runs — we fall back to a sentinel
    so single-track videos still render overlays.
    """
    has_track = "track_id" in detections_df.columns
    has_dir = "directory" in detections_df.columns
    keep_cols = [c for c in ("bbox", "keypoints") if c in detections_df.columns]
    index: dict = {}
    for _, row in detections_df.iterrows():
        try:
            frame = int(row["frame"])
        except Exception:
            continue
        track = row["track_id"] if has_track else 0
        directory = str(row["directory"]) if has_dir else ""
        record = {c: row[c] for c in keep_cols}
        index[(directory, frame, track)] = record
    return index


def _row_template(bout: dict, class_names: dict, state_names: dict) -> dict:
    parsed = _parse_namespaced_label(bout.get("state"))
    if parsed is None:
        class_id, sub_id = -1, -1
        folder = "noise"
    else:
        class_id, sub_id = parsed
        folder = _behavior_folder_name(
            class_id, sub_id, class_names=class_names, state_names=state_names,
        )
    return {
        "_input_index": bout.get("_input_index", 0),
        "behavior_folder": folder,
        "namespaced_label": (
            f"{class_id}:{sub_id}" if parsed is not None else str(bout.get("state", ""))
        ),
        "class_id": class_id,
        "sub_id": sub_id,
        "video_basename": "",
        "start_frame": int(bout.get("start_frame", 0) or 0),
        "end_frame": int(bout.get("end_frame", 0) or 0),
        "duration_frames": int(bout.get("duration_frames", 0) or 0),
        "track_id": bout.get("track_id", ""),
        "subject_id": bout.get("subject_id", ""),
        "directory": str(bout.get("directory", "")),
        "clip_path": None,
        "status": "skipped",
        "reason": "",
    }


def _skip_row(bout: dict, class_names: dict, state_names: dict, *, reason: str) -> dict:
    row = _row_template(bout, class_names, state_names)
    row["status"] = "skipped"
    row["reason"] = reason
    return row


def _write_one_bout(
    cap,
    bout: dict,
    *,
    video_basename: str,
    output_dir: str,
    class_names: dict,
    state_names: dict,
    fourcc,
    fps: float,
    width: int,
    height: int,
    overlay: bool,
    overlay_index,
    directory: str,
    conf_threshold: float,
    cv2,
) -> dict:
    """Write one bout's worth of contiguous frames as a single .mp4."""
    row = _row_template(bout, class_names, state_names)
    row["video_basename"] = video_basename

    parsed = _parse_namespaced_label(bout.get("state"))
    if parsed is None:
        row["status"] = "skipped"
        row["reason"] = "noise label"
        return row

    class_id, sub_id = parsed
    behavior_folder = row["behavior_folder"]
    target_dir = os.path.join(output_dir, behavior_folder)
    os.makedirs(target_dir, exist_ok=True)

    track_part = _safe_filename_part(str(bout.get("track_id", "0")))
    bout_idx_part = f"b{int(bout.get('_input_index', 0)):04d}"
    start = int(bout.get("start_frame", 0) or 0)
    end = int(bout.get("end_frame", start) or start)
    frames_part = f"f{start}-{end}"
    filename = (
        f"{video_basename}__t{track_part}__{bout_idx_part}__{frames_part}.mp4"
    )
    clip_path = os.path.join(target_dir, filename)
    row["clip_path"] = clip_path

    writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        row["status"] = "failed"
        row["reason"] = "cv2.VideoWriter failed to open"
        return row

    # 1-indexed convention: frame 1 is the first cv2 frame (zero index 0).
    seek_frame = max(0, start - 1)
    end_frame_zero = max(0, end - 1)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
        track_id = bout.get("track_id", 0)
        for frame_idx_zero in range(seek_frame, end_frame_zero + 1):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_one = frame_idx_zero + 1
            if overlay and overlay_index is not None:
                det = overlay_index.get((directory, frame_one, track_id))
                if det is not None:
                    frame = _annotate_frame(
                        frame.copy(), det,
                        class_id=class_id, sub_id=sub_id,
                        frame_idx=frame_one,
                        class_names=class_names,
                        state_names=state_names,
                        conf_threshold=conf_threshold,
                        cv2=cv2,
                    )
            writer.write(frame)
        row["status"] = "written"
    except Exception as exc:
        row["status"] = "failed"
        row["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            writer.release()
        except Exception:
            pass
    return row


def _report(callback, done: int, total: int, message: str) -> None:
    if callback is None:
        return
    try:
        callback(done, total, message)
    except Exception as exc:
        logger.debug("progress_callback raised: %s", exc)


def _write_manifest(output_dir: str, rows: list[dict], *, run_id: str = "") -> None:
    """Write ``clip_manifest.csv`` summarizing every clip attempt.

    Columns: status, behavior_folder, namespaced_label, class_id, sub_id,
    video_basename, track_id, subject_id, start_frame, end_frame,
    duration_frames, directory, clip_path, reason, run_id.
    """
    try:
        import csv  # noqa: PLC0415

        path = os.path.join(output_dir, CLIP_MANIFEST_FILENAME)
        cols = [
            "status", "behavior_folder", "namespaced_label", "class_id", "sub_id",
            "video_basename", "track_id", "subject_id",
            "start_frame", "end_frame", "duration_frames",
            "directory", "clip_path", "reason", "run_id",
        ]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "status": row.get("status", ""),
                    "behavior_folder": row.get("behavior_folder", ""),
                    "namespaced_label": row.get("namespaced_label", ""),
                    "class_id": row.get("class_id", ""),
                    "sub_id": row.get("sub_id", ""),
                    "video_basename": row.get("video_basename", ""),
                    "track_id": row.get("track_id", ""),
                    "subject_id": row.get("subject_id", ""),
                    "start_frame": row.get("start_frame", ""),
                    "end_frame": row.get("end_frame", ""),
                    "duration_frames": row.get("duration_frames", ""),
                    "directory": row.get("directory", ""),
                    "clip_path": row.get("clip_path", "") or "",
                    "reason": row.get("reason", ""),
                    "run_id": run_id,
                })
    except Exception as exc:
        logger.warning("Could not write clip manifest: %s", exc)


__all__ = [
    "export_sub_cluster_clips",
    "export_bout_clips",
    "CLIP_MANIFEST_FILENAME",
]

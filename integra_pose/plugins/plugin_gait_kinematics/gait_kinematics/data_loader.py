import os
import re
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


def _extract_frame_number(filename: str) -> int:
    stem = os.path.splitext(filename)[0]
    match = re.search(r'_(\d+)$', stem)
    if match:
        return int(match.group(1))
    return 0


def _is_int_like(value: float, tolerance: float = 1e-6) -> bool:
    return abs(value - round(value)) < tolerance


def _parse_pose_line(parts: list[str], n_keypoints: int):
    if len(parts) < 1 + 3 * n_keypoints:
        raise ValueError(f"Pose line too short ({len(parts)} tokens).")

    class_id = int(float(parts[0]))
    remainder = parts[1:]

    bbox = None
    if len(remainder) >= 4 + 3 * n_keypoints:
        bbox = [float(tok) for tok in remainder[:4]]
        remainder = remainder[4:]

    pose_tokens = remainder[-3 * n_keypoints :]
    pose_data = [float(tok) for tok in pose_tokens]
    prefix = remainder[:-3 * n_keypoints]

    bbox_conf = None
    track_id = None
    if prefix:
        candidate = float(prefix[-1])
        if _is_int_like(candidate):
            track_id = int(round(candidate))
            if track_id < 0:
                track_id = None
            prefix = prefix[:-1]
    if prefix:
        try:
            bbox_conf = float(prefix[-1])
        except ValueError:
            bbox_conf = None

    return class_id, bbox, pose_data, track_id, bbox_conf


def _compute_centroid(det: dict) -> tuple[np.ndarray | None, float]:
    if det.get("bbox"):
        x_c, y_c, _, _ = det["bbox"]
        centroid = np.array([x_c, y_c], dtype=float)
        threshold = max(det.get("video_width") or 1, det.get("video_height") or 1) * 0.08
        return centroid, threshold

    keypoints = det.get("keypoints") or {}
    coords = [
        np.array((kp[0], kp[1]), dtype=float)
        for kp in keypoints.values()
        if kp and not any(np.isnan(kp[:2]))
    ]
    if not coords:
        return None, 0.0
    centroid = np.vstack(coords).mean(axis=0)
    if det.get("video_width") and det.get("video_height"):
        threshold = max(det["video_width"], det["video_height"]) * 0.08
    else:
        threshold = 0.12
    return centroid, threshold


def _ensure_track_ids(detections: list[dict], max_frame_gap: int = 10) -> None:
    if not detections:
        return

    detections.sort(key=lambda d: d["frame"])
    existing = [
        int(round(float(det["track_id"])))
        for det in detections
        if det.get("track_id") is not None
        and _is_int_like(float(det["track_id"]))
        and int(round(float(det["track_id"]))) >= 0
    ]
    next_track_id = (max(existing) + 1) if existing else 0
    active_tracks: dict[int, dict] = {}
    detections_by_frame: dict[int, list[dict]] = defaultdict(list)

    for det in detections:
        detections_by_frame[det["frame"]].append(det)

    for frame in sorted(detections_by_frame):
        for track_id in list(active_tracks.keys()):
            if frame - active_tracks[track_id]["frame"] > max_frame_gap:
                del active_tracks[track_id]

        frame_dets = detections_by_frame[frame]
        to_assign: list[dict] = []

        for det in frame_dets:
            centroid, threshold = _compute_centroid(det)
            det["_centroid"] = centroid
            det["_threshold"] = threshold

            tid = det.get("track_id")
            if tid is not None and _is_int_like(float(tid)):
                tid = int(round(float(tid)))
                if tid >= 0:
                    det["track_id"] = tid
                    active_tracks[tid] = {
                        "centroid": centroid,
                        "threshold": threshold,
                        "frame": frame,
                    }
                    continue
            to_assign.append(det)

        available_tracks = {
            track_id: info
            for track_id, info in active_tracks.items()
            if info["centroid"] is not None
        }

        candidates = [det for det in to_assign if det["_centroid"] is not None]
        unmatched = [det for det in to_assign if det["_centroid"] is None]

        if available_tracks and candidates:
            track_ids = list(available_tracks.keys())
            track_centroids = np.stack([available_tracks[tid]["centroid"] for tid in track_ids])
            candidate_centroids = np.stack([det["_centroid"] for det in candidates])
            cost_matrix = np.linalg.norm(
                track_centroids[:, np.newaxis, :] - candidate_centroids[np.newaxis, :, :], axis=2
            )

            assigned = set()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                track_id = track_ids[r]
                det = candidates[c]
                distance = cost_matrix[r, c]
                max_distance = max(
                    available_tracks[track_id]["threshold"], det["_threshold"] or 0.0
                )
                if distance <= max_distance:
                    det["track_id"] = track_id
                    active_tracks[track_id] = {
                        "centroid": det["_centroid"],
                        "threshold": det["_threshold"],
                        "frame": frame,
                    }
                    assigned.add(c)

            candidates = [
                det for idx, det in enumerate(candidates) if idx not in assigned
            ]

        for det in candidates + unmatched:
            det["track_id"] = next_track_id
            active_tracks[next_track_id] = {
                "centroid": det["_centroid"],
                "threshold": det["_threshold"],
                "frame": frame,
            }
            next_track_id += 1

        for det in frame_dets:
            det.pop("_centroid", None)
            det.pop("_threshold", None)


def load_yolo_data(txt_dir, video_width, video_height, keypoint_order, conf_threshold, total_frames, video_base_name):
    """
    Load YOLO pose exports, supporting both predict (no tracking) and track outputs.
    """
    if not os.path.isdir(txt_dir):
        logger.error("YOLO text directory not found: %s", txt_dir)
        raise FileNotFoundError(f"YOLO text directory not found at: {txt_dir}")

    num_keypoints = len(keypoint_order)
    detections: list[dict] = []

    logger.info("Scanning for detections across %s frames for base '%s'...", total_frames, video_base_name)
    txt_files = [
        fname
        for fname in os.listdir(txt_dir)
        if fname.endswith(".txt") and fname.startswith(video_base_name)
    ]
    if not txt_files:
        logger.warning("No pose files matching base name '%s' found in %s.", video_base_name, txt_dir)
        return pd.DataFrame()

    for filename in sorted(txt_files, key=_extract_frame_number):
        frame_idx = _extract_frame_number(filename)
        filepath = os.path.join(txt_dir, filename)
        try:
            with open(filepath, "r") as fh:
                for line_num, line in enumerate(fh, 1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        class_id, bbox_norm, pose_norm, track_id, bbox_conf = _parse_pose_line(parts, num_keypoints)
                    except ValueError as exc:
                        logger.warning("Skipping malformed line %s in %s: %s", line_num, filename, exc)
                        continue

                    confidence = bbox_conf if bbox_conf is not None else 1.0
                    if confidence < conf_threshold:
                        continue

                    center_x_px = np.nan
                    center_y_px = np.nan
                    bbox_pixels = None
                    if bbox_norm:
                        bbox_pixels = [
                            bbox_norm[0] * video_width,
                            bbox_norm[1] * video_height,
                            bbox_norm[2] * video_width,
                            bbox_norm[3] * video_height,
                        ]
                        center_x_px = bbox_pixels[0]
                        center_y_px = bbox_pixels[1]

                    pose_data = np.array(pose_norm, dtype=float).reshape(num_keypoints, 3)
                    keypoints_px = pose_data[:, :2] * np.array([video_width, video_height])
                    keypoints_px[np.where((keypoints_px[:, 0] == 0) & (keypoints_px[:, 1] == 0))] = np.nan

                    record = {
                        "frame": frame_idx,
                        "track_id": track_id,
                        "behavior_id": class_id,
                        "confidence": confidence,
                        "center_x": center_x_px,
                        "center_y": center_y_px,
                        "bbox": bbox_pixels,
                        "video_width": video_width,
                        "video_height": video_height,
                        "keypoints": {
                            keypoint_order[idx]: (
                                keypoints_px[idx, 0],
                                keypoints_px[idx, 1],
                                pose_data[idx, 2],
                            )
                            for idx in range(num_keypoints)
                        },
                    }
                    detections.append(record)
        except Exception as exc:
            logger.error("Failed to process %s: %s", filepath, exc)

    if not detections:
        logger.warning("No valid detections were loaded.")
        return pd.DataFrame()

    _ensure_track_ids(detections)

    rows = []
    for det in detections:
        row = {
            "frame": det["frame"],
            "track_id": det["track_id"],
            "behavior_id": det["behavior_id"],
            "confidence": det["confidence"],
            "center_x": det["center_x"],
            "center_y": det["center_y"],
        }
        for name in keypoint_order:
            kp = det["keypoints"].get(name, (np.nan, np.nan, np.nan))
            row[f"{name}_x"] = kp[0]
            row[f"{name}_y"] = kp[1]
            row[f"{name}_conf"] = kp[2]
        rows.append(row)

    logger.info("Loaded %s detections from the video frames.", len(rows))
    return pd.DataFrame(rows)

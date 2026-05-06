"""Canonical motion metrics compiled from saved labels.csv outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import json
import math

import numpy as np
import pandas as pd

LogFn = Callable[[str, str], None]


@dataclass(slots=True)
class CanonicalMetricsResult:
    metrics_csv: Path
    metadata_json: Path


def _default_log(_message: str, _level: str = "INFO") -> None:
    return


def _coerce_object_id(raw_track, *, frame_index: int, detection_index: int) -> str:
    if raw_track is None:
        return f"frame{frame_index:06d}_det{detection_index}"
    if isinstance(raw_track, str):
        token = raw_track.strip()
        if not token:
            return f"frame{frame_index:06d}_det{detection_index}"
        try:
            return str(int(float(token)))
        except Exception:
            return token
    try:
        value = float(raw_track)
    except Exception:
        token = str(raw_track).strip()
        return token or f"frame{frame_index:06d}_det{detection_index}"
    if not np.isfinite(value):
        return f"frame{frame_index:06d}_det{detection_index}"
    return str(int(value))


def _bearing(dx: float, dy: float) -> float:
    if dx == 0.0 and dy == 0.0:
        return 0.0
    angle = math.degrees(math.atan2(dx, dy))
    return float((angle + 360.0) % 360.0)


def _angular_delta(a: float, b: float) -> float:
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


def _signed_angular_delta(a: float, b: float) -> float:
    return float(((b - a + 540.0) % 360.0) - 180.0)


def _extract_keypoint_prefixes(columns: list[str]) -> list[str]:
    prefixes: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if not str(column).startswith("kp_") or not str(column).endswith("_x_n"):
            continue
        prefix = str(column)[:-4]
        y_col = f"{prefix}_y_n"
        if y_col not in columns or prefix in seen:
            continue
        prefixes.append(prefix)
        seen.add(prefix)
    return prefixes


def compile_metrics_from_labels(
    *,
    labels_csv_path: Path,
    run_dir: Path,
    frame_width: int,
    frame_height: int,
    fps: float = 0.0,
    motion_direction_threshold_deg: float = 15.0,
    motion_velocity_threshold_px: float = 0.0,
    heading_indices: Optional[tuple[int, int]] = None,
    log_fn: Optional[LogFn] = None,
) -> CanonicalMetricsResult:
    log = log_fn or _default_log
    metrics_csv = run_dir / "metrics.csv"
    metadata_json = run_dir / "metrics_metadata.json"

    if not labels_csv_path.is_file():
        return CanonicalMetricsResult(metrics_csv=metrics_csv, metadata_json=metadata_json)

    try:
        df = pd.read_csv(labels_csv_path)
    except Exception as exc:
        log(f"Failed to load labels CSV for canonical metrics: {exc}", "WARNING")
        return CanonicalMetricsResult(metrics_csv=metrics_csv, metadata_json=metadata_json)

    required = {"frame", "x_center_n", "y_center_n"}
    if not required.issubset(set(df.columns)):
        log("Skipping canonical metrics: labels.csv missing required center columns.", "WARNING")
        return CanonicalMetricsResult(metrics_csv=metrics_csv, metadata_json=metadata_json)

    work = df.copy()
    work["frame"] = pd.to_numeric(work["frame"], errors="coerce")
    work["x_center_n"] = pd.to_numeric(work["x_center_n"], errors="coerce")
    work["y_center_n"] = pd.to_numeric(work["y_center_n"], errors="coerce")
    if "width_n" in work.columns:
        work["width_n"] = pd.to_numeric(work["width_n"], errors="coerce")
    if "height_n" in work.columns:
        work["height_n"] = pd.to_numeric(work["height_n"], errors="coerce")
    if "bbox_conf" in work.columns:
        work["bbox_conf"] = pd.to_numeric(work["bbox_conf"], errors="coerce")
    work = work.dropna(subset=["frame", "x_center_n", "y_center_n"]).copy()
    if work.empty:
        return CanonicalMetricsResult(metrics_csv=metrics_csv, metadata_json=metadata_json)

    work["frame"] = work["frame"].astype(int)
    work.sort_values(["frame"], inplace=True, kind="stable")
    work["det_index"] = work.groupby("frame").cumcount()

    fw = max(1.0, float(frame_width or 0.0))
    fh = max(1.0, float(frame_height or 0.0))
    cx_frame = fw / 2.0
    cy_frame = fh / 2.0

    kp_prefixes = _extract_keypoint_prefixes([str(col) for col in work.columns])
    kp_x_cols = [f"{prefix}_x_n" for prefix in kp_prefixes]
    kp_y_cols = [f"{prefix}_y_n" for prefix in kp_prefixes]
    for col in kp_x_cols + kp_y_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    threshold = max(0.0, float(motion_direction_threshold_deg))
    velocity_floor = max(0.0, float(motion_velocity_threshold_px))
    heading_pair = None
    if heading_indices and len(kp_prefixes) > max(heading_indices):
        heading_pair = (int(heading_indices[0]), int(heading_indices[1]))

    base_rows: list[dict[str, object]] = []
    for row in work.itertuples(index=False):
        frame_index = int(getattr(row, "frame"))
        det_index = int(getattr(row, "det_index"))
        center_x = float(getattr(row, "x_center_n")) * fw
        center_y = float(getattr(row, "y_center_n")) * fh
        anchor_x = center_x
        anchor_y = center_y
        orientation_bearing = _bearing(center_x - cx_frame, cy_frame - center_y)
        body_length = np.nan
        body_aspect = np.nan

        kp_points: list[tuple[float, float]] = []
        for x_col, y_col in zip(kp_x_cols, kp_y_cols):
            kp_x = getattr(row, x_col, np.nan)
            kp_y = getattr(row, y_col, np.nan)
            if pd.notna(kp_x) and pd.notna(kp_y):
                kp_points.append((float(kp_x) * fw, float(kp_y) * fh))
            else:
                kp_points.append((float("nan"), float("nan")))

        if heading_pair is not None:
            try:
                kp_root = kp_points[heading_pair[0]]
                kp_tip = kp_points[heading_pair[1]]
                if all(np.isfinite(value) for value in (*kp_root, *kp_tip)):
                    anchor_x = float(kp_root[0] + kp_tip[0]) / 2.0
                    anchor_y = float(kp_root[1] + kp_tip[1]) / 2.0
                    vec_x = float(kp_tip[0] - kp_root[0])
                    vec_y = float(kp_tip[1] - kp_root[1])
                    orientation_bearing = _bearing(vec_x, -vec_y)
                    body_length = float(math.hypot(vec_x, vec_y))
            except Exception:
                pass

        finite_points = [(x, y) for x, y in kp_points if np.isfinite(x) and np.isfinite(y)]
        if finite_points:
            xs = [x for x, _ in finite_points]
            ys = [y for _, y in finite_points]
            width = float(max(xs) - min(xs))
            height = float(max(ys) - min(ys))
            minor = max(min(width, height), 1e-3)
            if np.isfinite(body_length):
                body_aspect = float(body_length / minor)
            else:
                body_aspect = float(max(width, height) / minor) if minor > 0 else np.nan

        raw_track = getattr(row, "track_id", None) if "track_id" in work.columns else None
        object_id = _coerce_object_id(raw_track, frame_index=frame_index, detection_index=det_index)

        class_id = 0
        if "class_id" in work.columns:
            try:
                raw_class = getattr(row, "class_id")
                if pd.notna(raw_class):
                    class_id = int(raw_class)
            except Exception:
                class_id = 0

        confidence = np.nan
        if "bbox_conf" in work.columns:
            raw_conf = getattr(row, "bbox_conf")
            if pd.notna(raw_conf):
                confidence = float(raw_conf)

        base_rows.append(
            {
                "frame": frame_index,
                "object_id": object_id,
                "class_id": class_id,
                "confidence": confidence,
                "anchor_x_px": float(anchor_x),
                "anchor_y_px": float(anchor_y),
                "distance_from_frame_center_px": float(math.hypot(anchor_x - cx_frame, anchor_y - cy_frame)),
                "orientation_deg": float(orientation_bearing),
                "body_length_px": body_length,
                "body_aspect_ratio": body_aspect,
            }
        )

    metrics_df = pd.DataFrame(base_rows)
    if metrics_df.empty:
        return CanonicalMetricsResult(metrics_csv=metrics_csv, metadata_json=metadata_json)

    metrics_df["objects_in_frame"] = metrics_df.groupby("frame")["object_id"].transform("count").astype(int)
    metrics_df["pairwise_anchor_distances_px"] = ""
    metrics_df["nearest_object_id"] = ""
    metrics_df["nearest_distance_px"] = np.nan

    for frame_value, frame_group in metrics_df.groupby("frame", sort=False):
        if len(frame_group) <= 1:
            continue
        frame_indices = list(frame_group.index)
        anchors = frame_group[["anchor_x_px", "anchor_y_px"]].to_numpy(dtype=float)
        object_ids = frame_group["object_id"].astype(str).tolist()
        distance_map: dict[int, list[tuple[str, float]]] = {idx: [] for idx in frame_indices}
        for first in range(len(frame_indices)):
            for second in range(first + 1, len(frame_indices)):
                first_idx = frame_indices[first]
                second_idx = frame_indices[second]
                first_anchor = anchors[first]
                second_anchor = anchors[second]
                distance = float(math.hypot(first_anchor[0] - second_anchor[0], first_anchor[1] - second_anchor[1]))
                distance_map[first_idx].append((object_ids[second], distance))
                distance_map[second_idx].append((object_ids[first], distance))
        for idx in frame_indices:
            neighbors = distance_map[idx]
            if not neighbors:
                continue
            neighbors_sorted = sorted(neighbors, key=lambda item: item[0])
            metrics_df.at[idx, "pairwise_anchor_distances_px"] = ";".join(
                f"{other_id}={distance:.6f}" for other_id, distance in neighbors_sorted
            )
            nearest_id, nearest_distance = min(neighbors, key=lambda item: item[1])
            metrics_df.at[idx, "nearest_object_id"] = nearest_id
            metrics_df.at[idx, "nearest_distance_px"] = float(nearest_distance)

    metrics_df["angular_velocity_deg_per_frame"] = np.nan
    metrics_df["signed_angular_velocity_deg_per_frame"] = np.nan
    metrics_df["movement_speed_px_per_frame"] = 0.0
    metrics_df["acceleration_px_per_frame2"] = np.nan
    metrics_df["movement_heading_deg"] = np.nan
    metrics_df["total_path_length_px"] = 0.0
    metrics_df["turn_count"] = 0

    metrics_df.sort_values(["object_id", "frame"], inplace=True, kind="stable")
    for object_id, group in metrics_df.groupby("object_id", sort=False):
        last_center = None
        total_distance = 0.0
        last_movement_bearing = None
        direction_shifts = 0
        last_velocity = None
        last_orientation = None
        for idx in group.index:
            anchor_x = float(metrics_df.at[idx, "anchor_x_px"])
            anchor_y = float(metrics_df.at[idx, "anchor_y_px"])
            orientation = float(metrics_df.at[idx, "orientation_deg"])
            velocity = 0.0
            movement_bearing = np.nan
            acceleration = np.nan
            angular_velocity = np.nan
            signed_angular_velocity = np.nan

            if last_center is not None:
                dx = anchor_x - last_center[0]
                dy = anchor_y - last_center[1]
                step_distance = float(math.hypot(dx, dy))
                if step_distance > 0.0 and step_distance >= velocity_floor:
                    velocity = step_distance
                    movement_bearing = float(_bearing(dx, -dy))
                    if last_movement_bearing is not None:
                        delta = _angular_delta(last_movement_bearing, movement_bearing)
                        if delta >= threshold:
                            direction_shifts += 1
                    last_movement_bearing = movement_bearing
                    total_distance += step_distance
                    if last_velocity is not None:
                        acceleration = float(velocity - last_velocity)
                    last_velocity = velocity
                else:
                    last_velocity = 0.0
            else:
                last_movement_bearing = None
                last_velocity = 0.0

            if last_orientation is not None:
                signed_angular_velocity = float(_signed_angular_delta(last_orientation, orientation))
                angular_velocity = float(abs(signed_angular_velocity))
            last_orientation = orientation
            last_center = (anchor_x, anchor_y)

            metrics_df.at[idx, "movement_speed_px_per_frame"] = float(velocity)
            metrics_df.at[idx, "acceleration_px_per_frame2"] = acceleration
            metrics_df.at[idx, "movement_heading_deg"] = movement_bearing
            metrics_df.at[idx, "total_path_length_px"] = float(total_distance)
            metrics_df.at[idx, "turn_count"] = int(direction_shifts)
            metrics_df.at[idx, "angular_velocity_deg_per_frame"] = angular_velocity
            metrics_df.at[idx, "signed_angular_velocity_deg_per_frame"] = signed_angular_velocity

    metrics_df.sort_values(["frame", "object_id"], inplace=True, kind="stable")

    try:
        metrics_df.to_csv(metrics_csv, index=False)
    except Exception as exc:
        log(f"Failed to write canonical metrics.csv: {exc}", "WARNING")
        return CanonicalMetricsResult(metrics_csv=metrics_csv, metadata_json=metadata_json)

    metadata = {
        "schema_version": 1,
        "source": "labels_csv_canonical_compiler",
        "labels_csv": str(labels_csv_path),
        "frame_width": int(frame_width or 0),
        "frame_height": int(frame_height or 0),
        "fps": float(fps or 0.0),
        "motion_direction_threshold_deg": float(motion_direction_threshold_deg),
        "motion_velocity_threshold_px": float(motion_velocity_threshold_px),
        "heading_indices": [int(idx) for idx in heading_indices] if heading_indices else None,
    }
    try:
        metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception as exc:
        log(f"Failed to write canonical metrics metadata: {exc}", "WARNING")

    return CanonicalMetricsResult(metrics_csv=metrics_csv, metadata_json=metadata_json)

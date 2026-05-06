import ast
import csv
import cv2
import pandas as pd
import numpy as np
import os
import yaml
from tqdm import tqdm
import re
import logging
import time
import json
import textwrap
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

class BoutAnalysisError(Exception):
    """Custom exception for errors during bout analysis."""
    pass


class BoutAnalysisCancelledError(BoutAnalysisError):
    """Raised when a cooperative stop request interrupts analytics."""
    pass


def _is_stop_requested(stop_check: Optional[Callable[[], bool]]) -> bool:
    if stop_check is None:
        return False
    try:
        return bool(stop_check())
    except Exception:
        return False


def _raise_if_stop_requested(stop_check: Optional[Callable[[], bool]], stage: str) -> None:
    if _is_stop_requested(stop_check):
        raise BoutAnalysisCancelledError(f"{stage} cancelled.")


def _check_stop_interval(
    stop_check: Optional[Callable[[], bool]],
    index: int,
    *,
    stage: str,
    interval: int = 256,
) -> None:
    if stop_check is None:
        return
    if index <= 1 or (index % max(1, int(interval))) == 0:
        _raise_if_stop_requested(stop_check, stage)

def _get_frame_from_filename_robust(filename):
    """
    Robustly extracts a frame number from a filename.
    Looks for digits following the last underscore in the filename.
    """
    match = re.search(r'_(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))

    matches = re.findall(r'\d+', filename)
    if matches:
        return int(matches[-1])
        
    return None

def _normalize_class_names(names: Any) -> Dict[Any, str]:
    """Coerce class name payloads from YAML into an indexable mapping."""
    if isinstance(names, dict):
        normalized: Dict[Any, str] = {}
        for key, value in names.items():
            try:
                parsed_key = int(key)
            except (TypeError, ValueError):
                parsed_key = key
            normalized[parsed_key] = str(value)
        return normalized
    if isinstance(names, (list, tuple)):
        return {idx: str(value) for idx, value in enumerate(names)}
    return {}


def _normalize_roi_polygons(rois: Any) -> Dict[str, List[np.ndarray]]:
    """
    Coerce ROI polygon payloads from YAML into the structure expected by analytics:
    {roi_name: [np.ndarray(points), ...]}
    """
    normalized: Dict[str, List[np.ndarray]] = {}
    if not isinstance(rois, dict):
        return normalized

    def _iter_polygons(payload: Any) -> Iterable[Iterable[Tuple[float, float]]]:
        if payload is None:
            return []
        if isinstance(payload, dict):
            values = payload.get('polygons', payload.values())
            return _iter_polygons(values)
        if isinstance(payload, np.ndarray):
            if payload.ndim == 2 and payload.shape[0] >= 3 and payload.shape[1] == 2:
                return [payload]
            if payload.ndim == 3 and payload.shape[1] >= 3 and payload.shape[2] == 2:
                return [poly for poly in payload]
            return []
        if isinstance(payload, (list, tuple)):
            if not payload:
                return []
            first = payload[0]
            if isinstance(first, np.ndarray):
                if first.ndim == 2 and first.shape[0] >= 3 and first.shape[1] == 2:
                    return payload
                if first.ndim == 1 and first.shape[0] == 2:
                    return [payload]
            if isinstance(first, (list, tuple)) and len(first) and isinstance(first[0], (list, tuple, np.ndarray)):
                # Already a list of polygons
                return payload
            # Treat as single polygon
            return [payload]
        return []

    for name, payload in rois.items():
        polygons: List[np.ndarray] = []
        for polygon in _iter_polygons(payload):
            poly_arr = np.asarray(polygon, dtype=np.int32)
            if poly_arr.ndim != 2 or poly_arr.shape[0] < 3 or poly_arr.shape[1] != 2:
                continue
            polygons.append(poly_arr)
        if polygons:
            normalized[str(name)] = polygons
    return normalized


def _load_yaml_config(yaml_path: Any) -> Tuple[Dict[Any, str], Dict[str, List[np.ndarray]]]:
    if not yaml_path:
        return {}, {}
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise BoutAnalysisError(f"YAML file not found at {yaml_path}") from exc
    except Exception as exc:
        raise BoutAnalysisError(f"Error parsing YAML file: {exc}") from exc
    return (
        _normalize_class_names(config_data.get('names', {})),
        _normalize_roi_polygons(config_data.get('rois', {})),
    )


def _load_labels_csv_row_map(yolo_txt_folder: str) -> Dict[int, List[Dict[str, Optional[float]]]]:
    csv_path = os.path.join(yolo_txt_folder, "labels.csv")
    if not os.path.isfile(csv_path):
        return {}

    row_map: Dict[int, List[Dict[str, Optional[float]]]] = defaultdict(list)
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    frame = int(float(row.get("frame", "")))
                except (TypeError, ValueError):
                    continue
                raw_conf = row.get("bbox_conf", "")
                conf_value: Optional[float] = None
                if raw_conf not in (None, ""):
                    try:
                        parsed = float(raw_conf)
                    except (TypeError, ValueError):
                        conf_value = None
                    else:
                        conf_value = parsed if np.isfinite(parsed) else None
                raw_track_id = row.get("track_id", "")
                track_value: Optional[int] = None
                if raw_track_id not in (None, ""):
                    try:
                        parsed_track = float(raw_track_id)
                    except (TypeError, ValueError):
                        track_value = None
                    else:
                        if np.isfinite(parsed_track):
                            rounded_track = round(parsed_track)
                            if abs(parsed_track - rounded_track) <= 1e-6:
                                track_value = int(rounded_track)
                row_map[frame].append(
                    {
                        "bbox_conf": conf_value,
                        "track_id": track_value,
                    }
                )
    except Exception as exc:
        logger.warning("Failed to load labels.csv row map from %s: %s", csv_path, exc)
        return {}
    return row_map


def _infer_class_names_from_dataframe(per_frame_df: pd.DataFrame, class_names: Dict[Any, str]) -> Dict[Any, str]:
    resolved = dict(class_names or {})
    if per_frame_df is None or per_frame_df.empty or 'class_id' not in per_frame_df.columns:
        return resolved
    try:
        class_ids = (
            pd.to_numeric(per_frame_df['class_id'], errors='coerce')
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
    except Exception:
        return resolved
    for class_id in sorted(class_ids):
        if class_id in resolved or str(class_id) in resolved:
            continue
        resolved[class_id] = f"Class_{class_id}"
    return resolved


def load_and_preprocess_data(
    yaml_path,
    yolo_txt_folder,
    single_animal_mode=False,
    class_names=None,
    roi_polygons=None,
):
    """
    Load detection text files into a DataFrame with optional YAML-backed metadata.

    Explicit class names / ROI polygons override any YAML-derived metadata.
    When single_animal_mode=True, the highest-confidence valid detection line from each
    frame file is used when confidence is available; otherwise the first valid detection
    is kept as a fallback.

    Returns: (per_frame_df, roi_polygons, class_names)
    """
    yaml_class_names: Dict[Any, str] = {}
    yaml_roi_polygons: Dict[str, List[np.ndarray]] = {}
    if yaml_path:
        try:
            yaml_class_names, yaml_roi_polygons = _load_yaml_config(yaml_path)
        except BoutAnalysisError:
            if class_names or roi_polygons:
                logger.warning(
                    "Falling back to explicit analytics metadata because YAML could not be loaded: %s",
                    yaml_path,
                )
            else:
                raise
    class_names = _normalize_class_names(class_names) or yaml_class_names
    roi_polygons = _normalize_roi_polygons(roi_polygons) or yaml_roi_polygons

    all_files = [f for f in os.listdir(yolo_txt_folder) if f.endswith('.txt')]
    if not all_files:
        raise BoutAnalysisError(f"No .txt files found in {yolo_txt_folder}")

    def _frame_sort_key(filename):
        frame_idx = _get_frame_from_filename_robust(filename)
        return (frame_idx if frame_idx is not None else float('inf'), filename)

    all_files = sorted(all_files, key=_frame_sort_key)

    skipped_files = []
    skipped_lines = 0
    labels_csv_row_map = _load_labels_csv_row_map(yolo_txt_folder)
    single_animal_confidence_warning_emitted = False

    def _looks_like_track_id(value: float) -> bool:
        if value is None or not np.isfinite(value):
            return False
        rounded = round(value)
        return abs(value - rounded) <= 1e-6 and 0 <= rounded <= 1e9

    def _looks_like_confidence(value: float) -> bool:
        return value is not None and np.isfinite(value) and 0.0 <= float(value) <= 1.0

    def _matches_fallback_confidence(value: float, fallback_confidence: Optional[float]) -> bool:
        if fallback_confidence is None or not _looks_like_confidence(value):
            return False
        try:
            return abs(float(value) - float(fallback_confidence)) <= 1e-6
        except (TypeError, ValueError):
            return False

    def _matches_fallback_track_id(value: float, fallback_track_id: Optional[int]) -> bool:
        if fallback_track_id is None or not _looks_like_track_id(value):
            return False
        try:
            return int(round(float(value))) == int(fallback_track_id)
        except (TypeError, ValueError):
            return False

    def _prefer_ambiguous_single_suffix(
        value: float,
        *,
        fallback_confidence: Optional[float],
        fallback_track_id: Optional[int],
    ) -> str:
        """
        Resolve the rare case where a trailing value like ``0`` or ``1`` could be
        either confidence or track_id. Prefer metadata matches when available;
        otherwise default to track IDs because tracked TXT exports are the common
        analytics path and exact confidence values of 0/1 are comparatively rare.
        """
        track_match = _matches_fallback_track_id(value, fallback_track_id)
        conf_match = _matches_fallback_confidence(value, fallback_confidence)

        if track_match and not conf_match:
            return "track"
        if conf_match and not track_match:
            return "conf"
        if fallback_track_id is not None and fallback_confidence is None:
            return "track"
        if fallback_confidence is not None and fallback_track_id is None:
            return "conf"
        return "track"

    def _try_keypoint_payload(values: List[float]) -> List[Tuple[List[float], int]]:
        candidates: List[Tuple[List[float], int]] = []
        if not values:
            return candidates
        if len(values) % 3 == 0:
            candidates.append((values, 3))
        if len(values) % 2 == 0:
            candidates.append((values, 2))
        return candidates

    def _split_remainder(
        values: List[float],
        *,
        fallback_confidence: Optional[float] = None,
        fallback_track_id: Optional[int] = None,
    ) -> Tuple[Optional[float], List[float], int, int]:
        """
        Returns (det_confidence, keypoint_values, keypoint_dim, track_id)
        Track ID defaults to 0 when not present.
        Supports the common Ultralytics layouts for detect and pose TXT rows:
        class box [conf] [track_id]
        class box keypoints... [conf] [track_id]
        """
        if not values:
            return fallback_confidence, [], 2, int(fallback_track_id or 0)

        # Detect-only outputs have at most two trailing fields after the box.
        if len(values) <= 2:
            if len(values) == 1:
                suffix = values[0]
                if _looks_like_track_id(suffix) and not _looks_like_confidence(suffix):
                    return fallback_confidence, [], 2, int(round(suffix))
                if _looks_like_track_id(suffix) and _looks_like_confidence(suffix):
                    preferred_kind = _prefer_ambiguous_single_suffix(
                        suffix,
                        fallback_confidence=fallback_confidence,
                        fallback_track_id=fallback_track_id,
                    )
                    if preferred_kind == "conf":
                        return float(suffix), [], 2, int(fallback_track_id or 0)
                    return fallback_confidence, [], 2, int(round(suffix))
                if _looks_like_confidence(suffix):
                    return float(suffix), [], 2, int(fallback_track_id or 0)
                raise ValueError("Unable to interpret detect-only suffix payload.")

            conf_candidate, track_candidate = values
            if _looks_like_track_id(track_candidate):
                conf_value = float(conf_candidate) if _looks_like_confidence(conf_candidate) else fallback_confidence
                return conf_value, [], 2, int(round(track_candidate))
            raise ValueError("Unable to interpret detect-only confidence/track payload.")

        # Pose or keypoint-bearing outputs.
        last_val = values[-1]
        second_last = values[-2] if len(values) >= 2 else None
        candidates: List[Dict[str, object]] = []

        def _add_candidates(
            kp_payload: List[float],
            *,
            confidence: Optional[float],
            track_id: int,
            stripped_count: int,
            kind: str,
        ) -> None:
            for kp_vals, kp_dim in _try_keypoint_payload(kp_payload):
                kp_count = len(kp_vals) // kp_dim if kp_dim else 0
                candidates.append(
                    {
                        "confidence": confidence,
                        "kp_vals": kp_vals,
                        "kp_dim": kp_dim,
                        "track_id": int(track_id),
                        "stripped_count": int(stripped_count),
                        "kind": kind,
                        "kp_count": int(kp_count),
                    }
                )

        _add_candidates(
            values,
            confidence=fallback_confidence,
            track_id=int(fallback_track_id or 0),
            stripped_count=0,
            kind="no_suffix",
        )

        if second_last is not None and _looks_like_track_id(last_val) and _looks_like_confidence(second_last):
            _add_candidates(
                values[:-2],
                confidence=float(second_last),
                track_id=int(round(last_val)),
                stripped_count=2,
                kind="conf_track",
            )

        if _looks_like_track_id(last_val):
            _add_candidates(
                values[:-1],
                confidence=fallback_confidence,
                track_id=int(round(last_val)),
                stripped_count=1,
                kind="track",
            )

        if _looks_like_confidence(last_val):
            _add_candidates(
                values[:-1],
                confidence=float(last_val),
                track_id=int(fallback_track_id or 0),
                stripped_count=1,
                kind="conf",
            )

        if candidates:
            multi_point_candidates = [candidate for candidate in candidates if int(candidate["kp_count"]) >= 2]
            if any(int(candidate["kp_dim"]) == 3 for candidate in multi_point_candidates):
                pool = [candidate for candidate in multi_point_candidates if int(candidate["kp_dim"]) == 3]
            elif multi_point_candidates:
                pool = multi_point_candidates
            else:
                pool = candidates

            kind_priority = {
                "no_suffix": 0,
                "track": 1,
                "conf": 2,
                "conf_track": 3,
            }
            if _looks_like_track_id(last_val) and _looks_like_confidence(last_val):
                preferred_kind = _prefer_ambiguous_single_suffix(
                    last_val,
                    fallback_confidence=fallback_confidence,
                    fallback_track_id=fallback_track_id,
                )
                if preferred_kind == "conf":
                    kind_priority["conf"] = 1
                    kind_priority["track"] = 2
            best = min(
                pool,
                key=lambda candidate: (
                    int(candidate["stripped_count"]),
                    kind_priority.get(str(candidate["kind"]), 99),
                    -int(candidate["kp_count"]),
                    -int(candidate["kp_dim"]),
                ),
            )
            return (
                best["confidence"],  # type: ignore[return-value]
                best["kp_vals"],  # type: ignore[return-value]
                int(best["kp_dim"]),
                int(best["track_id"]),
            )

        raise ValueError("Unable to interpret YOLO remainder payload.")

    records = []
    for filename in all_files:
        try:
            frame_num = None
            try:
                frame_num = _get_frame_from_filename_robust(filename)  # type: ignore[name-defined]
            except Exception:
                pass
            if frame_num is None:
                import re as _re
                m1 = _re.search(r'frame_(\d+)\.txt$', filename)
                m2 = _re.search(r'_(\d+)\.txt$', filename)
                m3 = _re.search(r'(\d+)\.txt$', filename)
                for m_ in (m1, m2, m3):
                    if m_:
                        frame_num = int(m_.group(1)); break
            if frame_num is None:
                logging.warning("Skipping detection file without frame index: %s", filename)
                skipped_files.append(filename)
                continue

            filepath = os.path.join(yolo_txt_folder, filename)
            frame_records = []
            frame_row_meta = labels_csv_row_map.get(int(frame_num), [])
            valid_detection_idx = 0
            with open(filepath, 'r') as fh:
                for line_idx, raw_line in enumerate(fh, 1):
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    if len(parts) < 3:
                        skipped_lines += 1
                        continue
                    try:
                        values = [float(part) for part in parts]
                    except ValueError:
                        skipped_lines += 1
                        continue

                    class_id = int(round(values[0])) if len(values) > 0 else 0
                    x_center = values[1] if len(values) > 1 else np.nan
                    y_center = values[2] if len(values) > 2 else np.nan
                    w = values[3] if len(values) > 3 else np.nan
                    h = values[4] if len(values) > 4 else np.nan

                    remainder = values[5:]
                    row_meta = frame_row_meta[valid_detection_idx] if valid_detection_idx < len(frame_row_meta) else {}
                    fallback_confidence = row_meta.get("bbox_conf") if isinstance(row_meta, dict) else None
                    fallback_track_id = row_meta.get("track_id") if isinstance(row_meta, dict) else None
                    try:
                        det_confidence, kp_vals, keypoint_dim, track_id = _split_remainder(
                            remainder,
                            fallback_confidence=fallback_confidence,
                            fallback_track_id=fallback_track_id if isinstance(fallback_track_id, int) else None,
                        )
                    except ValueError:
                        skipped_lines += 1
                        continue

                    step = keypoint_dim if keypoint_dim in (2, 3) else 3
                    keypoints = []
                    for i in range(0, len(kp_vals), step):
                        if i + step <= len(kp_vals):
                            kp_tuple = tuple(kp_vals[i:i + step])
                            keypoints.append(kp_tuple)

                    record = {
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'w': w,
                        'h': h,
                        'track_id': track_id,
                        'frame': frame_num,
                        'keypoints': keypoints,
                        'confidence': det_confidence,
                        'keypoint_dim': keypoint_dim,
                    }
                    valid_detection_idx += 1
                    if single_animal_mode:
                        frame_records.append(record)
                    else:
                        records.append(record)
            if single_animal_mode and frame_records:
                confidences = [
                    float(entry['confidence'])
                    for entry in frame_records
                    if entry.get('confidence') is not None and np.isfinite(entry['confidence'])
                ]
                if confidences:
                    best_record = max(
                        frame_records,
                        key=lambda entry: float(entry['confidence']) if entry.get('confidence') is not None and np.isfinite(entry['confidence']) else -np.inf,
                    )
                else:
                    best_record = frame_records[0]
                    if len(frame_records) > 1 and not single_animal_confidence_warning_emitted:
                        logger.warning(
                            "Single-animal mode found frames with multiple detections but no confidence metadata in %s. "
                            "Falling back to first valid detection for those frames.",
                            yolo_txt_folder,
                        )
                        single_animal_confidence_warning_emitted = True
                records.append(best_record)
        except FileNotFoundError:
            continue
        except Exception:
            continue

    if skipped_files:
        examples = ", ".join(sorted(set(skipped_files))[:3])
        logging.warning("Skipped %d detection files lacking frame numbers%s",
                        len(skipped_files), f" (e.g. {examples})" if examples else '')
    if skipped_lines:
        logging.warning("Skipped %d malformed detection lines while parsing YOLO TXT files.", skipped_lines)

    if not records:
        raise BoutAnalysisError("No valid detection files could be processed. Please check filename format and content.")

    full_df = pd.DataFrame.from_records(records)
    full_df['track_id'] = pd.to_numeric(full_df['track_id'], errors='coerce').fillna(0).astype(int)
    full_df = full_df.sort_values(by=['track_id', 'frame']).reset_index(drop=True)

    if not single_animal_mode and not full_df.empty:
        try:
            has_only_default_track = full_df['track_id'].nunique(dropna=False) == 1 and int(full_df['track_id'].iloc[0]) == 0
        except Exception:
            has_only_default_track = False
        if has_only_default_track:
            try:
                max_detections_per_frame = int(full_df.groupby('frame').size().max())
            except Exception:
                max_detections_per_frame = 1
            if max_detections_per_frame > 1:
                raise BoutAnalysisError(
                    "Multiple detections per frame were found but no track IDs were detected. "
                    "Multi-animal bout analytics requires running inference with tracking enabled (mode=track / Use Tracker), "
                    "or enable 'Single Animal Analysis' to force all detections onto track ID 0."
                )

    return full_df, roi_polygons, _infer_class_names_from_dataframe(full_df, class_names)


def _build_presence_segments(
    frames: Sequence[int],
    *,
    gap_threshold: int = 0,
    min_duration: int = 1,
) -> list[tuple[int, int]]:
    ordered = sorted({int(frame) for frame in frames})
    if not ordered:
        return []

    max_gap = max(0, int(gap_threshold or 0))
    min_len = max(1, int(min_duration or 1))
    segments: list[tuple[int, int]] = []
    start = ordered[0]
    end = ordered[0]

    for frame in ordered[1:]:
        if frame - end <= max_gap + 1:
            end = frame
            continue
        if end - start + 1 >= min_len:
            segments.append((start, end))
        start = frame
        end = frame

    if end - start + 1 >= min_len:
        segments.append((start, end))
    return segments


def assign_roi_membership(
    per_frame_df,
    roi_polygons,
    video_width,
    video_height,
    entry_threshold=0.75,
    exit_threshold=0.25,
    event_mode: str = "tab6_hybrid",
    keypoint_index: int = 0,
    keypoint_indices: Optional[List[int]] = None,
    keypoint_ratio_threshold: float = 0.5,
    max_gap_frames: int = 0,
    min_dwell_frames: int = 1,
    stop_check: Optional[Callable[[], bool]] = None,
):
    """Assigns each detection to an ROI using area-overlap thresholds and records entry/exit events."""

    if per_frame_df.empty or not roi_polygons or video_width <= 0 or video_height <= 0:
        df_copy = per_frame_df.copy()
        df_copy['ROI Name'] = ''
        return df_copy, {'entries': [], 'exits': []}

    roi_masks: Dict[str, np.ndarray] = {}
    roi_info: List[Tuple[str, np.ndarray, int]] = []
    for name, polygons in roi_polygons.items():
        if not polygons:
            continue
        mask = np.zeros((video_height, video_width), dtype=np.uint8)
        for polygon in polygons:
            poly_np = np.array(polygon, dtype=np.int32)
            if poly_np.ndim != 2 or poly_np.shape[0] < 3 or poly_np.shape[1] != 2:
                continue
            cv2.fillPoly(mask, [poly_np], 1)
        if mask.any():
            roi_masks[name] = mask
            roi_info.append((name, mask, int(mask.sum())))

    if not roi_masks:
        df_copy = per_frame_df.copy()
        df_copy['ROI Name'] = ''
        return df_copy, {'entries': [], 'exits': []}

    # Sort by area ascending so inner ROIs (smaller area) get priority for primary assignment.
    roi_info.sort(key=lambda item: item[2])
    sorted_roi_names = [name for name, _mask, _area in roi_info]
    _ = sorted_roi_names  # retained for backward-compatible debugging variables

    mode = str(event_mode or "tab6_hybrid").strip().lower()
    if mode not in {"tab6_hybrid", "bbox_only", "keypoint_index", "keypoint_any", "keypoint_ratio"}:
        mode = "tab6_hybrid"
    kp_idx = max(0, int(keypoint_index or 0))
    kp_idx_list = [max(0, int(idx)) for idx in (keypoint_indices or []) if isinstance(idx, (int, np.integer, float))]
    kp_ratio_threshold = max(0.0, min(1.0, float(keypoint_ratio_threshold if keypoint_ratio_threshold is not None else 0.5)))
    gap_threshold = max(0, int(max_gap_frames or 0))
    min_dwell_frames = max(1, int(min_dwell_frames or 1))

    def _to_pixel_coordinate(value, scale):
        if pd.isna(value):
            return None
        value = float(value)
        if 0.0 <= value <= 1.0:
            return value * scale
        return value

    def _to_pixel_length(value, scale):
        if pd.isna(value):
            return 0.0
        value = float(value)
        if value <= 0:
            return 0.0
        if 0.0 < value <= 1.0:
            return value * scale
        return value

    def _collect_coverage(mask, x1, y1, x2, y2):
        if x2 <= x1 or y2 <= y1:
            return 0.0, 0
        sub_region = mask[y1:y2, x1:x2]
        covered_pixels = int(sub_region.sum())
        area = max((y2 - y1) * (x2 - x1), 1)
        return covered_pixels / area, covered_pixels

    df_sorted = (
        per_frame_df
        .sort_values(['track_id', 'frame'])
        .reset_index()
        .rename(columns={'index': 'orig_index'})
    )

    assignments = {}
    track_states = {}
    entry_events = []
    exit_events = []
    memberships_map: Dict[int, Tuple[str, ...]] = {}

    for row_idx, (_, row) in enumerate(df_sorted.iterrows(), start=1):
        _check_stop_interval(stop_check, row_idx, stage="ROI membership assignment")
        track_id = row['track_id']
        frame_num = int(row['frame'])
        orig_index = row['orig_index']

        state = track_states.setdefault(
            track_id,
            {
                'active_rois': {},
                'last_frame': None,
            },
        )

        keypoints = row.get('keypoints', None)
        kp_entries: List[Tuple[float, float, float]] = []
        kp_coords = []
        if isinstance(keypoints, (list, tuple)):
            for kp_entry in keypoints:
                if not isinstance(kp_entry, (list, tuple)) or len(kp_entry) < 2:
                    continue
                kp_x, kp_y = kp_entry[0], kp_entry[1]
                kp_conf = kp_entry[2] if len(kp_entry) > 2 else 1.0
                kp_entries.append((kp_x, kp_y, kp_conf))
                try:
                    kp_conf_value = float(kp_conf)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(kp_conf_value) or kp_conf_value <= 0:
                    continue
                px = _to_pixel_coordinate(kp_x, video_width)
                py = _to_pixel_coordinate(kp_y, video_height)
                if px is None or py is None:
                    continue
                kp_coords.append((px, py))

        x_center_val = _to_pixel_coordinate(row.get('x_center'), video_width)
        y_center_val = _to_pixel_coordinate(row.get('y_center'), video_height)
        bbox_width_px = _to_pixel_length(row.get('w', np.nan), video_width)
        bbox_height_px = _to_pixel_length(row.get('h', np.nan), video_height)
        valid_bbox = (
            x_center_val is not None
            and y_center_val is not None
            and bbox_width_px > 0
            and bbox_height_px > 0
        )

        # Hybrid ROI scoring should keep using the detector box when it exists.
        # Pose keypoints add evidence, but they should not replace the box geometry.
        bbox_x1 = bbox_y1 = bbox_x2 = bbox_y2 = 0
        bbox_center_x = x_center_val if x_center_val is not None else None
        bbox_center_y = y_center_val if y_center_val is not None else None
        if valid_bbox:
            half_w = bbox_width_px / 2.0
            half_h = bbox_height_px / 2.0
            bbox_x1 = int(np.clip(round(x_center_val - half_w), 0, video_width - 1))
            bbox_x2 = int(np.clip(round(x_center_val + half_w), 0, video_width - 1))
            bbox_y1 = int(np.clip(round(y_center_val - half_h), 0, video_height - 1))
            bbox_y2 = int(np.clip(round(y_center_val + half_h), 0, video_height - 1))
            if bbox_x2 <= bbox_x1:
                bbox_x2 = min(video_width, bbox_x1 + 1)
            if bbox_y2 <= bbox_y1:
                bbox_y2 = min(video_height, bbox_y1 + 1)

        kp_hull_x1 = kp_hull_y1 = kp_hull_x2 = kp_hull_y2 = 0
        kp_center_x = kp_center_y = None
        if kp_coords:
            xs = [coord[0] for coord in kp_coords]
            ys = [coord[1] for coord in kp_coords]
            kp_hull_x1 = int(np.clip(np.floor(min(xs)), 0, video_width - 1))
            kp_hull_x2 = int(np.clip(np.ceil(max(xs)), 0, video_width - 1))
            kp_hull_y1 = int(np.clip(np.floor(min(ys)), 0, video_height - 1))
            kp_hull_y2 = int(np.clip(np.ceil(max(ys)), 0, video_height - 1))
            kp_center_x = float(np.mean(xs))
            kp_center_y = float(np.mean(ys))

        center_x_for_roi = bbox_center_x if valid_bbox else kp_center_x
        center_y_for_roi = bbox_center_y if valid_bbox else kp_center_y
        if center_x_for_roi is None:
            center_x_for_roi = x_center_val
        if center_y_for_roi is None:
            center_y_for_roi = y_center_val

        cx_idx = int(np.clip(round(center_x_for_roi), 0, video_width - 1)) if center_x_for_roi is not None else None
        cy_idx = int(np.clip(round(center_y_for_roi), 0, video_height - 1)) if center_y_for_roi is not None else None

        coverage_info = {}
        center_inside_map = {}
        for roi_name, mask in roi_masks.items():
            center_inside = False
            if cx_idx is not None and cy_idx is not None:
                center_inside = bool(mask[cy_idx, cx_idx])
            center_inside_map[roi_name] = center_inside

            if bbox_x2 > bbox_x1 and bbox_y2 > bbox_y1:
                bbox_ratio, pixels = _collect_coverage(mask, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            elif kp_hull_x2 > kp_hull_x1 and kp_hull_y2 > kp_hull_y1:
                bbox_ratio, pixels = _collect_coverage(mask, kp_hull_x1, kp_hull_y1, kp_hull_x2, kp_hull_y2)
            else:
                bbox_ratio = 1.0 if center_inside else 0.0
                pixels = int(center_inside)

            if kp_coords:
                kp_inside = 0
                for px, py in kp_coords:
                    xi = int(np.clip(round(px), 0, video_width - 1))
                    yi = int(np.clip(round(py), 0, video_height - 1))
                    if mask[yi, xi]:
                        kp_inside += 1
                kp_ratio = kp_inside / len(kp_coords) if kp_coords else 0.0
            else:
                kp_ratio = 0.0

            combined_ratio = max(bbox_ratio, kp_ratio)
            kp_any_inside = False
            kp_index_inside = False
            kp_set_ratio = 0.0
            valid_set_points = 0
            inside_set_points = 0

            if kp_entries:
                for idx, (raw_x, raw_y, raw_conf) in enumerate(kp_entries):
                    try:
                        raw_conf_value = float(raw_conf)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(raw_conf_value) or raw_conf_value <= 0:
                        continue
                    px = _to_pixel_coordinate(raw_x, video_width)
                    py = _to_pixel_coordinate(raw_y, video_height)
                    if px is None or py is None:
                        continue
                    xi = int(np.clip(round(px), 0, video_width - 1))
                    yi = int(np.clip(round(py), 0, video_height - 1))
                    inside = bool(mask[yi, xi])
                    if inside:
                        kp_any_inside = True
                    if idx == kp_idx and inside:
                        kp_index_inside = True
                    if idx in kp_idx_list:
                        valid_set_points += 1
                        if inside:
                            inside_set_points += 1
                if kp_idx_list:
                    kp_set_ratio = (inside_set_points / valid_set_points) if valid_set_points > 0 else 0.0

            coverage_info[roi_name] = {
                "combined_ratio": combined_ratio,
                "pixels": pixels,
                "kp_ratio": kp_ratio,
                "bbox_ratio": bbox_ratio,
                "center_inside": center_inside,
                "kp_any_inside": kp_any_inside,
                "kp_index_inside": kp_index_inside,
                "kp_set_ratio": kp_set_ratio,
                "has_keypoints": bool(kp_coords),
            }

        active_rois_prev = state['active_rois']
        active_rois_next = {}
        membership_names: List[str] = []
        preferred_roi = ''
        preferred_score = -1.0
        preferred_area = float('inf')

        for roi_name, mask, area in roi_info:
            prev_info = active_rois_prev.get(roi_name)
            prev_active = prev_info is not None
            cinfo = coverage_info.get(roi_name, {})
            box_ratio = float(cinfo.get("bbox_ratio", 0.0))
            combined_ratio = float(cinfo.get("combined_ratio", box_ratio))
            kp_ratio = float(cinfo.get("kp_ratio", 0.0))
            center_inside = bool(cinfo.get("center_inside", center_inside_map.get(roi_name, False)))
            kp_any_inside = bool(cinfo.get("kp_any_inside", False))
            kp_index_inside = bool(cinfo.get("kp_index_inside", False))
            kp_set_ratio = float(cinfo.get("kp_set_ratio", 0.0))
            has_keypoints = bool(cinfo.get("has_keypoints", False))

            if mode == "bbox_only":
                should_enter = center_inside or box_ratio >= entry_threshold
                should_stay = center_inside or box_ratio > exit_threshold
                current_score = (2.0 if center_inside else 0.0) + box_ratio
            elif mode == "keypoint_index":
                if has_keypoints:
                    should_enter = kp_index_inside
                    should_stay = kp_index_inside
                    current_score = (2.0 if kp_index_inside else 0.0) + kp_ratio
                else:
                    should_enter = center_inside or box_ratio >= entry_threshold
                    should_stay = center_inside or box_ratio > exit_threshold
                    current_score = (2.0 if center_inside else 0.0) + box_ratio
            elif mode == "keypoint_any":
                if has_keypoints:
                    should_enter = kp_any_inside
                    should_stay = kp_any_inside
                    current_score = (2.0 if kp_any_inside else 0.0) + kp_ratio
                else:
                    should_enter = center_inside or box_ratio >= entry_threshold
                    should_stay = center_inside or box_ratio > exit_threshold
                    current_score = (2.0 if center_inside else 0.0) + box_ratio
            elif mode == "keypoint_ratio":
                if has_keypoints:
                    should_enter = kp_set_ratio >= kp_ratio_threshold if kp_idx_list else kp_ratio >= kp_ratio_threshold
                    stay_threshold = min(exit_threshold, kp_ratio_threshold) if exit_threshold >= 0 else kp_ratio_threshold
                    should_stay = kp_set_ratio > stay_threshold if kp_idx_list else kp_ratio > stay_threshold
                    current_score = (2.0 if center_inside else 0.0) + max(kp_set_ratio, kp_ratio)
                else:
                    should_enter = center_inside or box_ratio >= entry_threshold
                    should_stay = center_inside or box_ratio > exit_threshold
                    current_score = (2.0 if center_inside else 0.0) + box_ratio
            else:
                should_enter = center_inside or box_ratio >= entry_threshold or kp_ratio >= entry_threshold
                should_stay = center_inside or box_ratio > exit_threshold or kp_ratio > exit_threshold
                current_score = (2.0 if center_inside else 0.0) + combined_ratio + kp_ratio

            if prev_active:
                if should_stay:
                    start_frame = prev_info['start_frame']
                    active_rois_next[roi_name] = {'start_frame': start_frame, 'last_frame': frame_num}
                    membership_names.append(roi_name)
                else:
                    exit_events.append({'frame': frame_num, 'track_id': track_id, 'roi_name': roi_name})
                if should_stay and (
                    current_score > preferred_score
                    or (abs(current_score - preferred_score) <= 1e-6 and area < preferred_area)
                ):
                    preferred_roi = roi_name
                    preferred_score = current_score
                    preferred_area = area
                continue

            if should_enter:
                active_rois_next[roi_name] = {'start_frame': frame_num, 'last_frame': frame_num}
                membership_names.append(roi_name)
                entry_events.append({'frame': frame_num, 'track_id': track_id, 'roi_name': roi_name})
                if current_score > preferred_score or (abs(current_score - preferred_score) <= 1e-6 and area < preferred_area):
                    preferred_roi = roi_name
                    preferred_score = current_score
                    preferred_area = area

        # If no ROI selected for primary but memberships exist, pick the smallest-area active ROI.
        if not preferred_roi and membership_names:
            for roi_name in membership_names:
                for name, _mask, area in roi_info:
                    if name == roi_name:
                        preferred_roi = name
                        preferred_area = area
                        break
                if preferred_roi:
                    break

        assignments[orig_index] = preferred_roi or ''
        memberships_map[orig_index] = tuple(membership_names)
        state['active_rois'] = active_rois_next
        state['last_frame'] = frame_num

    # Finalize lingering ROI memberships
    for track_id, state in track_states.items():
        last_frame = state.get('last_frame')
        for roi_name, info in state.get('active_rois', {}).items():
            exit_frame = info.get('last_frame', last_frame)
            if exit_frame is None:
                exit_frame = last_frame
            exit_events.append({'frame': int(exit_frame or 0), 'track_id': track_id, 'roi_name': roi_name})

    df_result = per_frame_df.copy()
    df_result['ROI Name'] = df_result.index.map(assignments).fillna('')
    # Store memberships as a plain object column; Index.map() auto-converts tuple outputs
    # into a MultiIndex (and crashes when empty tuples exist).
    df_result['ROI Memberships'] = [memberships_map.get(idx, ()) for idx in df_result.index]
    presence_frames_by_track = defaultdict(lambda: defaultdict(list))
    for _, row in df_sorted.iterrows():
        track_id = int(row['track_id'])
        frame_num = int(row['frame'])
        orig_index = int(row['orig_index'])
        for roi_name in memberships_map.get(orig_index, ()):
            presence_frames_by_track[track_id][str(roi_name)].append(frame_num)

    debounced_entries = []
    debounced_exits = []
    for track_idx, track_id in enumerate(sorted(presence_frames_by_track.keys()), start=1):
        _check_stop_interval(stop_check, track_idx, stage="ROI event summarization", interval=32)
        for roi_name in sorted(presence_frames_by_track[track_id].keys()):
            segments = _build_presence_segments(
                presence_frames_by_track[track_id][roi_name],
                gap_threshold=gap_threshold,
                min_duration=min_dwell_frames,
            )
            for start_frame, end_frame in segments:
                debounced_entries.append({'frame': int(start_frame), 'track_id': int(track_id), 'roi_name': str(roi_name)})
                debounced_exits.append({'frame': int(end_frame), 'track_id': int(track_id), 'roi_name': str(roi_name)})

    event_log = {'entries': debounced_entries, 'exits': debounced_exits}
    return df_result, event_log


def compute_object_interactions(
    per_frame_df,
    object_roi_polygons,
    video_width,
    video_height,
    *,
    keypoint_index: int = 0,
    distance_threshold_px: float = 0.0,
    fps: float = 30.0,
    max_gap_frames: int = 0,
    min_dwell_frames: int = 1,
    stop_check: Optional[Callable[[], bool]] = None,
):
    """
    Compute object interaction metrics using a keypoint-centric proximity rule.

    Interaction is positive when the selected keypoint is either:
    - inside an object ROI, or
    - within ``distance_threshold_px`` from the object ROI boundary.
    """

    per_frame_columns = [
        "Object Interaction ROI",
        "Object Interaction Memberships",
        "Object Contact ROI",
        "Object Contact Memberships",
        "Object Proximity ROI",
        "Object Proximity Memberships",
        "Object Interaction State",
        "Object Interaction Distance (px)",
    ]
    summary_columns = [
        "Object ROI",
        "Entries",
        "Exits",
        "Frames Interacting",
        "Time Interacting (s)",
        "Frames Contact",
        "Time Contact (s)",
        "Frames Proximity Only",
        "Time Proximity Only (s)",
        "Dwell Events",
        "Mean Dwell Duration (frames)",
        "Mean Dwell Duration (s)",
        "Median Dwell Duration (frames)",
        "Median Dwell Duration (s)",
    ]
    per_track_columns = [
        "Track ID",
        "Object ROI",
        "Entries",
        "Exits",
        "Frames Interacting",
        "Time Interacting (s)",
        "Frames Contact",
        "Time Contact (s)",
        "Frames Proximity Only",
        "Time Proximity Only (s)",
        "Dwell Events",
        "Mean Dwell Duration (frames)",
        "Mean Dwell Duration (s)",
        "Median Dwell Duration (frames)",
        "Median Dwell Duration (s)",
    ]
    dwell_columns = [
        "Track ID",
        "Object ROI",
        "Start Frame",
        "End Frame",
        "Duration (Frames)",
        "Duration (s)",
    ]
    approach_retreat_summary_columns = [
        "Object ROI",
        "Approach Events",
        "Retreat Events",
        "Approach Frames",
        "Retreat Frames",
        "Approach Time (s)",
        "Retreat Time (s)",
        "Mean Approach Rate (px/frame)",
        "Mean Retreat Rate (px/frame)",
        "Net Distance Change (px)",
    ]
    approach_retreat_track_columns = [
        "Track ID",
        "Object ROI",
        "Approach Events",
        "Retreat Events",
        "Approach Frames",
        "Retreat Frames",
        "Approach Time (s)",
        "Retreat Time (s)",
        "Mean Approach Rate (px/frame)",
        "Mean Retreat Rate (px/frame)",
        "Net Distance Change (px)",
    ]
    approach_retreat_event_columns = [
        "Track ID",
        "Object ROI",
        "State",
        "Start Frame",
        "End Frame",
        "Duration (Frames)",
        "Duration (s)",
        "Mean Delta Distance (px/frame)",
        "Total Distance Change (px)",
    ]

    empty_summary = pd.DataFrame(columns=summary_columns)
    empty_per_track = pd.DataFrame(columns=per_track_columns)
    empty_dwell = pd.DataFrame(columns=dwell_columns)
    empty_ar_summary = pd.DataFrame(columns=approach_retreat_summary_columns)
    empty_ar_track = pd.DataFrame(columns=approach_retreat_track_columns)
    empty_ar_events = pd.DataFrame(columns=approach_retreat_event_columns)

    df_base = per_frame_df.copy() if per_frame_df is not None else pd.DataFrame()
    for col in per_frame_columns:
        if col not in df_base.columns:
            if col in {"Object Interaction Memberships", "Object Contact Memberships", "Object Proximity Memberships"}:
                df_base[col] = [() for _ in range(len(df_base))]
            elif col == "Object Interaction Distance (px)":
                df_base[col] = np.nan
            else:
                df_base[col] = ""

    if (
        df_base.empty
        or "track_id" not in df_base.columns
        or "frame" not in df_base.columns
        or video_width <= 0
        or video_height <= 0
    ):
        return {
            "per_frame_df": df_base,
            "summary": empty_summary,
            "per_track": empty_per_track,
            "dwell_events": empty_dwell,
            "approach_retreat_summary": empty_ar_summary,
            "approach_retreat_by_track": empty_ar_track,
            "approach_retreat_events": empty_ar_events,
            "events": {"entries": [], "exits": []},
        }

    work_df = df_base.copy()
    work_df["track_id"] = pd.to_numeric(work_df["track_id"], errors="coerce")
    work_df["frame"] = pd.to_numeric(work_df["frame"], errors="coerce")
    work_df = work_df.dropna(subset=["track_id", "frame"])
    if work_df.empty:
        return {
            "per_frame_df": df_base,
            "summary": empty_summary,
            "per_track": empty_per_track,
            "dwell_events": empty_dwell,
            "approach_retreat_summary": empty_ar_summary,
            "approach_retreat_by_track": empty_ar_track,
            "approach_retreat_events": empty_ar_events,
            "events": {"entries": [], "exits": []},
        }
    work_df["track_id"] = work_df["track_id"].astype(int)
    work_df["frame"] = work_df["frame"].astype(int)

    roi_map = _normalize_roi_polygons(object_roi_polygons)
    if not roi_map:
        return {
            "per_frame_df": df_base,
            "summary": empty_summary,
            "per_track": empty_per_track,
            "dwell_events": empty_dwell,
            "approach_retreat_summary": empty_ar_summary,
            "approach_retreat_by_track": empty_ar_track,
            "approach_retreat_events": empty_ar_events,
            "events": {"entries": [], "exits": []},
        }

    object_info: List[Tuple[str, List[np.ndarray], np.ndarray, int]] = []
    for name, polygons in roi_map.items():
        if not polygons:
            continue
        mask = np.zeros((video_height, video_width), dtype=np.uint8)
        valid_polys: List[np.ndarray] = []
        for polygon in polygons:
            poly_np = np.asarray(polygon, dtype=np.int32)
            if poly_np.ndim != 2 or poly_np.shape[0] < 3 or poly_np.shape[1] != 2:
                continue
            valid_polys.append(poly_np)
            cv2.fillPoly(mask, [poly_np], 1)
        area = int(mask.sum())
        if valid_polys and area > 0:
            object_info.append((str(name), valid_polys, mask, area))
    if not object_info:
        return {
            "per_frame_df": df_base,
            "summary": empty_summary,
            "per_track": empty_per_track,
            "dwell_events": empty_dwell,
            "approach_retreat_summary": empty_ar_summary,
            "approach_retreat_by_track": empty_ar_track,
            "approach_retreat_events": empty_ar_events,
            "events": {"entries": [], "exits": []},
        }

    object_info.sort(key=lambda row: row[3])
    kp_idx = max(0, int(keypoint_index or 0))
    threshold_px = max(0.0, float(distance_threshold_px or 0.0))
    has_valid_fps = bool(fps and fps > 0)
    gap_threshold = max(0, int(max_gap_frames or 0))
    min_dwell_frames = max(1, int(min_dwell_frames or 1))

    def _to_pixel_coordinate(value, scale):
        if pd.isna(value):
            return None
        value = float(value)
        if 0.0 <= value <= 1.0:
            return value * scale
        return value

    def _point_from_row(row) -> Tuple[Optional[float], Optional[float], str]:
        keypoints = row.get("keypoints", None)
        if isinstance(keypoints, (list, tuple)) and len(keypoints) > kp_idx:
            selected = keypoints[kp_idx]
            if isinstance(selected, (list, tuple)) and len(selected) >= 2:
                conf = selected[2] if len(selected) > 2 else 1.0
                try:
                    conf_value = float(conf)
                except (TypeError, ValueError):
                    return None, None, "none"
                if np.isfinite(conf_value) and conf_value > 0:
                    px = _to_pixel_coordinate(selected[0], video_width)
                    py = _to_pixel_coordinate(selected[1], video_height)
                    if px is None or py is None:
                        return None, None, "none"
                    try:
                        px_value = float(px)
                        py_value = float(py)
                    except (TypeError, ValueError):
                        return None, None, "none"
                    if np.isfinite(px_value) and np.isfinite(py_value):
                        return px_value, py_value, "keypoint"
        return None, None, "none"

    per_frame_sorted = (
        work_df.sort_values(["track_id", "frame"])
        .reset_index()
        .rename(columns={"index": "__orig_index__"})
    )
    assignments: Dict[int, str] = {}
    memberships_map: Dict[int, Tuple[str, ...]] = {}
    contact_assignments: Dict[int, str] = {}
    contact_memberships_map: Dict[int, Tuple[str, ...]] = {}
    proximity_assignments: Dict[int, str] = {}
    proximity_memberships_map: Dict[int, Tuple[str, ...]] = {}
    state_map: Dict[int, str] = {}
    distance_map: Dict[int, float] = {}

    entries = Counter()
    exits = Counter()
    frame_counts = Counter()
    contact_frame_counts = Counter()
    proximity_frame_counts = Counter()
    dwell_records: List[Dict[str, Any]] = []
    distance_records: List[Dict[str, Any]] = []
    entry_events: List[Dict[str, Any]] = []
    exit_events: List[Dict[str, Any]] = []
    per_track_entries = defaultdict(Counter)
    per_track_exits = defaultdict(Counter)
    per_track_frames = defaultdict(Counter)
    per_track_contact_frames = defaultdict(Counter)
    per_track_proximity_frames = defaultdict(Counter)
    interaction_frames_by_track = defaultdict(lambda: defaultdict(list))

    track_state: Dict[Any, Dict[str, Any]] = {}

    def _close_active(track_id, object_name: str, info: Dict[str, int]) -> None:
        start_frame = int(info.get("start_frame", 0))
        end_frame = int(info.get("last_frame", start_frame))
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        duration_frames = max(end_frame - start_frame + 1, 1)
        if duration_frames < min_dwell_frames:
            return
        duration_s = (duration_frames / fps) if has_valid_fps else np.nan
        dwell_records.append(
            {
                "Track ID": track_id,
                "Object ROI": object_name,
                "Start Frame": start_frame,
                "End Frame": end_frame,
                "Duration (Frames)": duration_frames,
                "Duration (s)": duration_s,
            }
        )

    for row_idx, (_, row) in enumerate(per_frame_sorted.iterrows(), start=1):
        _check_stop_interval(stop_check, row_idx, stage="Object interaction analysis")
        track_id = row.get("track_id")
        frame = int(row.get("frame"))
        orig_index = int(row.get("__orig_index__"))
        state = track_state.setdefault(
            track_id,
            {
                "active": {},
                "last_frame": None,
                "last_point": None,
                "last_point_frame": None,
            },
        )
        last_frame = state.get("last_frame")
        active = state.get("active", {})
        if last_frame is not None and frame - int(last_frame) > gap_threshold + 1:
            for object_name, info in list(active.items()):
                exits[object_name] += 1
                per_track_exits[track_id][object_name] += 1
                exit_events.append(
                    {
                        "frame": int(info.get("last_frame", info.get("start_frame", frame))),
                        "track_id": track_id,
                        "object_roi": object_name,
                    }
                )
                _close_active(track_id, object_name, info)
            active.clear()

        px, py, _point_source = _point_from_row(row)
        if px is not None and py is not None:
            state["last_point"] = (float(px), float(py))
            state["last_point_frame"] = frame
        else:
            last_point = state.get("last_point")
            last_point_frame = state.get("last_point_frame")
            if (
                last_point is not None
                and last_point_frame is not None
                and gap_threshold > 0
                and (frame - int(last_point_frame)) <= gap_threshold
            ):
                px, py = last_point
                _point_source = "carry_forward"
        memberships: List[str] = []
        contact_memberships: List[str] = []
        proximity_memberships: List[str] = []
        closest_name = ""
        closest_distance = np.nan
        best_distance = np.inf
        best_area = np.inf

        if px is not None and py is not None:
            xi = int(np.clip(round(px), 0, video_width - 1))
            yi = int(np.clip(round(py), 0, video_height - 1))
            for object_name, polygons, mask, area in object_info:
                inside = bool(mask[yi, xi])
                signed_distance = -np.inf
                for poly in polygons:
                    try:
                        dist = float(cv2.pointPolygonTest(poly, (float(px), float(py)), True))
                    except Exception:
                        continue
                    if dist > signed_distance:
                        signed_distance = dist
                if not np.isfinite(signed_distance):
                    continue
                distance_to_roi = 0.0 if signed_distance >= 0 else abs(signed_distance)
                contact = bool(inside)
                proximity_only = (not contact) and (distance_to_roi <= threshold_px)
                interacting = contact or proximity_only
                distance_records.append(
                    {
                        "track_id": track_id,
                        "frame": frame,
                        "object_roi": object_name,
                        "distance_px": float(distance_to_roi),
                        "interacting": bool(interacting),
                        "contact": bool(contact),
                        "proximity_only": bool(proximity_only),
                    }
                )
                if interacting:
                    memberships.append(object_name)
                    frame_counts[object_name] += 1
                    per_track_frames[track_id][object_name] += 1
                    interaction_frames_by_track[track_id][object_name].append(frame)
                if contact:
                    contact_memberships.append(object_name)
                    contact_frame_counts[object_name] += 1
                    per_track_contact_frames[track_id][object_name] += 1
                elif proximity_only:
                    proximity_memberships.append(object_name)
                    proximity_frame_counts[object_name] += 1
                    per_track_proximity_frames[track_id][object_name] += 1
                if distance_to_roi < best_distance or (
                    abs(distance_to_roi - best_distance) <= 1e-6 and area < best_area
                ):
                    best_distance = distance_to_roi
                    best_area = area
                    closest_name = object_name
                    closest_distance = float(distance_to_roi)
        else:
            for object_name, _polygons, _mask, _area in object_info:
                distance_records.append(
                    {
                        "track_id": track_id,
                        "frame": frame,
                        "object_roi": object_name,
                        "distance_px": np.nan,
                        "interacting": False,
                    }
                )

        memberships_tuple = tuple(sorted(set(memberships)))
        contact_tuple = tuple(sorted(set(contact_memberships)))
        proximity_tuple = tuple(sorted(set(proximity_memberships)))
        primary_name = closest_name if closest_name and (closest_name in memberships_tuple) else ""
        contact_name = closest_name if closest_name and (closest_name in contact_tuple) else (contact_tuple[0] if contact_tuple else "")
        proximity_name = (
            closest_name if closest_name and (closest_name in proximity_tuple) else (proximity_tuple[0] if proximity_tuple else "")
        )
        assignments[orig_index] = primary_name
        memberships_map[orig_index] = memberships_tuple
        contact_assignments[orig_index] = contact_name
        contact_memberships_map[orig_index] = contact_tuple
        proximity_assignments[orig_index] = proximity_name
        proximity_memberships_map[orig_index] = proximity_tuple
        state_map[orig_index] = "contact" if contact_tuple else ("proximity" if proximity_tuple else "")
        distance_map[orig_index] = float(closest_distance) if np.isfinite(closest_distance) else np.nan

        current_memberships = set(memberships_tuple)
        for object_name in current_memberships:
            if object_name in active:
                active[object_name]["last_frame"] = frame
            else:
                active[object_name] = {"start_frame": frame, "last_frame": frame}
                entries[object_name] += 1
                per_track_entries[track_id][object_name] += 1
                entry_events.append({"frame": frame, "track_id": track_id, "object_roi": object_name})

        stale_names = [name for name in active.keys() if name not in current_memberships]
        for object_name in stale_names:
            info = active.pop(object_name)
            exits[object_name] += 1
            per_track_exits[track_id][object_name] += 1
            exit_events.append({"frame": frame, "track_id": track_id, "object_roi": object_name})
            _close_active(track_id, object_name, info)
        state["last_frame"] = frame

    for track_id, state in track_state.items():
        active = state.get("active", {})
        for object_name, info in active.items():
            exits[object_name] += 1
            per_track_exits[track_id][object_name] += 1
            exit_events.append(
                {
                    "frame": int(info.get("last_frame", info.get("start_frame", 0))),
                    "track_id": track_id,
                    "object_roi": object_name,
                }
            )
            _close_active(track_id, object_name, info)

    entries = Counter()
    exits = Counter()
    dwell_records = []
    entry_events = []
    exit_events = []
    per_track_entries = defaultdict(Counter)
    per_track_exits = defaultdict(Counter)
    for track_idx, track_id in enumerate(sorted(interaction_frames_by_track.keys()), start=1):
        _check_stop_interval(stop_check, track_idx, stage="Object interaction event summarization", interval=32)
        for object_name in sorted(interaction_frames_by_track[track_id].keys()):
            segments = _build_presence_segments(
                interaction_frames_by_track[track_id][object_name],
                gap_threshold=gap_threshold,
                min_duration=min_dwell_frames,
            )
            for start_frame, end_frame in segments:
                entries[object_name] += 1
                exits[object_name] += 1
                per_track_entries[track_id][object_name] += 1
                per_track_exits[track_id][object_name] += 1
                entry_events.append({"frame": int(start_frame), "track_id": track_id, "object_roi": object_name})
                exit_events.append({"frame": int(end_frame), "track_id": track_id, "object_roi": object_name})
                _close_active(
                    track_id,
                    object_name,
                    {"start_frame": int(start_frame), "last_frame": int(end_frame)},
                )

    df_result = df_base.copy()
    df_result["Object Interaction ROI"] = df_result.index.map(assignments).fillna("")
    df_result["Object Interaction Memberships"] = [memberships_map.get(idx, ()) for idx in df_result.index]
    df_result["Object Contact ROI"] = df_result.index.map(contact_assignments).fillna("")
    df_result["Object Contact Memberships"] = [contact_memberships_map.get(idx, ()) for idx in df_result.index]
    df_result["Object Proximity ROI"] = df_result.index.map(proximity_assignments).fillna("")
    df_result["Object Proximity Memberships"] = [proximity_memberships_map.get(idx, ()) for idx in df_result.index]
    df_result["Object Interaction State"] = df_result.index.map(state_map).fillna("")
    df_result["Object Interaction Distance (px)"] = df_result.index.map(distance_map)

    dwell_df = pd.DataFrame(dwell_records, columns=dwell_columns)
    if not dwell_df.empty:
        dwell_df = dwell_df.sort_values(["Object ROI", "Track ID", "Start Frame"]).reset_index(drop=True)
    dwell_by_object = dwell_df.groupby("Object ROI") if not dwell_df.empty else None
    dwell_by_track = dwell_df.groupby(["Track ID", "Object ROI"]) if not dwell_df.empty else None

    summary_rows: List[Dict[str, Any]] = []
    object_names = [name for name, _polys, _mask, _area in object_info]
    for object_name in sorted(object_names):
        total_frames = int(frame_counts.get(object_name, 0))
        events_df = dwell_by_object.get_group(object_name) if dwell_by_object is not None and object_name in dwell_by_object.groups else None
        dwell_events_count = int(len(events_df)) if events_df is not None else 0
        mean_dwell_frames = float(events_df["Duration (Frames)"].mean()) if events_df is not None and dwell_events_count else np.nan
        mean_dwell_s = float(events_df["Duration (s)"].mean()) if events_df is not None and dwell_events_count and has_valid_fps else np.nan
        median_dwell_frames = float(events_df["Duration (Frames)"].median()) if events_df is not None and dwell_events_count else np.nan
        median_dwell_s = float(events_df["Duration (s)"].median()) if events_df is not None and dwell_events_count and has_valid_fps else np.nan
        summary_rows.append(
            {
                "Object ROI": object_name,
                "Entries": int(entries.get(object_name, 0)),
                "Exits": int(exits.get(object_name, 0)),
                "Frames Interacting": total_frames,
                "Time Interacting (s)": (total_frames / fps) if has_valid_fps else np.nan,
                "Frames Contact": int(contact_frame_counts.get(object_name, 0)),
                "Time Contact (s)": (float(contact_frame_counts.get(object_name, 0)) / fps) if has_valid_fps else np.nan,
                "Frames Proximity Only": int(proximity_frame_counts.get(object_name, 0)),
                "Time Proximity Only (s)": (float(proximity_frame_counts.get(object_name, 0)) / fps) if has_valid_fps else np.nan,
                "Dwell Events": dwell_events_count,
                "Mean Dwell Duration (frames)": mean_dwell_frames,
                "Mean Dwell Duration (s)": mean_dwell_s,
                "Median Dwell Duration (frames)": median_dwell_frames,
                "Median Dwell Duration (s)": median_dwell_s,
            }
        )
    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)

    per_track_rows: List[Dict[str, Any]] = []
    for track_id in sorted(per_track_frames.keys()):
        for object_name in sorted(object_names):
            frames = int(per_track_frames[track_id].get(object_name, 0))
            if frames <= 0 and per_track_entries[track_id].get(object_name, 0) <= 0 and per_track_exits[track_id].get(object_name, 0) <= 0:
                continue
            events_df = (
                dwell_by_track.get_group((track_id, object_name))
                if dwell_by_track is not None and (track_id, object_name) in dwell_by_track.groups
                else None
            )
            dwell_events_count = int(len(events_df)) if events_df is not None else 0
            mean_dwell_frames = float(events_df["Duration (Frames)"].mean()) if events_df is not None and dwell_events_count else np.nan
            mean_dwell_s = float(events_df["Duration (s)"].mean()) if events_df is not None and dwell_events_count and has_valid_fps else np.nan
            median_dwell_frames = float(events_df["Duration (Frames)"].median()) if events_df is not None and dwell_events_count else np.nan
            median_dwell_s = float(events_df["Duration (s)"].median()) if events_df is not None and dwell_events_count and has_valid_fps else np.nan
            per_track_rows.append(
                {
                    "Track ID": track_id,
                    "Object ROI": object_name,
                    "Entries": int(per_track_entries[track_id].get(object_name, 0)),
                    "Exits": int(per_track_exits[track_id].get(object_name, 0)),
                    "Frames Interacting": frames,
                    "Time Interacting (s)": (frames / fps) if has_valid_fps else np.nan,
                    "Frames Contact": int(per_track_contact_frames[track_id].get(object_name, 0)),
                    "Time Contact (s)": (float(per_track_contact_frames[track_id].get(object_name, 0)) / fps) if has_valid_fps else np.nan,
                    "Frames Proximity Only": int(per_track_proximity_frames[track_id].get(object_name, 0)),
                    "Time Proximity Only (s)": (float(per_track_proximity_frames[track_id].get(object_name, 0)) / fps) if has_valid_fps else np.nan,
                    "Dwell Events": dwell_events_count,
                    "Mean Dwell Duration (frames)": mean_dwell_frames,
                    "Mean Dwell Duration (s)": mean_dwell_s,
                    "Median Dwell Duration (frames)": median_dwell_frames,
                    "Median Dwell Duration (s)": median_dwell_s,
                }
            )
    per_track_df = pd.DataFrame(per_track_rows, columns=per_track_columns)
    if not per_track_df.empty:
        per_track_df = per_track_df.sort_values(["Track ID", "Object ROI"]).reset_index(drop=True)

    distance_df = pd.DataFrame(distance_records)
    approach_retreat_events_df = empty_ar_events.copy()
    approach_retreat_summary_df = empty_ar_summary.copy()
    approach_retreat_track_df = empty_ar_track.copy()
    if not distance_df.empty:
        distance_df["frame"] = pd.to_numeric(distance_df["frame"], errors="coerce")
        distance_df["distance_px"] = pd.to_numeric(distance_df["distance_px"], errors="coerce")
        distance_df = distance_df.dropna(subset=["track_id", "object_roi", "frame"])
        if not distance_df.empty:
            distance_df["track_id"] = pd.to_numeric(distance_df["track_id"], errors="coerce")
            distance_df = distance_df.dropna(subset=["track_id"])
            distance_df["track_id"] = distance_df["track_id"].astype(int)
            distance_df["frame"] = distance_df["frame"].astype(int)
            distance_df = distance_df.sort_values(["track_id", "object_roi", "frame"]).reset_index(drop=True)

            state_rows: List[Dict[str, Any]] = []
            event_rows: List[Dict[str, Any]] = []
            for (track_id, object_name), group in distance_df.groupby(["track_id", "object_roi"], sort=False):
                prev_frame = None
                prev_dist = np.nan
                current_event: Dict[str, Any] | None = None
                for row in group.itertuples(index=False):
                    frame = int(getattr(row, "frame"))
                    curr_dist = float(getattr(row, "distance_px")) if pd.notna(getattr(row, "distance_px")) else np.nan
                    delta = np.nan
                    state = "neutral"
                    if (
                        prev_frame is not None
                        and frame == prev_frame + 1
                        and np.isfinite(prev_dist)
                        and np.isfinite(curr_dist)
                    ):
                        delta = float(curr_dist - prev_dist)
                        if delta < 0:
                            state = "approach"
                        elif delta > 0:
                            state = "retreat"
                        else:
                            state = "neutral"
                    state_rows.append(
                        {
                            "Track ID": int(track_id),
                            "Object ROI": str(object_name),
                            "Frame": frame,
                            "Distance (px)": curr_dist,
                            "Delta Distance (px/frame)": delta,
                            "State": state,
                        }
                    )

                    if state in {"approach", "retreat"} and np.isfinite(delta):
                        if (
                            current_event is not None
                            and current_event["State"] == state
                            and frame == current_event["End Frame"] + 1
                        ):
                            current_event["End Frame"] = frame
                            current_event["_deltas"].append(float(delta))
                        else:
                            if current_event is not None:
                                deltas = current_event.pop("_deltas")
                                duration_frames = int(current_event["End Frame"] - current_event["Start Frame"] + 1)
                                current_event["Duration (Frames)"] = duration_frames
                                current_event["Duration (s)"] = (duration_frames / fps) if has_valid_fps else np.nan
                                current_event["Mean Delta Distance (px/frame)"] = float(np.mean(deltas)) if deltas else np.nan
                                current_event["Total Distance Change (px)"] = float(np.sum(deltas)) if deltas else np.nan
                                event_rows.append(current_event)
                            current_event = {
                                "Track ID": int(track_id),
                                "Object ROI": str(object_name),
                                "State": state,
                                "Start Frame": frame,
                                "End Frame": frame,
                                "_deltas": [float(delta)],
                            }
                    else:
                        if current_event is not None:
                            deltas = current_event.pop("_deltas")
                            duration_frames = int(current_event["End Frame"] - current_event["Start Frame"] + 1)
                            current_event["Duration (Frames)"] = duration_frames
                            current_event["Duration (s)"] = (duration_frames / fps) if has_valid_fps else np.nan
                            current_event["Mean Delta Distance (px/frame)"] = float(np.mean(deltas)) if deltas else np.nan
                            current_event["Total Distance Change (px)"] = float(np.sum(deltas)) if deltas else np.nan
                            event_rows.append(current_event)
                            current_event = None

                    prev_frame = frame
                    prev_dist = curr_dist
                if current_event is not None:
                    deltas = current_event.pop("_deltas")
                    duration_frames = int(current_event["End Frame"] - current_event["Start Frame"] + 1)
                    current_event["Duration (Frames)"] = duration_frames
                    current_event["Duration (s)"] = (duration_frames / fps) if has_valid_fps else np.nan
                    current_event["Mean Delta Distance (px/frame)"] = float(np.mean(deltas)) if deltas else np.nan
                    current_event["Total Distance Change (px)"] = float(np.sum(deltas)) if deltas else np.nan
                    event_rows.append(current_event)

            state_df = pd.DataFrame(state_rows)
            approach_retreat_events_df = pd.DataFrame(event_rows, columns=approach_retreat_event_columns)
            if not approach_retreat_events_df.empty:
                approach_retreat_events_df = approach_retreat_events_df.sort_values(
                    ["Object ROI", "Track ID", "Start Frame"]
                ).reset_index(drop=True)

            if not state_df.empty:
                state_df["IsApproach"] = state_df["State"] == "approach"
                state_df["IsRetreat"] = state_df["State"] == "retreat"
                state_df["ApproachRate"] = np.where(
                    state_df["IsApproach"],
                    state_df["Delta Distance (px/frame)"].abs(),
                    np.nan,
                )
                state_df["RetreatRate"] = np.where(
                    state_df["IsRetreat"],
                    state_df["Delta Distance (px/frame)"],
                    np.nan,
                )
                state_df["DeltaValid"] = np.where(
                    np.isfinite(state_df["Delta Distance (px/frame)"]),
                    state_df["Delta Distance (px/frame)"],
                    0.0,
                )

                event_counts_track = defaultdict(lambda: {"Approach Events": 0, "Retreat Events": 0})
                if not approach_retreat_events_df.empty:
                    for _, erow in approach_retreat_events_df.iterrows():
                        key = (int(erow["Track ID"]), str(erow["Object ROI"]))
                        if str(erow["State"]) == "approach":
                            event_counts_track[key]["Approach Events"] += 1
                        elif str(erow["State"]) == "retreat":
                            event_counts_track[key]["Retreat Events"] += 1

                track_rows: List[Dict[str, Any]] = []
                for (track_id, object_name), g in state_df.groupby(["Track ID", "Object ROI"]):
                    approach_frames = int(g["IsApproach"].sum())
                    retreat_frames = int(g["IsRetreat"].sum())
                    key = (int(track_id), str(object_name))
                    events_meta = event_counts_track.get(key, {"Approach Events": 0, "Retreat Events": 0})
                    track_rows.append(
                        {
                            "Track ID": int(track_id),
                            "Object ROI": str(object_name),
                            "Approach Events": int(events_meta.get("Approach Events", 0)),
                            "Retreat Events": int(events_meta.get("Retreat Events", 0)),
                            "Approach Frames": approach_frames,
                            "Retreat Frames": retreat_frames,
                            "Approach Time (s)": (approach_frames / fps) if has_valid_fps else np.nan,
                            "Retreat Time (s)": (retreat_frames / fps) if has_valid_fps else np.nan,
                            "Mean Approach Rate (px/frame)": float(g["ApproachRate"].dropna().mean())
                            if g["ApproachRate"].dropna().size
                            else np.nan,
                            "Mean Retreat Rate (px/frame)": float(g["RetreatRate"].dropna().mean())
                            if g["RetreatRate"].dropna().size
                            else np.nan,
                            "Net Distance Change (px)": float(g["DeltaValid"].sum()),
                        }
                    )
                approach_retreat_track_df = pd.DataFrame(track_rows, columns=approach_retreat_track_columns)
                if not approach_retreat_track_df.empty:
                    approach_retreat_track_df = approach_retreat_track_df.sort_values(
                        ["Track ID", "Object ROI"]
                    ).reset_index(drop=True)

                summary_rows_ar: List[Dict[str, Any]] = []
                for object_name, g in state_df.groupby("Object ROI"):
                    approach_frames = int(g["IsApproach"].sum())
                    retreat_frames = int(g["IsRetreat"].sum())
                    object_track = approach_retreat_track_df[approach_retreat_track_df["Object ROI"] == object_name]
                    summary_rows_ar.append(
                        {
                            "Object ROI": str(object_name),
                            "Approach Events": int(object_track["Approach Events"].sum()) if not object_track.empty else 0,
                            "Retreat Events": int(object_track["Retreat Events"].sum()) if not object_track.empty else 0,
                            "Approach Frames": approach_frames,
                            "Retreat Frames": retreat_frames,
                            "Approach Time (s)": (approach_frames / fps) if has_valid_fps else np.nan,
                            "Retreat Time (s)": (retreat_frames / fps) if has_valid_fps else np.nan,
                            "Mean Approach Rate (px/frame)": float(g["ApproachRate"].dropna().mean())
                            if g["ApproachRate"].dropna().size
                            else np.nan,
                            "Mean Retreat Rate (px/frame)": float(g["RetreatRate"].dropna().mean())
                            if g["RetreatRate"].dropna().size
                            else np.nan,
                            "Net Distance Change (px)": float(g["DeltaValid"].sum()),
                        }
                    )
                approach_retreat_summary_df = pd.DataFrame(summary_rows_ar, columns=approach_retreat_summary_columns)
                if not approach_retreat_summary_df.empty:
                    approach_retreat_summary_df = approach_retreat_summary_df.sort_values("Object ROI").reset_index(drop=True)

    return {
        "per_frame_df": df_result,
        "summary": summary_df,
        "per_track": per_track_df,
        "dwell_events": dwell_df,
        "approach_retreat_summary": approach_retreat_summary_df,
        "approach_retreat_by_track": approach_retreat_track_df,
        "approach_retreat_events": approach_retreat_events_df,
        "events": {"entries": entry_events, "exits": exit_events},
    }

def analyze_bouts(
    per_frame_df,
    class_names,
    max_gap_frames,
    min_bout_frames,
    fps,
    roi_column=None,
    stop_check: Optional[Callable[[], bool]] = None,
):
    """Analyzes the preprocessed data to identify and summarize behavioral bouts.

    ``fps`` is required and must be a positive number. The previous default of
    ``30`` was removed to prevent silent miscalibration on non-30Hz recordings;
    resolve the value via ``integra_pose.utils.fps_resolver.resolve_fps`` at
    the entry point of each workflow.
    """
    if fps is None or not (float(fps) > 0):
        raise ValueError(
            "analyze_bouts: fps must be a positive number; "
            "resolve it from the workflow's Video FPS field before calling."
        )
    fps = float(fps)
    if per_frame_df is None or per_frame_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    required_cols = {"track_id", "frame", "class_id"}
    if not required_cols.issubset(per_frame_df.columns):
        return pd.DataFrame(), pd.DataFrame()

    use_roi = bool(roi_column) and roi_column in per_frame_df.columns

    has_valid_fps = True
    fps_value = float(fps) if has_valid_fps else 0.0

    columns = ["track_id", "frame", "class_id"]
    if use_roi:
        columns.append(roi_column)
    df = per_frame_df[columns].copy()

    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["class_id"] = pd.to_numeric(df["class_id"], errors="coerce")
    df = df.dropna(subset=["track_id", "frame", "class_id"])
    df["track_id"] = df["track_id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df["class_id"] = df["class_id"].astype(int)

    def _normalize_roi(value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        text = str(value).strip()
        if not text or text.lower() == "none":
            return ""
        return text

    if use_roi:
        df["_roi_label"] = df[roi_column].map(_normalize_roi)
        # Ensure one ROI label per (track, class, frame) so ROI metadata is deterministic.
        df = (
            df.groupby(["track_id", "class_id", "frame"], as_index=False)["_roi_label"]
            .agg(lambda values: next((v for v in values if v), ""))
        )
    else:
        df = df.drop_duplicates(subset=["track_id", "class_id", "frame"])

    detailed_bouts = []
    grouped = df.groupby("track_id")

    for track_idx, (track_id, track_df) in enumerate(tqdm(grouped, desc="Analyzing Bouts per Animal"), start=1):
        _check_stop_interval(stop_check, track_idx, stage="Bout analysis", interval=8)
        track_df = track_df.sort_values("frame")
        for class_idx, (class_id, class_group) in enumerate(track_df.groupby("class_id"), start=1):
            _check_stop_interval(stop_check, class_idx, stage="Bout analysis", interval=16)
            if class_group.empty:
                continue

            class_name = class_names.get(class_id) or class_names.get(str(class_id)) or f"Class_{class_id}"
            frames = class_group["frame"].to_numpy(dtype=int)
            roi_labels = (
                class_group["_roi_label"].to_numpy(dtype=object) if use_roi else None
            )

            gaps = np.diff(frames) > max_gap_frames + 1
            breaks = gaps

            bout_ends = np.where(breaks)[0]
            bout_starts_indices = np.insert(bout_ends + 1, 0, 0)
            bout_ends_indices = np.append(bout_ends, len(frames) - 1)

            for start_idx, end_idx in zip(bout_starts_indices, bout_ends_indices):
                bout_start_frame = int(frames[start_idx])
                bout_end_frame = int(frames[end_idx])
                duration_frames = int(bout_end_frame - bout_start_frame + 1)

                if duration_frames < min_bout_frames:
                    continue

                if has_valid_fps:
                    start_time = bout_start_frame / fps_value
                    end_time = bout_end_frame / fps_value
                    duration_time = duration_frames / fps_value
                else:
                    start_time = np.nan
                    end_time = np.nan
                    duration_time = np.nan

                entry = {
                    "Track ID": int(track_id),
                    "Behavior": class_name,
                    "Start Frame": bout_start_frame,
                    "End Frame": bout_end_frame,
                    "Duration (Frames)": duration_frames,
                    "Start Time (s)": start_time,
                    "End Time (s)": end_time,
                    "Duration (s)": duration_time,
                }
                if use_roi and roi_labels is not None and roi_labels.size:
                    roi_slice = roi_labels[start_idx:end_idx + 1]
                    non_empty = [label for label in roi_slice if label]
                    memberships = tuple(dict.fromkeys(non_empty))
                    primary_roi = ""
                    if non_empty:
                        counts = Counter(non_empty)
                        max_count = max(counts.values())
                        candidates = {roi for roi, count in counts.items() if count == max_count}
                        if len(candidates) == 1:
                            primary_roi = next(iter(candidates))
                        else:
                            for roi in non_empty:
                                if roi in candidates:
                                    primary_roi = roi
                                    break
                    entry["ROI Name"] = str(primary_roi)
                    entry["ROI Memberships"] = memberships
                elif use_roi:
                    entry["ROI Name"] = ""
                    entry["ROI Memberships"] = ()
                detailed_bouts.append(entry)

    if not detailed_bouts:
        return pd.DataFrame(), pd.DataFrame()

    detailed_bouts_df = pd.DataFrame(detailed_bouts)

    summary_group_cols = ['Track ID', 'Behavior']
    if use_roi:
        summary_group_cols.append('ROI Name')

    summary_df = detailed_bouts_df.groupby(summary_group_cols).agg(
        Bout_Count=('Behavior', 'size'),
        Total_Duration_s=('Duration (s)', 'sum'),
        Mean_Duration_s=('Duration (s)', 'mean')
    ).reset_index()

    return detailed_bouts_df, summary_df

def compute_roi_metrics(
    per_frame_df,
    roi_column,
    fps,
    class_names=None,
    *,
    max_gap_frames=0,
    min_dwell_frames=1,
    stop_check: Optional[Callable[[], bool]] = None,
):
    roi_column = roi_column or ''
    if per_frame_df.empty or 'track_id' not in per_frame_df.columns or 'frame' not in per_frame_df.columns:
        return None

    membership_column = 'ROI Memberships' if 'ROI Memberships' in per_frame_df.columns else roi_column
    if membership_column not in per_frame_df.columns:
        return None

    has_valid_fps = bool(fps and fps > 0)
    gap_threshold = max(0, int(max_gap_frames or 0))
    min_dwell_frames = max(1, int(min_dwell_frames or 1))

    entries = Counter()
    exits = Counter()
    transitions = Counter()
    frame_counts = Counter()
    per_track_transitions = defaultdict(Counter)

    def _new_roi_stats():
        return {
            'Entries': 0,
            'Exits': 0,
            'Total Frames': 0,
            'Dwell Events': 0,
            'Total Dwell Frames': 0,
            'Total Dwell Time (s)': 0.0,
        }

    per_track_stats = defaultdict(lambda: defaultdict(_new_roi_stats))
    roi_dwell_totals = defaultdict(lambda: {'events': 0, 'frames': 0, 'time': 0.0})
    roi_behavior_frame_counts = Counter()
    per_track_behavior_frame_counts = Counter()
    dwell_records = []
    all_roi_names = set()
    has_class_column = 'class_id' in per_frame_df.columns
    presence_frames_by_track = defaultdict(lambda: defaultdict(list))

    def _normalise_primary(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        if not text or text.lower() == 'none':
            return None
        return text

    def _normalise_memberships(value) -> Tuple[str, ...]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ()
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() == 'none':
                return ()
            return (value,)
        memberships: List[str] = []
        if isinstance(value, (list, tuple, set)):
            for item in value:
                name = _normalise_primary(item)
                if name:
                    memberships.append(name)
        else:
            name = _normalise_primary(value)
            if name:
                memberships.append(name)
        return tuple(sorted(set(memberships)))

    def _resolve_behavior(class_id):
        if class_id is None or (isinstance(class_id, float) and np.isnan(class_id)):
            return None
        try:
            class_id_int = int(class_id)
        except (TypeError, ValueError):
            class_id_int = class_id

        if isinstance(class_names, dict):
            if class_id_int in class_names:
                return class_names[class_id_int]
            if isinstance(class_id_int, int):
                str_key = str(class_id_int)
                if str_key in class_names:
                    return class_names[str_key]
            if class_id in class_names:
                return class_names[class_id]
        elif isinstance(class_names, (list, tuple)):
            try:
                idx = int(class_id_int)
                if 0 <= idx < len(class_names):
                    return class_names[idx]
            except (TypeError, ValueError):
                pass
        return f"Class_{class_id_int}"

    def _record_dwell(track_id, roi_name, start_frame, end_frame):
        if roi_name is None or start_frame is None or end_frame is None:
            return
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        frames = max(end_frame - start_frame + 1, 1)
        if frames < min_dwell_frames:
            return
        duration_s = frames / fps if has_valid_fps else np.nan

        dwell_records.append({
            'Track ID': track_id,
            'ROI Name': roi_name,
            'Start Frame': start_frame,
            'End Frame': end_frame,
            'Duration (Frames)': frames,
            'Duration (s)': duration_s,
        })

        roi_stats = per_track_stats[track_id][roi_name]
        roi_stats['Dwell Events'] += 1
        roi_stats['Total Dwell Frames'] += frames
        if has_valid_fps:
            roi_stats['Total Dwell Time (s)'] += duration_s
        else:
            roi_stats['Total Dwell Time (s)'] = np.nan

        dwell_totals = roi_dwell_totals[roi_name]
        dwell_totals['events'] += 1
        dwell_totals['frames'] += frames
        if has_valid_fps:
            dwell_totals['time'] += duration_s
        else:
            dwell_totals['time'] = np.nan

    per_frame_sorted = per_frame_df.sort_values(['track_id', 'frame'])
    # Guard against duplicated detections for the same (track, frame) which would otherwise
    # inflate dwell/occupancy counts and duplicate entry/exit events.
    try:
        duplicate_mask = per_frame_sorted.duplicated(subset=['track_id', 'frame'], keep='first')
    except Exception:
        duplicate_mask = None
    if duplicate_mask is not None and bool(duplicate_mask.any()):
        if 'confidence' in per_frame_sorted.columns:
            conf_series = pd.to_numeric(per_frame_sorted['confidence'], errors='coerce').fillna(-np.inf)
            per_frame_sorted = per_frame_sorted.assign(_confidence_rank=conf_series)
            per_frame_sorted = (
                per_frame_sorted.sort_values(['track_id', 'frame', '_confidence_rank'])
                .drop_duplicates(subset=['track_id', 'frame'], keep='last')
                .drop(columns=['_confidence_rank'])
            )
        else:
            per_frame_sorted = per_frame_sorted.drop_duplicates(subset=['track_id', 'frame'], keep='last')

    for track_idx, (track_id, track_df) in enumerate(per_frame_sorted.groupby('track_id'), start=1):
        _check_stop_interval(stop_check, track_idx, stage="ROI metric aggregation", interval=8)
        track_df = track_df.sort_values('frame')
        active_rois: Dict[str, Dict[str, int]] = {}
        prev_primary = None
        last_frame = None

        for row_idx, (_, row) in enumerate(track_df.iterrows(), start=1):
            _check_stop_interval(stop_check, row_idx, stage="ROI metric aggregation")
            frame = int(row['frame'])
            if last_frame is not None and frame - last_frame > gap_threshold + 1:
                for roi_name, info in active_rois.items():
                    exits[roi_name] += 1
                    roi_stats = per_track_stats[track_id][roi_name]
                    roi_stats['Exits'] += 1
                    _record_dwell(track_id, roi_name, info['start_frame'], info['last_frame'])
                active_rois = {}
                prev_primary = None
            memberships = _normalise_memberships(row.get(membership_column))
            primary_roi = _normalise_primary(row.get(roi_column))
            class_id = row['class_id'] if has_class_column else None
            behavior_name = _resolve_behavior(class_id) if has_class_column else None

            membership_set = set(memberships)

            # Update frame counts and per-behavior accumulators.
            for roi_name in membership_set:
                all_roi_names.add(roi_name)
                frame_counts[roi_name] += 1
                presence_frames_by_track[track_id][roi_name].append(frame)
                roi_stats = per_track_stats[track_id][roi_name]
                roi_stats['Total Frames'] += 1
                if behavior_name:
                    roi_behavior_frame_counts[(roi_name, behavior_name)] += 1
                    per_track_behavior_frame_counts[(track_id, roi_name, behavior_name)] += 1

            # Handle entries and exits for each ROI independently.
            next_active: Dict[str, Dict[str, int]] = {}
            for roi_name in membership_set:
                prev_info = active_rois.get(roi_name)
                if prev_info is None:
                    entries[roi_name] += 1
                    roi_stats = per_track_stats[track_id][roi_name]
                    roi_stats['Entries'] += 1
                    next_active[roi_name] = {'start_frame': frame, 'last_frame': frame}
                else:
                    next_active[roi_name] = {'start_frame': prev_info['start_frame'], 'last_frame': frame}

            for roi_name, info in active_rois.items():
                if roi_name not in membership_set:
                    exits[roi_name] += 1
                    roi_stats = per_track_stats[track_id][roi_name]
                    roi_stats['Exits'] += 1
                    _record_dwell(track_id, roi_name, info['start_frame'], info['last_frame'])

            active_rois = next_active

            # Track primary ROI transitions for reporting.
            if primary_roi and primary_roi not in membership_set:
                membership_set.add(primary_roi)
                all_roi_names.add(primary_roi)

            if prev_primary and primary_roi and prev_primary != primary_roi:
                transitions[(prev_primary, primary_roi)] += 1
                per_track_transitions[track_id][(prev_primary, primary_roi)] += 1

            prev_primary = primary_roi
            last_frame = frame

        # Close any lingering ROI memberships after the last frame.
        if last_frame is None:
            continue
        for roi_name, info in active_rois.items():
            exits[roi_name] += 1
            roi_stats = per_track_stats[track_id][roi_name]
            roi_stats['Exits'] += 1
            _record_dwell(track_id, roi_name, info['start_frame'], info['last_frame'])

    entries = Counter()
    exits = Counter()
    dwell_records = []
    roi_dwell_totals = defaultdict(lambda: {'events': 0, 'frames': 0, 'time': 0.0})
    for track_stats in per_track_stats.values():
        for stats in track_stats.values():
            stats['Entries'] = 0
            stats['Exits'] = 0
            stats['Dwell Events'] = 0
            stats['Total Dwell Frames'] = 0
            stats['Total Dwell Time (s)'] = 0.0 if has_valid_fps else np.nan

    for track_id in sorted(presence_frames_by_track.keys()):
        for roi_name in sorted(presence_frames_by_track[track_id].keys()):
            segments = _build_presence_segments(
                presence_frames_by_track[track_id][roi_name],
                gap_threshold=gap_threshold,
                min_duration=min_dwell_frames,
            )
            for start_frame, end_frame in segments:
                entries[roi_name] += 1
                exits[roi_name] += 1
                roi_stats = per_track_stats[track_id][roi_name]
                roi_stats['Entries'] += 1
                roi_stats['Exits'] += 1
                _record_dwell(track_id, roi_name, start_frame, end_frame)

    if not all_roi_names:
        empty_overview_cols = [
            'ROI Name', 'Entries', 'Exits', 'Frames in ROI', 'Time in ROI (s)',
            'Dwell Events', 'Mean Dwell Duration (frames)', 'Mean Dwell Duration (s)',
            'Median Dwell Duration (frames)', 'Median Dwell Duration (s)', 'Total Dwell Frames',
            'Total Dwell Time (s)'
        ]
        return {
            'entries_exits': pd.DataFrame(columns=empty_overview_cols),
            'transitions': pd.DataFrame(columns=['From ROI', 'To ROI', 'Transition Count']),
            'entries_exits_by_track': pd.DataFrame(columns=[
                'Track ID', 'ROI Name', 'Entries', 'Exits', 'Frames in ROI', 'Time in ROI (s)',
                'Dwell Events', 'Mean Dwell Duration (frames)', 'Mean Dwell Duration (s)',
                'Median Dwell Duration (frames)', 'Median Dwell Duration (s)',
                'Total Dwell Frames', 'Total Dwell Time (s)'
            ]),
            'dwell_events': pd.DataFrame(columns=['Track ID', 'ROI Name', 'Start Frame', 'End Frame', 'Duration (Frames)', 'Duration (s)']),
            'transitions_by_track': pd.DataFrame(columns=['Track ID', 'From ROI', 'To ROI', 'Transition Count']),
            'roi_behavior_summary': pd.DataFrame(columns=['ROI Name', 'Behavior', 'Frames in ROI', 'Time in ROI (s)', 'Occupancy Within ROI (%)']),
            'roi_behavior_by_track': pd.DataFrame(columns=['Track ID', 'ROI Name', 'Behavior', 'Frames in ROI', 'Time in ROI (s)'])
        }

    dwell_events_df = pd.DataFrame(dwell_records)
    if not dwell_events_df.empty:
        dwell_events_df = dwell_events_df.sort_values(['ROI Name', 'Track ID', 'Start Frame']).reset_index(drop=True)
        roi_median_s = dwell_events_df.groupby('ROI Name')['Duration (s)'].median()
        roi_median_frames = dwell_events_df.groupby('ROI Name')['Duration (Frames)'].median()
        per_track_median_s = dwell_events_df.groupby(['Track ID', 'ROI Name'])['Duration (s)'].median()
        per_track_median_frames = dwell_events_df.groupby(['Track ID', 'ROI Name'])['Duration (Frames)'].median()
    else:
        roi_median_s = pd.Series(dtype=float)
        roi_median_frames = pd.Series(dtype=float)
        per_track_median_s = pd.Series(dtype=float)
        per_track_median_frames = pd.Series(dtype=float)

    overview_rows = []
    for roi_name in sorted(all_roi_names):
        total_frames = frame_counts.get(roi_name, 0)
        dwell_totals = roi_dwell_totals.get(roi_name, {'events': 0, 'frames': 0, 'time': 0.0})
        dwell_events_count = dwell_totals['events']
        mean_dwell_frames = (dwell_totals['frames'] / dwell_events_count) if dwell_events_count else np.nan
        mean_dwell_s = (dwell_totals['time'] / dwell_events_count) if dwell_events_count and has_valid_fps else np.nan
        overview_rows.append({
            'ROI Name': roi_name,
            'Entries': entries.get(roi_name, 0),
            'Exits': exits.get(roi_name, 0),
            'Frames in ROI': total_frames,
            'Time in ROI (s)': (total_frames / fps) if has_valid_fps else np.nan,
            'Dwell Events': dwell_events_count,
            'Mean Dwell Duration (frames)': mean_dwell_frames,
            'Mean Dwell Duration (s)': mean_dwell_s,
            'Median Dwell Duration (frames)': roi_median_frames.get(roi_name, np.nan) if not dwell_events_df.empty else np.nan,
            'Median Dwell Duration (s)': roi_median_s.get(roi_name, np.nan) if not dwell_events_df.empty else np.nan,
            'Total Dwell Frames': dwell_totals['frames'],
            'Total Dwell Time (s)': dwell_totals['time'] if has_valid_fps else np.nan,
        })
    entries_exits_df = pd.DataFrame(overview_rows).sort_values('ROI Name').reset_index(drop=True)

    per_track_rows = []
    for track_id, roi_dict in per_track_stats.items():
        for roi_name, stats in roi_dict.items():
            if not roi_name:
                continue
            dwell_events_count = stats['Dwell Events']
            mean_dwell_frames = (stats['Total Dwell Frames'] / dwell_events_count) if dwell_events_count else np.nan
            mean_dwell_s = (stats['Total Dwell Time (s)'] / dwell_events_count) if dwell_events_count and has_valid_fps else np.nan
            per_track_rows.append({
                'Track ID': track_id,
                'ROI Name': roi_name,
                'Entries': stats['Entries'],
                'Exits': stats['Exits'],
                'Frames in ROI': stats['Total Frames'],
                'Time in ROI (s)': (stats['Total Frames'] / fps) if has_valid_fps else np.nan,
                'Dwell Events': dwell_events_count,
                'Mean Dwell Duration (frames)': mean_dwell_frames,
                'Mean Dwell Duration (s)': mean_dwell_s,
                'Total Dwell Frames': stats['Total Dwell Frames'],
                'Total Dwell Time (s)': stats['Total Dwell Time (s)'] if has_valid_fps else np.nan,
            })
    per_track_df = pd.DataFrame(per_track_rows)
    if not per_track_df.empty:
        if not per_track_median_s.empty:
            per_track_df = per_track_df.merge(
                per_track_median_s.reset_index().rename(columns={'Duration (s)': 'Median Dwell Duration (s)'}),
                on=['Track ID', 'ROI Name'],
                how='left'
            )
        else:
            per_track_df['Median Dwell Duration (s)'] = np.nan
        if not per_track_median_frames.empty:
            per_track_df = per_track_df.merge(
                per_track_median_frames.reset_index().rename(columns={'Duration (Frames)': 'Median Dwell Duration (frames)'}),
                on=['Track ID', 'ROI Name'],
                how='left'
            )
        else:
            per_track_df['Median Dwell Duration (frames)'] = np.nan
        per_track_df = per_track_df.sort_values(['Track ID', 'ROI Name']).reset_index(drop=True)
    else:
        per_track_df = pd.DataFrame(columns=[
            'Track ID', 'ROI Name', 'Entries', 'Exits', 'Frames in ROI', 'Time in ROI (s)',
            'Dwell Events', 'Mean Dwell Duration (frames)', 'Mean Dwell Duration (s)',
            'Total Dwell Frames', 'Total Dwell Time (s)', 'Median Dwell Duration (s)',
            'Median Dwell Duration (frames)'
        ])

    transition_rows = []
    for (src_roi, dst_roi), count in transitions.items():
        if not src_roi or not dst_roi:
            continue
        transition_rows.append({
            'From ROI': src_roi,
            'To ROI': dst_roi,
            'Transition Count': count
        })
    transitions_df = pd.DataFrame(transition_rows)
    if not transitions_df.empty:
        transitions_df = transitions_df.sort_values('Transition Count', ascending=False).reset_index(drop=True)
    else:
        transitions_df = pd.DataFrame(columns=['From ROI', 'To ROI', 'Transition Count'])

    per_track_transition_rows = []
    for track_id, transition_counter in per_track_transitions.items():
        for (src_roi, dst_roi), count in transition_counter.items():
            if not src_roi or not dst_roi:
                continue
            per_track_transition_rows.append({
                'Track ID': track_id,
                'From ROI': src_roi,
                'To ROI': dst_roi,
                'Transition Count': count
            })
    per_track_transitions_df = pd.DataFrame(per_track_transition_rows)
    if not per_track_transitions_df.empty:
        per_track_transitions_df = per_track_transitions_df.sort_values(
            ['Track ID', 'Transition Count'], ascending=[True, False]
        ).reset_index(drop=True)
    else:
        per_track_transitions_df = pd.DataFrame(columns=['Track ID', 'From ROI', 'To ROI', 'Transition Count'])

    roi_behavior_rows = []
    for (roi_name, behavior_name), frames in roi_behavior_frame_counts.items():
        if not roi_name or not behavior_name:
            continue
        total_roi_frames = frame_counts.get(roi_name, 0)
        occupancy_ratio = (frames / total_roi_frames) if total_roi_frames else np.nan
        roi_behavior_rows.append({
            'ROI Name': roi_name,
            'Behavior': behavior_name,
            'Frames in ROI': frames,
            'Time in ROI (s)': (frames / fps) if has_valid_fps else np.nan,
            'Occupancy Within ROI (%)': (occupancy_ratio * 100.0) if occupancy_ratio is not None else np.nan,
        })
    roi_behavior_df = pd.DataFrame(roi_behavior_rows)
    if not roi_behavior_df.empty:
        roi_behavior_df = roi_behavior_df.sort_values(['ROI Name', 'Behavior']).reset_index(drop=True)
    else:
        roi_behavior_df = pd.DataFrame(columns=['ROI Name', 'Behavior', 'Frames in ROI', 'Time in ROI (s)', 'Occupancy Within ROI (%)'])

    per_track_behavior_rows = []
    for (track_id, roi_name, behavior_name), frames in per_track_behavior_frame_counts.items():
        if not roi_name or not behavior_name:
            continue
        per_track_behavior_rows.append({
            'Track ID': track_id,
            'ROI Name': roi_name,
            'Behavior': behavior_name,
            'Frames in ROI': frames,
            'Time in ROI (s)': (frames / fps) if has_valid_fps else np.nan,
        })
    per_track_behavior_df = pd.DataFrame(per_track_behavior_rows)
    if not per_track_behavior_df.empty:
        per_track_behavior_df = per_track_behavior_df.sort_values(['Track ID', 'ROI Name', 'Behavior']).reset_index(drop=True)
    else:
        per_track_behavior_df = pd.DataFrame(columns=['Track ID', 'ROI Name', 'Behavior', 'Frames in ROI', 'Time in ROI (s)'])

    return {
        'entries_exits': entries_exits_df,
        'transitions': transitions_df,
        'entries_exits_by_track': per_track_df,
        'dwell_events': dwell_events_df,
        'transitions_by_track': per_track_transitions_df,
        'roi_behavior_summary': roi_behavior_df,
        'roi_behavior_by_track': per_track_behavior_df,
    }

def save_bout_summary_to_excel(detailed_bouts_df, output_folder, class_names, fps, video_name):
    """
    Saves a summary of bout analysis per track and behavior to an Excel file.

    Args:
        detailed_bouts_df (pd.DataFrame): DataFrame with detailed bout information.
        output_folder (str): Folder to save the Excel file.
        class_names (dict): Mapping of class IDs to names.
        fps (float): Video frame rate.
        video_name (str): Base name for the Excel file.

    Returns:
        str: Path to the generated Excel file, or None if an error occurs.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Generating bout summary Excel for video: {video_name}")

    if detailed_bouts_df.empty:
        logger.warning("No bouts to summarize, skipping Excel generation")
        return None

    excel_path = os.path.join(output_folder, f"{video_name}_bout_summary.xlsx")
    summary_data = []

    group_columns = ['Track ID', 'Behavior']
    include_roi = 'ROI Name' in detailed_bouts_df.columns
    if include_roi:
        group_columns.append('ROI Name')

    grouped = detailed_bouts_df.groupby(group_columns)

    for group_key, group in grouped:
        if include_roi:
            if isinstance(group_key, tuple):
                track_id, behavior, roi_name = group_key
            else:
                track_id, behavior, roi_name = group_key, None, None
        else:
            if isinstance(group_key, tuple):
                track_id, behavior = group_key
            else:
                track_id, behavior = group_key, None
            roi_name = None

        num_bouts = len(group)
        avg_duration_frames = group['Duration (Frames)'].mean()
        avg_duration_seconds = group['Duration (s)'].mean()
        all_durations_frames = group['Duration (Frames)'].tolist()
        all_durations_str = ", ".join(map(str, all_durations_frames))

        if num_bouts > 1:
            start_times = group['Start Time (s)'].sort_values().values
            gaps = np.diff(start_times)
            avg_time_between_bouts = gaps.mean()
        else:
            avg_time_between_bouts = np.nan

        entry = {
            'Class Label': behavior,
            'Track ID': track_id,
            'Number of Bouts': num_bouts,
            'Avg Bout Duration (frames)': avg_duration_frames,
            'Avg Bout Duration (seconds)': avg_duration_seconds,
            'All Bout Durations (frames) for this Track/Class': all_durations_str,
            'Avg Time Between Bouts (seconds) for this Track/Class': avg_time_between_bouts
        }
        if include_roi:
            entry['ROI Name'] = roi_name
        summary_data.append(entry)

    if not summary_data:
        logger.warning("No summary data generated, skipping Excel generation")
        return None

    summary_df = pd.DataFrame(summary_data)
    summary_columns = [
        'Class Label', 'Track ID', 'Number of Bouts',
        'Avg Bout Duration (frames)', 'Avg Bout Duration (seconds)',
        'All Bout Durations (frames) for this Track/Class',
        'Avg Time Between Bouts (seconds) for this Track/Class'
    ]
    if include_roi:
        summary_columns.insert(2, 'ROI Name')
    summary_df = summary_df[summary_columns]

    try:
        summary_df.to_excel(excel_path, index=False)
        logger.info(f"Bout summary saved to {excel_path}")
        return excel_path
    except Exception as e:
        fallback_path = os.path.join(output_folder, f"{video_name}_bout_summary.csv")
        logger.warning(
            "Failed to save bout summary Excel (%s). Falling back to CSV at %s.",
            e,
            fallback_path,
        )
        try:
            summary_df.to_csv(fallback_path, index=False)
            return fallback_path
        except Exception as csv_exc:
            logger.error(f"Failed to save bout summary fallback CSV: {csv_exc}")
            return None

def generate_analytics_dashboard(output_folder, video_name, detailed_bouts_df, summary_df, roi_metrics):
    """Create a multi-panel analytics dashboard visualizing key bout and ROI metrics."""
    logger = logging.getLogger(__name__)
    try:
        import matplotlib
        try:
            matplotlib.get_backend()
        except Exception:
            pass
        if matplotlib.get_backend().lower() != 'agg':
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib is not installed; skipping analytics dashboard generation")
        return None

    roi_metrics = roi_metrics or {}

    def _render_empty(ax, title):
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    def _wrap_label(value: Any, width: int = 14) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        wrapped = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
        return "\n".join(wrapped) if wrapped else text

    def _format_x_labels(ax, *, width: int = 14):
        ax.tick_params(axis='x', rotation=45)
        labels = [_wrap_label(label.get_text(), width=width) for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()
    fig.suptitle(f"Bout Analytics Summary - {video_name}", fontsize=16, fontweight='bold')

    ax = axes[0]
    if summary_df is not None and not summary_df.empty and {'Behavior', 'Bout_Count'}.issubset(summary_df.columns):
        behavior_counts = summary_df.groupby('Behavior')['Bout_Count'].sum().sort_values(ascending=False)
        if not behavior_counts.empty:
            ax.bar(behavior_counts.index, behavior_counts.values, color='#4C72B0')
            ax.set_xlabel('Behavior')
            ax.set_ylabel('Bout Count')
            _format_x_labels(ax)
        else:
            _render_empty(ax, 'Total Bout Counts per Behavior')
        ax.set_title('Total Bout Counts per Behavior')
    else:
        _render_empty(ax, 'Total Bout Counts per Behavior')

    ax = axes[1]
    if summary_df is not None and not summary_df.empty and {'Behavior', 'Mean_Duration_s'}.issubset(summary_df.columns):
        avg_duration = summary_df.groupby('Behavior')['Mean_Duration_s'].mean().sort_values(ascending=False)
        if not avg_duration.empty:
            ax.bar(avg_duration.index, avg_duration.values, color='#55A868')
            ax.set_xlabel('Behavior')
            ax.set_ylabel('Seconds')
            _format_x_labels(ax)
        else:
            _render_empty(ax, 'Average Bout Duration (s) by Behavior')
        ax.set_title('Average Bout Duration (s) by Behavior')
    else:
        _render_empty(ax, 'Average Bout Duration (s) by Behavior')

    ax = axes[2]
    roi_overview = roi_metrics.get('entries_exits')
    if roi_overview is not None and not roi_overview.empty and 'Time in ROI (s)' in roi_overview.columns:
        time_series = roi_overview.set_index('ROI Name')['Time in ROI (s)'].fillna(0).sort_values(ascending=False)
        if not time_series.empty:
            ax.bar(time_series.index, time_series.values, color='#C44E52')
            ax.set_xlabel('ROI Name')
            ax.set_ylabel('Seconds')
            _format_x_labels(ax)
        else:
            _render_empty(ax, 'Total Time Spent per ROI (s)')
        ax.set_title('Total Time Spent per ROI (s)')
    else:
        _render_empty(ax, 'Total Time Spent per ROI (s)')

    ax = axes[3]
    transitions_df = roi_metrics.get('transitions')
    if transitions_df is not None and not transitions_df.empty:
        pivot = transitions_df.pivot(index='From ROI', columns='To ROI', values='Transition Count').fillna(0)
        if not pivot.empty:
            im = ax.imshow(pivot.values, cmap='Blues')
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([_wrap_label(col, width=12) for col in pivot.columns], rotation=45, ha='right')
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels([_wrap_label(idx, width=12) for idx in pivot.index])
            ax.set_title('ROI Transition Heatmap')
            ax.set_xlabel('To ROI')
            ax.set_ylabel('From ROI')
            max_val = pivot.values.max() if pivot.values.size else 0
            denom = max(max_val, 1)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    value = pivot.iloc[i, j]
                    text_color = 'white' if value > denom / 2 else 'black'
                    ax.text(j, i, int(value), ha='center', va='center', color=text_color, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            _render_empty(ax, 'ROI Transition Heatmap')
    else:
        _render_empty(ax, 'ROI Transition Heatmap')

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    dashboard_path = os.path.join(output_folder, f"{video_name}_analytics_dashboard.png")
    try:
        fig.savefig(dashboard_path, dpi=200, bbox_inches='tight')
        logger.info(f"Analytics dashboard saved to {dashboard_path}")
        return dashboard_path
    except Exception as exc:
        logger.error(f"Failed to save analytics dashboard: {exc}")
        return None
    finally:
        plt.close(fig)

def save_analysis_outputs(
    per_frame_df,
    yaml_path,
    output_folder,
    max_gap_frames,
    min_bout_frames,
    fps,
    video_name="analysis",
    roi_column=None,
    enabled_modules=None,
    visual_prefs=None,
    run_id=None,
    class_names=None,
    object_metrics=None,
    roi_max_gap_frames=0,
    roi_min_dwell_frames=1,
    stop_check: Optional[Callable[[], bool]] = None,
):
    if fps is None or not (float(fps) > 0):
        raise ValueError(
            "save_analysis_outputs: fps must be a positive number; "
            "resolve it from the workflow's Video FPS field before calling."
        )
    fps = float(fps)
    """
    Wrapper to analyze bouts and save both detailed/summary DataFrames and the new bout summary Excel.

    Args:
        per_frame_df (pd.DataFrame): Preprocessed DataFrame from load_and_preprocess_data.
        yaml_path (str): Optional path to YAML configuration file.
        output_folder (str): Folder to save outputs.
        max_gap_frames (int): Maximum frame gap to consider a bout continuous.
        min_bout_frames (int): Minimum frame duration for a valid bout.
        fps (float): Video frame rate.
        video_name (str): Base name for output files.
        roi_column (str, optional): Name of ROI column if ROI-aware analytics enabled.
        enabled_modules (List[str], optional): Additional analytics modules to execute.
        object_metrics (dict, optional): Object interaction outputs from compute_object_interactions.
        roi_max_gap_frames (int): Maximum frame gap for ROI event summarization.
        roi_min_dwell_frames (int): Minimum frame duration for ROI entry/exit event summarization.

    Returns:
        tuple: (detailed_bouts_df, summary_df, bout_summary_excel_path, roi_metrics, module_outputs)
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Starting analysis output for video: {video_name}")

    os.makedirs(output_folder, exist_ok=True)
    enabled_modules = enabled_modules or []
    visual_prefs = visual_prefs or {}

    try:
        yaml_class_names, _yaml_roi_polygons = _load_yaml_config(yaml_path)
    except Exception as e:
        if class_names:
            logger.warning("Proceeding with explicit class names because YAML could not be loaded: %s", e)
            yaml_class_names = {}
        else:
            logger.error(f"Error loading YAML for class names: {e}")
            raise BoutAnalysisError(f"Error loading YAML: {e}")
    class_names = _normalize_class_names(class_names) or yaml_class_names
    class_names = _infer_class_names_from_dataframe(per_frame_df, class_names)

    _raise_if_stop_requested(stop_check, "Analysis export")
    detailed_bouts_df, summary_df = analyze_bouts(
        per_frame_df,
        class_names,
        max_gap_frames,
        min_bout_frames,
        fps,
        roi_column=roi_column,
        stop_check=stop_check,
    )

    if run_id is None:
        import uuid

        run_id = uuid.uuid4().hex

    if detailed_bouts_df is not None and not detailed_bouts_df.empty:
        detailed_bouts_df = detailed_bouts_df.copy()
        if "Run ID" not in detailed_bouts_df.columns:
            detailed_bouts_df.insert(0, "Run ID", str(run_id))
        else:
            detailed_bouts_df["Run ID"] = str(run_id)

        if "Bout ID" not in detailed_bouts_df.columns:
            import uuid

            def _make_bout_id(row):
                roi_name = row.get("ROI Name", "") if isinstance(row, dict) else ""
                if roi_name is None:
                    roi_name = ""
                name = (
                    f"{run_id}|{row.get('Track ID')}|{row.get('Behavior')}|"
                    f"{row.get('Start Frame')}|{row.get('End Frame')}|{roi_name}"
                )
                return str(uuid.uuid5(uuid.NAMESPACE_URL, name))

            detailed_bouts_df.insert(
                1,
                "Bout ID",
                [
                    _make_bout_id(row)
                    for row in detailed_bouts_df.to_dict(orient="records")
                ],
            )

    if summary_df is not None and not summary_df.empty:
        summary_df = summary_df.copy()
        if "Run ID" not in summary_df.columns:
            summary_df.insert(0, "Run ID", str(run_id))
        else:
            summary_df["Run ID"] = str(run_id)

    if not detailed_bouts_df.empty:
        detailed_csv_path = os.path.join(output_folder, f"{video_name}_detailed_bouts.csv")
        summary_csv_path = os.path.join(output_folder, f"{video_name}_summary.csv")
        try:
            detailed_bouts_df.to_csv(detailed_csv_path, index=False)
            summary_df.to_csv(summary_csv_path, index=False)
            logger.info(f"Saved detailed bouts to {detailed_csv_path}")
            logger.info(f"Saved summary to {summary_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV files: {e}")
    else:
        logger.warning("No bouts detected, skipping CSV generation")

    _raise_if_stop_requested(stop_check, "Analysis export")
    bout_summary_excel_path = save_bout_summary_to_excel(detailed_bouts_df, output_folder, class_names, fps, video_name)

    roi_metrics = (
        compute_roi_metrics(
            per_frame_df,
            roi_column,
            fps,
            class_names,
            max_gap_frames=roi_max_gap_frames,
            min_dwell_frames=roi_min_dwell_frames,
            stop_check=stop_check,
        )
        if roi_column
        else None
    )

    if roi_metrics:
        roi_metric_files = {}
        csv_exports = [
            ('entries_exits', 'roi_overview'),
            ('entries_exits_by_track', 'roi_per_track'),
            ('dwell_events', 'roi_dwell_events'),
            ('transitions', 'roi_transitions'),
            ('transitions_by_track', 'roi_transitions_per_track'),
            ('roi_behavior_summary', 'roi_behavior_summary'),
            ('roi_behavior_by_track', 'roi_behavior_by_track'),
        ]
        for key, suffix in csv_exports:
            df = roi_metrics.get(key)
            if df is not None and not df.empty:
                csv_path = os.path.join(output_folder, f"{video_name}_{suffix}.csv")
                try:
                    df.to_csv(csv_path, index=False)
                    roi_metric_files[key] = csv_path
                    logger.info(f"Saved {key} metrics to {csv_path}")
                except Exception as e:
                    logger.error(f"Failed to save {key} metrics CSV: {e}")
        if roi_metric_files:
            roi_metrics['files'] = roi_metric_files

    _raise_if_stop_requested(stop_check, "Analysis export")
    dashboard_path = generate_analytics_dashboard(output_folder, video_name, detailed_bouts_df, summary_df, roi_metrics)
    if dashboard_path:
        if roi_metrics is None:
            roi_metrics = {}
        roi_metrics.setdefault('files', {})['analytics_dashboard'] = dashboard_path

    _raise_if_stop_requested(stop_check, "Analysis export")
    module_outputs = run_optional_modules(
        enabled_modules=enabled_modules,
        per_frame_df=per_frame_df,
        detailed_bouts_df=detailed_bouts_df,
        summary_df=summary_df,
        roi_metrics=roi_metrics,
        fps=fps,
        class_names=class_names,
        output_folder=output_folder,
        video_name=video_name,
        roi_column=roi_column,
        visual_prefs=visual_prefs,
        object_metrics=object_metrics,
        stop_check=stop_check,
    )

    return detailed_bouts_df, summary_df, bout_summary_excel_path, roi_metrics, module_outputs


# ---------------------------------------------------------------------------
# Optional analytics modules
# ---------------------------------------------------------------------------

class AnalyticsModuleContext:
    """Shared inputs for optional analytics modules."""

    def __init__(
        self,
        per_frame_df: pd.DataFrame,
        detailed_bouts_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        roi_metrics: Optional[Dict[str, Any]],
        object_metrics: Optional[Dict[str, Any]],
        fps: float,
        class_names: Dict[Any, str],
        output_folder: str,
        video_name: str,
        roi_column: Optional[str],
        visual_preferences: Optional[Dict[str, Any]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ):
        self.per_frame_df = per_frame_df if per_frame_df is not None else pd.DataFrame()
        self.detailed_bouts_df = detailed_bouts_df if detailed_bouts_df is not None else pd.DataFrame()
        self.summary_df = summary_df if summary_df is not None else pd.DataFrame()
        self.roi_metrics = roi_metrics or {}
        self.object_metrics = object_metrics or {}
        self.fps = fps or 0.0
        self.class_names = class_names or {}
        self.output_folder = output_folder
        self.video_name = video_name
        self.roi_column = roi_column if roi_column and roi_column in self.per_frame_df.columns else None
        self._module_dirs: Dict[str, str] = {}
        self.behavior_to_class_id = self._invert_class_names()
        self.session_duration_s = self._compute_session_duration()
        self.visual_prefs = visual_preferences or {}
        self.stop_check = stop_check
        self.roi_events = self.roi_metrics.get('events', {}) if isinstance(self.roi_metrics, dict) else {}
        self.object_events = self.object_metrics.get('events', {}) if isinstance(self.object_metrics, dict) else {}
        self.object_summary_df = self.object_metrics.get('summary') if isinstance(self.object_metrics, dict) else pd.DataFrame()
        self.object_per_track_df = self.object_metrics.get('per_track') if isinstance(self.object_metrics, dict) else pd.DataFrame()
        self.object_dwell_df = self.object_metrics.get('dwell_events') if isinstance(self.object_metrics, dict) else pd.DataFrame()
        self.object_per_frame_df = self.object_metrics.get('per_frame_df') if isinstance(self.object_metrics, dict) else pd.DataFrame()

    def _invert_class_names(self) -> Dict[str, Any]:
        inverted = {}
        for key, value in self.class_names.items():
            if value is None:
                continue
            inverted[str(value)] = key
        return inverted

    def _compute_session_duration(self) -> float:
        if not self.per_frame_df.empty and 'frame' in self.per_frame_df.columns and self.fps:
            frames = self.per_frame_df['frame'].to_numpy()
            if frames.size:
                min_frame = np.nanmin(frames)
                max_frame = np.nanmax(frames)
                if np.isfinite(min_frame) and np.isfinite(max_frame) and max_frame >= min_frame:
                    return (float(max_frame) - float(min_frame) + 1.0) / max(self.fps, 1e-6)
        if not self.detailed_bouts_df.empty and 'End Time (s)' in self.detailed_bouts_df.columns:
            max_time = self.detailed_bouts_df['End Time (s)'].max()
            if pd.notna(max_time):
                return float(max_time)
        return float('nan')

    def get_module_dir(self, key: str) -> str:
        if key not in self._module_dirs:
            safe_name = key.replace(' ', '_')
            path = os.path.join(self.output_folder, safe_name)
            os.makedirs(path, exist_ok=True)
            self._module_dirs[key] = path
        return self._module_dirs[key]

    def get_visual_pref(self, key: str, default: str) -> str:
        value = self.visual_prefs.get(key, default) if isinstance(self.visual_prefs, dict) else default
        if value in (None, ""):
            return default
        text = str(value).strip().lower()
        text = text.replace('+', ' ').replace('-', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text if text else default.lower()

    def raise_if_cancelled(self, stage: str) -> None:
        _raise_if_stop_requested(self.stop_check, stage)


def run_optional_modules(
    enabled_modules: List[str],
    per_frame_df: pd.DataFrame,
    detailed_bouts_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    roi_metrics: Optional[Dict[str, Any]],
    object_metrics: Optional[Dict[str, Any]],
    fps: float,
    class_names: Dict[Any, str],
    output_folder: str,
    video_name: str,
    roi_column: Optional[str],
    visual_prefs: Optional[Dict[str, Any]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Dict[str, Any]]:
    logger = logging.getLogger(__name__)
    results: Dict[str, Dict[str, Any]] = {}
    if not enabled_modules:
        return results

    context = AnalyticsModuleContext(
        per_frame_df=per_frame_df,
        detailed_bouts_df=detailed_bouts_df,
        summary_df=summary_df,
        roi_metrics=roi_metrics,
        object_metrics=object_metrics,
        fps=fps,
        class_names=class_names,
        output_folder=output_folder,
        video_name=video_name,
        roi_column=roi_column,
        visual_preferences=visual_prefs,
        stop_check=stop_check,
    )

    for key in enabled_modules:
        context.raise_if_cancelled("Optional analytics modules")
        module = OPTIONAL_ANALYTICS_MODULES.get(key)
        if module is None:
            logger.warning("Unknown analytics module requested: %s", key)
            continue
        compute_fn = module.get('compute')
        if compute_fn is None:
            logger.warning("Analytics module %s is missing a compute function", key)
            continue
        try:
            logger.debug("Running analytics module: %s", key)
            start_time = time.perf_counter()
            output = compute_fn(context)
            duration = time.perf_counter() - start_time
            logger.debug("Analytics module %s completed in %.2f s", key, duration)
            if output:
                if isinstance(output, dict):
                    output.setdefault('duration_s', duration)
                else:
                    output = {'result': output, 'duration_s': duration}
                results[key] = output
        except Exception as exc:
            logger.error("Analytics module %s failed: %s", key, exc, exc_info=True)
    return results


def _module_behavior_transitions(context: AnalyticsModuleContext) -> Dict[str, Any]:
    df = context.detailed_bouts_df
    if df.empty or 'Behavior' not in df.columns:
        return {}

    df_sorted = df.sort_values(['Track ID', 'Start Frame']).reset_index(drop=True)
    transitions = []
    for track_idx, (track_id, group) in enumerate(df_sorted.groupby('Track ID'), start=1):
        _check_stop_interval(context.stop_check, track_idx, stage="Behavior transition analysis", interval=8)
        group = group.sort_values('Start Frame')
        if group.shape[0] < 2:
            continue
        behaviors = group['Behavior'].to_numpy()
        roi_values = group['ROI Name'].to_numpy() if 'ROI Name' in group.columns else np.full(len(group), None, dtype=object)
        start_times = group['Start Time (s)'].to_numpy(dtype=float) if 'Start Time (s)' in group.columns else np.full(len(group), np.nan)
        end_times = group['End Time (s)'].to_numpy(dtype=float) if 'End Time (s)' in group.columns else np.full(len(group), np.nan)
        for idx in range(1, len(group)):
            curr_behavior = behaviors[idx]
            prev_behavior = behaviors[idx - 1]
            prev_roi = roi_values[idx - 1] if roi_values.size else None
            curr_roi = roi_values[idx] if roi_values.size else None
            gap = np.nan
            if np.isfinite(start_times[idx]) and np.isfinite(end_times[idx - 1]):
                gap = max(0.0, float(start_times[idx]) - float(end_times[idx - 1]))
            transitions.append({
                'Track ID': track_id,
                'From Behavior': prev_behavior,
                'To Behavior': curr_behavior,
                'Gap Before Transition (s)': gap,
                'From ROI': prev_roi,
                'To ROI': curr_roi,
            })

    if not transitions:
        return {}

    transitions_df = pd.DataFrame(transitions)
    module_dir = context.get_module_dir('behavior_transitions')
    summary_counts = transitions_df.groupby(['From Behavior', 'To Behavior']).size().reset_index(name='Transition Count')
    total_transitions = summary_counts['Transition Count'].sum()
    if total_transitions > 0:
        summary_counts['Transition Percentage'] = (summary_counts['Transition Count'] / total_transitions) * 100.0
    summary_counts.to_csv(os.path.join(module_dir, f"{context.video_name}_behavior_transition_matrix.csv"), index=False)

    per_track = transitions_df.groupby(['Track ID', 'From Behavior', 'To Behavior']).agg(
        Transition_Count=('Track ID', 'size'),
        Median_Gap_s=('Gap Before Transition (s)', 'median')
    ).reset_index()
    per_track.to_csv(os.path.join(module_dir, f"{context.video_name}_behavior_transition_by_track.csv"), index=False)

    roi_columns_present = transitions_df[['From ROI', 'To ROI']].dropna(how='all')
    if not roi_columns_present.empty:
        roi_summary = roi_columns_present.groupby(['From ROI', 'To ROI']).size().reset_index(name='Transition Count')
        roi_summary.to_csv(os.path.join(module_dir, f"{context.video_name}_behavior_transition_by_roi.csv"), index=False)
    return {
        'files': {
            'global_matrix': os.path.join(module_dir, f"{context.video_name}_behavior_transition_matrix.csv"),
            'per_track': os.path.join(module_dir, f"{context.video_name}_behavior_transition_by_track.csv"),
        }
    }


def _module_temporal_trends(context: AnalyticsModuleContext, bin_size_seconds: int = 60) -> Dict[str, Any]:
    df = context.detailed_bouts_df
    if df.empty:
        return {}

    records: List[Dict[str, Any]] = []
    count_records: List[Dict[str, Any]] = []
    roi_records: List[Dict[str, Any]] = []

    columns = list(df.columns)
    idx_behavior = columns.index('Behavior')
    idx_start_time = columns.index('Start Time (s)') if 'Start Time (s)' in columns else None
    idx_end_time = columns.index('End Time (s)') if 'End Time (s)' in columns else None
    idx_roi = columns.index('ROI Name') if 'ROI Name' in columns else None
    idx_start_frame = columns.index('Start Frame') if 'Start Frame' in columns else None
    idx_end_frame = columns.index('End Frame') if 'End Frame' in columns else None
    fps = max(context.fps, 1e-6)

    for row_idx, row in enumerate(df.itertuples(index=False), start=1):
        _check_stop_interval(context.stop_check, row_idx, stage="Temporal trend analysis")
        behavior = row[idx_behavior]
        start = row[idx_start_time] if idx_start_time is not None else None
        end = row[idx_end_time] if idx_end_time is not None else None
        if start is None or pd.isna(start):
            if idx_start_frame is not None:
                start = float(row[idx_start_frame]) / fps
            else:
                start = 0.0
        else:
            start = float(start)
        if end is None or pd.isna(end):
            if idx_end_frame is not None:
                end = float(row[idx_end_frame]) / fps
            else:
                end = start
        else:
            end = float(end)
        roi_name = row[idx_roi] if idx_roi is not None else None
        start_bin = int(np.floor(start / bin_size_seconds))
        end_bin = int(np.floor(end / bin_size_seconds))
        for bin_idx in range(start_bin, end_bin + 1):
            bin_start = bin_idx * bin_size_seconds
            bin_end = bin_start + bin_size_seconds
            overlap = max(0.0, min(end, bin_end) - max(start, bin_start))
            if overlap <= 0.0:
                continue
            records.append({
                'Time Bin Start (s)': bin_start,
                'Time Bin End (s)': bin_end,
                'Behavior': behavior,
                'Duration (s)': overlap,
            })
            if roi_name not in (None, '', 'None'):
                roi_records.append({
                    'Time Bin Start (s)': bin_start,
                    'Time Bin End (s)': bin_end,
                    'Behavior': behavior,
                    'ROI Name': roi_name,
                    'Duration (s)': overlap,
                })
        bin_for_count = start_bin
        count_records.append({
            'Time Bin Start (s)': bin_for_count,
            'Behavior': behavior,
            'Bout Count': 1,
        })

    if not records:
        return {}

    module_dir = context.get_module_dir('temporal_trends')
    duration_df = pd.DataFrame(records)
    duration_summary = duration_df.groupby(
        ['Time Bin Start (s)', 'Time Bin End (s)', 'Behavior']
    ).agg(Total_Duration_s=('Duration (s)', 'sum')).reset_index()
    duration_path = os.path.join(module_dir, f"{context.video_name}_behavior_time_by_bin.csv")
    duration_summary.to_csv(duration_path, index=False)

    count_df = pd.DataFrame(count_records)
    count_path = None
    if not count_df.empty:
        count_summary = count_df.groupby(['Time Bin Start (s)', 'Behavior']).agg(Bout_Count=('Bout Count', 'sum')).reset_index()
        count_summary['Time Bin End (s)'] = (count_summary['Time Bin Start (s)'] + 1) * bin_size_seconds
        count_summary['Time Bin Start (s)'] *= bin_size_seconds
        count_path = os.path.join(module_dir, f"{context.video_name}_behavior_counts_by_bin.csv")
        count_summary.to_csv(count_path, index=False)

    roi_path = None
    if roi_records:
        roi_df = pd.DataFrame(roi_records)
        roi_summary = roi_df.groupby(['Time Bin Start (s)', 'Time Bin End (s)', 'ROI Name', 'Behavior']).agg(
            Total_Duration_s=('Duration (s)', 'sum')
        ).reset_index()
        roi_path = os.path.join(module_dir, f"{context.video_name}_roi_behavior_time_by_bin.csv")
        roi_summary.to_csv(roi_path, index=False)

    cumulative_path = None
    if not duration_summary.empty:
        cumulative_summary = duration_summary.sort_values('Time Bin End (s)').copy()
        cumulative_summary['Cumulative_Duration_s'] = cumulative_summary.groupby('Behavior')['Total_Duration_s'].cumsum()
        cumulative_path = os.path.join(module_dir, f"{context.video_name}_behavior_time_cumulative.csv")
        cumulative_summary.to_csv(cumulative_path, index=False)
    else:
        cumulative_summary = pd.DataFrame()

    visual_mode = context.get_visual_pref('temporal_trends', 'line bar')
    plot_lines = 'line' in visual_mode and not cumulative_summary.empty
    plot_bars = 'bar' in visual_mode and not duration_summary.empty

    plot_files: Dict[str, str] = {}
    if plot_lines or plot_bars:
        try:
            import matplotlib
            try:
                matplotlib.get_backend()
            except Exception:
                pass
            if matplotlib.get_backend().lower() != 'agg':
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - matplotlib optional
            logger.warning("Temporal trends visualization skipped (matplotlib unavailable): %s", exc)
            plot_lines = plot_bars = False

    if plot_lines:
        try:
            pivot_cum = cumulative_summary.pivot_table(
                index='Time Bin End (s)',
                columns='Behavior',
                values='Cumulative_Duration_s',
                fill_value=0.0,
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            for behavior in pivot_cum.columns:
                ax.plot(pivot_cum.index, pivot_cum[behavior], label=str(behavior))
            ax.set_ylabel("Cumulative Duration (s)")
            ax.set_xlabel("Time (s)")
            ax.set_title("Cumulative Bout Duration per Behavior")
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
            ax.grid(True, alpha=0.2)
            line_path = os.path.join(module_dir, f"{context.video_name}_behavior_time_cumulative.png")
            fig.tight_layout()
            fig.savefig(line_path, dpi=200)
            plt.close(fig)
            plot_files['cumulative_plot'] = line_path
        except Exception as exc:  # pragma: no cover - best effort plotting
            logger.warning("Failed to render temporal cumulative plot: %s", exc)

    if plot_bars:
        try:
            pivot_bar = duration_summary.pivot_table(
                index='Time Bin Start (s)',
                columns='Behavior',
                values='Total_Duration_s',
                fill_value=0.0,
            ).sort_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            bottoms = np.zeros(len(pivot_bar))
            for behavior in pivot_bar.columns:
                ax.bar(
                    pivot_bar.index,
                    pivot_bar[behavior].to_numpy(),
                    bottom=bottoms,
                    width=bin_size_seconds * 0.9,
                    label=str(behavior),
                )
                bottoms += pivot_bar[behavior].to_numpy()
            ax.set_xlabel("Time Bin Start (s)")
            ax.set_ylabel("Bout Duration within Bin (s)")
            ax.set_title("Stacked Bout Duration per Behavior")
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
            ax.grid(True, axis='y', alpha=0.2)
            bar_path = os.path.join(module_dir, f"{context.video_name}_behavior_time_stacked.png")
            fig.tight_layout()
            fig.savefig(bar_path, dpi=200)
            plt.close(fig)
            plot_files['stacked_plot'] = bar_path
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to render temporal stacked bar plot: %s", exc)

    files = {'duration': duration_path}
    if count_path:
        files['counts'] = count_path
    if roi_path:
        files['roi_duration'] = roi_path
    if cumulative_path:
        files['cumulative'] = cumulative_path
    files.update(plot_files)

    return {'files': files}


def _module_activity_budgets(context: AnalyticsModuleContext) -> Dict[str, Any]:
    df = context.detailed_bouts_df
    if df.empty:
        return {}

    module_dir = context.get_module_dir('activity_budgets')
    session_duration = context.session_duration_s
    if not np.isfinite(session_duration) or session_duration <= 0.0:
        session_duration = df['Duration (s)'].sum()

    global_summary = df.groupby('Behavior').agg(
        Total_Time_s=('Duration (s)', 'sum')
    ).reset_index()
    global_summary['Proportion_of_Session'] = np.where(
        session_duration > 0.0,
        global_summary['Total_Time_s'] / session_duration,
        np.nan,
    )
    global_path = os.path.join(module_dir, f"{context.video_name}_activity_budget_global.csv")
    global_summary.to_csv(global_path, index=False)

    per_track = df.groupby(['Track ID', 'Behavior']).agg(
        Total_Time_s=('Duration (s)', 'sum')
    ).reset_index()
    per_track['Track_Total_Time_s'] = per_track.groupby('Track ID')['Total_Time_s'].transform('sum')
    per_track['Proportion_of_Track_Time'] = np.where(
        per_track['Track_Total_Time_s'] > 0.0,
        per_track['Total_Time_s'] / per_track['Track_Total_Time_s'],
        np.nan,
    )
    per_track_path = os.path.join(module_dir, f"{context.video_name}_activity_budget_per_track.csv")
    per_track.to_csv(per_track_path, index=False)

    roi_col = 'ROI Name' if 'ROI Name' in df.columns else context.roi_column
    if roi_col and roi_col in df.columns:
        roi_budget = df.groupby([roi_col, 'Behavior']).agg(
            Total_Time_s=('Duration (s)', 'sum')
        ).reset_index()
        roi_budget['ROI_Total_Time_s'] = roi_budget.groupby(roi_col)['Total_Time_s'].transform('sum')
        roi_budget['Proportion_of_ROI_Time'] = np.where(
            roi_budget['ROI_Total_Time_s'] > 0.0,
            roi_budget['Total_Time_s'] / roi_budget['ROI_Total_Time_s'],
            np.nan,
        )
        roi_budget_path = os.path.join(module_dir, f"{context.video_name}_activity_budget_per_roi.csv")
        roi_budget.to_csv(roi_budget_path, index=False)
    else:
        roi_budget_path = None

    visual_mode = context.get_visual_pref('activity_budgets', 'stacked bar')
    plot_stacked = 'stack' in visual_mode
    plot_violin = 'violin' in visual_mode

    plot_files: Dict[str, str] = {}
    if plot_stacked or plot_violin:
        try:
            import matplotlib
            try:
                matplotlib.get_backend()
            except Exception:
                pass
            if matplotlib.get_backend().lower() != 'agg':
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            logger.warning("Activity budget visualization skipped (matplotlib unavailable): %s", exc)
            plot_stacked = plot_violin = False

    if plot_stacked and not per_track.empty:
        try:
            pivot = per_track.pivot_table(
                index='Track ID',
                columns='Behavior',
                values='Proportion_of_Track_Time',
                fill_value=0.0,
            ).sort_index()
            width = max(6.0, min(14.0, 1.2 * len(pivot.index)))
            fig, ax = plt.subplots(figsize=(width, 5))
            bottoms = np.zeros(len(pivot.index))
            x_labels = [str(lbl) for lbl in pivot.index]
            x_positions = np.arange(len(pivot.index))
            for behavior in pivot.columns:
                ax.bar(
                    x_positions,
                    pivot[behavior].to_numpy(),
                    bottom=bottoms,
                    label=str(behavior),
                )
                bottoms += pivot[behavior].to_numpy()
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.set_xlabel("Track ID")
            ax.set_ylabel("Proportion of Track Time")
            ax.set_title("Per-Track Activity Budget (Stacked)")
            ax.set_ylim(0, 1.05)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
            ax.grid(True, axis='y', alpha=0.2)
            stacked_path = os.path.join(module_dir, f"{context.video_name}_activity_budget_stacked.png")
            fig.tight_layout()
            fig.savefig(stacked_path, dpi=200)
            plt.close(fig)
            plot_files['stacked_plot'] = stacked_path
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to render stacked activity budget plot: %s", exc)

    if plot_violin:
        try:
            durations = context.detailed_bouts_df[['Behavior', 'Duration (s)']].dropna()
            if not durations.empty:
                behaviors = sorted(durations['Behavior'].unique())
                data = [durations.loc[durations['Behavior'] == bh, 'Duration (s)'].to_numpy() for bh in behaviors]
                if any(len(arr) for arr in data):
                    fig, ax = plt.subplots(figsize=(max(6.0, 1.0 * len(behaviors)), 5))
                    parts = ax.violinplot(data, showmeans=True, showextrema=False)
                    for pc in parts['bodies']:
                        pc.set_facecolor('#4C72B0')
                        pc.set_alpha(0.6)
                    ax.set_xticks(np.arange(1, len(behaviors) + 1))
                    ax.set_xticklabels(behaviors, rotation=45, ha='right')
                    ax.set_xlabel("Behavior")
                    ax.set_ylabel("Bout Duration (s)")
                    ax.set_title("Bout Duration Distributions per Behavior")
                    violin_path = os.path.join(module_dir, f"{context.video_name}_activity_budget_violin.png")
                    fig.tight_layout()
                    fig.savefig(violin_path, dpi=200)
                    plt.close(fig)
                    plot_files['violin_plot'] = violin_path
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to render activity budget violin plot: %s", exc)

    files = {
        'global': global_path,
        'per_track': per_track_path,
    }
    if roi_budget_path:
        files['per_roi'] = roi_budget_path
    files.update(plot_files)
    return {'files': files}


def _module_multi_animal_descriptors(context: AnalyticsModuleContext) -> Dict[str, Any]:
    df = context.per_frame_df
    if df.empty or 'track_id' not in df.columns or df['track_id'].nunique() < 2:
        return {}

    module_dir = context.get_module_dir('multi_animal_descriptors')
    fps = max(context.fps, 1e-6)
    positions = (
        df[['frame', 'track_id', 'x_center', 'y_center']]
        .dropna(subset=['frame', 'track_id', 'x_center', 'y_center'])
        .drop_duplicates(subset=['frame', 'track_id'])
        .sort_values(['frame', 'track_id'])
    )

    threshold_array = np.asarray([0.05, 0.1, 0.2], dtype=float)
    pair_stats: Dict[Tuple[int, int], Dict[str, Any]] = defaultdict(
        lambda: {
            'frame_count': 0,
            'distance_sum': 0.0,
            'threshold_counts': np.zeros_like(threshold_array, dtype=float),
        }
    )

    for _, frame_group in positions.groupby('frame', sort=False):
        track_ids = frame_group['track_id'].to_numpy(dtype=int)
        coords = frame_group[['x_center', 'y_center']].to_numpy(dtype=float)
        count = coords.shape[0]
        if count < 2:
            continue
        diffs = coords[:, None, :] - coords[None, :, :]
        dist_matrix = np.sqrt(np.sum(diffs * diffs, axis=2))
        upper_idx = np.triu_indices(count, k=1)
        tracks_a = track_ids[upper_idx[0]]
        tracks_b = track_ids[upper_idx[1]]
        distances = dist_matrix[upper_idx]
        if distances.size == 0:
            continue
        comparisons = distances[:, None] <= threshold_array[None, :]
        for ta, tb, distance, comparison in zip(tracks_a, tracks_b, distances, comparisons):
            pair_key = (int(min(ta, tb)), int(max(ta, tb)))
            stats = pair_stats[pair_key]
            stats['frame_count'] += 1
            stats['distance_sum'] += float(distance)
            stats['threshold_counts'] += comparison.astype(float)

    proximity_rows = []
    for pair_key, stats in pair_stats.items():
        frame_count = stats['frame_count']
        threshold_counts = stats['threshold_counts']
        proximity_rows.append({
            'Track A': pair_key[0],
            'Track B': pair_key[1],
            'Frames_Evaluated': frame_count,
            'Time_Evaluated_s': frame_count / fps,
            'Mean_Centroid_Distance': stats['distance_sum'] / frame_count if frame_count else np.nan,
            **{f"Time_within_{threshold:.2f}_s": float(threshold_counts[idx]) / fps for idx, threshold in enumerate(threshold_array)},
        })
    proximity_df = pd.DataFrame(proximity_rows)
    proximity_path = os.path.join(module_dir, f"{context.video_name}_pairwise_proximity.csv")
    proximity_df.to_csv(proximity_path, index=False)

    # Behavior co-occurrence
    bout_df = context.detailed_bouts_df
    overlap_records = []
    if not bout_df.empty and 'Track ID' in bout_df.columns:
        track_groups = {
            track_id: group.sort_values('Start Time (s)')[['Start Time (s)', 'End Time (s)', 'Behavior']].to_numpy()
            for track_id, group in bout_df.groupby('Track ID')
        }
        track_ids = sorted(track_groups.keys())
        for idx_a in range(len(track_ids)):
            for idx_b in range(idx_a + 1, len(track_ids)):
                track_a = track_ids[idx_a]
                track_b = track_ids[idx_b]
                bouts_a = track_groups[track_a]
                bouts_b = track_groups[track_b]
                ia = ib = 0
                while ia < len(bouts_a) and ib < len(bouts_b):
                    start_a, end_a, behavior_a = bouts_a[ia]
                    start_b, end_b, behavior_b = bouts_b[ib]
                    overlap_start = max(start_a, start_b)
                    overlap_end = min(end_a, end_b)
                    if overlap_end > overlap_start:
                        overlap_records.append({
                            'Track A': track_a,
                            'Track B': track_b,
                            'Behavior A': behavior_a,
                            'Behavior B': behavior_b,
                            'Overlap Duration (s)': overlap_end - overlap_start,
                        })
                    if end_a < end_b:
                        ia += 1
                    else:
                        ib += 1
    if overlap_records:
        overlap_df = pd.DataFrame(overlap_records)
        overlap_path = os.path.join(module_dir, f"{context.video_name}_behavior_overlap.csv")
        overlap_df.to_csv(overlap_path, index=False)
        files = {'proximity': proximity_path, 'behavior_overlap': overlap_path}
    else:
        files = {'proximity': proximity_path}
    return {'files': files}


def _module_roi_time_heatmap(context: AnalyticsModuleContext, bin_size_seconds: int = 60) -> Dict[str, Any]:
    if not context.roi_column:
        return {}
    df = context.per_frame_df
    if df.empty or context.roi_column not in df.columns or 'frame' not in df.columns:
        return {}

    module_dir = context.get_module_dir('roi_time_heatmap')
    fps = max(context.fps, 1e-6)
    roi_col = context.roi_column
    roi_df = (
        df[['frame', 'track_id', roi_col]]
        .dropna(subset=[context.roi_column])
        .drop_duplicates(subset=['frame', 'track_id'])
    )
    if roi_df.empty:
        return {}

    roi_df['Time (s)'] = roi_df['frame'] / fps
    roi_df['Time Bin'] = (roi_df['Time (s)'] // bin_size_seconds).astype(int)
    occupancy = roi_df.groupby([roi_col, 'Time Bin']).size().reset_index(name='Frame_Count')
    occupancy['Time Bin Start (s)'] = occupancy['Time Bin'] * bin_size_seconds
    occupancy['Time Bin End (s)'] = occupancy['Time Bin Start (s)'] + bin_size_seconds
    occupancy['Time in ROI (s)'] = occupancy['Frame_Count'] / fps
    occupancy_csv = os.path.join(module_dir, f"{context.video_name}_roi_time_heatmap.csv")
    occupancy[[roi_col, 'Time Bin Start (s)', 'Time Bin End (s)', 'Time in ROI (s)']].to_csv(occupancy_csv, index=False)

    pivot = occupancy.pivot(index=roi_col, columns='Time Bin Start (s)', values='Time in ROI (s)').fillna(0.0)
    pivot_csv = os.path.join(module_dir, f"{context.video_name}_roi_time_heatmap_matrix.csv")
    pivot.to_csv(pivot_csv)

    return {'files': {'occupancy': occupancy_csv, 'matrix': pivot_csv}}


def _module_kinematic_descriptors(context: AnalyticsModuleContext) -> Dict[str, Any]:
    df = context.per_frame_df
    if df.empty or not {'track_id', 'frame', 'x_center', 'y_center'}.issubset(df.columns):
        return {}
    if context.detailed_bouts_df.empty:
        return {}

    module_dir = context.get_module_dir('kinematic_descriptors')
    fps = max(context.fps, 1e-6)

    def _extract_joint_angles(payload: Any) -> List[float]:
        if payload in (None, '', []):
            return []
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                return []
        if not isinstance(payload, (list, tuple)):
            return []
        try:
            arr = np.asarray(payload, dtype=float)
        except Exception:
            return []
        if arr.size == 0:
            return []
        if arr.ndim == 1:
            kp_dim = 3 if arr.size % 3 == 0 else 2
            try:
                arr = arr.reshape(-1, kp_dim)
            except ValueError:
                return []
        coords = arr[:, :2]
        if coords.shape[0] < 3:
            return []
        mask = np.isfinite(coords).all(axis=1)
        if arr.shape[1] >= 3:
            try:
                mask &= np.asarray(arr[:, 2], dtype=float) > 0.0
            except Exception:
                pass
        coords = coords[mask]
        if coords.shape[0] < 3:
            return []
        angles: List[float] = []
        for idx in range(1, coords.shape[0] - 1):
            v1 = coords[idx - 1] - coords[idx]
            v2 = coords[idx + 1] - coords[idx]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0.0 or norm2 == 0.0:
                continue
            cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
            angles.append(float(np.degrees(np.arccos(cos_theta))))
        return angles

    keypoint_df = context.per_frame_df[['track_id', 'frame', 'keypoints']].dropna(subset=['keypoints']).copy()
    angle_records = []
    for row in keypoint_df.itertuples(index=False):
        track_id = getattr(row, 'track_id', None)
        frame = getattr(row, 'frame', None)
        if track_id in (None, np.nan) or frame in (None, np.nan):
            continue
        try:
            track_id_int = int(track_id)
            frame_int = int(frame)
        except (TypeError, ValueError):
            continue
        angles = _extract_joint_angles(getattr(row, 'keypoints', None))
        if not angles:
            continue
        angle_records.append({
            'track_id': track_id_int,
            'frame': frame_int,
            'angle_mean': float(np.mean(angles)),
            'angle_std': float(np.std(angles)),
            'angle_range': float(np.ptp(angles)),
        })
    angle_df = pd.DataFrame(angle_records) if angle_records else pd.DataFrame()

    positions = (
        df[['track_id', 'frame', 'x_center', 'y_center']]
        .dropna()
        .drop_duplicates(subset=['track_id', 'frame'])
        .sort_values(['track_id', 'frame'])
    )
    if positions.empty:
        return {}

    positions['dx'] = positions.groupby('track_id')['x_center'].diff()
    positions['dy'] = positions.groupby('track_id')['y_center'].diff()
    positions['step_distance_norm'] = np.hypot(positions['dx'], positions['dy'])
    positions['speed_norm_per_s'] = positions['step_distance_norm'] * fps
    positions['accel_norm_per_s2'] = positions.groupby('track_id')['speed_norm_per_s'].diff() * fps

    track_maps = {track_id: group.set_index('frame') for track_id, group in positions.groupby('track_id')}

    columns = list(context.detailed_bouts_df.columns)
    idx_track = columns.index('Track ID')
    idx_behavior = columns.index('Behavior')
    idx_start_frame = columns.index('Start Frame')
    idx_end_frame = columns.index('End Frame')

    records = []
    for row in context.detailed_bouts_df.itertuples(index=False):
        track_id = row[idx_track]
        start_frame = row[idx_start_frame]
        end_frame = row[idx_end_frame]
        if pd.isna(track_id) or pd.isna(start_frame) or pd.isna(end_frame):
            continue
        track_id_int = int(track_id)
        frame_start = int(start_frame)
        frame_end = int(end_frame)
        track_df = track_maps.get(track_id_int)
        if track_df is None:
            continue
        window = track_df.loc[frame_start:frame_end]
        if window.empty:
            continue
        speeds = window['speed_norm_per_s'].dropna()
        accels = window['accel_norm_per_s2'].dropna()
        angle_metrics = angle_df[
            (angle_df['track_id'] == track_id_int)
            & (angle_df['frame'] >= frame_start)
            & (angle_df['frame'] <= frame_end)
        ] if not angle_df.empty else pd.DataFrame()
        angle_mean = angle_metrics['angle_mean'].mean() if not angle_metrics.empty else np.nan
        angle_std = angle_metrics['angle_std'].mean() if not angle_metrics.empty else np.nan
        angle_range = angle_metrics['angle_range'].mean() if not angle_metrics.empty else np.nan
        records.append({
            'Track ID': track_id_int,
            'Behavior': row[idx_behavior],
            'Start Frame': frame_start,
            'End Frame': frame_end,
            'Mean Speed (norm units/s)': speeds.mean() if not speeds.empty else np.nan,
            'Peak Speed (norm units/s)': speeds.max() if not speeds.empty else np.nan,
            'Mean Accel (norm units/s^2)': accels.mean() if not accels.empty else np.nan,
            'Peak Accel (norm units/s^2)': accels.max() if not accels.empty else np.nan,
            'Path Length (norm units)': window['step_distance_norm'].sum(),
            'Mean Joint Angle (deg)': angle_mean,
            'Joint Angle Std (deg)': angle_std,
            'Joint Angle Range (deg)': angle_range,
        })

    if not records:
        return {}

    kinematics_df = pd.DataFrame(records)
    kinematics_path = os.path.join(module_dir, f"{context.video_name}_bout_kinematics.csv")
    kinematics_df.to_csv(kinematics_path, index=False)
    return {'files': {'kinematics': kinematics_path}}


def _compute_keypoint_stats(keypoints: Any) -> Tuple[float, float, float]:
    if not keypoints:
        return np.nan, np.nan, np.nan
    confidences = []
    present_count = 0
    for kp in keypoints:
        if not isinstance(kp, (list, tuple)) or len(kp) < 3:
            continue
        conf = kp[2]
        if pd.isna(conf):
            continue
        confidences.append(float(conf))
        if float(conf) > 0.0:
            present_count += 1
    if not confidences:
        return np.nan, np.nan, np.nan
    mean_conf = float(np.mean(confidences))
    std_conf = float(np.std(confidences))
    completeness = present_count / max(len(keypoints), 1)
    return mean_conf, std_conf, completeness


def _module_detection_quality(context: AnalyticsModuleContext) -> Dict[str, Any]:
    df = context.per_frame_df
    if df.empty or 'keypoints' not in df.columns:
        return {}
    if context.detailed_bouts_df.empty:
        return {}

    module_dir = context.get_module_dir('detection_quality')
    quality_df = df[['track_id', 'frame', 'class_id', 'keypoints', 'w', 'h']].dropna(subset=['track_id', 'frame']).reset_index(drop=True)
    if quality_df.empty:
        return {}
    stats_list = [_compute_keypoint_stats(kps) for kps in quality_df['keypoints']]
    stats_df = pd.DataFrame(stats_list, columns=['kp_conf_mean', 'kp_conf_std', 'kp_completeness'])
    quality_df = pd.concat([quality_df, stats_df], axis=1)
    quality_df['bbox_area'] = quality_df['w'] * quality_df['h']

    track_class_groups = quality_df.groupby(['track_id', 'class_id'])
    records = []
    name_lookup = {v: k for k, v in context.class_names.items()}
    columns = list(context.detailed_bouts_df.columns)
    idx_track = columns.index('Track ID')
    idx_behavior = columns.index('Behavior')
    idx_start_frame = columns.index('Start Frame')
    idx_end_frame = columns.index('End Frame')
    for bout in context.detailed_bouts_df.itertuples(index=False):
        track_id = bout[idx_track]
        behavior = bout[idx_behavior]
        start_frame = bout[idx_start_frame]
        end_frame = bout[idx_end_frame]
        if pd.isna(track_id) or pd.isna(start_frame) or pd.isna(end_frame):
            continue
        track_id_int = int(track_id)
        class_id = context.behavior_to_class_id.get(str(behavior))
        if class_id is None:
            class_id = name_lookup.get(behavior)
        group_key = (track_id_int, class_id) if class_id is not None else None
        if group_key and group_key in track_class_groups.groups:
            class_frames = track_class_groups.get_group(group_key)
        else:
            class_frames = quality_df[quality_df['track_id'] == track_id_int]
        frame_start = int(start_frame)
        frame_end = int(end_frame)
        window = class_frames[(class_frames['frame'] >= frame_start) & (class_frames['frame'] <= frame_end)]
        if window.empty:
            continue
        records.append({
            'Track ID': track_id_int,
            'Behavior': behavior,
            'Start Frame': frame_start,
            'End Frame': frame_end,
            'Mean Keypoint Confidence': window['kp_conf_mean'].mean(),
            'Min Keypoint Confidence': window['kp_conf_mean'].min(),
            'Mean Keypoint Completeness': window['kp_completeness'].mean(),
            'BBox Area Variance': window['bbox_area'].var(ddof=0),
        })

    if not records:
        return {}

    quality_summary = pd.DataFrame(records)
    quality_path = os.path.join(module_dir, f"{context.video_name}_detection_quality.csv")
    quality_summary.to_csv(quality_path, index=False)
    return {'files': {'quality': quality_path}}


def _module_inter_bout_intervals(context: AnalyticsModuleContext) -> Dict[str, Any]:
    df = context.detailed_bouts_df
    if df.empty:
        return {}

    module_dir = context.get_module_dir('inter_bout_intervals')
    interval_records = []
    summary_records = []
    group_cols = ['Track ID', 'Behavior']
    for (track_id, behavior), group in df.groupby(group_cols):
        group_sorted = group.sort_values('Start Time (s)', kind='mergesort')
        start_times = group_sorted['Start Time (s)'].to_numpy(dtype=float)
        end_times = group_sorted['End Time (s)'].to_numpy(dtype=float)
        valid_mask = np.isfinite(start_times) & np.isfinite(end_times)
        start_times = start_times[valid_mask]
        end_times = end_times[valid_mask]
        if start_times.size == 0:
            continue
        first_start = start_times[0]
        prev_end = None
        intervals = []
        for start_time, end_time in zip(start_times, end_times):
            if prev_end is not None:
                gap = start_time - prev_end
                intervals.append(gap)
                interval_records.append({
                    'Track ID': track_id,
                    'Behavior': behavior,
                    'Interval Start (s)': prev_end,
                    'Interval End (s)': start_time,
                    'Inter-Bout Interval (s)': gap,
                })
            prev_end = end_time
        if intervals:
            summary_records.append({
                'Track ID': track_id,
                'Behavior': behavior,
                'Median Inter-Bout Interval (s)': float(np.median(intervals)),
                'Mean Inter-Bout Interval (s)': float(np.mean(intervals)),
            })
        summary_records.append({
            'Track ID': track_id,
            'Behavior': behavior,
            'Latency to First Bout (s)': first_start,
        })

    interval_df = pd.DataFrame(interval_records)
    summary_df = pd.DataFrame(summary_records)
    files = {}
    if not interval_df.empty:
        interval_path = os.path.join(module_dir, f"{context.video_name}_inter_bout_intervals.csv")
        interval_df.to_csv(interval_path, index=False)
        files['intervals'] = interval_path
    if not summary_df.empty:
        summary_path = os.path.join(module_dir, f"{context.video_name}_inter_bout_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        files['summary'] = summary_path
    return {'files': files}


def _module_roi_context_windows(context: AnalyticsModuleContext, frame_window: int = 1) -> Dict[str, Any]:
    if not context.roi_column:
        return {}
    df = context.per_frame_df
    if df.empty or 'frame' not in df.columns or 'track_id' not in df.columns:
        return {}
    if context.detailed_bouts_df.empty:
        return {}

    module_dir = context.get_module_dir('roi_context_windows')
    roi_lookup = (
        df[['track_id', 'frame', context.roi_column]]
        .dropna(subset=[context.roi_column])
        .drop_duplicates(subset=['track_id', 'frame'])
        .set_index(['track_id', 'frame'])
    )

    def _lookup(track_id: int, frame: int) -> Optional[str]:
        key = (track_id, frame)
        if key in roi_lookup.index:
            value = roi_lookup.loc[key]
            if isinstance(value, pd.Series):
                return value.iloc[0]
            return value
        return None

    columns = list(context.detailed_bouts_df.columns)
    idx_track = columns.index('Track ID')
    idx_behavior = columns.index('Behavior')
    idx_start_frame = columns.index('Start Frame') if 'Start Frame' in columns else None
    idx_end_frame = columns.index('End Frame') if 'End Frame' in columns else None
    idx_roi_name = columns.index('ROI Name') if 'ROI Name' in columns else None
    idx_roi_alt = columns.index(context.roi_column) if context.roi_column and context.roi_column in columns else None

    records = []
    for row in context.detailed_bouts_df.itertuples(index=False):
        track_id = row[idx_track]
        if pd.isna(track_id):
            continue
        track_id_int = int(track_id)
        start_frame = int(row[idx_start_frame]) if idx_start_frame is not None else 0
        end_frame = int(row[idx_end_frame]) if idx_end_frame is not None else start_frame
        pre_roi = _lookup(track_id_int, start_frame - frame_window)
        post_roi = _lookup(track_id_int, end_frame + frame_window)
        if idx_roi_name is not None:
            during_roi = row[idx_roi_name]
        elif idx_roi_alt is not None:
            during_roi = row[idx_roi_alt]
        else:
            during_roi = None
        records.append({
            'Track ID': track_id_int,
            'Behavior': row[idx_behavior],
            'Start Frame': start_frame,
            'End Frame': end_frame,
            'Pre ROI': pre_roi,
            'Bout ROI': during_roi,
            'Post ROI': post_roi,
        })

    context_df = pd.DataFrame(records)
    if context_df.empty:
        return {}
    context_path = os.path.join(module_dir, f"{context.video_name}_roi_context_windows.csv")
    context_df.to_csv(context_path, index=False)
    return {'files': {'context_windows': context_path}}


def _module_bout_timeline_export(context: AnalyticsModuleContext) -> Dict[str, Any]:
    df = context.detailed_bouts_df
    if df.empty:
        return {}

    module_dir = context.get_module_dir('bout_timeline_export')
    timeline_df = df.copy()
    timeline_df = timeline_df.sort_values(['Track ID', 'Start Time (s)'])
    session_duration = context.session_duration_s
    if np.isfinite(session_duration) and session_duration > 0.0:
        timeline_df['Normalized Start'] = timeline_df['Start Time (s)'] / session_duration
        timeline_df['Normalized End'] = timeline_df['End Time (s)'] / session_duration
    timeline_path = os.path.join(module_dir, f"{context.video_name}_bout_timeline.csv")
    timeline_df.to_csv(timeline_path, index=False)

    files = {'timeline': timeline_path}

    visual_mode = context.get_visual_pref('bout_timeline', 'gantt')
    plot_gantt = 'gantt' in visual_mode

    if plot_gantt and not timeline_df.empty:
        try:
            import matplotlib
            try:
                matplotlib.get_backend()
            except Exception:
                pass
            if matplotlib.get_backend().lower() != 'agg':
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
        except Exception as exc:  # pragma: no cover
            logger.warning("Bout timeline visualization skipped (matplotlib unavailable): %s", exc)
            plot_gantt = False

    if plot_gantt and not timeline_df.empty:
        try:
            tracks = sorted(timeline_df['Track ID'].dropna().astype(int).unique().tolist())
            if not tracks:
                plot_gantt = False
            else:
                behaviors = sorted(timeline_df['Behavior'].dropna().unique().tolist())
                cmap = plt.get_cmap('tab20', max(len(behaviors), 1))
                behavior_colors = {beh: cmap(idx % cmap.N) for idx, beh in enumerate(behaviors)}

                status_column = next((col for col in timeline_df.columns if 'status' in col.lower()), None)
                status_styles = {
                    'confirmed': {'alpha': 0.95, 'edgecolor': '#2ca02c', 'linewidth': 1.5},
                    'rejected': {'alpha': 0.4, 'edgecolor': '#d62728', 'linewidth': 1.5},
                    'unreviewed': {'alpha': 0.65, 'edgecolor': '#7f7f7f', 'linewidth': 1.0},
                }

                height = max(4.0, min(12.0, 0.6 * len(tracks) + 2))
                fig, ax = plt.subplots(figsize=(12, height))
                for idx, track_id in enumerate(tracks):
                    track_rows = timeline_df[timeline_df['Track ID'] == track_id]
                    for _, row in track_rows.iterrows():
                        start = float(row.get('Start Time (s)', 0.0))
                        end = float(row.get('End Time (s)', start))
                        duration = max(end - start, 0.0)
                        color = behavior_colors.get(row.get('Behavior'), '#1f77b4')
                        style = {'alpha': 0.75, 'edgecolor': 'black', 'linewidth': 0.8}
                        if status_column:
                            status_value = str(row.get(status_column, '')).lower()
                            for key, style_values in status_styles.items():
                                if key in status_value:
                                    style.update(style_values)
                                    break
                        ax.barh(
                            y=idx,
                            width=duration,
                            left=start,
                            height=0.5,
                            color=color,
                            **style,
                        )

                ax.set_yticks(range(len(tracks)))
                ax.set_yticklabels([str(t) for t in tracks])
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Track ID")
                ax.set_title("Bout Timeline")
                ax.grid(True, axis='x', alpha=0.2)

                behavior_handles = [Patch(facecolor=behavior_colors[beh], label=str(beh)) for beh in behaviors]
                legends = behavior_handles
                if status_column:
                    status_handles = [
                        Patch(facecolor='#ffffff', edgecolor=style['edgecolor'], linewidth=style['linewidth'], alpha=style['alpha'], label=label.title())
                        for label, style in status_styles.items()
                    ]
                    legends = [*behavior_handles, *status_handles]
                if legends:
                    ax.legend(handles=legends, loc='upper left', bbox_to_anchor=(1.02, 1.0))

                gantt_path = os.path.join(module_dir, f"{context.video_name}_bout_timeline_gantt.png")
                fig.tight_layout()
                fig.savefig(gantt_path, dpi=200)
                plt.close(fig)
                files['timeline_plot'] = gantt_path
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to render bout timeline Gantt plot: %s", exc)

    return {'files': files}


def _coerce_dataframe(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _parse_membership_names(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = text
    else:
        parsed = value

    if isinstance(parsed, str):
        text = parsed.strip()
        return (text,) if text else ()
    if isinstance(parsed, (list, tuple, set)):
        out: List[str] = []
        for item in parsed:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return tuple(sorted(set(out)))
    return ()


def _event_records_to_df(events: Any, *, name_key: str, source_label: str) -> pd.DataFrame:
    if not isinstance(events, dict):
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for event_type in ("entries", "exits"):
        event_rows = events.get(event_type, [])
        if not isinstance(event_rows, list):
            continue
        for event in event_rows:
            if not isinstance(event, dict):
                continue
            name = str(event.get(name_key, "") or "").strip()
            frame_val = pd.to_numeric(event.get("frame"), errors="coerce")
            track_val = pd.to_numeric(event.get("track_id"), errors="coerce")
            if not name or pd.isna(frame_val):
                continue
            rows.append(
                {
                    "Source": source_label,
                    "Event Type": "entry" if event_type == "entries" else "exit",
                    "Target Name": name,
                    "Frame": int(frame_val),
                    "Track ID": int(track_val) if pd.notna(track_val) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _build_first_membership_df(per_frame_df: pd.DataFrame, membership_col: str, *, source_label: str) -> pd.DataFrame:
    if per_frame_df.empty or membership_col not in per_frame_df.columns or "frame" not in per_frame_df.columns:
        return pd.DataFrame()
    work_df = per_frame_df.copy()
    work_df["frame"] = pd.to_numeric(work_df["frame"], errors="coerce")
    if "track_id" not in work_df.columns:
        work_df["track_id"] = 0
    work_df["track_id"] = pd.to_numeric(work_df["track_id"], errors="coerce").fillna(0).astype(int)
    work_df = work_df.dropna(subset=["frame"])
    rows: List[Dict[str, Any]] = []
    for _, row in work_df.iterrows():
        memberships = _parse_membership_names(row.get(membership_col))
        frame = row.get("frame")
        track_id = row.get("track_id", 0)
        if frame is None or pd.isna(frame):
            continue
        for name in memberships:
            rows.append(
                {
                    "Source": source_label,
                    "Target Name": name,
                    "Track ID": int(track_id),
                    "Frame": int(frame),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["Target Name", "Track ID", "Frame"]).reset_index(drop=True)


def _safe_preference_index(value_a: float, value_b: float) -> float:
    denom = float(value_a) + float(value_b)
    if not np.isfinite(denom) or abs(denom) <= 1e-12:
        return np.nan
    return (float(value_a) - float(value_b)) / denom


def _build_pairwise_preference_rows(
    summary_df: pd.DataFrame,
    *,
    source_label: str,
    name_col: str,
    measure_specs: Sequence[Tuple[str, str]],
    first_event_frames: Dict[str, int] | None = None,
) -> pd.DataFrame:
    if summary_df.empty or name_col not in summary_df.columns:
        return pd.DataFrame()
    lookup = {
        str(row[name_col]).strip(): row.to_dict()
        for _, row in summary_df.iterrows()
        if str(row.get(name_col, "")).strip()
    }
    names = sorted(lookup.keys())
    rows: List[Dict[str, Any]] = []
    first_event_frames = first_event_frames or {}
    for name_a, name_b in combinations(names, 2):
        record: Dict[str, Any] = {
            "Source": source_label,
            "Target A": name_a,
            "Target B": name_b,
        }
        for value_col, output_label in measure_specs:
            value_a = pd.to_numeric(lookup[name_a].get(value_col), errors="coerce")
            value_b = pd.to_numeric(lookup[name_b].get(value_col), errors="coerce")
            record[f"{output_label} A"] = float(value_a) if pd.notna(value_a) else np.nan
            record[f"{output_label} B"] = float(value_b) if pd.notna(value_b) else np.nan
            record[f"{output_label} Index"] = (
                _safe_preference_index(float(value_a), float(value_b))
                if pd.notna(value_a) and pd.notna(value_b)
                else np.nan
            )
        first_a = first_event_frames.get(name_a)
        first_b = first_event_frames.get(name_b)
        if first_a is None and first_b is None:
            record["First Choice"] = ""
        elif first_b is None or (first_a is not None and first_a < first_b):
            record["First Choice"] = name_a
        elif first_a is None or first_b < first_a:
            record["First Choice"] = name_b
        else:
            record["First Choice"] = "tie"
        rows.append(record)
    return pd.DataFrame(rows)


def _module_preference_indices(context: AnalyticsModuleContext) -> Dict[str, Any]:
    module_dir = context.get_module_dir("preference_indices")
    session_duration = context.session_duration_s if np.isfinite(context.session_duration_s) else np.nan
    files: Dict[str, str] = {}

    zone_summary = _coerce_dataframe(context.roi_metrics.get("entries_exits") if isinstance(context.roi_metrics, dict) else None)
    zone_events = _event_records_to_df(context.roi_events, name_key="roi_name", source_label="zone")
    if not zone_summary.empty and "ROI Name" in zone_summary.columns:
        zone_summary = zone_summary.copy()
        zone_summary["Percent of Session"] = np.where(
            np.isfinite(session_duration) & (session_duration > 0.0),
            pd.to_numeric(zone_summary.get("Time in ROI (s)"), errors="coerce") / float(session_duration),
            np.nan,
        )
        first_zone_frames = {}
        if not zone_events.empty:
            entries_df = zone_events[zone_events["Event Type"] == "entry"]
            if not entries_df.empty:
                first_zone_frames = (
                    entries_df.groupby("Target Name")["Frame"].min().astype(int).to_dict()
                )
                zone_summary["First Entry Frame"] = zone_summary["ROI Name"].map(first_zone_frames)
                zone_summary["First Entry Latency (s)"] = zone_summary["First Entry Frame"] / max(context.fps, 1e-6)
        zone_summary_path = os.path.join(module_dir, f"{context.video_name}_zone_preference_summary.csv")
        zone_summary.to_csv(zone_summary_path, index=False)
        files["zone_summary"] = zone_summary_path

        pairwise_zone = _build_pairwise_preference_rows(
            zone_summary,
            source_label="zone",
            name_col="ROI Name",
            measure_specs=(
                ("Time in ROI (s)", "Time Preference"),
                ("Entries", "Entry Preference"),
            ),
            first_event_frames=first_zone_frames,
        )
        if not pairwise_zone.empty:
            pairwise_zone_path = os.path.join(module_dir, f"{context.video_name}_zone_preference_pairwise.csv")
            pairwise_zone.to_csv(pairwise_zone_path, index=False)
            files["zone_pairwise"] = pairwise_zone_path

    object_summary = _coerce_dataframe(context.object_summary_df)
    object_events = _event_records_to_df(context.object_events, name_key="object_roi", source_label="object")
    if not object_summary.empty and "Object ROI" in object_summary.columns:
        object_summary = object_summary.copy()
        object_summary["Percent of Session"] = np.where(
            np.isfinite(session_duration) & (session_duration > 0.0),
            pd.to_numeric(object_summary.get("Time Interacting (s)"), errors="coerce") / float(session_duration),
            np.nan,
        )
        first_object_frames = {}
        if not object_events.empty:
            entries_df = object_events[object_events["Event Type"] == "entry"]
            if not entries_df.empty:
                first_object_frames = entries_df.groupby("Target Name")["Frame"].min().astype(int).to_dict()
                object_summary["First Interaction Frame"] = object_summary["Object ROI"].map(first_object_frames)
                object_summary["First Interaction Latency (s)"] = object_summary["First Interaction Frame"] / max(context.fps, 1e-6)
        object_summary_path = os.path.join(module_dir, f"{context.video_name}_object_preference_summary.csv")
        object_summary.to_csv(object_summary_path, index=False)
        files["object_summary"] = object_summary_path

        pairwise_object = _build_pairwise_preference_rows(
            object_summary,
            source_label="object",
            name_col="Object ROI",
            measure_specs=(
                ("Time Interacting (s)", "Interaction Preference"),
                ("Time Contact (s)", "Contact Preference"),
                ("Time Proximity Only (s)", "Proximity Preference"),
                ("Entries", "Entry Preference"),
            ),
            first_event_frames=first_object_frames,
        )
        if not pairwise_object.empty:
            pairwise_object_path = os.path.join(module_dir, f"{context.video_name}_object_preference_pairwise.csv")
            pairwise_object.to_csv(pairwise_object_path, index=False)
            files["object_pairwise"] = pairwise_object_path

    return {"files": files} if files else {}


def _module_latency_metrics(context: AnalyticsModuleContext) -> Dict[str, Any]:
    module_dir = context.get_module_dir("latency_metrics")
    fps = max(context.fps, 1e-6)
    files: Dict[str, str] = {}

    roi_event_df = _event_records_to_df(context.roi_events, name_key="roi_name", source_label="zone")
    if not roi_event_df.empty:
        zone_rows: List[Dict[str, Any]] = []
        for target_name, group in roi_event_df.groupby("Target Name"):
            entries = sorted(group.loc[group["Event Type"] == "entry", "Frame"].astype(int).tolist())
            exits = sorted(group.loc[group["Event Type"] == "exit", "Frame"].astype(int).tolist())
            first_entry = entries[0] if entries else np.nan
            first_exit = exits[0] if exits else np.nan
            second_entry = entries[1] if len(entries) >= 2 else np.nan
            reentry_delay = np.nan
            if entries and exits:
                later_exits = [frame for frame in exits if frame >= entries[0]]
                if later_exits and len(entries) >= 2:
                    reentry_delay = float(entries[1] - later_exits[0]) / fps
            zone_rows.append(
                {
                    "Target Type": "zone",
                    "Target Name": str(target_name),
                    "First Entry Frame": first_entry,
                    "First Entry Latency (s)": (float(first_entry) / fps) if pd.notna(first_entry) else np.nan,
                    "First Exit Frame": first_exit,
                    "First Exit Latency (s)": (float(first_exit) / fps) if pd.notna(first_exit) else np.nan,
                    "Second Entry Frame": second_entry,
                    "Second Entry Latency (s)": (float(second_entry) / fps) if pd.notna(second_entry) else np.nan,
                    "Re-entry Delay (s)": reentry_delay,
                }
            )
        zone_df = pd.DataFrame(zone_rows)
        zone_path = os.path.join(module_dir, f"{context.video_name}_zone_latency_metrics.csv")
        zone_df.to_csv(zone_path, index=False)
        files["zone_latency"] = zone_path

    object_event_df = _event_records_to_df(context.object_events, name_key="object_roi", source_label="object")
    object_contact_df = _build_first_membership_df(_coerce_dataframe(context.object_per_frame_df), "Object Contact Memberships", source_label="object_contact")
    object_prox_df = _build_first_membership_df(_coerce_dataframe(context.object_per_frame_df), "Object Proximity Memberships", source_label="object_proximity")
    if not object_event_df.empty or not object_contact_df.empty or not object_prox_df.empty:
        interaction_frames = (
            object_event_df[object_event_df["Event Type"] == "entry"].groupby("Target Name")["Frame"].min().to_dict()
            if not object_event_df.empty
            else {}
        )
        contact_frames = object_contact_df.groupby("Target Name")["Frame"].min().to_dict() if not object_contact_df.empty else {}
        proximity_frames = object_prox_df.groupby("Target Name")["Frame"].min().to_dict() if not object_prox_df.empty else {}
        all_names = sorted(set(interaction_frames) | set(contact_frames) | set(proximity_frames))
        object_rows = []
        for target_name in all_names:
            first_interaction = interaction_frames.get(target_name, np.nan)
            first_contact = contact_frames.get(target_name, np.nan)
            first_proximity = proximity_frames.get(target_name, np.nan)
            object_rows.append(
                {
                    "Target Type": "object",
                    "Target Name": str(target_name),
                    "First Interaction Frame": first_interaction,
                    "First Interaction Latency (s)": (float(first_interaction) / fps) if pd.notna(first_interaction) else np.nan,
                    "First Contact Frame": first_contact,
                    "First Contact Latency (s)": (float(first_contact) / fps) if pd.notna(first_contact) else np.nan,
                    "First Proximity Frame": first_proximity,
                    "First Proximity Latency (s)": (float(first_proximity) / fps) if pd.notna(first_proximity) else np.nan,
                }
            )
        object_df = pd.DataFrame(object_rows)
        object_path = os.path.join(module_dir, f"{context.video_name}_object_latency_metrics.csv")
        object_df.to_csv(object_path, index=False)
        files["object_latency"] = object_path

    return {"files": files} if files else {}


def _summarize_dwell_events(dwell_df: pd.DataFrame, *, name_col: str, source_label: str, fps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dwell_df.empty or name_col not in dwell_df.columns or "Duration (Frames)" not in dwell_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    work_df = dwell_df.copy()
    if "Track ID" not in work_df.columns:
        work_df["Track ID"] = 0
    work_df["Duration (Frames)"] = pd.to_numeric(work_df["Duration (Frames)"], errors="coerce")
    work_df["Duration (s)"] = pd.to_numeric(work_df.get("Duration (s)"), errors="coerce")
    work_df = work_df.dropna(subset=["Duration (Frames)"])
    if work_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = work_df.groupby(name_col).agg(
        Visits=("Duration (Frames)", "size"),
        Mean_Duration_Frames=("Duration (Frames)", "mean"),
        Median_Duration_Frames=("Duration (Frames)", "median"),
        P25_Duration_Frames=("Duration (Frames)", lambda s: float(np.percentile(s, 25))),
        P75_Duration_Frames=("Duration (Frames)", lambda s: float(np.percentile(s, 75))),
        Max_Duration_Frames=("Duration (Frames)", "max"),
    ).reset_index()
    summary.insert(0, "Source", source_label)
    if "Duration (s)" in work_df.columns and work_df["Duration (s)"].notna().any():
        summary["Mean_Duration_s"] = work_df.groupby(name_col)["Duration (s)"].mean().to_numpy()
        summary["Median_Duration_s"] = work_df.groupby(name_col)["Duration (s)"].median().to_numpy()
    else:
        summary["Mean_Duration_s"] = summary["Mean_Duration_Frames"] / max(fps, 1e-6)
        summary["Median_Duration_s"] = summary["Median_Duration_Frames"] / max(fps, 1e-6)

    per_track = work_df.groupby(["Track ID", name_col]).agg(
        Visits=("Duration (Frames)", "size"),
        Mean_Duration_Frames=("Duration (Frames)", "mean"),
        Median_Duration_Frames=("Duration (Frames)", "median"),
    ).reset_index()
    per_track.insert(0, "Source", source_label)
    per_track["Mean_Duration_s"] = per_track["Mean_Duration_Frames"] / max(fps, 1e-6)
    per_track["Median_Duration_s"] = per_track["Median_Duration_Frames"] / max(fps, 1e-6)
    return summary, per_track


def _module_visit_structure(context: AnalyticsModuleContext) -> Dict[str, Any]:
    module_dir = context.get_module_dir("visit_structure")
    files: Dict[str, str] = {}
    fps = max(context.fps, 1e-6)

    roi_dwell = _coerce_dataframe(context.roi_metrics.get("dwell_events") if isinstance(context.roi_metrics, dict) else None)
    if not roi_dwell.empty and "ROI Name" in roi_dwell.columns:
        zone_summary, zone_track = _summarize_dwell_events(roi_dwell, name_col="ROI Name", source_label="zone", fps=fps)
        if not zone_summary.empty:
            path = os.path.join(module_dir, f"{context.video_name}_zone_visit_structure.csv")
            zone_summary.to_csv(path, index=False)
            files["zone_summary"] = path
        if not zone_track.empty:
            path = os.path.join(module_dir, f"{context.video_name}_zone_visit_structure_by_track.csv")
            zone_track.to_csv(path, index=False)
            files["zone_by_track"] = path

    object_dwell = _coerce_dataframe(context.object_dwell_df)
    if not object_dwell.empty and "Object ROI" in object_dwell.columns:
        object_summary, object_track = _summarize_dwell_events(object_dwell, name_col="Object ROI", source_label="object", fps=fps)
        if not object_summary.empty:
            path = os.path.join(module_dir, f"{context.video_name}_object_visit_structure.csv")
            object_summary.to_csv(path, index=False)
            files["object_summary"] = path
        if not object_track.empty:
            path = os.path.join(module_dir, f"{context.video_name}_object_visit_structure_by_track.csv")
            object_track.to_csv(path, index=False)
            files["object_by_track"] = path

    return {"files": files} if files else {}


def _module_object_transition_analysis(context: AnalyticsModuleContext) -> Dict[str, Any]:
    object_event_df = _event_records_to_df(context.object_events, name_key="object_roi", source_label="object")
    if object_event_df.empty:
        return {}
    entries_df = object_event_df[object_event_df["Event Type"] == "entry"].copy()
    if entries_df.empty:
        return {}

    sequence_rows: List[Dict[str, Any]] = []
    transition_rows: List[Dict[str, Any]] = []
    for track_idx, (track_id, group) in enumerate(entries_df.groupby("Track ID", dropna=False), start=1):
        _check_stop_interval(context.stop_check, track_idx, stage="Object transition analysis", interval=8)
        group_sorted = group.sort_values("Frame", kind="mergesort")
        prev_name = None
        visit_order = 0
        for _, row in group_sorted.iterrows():
            target_name = str(row.get("Target Name", "") or "").strip()
            frame = int(row.get("Frame"))
            if not target_name:
                continue
            if target_name == prev_name:
                continue
            visit_order += 1
            sequence_rows.append(
                {
                    "Track ID": int(track_id) if pd.notna(track_id) else 0,
                    "Visit Order": visit_order,
                    "Object ROI": target_name,
                    "Entry Frame": frame,
                    "Entry Time (s)": frame / max(context.fps, 1e-6),
                }
            )
            if prev_name is not None:
                transition_rows.append(
                    {
                        "Track ID": int(track_id) if pd.notna(track_id) else 0,
                        "From Object": prev_name,
                        "To Object": target_name,
                        "Transition Count": 1,
                    }
                )
            prev_name = target_name

    if not sequence_rows:
        return {}

    module_dir = context.get_module_dir("object_transition_analysis")
    files: Dict[str, str] = {}
    sequence_df = pd.DataFrame(sequence_rows)
    sequence_path = os.path.join(module_dir, f"{context.video_name}_object_visit_sequence.csv")
    sequence_df.to_csv(sequence_path, index=False)
    files["sequence"] = sequence_path

    if transition_rows:
        transition_df = pd.DataFrame(transition_rows)
        summary_df = transition_df.groupby(["From Object", "To Object"]).agg(
            Transition_Count=("Transition Count", "sum")
        ).reset_index()
        summary_path = os.path.join(module_dir, f"{context.video_name}_object_transition_matrix.csv")
        summary_df.to_csv(summary_path, index=False)
        files["matrix"] = summary_path

        by_track_df = transition_df.groupby(["Track ID", "From Object", "To Object"]).agg(
            Transition_Count=("Transition Count", "sum")
        ).reset_index()
        by_track_path = os.path.join(module_dir, f"{context.video_name}_object_transition_by_track.csv")
        by_track_df.to_csv(by_track_path, index=False)
        files["by_track"] = by_track_path

    return {"files": files}


def _module_event_aligned_windows(context: AnalyticsModuleContext, frame_window: int | None = None) -> Dict[str, Any]:
    df = context.per_frame_df
    if df.empty or not {"frame", "track_id", "class_id"}.issubset(df.columns):
        return {}

    frame_window = frame_window if frame_window is not None else max(1, int(round(context.fps)) if context.fps else 30)
    roi_events_df = _event_records_to_df(context.roi_events, name_key="roi_name", source_label="zone")
    object_events_df = _event_records_to_df(context.object_events, name_key="object_roi", source_label="object")
    events_df = pd.concat([roi_events_df, object_events_df], ignore_index=True) if not roi_events_df.empty or not object_events_df.empty else pd.DataFrame()
    if events_df.empty:
        return {}

    work_df = df[["track_id", "frame", "class_id"]].copy()
    work_df["track_id"] = pd.to_numeric(work_df["track_id"], errors="coerce").fillna(0).astype(int)
    work_df["frame"] = pd.to_numeric(work_df["frame"], errors="coerce")
    work_df["class_id"] = pd.to_numeric(work_df["class_id"], errors="coerce")
    work_df = work_df.dropna(subset=["frame", "class_id"])
    if work_df.empty:
        return {}
    work_df["frame"] = work_df["frame"].astype(int)
    work_df["class_id"] = work_df["class_id"].astype(int)
    work_df["Behavior"] = work_df["class_id"].map(context.class_names).fillna(work_df["class_id"].astype(str))

    event_rows: List[Dict[str, Any]] = []
    for event_idx, (_, event) in enumerate(events_df.iterrows(), start=1):
        _check_stop_interval(context.stop_check, event_idx, stage="Event-aligned window analysis")
        track_id_value = pd.to_numeric(event.get("Track ID"), errors="coerce")
        track_id = int(track_id_value) if pd.notna(track_id_value) else 0
        frame = int(event.get("Frame"))
        target_name = str(event.get("Target Name", "") or "").strip()
        event_type = str(event.get("Event Type", "") or "").strip()
        source = str(event.get("Source", "") or "").strip()
        track_df = work_df[work_df["track_id"] == int(track_id)]
        if track_df.empty:
            continue
        window_df = track_df[(track_df["frame"] >= frame - frame_window) & (track_df["frame"] <= frame + frame_window)]
        if window_df.empty:
            continue
        for _, row in window_df.iterrows():
            relative_frame = int(row.get("frame")) - frame
            behavior = str(row.get("Behavior"))
            event_rows.append(
                {
                    "Source": source,
                    "Event Type": event_type,
                    "Target Name": target_name,
                    "Track ID": int(track_id),
                    "Relative Frame": relative_frame,
                    "Relative Time (s)": relative_frame / max(context.fps, 1e-6),
                    "Behavior": behavior,
                    "Frame Count": 1,
                }
            )

    if not event_rows:
        return {}

    event_df = pd.DataFrame(event_rows)
    summary_df = event_df.groupby(
        ["Source", "Event Type", "Target Name", "Relative Frame", "Relative Time (s)", "Behavior"]
    ).agg(Frame_Count=("Frame Count", "sum")).reset_index()
    totals_df = summary_df.groupby(["Source", "Event Type", "Target Name", "Relative Frame"])["Frame_Count"].transform("sum")
    summary_df["Behavior Fraction"] = np.where(totals_df > 0, summary_df["Frame_Count"] / totals_df, np.nan)

    module_dir = context.get_module_dir("event_aligned_windows")
    summary_path = os.path.join(module_dir, f"{context.video_name}_event_aligned_behavior_windows.csv")
    summary_df.to_csv(summary_path, index=False)
    return {"files": {"summary": summary_path}}


def _module_normalization_summary(context: AnalyticsModuleContext) -> Dict[str, Any]:
    module_dir = context.get_module_dir("normalization_summary")
    session_duration = float(context.session_duration_s) if np.isfinite(context.session_duration_s) else np.nan
    session_minutes = session_duration / 60.0 if np.isfinite(session_duration) and session_duration > 0.0 else np.nan
    files: Dict[str, str] = {}

    zone_summary = _coerce_dataframe(context.roi_metrics.get("entries_exits") if isinstance(context.roi_metrics, dict) else None)
    if not zone_summary.empty and "ROI Name" in zone_summary.columns:
        zone_summary = zone_summary.copy()
        zone_summary.insert(0, "Source", "zone")
        zone_summary["Time per Minute (s/min)"] = np.where(
            np.isfinite(session_minutes) & (session_minutes > 0.0),
            pd.to_numeric(zone_summary.get("Time in ROI (s)"), errors="coerce") / session_minutes,
            np.nan,
        )
        zone_summary["Entries per Minute"] = np.where(
            np.isfinite(session_minutes) & (session_minutes > 0.0),
            pd.to_numeric(zone_summary.get("Entries"), errors="coerce") / session_minutes,
            np.nan,
        )
        zone_summary["Percent of Session"] = np.where(
            np.isfinite(session_duration) & (session_duration > 0.0),
            pd.to_numeric(zone_summary.get("Time in ROI (s)"), errors="coerce") / session_duration,
            np.nan,
        )
        zone_path = os.path.join(module_dir, f"{context.video_name}_zone_normalized_summary.csv")
        zone_summary.to_csv(zone_path, index=False)
        files["zone_summary"] = zone_path

    object_summary = _coerce_dataframe(context.object_summary_df)
    if not object_summary.empty and "Object ROI" in object_summary.columns:
        object_summary = object_summary.copy()
        object_summary.insert(0, "Source", "object")
        object_summary["Interaction Time per Minute (s/min)"] = np.where(
            np.isfinite(session_minutes) & (session_minutes > 0.0),
            pd.to_numeric(object_summary.get("Time Interacting (s)"), errors="coerce") / session_minutes,
            np.nan,
        )
        object_summary["Contact Time per Minute (s/min)"] = np.where(
            np.isfinite(session_minutes) & (session_minutes > 0.0),
            pd.to_numeric(object_summary.get("Time Contact (s)"), errors="coerce") / session_minutes,
            np.nan,
        )
        object_summary["Entries per Minute"] = np.where(
            np.isfinite(session_minutes) & (session_minutes > 0.0),
            pd.to_numeric(object_summary.get("Entries"), errors="coerce") / session_minutes,
            np.nan,
        )
        object_summary["Percent of Session"] = np.where(
            np.isfinite(session_duration) & (session_duration > 0.0),
            pd.to_numeric(object_summary.get("Time Interacting (s)"), errors="coerce") / session_duration,
            np.nan,
        )
        object_path = os.path.join(module_dir, f"{context.video_name}_object_normalized_summary.csv")
        object_summary.to_csv(object_path, index=False)
        files["object_summary"] = object_path

    return {"files": files} if files else {}


OPTIONAL_ANALYTICS_MODULES: Dict[str, Dict[str, Any]] = {
    'preference_indices': {
        'label': 'Preference Indices',
        'compute': _module_preference_indices,
    },
    'latency_metrics': {
        'label': 'Latency Metrics',
        'compute': _module_latency_metrics,
    },
    'visit_structure': {
        'label': 'Visit Structure',
        'compute': _module_visit_structure,
    },
    'object_transition_analysis': {
        'label': 'Object Transition Analysis',
        'compute': _module_object_transition_analysis,
    },
    'event_aligned_windows': {
        'label': 'Event-Aligned Windows',
        'compute': _module_event_aligned_windows,
    },
    'normalization_summary': {
        'label': 'Normalization Summary',
        'compute': _module_normalization_summary,
    },
    'behavior_transitions': {
        'label': 'Behavior Transition Metrics',
        'compute': _module_behavior_transitions,
    },
    'temporal_trends': {
        'label': 'Temporal Bout Trends',
        'compute': _module_temporal_trends,
    },
    'activity_budgets': {
        'label': 'Activity Budgets',
        'compute': _module_activity_budgets,
    },
    'multi_animal_descriptors': {
        'label': 'Multi-Animal Interaction Descriptors',
        'compute': _module_multi_animal_descriptors,
    },
    'roi_time_heatmap': {
        'label': 'ROI Temporal Occupancy',
        'compute': _module_roi_time_heatmap,
    },
    'kinematic_descriptors': {
        'label': 'Kinematic Descriptors',
        'compute': _module_kinematic_descriptors,
    },
    'detection_quality': {
        'label': 'Detection Quality Metrics',
        'compute': _module_detection_quality,
    },
    'inter_bout_intervals': {
        'label': 'Inter-Bout Interval Metrics',
        'compute': _module_inter_bout_intervals,
    },
    'roi_context_windows': {
        'label': 'ROI Context Windows',
        'compute': _module_roi_context_windows,
    },
    'bout_timeline_export': {
        'label': 'Bout Timeline Export',
        'compute': _module_bout_timeline_export,
    },
}

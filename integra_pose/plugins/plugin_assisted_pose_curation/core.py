from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

DEFAULT_CLASS_NAME = "mouse"
DEFAULT_KEYPOINT_NAMES: tuple[str, ...] = (
    "Nose",
    "Left Ear",
    "Right Ear",
    "Base of Neck",
    "Left Front Paw",
    "Right Front Paw",
    "Center Spine",
    "Left Rear Paw",
    "Right Rear Paw",
    "Base of Tail",
    "Mid Tail",
    "Tail Tip",
)
DEFAULT_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (6, 7),
    (6, 8),
    (6, 9),
    (9, 10),
    (10, 11),
)

VISIBILITY_MISSING = 0
VISIBILITY_OCCLUDED = 1
VISIBILITY_VISIBLE = 2
MIN_BBOX_SIZE = 4
CURATION_MANIFEST_FILENAME = ".integrapose_assisted_pose_manifest.json"
ACTIVE_LEARNING_CSV_FILENAME = ".integrapose_active_learning_candidates.csv"
ACTIVE_LEARNING_CONTEXT_SCALE = 2.0
ACTIVE_LEARNING_WEIGHT_UNCERTAINTY = 0.45
ACTIVE_LEARNING_WEIGHT_TRANSITION = 0.25
ACTIVE_LEARNING_WEIGHT_DIVERSITY = 0.30
PROVENANCE_PENDING_REVIEW = "pending_review"
PROVENANCE_MANUAL = "manual"
PROVENANCE_ASSIST_ACCEPTED = "assist_accepted"
PROVENANCE_ASSIST_CORRECTED = "assist_corrected"
PROVENANCE_COPIED_FORWARD = "copied_forward"
KNOWN_PROVENANCE_STATES = {
    PROVENANCE_PENDING_REVIEW,
    PROVENANCE_MANUAL,
    PROVENANCE_ASSIST_ACCEPTED,
    PROVENANCE_ASSIST_CORRECTED,
    PROVENANCE_COPIED_FORWARD,
}


@dataclass
class PosePoint:
    name: str
    x: float | None = None
    y: float | None = None
    v: int = VISIBILITY_MISSING
    conf: float = 0.0

    def has_coords(self) -> bool:
        return self.x is not None and self.y is not None


@dataclass
class ActiveLearningCandidate:
    frame_index: int
    image_name: str = ""
    source_video_path: str = ""
    bbox_xyxy: tuple[int, int, int, int] | None = None
    context_bbox_xyxy: tuple[int, int, int, int] | None = None
    detection_conf: float = 0.0
    uncertainty_score: float = 0.0
    novelty_score: float = 0.0
    transition_score: float = 0.0
    diversity_score: float = 0.0
    selection_score: float = 0.0
    nearest_reference_distance: float | None = None
    nearest_reference_image: str = ""
    nearest_reference_source_video: str = ""
    nearest_selected_distance: float | None = None
    nearest_selected_image: str = ""
    nearest_selected_source_video: str = ""
    selected: bool = False
    embedding: np.ndarray | None = None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_keypoint_names(raw_names: str | Sequence[str] | None) -> list[str]:
    if raw_names is None:
        return []
    if isinstance(raw_names, str):
        parts = re.split(r"[\n\r,]+", raw_names)
        return [part.strip() for part in parts if part and part.strip()]
    out: list[str] = []
    for item in raw_names:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def sanitize_skeleton_edges(raw_edges: Iterable[Sequence[Any]] | None, keypoint_names: Sequence[str]) -> list[tuple[int, int]]:
    names = list(keypoint_names or [])

    def _resolve_endpoint(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return int(round(value))
        text = str(value).strip() if value is not None else ""
        if not text:
            return None
        if text.lstrip("+-").isdigit():
            try:
                return int(text)
            except Exception:
                return None
        if names and text in names:
            return names.index(text)
        return None

    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    if raw_edges is None:
        return normalized
    for edge in raw_edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        idx_a = _resolve_endpoint(edge[0])
        idx_b = _resolve_endpoint(edge[1])
        if idx_a is None or idx_b is None:
            continue
        if idx_a < 0 or idx_b < 0 or idx_a == idx_b:
            continue
        if idx_a >= len(names) or idx_b >= len(names):
            continue
        pair = (int(idx_a), int(idx_b))
        if pair in seen:
            continue
        seen.add(pair)
        normalized.append(pair)
    return normalized


def make_empty_pose(keypoint_names: Sequence[str]) -> list[PosePoint]:
    return [PosePoint(name=str(name).strip()) for name in keypoint_names if str(name).strip()]


def clone_pose(points: Sequence[PosePoint]) -> list[PosePoint]:
    return [replace(point) for point in points]


def pose_has_any_labels(points: Sequence[PosePoint]) -> bool:
    return any(point.has_coords() and int(point.v) != VISIBILITY_MISSING for point in points)


def derive_review_provenance(
    *,
    source_origin: str | None,
    manual_edits: bool,
    has_labels: bool,
) -> str | None:
    if not has_labels:
        return None
    origin = str(source_origin or "").strip().lower()
    if origin == "assist":
        return PROVENANCE_ASSIST_CORRECTED if manual_edits else PROVENANCE_ASSIST_ACCEPTED
    if origin == "copied_forward":
        return PROVENANCE_COPIED_FORWARD
    return PROVENANCE_MANUAL


def summarize_curation_manifest_records(
    frame_records: dict[str, Any] | None,
    *,
    expected_frame_keys: Sequence[str] | None = None,
) -> dict[str, int]:
    records = frame_records if isinstance(frame_records, dict) else {}
    counts = {
        "total_frames": 0,
        "reviewed_frames": 0,
        "pending_review_frames": 0,
        PROVENANCE_MANUAL: 0,
        PROVENANCE_ASSIST_ACCEPTED: 0,
        PROVENANCE_ASSIST_CORRECTED: 0,
        PROVENANCE_COPIED_FORWARD: 0,
        "unknown_frames": 0,
        "labeled_frames": 0,
    }
    target_keys = list(expected_frame_keys) if expected_frame_keys is not None else list(records.keys())
    counts["total_frames"] = len(target_keys)
    for key in target_keys:
        record = records.get(str(key), {})
        if not isinstance(record, dict):
            record = {}
        provenance = str(record.get("provenance") or "").strip()
        has_saved_labels = bool(record.get("has_saved_labels"))
        if has_saved_labels:
            counts["labeled_frames"] += 1
        if provenance == PROVENANCE_PENDING_REVIEW:
            counts["pending_review_frames"] += 1
            continue
        if provenance in {
            PROVENANCE_MANUAL,
            PROVENANCE_ASSIST_ACCEPTED,
            PROVENANCE_ASSIST_CORRECTED,
            PROVENANCE_COPIED_FORWARD,
        }:
            counts[provenance] += 1
            counts["reviewed_frames"] += 1
            continue
        if record:
            counts["unknown_frames"] += 1
    return counts


def pose_bbox_xyxy(
    points: Sequence[PosePoint],
    img_w: int,
    img_h: int,
    *,
    padding_ratio: float = 0.15,
    min_size: int = MIN_BBOX_SIZE,
) -> tuple[int, int, int, int] | None:
    coords = [(float(point.x), float(point.y)) for point in points if point.has_coords() and int(point.v) != VISIBILITY_MISSING]
    if not coords or img_w <= 0 or img_h <= 0:
        return None

    xs = [coord[0] for coord in coords]
    ys = [coord[1] for coord in coords]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y
    pad_x = max(float(min_size) / 2.0, span_x * max(0.0, float(padding_ratio)))
    pad_y = max(float(min_size) / 2.0, span_y * max(0.0, float(padding_ratio)))

    x1 = int(math.floor(min_x - pad_x))
    y1 = int(math.floor(min_y - pad_y))
    x2 = int(math.ceil(max_x + pad_x))
    y2 = int(math.ceil(max_y + pad_y))

    x1 = max(0, min(x1, max(img_w - 1, 0)))
    y1 = max(0, min(y1, max(img_h - 1, 0)))
    x2 = max(0, min(x2, max(img_w - 1, 0)))
    y2 = max(0, min(y2, max(img_h - 1, 0)))

    if x2 <= x1:
        cx = int(round((x1 + x2) / 2.0))
        x1 = max(0, cx - max(1, min_size // 2))
        x2 = min(max(img_w - 1, 0), x1 + max(min_size, 1))
    if y2 <= y1:
        cy = int(round((y1 + y2) / 2.0))
        y1 = max(0, cy - max(1, min_size // 2))
        y2 = min(max(img_h - 1, 0), y1 + max(min_size, 1))

    if (x2 - x1) < min_size:
        deficit = int(math.ceil((min_size - (x2 - x1)) / 2.0))
        x1 = max(0, x1 - deficit)
        x2 = min(max(img_w - 1, 0), x2 + deficit)
    if (y2 - y1) < min_size:
        deficit = int(math.ceil((min_size - (y2 - y1)) / 2.0))
        y1 = max(0, y1 - deficit)
        y2 = min(max(img_h - 1, 0), y2 + deficit)

    if x2 <= x1:
        x2 = min(max(img_w - 1, 0), x1 + max(min_size, 1))
    if y2 <= y1:
        y2 = min(max(img_h - 1, 0), y1 + max(min_size, 1))
    return int(x1), int(y1), int(x2), int(y2)


def scale_bbox_xyxy(
    bbox_xyxy: Sequence[int | float] | None,
    img_w: int,
    img_h: int,
    *,
    scale: float = ACTIVE_LEARNING_CONTEXT_SCALE,
    min_size: int = MIN_BBOX_SIZE,
) -> tuple[int, int, int, int] | None:
    if bbox_xyxy is None or img_w <= 0 or img_h <= 0:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    except Exception:
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    scaled_w = max(float(min_size), width * max(1.0, float(scale)))
    scaled_h = max(float(min_size), height * max(1.0, float(scale)))
    half_w = scaled_w / 2.0
    half_h = scaled_h / 2.0
    sx1 = int(math.floor(cx - half_w))
    sy1 = int(math.floor(cy - half_h))
    sx2 = int(math.ceil(cx + half_w))
    sy2 = int(math.ceil(cy + half_h))
    sx1 = max(0, min(sx1, max(img_w - 1, 0)))
    sy1 = max(0, min(sy1, max(img_h - 1, 0)))
    sx2 = max(0, min(sx2, max(img_w - 1, 0)))
    sy2 = max(0, min(sy2, max(img_h - 1, 0)))
    if sx2 <= sx1:
        sx2 = min(max(img_w - 1, 0), sx1 + max(min_size, 1))
    if sy2 <= sy1:
        sy2 = min(max(img_h - 1, 0), sy1 + max(min_size, 1))
    return int(sx1), int(sy1), int(sx2), int(sy2)


def crop_image_to_bbox(
    image: np.ndarray | None,
    bbox_xyxy: Sequence[int | float] | None,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    if image is None or bbox_xyxy is None or not hasattr(image, "shape"):
        return None, None
    img_h, img_w = image.shape[:2]
    bbox = scale_bbox_xyxy(bbox_xyxy, int(img_w), int(img_h), scale=1.0)
    if bbox is None:
        return None, None
    x1, y1, x2, y2 = bbox
    x2 = min(int(img_w), max(x1 + 1, x2))
    y2 = min(int(img_h), max(y1 + 1, y2))
    crop = image[y1:y2, x1:x2].copy()
    return crop, (int(x1), int(y1), int(x2), int(y2))


def serialize_pose_points(points: Sequence[PosePoint]) -> list[dict[str, Any]]:
    return [
        {
            "name": point.name,
            "x": None if point.x is None else float(point.x),
            "y": None if point.y is None else float(point.y),
            "v": int(point.v),
            "conf": float(point.conf),
        }
        for point in points
    ]


def deserialize_pose_points(records: Sequence[dict[str, Any]] | None, keypoint_names: Sequence[str]) -> list[PosePoint]:
    points = make_empty_pose(keypoint_names)
    if not isinstance(records, (list, tuple)):
        return points
    record_map: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            record_map[name] = item
    for idx, name in enumerate(keypoint_names):
        item = record_map.get(str(name))
        if item is None:
            continue
        x = item.get("x")
        y = item.get("y")
        points[idx] = PosePoint(
            name=str(name),
            x=float(x) if x is not None else None,
            y=float(y) if y is not None else None,
            v=int(item.get("v", VISIBILITY_MISSING) or VISIBILITY_MISSING),
            conf=float(item.get("conf", 0.0) or 0.0),
        )
    return points


def normalize_pose_to_bbox(points: Sequence[PosePoint], bbox_xyxy: Sequence[int | float] | None) -> list[dict[str, Any]]:
    if bbox_xyxy is None:
        return serialize_pose_points(points)
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    except Exception:
        return serialize_pose_points(points)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    normalized: list[dict[str, Any]] = []
    for point in points:
        if not point.has_coords() or int(point.v) == VISIBILITY_MISSING:
            normalized.append({"name": point.name, "x": None, "y": None, "v": int(point.v), "conf": float(point.conf)})
            continue
        normalized.append(
            {
                "name": point.name,
                "x": float(point.x - x1) / width if point.x is not None else None,
                "y": float(point.y - y1) / height if point.y is not None else None,
                "v": int(point.v),
                "conf": float(point.conf),
            }
        )
    return normalized


def pose_from_normalized_bbox_points(
    records: Sequence[dict[str, Any]] | None,
    bbox_xyxy: Sequence[int | float] | None,
    keypoint_names: Sequence[str],
) -> list[PosePoint]:
    points = make_empty_pose(keypoint_names)
    if bbox_xyxy is None or not isinstance(records, (list, tuple)):
        return points
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    except Exception:
        return points
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    record_map: dict[str, dict[str, Any]] = {}
    for item in records:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if name:
                record_map[name] = item
    for idx, name in enumerate(keypoint_names):
        item = record_map.get(str(name))
        if item is None:
            continue
        norm_x = item.get("x")
        norm_y = item.get("y")
        points[idx] = PosePoint(
            name=str(name),
            x=None if norm_x is None else (x1 + (float(norm_x) * width)),
            y=None if norm_y is None else (y1 + (float(norm_y) * height)),
            v=int(item.get("v", VISIBILITY_MISSING) or VISIBILITY_MISSING),
            conf=float(item.get("conf", 0.0) or 0.0),
        )
    return points


def blend_memory_pose_into_assist_pose(
    assist_pose: Sequence[PosePoint],
    assist_bbox_xyxy: Sequence[int | float] | None,
    *,
    memory_normalized_pose: Sequence[dict[str, Any]] | None,
    keypoint_names: Sequence[str],
    similarity: float,
    min_similarity: float = 0.88,
    fill_similarity: float = 0.93,
    max_alpha: float = 0.28,
    low_conf_threshold: float = 0.85,
    protected_low_conf_threshold: float = 0.35,
    max_shift_ratio: float = 0.18,
    protected_keypoint_names: Sequence[str] | None = ("Nose", "Center Spine", "Base of Tail"),
) -> tuple[list[PosePoint], dict[str, Any]]:
    blended = clone_pose(assist_pose)
    protected_names = {str(name).strip().lower() for name in (protected_keypoint_names or []) if str(name).strip()}
    meta = {
        "memory_similarity": float(similarity),
        "memory_refined": False,
        "memory_blend_alpha": 0.0,
        "memory_adjusted_points": 0,
        "memory_filled_points": 0,
    }
    if assist_bbox_xyxy is None or not isinstance(memory_normalized_pose, (list, tuple)):
        return blended, meta
    if float(similarity) < float(min_similarity):
        return blended, meta
    try:
        x1, y1, x2, y2 = [float(value) for value in assist_bbox_xyxy]
    except Exception:
        return blended, meta
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    alpha = max(0.0, min(float(max_alpha), ((float(similarity) - float(min_similarity)) / 0.08) * float(max_alpha)))
    if alpha <= 0.0:
        return blended, meta
    memory_pose = pose_from_normalized_bbox_points(memory_normalized_pose, assist_bbox_xyxy, keypoint_names)
    adjusted_points = 0
    filled_points = 0
    for idx, point in enumerate(blended):
        if idx >= len(memory_pose):
            continue
        memory_point = memory_pose[idx]
        if not memory_point.has_coords() or int(memory_point.v) == VISIBILITY_MISSING:
            continue
        if not point.has_coords() or int(point.v) == VISIBILITY_MISSING:
            if float(similarity) >= float(fill_similarity):
                blended[idx] = replace(memory_point, conf=max(float(memory_point.conf), float(similarity)))
                filled_points += 1
            continue
        is_protected = str(point.name or "").strip().lower() in protected_names
        raw_conf = max(0.0, min(1.0, float(point.conf or 0.0)))
        point_threshold = float(protected_low_conf_threshold if is_protected else low_conf_threshold)
        if raw_conf >= point_threshold:
            continue
        point_max_shift_px = max(
            6.0,
            max(width, height) * (0.10 if is_protected else max(0.01, float(max_shift_ratio))),
        )
        dx = float(memory_point.x or 0.0) - float(point.x or 0.0)
        dy = float(memory_point.y or 0.0) - float(point.y or 0.0)
        if math.hypot(dx, dy) > point_max_shift_px:
            continue
        conf_weight = 1.0 - min(1.0, raw_conf / max(point_threshold, 1e-6))
        if is_protected:
            local_alpha = float(alpha) * 0.35 * conf_weight
        else:
            local_alpha = float(alpha) * (0.25 + (0.75 * conf_weight))
        if local_alpha <= 0.0:
            continue
        point.x = float(point.x or 0.0) + (local_alpha * dx)
        point.y = float(point.y or 0.0) + (local_alpha * dy)
        point.conf = max(float(point.conf or 0.0), float(memory_point.conf or 0.0), float(similarity))
        adjusted_points += 1
    meta.update(
        {
            "memory_refined": bool(adjusted_points or filled_points),
            "memory_blend_alpha": float(alpha),
            "memory_adjusted_points": int(adjusted_points),
            "memory_filled_points": int(filled_points),
        }
    )
    return blended, meta


def cosine_distance(vec_a: Sequence[float] | np.ndarray | None, vec_b: Sequence[float] | np.ndarray | None) -> float:
    if vec_a is None or vec_b is None:
        return 1.0
    a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 1.0
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 1e-8 or b_norm <= 1e-8:
        return 1.0
    similarity = float(np.dot(a, b) / (a_norm * b_norm))
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def summarize_pose_uncertainty(points: Sequence[PosePoint]) -> float:
    confidences = [
        max(0.0, min(1.0, float(point.conf)))
        for point in points
        if point.has_coords() and int(point.v) != VISIBILITY_MISSING
    ]
    if not confidences:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (sum(confidences) / len(confidences))))


def compute_bbox_transition_score(
    bbox_xyxy: Sequence[int | float] | None,
    prev_bbox_xyxy: Sequence[int | float] | None,
    *,
    img_w: int,
    img_h: int,
) -> float:
    if bbox_xyxy is None or prev_bbox_xyxy is None or img_w <= 0 or img_h <= 0:
        return 0.0
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
        px1, py1, px2, py2 = [float(value) for value in prev_bbox_xyxy]
    except Exception:
        return 0.0
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0
    center_dist = math.hypot(cx - pcx, cy - pcy)
    diag = max(1.0, math.hypot(float(img_w), float(img_h)))
    motion_score = center_dist / diag
    height = max(1.0, y2 - y1)
    prev_height = max(1.0, py2 - py1)
    area = max(1.0, (x2 - x1) * (y2 - y1))
    prev_area = max(1.0, (px2 - px1) * (py2 - py1))
    height_delta = abs(height - prev_height) / max(height, prev_height, 1.0)
    area_delta = abs(area - prev_area) / max(area, prev_area, 1.0)
    return max(0.0, min(1.0, (motion_score * 2.0) + (height_delta * 0.5) + (area_delta * 0.5)))


def _min_distance_to_embeddings(
    embedding: np.ndarray | None,
    reference_embeddings: Sequence[np.ndarray],
) -> float:
    if embedding is None or not reference_embeddings:
        return 0.0
    distances = [cosine_distance(embedding, candidate) for candidate in reference_embeddings if candidate is not None]
    if not distances:
        return 0.0
    return max(0.0, min(1.0, min(distances)))


def _nearest_candidate_distance(
    candidate: ActiveLearningCandidate,
    references: Sequence[ActiveLearningCandidate],
    *,
    exclude_self: bool = False,
) -> tuple[float | None, str, str]:
    if candidate.embedding is None:
        return None, "", ""
    best_distance: float | None = None
    best_image = ""
    best_source_video = ""
    for reference in references:
        if exclude_self and reference is candidate:
            continue
        if reference.embedding is None:
            continue
        distance = cosine_distance(candidate.embedding, reference.embedding)
        if best_distance is None or float(distance) < float(best_distance):
            best_distance = float(distance)
            best_image = str(reference.image_name or "")
            best_source_video = str(reference.source_video_path or "")
    return best_distance, best_image, best_source_video


def select_active_learning_candidates(
    candidates: Sequence[ActiveLearningCandidate],
    *,
    target_count: int,
    min_frame_gap: int,
    reference_embeddings: Sequence[np.ndarray] | None = None,
) -> list[ActiveLearningCandidate]:
    candidate_pool = [candidate for candidate in candidates]
    if target_count <= 0 or not candidate_pool:
        return []
    selected: list[ActiveLearningCandidate] = []
    selected_embeddings = [np.asarray(emb, dtype=np.float32).reshape(-1) for emb in (reference_embeddings or []) if emb is not None]
    remaining = list(range(len(candidate_pool)))

    for candidate in candidate_pool:
        candidate.selected = False
        candidate.diversity_score = float(candidate.novelty_score)
        candidate.selection_score = (
            (ACTIVE_LEARNING_WEIGHT_UNCERTAINTY * float(candidate.uncertainty_score))
            + (ACTIVE_LEARNING_WEIGHT_TRANSITION * float(candidate.transition_score))
            + (ACTIVE_LEARNING_WEIGHT_DIVERSITY * float(candidate.novelty_score))
        )

    def _allowed(index: int, enforce_gap: bool) -> bool:
        if not enforce_gap:
            return True
        candidate = candidate_pool[index]
        frame_index = int(candidate.frame_index)
        source_video_path = str(candidate.source_video_path or "")
        return all(
            str(chosen.source_video_path or "") != source_video_path
            or abs(frame_index - int(chosen.frame_index)) >= max(0, int(min_frame_gap))
            for chosen in selected
        )

    def _score(index: int) -> float:
        candidate = candidate_pool[index]
        nearest_selected_distance, nearest_selected_image, nearest_selected_source_video = _nearest_candidate_distance(
            candidate,
            selected,
            exclude_self=False,
        )
        diversity = max(
            float(candidate.novelty_score),
            0.0 if nearest_selected_distance is None else float(nearest_selected_distance),
        )
        candidate.diversity_score = diversity
        candidate.nearest_selected_distance = nearest_selected_distance
        candidate.nearest_selected_image = nearest_selected_image
        candidate.nearest_selected_source_video = nearest_selected_source_video
        candidate.selection_score = (
            (ACTIVE_LEARNING_WEIGHT_UNCERTAINTY * float(candidate.uncertainty_score))
            + (ACTIVE_LEARNING_WEIGHT_TRANSITION * float(candidate.transition_score))
            + (ACTIVE_LEARNING_WEIGHT_DIVERSITY * float(diversity))
        )
        return float(candidate.selection_score)

    for enforce_gap in (True, False):
        while remaining and len(selected) < int(target_count):
            best_index: int | None = None
            best_score = -1.0
            for pool_index in remaining:
                if not _allowed(pool_index, enforce_gap):
                    continue
                score = _score(pool_index)
                if score > best_score:
                    best_index = pool_index
                    best_score = score
            if best_index is None:
                break
            candidate = candidate_pool[best_index]
            candidate.selected = True
            selected.append(candidate)
            if candidate.embedding is not None:
                selected_embeddings.append(np.asarray(candidate.embedding, dtype=np.float32).reshape(-1))
            remaining.remove(best_index)
        if len(selected) >= int(target_count):
            break
    for candidate in candidate_pool:
        nearest_selected_distance, nearest_selected_image, nearest_selected_source_video = _nearest_candidate_distance(
            candidate,
            selected,
            exclude_self=bool(candidate.selected),
        )
        candidate.nearest_selected_distance = nearest_selected_distance
        candidate.nearest_selected_image = nearest_selected_image
        candidate.nearest_selected_source_video = nearest_selected_source_video
    return sorted(selected, key=lambda candidate: (str(candidate.source_video_path or ""), int(candidate.frame_index)))


def write_active_learning_candidates_csv(
    output_path: str | Path,
    candidates: Sequence[ActiveLearningCandidate],
    *,
    run_metadata: dict[str, Any] | None = None,
) -> Path:
    header = [
        "run_id",
        "run_timestamp",
        "app_version",
        "assist_model_path",
        "assist_model_sha256",
        "score_weight_uncertainty",
        "score_weight_transition",
        "score_weight_diversity",
        "stride",
        "target_count",
        "min_frame_gap",
        "context_crop_scale",
        "assist_conf_threshold",
        "reviewed_reference_count",
        "source_video_path",
        "source_video_name",
        "frame_index",
        "image_name",
        "selected",
        "detection_conf",
        "uncertainty_score",
        "novelty_score",
        "nearest_reference_distance",
        "nearest_reference_image",
        "nearest_reference_source_video",
        "transition_score",
        "diversity_score",
        "nearest_selected_distance",
        "nearest_selected_image",
        "nearest_selected_source_video",
        "selection_score",
        "bbox_xyxy",
        "context_bbox_xyxy",
    ]
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = run_metadata if isinstance(run_metadata, dict) else {}
    write_header = True
    if out_path.is_file() and out_path.stat().st_size > 0:
        try:
            with out_path.open("r", encoding="utf-8", newline="") as existing_handle:
                existing_reader = csv.reader(existing_handle)
                existing_header = next(existing_reader, [])
            if list(existing_header) == header:
                write_header = False
            else:
                legacy_index = 1
                legacy_path = out_path.with_name(f"{out_path.stem}.legacy{out_path.suffix}")
                while legacy_path.exists():
                    legacy_index += 1
                    legacy_path = out_path.with_name(f"{out_path.stem}.legacy{legacy_index}{out_path.suffix}")
                out_path.rename(legacy_path)
                write_header = True
        except Exception:
            write_header = True
    open_mode = "a" if out_path.exists() and not write_header else "w"
    with out_path.open(open_mode, encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(header)
        for candidate in sorted(candidates, key=lambda item: (str(item.source_video_path or ""), int(item.frame_index))):
            source_video_path = str(candidate.source_video_path or "")
            source_video_name = Path(source_video_path).name if source_video_path else ""
            writer.writerow(
                [
                    str(meta.get("run_id") or ""),
                    str(meta.get("run_timestamp") or ""),
                    str(meta.get("app_version") or ""),
                    str(meta.get("assist_model_path") or ""),
                    str(meta.get("assist_model_sha256") or ""),
                    str(meta.get("score_weight_uncertainty") or ""),
                    str(meta.get("score_weight_transition") or ""),
                    str(meta.get("score_weight_diversity") or ""),
                    str(meta.get("stride") or ""),
                    str(meta.get("target_count") or ""),
                    str(meta.get("min_frame_gap") or ""),
                    str(meta.get("context_crop_scale") or ""),
                    str(meta.get("assist_conf_threshold") or ""),
                    str(meta.get("reviewed_reference_count") or ""),
                    source_video_path,
                    source_video_name,
                    int(candidate.frame_index),
                    candidate.image_name,
                    int(bool(candidate.selected)),
                    f"{float(candidate.detection_conf):.6f}",
                    f"{float(candidate.uncertainty_score):.6f}",
                    f"{float(candidate.novelty_score):.6f}",
                    "" if candidate.nearest_reference_distance is None else f"{float(candidate.nearest_reference_distance):.6f}",
                    candidate.nearest_reference_image,
                    candidate.nearest_reference_source_video,
                    f"{float(candidate.transition_score):.6f}",
                    f"{float(candidate.diversity_score):.6f}",
                    "" if candidate.nearest_selected_distance is None else f"{float(candidate.nearest_selected_distance):.6f}",
                    candidate.nearest_selected_image,
                    candidate.nearest_selected_source_video,
                    f"{float(candidate.selection_score):.6f}",
                    "" if candidate.bbox_xyxy is None else list(candidate.bbox_xyxy),
                    "" if candidate.context_bbox_xyxy is None else list(candidate.context_bbox_xyxy),
                ]
            )
    return out_path


def encode_pose_label_line(
    class_id: int,
    points: Sequence[PosePoint],
    img_w: int,
    img_h: int,
    keypoint_names: Sequence[str],
    *,
    bbox_xyxy: tuple[int, int, int, int] | None = None,
) -> str:
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Image dimensions must be positive.")
    bbox = bbox_xyxy or pose_bbox_xyxy(points, img_w, img_h)
    if bbox is None:
        raise ValueError("Cannot encode a pose label without labeled keypoints.")
    x1, y1, x2, y2 = bbox
    box_w = max(float(x2 - x1), 1.0)
    box_h = max(float(y2 - y1), 1.0)
    cx = _clip01((x1 + box_w / 2.0) / float(img_w))
    cy = _clip01((y1 + box_h / 2.0) / float(img_h))
    bw = _clip01(box_w / float(img_w))
    bh = _clip01(box_h / float(img_h))

    point_map = {point.name: point for point in points}
    flat_parts: list[str] = [str(int(class_id)), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for name in keypoint_names:
        point = point_map.get(str(name))
        if point is None or not point.has_coords() or int(point.v) == VISIBILITY_MISSING:
            flat_parts.extend(["0.000000", "0.000000", f"{float(VISIBILITY_MISSING):.6f}"])
            continue
        flat_parts.extend(
            [
                f"{_clip01(float(point.x) / float(img_w)):.6f}",
                f"{_clip01(float(point.y) / float(img_h)):.6f}",
                f"{float(int(point.v)):.6f}",
            ]
        )
    return " ".join(flat_parts)


def decode_pose_label_line(
    line: str,
    keypoint_names: Sequence[str],
    img_w: int,
    img_h: int,
) -> tuple[int, list[PosePoint], tuple[int, int, int, int]]:
    tokens = [token for token in str(line).strip().split() if token]
    expected = 5 + (len(keypoint_names) * 3)
    if len(tokens) < expected:
        raise ValueError(f"Expected at least {expected} tokens, got {len(tokens)}.")
    values = [float(token) for token in tokens]
    class_id = int(values[0])
    norm_cx, norm_cy, norm_w, norm_h = values[1:5]
    box_w = norm_w * float(img_w)
    box_h = norm_h * float(img_h)
    x1 = int(round((norm_cx * float(img_w)) - (box_w / 2.0)))
    y1 = int(round((norm_cy * float(img_h)) - (box_h / 2.0)))
    x2 = int(round(x1 + box_w))
    y2 = int(round(y1 + box_h))

    points = make_empty_pose(keypoint_names)
    start = 5
    for idx, name in enumerate(keypoint_names):
        offset = start + (idx * 3)
        norm_x = values[offset]
        norm_y = values[offset + 1]
        visibility = int(round(values[offset + 2]))
        if visibility == VISIBILITY_MISSING:
            points[idx] = PosePoint(name=str(name))
            continue
        points[idx] = PosePoint(
            name=str(name),
            x=float(norm_x) * float(img_w),
            y=float(norm_y) * float(img_h),
            v=visibility,
            conf=1.0,
        )
    return class_id, points, (x1, y1, x2, y2)


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return value.detach().cpu().numpy()
    except Exception:
        pass
    try:
        return value.cpu().numpy()
    except Exception:
        pass
    try:
        return np.asarray(value)
    except Exception:
        return None


def poses_from_ultralytics_result(
    result: Any,
    keypoint_names: Sequence[str],
    *,
    min_keypoint_conf: float = 0.0,
) -> list[tuple[int, list[PosePoint], tuple[int, int, int, int] | None, dict[str, Any]]]:
    if result is None:
        return []
    keypoints_obj = getattr(result, "keypoints", None)
    keypoint_data = _to_numpy(getattr(keypoints_obj, "data", None))
    if keypoint_data is None or keypoint_data.size == 0:
        return []
    if keypoint_data.ndim == 2:
        keypoint_data = keypoint_data[None, ...]
    if keypoint_data.ndim != 3 or keypoint_data.shape[0] <= 0:
        return []

    boxes_obj = getattr(result, "boxes", None)
    xyxy = _to_numpy(getattr(boxes_obj, "xyxy", None))
    confs = _to_numpy(getattr(boxes_obj, "conf", None))
    classes = _to_numpy(getattr(boxes_obj, "cls", None))
    conf_arr = None
    cls_arr = None
    if confs is not None and np.size(confs) > 0:
        conf_arr = np.asarray(confs, dtype=np.float32).reshape(-1)
    if classes is not None and np.size(classes) > 0:
        cls_arr = np.asarray(classes, dtype=np.float32).reshape(-1)
    if conf_arr is not None and conf_arr.size > 0:
        order = list(np.argsort(conf_arr)[::-1])
    else:
        order = list(range(int(keypoint_data.shape[0])))

    detections: list[tuple[int, list[PosePoint], tuple[int, int, int, int] | None, dict[str, Any]]] = []
    for raw_index in order:
        detection_index = max(0, min(int(raw_index), keypoint_data.shape[0] - 1))
        point_rows = keypoint_data[detection_index]
        points = make_empty_pose(keypoint_names)
        for idx, point in enumerate(points):
            if idx >= point_rows.shape[0]:
                break
            row = point_rows[idx]
            if row.shape[0] < 2:
                continue
            x = float(row[0])
            y = float(row[1])
            conf = float(row[2]) if row.shape[0] > 2 else 1.0
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(conf)):
                continue
            if conf < float(min_keypoint_conf):
                continue
            point.x = x
            point.y = y
            point.v = VISIBILITY_VISIBLE
            point.conf = conf

        bbox_xyxy: tuple[int, int, int, int] | None = None
        if xyxy is not None and np.size(xyxy) > 0:
            try:
                box_row = np.asarray(xyxy).reshape(-1, 4)[detection_index]
                bbox_xyxy = tuple(int(round(float(value))) for value in box_row)  # type: ignore[assignment]
            except Exception:
                bbox_xyxy = None
        if bbox_xyxy is None:
            orig_img = getattr(result, "orig_img", None)
            if orig_img is not None and hasattr(orig_img, "shape"):
                img_h, img_w = orig_img.shape[:2]
                bbox_xyxy = pose_bbox_xyxy(points, int(img_w), int(img_h))

        class_id = 0
        if cls_arr is not None and cls_arr.size > detection_index:
            try:
                class_id = int(round(float(cls_arr[detection_index])))
            except Exception:
                class_id = 0
        detection_conf = 0.0
        if conf_arr is not None and conf_arr.size > detection_index:
            detection_conf = float(conf_arr[detection_index])
        meta = {
            "class_id": int(class_id),
            "detection_index": detection_index,
            "detection_conf": detection_conf,
            "model_keypoint_count": int(point_rows.shape[0]),
            "detection_count": int(keypoint_data.shape[0]),
        }
        detections.append((int(class_id), points, bbox_xyxy, meta))
    return detections


def pose_from_ultralytics_result(
    result: Any,
    keypoint_names: Sequence[str],
    *,
    min_keypoint_conf: float = 0.0,
) -> tuple[list[PosePoint], tuple[int, int, int, int] | None, dict[str, Any]] | None:
    detections = poses_from_ultralytics_result(
        result,
        keypoint_names,
        min_keypoint_conf=min_keypoint_conf,
    )
    if not detections:
        return None
    _class_id, points, bbox_xyxy, meta = detections[0]
    return points, bbox_xyxy, meta


def write_dataset_yaml(
    output_path: str | Path,
    *,
    dataset_root: str | Path,
    train_images_path: str | Path,
    val_images_path: str | Path,
    class_name: str | None = None,
    class_names: Sequence[str] | None = None,
    keypoint_names: Sequence[str],
    skeleton_edges: Sequence[Sequence[int]],
) -> Path:
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    train_images = Path(train_images_path).expanduser().resolve()
    val_images = Path(val_images_path).expanduser().resolve()
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        rel_train = train_images.relative_to(dataset_root_path).as_posix()
        rel_val = val_images.relative_to(dataset_root_path).as_posix()
    except ValueError as exc:
        raise ValueError(
            "Train and validation image folders must be inside the dataset root."
        ) from exc
    normalized_class_names = [str(name).strip() for name in (class_names or []) if str(name).strip()]
    if not normalized_class_names:
        normalized_class_names = [str(class_name).strip() or DEFAULT_CLASS_NAME]
    payload = {
        "path": dataset_root_path.as_posix(),
        "train": rel_train,
        "val": rel_val,
        "kpt_shape": [len(list(keypoint_names)), 3],
        "nc": int(len(normalized_class_names)),
        "names": {idx: name for idx, name in enumerate(normalized_class_names)},
    }
    header_lines = [
        "# YOLO pose dataset configuration generated by Assisted Pose Curation",
        "#",
        "# Keypoint names (metadata):",
    ]
    header_lines.extend([f"# - {name}" for name in keypoint_names])
    if skeleton_edges:
        header_lines.extend(["#", "# Skeleton edges (0-based metadata):"])
        header_lines.extend([f"# - [{int(edge[0])}, {int(edge[1])}]" for edge in skeleton_edges if len(edge) == 2])
    header_lines.append("")
    header = "\n".join(header_lines)
    out_path.write_text(header + "\n" + yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out_path


__all__ = [
    "ACTIVE_LEARNING_CONTEXT_SCALE",
    "ACTIVE_LEARNING_CSV_FILENAME",
    "ACTIVE_LEARNING_WEIGHT_DIVERSITY",
    "ACTIVE_LEARNING_WEIGHT_TRANSITION",
    "ACTIVE_LEARNING_WEIGHT_UNCERTAINTY",
    "ActiveLearningCandidate",
    "CURATION_MANIFEST_FILENAME",
    "DEFAULT_CLASS_NAME",
    "DEFAULT_KEYPOINT_NAMES",
    "DEFAULT_SKELETON_EDGES",
    "KNOWN_PROVENANCE_STATES",
    "MIN_BBOX_SIZE",
    "PosePoint",
    "PROVENANCE_ASSIST_ACCEPTED",
    "PROVENANCE_ASSIST_CORRECTED",
    "PROVENANCE_COPIED_FORWARD",
    "PROVENANCE_MANUAL",
    "PROVENANCE_PENDING_REVIEW",
    "VISIBILITY_MISSING",
    "VISIBILITY_OCCLUDED",
    "VISIBILITY_VISIBLE",
    "clone_pose",
    "blend_memory_pose_into_assist_pose",
    "compute_bbox_transition_score",
    "cosine_distance",
    "crop_image_to_bbox",
    "deserialize_pose_points",
    "decode_pose_label_line",
    "derive_review_provenance",
    "encode_pose_label_line",
    "make_empty_pose",
    "normalize_pose_to_bbox",
    "parse_keypoint_names",
    "pose_from_normalized_bbox_points",
    "pose_bbox_xyxy",
    "pose_from_ultralytics_result",
    "pose_has_any_labels",
    "scale_bbox_xyxy",
    "select_active_learning_candidates",
    "serialize_pose_points",
    "sanitize_skeleton_edges",
    "summarize_pose_uncertainty",
    "summarize_curation_manifest_records",
    "write_active_learning_candidates_csv",
    "write_dataset_yaml",
]

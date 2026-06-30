from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np

try:  # pragma: no cover - package/script dual use
    from utils.pose_features_y import extract_relational_features
except Exception:  # pragma: no cover
    from .utils.pose_features_y import extract_relational_features

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


@dataclass
class NCropDetection:
    frame_idx: int
    animal_mask: np.ndarray  # [N]
    pose_mask: np.ndarray  # [N]
    pose_conf: np.ndarray  # [N]
    crop_conf: np.ndarray  # [N]
    track_conf: np.ndarray  # [N]
    track_age: np.ndarray  # [N]
    relation_features: np.ndarray  # [N, N, R]
    relation_pose_mask: np.ndarray  # [N, N]
    relation_present: np.ndarray  # [N, N]
    bbox_xyxy_animals: np.ndarray  # [N, 4]
    bbox_xyxy_group: Tuple[int, int, int, int]
    crop_xyxy_animals: np.ndarray  # [N, 4]
    crop_xyxy_group: Tuple[int, int, int, int]
    track_ids: np.ndarray  # [N]
    keypoints_xy_raw: np.ndarray  # [N, K, 2]
    keypoints_conf_raw: np.ndarray  # [N, K]
    keypoints_crop_norm: np.ndarray  # [N, K, 3]
    yolo_status: str
    # Optional raw BGR source frame (full-resolution). Only populated when
    # `iterate_yolo_n_crops(..., yield_orig_frame=True)`. Used by the
    # annotated-video exporter in infer_n.py — kept off by default to avoid
    # the per-frame memcpy cost in the preprocessing pipeline.
    orig_frame_bgr: Optional[np.ndarray] = None


def _check_ultralytics():
    if YOLO is None:
        raise ImportError(
            "Ultralytics YOLO is required for this operation. "
            "Install it with 'pip install ultralytics'."
        )


def normalize_device(device: str | int | None) -> str:
    """Normalize GUI/CLI device values for torch and Ultralytics."""
    if device is None:
        return "cpu"
    raw = str(device).strip()
    if not raw:
        return "cpu"
    lowered = raw.lower()
    if lowered in {"cpu", "mps"} or lowered.startswith("cuda"):
        return raw
    if lowered.isdigit():
        try:
            import torch

            if torch.cuda.is_available():
                return f"cuda:{int(lowered)}"
        except Exception:
            pass
        return "cpu"
    return raw


def load_yolo_model(weights_path: Path | str, device: str = "cpu", task: str | None = None):
    _check_ultralytics()
    device = normalize_device(device)
    w_str = str(weights_path)
    model = YOLO(w_str, task=task)
    if w_str.endswith(".pt") or w_str.endswith(".yaml") or w_str.endswith(".pth"):
        model.to(device)
    return model


def _expand_bbox(
    bbox_xyxy: np.ndarray,
    pad_ratio: float,
    image_shape: Tuple[int, int, int],
):
    x1, y1, x2, y2 = bbox_xyxy.astype(float)
    width = x2 - x1
    height = y2 - y1
    cx = x1 + width / 2
    cy = y1 + height / 2
    scale = 1.0 + pad_ratio
    half_w = width * scale / 2
    half_h = height * scale / 2
    return np.array(
        [
            max(0.0, cx - half_w),
            max(0.0, cy - half_h),
            min(float(image_shape[1]), cx + half_w),
            min(float(image_shape[0]), cy + half_h),
        ],
        dtype=np.float32,
    )


def compute_crop_xyxy(
    frame_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    expanded = _expand_bbox(bbox_xyxy, pad_ratio, frame_bgr.shape)
    x1, y1, x2, y2 = expanded.astype(int)
    h, w = frame_bgr.shape[:2]
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w, x2)))
    y2 = int(max(0, min(h, y2)))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, int(w), int(h))
    return (x1, y1, x2, y2)


def compute_fixed_square_xyxy(
    image_shape: Tuple[int, int, int],
    center_xy: Tuple[float, float],
    side_px: float,
) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    side = max(float(side_px), 2.0)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))
    # Preserve intended side in metadata as much as possible; slicing is clipped later.
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return (x1, y1, x2, y2)


def crop_coverage_conf(crop_xyxy: Tuple[int, int, int, int], image_shape: Tuple[int, int, int]) -> float:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in crop_xyxy]
    area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    if area <= 1e-9:
        return 0.0
    ix1 = max(0.0, x1)
    iy1 = max(0.0, y1)
    ix2 = min(float(w), x2)
    iy2 = min(float(h), y2)
    inside = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    return float(max(0.0, min(1.0, inside / area)))


def _extract_result_arrays(result, n_animals: int, frame_shape) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 0, 2), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    conf = boxes.conf.detach().cpu().numpy().astype(np.float32) if boxes.conf is not None else np.ones((xyxy.shape[0],), dtype=np.float32)
    ids_attr = getattr(boxes, "id", None)
    if ids_attr is not None:
        ids = ids_attr.detach().cpu().numpy().astype(np.int32)
    else:
        ids = np.arange(xyxy.shape[0], dtype=np.int32)

    h_img, w_img = frame_shape[:2]
    if xyxy.size and float(np.nanmax(xyxy)) <= 1.0:
        xyxy[:, [0, 2]] *= float(w_img)
        xyxy[:, [1, 3]] *= float(h_img)

    kxy = np.zeros((xyxy.shape[0], 0, 2), dtype=np.float32)
    kconf = np.zeros((xyxy.shape[0], 0), dtype=np.float32)
    keypoints = getattr(result, "keypoints", None)
    if keypoints is not None and getattr(keypoints, "xy", None) is not None:
        xy_obj = keypoints.xy
        xy_np = xy_obj.detach().cpu().numpy().astype(np.float32) if hasattr(xy_obj, "detach") else np.asarray(xy_obj, dtype=np.float32)
        if xy_np.size and float(np.nanmax(xy_np)) <= 1.0:
            xy_np[..., 0] *= float(w_img)
            xy_np[..., 1] *= float(h_img)
        kxy = xy_np
        conf_obj = getattr(keypoints, "conf", None)
        if conf_obj is not None:
            kconf = conf_obj.detach().cpu().numpy().astype(np.float32) if hasattr(conf_obj, "detach") else np.asarray(conf_obj, dtype=np.float32)
        else:
            kconf = np.ones(kxy.shape[:2], dtype=np.float32)

    order = np.argsort(-conf)[: int(n_animals)]
    return xyxy[order], conf[order], ids[order], kxy[order], kconf[order]


def _assign_slots(
    detections: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    n_animals: int,
    slot_track_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bboxes, conf, ids, kxy, kconf = detections
    k = int(kxy.shape[1]) if kxy.ndim == 3 else 0
    out_boxes = np.zeros((n_animals, 4), dtype=np.float32)
    out_conf = np.zeros((n_animals,), dtype=np.float32)
    out_ids = np.full((n_animals,), -1, dtype=np.int32)
    out_kxy = np.zeros((n_animals, k, 2), dtype=np.float32)
    out_kconf = np.zeros((n_animals, k), dtype=np.float32)
    present = np.zeros((n_animals,), dtype=bool)

    used = set()
    for slot, tid in enumerate(slot_track_ids.tolist()):
        if tid < 0:
            continue
        matches = np.where(ids == tid)[0]
        if matches.size:
            idx = int(matches[0])
            out_boxes[slot] = bboxes[idx]
            out_conf[slot] = conf[idx]
            out_ids[slot] = ids[idx]
            if k:
                out_kxy[slot] = kxy[idx]
                out_kconf[slot] = kconf[idx]
            present[slot] = True
            used.add(idx)

    for idx in range(bboxes.shape[0]):
        if idx in used:
            continue
        empty = np.where(~present)[0]
        if empty.size == 0:
            break
        slot = int(empty[0])
        out_boxes[slot] = bboxes[idx]
        out_conf[slot] = conf[idx]
        out_ids[slot] = ids[idx]
        if k:
            out_kxy[slot] = kxy[idx]
            out_kconf[slot] = kconf[idx]
        present[slot] = True
        slot_track_ids[slot] = ids[idx]

    return out_boxes, out_conf, out_ids, out_kxy, out_kconf, present


def iterate_yolo_n_crops(
    model,
    video_path: Path | str | int,
    *,
    n_animals: int = 2,
    crop_size: int = 224,
    animal_scale_factor: float = 4.0,
    group_scale_factor: float = 8.0,
    body_length_px: float | None = None,
    crop_strategy: str = "fixed_scale",
    pad_ratio: float = 0.2,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    device: str = "cpu",
    keep_last_box: bool = True,
    pose_conf_threshold: float = 0.3,
    fps: float = 30.0,
    batch_size: int = 25,
    timing: dict[str, float] | None = None,
    yield_orig_frame: bool = False,
) -> Generator[NCropDetection, None, None]:
    _check_ultralytics()
    device = normalize_device(device)
    n_animals = int(n_animals)
    if n_animals <= 0:
        raise ValueError("n_animals must be >= 1")
    source = str(video_path) if isinstance(video_path, Path) else video_path
    crop_strategy = str(crop_strategy or "fixed_scale").lower()

    if n_animals == 1 and crop_strategy == "bbox":
        raise ValueError("Use legacy iterate_yolo_crops for n_animals=1 + crop_strategy=bbox.")

    if hasattr(model, "track"):
        results = model.track(
            source=source,
            stream=True,
            persist=True,
            device=device,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            max_det=n_animals,
            batch=int(batch_size),
            verbose=False,
        )
    else:
        results = model.predict(
            source=source,
            stream=True,
            device=device,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            max_det=n_animals,
            batch=int(batch_size),
            verbose=False,
        )

    slot_track_ids = np.full((n_animals,), -1, dtype=np.int32)
    track_first_seen: dict[int, int] = {}
    last_boxes = None
    last_kxy = None
    last_kconf = None
    last_present = None
    last_track_ids = None
    last_conf = None
    prev_boxes = None
    prev_kxy = None

    for frame_idx, result in enumerate(results):
        if timing is not None:
            timing["yolo_frames"] = timing.get("yolo_frames", 0.0) + 1.0
        frame_bgr = result.orig_img
        detections = _extract_result_arrays(result, n_animals, frame_bgr.shape)
        boxes, conf, track_ids, kxy, kconf, present = _assign_slots(detections, n_animals, slot_track_ids)
        crop_present = present.copy()

        status = "detected" if bool(present.any()) else "missed"
        if not present.any() and keep_last_box and last_boxes is not None:
            boxes = last_boxes.copy()
            kxy = last_kxy.copy() if last_kxy is not None else kxy
            kconf = last_kconf.copy() if last_kconf is not None else kconf
            crop_present = last_present.copy() if last_present is not None else crop_present
            # Reused geometry keeps crops stable, but it is not a fresh
            # detection. Keep the hard animal mask false so model pooling and
            # relation masks do not treat stale boxes as present animals.
            present = np.zeros_like(crop_present, dtype=bool)
            track_ids = last_track_ids.copy() if last_track_ids is not None else track_ids
            conf = last_conf.copy() if last_conf is not None else conf
            status = "reused"
        elif present.any():
            crop_present = present.copy()
            last_boxes = boxes.copy()
            last_kxy = kxy.copy()
            last_kconf = kconf.copy()
            last_present = present.copy()
            last_track_ids = track_ids.copy()
            last_conf = conf.copy()

        crop_started = time.perf_counter()
        centers = np.array([(bb[0] + bb[2], bb[1] + bb[3]) for bb in boxes], dtype=np.float32) * 0.5
        valid_centers = centers[crop_present]
        if valid_centers.size:
            group_center = valid_centers.mean(axis=0)
        else:
            h, w = frame_bgr.shape[:2]
            group_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        body_len = float(body_length_px or 0.0)
        if body_len <= 0:
            valid_w = boxes[crop_present, 2] - boxes[crop_present, 0] if crop_present.any() else np.array([crop_size], dtype=np.float32)
            valid_h = boxes[crop_present, 3] - boxes[crop_present, 1] if crop_present.any() else np.array([crop_size], dtype=np.float32)
            body_len = float(np.nanmedian(np.maximum(valid_w, valid_h))) if valid_w.size else float(crop_size)
            body_len = max(body_len, 1.0)

        if crop_strategy == "bbox":
            union = np.array(
                [
                    float(np.min(boxes[crop_present, 0])),
                    float(np.min(boxes[crop_present, 1])),
                    float(np.max(boxes[crop_present, 2])),
                    float(np.max(boxes[crop_present, 3])),
                ],
                dtype=np.float32,
            ) if crop_present.any() else np.array([0, 0, frame_bgr.shape[1], frame_bgr.shape[0]], dtype=np.float32)
            group_bbox_xyxy = tuple(int(v) for v in union)
            group_xyxy = compute_crop_xyxy(frame_bgr, union, pad_ratio)
            animal_xyxys = np.array([compute_crop_xyxy(frame_bgr, bb, pad_ratio) for bb in boxes], dtype=np.int32)
        else:
            group_xyxy = compute_fixed_square_xyxy(frame_bgr.shape, (float(group_center[0]), float(group_center[1])), body_len * float(group_scale_factor))
            # In fixed_scale mode there is no underlying "group bbox" distinct
            # from the crop — the crop *is* the geometry of interest. Mirror
            # the crop window into bbox_xyxy_group for downstream consistency.
            group_bbox_xyxy = tuple(int(v) for v in group_xyxy)
            animal_xyxys = np.array(
                [
                    compute_fixed_square_xyxy(frame_bgr.shape, (float(centers[i, 0]), float(centers[i, 1])), body_len * float(animal_scale_factor))
                    for i in range(n_animals)
                ],
                dtype=np.int32,
            )

        pose_conf = np.zeros((n_animals,), dtype=np.float32)
        if kconf.size:
            pose_conf = np.nan_to_num(kconf.mean(axis=1), nan=0.0).astype(np.float32)
        pose_conf = np.where(present, pose_conf, 0.0).astype(np.float32)
        pose_mask = (pose_conf >= float(pose_conf_threshold)) & present
        crop_conf = np.array([crop_coverage_conf(tuple(xyxy), frame_bgr.shape) for xyxy in animal_xyxys], dtype=np.float32)
        track_conf = np.where(present, conf, 0.0).astype(np.float32)
        track_age = np.zeros((n_animals,), dtype=np.float32)
        for i, tid in enumerate(track_ids.tolist()):
            if tid >= 0 and present[i]:
                track_first_seen.setdefault(int(tid), int(frame_idx))
                track_age[i] = min(float(frame_idx - track_first_seen[int(tid)] + 1) / 30.0, 1.0)

        keypoints_crop_norm = np.zeros((n_animals, kxy.shape[1], 3), dtype=np.float32) if kxy.ndim == 3 else np.zeros((n_animals, 0, 3), dtype=np.float32)
        if kxy.ndim == 3 and kxy.shape[1] > 0:
            for i in range(n_animals):
                x1, y1, x2, y2 = [float(v) for v in animal_xyxys[i]]
                cw = max(x2 - x1, 1.0)
                ch = max(y2 - y1, 1.0)
                keypoints_crop_norm[i, :, 0] = (kxy[i, :, 0] - x1) / cw
                keypoints_crop_norm[i, :, 1] = (kxy[i, :, 1] - y1) / ch
                # kconf shape is [n_animals, K] when YOLO returns per-keypoint
                # confidence (the expected case for YOLOv8/v11-pose).  Guard on
                # ndim==2 so a scalar per-animal confidence from an unusual YOLO
                # version cannot silently broadcast to all K keypoint slots.
                keypoints_crop_norm[i, :, 2] = kconf[i] if kconf.ndim == 2 else 0.0

        rel, rel_pose_mask, rel_present = extract_relational_features(
            kxy,
            boxes,
            present,
            pose_mask,
            body_len,
            prev_keypoints_raw=prev_kxy,
            prev_bboxes=prev_boxes,
            fps=fps,
            pose_conf_threshold=float(pose_conf_threshold),
            keypoints_conf_raw=kconf if kconf.size else None,
        )
        prev_boxes = boxes.copy()
        prev_kxy = kxy.copy()

        if timing is not None:
            timing["crop_ms"] = timing.get("crop_ms", 0.0) + ((time.perf_counter() - crop_started) * 1000.0)
            timing["crop_frames"] = timing.get("crop_frames", 0.0) + 1.0

        yield NCropDetection(
            frame_idx=frame_idx,
            animal_mask=present.astype(bool),
            pose_mask=pose_mask.astype(bool),
            pose_conf=pose_conf,
            crop_conf=crop_conf,
            track_conf=track_conf,
            track_age=track_age,
            relation_features=rel,
            relation_pose_mask=rel_pose_mask,
            relation_present=rel_present,
            bbox_xyxy_animals=boxes.astype(np.int32),
            bbox_xyxy_group=group_bbox_xyxy,
            crop_xyxy_animals=animal_xyxys.astype(np.int32),
            crop_xyxy_group=tuple(int(v) for v in group_xyxy),
            track_ids=track_ids.astype(np.int32),
            keypoints_xy_raw=kxy.astype(np.float32),
            keypoints_conf_raw=kconf.astype(np.float32),
            keypoints_crop_norm=keypoints_crop_norm.astype(np.float32),
            yolo_status=status,
            orig_frame_bgr=(frame_bgr.copy() if yield_orig_frame else None),
        )

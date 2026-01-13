from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled at call site
    YOLO = None


@dataclass
class CropConfig:
    crop_size: int = 160
    pad_ratio: float = 0.2
    keep_last_box: bool = True


@dataclass
class CropDetection:
    frame_idx: int
    crop_rgb: Optional[np.ndarray]
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int] | None
    yolo_status: str  # detected | reused | missed
    keypoints_xy: Tuple[Tuple[float, float], ...] | None = None
    keypoints_conf: Tuple[float, ...] | None = None


def _check_ultralytics():
    if YOLO is None:
        raise ImportError(
            "Ultralytics YOLO is required for this operation. "
            "Install it with 'pip install ultralytics'."
        )


def load_yolo_model(weights_path: Path | str, device: str = "cpu"):
    _check_ultralytics()
    model = YOLO(str(weights_path))
    model.to(device)
    return model


def _expand_bbox(
    bbox_xyxy: np.ndarray, pad_ratio: float, image_shape: Tuple[int, int, int]
):
    x1, y1, x2, y2 = bbox_xyxy.astype(float)
    width = x2 - x1
    height = y2 - y1
    cx = x1 + width / 2
    cy = y1 + height / 2
    scale = 1.0 + pad_ratio
    half_w = width * scale / 2
    half_h = height * scale / 2

    new_x1 = max(0.0, cx - half_w)
    new_y1 = max(0.0, cy - half_h)
    new_x2 = min(image_shape[1] - 1.0, cx + half_w)
    new_y2 = min(image_shape[0] - 1.0, cy + half_h)
    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)


def crop_and_resize(
    frame_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    pad_ratio: float,
    crop_size: int,
) -> np.ndarray:
    expanded = _expand_bbox(bbox_xyxy, pad_ratio, frame_bgr.shape)
    x1, y1, x2, y2 = expanded.astype(int)
    sub_img = frame_bgr[y1:y2, x1:x2]
    if sub_img.size == 0:
        sub_img = frame_bgr
    resized = cv2.resize(sub_img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb


def iterate_yolo_crops(
    model,
    video_path: Path | str | int,
    crop_size: int,
    pad_ratio: float,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    device: str = "cpu",
    keep_last_box: bool = True,
    max_det: int = 1,
    nms: bool = True,
    yield_misses: bool = False,
    timing: dict[str, float] | None = None,
) -> Generator[CropDetection, None, None]:
    """
    Run YOLO on a video and yield per-frame padded crops.
    """
    _check_ultralytics()
    source = video_path
    if isinstance(video_path, Path):
        source = str(video_path)
    results = model.predict(
        source=source,
        stream=True,
        device=device,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        max_det=1,  # single-animal only (locked)
        nms=True,  # single-animal only (locked)
        verbose=False,
    )

    last_bbox = None
    last_conf = 0.0
    last_keypoints_xy = None
    last_keypoints_conf = None
    for frame_idx, result in enumerate(results):
        if timing is not None:
            timing["yolo_frames"] = timing.get("yolo_frames", 0.0) + 1.0
            speed = getattr(result, "speed", None)
            if isinstance(speed, dict):
                timing["yolo_preprocess_ms"] = timing.get("yolo_preprocess_ms", 0.0) + float(
                    speed.get("preprocess", 0.0) or 0.0
                )
                timing["yolo_inference_ms"] = timing.get("yolo_inference_ms", 0.0) + float(
                    speed.get("inference", 0.0) or 0.0
                )
                timing["yolo_postprocess_ms"] = timing.get("yolo_postprocess_ms", 0.0) + float(
                    speed.get("postprocess", 0.0) or 0.0
                )
        frame_bgr = result.orig_img
        boxes = result.boxes
        best_bbox = None
        best_conf = None
        best_keypoints_xy = None
        best_keypoints_conf = None
        status = "missed"
        if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > 0:
            confs = boxes.conf.cpu().numpy()
            idx = int(confs.argmax())
            best_bbox = boxes.xyxy[idx].cpu().numpy()
            best_conf = float(confs[idx])
            try:
                keypoints = getattr(result, "keypoints", None)
                if keypoints is not None:
                    xy = getattr(keypoints, "xy", None)
                    if xy is not None and len(xy) > idx:
                        xy_i = xy[idx]
                        if hasattr(xy_i, "detach"):
                            xy_i = xy_i.detach()
                        if hasattr(xy_i, "cpu"):
                            xy_i = xy_i.cpu()
                        xy_np = xy_i.numpy() if hasattr(xy_i, "numpy") else np.asarray(xy_i)
                        best_keypoints_xy = tuple(
                            (float(x), float(y)) for x, y in xy_np.tolist()
                        )
                    conf = getattr(keypoints, "conf", None)
                    if conf is not None and len(conf) > idx:
                        conf_i = conf[idx]
                        if hasattr(conf_i, "detach"):
                            conf_i = conf_i.detach()
                        if hasattr(conf_i, "cpu"):
                            conf_i = conf_i.cpu()
                        conf_np = (
                            conf_i.numpy()
                            if hasattr(conf_i, "numpy")
                            else np.asarray(conf_i)
                        )
                        best_keypoints_conf = tuple(float(c) for c in conf_np.tolist())
            except Exception:
                best_keypoints_xy = None
                best_keypoints_conf = None
            last_bbox = best_bbox
            last_conf = best_conf
            last_keypoints_xy = best_keypoints_xy
            last_keypoints_conf = best_keypoints_conf
            status = "detected"
        elif keep_last_box and last_bbox is not None:
            best_bbox = last_bbox
            best_conf = last_conf
            best_keypoints_xy = last_keypoints_xy
            best_keypoints_conf = last_keypoints_conf
            status = "reused"

        if best_bbox is None:
            if not yield_misses:
                continue
            yield CropDetection(
                frame_idx=frame_idx,
                crop_rgb=None,
                confidence=0.0,
                bbox_xyxy=None,
                yolo_status="missed",
                keypoints_xy=None,
                keypoints_conf=None,
            )
            continue

        crop_started = time.perf_counter()
        crop_rgb = crop_and_resize(frame_bgr, best_bbox, pad_ratio, crop_size)
        if timing is not None:
            timing["crop_ms"] = timing.get("crop_ms", 0.0) + (
                (time.perf_counter() - crop_started) * 1000.0
            )
            timing["crop_frames"] = timing.get("crop_frames", 0.0) + 1.0
        yield CropDetection(
            frame_idx=frame_idx,
            crop_rgb=crop_rgb,
            confidence=best_conf if best_conf is not None else 0.0,
            bbox_xyxy=tuple(int(v) for v in best_bbox),
            yolo_status=status,
            keypoints_xy=best_keypoints_xy,
            keypoints_conf=best_keypoints_conf,
        )

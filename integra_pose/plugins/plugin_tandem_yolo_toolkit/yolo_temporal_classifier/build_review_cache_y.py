"""Build per-frame pose overlay caches for TandemYTC annotation review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:  # pragma: no cover - CLI/package dual use
    from cropping_y import iterate_yolo_n_crops, load_yolo_model
except Exception:  # pragma: no cover
    from .cropping_y import iterate_yolo_n_crops, load_yolo_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TandemYTC review skeleton overlay cache.")
    p.add_argument("--video", required=True, help="Source video path.")
    p.add_argument("--yolo_weights", required=True, help="YOLO-pose .pt checkpoint.")
    p.add_argument("--output", required=True, help="Output .npz cache path.")
    p.add_argument("--n_animals", type=int, default=2)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--pose_conf_threshold", type=float, default=0.3)
    p.add_argument("--keep_last_box", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = load_yolo_model(args.yolo_weights, device=args.device, task="pose")
    frame_indices: list[int] = []
    bboxes: list[np.ndarray] = []
    keypoints_xy: list[np.ndarray] = []
    keypoints_conf: list[np.ndarray] = []
    track_ids: list[np.ndarray] = []
    pose_mask: list[np.ndarray] = []
    animal_mask: list[np.ndarray] = []
    statuses: list[str] = []
    width = 0
    height = 0

    for det in iterate_yolo_n_crops(
        model,
        args.video,
        n_animals=args.n_animals,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        keep_last_box=bool(args.keep_last_box),
        pose_conf_threshold=args.pose_conf_threshold,
        fps=args.fps,
        batch_size=max(1, int(args.batch)),
        yield_orig_frame=True,
    ):
        frame_indices.append(int(det.frame_idx))
        bboxes.append(det.bbox_xyxy_animals.astype(np.float32, copy=False))
        keypoints_xy.append(det.keypoints_xy_raw.astype(np.float32, copy=False))
        keypoints_conf.append(det.keypoints_conf_raw.astype(np.float32, copy=False))
        track_ids.append(det.track_ids.astype(np.int32, copy=False))
        pose_mask.append(det.pose_mask.astype(bool, copy=False))
        animal_mask.append(det.animal_mask.astype(bool, copy=False))
        statuses.append(str(det.yolo_status))
        if det.orig_frame_bgr is not None and width <= 0:
            height, width = det.orig_frame_bgr.shape[:2]
        if len(frame_indices) % 500 == 0:
            print(f"[review-cache] processed {len(frame_indices)} frames", flush=True)

    if not frame_indices:
        raise SystemExit("No frames were processed; review cache was not written.")

    status_vocab = {status: idx for idx, status in enumerate(sorted(set(statuses)))}
    status_ids = np.asarray([status_vocab[s] for s in statuses], dtype=np.int16)
    metadata = {
        "schema_version": "tandemytc-review-cache-v1",
        "video": str(args.video),
        "yolo_weights": str(args.yolo_weights),
        "n_animals": int(args.n_animals),
        "width": int(width),
        "height": int(height),
        "status_vocab": status_vocab,
    }
    np.savez_compressed(
        output,
        frame_idx=np.asarray(frame_indices, dtype=np.int64),
        bbox_xyxy=np.stack(bboxes).astype(np.float32, copy=False),
        keypoints_xy=np.stack(keypoints_xy).astype(np.float32, copy=False),
        keypoints_conf=np.stack(keypoints_conf).astype(np.float32, copy=False),
        track_ids=np.stack(track_ids).astype(np.int32, copy=False),
        pose_mask=np.stack(pose_mask).astype(bool, copy=False),
        animal_mask=np.stack(animal_mask).astype(bool, copy=False),
        status_ids=status_ids,
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(f"[review-cache] wrote {output} ({len(frame_indices)} frames)", flush=True)


if __name__ == "__main__":
    main()

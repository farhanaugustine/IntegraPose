from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import cv2
import torch
from torch.nn import functional as F

if __package__ is None or __package__ == "":  # pragma: no cover - CLI entry
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.append(str(SCRIPT_DIR))
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from cropping import (
        CropDetection,
        iterate_yolo_crops,
        load_yolo_model,
    )
    from data import build_frame_transform
    from model import BehaviorSequenceClassifier
else:
    from .cropping import CropDetection, iterate_yolo_crops, load_yolo_model
    from .data import build_frame_transform
    from .model import BehaviorSequenceClassifier

from integra_pose.utils.safe_model_io import load_torch_artifact


@dataclass
class WindowPrediction:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    class_idx: int | None
    class_name: str
    confidence: float


class JsonStreamWriter:
    def __init__(self, path: Path | None):
        self.path = path
        self.file = None
        self.first = True
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("w", encoding="utf-8")
            self.file.write("[\n")

    def write(self, prediction: WindowPrediction):
        if not self.file:
            return
        if not self.first:
            self.file.write(",\n")
        json.dump(prediction.__dict__, self.file)
        self.file.flush()
        self.first = False

    def close(self):
        if not self.file:
            return
        if self.first:
            self.file.write("]\n")
        else:
            self.file.write("\n]\n")
        self.file.close()
        self.file = None


class CsvMergeWriter:
    def __init__(self, path: Path, keep_unknown: bool):
        self.path = path
        self.keep_unknown = keep_unknown
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            [
                "start_time",
                "end_time",
                "start_frame",
                "end_frame",
                "label",
                "confidence",
            ]
        )
        self.current: WindowPrediction | None = None
        self.count = 0
        self.conf_sum = 0.0

    def _flush_current(self):
        if self.current is None:
            return
        self.current.confidence = self.conf_sum / max(self.count, 1)
        self.writer.writerow(
            [
                f"{self.current.start_time:.3f}",
                f"{self.current.end_time:.3f}",
                self.current.start_frame,
                self.current.end_frame,
                self.current.class_name,
                f"{self.current.confidence:.3f}",
            ]
        )
        self.file.flush()
        self.current = None
        self.count = 0
        self.conf_sum = 0.0

    def add(self, prediction: WindowPrediction):
        if not self.keep_unknown and prediction.class_idx is None:
            self._flush_current()
            return

        if (
            self.current is not None
            and self.current.class_idx == prediction.class_idx
            and self.current.class_name == prediction.class_name
        ):
            self.current.end_frame = prediction.end_frame
            self.current.end_time = prediction.end_time
            self.count += 1
            self.conf_sum += prediction.confidence
        else:
            self._flush_current()
            self.current = replace(prediction)
            self.count = 1
            self.conf_sum = prediction.confidence

    def close(self):
        self._flush_current()
        self.file.close()


def resolve_output_path(
    path: Path | None,
    video_path: Path,
    suffix: str,
) -> Path | None:
    if path is None:
        return None
    if path.exists() and path.is_dir():
        return path / f"{video_path.stem}{suffix}"
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained behavior LSTM model on a long video."
    )
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Path for the merged behavior annotations (.csv).",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional JSON dump of per-window predictions.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Number of frames per inference window (defaults to training num_frames).",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=None,
        help="Stride (in frames) between inference windows. Defaults to window_size//2.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Minimum probability to accept a prediction. Otherwise labeled as 'Unknown'.",
    )
    parser.add_argument(
        "--keep_unknown",
        action="store_true",
        help="Include unknown segments in the merged CSV.",
    )
    parser.add_argument(
        "--skip_tail",
        action="store_true",
        help="Skip the final partial window instead of forcing an evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on.",
    )
    parser.add_argument("--yolo_weights", type=Path, required=True)
    parser.add_argument(
        "--crop_size",
        type=int,
        default=None,
        help="Resize crop size (defaults to training frame_size).",
    )
    parser.add_argument("--pad_ratio", type=float, default=0.2)
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--yolo_iou", type=float, default=0.45)
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument(
        "--yolo_max_det",
        type=int,
        default=1,
        help="Maximum detections per frame returned by YOLO.",
    )
    parser.add_argument("--keep_last_box", action="store_true")
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=50,
        help="Print progress every N processed windows (0 to disable).",
    )
    return parser.parse_args()


def load_checkpoint(model_path: Path, device: torch.device):
    checkpoint = load_torch_artifact(model_path, map_location=device, description="behavior checkpoint")
    hparams = checkpoint.get("hparams", {})
    idx_to_class = checkpoint.get("idx_to_class") or hparams.get("idx_to_class")
    if idx_to_class is None:
        raise ValueError("Checkpoint is missing idx_to_class metadata.")

    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    num_classes = len(class_names)

    model = BehaviorSequenceClassifier(
        num_classes=num_classes,
        backbone=hparams.get("backbone", "mobilenet_v3_small"),
        pretrained_backbone=False,
        train_backbone=False,
        hidden_dim=int(hparams.get("hidden_dim", 256)),
        num_layers=int(hparams.get("num_layers", 1)),
        dropout=float(hparams.get("dropout", 0.2)),
        bidirectional=bool(hparams.get("bidirectional", False)),
        sequence_model=hparams.get("sequence_model", "lstm"),
        temporal_attention_layers=int(
            hparams.get("temporal_attention_layers", 0)
        ),
        attention_heads=int(hparams.get("attention_heads", 4)),
        use_attention_pooling=bool(hparams.get("attention_pool", False)),
        use_feature_se=bool(hparams.get("feature_se", False)),
        slowfast_alpha=int(hparams.get("slowfast_alpha", 4)),
        slowfast_fusion_ratio=float(hparams.get("slowfast_fusion_ratio", 0.25)),
        slowfast_base_channels=int(hparams.get("slowfast_base_channels", 48)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, class_names, hparams


def probe_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 1e-3:
        return 30.0
    return float(fps)


def probe_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if total is None or total <= 0:
        return 0
    return int(total)


def predict_windows(
    model: BehaviorSequenceClassifier,
    class_names: List[str],
    video_path: Path,
    window_size: int,
    stride: int,
    batch_size: int,
    prob_threshold: float,
    device: torch.device,
    yolo_model,
    crop_size: int,
    pad_ratio: float,
    yolo_conf: float,
    yolo_iou: float,
    yolo_imgsz: int,
    yolo_max_det: int,
    keep_last_box: bool,
    skip_tail: bool,
) -> List[WindowPrediction]:
    transform = build_frame_transform(crop_size, augment=False)
    fps = probe_fps(video_path)
    batch_frames: List[torch.Tensor] = []
    batch_meta: List[Dict] = []
    buffer: deque[CropDetection] = deque()

    def flush_batch():
        if not batch_frames:
            return []
        clips = torch.stack(batch_frames, dim=0).to(device)
        with torch.no_grad():
            logits = model(clips)
            probs = F.softmax(logits, dim=1)
        top_p, top_idx = probs.max(dim=1)
        results = []
        for meta, prob, idx in zip(batch_meta, top_p.tolist(), top_idx.tolist()):
            if prob < prob_threshold:
                class_idx = None
                class_name = "Unknown"
            else:
                class_idx = idx
                class_name = class_names[idx]
            results.append(
                WindowPrediction(
                    start_frame=meta["start_frame"],
                    end_frame=meta["end_frame"],
                    start_time=meta["start_time"],
                    end_time=meta["end_time"],
                    class_idx=class_idx,
                    class_name=class_name,
                    confidence=float(prob),
                )
            )
        batch_frames.clear()
        batch_meta.clear()
        return results

    def enqueue_window(entries: List[CropDetection]):
        processed = [transform(det.crop_rgb) for det in entries]
        clip_tensor = torch.stack(processed, dim=0)
        start_frame = entries[0].frame_idx
        end_frame = entries[-1].frame_idx
        batch_frames.append(clip_tensor)
        batch_meta.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_frame / fps,
                "end_time": end_frame / fps,
            }
        )
        if len(batch_frames) >= batch_size:
            for item in flush_batch():
                yield item

    stride = max(1, stride)
    if device.type == "cuda":
        yolo_device = (
            f"cuda:{device.index}" if device.index is not None else "cuda"
        )
    else:
        yolo_device = device.type

    crop_generator = iterate_yolo_crops(
        model=yolo_model,
        video_path=video_path,
        crop_size=crop_size,
        pad_ratio=pad_ratio,
        conf_threshold=yolo_conf,
        iou_threshold=yolo_iou,
        imgsz=yolo_imgsz,
        max_det=yolo_max_det,
        device=yolo_device,
        keep_last_box=keep_last_box,
    )

    for detection in crop_generator:
        buffer.append(detection)
        while len(buffer) >= window_size:
            entries = list(buffer)[:window_size]
            yield from enqueue_window(entries)
            for _ in range(stride):
                if buffer:
                    buffer.popleft()

    if not skip_tail and buffer:
        tail_entries = list(buffer)
        if tail_entries:
            while len(tail_entries) < window_size:
                tail_entries.append(tail_entries[-1])
            yield from enqueue_window(tail_entries[:window_size])

    yield from flush_batch()


def main():
    args = parse_args()
    if args.yolo_max_det <= 0:
        raise SystemExit("yolo_max_det must be a positive integer.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, class_names, hparams = load_checkpoint(args.model_path, device)
    default_num_frames = int(hparams.get("num_frames", 16))
    default_frame_size = int(hparams.get("frame_size", 160))

    window_size = args.window_size or default_num_frames
    stride = args.window_stride or max(1, window_size // 2)
    crop_size = args.crop_size or default_frame_size

    yolo_device = (
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    yolo_model = load_yolo_model(args.yolo_weights, device=yolo_device)

    output_csv = (
        resolve_output_path(args.output_csv, args.video_path, ".behavior.csv")
        if args.output_csv
        else args.video_path.with_suffix(".behavior.csv")
    )
    output_json = resolve_output_path(
        args.output_json, args.video_path, ".windows.json"
    )

    total_frames = probe_frame_count(args.video_path)
    json_stream = JsonStreamWriter(output_json)
    csv_stream = CsvMergeWriter(output_csv, keep_unknown=args.keep_unknown)
    windows_processed = 0

    try:
        for prediction in predict_windows(
            model=model,
            class_names=class_names,
            video_path=args.video_path,
            window_size=window_size,
            stride=stride,
            batch_size=args.batch_size,
            prob_threshold=args.prob_threshold,
            device=device,
            yolo_model=yolo_model,
            crop_size=crop_size,
            pad_ratio=args.pad_ratio,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_imgsz=args.yolo_imgsz,
            yolo_max_det=args.yolo_max_det,
            keep_last_box=args.keep_last_box,
            skip_tail=args.skip_tail,
        ):
            windows_processed += 1
            json_stream.write(prediction)
            csv_stream.add(prediction)

            if (
                args.progress_interval > 0
                and windows_processed % args.progress_interval == 0
            ):
                if total_frames > 0:
                    percent = prediction.end_frame / max(total_frames - 1, 1) * 100.0
                    frame_info = (
                        f"{prediction.end_frame}/{total_frames} "
                        f"({percent:.1f}%)"
                    )
                else:
                    frame_info = f"{prediction.end_frame}"
                print(
                    f"[Inference] windows={windows_processed} frame={frame_info}",
                    flush=True,
                )
    finally:
        csv_stream.close()
        json_stream.close()

    print(f"Merged annotations saved to: {output_csv}")
    if output_json:
        print(f"Per-window predictions saved to: {output_json}")
    print(f"Total windows processed: {windows_processed}")


if __name__ == "__main__":
    main()

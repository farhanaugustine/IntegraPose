from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

if __package__ is None or __package__ == "":  # pragma: no cover - CLI entry
    from cropping import CropDetection, iterate_yolo_crops, load_yolo_model
    from data import (
        SequenceSample,
        save_sequence_manifest,
        stratified_split,
        stratified_split_by_source_video,
    )
else:
    from .cropping import CropDetection, iterate_yolo_crops, load_yolo_model
    from .data import (
        SequenceSample,
        save_sequence_manifest,
        stratified_split,
        stratified_split_by_source_video,
    )

ALLOWED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def parse_args():
    def _parse_bool(value: str | bool) -> bool:
        if isinstance(value, bool):
            return value
        truthy = {"1", "true", "t", "yes", "y", "on"}
        falsy = {"0", "false", "f", "no", "n", "off"}
        normalized = value.strip().lower()
        if normalized in truthy:
            return True
        if normalized in falsy:
            return False
        raise argparse.ArgumentTypeError(
            f"Expected a boolean value (true/false/1/0), got: {value!r}"
        )

    parser = argparse.ArgumentParser(
        description="Run YOLO on behavior clips and build cropped sequences."
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--yolo_weights", type=Path, required=True)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Destination directory for QA crops, NPZ tensors, and the manifest.",
    )
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--window_stride", type=int, default=15)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--pad_ratio", type=float, default=0.2)
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--yolo_iou", type=float, default=0.45)
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--keep_last_box", action="store_true")
    parser.add_argument(
        "--save_frame_crops",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Persist individual frame crops for QA (option A).",
    )
    parser.add_argument(
        "--save_sequence_npz",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Persist NPZ tensors per window for fast training (option B).",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="window",
        choices=["window", "video"],
        help=(
            "How to split train/val/test. "
            "window: splits individual windows (can leak overlapping windows across splits). "
            "video: keeps all windows from the same source video together (recommended)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=None,
        help="Optional override for the manifest JSON path.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos that already have QA crops or NPZ tensors.",
    )
    parser.add_argument(
        "--min_sequence_count",
        type=int,
        default=1,
        help="Drop clips that yield fewer than this many windows.",
    )
    parser.add_argument(
        "--default_fps",
        type=float,
        default=30.0,
        help="Fallback FPS when the video metadata is unavailable.",
    )
    args = parser.parse_args()
    if args.window_size <= 0 or args.window_stride <= 0:
        raise SystemExit("window_size and window_stride must be positive.")
    if not args.save_frame_crops and not args.save_sequence_npz:
        raise SystemExit(
            "Enable --save_frame_crops and/or --save_sequence_npz to store data."
        )
    return args


def find_class_dirs(dataset_root: Path) -> Tuple[List[Path], Dict[str, int]]:
    class_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No behavior folders found in {dataset_root}")
    class_to_idx = {cls_dir.name: idx for idx, cls_dir in enumerate(class_dirs)}
    return class_dirs, class_to_idx


def iter_video_files(class_dir: Path):
    for path in sorted(class_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        yield path


def ensure_output_dirs(
    output_root: Path,
    class_name: str,
    video_stem: str,
    save_frame_crops: bool,
    save_sequence_npz: bool,
) -> Tuple[Path | None, Path | None]:
    qa_dir = None
    seq_dir = None
    if save_frame_crops:
        qa_dir = output_root / "qa_crops" / class_name / video_stem
        qa_dir.mkdir(parents=True, exist_ok=True)
    if save_sequence_npz:
        seq_dir = output_root / "sequence_npz" / class_name
        seq_dir.mkdir(parents=True, exist_ok=True)
    return qa_dir, seq_dir


def save_frame_crop(qa_dir: Path, detection: CropDetection) -> Path:
    path = qa_dir / f"{detection.frame_idx:06d}.jpg"
    if not path.exists():
        Image.fromarray(detection.crop_rgb).save(path, quality=95)
    return path


def save_sequence_npz(seq_dir: Path, name: str, frames: np.ndarray) -> Path:
    path = seq_dir / f"{name}.npz"
    np.savez_compressed(path, frames=frames.astype(np.uint8))
    return path


def probe_fps(video_path: Path, fallback: float) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 1e-3:
        return fallback
    return float(fps)


def build_sequences_for_video(
    model,
    video_path: Path,
    class_name: str,
    label: int,
    args,
    output_root: Path,
) -> List[SequenceSample]:
    qa_dir, seq_dir = ensure_output_dirs(
        output_root,
        class_name,
        video_path.stem,
        args.save_frame_crops,
        args.save_sequence_npz,
    )
    if args.skip_existing:
        seq_pattern = seq_dir.glob(f"{video_path.stem}_*.npz") if seq_dir else []
        qa_pattern = qa_dir.glob("*.jpg") if qa_dir else []
        if seq_dir and any(seq_pattern):
            print(f"Skipping {video_path} (sequence NPZ already exists).")
            return []
        if qa_dir and not seq_dir and any(qa_pattern):
            print(f"Skipping {video_path} (QA crops already exist).")
            return []

    fps = probe_fps(video_path, args.default_fps)

    detections = iterate_yolo_crops(
        model=model,
        video_path=video_path,
        crop_size=args.crop_size,
        pad_ratio=args.pad_ratio,
        conf_threshold=args.yolo_conf,
        iou_threshold=args.yolo_iou,
        imgsz=args.yolo_imgsz,
        device=args.device,
        keep_last_box=args.keep_last_box,
    )

    buffer: deque[CropDetection] = deque()
    sequences: List[SequenceSample] = []
    seq_index = 0

    for detection in detections:
        if qa_dir:
            save_frame_crop(qa_dir, detection)
        buffer.append(detection)

        while len(buffer) >= args.window_size:
            window_entries = list(buffer)[: args.window_size]
            start_frame = window_entries[0].frame_idx
            end_frame = window_entries[-1].frame_idx
            window_frames = np.stack(
                [entry.crop_rgb for entry in window_entries], axis=0
            )

            seq_name = f"{video_path.stem}_f{start_frame:06d}_s{seq_index:04d}"
            frame_paths = []
            if qa_dir:
                frame_paths = [
                    qa_dir / f"{entry.frame_idx:06d}.jpg" for entry in window_entries
                ]
            seq_path = (
                save_sequence_npz(seq_dir, seq_name, window_frames) if seq_dir else None
            )

            sequences.append(
                SequenceSample(
                    id=f"{class_name}_{seq_name}",
                    label=label,
                    class_name=class_name,
                    sequence_npz=seq_path,
                    frame_paths=tuple(frame_paths),
                    start_frame=start_frame,
                    end_frame=end_frame,
                    fps=fps,
                    source_video=str(video_path),
                )
            )
            seq_index += 1

            for _ in range(args.window_stride):
                if buffer:
                    buffer.popleft()

    if len(sequences) < args.min_sequence_count:
        print(f"Skipping {video_path}: only {len(sequences)} sequences.")
        return []
    return sequences


def main():
    args = parse_args()
    output_root = args.output_root or (args.dataset_root / "processed_sequences")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_path or (output_root / "sequence_manifest.json")

    class_dirs, class_to_idx = find_class_dirs(args.dataset_root)
    print(f"Found {len(class_dirs)} behavior folders.")

    model = load_yolo_model(args.yolo_weights, device=args.device)
    print(f"Loaded YOLO weights from {args.yolo_weights}.")

    all_samples: List[SequenceSample] = []
    for class_dir in class_dirs:
        label = class_to_idx[class_dir.name]
        videos = list(iter_video_files(class_dir))
        if not videos:
            print(f"Warning: no videos found under {class_dir}.")
            continue
        for video_path in videos:
            print(f"Processing {video_path}...")
            sequences = build_sequences_for_video(
                model=model,
                video_path=video_path,
                class_name=class_dir.name,
                label=label,
                args=args,
                output_root=output_root,
            )
            all_samples.extend(sequences)

    if not all_samples:
        raise RuntimeError("No sequences were generated. Check your inputs.")

    if args.split_strategy == "video":
        splits = stratified_split_by_source_video(
            all_samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        splits = stratified_split(
            all_samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    metadata = {
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "crop_size": args.crop_size,
        "pad_ratio": args.pad_ratio,
        "yolo_conf": args.yolo_conf,
        "yolo_iou": args.yolo_iou,
        "yolo_imgsz": args.yolo_imgsz,
        "split_strategy": args.split_strategy,
        "save_frame_crops": args.save_frame_crops,
        "save_sequence_npz": args.save_sequence_npz,
        "source_dataset_root": str(args.dataset_root),
    }

    save_sequence_manifest(
        manifest_path,
        splits,
        class_to_idx,
        metadata,
    )
    print(f"Manifest saved to {manifest_path}")
    print(
        "Sequences per split:",
        {split: len(entries) for split, entries in splits.items()},
    )


if __name__ == "__main__":
    main()

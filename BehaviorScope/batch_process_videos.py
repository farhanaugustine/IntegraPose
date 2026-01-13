from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch

if __package__ is None or __package__ == "":  # pragma: no cover - CLI entry
    import cropping
    import infer_behavior_lstm
    import review_annotations
else:  # pragma: no cover - package usage
    from . import cropping, infer_behavior_lstm, review_annotations

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch process videos: run inference (CSV + optional JSON) and "
            "optionally export annotated review videos."
        )
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Folder containing input videos (searched recursively).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Folder to save results (mirrors input_dir structure).",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to trained .pt checkpoint (best_model.pt).",
    )
    parser.add_argument(
        "--yolo_weights",
        type=Path,
        required=True,
        help="Path to YOLO weights (.pt).",
    )
    parser.add_argument(
        "--yolo_mode",
        type=str,
        default="auto",
        choices=("auto", "bbox", "pose"),
        help=(
            "How to interpret YOLO outputs: "
            "'auto' uses keypoints if present, "
            "'bbox' forces bbox-only (ignores keypoints), "
            "'pose' requires keypoints (errors if missing)."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cuda:0/cpu/auto).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference batch size (windows per forward pass).",
    )
    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Minimum probability to accept a prediction (else 'Unknown').",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Frames per window (defaults to checkpoint num_frames).",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=None,
        help="Stride in frames (defaults to window_size//2).",
    )
    parser.add_argument(
        "--skip_tail",
        action="store_true",
        help="Skip the final partial window instead of forcing an evaluation.",
    )
    parser.add_argument(
        "--drop_unknown",
        action="store_true",
        help="Do not include 'Unknown' segments in the merged CSV.",
    )
    parser.add_argument(
        "--skip_json",
        action="store_true",
        help="Do not write per-window JSON files.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos whose expected outputs already exist.",
    )

    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--pad_ratio", type=float, default=0.2)
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--yolo_iou", type=float, default=0.45)
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument(
        "--keep_last_box",
        action="store_true",
        help="Reuse the last box when YOLO misses a frame.",
    )

    parser.add_argument(
        "--skip_review_video",
        action="store_true",
        help="If set, only CSV/JSON are generated (no MP4 review export).",
    )
    parser.add_argument(
        "--label_bg_alpha",
        type=float,
        default=1.0,
        help=(
            "Opacity (0-1) for the black text background rectangles in review videos. "
            "Lower values reduce occlusion during QA."
        ),
    )

    parser.add_argument(
        "--export_xlsx",
        action="store_true",
        help="Export per-frame YOLO+behavior metrics workbook (.xlsx) per video.",
    )
    parser.add_argument(
        "--per_frame_mode",
        type=str,
        default="true",
        choices=["none", "simple", "true"],
        help="Per-frame behavior mapping for XLSX: none/simple/true.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top-K behavior probabilities to export (1-5).",
    )
    parser.add_argument(
        "--smoothing",
        type=str,
        default="none",
        choices=["none", "ema", "kalman", "interpolation"],
        help="Smoothing method for bbox centers when computing kinematics (XLSX).",
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.2,
        help="EMA alpha for smoothing (if smoothing=ema).",
    )
    parser.add_argument(
        "--interp_max_gap",
        type=int,
        default=5,
        help="Maximum gap (frames) to interpolate (if smoothing=interpolation).",
    )

    return parser.parse_args()


def _iter_video_files(input_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
    )


def _safe_release(resource) -> None:
    try:
        resource.release()
    except Exception:
        return


def generate_review_video(
    video_path: Path,
    csv_path: Path,
    output_video_path: Path,
    metrics_xlsx: Path | None = None,
    label_bg_alpha: float = 1.0,
) -> None:
    segments = []
    metrics_iter = None
    current_metric = None
    if metrics_xlsx is not None and metrics_xlsx.exists():
        try:
            metrics_iter = review_annotations.iter_metrics_per_frame(metrics_xlsx)
            current_metric = next(metrics_iter, None)
        except Exception as exc:
            print(
                f"  [Review] Warning: failed to load metrics XLSX ({exc}); falling back to CSV overlay."
            )
            metrics_iter = None
            current_metric = None

    if metrics_iter is None:
        if not csv_path.exists():
            print(f"  [Review] Error: CSV not found at {csv_path}")
            return
        segments = review_annotations.load_segments(csv_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [Review] Error: Could not open source video {video_path}")
        _safe_release(cap)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print("  [Review] Error: Could not create output video file.")
        _safe_release(cap)
        _safe_release(writer)
        return

    frame_idx = 0
    current_seg_idx = 0
    if segments:
        segments.sort(key=lambda seg: seg.start_frame)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if metrics_iter is not None:
                while (
                    current_metric is not None
                    and int(current_metric["frame_idx"]) < frame_idx
                ):
                    current_metric = next(metrics_iter, None)
                if (
                    current_metric is not None
                    and int(current_metric["frame_idx"]) == frame_idx
                ):
                    annotated = review_annotations.annotate_frame_bbox(
                        frame,
                        current_metric.get("bbox_xyxy"),
                        str(current_metric.get("label", "Unknown")),
                        current_metric.get("confidence"),
                        current_metric.get("yolo_status"),
                        font_scale=0.7,
                        thickness=2,
                        keypoints=current_metric.get("keypoints"),
                        label_bg_alpha=float(label_bg_alpha),
                    )
                else:
                    annotated = frame
            else:
                while (
                    current_seg_idx < len(segments)
                    and frame_idx > segments[current_seg_idx].end_frame
                ):
                    current_seg_idx += 1

                active_segment = None
                if current_seg_idx < len(segments):
                    seg = segments[current_seg_idx]
                    if seg.start_frame <= frame_idx <= seg.end_frame:
                        active_segment = seg

                annotated = review_annotations.annotate_frame(
                    frame,
                    active_segment,
                    frame_idx,
                    fps,
                    total_frames,
                    font_scale=0.7,
                    thickness=2,
                    label_bg_alpha=float(label_bg_alpha),
                )
            writer.write(annotated)
            frame_idx += 1
    finally:
        _safe_release(cap)
        _safe_release(writer)

    print(f"  [Review] Saved annotated video to: {output_video_path}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        sys.exit(f"Input directory not found: {input_dir}")
    if not args.model_path.exists():
        sys.exit(f"Model checkpoint not found: {args.model_path}")
    if not args.yolo_weights.exists():
        sys.exit(f"YOLO weights not found: {args.yolo_weights}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Running on device: {device}")

    print("Loading behavior model...")
    model, class_names, hparams = infer_behavior_lstm.load_checkpoint(
        args.model_path, device
    )

    print("Loading YOLO model...")
    if device.type == "cuda":
        yolo_device = f"cuda:{device.index}" if device.index is not None else "cuda"
    else:
        yolo_device = "cpu"
    yolo_model = cropping.load_yolo_model(args.yolo_weights, device=yolo_device)

    default_num_frames = int(hparams.get("num_frames", 16))
    default_frame_size = int(hparams.get("frame_size", 160))

    window_size = args.window_size or default_num_frames
    stride = args.window_stride or max(1, window_size // 2)
    crop_size = args.crop_size or default_frame_size

    backbone_name = str(hparams.get("backbone", "")).lower()
    if backbone_name.startswith("vit") and int(crop_size) != 224:
        raise ValueError(
            "ViT backbones require 224x224 crops in this implementation. "
            "Leave --crop_size blank or set it to 224."
        )

    video_files = _iter_video_files(input_dir)
    if not video_files:
        sys.exit("No video files found in input directory.")

    print(f"\nFound {len(video_files)} videos. Starting batch processing...\n")

    overall_started = time.perf_counter()
    processed = 0
    skipped = 0
    failed = 0

    for idx, video_path in enumerate(video_files, 1):
        print(f"Processing [{idx}/{len(video_files)}]: {video_path.name}")

        rel_path = video_path.relative_to(input_dir)
        out_subdir = output_dir / rel_path.parent
        out_subdir.mkdir(parents=True, exist_ok=True)

        base_name = rel_path.stem
        output_csv = out_subdir / f"{base_name}.behavior.csv"
        output_json = out_subdir / f"{base_name}.behavior.windows.json"
        output_vid = out_subdir / f"{base_name}.review.mp4"
        export_xlsx = bool(args.export_xlsx) or (not bool(args.skip_review_video))
        output_xlsx = out_subdir / f"{base_name}.metrics.xlsx" if export_xlsx else None

        if args.skip_existing:
            csv_ok = output_csv.exists()
            json_ok = args.skip_json or output_json.exists()
            vid_ok = args.skip_review_video or output_vid.exists()
            xlsx_ok = (not export_xlsx) or (output_xlsx is not None and output_xlsx.exists())
            if csv_ok and json_ok and vid_ok and xlsx_ok:
                print("  [Skip] Outputs already exist.")
                skipped += 1
                continue

        video_started = time.perf_counter()
        csv_writer = infer_behavior_lstm.CsvMergeWriter(
            output_csv, keep_unknown=not args.drop_unknown
        )
        json_writer = infer_behavior_lstm.JsonStreamWriter(
            None if args.skip_json else output_json
        )

        frame_meta: list[dict] | None = [] if export_xlsx else None
        window_meta: list[dict] | None = [] if export_xlsx else None
        window_probs: list[object] | None = None
        if export_xlsx and str(args.per_frame_mode).strip().lower() == "true":
            window_probs = []

        win_count = 0
        last_end_frame: int | None = None
        try:
            for prediction in infer_behavior_lstm.predict_windows(
                model=model,
                class_names=class_names,
                video_path=video_path,
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
                keep_last_box=args.keep_last_box,
                skip_tail=args.skip_tail,
                yolo_max_det=1,
                yolo_nms=True,
                top_k=args.top_k,
                frame_meta=frame_meta,
                window_meta=window_meta,
                window_probs=window_probs,  # type: ignore[arg-type]
                yolo_mode=args.yolo_mode,
            ):
                csv_writer.add(prediction)
                json_writer.write(prediction)
                win_count += 1
                last_end_frame = prediction.end_frame
        except Exception as exc:
            failed += 1
            print(f"  [ERROR] Inference failed: {exc}")
            continue
        finally:
            try:
                csv_writer.close()
            except Exception:
                pass
            try:
                json_writer.close()
            except Exception:
                pass

        elapsed = time.perf_counter() - video_started
        speed = ""
        if last_end_frame is not None and elapsed > 1e-6:
            effective_fps = (last_end_frame + 1) / elapsed
            windows_per_sec = win_count / elapsed
            speed = f" speed={effective_fps:.2f}fps ({windows_per_sec:.2f} win/s)"
        print(f"  [Inference] Done. windows={win_count}{speed}")

        if export_xlsx and output_xlsx is not None:
            try:
                infer_behavior_lstm._export_metrics_xlsx(  # type: ignore[attr-defined]
                    output_xlsx,
                    video_path=video_path,
                    class_names=class_names,
                    fps=infer_behavior_lstm.probe_fps(video_path),
                    prob_threshold=args.prob_threshold,
                    window_size=window_size,
                    window_stride=stride,
                    crop_size=crop_size,
                    pad_ratio=args.pad_ratio,
                    yolo_conf=args.yolo_conf,
                    yolo_iou=args.yolo_iou,
                    yolo_imgsz=args.yolo_imgsz,
                    keep_last_box=args.keep_last_box,
                    smoothing=args.smoothing,
                    ema_alpha=args.ema_alpha,
                    interp_max_gap=args.interp_max_gap,
                    per_frame_mode=args.per_frame_mode,
                    top_k=args.top_k,
                    frame_meta=frame_meta or [],
                    window_meta=window_meta or [],
                    window_probs=window_probs,  # type: ignore[arg-type]
                    model_path=args.model_path,
                    yolo_weights=args.yolo_weights,
                    yolo_mode=args.yolo_mode,
                )
                print(f"  [XLSX] Saved metrics workbook to: {output_xlsx}")
            except Exception as exc:
                failed += 1
                print(f"  [ERROR] XLSX export failed: {exc}")
                continue

        if not args.skip_review_video:
            try:
                generate_review_video(
                    video_path,
                    output_csv,
                    output_vid,
                    metrics_xlsx=output_xlsx,
                    label_bg_alpha=float(args.label_bg_alpha),
                )
            except Exception as exc:
                failed += 1
                print(f"  [ERROR] Review export failed: {exc}")
                continue

        processed += 1

    elapsed_total = time.perf_counter() - overall_started
    print(
        "\nBatch processing complete. "
        f"processed={processed} skipped={skipped} failed={failed} "
        f"runtime={elapsed_total:.2f}s"
    )


if __name__ == "__main__":
    main()

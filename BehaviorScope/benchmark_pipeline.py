from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

if __package__ is None or __package__ == "":  # pragma: no cover - CLI entry
    import cropping
    import infer_behavior_lstm
    from data import load_sequence_manifest
else:  # pragma: no cover - package usage
    from . import cropping, infer_behavior_lstm
    from .data import load_sequence_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the full pipeline (YOLO + crop + BehaviorScope classifier) "
            "using windows from a manifest split (default: val)."
        )
    )
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--yolo_weights", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Manifest split to benchmark.",
    )

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--prob_threshold", type=float, default=0.5)

    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Override frames per window (defaults to manifest meta or checkpoint).",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=None,
        help="Override stride in frames (defaults to manifest meta or window_size//2).",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=None,
        help="Override crop size (defaults to checkpoint frame_size).",
    )

    parser.add_argument("--pad_ratio", type=float, default=None)
    parser.add_argument("--yolo_conf", type=float, default=None)
    parser.add_argument("--yolo_iou", type=float, default=None)
    parser.add_argument("--yolo_imgsz", type=int, default=None)
    parser.add_argument("--yolo_max_det", type=int, default=1)
    parser.add_argument("--no_yolo_nms", action="store_true")
    parser.add_argument(
        "--no_keep_last_box",
        action="store_true",
        help="Disable reusing the last YOLO box when detections are missed.",
    )
    parser.add_argument(
        "--include_tail",
        action="store_true",
        help="Include the final partial window (does not match preprocessing).",
    )

    parser.add_argument(
        "--limit_videos",
        type=int,
        default=None,
        help="Optional cap on number of unique videos processed.",
    )
    parser.add_argument(
        "--limit_windows",
        type=int,
        default=None,
        help="Optional cap on number of windows evaluated (across all videos).",
    )
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be > 0.")
    if args.yolo_max_det <= 0:
        raise SystemExit("--yolo_max_det must be >= 1.")
    return args


def _resolve_video_path(
    raw: str,
    manifest_path: Path,
    meta: dict,
) -> Path:
    candidate = Path(raw)
    if candidate.exists():
        return candidate
    dataset_root = meta.get("source_dataset_root")
    if dataset_root:
        resolved = Path(dataset_root) / candidate
        if resolved.exists():
            return resolved
    resolved = manifest_path.parent / candidate
    if resolved.exists():
        return resolved
    return candidate


def _accumulate(dst: dict[str, float], src: dict[str, float]) -> None:
    for key, value in src.items():
        try:
            dst[key] = dst.get(key, 0.0) + float(value)
        except Exception:
            continue


def main() -> None:
    args = parse_args()
    if not args.manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if not args.yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.yolo_weights}")

    splits, class_to_idx, idx_to_class, meta = load_sequence_manifest(args.manifest_path)
    samples = list(splits.get(args.split, []))
    if not samples:
        raise RuntimeError(f"No samples found in manifest split: {args.split}")

    if args.limit_windows is not None and args.limit_windows > 0:
        samples = samples[: args.limit_windows]

    expected_by_video: dict[Path, dict[tuple[int, int], int]] = defaultdict(dict)
    for sample in samples:
        if not sample.source_video:
            continue
        video_path = _resolve_video_path(sample.source_video, args.manifest_path, meta)
        expected_by_video[video_path][(sample.start_frame, sample.end_frame)] = sample.label

    video_paths = sorted(expected_by_video.keys(), key=lambda p: p.as_posix())
    if args.limit_videos is not None and args.limit_videos > 0:
        video_paths = video_paths[: args.limit_videos]
        expected_by_video = {vp: expected_by_video[vp] for vp in video_paths}

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, class_names, hparams = infer_behavior_lstm.load_checkpoint(args.model_path, device)

    default_num_frames = int(hparams.get("num_frames", 16))
    default_frame_size = int(hparams.get("frame_size", 160))

    window_size = args.window_size or int(meta.get("window_size") or 0) or default_num_frames
    stride = (
        args.window_stride
        or int(meta.get("window_stride") or 0)
        or max(1, window_size // 2)
    )
    crop_size = args.crop_size or default_frame_size

    backbone_name = str(hparams.get("backbone", "")).lower()
    if backbone_name.startswith("vit") and int(crop_size) != 224:
        raise ValueError(
            "ViT backbones require 224x224 crops in this implementation. "
            "Leave --crop_size blank or set it to 224."
        )

    pad_ratio = float(args.pad_ratio if args.pad_ratio is not None else meta.get("pad_ratio", 0.2))
    yolo_conf = float(args.yolo_conf if args.yolo_conf is not None else meta.get("yolo_conf", 0.25))
    yolo_iou = float(args.yolo_iou if args.yolo_iou is not None else meta.get("yolo_iou", 0.45))
    yolo_imgsz = int(args.yolo_imgsz if args.yolo_imgsz is not None else meta.get("yolo_imgsz", 640))
    yolo_max_det = int(args.yolo_max_det)
    yolo_nms = not bool(args.no_yolo_nms)
    keep_last_box = not bool(args.no_keep_last_box)
    skip_tail = not bool(args.include_tail)

    if device.type == "cuda":
        yolo_device = f"cuda:{device.index}" if device.index is not None else "cuda"
    else:
        yolo_device = "cpu"

    yolo_model = cropping.load_yolo_model(args.yolo_weights, device=yolo_device)

    idx_to_name = {int(i): name for i, name in enumerate(class_names)}
    mismatches = []
    for idx, name in idx_to_class.items():
        if idx in idx_to_name and idx_to_name[idx] != name:
            mismatches.append((idx, idx_to_name[idx], name))
    if mismatches:
        print("[Benchmark] Warning: checkpoint class mapping differs from manifest idx_to_class.", flush=True)
        for idx, ck, mf in mismatches[:10]:
            print(f"  idx={idx} checkpoint={ck!r} manifest={mf!r}", flush=True)

    print(
        f"[Benchmark] split={args.split} videos={len(video_paths)} windows={len(samples)}",
        flush=True,
    )
    print(
        f"[Benchmark] window_size={window_size} stride={stride} batch_size={args.batch_size} prob_threshold={args.prob_threshold}",
        flush=True,
    )
    print(
        f"[Benchmark] YOLO conf={yolo_conf} iou={yolo_iou} imgsz={yolo_imgsz} max_det={yolo_max_det} nms={yolo_nms} keep_last_box={keep_last_box}",
        flush=True,
    )

    total_correct = 0
    total_eval = 0
    total_unknown = 0
    per_class_total: dict[int, int] = defaultdict(int)
    per_class_correct: dict[int, int] = defaultdict(int)
    per_class_unknown: dict[int, int] = defaultdict(int)

    timing_total: dict[str, float] = {}
    missing_total = 0
    missing_videos = 0
    processed_videos = 0

    for vid_idx, video_path in enumerate(video_paths, 1):
        expected = expected_by_video[video_path]
        remaining = dict(expected)
        if not video_path.exists():
            print(f"[Benchmark] Skipping missing video: {video_path}", flush=True)
            continue

        processed_videos += 1
        print(
            f"[Benchmark] Video [{vid_idx}/{len(video_paths)}]: {video_path.name} (eval_windows={len(expected)})",
            flush=True,
        )

        timing: dict[str, float] = {}
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
                pad_ratio=pad_ratio,
                yolo_conf=yolo_conf,
                yolo_iou=yolo_iou,
                yolo_imgsz=yolo_imgsz,
                keep_last_box=keep_last_box,
                skip_tail=skip_tail,
                yolo_max_det=yolo_max_det,
                yolo_nms=yolo_nms,
                timing=timing,
            ):
                key = (prediction.start_frame, prediction.end_frame)
                if key not in remaining:
                    continue
                true_label = remaining.pop(key)
                total_eval += 1
                per_class_total[true_label] += 1

                if prediction.class_idx is None:
                    total_unknown += 1
                    per_class_unknown[true_label] += 1
                elif int(prediction.class_idx) == int(true_label):
                    total_correct += 1
                    per_class_correct[true_label] += 1
        except Exception as exc:
            print(f"[Benchmark] Error processing {video_path}: {exc}", flush=True)
            continue
        finally:
            _accumulate(timing_total, timing)

        if remaining:
            missing_total += len(remaining)
            missing_videos += 1
            print(
                f"[Benchmark] Warning: missing {len(remaining)}/{len(expected)} expected windows in {video_path.name}.",
                flush=True,
            )

    if total_eval <= 0:
        raise RuntimeError("No evaluation windows were matched. Check window_size/stride and YOLO settings.")

    overall_acc = total_correct / total_eval
    print(f"\n[Benchmark] Accuracy: {total_correct}/{total_eval} = {overall_acc:.4f}", flush=True)
    print(f"[Benchmark] Unknown: {total_unknown}/{total_eval} = {total_unknown / total_eval:.4f}", flush=True)
    print("[Benchmark] Per-class accuracy:", flush=True)
    for idx in sorted(per_class_total.keys()):
        total = per_class_total[idx]
        correct = per_class_correct.get(idx, 0)
        unknown = per_class_unknown.get(idx, 0)
        name = idx_to_name.get(idx, str(idx_to_class.get(idx, f"class_{idx}")))
        acc = correct / total if total else 0.0
        unk = unknown / total if total else 0.0
        print(f"  {name}: acc={acc:.4f} ({correct}/{total}) unknown={unk:.4f} ({unknown}/{total})", flush=True)

    yolo_ms = float(
        timing_total.get("yolo_preprocess_ms", 0.0)
        + timing_total.get("yolo_inference_ms", 0.0)
        + timing_total.get("yolo_postprocess_ms", 0.0)
    )
    crop_ms = float(timing_total.get("crop_ms", 0.0))
    clip_prep_ms = float(timing_total.get("clip_prep_ms", 0.0))
    batch_to_device_ms = float(timing_total.get("batch_to_device_ms", 0.0))
    model_forward_ms = float(timing_total.get("model_forward_ms", 0.0))
    bs_ms = crop_ms + clip_prep_ms + batch_to_device_ms + model_forward_ms
    pipeline_ms = yolo_ms + bs_ms

    frames = float(timing_total.get("yolo_frames", 0.0))
    windows_pred = float(timing_total.get("windows_predicted", 0.0))
    pipeline_s = pipeline_ms / 1000.0
    yolo_s = yolo_ms / 1000.0
    bs_s = bs_ms / 1000.0

    fps_pipeline = (frames / pipeline_s) if pipeline_s > 1e-9 else 0.0
    fps_yolo = (frames / yolo_s) if yolo_s > 1e-9 else 0.0
    win_per_s = (windows_pred / pipeline_s) if pipeline_s > 1e-9 else 0.0

    print("\n[Benchmark] Timing breakdown (ms):", flush=True)
    print(
        f"  YOLO: preprocess={timing_total.get('yolo_preprocess_ms', 0.0):.1f} "
        f"infer={timing_total.get('yolo_inference_ms', 0.0):.1f} "
        f"post={timing_total.get('yolo_postprocess_ms', 0.0):.1f} "
        f"total={yolo_ms:.1f}",
        flush=True,
    )
    print(
        f"  Crop: {crop_ms:.1f}  ClipPrep(transform+stack): {clip_prep_ms:.1f}  "
        f"ToDevice: {batch_to_device_ms:.1f}  Model: {model_forward_ms:.1f}  "
        f"BehaviorScope(total): {bs_ms:.1f}",
        flush=True,
    )
    print(
        f"[Benchmark] Throughput (YOLO+BehaviorScope): {fps_pipeline:.2f} FPS  ({win_per_s:.2f} windows/s)",
        flush=True,
    )
    print(f"[Benchmark] YOLO-only throughput (from YOLO speed dict): {fps_yolo:.2f} FPS", flush=True)
    if missing_total:
        print(
            f"[Benchmark] Missing windows: {missing_total} across {missing_videos} videos (processed_videos={processed_videos})",
            flush=True,
        )

    summary = {
        "split": args.split,
        "videos": len(video_paths),
        "windows_manifest": len(samples),
        "windows_evaluated": total_eval,
        "accuracy": overall_acc,
        "unknown_rate": total_unknown / total_eval,
        "per_class": {
            idx_to_name.get(idx, str(idx)): {
                "correct": per_class_correct.get(idx, 0),
                "total": per_class_total[idx],
                "accuracy": per_class_correct.get(idx, 0) / per_class_total[idx]
                if per_class_total[idx]
                else 0.0,
                "unknown": per_class_unknown.get(idx, 0),
            }
            for idx in per_class_total
        },
        "timing_ms": {
            "yolo_preprocess_ms": timing_total.get("yolo_preprocess_ms", 0.0),
            "yolo_inference_ms": timing_total.get("yolo_inference_ms", 0.0),
            "yolo_postprocess_ms": timing_total.get("yolo_postprocess_ms", 0.0),
            "crop_ms": crop_ms,
            "clip_prep_ms": clip_prep_ms,
            "batch_to_device_ms": batch_to_device_ms,
            "model_forward_ms": model_forward_ms,
        },
        "throughput": {
            "frames": frames,
            "windows_predicted": windows_pred,
            "pipeline_fps": fps_pipeline,
            "yolo_fps": fps_yolo,
            "windows_per_sec": win_per_s,
        },
        "config": {
            "device": str(device),
            "batch_size": args.batch_size,
            "prob_threshold": args.prob_threshold,
            "window_size": window_size,
            "window_stride": stride,
            "crop_size": crop_size,
            "pad_ratio": pad_ratio,
            "yolo_conf": yolo_conf,
            "yolo_iou": yolo_iou,
            "yolo_imgsz": yolo_imgsz,
            "yolo_max_det": yolo_max_det,
            "yolo_nms": yolo_nms,
            "keep_last_box": keep_last_box,
            "skip_tail": skip_tail,
        },
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[Benchmark] Wrote summary JSON: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()


from __future__ import annotations

import hashlib
import json
import csv
from collections import Counter
from pathlib import Path

from .models import AnnotationRecord, BehaviorRecord, VideoRecord
from .store import AnnotationStore


def export_full_video_annotations(
    store: AnnotationStore,
    *,
    project_id: int,
    output_root: Path,
    val_ratio: float = 0.2,
    recommended_window_frames: int = 32,
    recommended_stride_frames: int = 16,
    split_by_video_id: dict[int, str] | None = None,
    source_split_by_video_id: dict[int, str] | None = None,
) -> dict:
    """Export GUI annotations as full-video BENTO files and a builder CSV.

    This is the bridge from user-owned videos to the full-video window builder.
    It preserves full source videos, writes one .annot file per video, and
    emits a source_manifest.csv with train/val assignments.
    """
    output_root = Path(output_root)
    annot_dir = output_root / "annotations"
    annot_dir.mkdir(parents=True, exist_ok=True)

    videos = store.list_videos(project_id)
    behaviors = store.list_behaviors(project_id)
    behavior_by_id = {b.id: b for b in behaviors}
    annotations_by_video: dict[int, list[AnnotationRecord]] = {}
    for annotation in store.approved_annotations_for_export(project_id):
        if annotation.behavior_id in behavior_by_id:
            annotations_by_video.setdefault(annotation.video_id, []).append(annotation)

    class_names = [b.slug for b in behaviors if b.slug != "other"]
    if "other" not in class_names:
        class_names.append("other")
    (output_root / "class_names.txt").write_text(
        "\n".join(class_names) + "\n",
        encoding="utf-8",
    )

    split_by_video = (
        {int(key): str(value) for key, value in split_by_video_id.items()}
        if split_by_video_id is not None
        else _assign_video_splits(videos, val_ratio=float(val_ratio))
    )
    source_split_by_video_id = source_split_by_video_id or {}
    rows: list[dict] = []
    batch_videos: list[dict] = []
    class_counts_by_split: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}
    span_counts_by_class: Counter[str] = Counter()
    for video in videos:
        anns = sorted(
            annotations_by_video.get(video.id, []),
            key=lambda a: (a.start_ms, a.end_ms, a.id),
        )
        annot_path = annot_dir / f"{Path(video.filename).stem}.annot"
        _write_bento_annot(annot_path, video, anns, behavior_by_id)
        split = split_by_video.get(video.id, "train")
        batch_annotations = []
        for annotation in anns:
            behavior = behavior_by_id.get(annotation.behavior_id)
            if behavior is None:
                continue
            class_counts_by_split.setdefault(split, Counter())[behavior.slug] += 1
            span_counts_by_class[behavior.slug] += 1
            batch_annotations.append(
                {
                    "annotation_id": int(annotation.id),
                    "behavior": behavior.name,
                    "behavior_slug": behavior.slug,
                    "start_frame": int(annotation.start_frame),
                    "end_frame": int(annotation.end_frame),
                    "start_ms": int(annotation.start_ms),
                    "end_ms": int(annotation.end_ms),
                    "duration_ms": int(annotation.duration_ms),
                    "confidence": annotation.confidence,
                    "is_ambiguous": bool(annotation.is_ambiguous),
                    "is_locked": bool(annotation.is_locked),
                    "notes": annotation.notes,
                }
            )
        rows.append(
            {
                "split": split,
                "source_split": str(source_split_by_video_id.get(video.id, split)),
                "video_id": Path(video.filename).stem,
                "video_path": video.path,
                "annot_path": str(annot_path),
                "fps": float(video.fps),
                "total_frames": int(video.total_frames),
                "width": int(video.width),
                "height": int(video.height),
                "approved_annotation_count": len(anns),
            }
        )
        batch_videos.append(
            {
                "video_id": Path(video.filename).stem,
                "filename": video.filename,
                "video_path": video.path,
                "annot_path": str(annot_path),
                "split": split,
                "source_split": str(source_split_by_video_id.get(video.id, split)),
                "fps": float(video.fps),
                "total_frames": int(video.total_frames),
                "duration_ms": int(video.duration_ms),
                "width": int(video.width),
                "height": int(video.height),
                "sha256": video.sha256,
                "approved_annotation_count": len(anns),
                "annotations": batch_annotations,
            }
        )

    manifest_csv = output_root / "source_manifest.csv"
    fieldnames = [
        "split",
        "source_split",
        "video_id",
        "video_path",
        "annot_path",
        "fps",
        "total_frames",
        "width",
        "height",
        "approved_annotation_count",
    ]
    with manifest_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    batch_json = output_root / "full_video_annotations.batch.json"
    batch_payload = {
        "version": "yolo-temporal-full-video-batch-v1",
        "output_root": str(output_root),
        "source_manifest_csv": str(manifest_csv),
        "class_names_file": str(output_root / "class_names.txt"),
        "annotation_format": "bento-per-video",
        "video_count": len(videos),
        "approved_annotation_count": sum(len(v) for v in annotations_by_video.values()),
        "class_names": class_names,
        "val_ratio": float(val_ratio),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "videos": batch_videos,
    }
    batch_json.write_text(json.dumps(batch_payload, indent=2), encoding="utf-8")

    preflight = _build_full_video_preflight(
        rows=rows,
        batch_videos=batch_videos,
        class_names=class_names,
        class_counts_by_split=class_counts_by_split,
        span_counts_by_class=span_counts_by_class,
        recommended_window_frames=int(recommended_window_frames),
        recommended_stride_frames=int(recommended_stride_frames),
    )
    preflight_json = output_root / "preflight_report.json"
    preflight_json.write_text(json.dumps(preflight, indent=2), encoding="utf-8")

    summary = {
        "version": "yolo-temporal-full-video-annotation-export-v1",
        "output_root": str(output_root),
        "source_manifest_csv": str(manifest_csv),
        "class_names_file": str(output_root / "class_names.txt"),
        "batch_annotations_json": str(batch_json),
        "preflight_report_json": str(preflight_json),
        "video_count": len(videos),
        "approved_annotation_count": sum(len(v) for v in annotations_by_video.values()),
        "class_names": class_names,
        "val_ratio": float(val_ratio),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "preflight": preflight,
    }
    (output_root / "full_video_annotations.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _assign_video_splits(videos: list[VideoRecord], val_ratio: float) -> dict[int, str]:
    if len(videos) <= 1 or val_ratio <= 0.0:
        return {video.id: "train" for video in videos}
    val_ratio = max(0.0, min(1.0, val_ratio))
    n_val = int(round(len(videos) * val_ratio))
    n_val = min(max(1, n_val), len(videos) - 1)
    ranked = sorted(
        videos,
        key=lambda video: hashlib.sha1(str(video.sha256).encode("utf-8")).hexdigest(),
    )
    val_ids = {video.id for video in ranked[:n_val]}
    return {video.id: ("val" if video.id in val_ids else "train") for video in videos}


def _build_full_video_preflight(
    *,
    rows: list[dict],
    batch_videos: list[dict],
    class_names: list[str],
    class_counts_by_split: dict[str, Counter[str]],
    span_counts_by_class: Counter[str],
    recommended_window_frames: int,
    recommended_stride_frames: int,
) -> dict:
    warnings: list[str] = []
    info: list[str] = []
    real_class_names = [name for name in class_names if name != "other"]
    zero_annotation_videos = [
        str(video["video_id"])
        for video in batch_videos
        if int(video.get("approved_annotation_count", 0)) == 0
    ]
    if zero_annotation_videos:
        preview = ", ".join(zero_annotation_videos[:5])
        suffix = "..." if len(zero_annotation_videos) > 5 else ""
        info.append(
            f"{len(zero_annotation_videos)} video(s) have no approved spans and will train as background-only unless annotations are added: {preview}{suffix}"
        )

    for split in ("train", "val"):
        split_rows = [row for row in rows if row.get("split") == split]
        if not split_rows:
            warnings.append(f"No videos are assigned to the {split} split.")
            continue
        missing = [
            name
            for name in real_class_names
            if int(class_counts_by_split.get(split, Counter()).get(name, 0)) == 0
        ]
        if missing:
            warnings.append(
                f"{split} split has no approved spans for: {', '.join(missing)}."
            )

    for class_name in real_class_names:
        count = int(span_counts_by_class.get(class_name, 0))
        if count == 0:
            warnings.append(f"No approved spans were exported for class '{class_name}'.")
        elif count < 5:
            info.append(
                f"Class '{class_name}' has only {count} approved span(s); consider adding more examples before training."
            )

    short_spans: list[str] = []
    overlap_videos: list[str] = []
    for video in batch_videos:
        anns = sorted(
            video.get("annotations", []),
            key=lambda row: (int(row.get("start_frame", 0)), int(row.get("end_frame", 0))),
        )
        for annotation in anns:
            if int(annotation.get("end_frame", 0)) - int(annotation.get("start_frame", 0)) + 1 < recommended_window_frames:
                short_spans.append(
                    f"{video['video_id']}:{annotation.get('annotation_id')}"
                )
        for prev, current in zip(anns, anns[1:]):
            if int(current.get("start_frame", 0)) <= int(prev.get("end_frame", 0)):
                overlap_videos.append(str(video["video_id"]))
                break
    if short_spans:
        preview = ", ".join(short_spans[:8])
        suffix = "..." if len(short_spans) > 8 else ""
        info.append(
            f"{len(short_spans)} approved span(s) are shorter than the recommended {recommended_window_frames}-frame behavior window: {preview}{suffix}"
        )
    if overlap_videos:
        warnings.append(
            "Overlapping approved spans were found in: "
            + ", ".join(sorted(set(overlap_videos))[:8])
            + ("..." if len(set(overlap_videos)) > 8 else "")
        )

    total_windows = 0
    for row in rows:
        total_frames = int(row.get("total_frames") or 0)
        if total_frames >= recommended_window_frames:
            total_windows += 1 + max(
                0,
                (total_frames - recommended_window_frames) // max(1, recommended_stride_frames),
            )
    return {
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "info": info,
        "video_count": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "approved_annotation_count": sum(int(row.get("approved_annotation_count", 0)) for row in rows),
        "class_span_counts": {name: int(span_counts_by_class.get(name, 0)) for name in class_names},
        "class_span_counts_by_split": {
            split: {name: int(counts.get(name, 0)) for name in class_names}
            for split, counts in sorted(class_counts_by_split.items())
        },
        "zero_annotation_video_count": len(zero_annotation_videos),
        "recommended_window_frames": int(recommended_window_frames),
        "recommended_stride_frames": int(recommended_stride_frames),
        "estimated_windows_at_recommended_settings": int(total_windows),
    }


def _write_bento_annot(
    path: Path,
    video: VideoRecord,
    annotations: list[AnnotationRecord],
    behavior_by_id: dict[int, BehaviorRecord],
) -> None:
    by_behavior: dict[str, list[AnnotationRecord]] = {}
    for annotation in annotations:
        behavior = behavior_by_id.get(annotation.behavior_id)
        if behavior is None:
            continue
        by_behavior.setdefault(behavior.slug, []).append(annotation)

    stop_time = float(video.duration_ms) / 1000.0
    lines = [
        "Bento annotation file",
        f"Annotation framerate: {float(video.fps):.6f}",
        f"Annotation stop time: {stop_time:.6f}",
        "",
        "Ch1----------",
    ]
    for behavior_name in sorted(by_behavior):
        rows = by_behavior[behavior_name]
        lines.append(f">{behavior_name}")
        lines.append("Start Stop Duration")
        for annotation in rows:
            fps = max(float(video.fps), 1e-6)
            start_s = float(annotation.start_frame) / fps
            stop_s = float(annotation.end_frame + 1) / fps
            lines.append(f"{start_s:.6f} {stop_s:.6f} {max(0.0, stop_s - start_s):.6f}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Callable, Dict, List


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".mpeg", ".mpg"}


def format_grounding_dino_import_error(exc: BaseException) -> str:
    raw = str(exc).strip() or exc.__class__.__name__
    if isinstance(exc, OSError) and "c10.dll" in raw.lower():
        return (
            "PyTorch failed to initialize while AutoLabel Forge was importing GroundingDINO. "
            "On Windows this usually means the current process is loading DLLs from a different "
            "Conda/venv than the one where IntegraPose is installed, or the installed "
            "torch/torchvision build does not match the machine. "
            f"Current Python executable: {sys.executable}. "
            "Activate the target environment before launching IntegraPose so its PATH starts with "
            "that environment, and reinstall a matching torch/torchvision pair if needed "
            "(CPU-only is the safest baseline for verification). "
            f"Original error: {raw}"
        )
    return raw


def install_groundingdino_yapf_shim() -> None:
    """GroundingDINO's config loader imports yapf for pretty-printing only.

    Some Windows environments hang while importing yapf. Inference does not need
    the formatter, so we provide the minimal API that slconfig expects.
    """

    if "yapf.yapflib.yapf_api" in sys.modules:
        return

    yapf_module = types.ModuleType("yapf")
    yapflib_module = types.ModuleType("yapf.yapflib")
    yapf_api_module = types.ModuleType("yapf.yapflib.yapf_api")

    def _format_code(text: str, style_config=None, verify=None):
        return text, False

    yapf_api_module.FormatCode = _format_code  # type: ignore[attr-defined]
    sys.modules["yapf"] = yapf_module
    sys.modules["yapf.yapflib"] = yapflib_module
    sys.modules["yapf.yapflib.yapf_api"] = yapf_api_module


def load_job_from_path(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("AutoLabel Forge job payload must be a JSON object.")
    return payload


def _parse_ontology(text: str) -> Dict[str, str]:
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("Ontology cannot be empty. Use one entry per line as: caption=class")
    mapping: Dict[str, str] = {}
    for row in rows:
        if "=" not in row:
            raise ValueError(f"Ontology line is missing '=': {row}")
        caption, cls = row.split("=", 1)
        caption = caption.strip()
        cls = cls.strip()
        if not caption or not cls:
            raise ValueError(f"Ontology line is incomplete: {row}")
        mapping[caption] = cls
    return mapping


def _list_videos(video_dir: Path) -> List[Path]:
    if not video_dir.is_dir():
        return []
    return sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def extract_frames(
    videos: List[Path],
    frames_dir: Path,
    *,
    stride: int,
    ext: str,
    max_frames: int,
    logger: Callable[[str], None],
) -> int:
    import cv2

    saved_total = 0
    for video_path in videos:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            logger(f"[ERROR] Could not open video: {video_path}")
            continue
        frame_idx = 0
        saved_for_video = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                out_path = frames_dir / f"{video_path.stem}_{frame_idx:06d}{ext}"
                cv2.imwrite(str(out_path), frame)
                saved_total += 1
                saved_for_video += 1
                if max_frames > 0 and saved_for_video >= max_frames:
                    break
            frame_idx += 1
        capture.release()
        logger(f"Extracted {saved_for_video} frame(s) from {video_path.name}")
    return saved_total


def _write_fallback_data_yaml(dataset_dir: Path, class_names: List[str], logger: Callable[[str], None]) -> None:
    import yaml

    data_yaml_path = dataset_dir / "data.yaml"
    if data_yaml_path.exists():
        return
    payload = {
        "path": str(dataset_dir),
        "train": "images",
        "val": "images",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    data_yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    logger(f"Generated fallback data.yaml: {data_yaml_path}")


def _load_yolo_boxes(label_path: Path, image_w: int, image_h: int) -> List[tuple[int, tuple[int, int, int, int]]]:
    out: List[tuple[int, tuple[int, int, int, int]]] = []
    if not label_path.exists():
        return out
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
        except Exception:
            continue
        x1 = max(0, min(image_w - 1, int(round((cx - bw / 2.0) * image_w))))
        y1 = max(0, min(image_h - 1, int(round((cy - bh / 2.0) * image_h))))
        x2 = max(0, min(image_w - 1, int(round((cx + bw / 2.0) * image_w))))
        y2 = max(0, min(image_h - 1, int(round((cy + bh / 2.0) * image_h))))
        out.append((class_id, (x1, y1, x2, y2)))
    return out


def _iter_preview_sources(dataset_dir: Path) -> List[tuple[str, Path, Path, Path]]:
    previews_root = dataset_dir / "previews"
    candidates = [
        ("train", dataset_dir / "train" / "images", dataset_dir / "train" / "labels", previews_root / "train"),
        ("valid", dataset_dir / "valid" / "images", dataset_dir / "valid" / "labels", previews_root / "valid"),
        ("images", dataset_dir / "images", dataset_dir / "labels", previews_root / "images"),
    ]
    return [(name, images_dir, labels_dir, preview_dir) for name, images_dir, labels_dir, preview_dir in candidates if images_dir.is_dir() and labels_dir.is_dir()]


def _sample_preview_items(
    items: List[tuple[str, Path, Path, Path]],
    max_preview_images: int,
) -> List[tuple[str, Path, Path, Path]]:
    if max_preview_images <= 0 or len(items) <= max_preview_images:
        return items
    if max_preview_images == 1:
        return [items[len(items) // 2]]
    last_index = len(items) - 1
    indices = [round(i * last_index / (max_preview_images - 1)) for i in range(max_preview_images)]
    return [items[idx] for idx in indices]


def generate_preview_overlays(
    dataset_dir: Path,
    class_names: List[str],
    logger: Callable[[str], None],
    *,
    max_preview_images: int = 0,
) -> int:
    import cv2

    written = 0
    preview_sources = _iter_preview_sources(dataset_dir)
    if not preview_sources:
        logger(f"No previewable image/label folders found under {dataset_dir}")
        return 0

    available_by_split: Dict[str, int] = {}
    preview_items: List[tuple[str, Path, Path, Path]] = []
    preview_dirs: Dict[str, Path] = {}
    for split_name, images_dir, labels_dir, split_preview_dir in preview_sources:
        image_paths = sorted([p for p in images_dir.iterdir() if p.is_file()])
        available_by_split[split_name] = len(image_paths)
        preview_dirs[split_name] = split_preview_dir
        for image_path in image_paths:
            preview_items.append((split_name, image_path, labels_dir / f"{image_path.stem}.txt", split_preview_dir))

    selected_items = _sample_preview_items(preview_items, max_preview_images)
    selected_by_split: Dict[str, int] = {}
    for split_name, image_path, label_path, split_preview_dir in selected_items:
        split_preview_dir.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(image_path))
        if image is None:
            logger(f"[ERROR] Could not load image for preview: {image_path}")
            continue
        image_h, image_w = image.shape[:2]
        boxes = _load_yolo_boxes(label_path, image_w, image_h)
        preview = image.copy()

        if boxes:
            for class_id, (x1, y1, x2, y2) in boxes:
                class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
                cv2.rectangle(preview, (x1, y1), (x2, y2), (80, 190, 255), 2)
                cv2.putText(
                    preview,
                    class_name,
                    (x1 + 4, max(18, y1 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (80, 190, 255),
                    2,
                )
        else:
            cv2.putText(
                preview,
                "No detections",
                (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (120, 120, 255),
                2,
            )

        out_path = split_preview_dir / image_path.name
        cv2.imwrite(str(out_path), preview)
        written += 1
        selected_by_split[split_name] = selected_by_split.get(split_name, 0) + 1

    for split_name, _, _, _ in preview_sources:
        generated = selected_by_split.get(split_name, 0)
        available = available_by_split.get(split_name, 0)
        logger(
            f"Generated {generated} preview image(s) for {split_name} at {preview_dirs[split_name]} "
            f"({available} image(s) available)"
        )

    return written


def run_autolabel_job(job: Dict[str, object], logger: Callable[[str], None]) -> Path:
    project_dir = Path(str(job["project_dir"]))
    video_dir = Path(str(job["video_dir"]))
    frames_dir = Path(str(job["frames_dir"]))
    dataset_dir = Path(str(job["dataset_dir"]))
    stride = int(job["stride"])
    max_frames = int(job["max_frames"])
    max_preview_images = int(job.get("max_preview_images", 0))
    ext = str(job["ext"])
    ontology_text = str(job["ontology_text"])

    if ext not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported frame extension: {ext}")
    if not video_dir.exists():
        raise ValueError(f"Video folder not found: {video_dir}")

    project_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    videos = _list_videos(video_dir)
    if not videos:
        raise ValueError(f"No videos found in {video_dir}")

    ontology = _parse_ontology(ontology_text)
    class_names = list(dict.fromkeys(ontology.values()))

    logger("Initializing GroundingDINO. If model weights are missing, autodistill will download them automatically.")
    try:
        install_groundingdino_yapf_shim()
        # Import GroundingDINO first. In some Windows envs, importing CaptionOntology
        # ahead of torch-dependent groundingdino modules can poison the later torch DLL load.
        from autodistill_grounding_dino import GroundingDINO
        from autodistill.detection import CaptionOntology
    except Exception as exc:
        raise RuntimeError(format_grounding_dino_import_error(exc)) from exc

    logger(f"Extracting frames from {len(videos)} video(s) ...")
    total_frames = extract_frames(videos, frames_dir, stride=stride, ext=ext, max_frames=max_frames, logger=logger)
    logger(f"Extracted {total_frames} frame(s) into {frames_dir}")

    base_model = GroundingDINO(ontology=CaptionOntology(ontology))
    base_model.label(input_folder=str(frames_dir), extension=ext, output_folder=str(dataset_dir))
    logger(f"GroundingDINO autolabel complete: {dataset_dir}")

    _write_fallback_data_yaml(dataset_dir, class_names, logger)
    _write_label_provenance(
        dataset_dir,
        ontology=ontology,
        class_names=class_names,
        model_name="GroundingDINO",
        logger=logger,
    )
    preview_count = generate_preview_overlays(dataset_dir, class_names, logger, max_preview_images=max_preview_images)
    if preview_count:
        logger(f"Preview overlays saved under {dataset_dir / 'previews'}")
    return dataset_dir


def _write_label_provenance(
    dataset_dir: Path,
    *,
    ontology: Dict[str, str],
    class_names: list,
    model_name: str,
    logger: Callable[[str], None],
) -> None:
    """Write ``label_provenance.json`` next to a freshly auto-labelled dataset.

    Records that every label file currently in the dataset was generated by
    the model (not human-curated). Downstream merges / training loops can use
    this to distinguish model-suggested annotations from human ones — without
    it, the two are indistinguishable once they share the same YOLO label
    format.

    The sidecar is intended to be **immutable** for the dataset's lifetime as
    a model output. If a human later edits a label, the curation tool is
    responsible for updating its provenance entry to "model+human-edit"; the
    autolabel step only ever writes "model".
    """
    from datetime import datetime, timezone

    from integra_pose.utils.safe_io import safe_write_json

    label_files: list[dict] = []
    # Walk the YOLO layout. Autodistill's GroundingDINO produces
    # train/valid/test splits, each containing labels/ and images/. We record
    # every .txt under any labels/ directory.
    for split_dir in dataset_dir.rglob("labels"):
        if not split_dir.is_dir():
            continue
        rel_split = split_dir.relative_to(dataset_dir).as_posix()
        for label_path in sorted(split_dir.glob("*.txt")):
            label_files.append(
                {
                    "split": rel_split,
                    "label_file": label_path.name,
                    "provenance": "model",
                }
            )

    payload = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": str(model_name),
        "model_provenance_note": (
            "All label files listed below were produced by an automatic "
            "labelling model. They are not human-reviewed annotations. "
            "If you edit a label downstream, update its 'provenance' entry "
            "to 'model+human-edit' so the distinction is preserved."
        ),
        "ontology": dict(ontology or {}),
        "class_names": list(class_names or []),
        "labels": label_files,
    }
    try:
        safe_write_json(dataset_dir / "label_provenance.json", payload, indent=2)
        logger(
            f"Wrote label provenance manifest with {len(label_files)} entries: "
            f"{dataset_dir / 'label_provenance.json'}"
        )
    except Exception as exc:
        # Provenance is a reproducibility aid; an export failure should not
        # invalidate the labels themselves.
        logger(f"[WARN] Could not write label_provenance.json: {exc}")


__all__ = [
    "VIDEO_EXTS",
    "extract_frames",
    "format_grounding_dino_import_error",
    "generate_preview_overlays",
    "install_groundingdino_yapf_shim",
    "_iter_preview_sources",
    "load_job_from_path",
    "run_autolabel_job",
]

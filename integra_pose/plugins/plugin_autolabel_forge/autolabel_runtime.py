from __future__ import annotations

import json
import hashlib
import os
import re
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping
from uuid import uuid4


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".mpeg", ".mpg"}
EXTRACTION_MANIFEST_FILENAME = "frame_extraction_manifest.json"
JOB_MANIFEST_FILENAME = "autolabel_job_manifest.json"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def create_run_id() -> str:
    """Return a sortable, collision-resistant AutoLabel Forge run identifier."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid4().hex[:12]}"


def _validate_run_id(raw: object) -> str:
    run_id = str(raw or "").strip()
    if not _RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."} or ".." in run_id:
        raise ValueError(
            "AutoLabel Forge run_id must contain only letters, numbers, '.', '_' or '-', "
            "must not contain '..', and must be at most 80 characters."
        )
    return run_id


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_key(path: Path, content_sha256: str) -> str:
    canonical = os.path.normcase(str(path.expanduser().resolve()))
    payload = f"{canonical}\0{content_sha256}".encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()[:16]


def _safe_source_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-") or "video"
    return stem[:48]


def _write_json_required(path: Path, payload: Mapping[str, object]) -> None:
    from integra_pose.utils.safe_io import safe_write_json

    safe_write_json(path, dict(payload), indent=2)


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
    run_id: str | None = None,
) -> int:
    import cv2

    if stride < 1:
        raise ValueError("Frame extraction stride must be at least 1.")
    if max_frames < 0:
        raise ValueError("Maximum frames per video cannot be negative.")
    ext = str(ext).strip().lower()
    if ext not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported frame extension: {ext}")

    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing_entries = list(frames_dir.iterdir())
    if existing_entries:
        raise FileExistsError(
            f"Frame extraction output must be a fresh, empty directory: {frames_dir}. "
            "Use a new run directory instead of mixing frames from different jobs."
        )

    resolved_run_id = _validate_run_id(run_id or frames_dir.name)
    saved_total = 0
    source_records: list[dict[str, object]] = []
    for video_path in videos:
        video_path = Path(video_path).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Source video not found: {video_path}")

        source_sha256 = _file_sha256(video_path)
        source_id = _source_key(video_path, source_sha256)
        source_prefix = f"{_safe_source_stem(video_path)}__{source_id}"
        source_stat = video_path.stat()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            try:
                capture.release()
            except Exception:
                pass
            raise OSError(f"Could not open source video: {video_path}")

        frame_idx = 0
        saved_for_video = 0
        extracted_frames: list[dict[str, object]] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_idx % stride == 0:
                    out_name = f"{source_prefix}__frame_{frame_idx:06d}{ext}"
                    out_path = frames_dir / out_name
                    if out_path.exists():
                        raise FileExistsError(f"Refusing to overwrite extracted frame: {out_path}")
                    if not bool(cv2.imwrite(str(out_path), frame)):
                        raise OSError(
                            "OpenCV failed to write extracted frame "
                            f"{frame_idx} from {video_path} to {out_path}"
                        )
                    if not out_path.is_file():
                        raise OSError(
                            "OpenCV reported a successful frame write, but no output file exists: "
                            f"{out_path}"
                        )
                    extracted_frames.append(
                        {
                            "source_frame_index": frame_idx,
                            "image_file": out_name,
                            "image_size_bytes": out_path.stat().st_size,
                        }
                    )
                    saved_total += 1
                    saved_for_video += 1
                    if max_frames > 0 and saved_for_video >= max_frames:
                        break
                frame_idx += 1
        finally:
            capture.release()

        if saved_for_video == 0:
            raise OSError(f"No frames could be extracted from source video: {video_path}")

        source_records.append(
            {
                "source_id": source_id,
                "source_path": str(video_path),
                "source_name": video_path.name,
                "source_size_bytes": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "source_sha256": source_sha256,
                "extracted_frame_count": saved_for_video,
                "frames": extracted_frames,
            }
        )
        logger(f"Extracted {saved_for_video} frame(s) from {video_path.name}")

    manifest = {
        "schema_version": 1,
        "run_id": resolved_run_id,
        "created_at_utc": _utc_now(),
        "frame_index_base": 0,
        "frames_directory": str(frames_dir.resolve()),
        "parameters": {
            "stride": stride,
            "max_frames_per_video": max_frames,
            "image_extension": ext,
        },
        "source_count": len(source_records),
        "extracted_frame_count": saved_total,
        "sources": source_records,
    }
    _write_json_required(frames_dir / EXTRACTION_MANIFEST_FILENAME, manifest)
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
        if not bool(cv2.imwrite(str(out_path), preview)) or not out_path.is_file():
            logger(f"[ERROR] OpenCV could not write preview image: {out_path}")
            continue
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


def _create_run_directories(frames_root: Path, dataset_root: Path, run_id: str) -> tuple[Path, Path]:
    frames_root = frames_root.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    frames_run_dir = frames_root / "runs" / run_id
    dataset_run_dir = dataset_root / "runs" / run_id
    if frames_run_dir == dataset_run_dir:
        raise ValueError("Frame and dataset roots must be different directories.")

    frames_run_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_run_dir.parent.mkdir(parents=True, exist_ok=True)
    if frames_run_dir.exists() or dataset_run_dir.exists():
        raise FileExistsError(
            f"AutoLabel Forge run '{run_id}' already exists. Choose a new run_id; existing outputs "
            "will not be overwritten."
        )
    frames_run_dir.mkdir()
    try:
        dataset_run_dir.mkdir()
    except Exception:
        try:
            frames_run_dir.rmdir()
        except OSError:
            pass
        raise
    return frames_run_dir, dataset_run_dir


def _read_extraction_manifest(frames_run_dir: Path) -> dict[str, object]:
    manifest_path = frames_run_dir / EXTRACTION_MANIFEST_FILENAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read required extraction manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Extraction manifest must be a JSON object: {manifest_path}")
    return payload


def _manifest_frame_lookup(extraction_manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    for raw_source in extraction_manifest.get("sources", []):
        if not isinstance(raw_source, dict):
            continue
        for raw_frame in raw_source.get("frames", []):
            if not isinstance(raw_frame, dict):
                continue
            image_file = str(raw_frame.get("image_file") or "")
            if not image_file:
                continue
            image_stem = Path(image_file).stem
            if image_stem in lookup:
                raise RuntimeError(f"Extraction manifest contains duplicate image identity: {image_file}")
            lookup[image_stem] = {
                "extracted_image_file": image_file,
                "source_frame_index": int(raw_frame.get("source_frame_index", -1)),
                "source_id": str(raw_source.get("source_id") or ""),
                "source_video": str(raw_source.get("source_path") or ""),
                "source_video_sha256": str(raw_source.get("source_sha256") or ""),
            }
    return lookup


def _validate_labeled_dataset(dataset_dir: Path, extraction_manifest: Mapping[str, object]) -> None:
    expected = _manifest_frame_lookup(extraction_manifest)
    if not expected:
        raise RuntimeError("Extraction manifest contains no frame records; refusing to accept an empty dataset.")

    image_paths: list[Path] = []
    for images_dir in dataset_dir.rglob("images"):
        if images_dir.is_dir():
            image_paths.extend(
                path
                for path in images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
    actual_stems = [path.stem for path in image_paths]
    seen_stems: set[str] = set()
    duplicates: set[str] = set()
    for stem in actual_stems:
        if stem in seen_stems:
            duplicates.add(stem)
        seen_stems.add(stem)
    if duplicates:
        raise RuntimeError(
            "Autolabel output contains duplicate image identities across dataset splits: "
            + ", ".join(sorted(duplicates)[:5])
        )

    missing = sorted(set(expected) - set(actual_stems))
    unexpected = sorted(set(actual_stems) - set(expected))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing {len(missing)} extracted image(s)")
        if unexpected:
            details.append(f"found {len(unexpected)} image(s) not created by this run")
        raise RuntimeError(
            "GroundingDINO dataset output does not match the current extraction manifest: "
            + "; ".join(details)
        )

    for labels_dir in dataset_dir.rglob("labels"):
        if not labels_dir.is_dir():
            continue
        unexpected_labels = sorted(
            path.name
            for path in labels_dir.glob("*.txt")
            if path.stem not in expected and path.name.lower() not in {"classes.txt"}
        )
        if unexpected_labels:
            raise RuntimeError(
                "GroundingDINO produced label files that do not belong to this extraction run: "
                + ", ".join(unexpected_labels[:5])
            )


def run_autolabel_job(job: Dict[str, object], logger: Callable[[str], None]) -> Path:
    project_dir = Path(str(job["project_dir"]))
    video_dir = Path(str(job["video_dir"]))
    frames_root = Path(str(job["frames_dir"]))
    dataset_root = Path(str(job["dataset_dir"]))
    stride = int(job["stride"])
    max_frames = int(job["max_frames"])
    max_preview_images = int(job.get("max_preview_images", 0))
    ext = str(job["ext"]).strip().lower()
    ontology_text = str(job["ontology_text"])
    run_id = _validate_run_id(job.get("run_id") or create_run_id())

    if ext not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported frame extension: {ext}")
    if stride < 1:
        raise ValueError("Frame extraction stride must be at least 1.")
    if max_frames < 0:
        raise ValueError("Maximum frames per video cannot be negative.")
    if not video_dir.exists():
        raise ValueError(f"Video folder not found: {video_dir}")

    project_dir.mkdir(parents=True, exist_ok=True)

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

    frames_run_dir, dataset_run_dir = _create_run_directories(frames_root, dataset_root, run_id)
    state_dir = project_dir / ".autolabel_forge_runs"
    state_path = state_dir / f"{run_id}.json"
    job_manifest: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at_utc": _utc_now(),
        "project_directory": str(project_dir.expanduser().resolve()),
        "video_directory": str(video_dir.expanduser().resolve()),
        "frames_root": str(frames_root.expanduser().resolve()),
        "dataset_root": str(dataset_root.expanduser().resolve()),
        "frames_run_directory": str(frames_run_dir),
        "dataset_run_directory": str(dataset_run_dir),
        "model": "GroundingDINO",
        "ontology": ontology,
        "class_names": class_names,
        "parameters": {
            "stride": stride,
            "max_frames_per_video": max_frames,
            "max_preview_images": max_preview_images,
            "image_extension": ext,
        },
    }
    _write_json_required(state_path, job_manifest)

    try:
        logger(f"AutoLabel Forge run: {run_id}")
        logger(f"Extracting frames from {len(videos)} video(s) into isolated run directory ...")
        total_frames = extract_frames(
            videos,
            frames_run_dir,
            stride=stride,
            ext=ext,
            max_frames=max_frames,
            logger=logger,
            run_id=run_id,
        )
        logger(f"Extracted {total_frames} frame(s) into {frames_run_dir}")
        extraction_manifest = _read_extraction_manifest(frames_run_dir)

        base_model = GroundingDINO(ontology=CaptionOntology(ontology))
        base_model.label(
            input_folder=str(frames_run_dir),
            extension=ext,
            output_folder=str(dataset_run_dir),
        )
        _validate_labeled_dataset(dataset_run_dir, extraction_manifest)
        logger(f"GroundingDINO autolabel complete: {dataset_run_dir}")

        _write_fallback_data_yaml(dataset_run_dir, class_names, logger)
        _write_label_provenance(
            dataset_run_dir,
            ontology=ontology,
            class_names=class_names,
            model_name="GroundingDINO",
            logger=logger,
            extraction_manifest=extraction_manifest,
        )
        preview_count = generate_preview_overlays(
            dataset_run_dir,
            class_names,
            logger,
            max_preview_images=max_preview_images,
        )
        if preview_count:
            logger(f"Preview overlays saved under {dataset_run_dir / 'previews'}")

        _write_json_required(
            dataset_run_dir / EXTRACTION_MANIFEST_FILENAME,
            extraction_manifest,
        )
        job_manifest.update(
            {
                "status": "completed",
                "completed_at_utc": _utc_now(),
                "extracted_frame_count": total_frames,
                "preview_image_count": preview_count,
                "extraction_manifest": EXTRACTION_MANIFEST_FILENAME,
                "label_provenance_manifest": "label_provenance.json",
            }
        )
        _write_json_required(dataset_run_dir / JOB_MANIFEST_FILENAME, job_manifest)
        _write_json_required(state_path, job_manifest)
        return dataset_run_dir
    except Exception as exc:
        job_manifest.update(
            {
                "status": "failed",
                "completed_at_utc": _utc_now(),
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
        )
        try:
            _write_json_required(state_path, job_manifest)
        except Exception as manifest_exc:
            logger(f"[ERROR] Could not update failed-run manifest {state_path}: {manifest_exc}")
        raise


def _write_label_provenance(
    dataset_dir: Path,
    *,
    ontology: Dict[str, str],
    class_names: list,
    model_name: str,
    logger: Callable[[str], None],
    extraction_manifest: Mapping[str, object] | None = None,
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
    autolabel step only ever writes "model". A run is not reported as
    successful unless this required provenance sidecar can be written.
    """
    from integra_pose.utils.safe_io import safe_write_json

    label_files: list[dict] = []
    frame_lookup = _manifest_frame_lookup(extraction_manifest or {})
    # Walk the YOLO layout. Autodistill's GroundingDINO produces
    # train/valid/test splits, each containing labels/ and images/. We record
    # every .txt under any labels/ directory.
    for split_dir in dataset_dir.rglob("labels"):
        if not split_dir.is_dir():
            continue
        rel_split = split_dir.relative_to(dataset_dir).as_posix()
        for label_path in sorted(split_dir.glob("*.txt")):
            if label_path.name.lower() == "classes.txt":
                continue
            record = {
                "split": rel_split,
                "label_file": label_path.name,
                "provenance": "model",
            }
            source_record = frame_lookup.get(label_path.stem)
            if source_record is not None:
                record.update(source_record)
            label_files.append(record)

    payload = {
        "schema_version": 1,
        "recorded_at_utc": _utc_now(),
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
    safe_write_json(dataset_dir / "label_provenance.json", payload, indent=2)
    logger(
        f"Wrote label provenance manifest with {len(label_files)} entries: "
        f"{dataset_dir / 'label_provenance.json'}"
    )


__all__ = [
    "EXTRACTION_MANIFEST_FILENAME",
    "JOB_MANIFEST_FILENAME",
    "VIDEO_EXTS",
    "create_run_id",
    "extract_frames",
    "format_grounding_dino_import_error",
    "generate_preview_overlays",
    "install_groundingdino_yapf_shim",
    "_iter_preview_sources",
    "load_job_from_path",
    "run_autolabel_job",
]

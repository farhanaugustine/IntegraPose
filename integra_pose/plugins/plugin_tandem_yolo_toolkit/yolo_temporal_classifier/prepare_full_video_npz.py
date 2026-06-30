"""Prepare TandemYTC full-video training data from source videos using a
sliding window over the entire timeline.

Unlike clip-based preparation, this script:

  * Discovers source videos under a dataset root organized by split folders
    (e.g. ``data_root/train/video_001/``), **or** accepts a flat CSV manifest
    pointing to arbitrary video + annotation file paths.
  * Converts NorPix ``.seq`` → ``.mp4`` if needed (pass ``--source_mode seq``).
  * Parses BENTO ``.annot`` ground-truth files to build a per-frame label
    vector for every source video.
  * Runs YOLO-pose on every frame.
  * Emits sliding-window samples spanning the ENTIRE video, including bout
    boundaries and non-behavior context. HDF5 storage writes one compact
    per-video pose/social store; NPZ storage writes one pose/social file per
    window for debugging.
  * Labels each window by the dominant frame class (with a configurable
    minimum dominance threshold); optionally subsamples "other" windows to
    keep the class balance reasonable.
  * Writes a manifest in the ``yolo-temporal-full-video-v1`` schema that
    ``train_y.py`` consumes.

The motivation: clip-based training inflates validation scores because
temporal splits within the same video leak source-video context. Full-video
sliding-window training closes the train/inference distribution gap and
gives the model bout-boundary training signal.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import queue
import random
import re
import sys
import threading
import time
import warnings
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from numpy.lib import format as np_format
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings(
    "ignore", message=".*pynvml.*deprecated.*", category=FutureWarning,
)

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Local package imports — same modules as v4
# ---------------------------------------------------------------------------
if __package__ is None or __package__ == "":  # pragma: no cover - CLI entry
    THIS = Path(__file__).resolve().parent
    sys.path.insert(0, str(THIS))
    from cropping_y import NCropDetection, iterate_yolo_n_crops, load_yolo_model
    from data_y import load_hdf5_window_payload, load_n_manifest, validate_n_npz_payload
    from utils.logging_utils import setup_logger
    from utils.pose_features_y import REL_FEATURE_DIM
else:  # pragma: no cover
    from .cropping_y import NCropDetection, iterate_yolo_n_crops, load_yolo_model
    from .data_y import load_hdf5_window_payload, load_n_manifest, validate_n_npz_payload
    from .utils.logging_utils import setup_logger
    from .utils.pose_features_y import REL_FEATURE_DIM

# ---------------------------------------------------------------------------
# Dataset discovery helpers
# ---------------------------------------------------------------------------
DEFAULT_CLASSES = ["attack", "investigation", "mount", "other"]


def find_seq(video_dir: Path) -> Path | None:
    """Find the top-view .seq file in a video directory (NorPix format)."""
    preferred = sorted(video_dir.glob("*_Top_J85.seq"))
    if preferred:
        return preferred[0]
    preferred = sorted(video_dir.glob("*_Top.seq"))
    if preferred:
        return preferred[0]
    any_top = sorted(video_dir.glob("*Top*.seq"))
    if any_top:
        return any_top[0]
    any_seq = sorted(video_dir.glob("*.seq"))
    return any_seq[0] if any_seq else None


def find_annot(video_dir: Path) -> Path | None:
    """Find the human-annotated .annot file in a video directory."""
    annots = sorted(video_dir.glob("*.annot"))
    if not annots:
        return None
    filtered = [
        p for p in annots
        if "pred" not in p.name.lower()
        and "prediction" not in p.name.lower()
    ]
    if not filtered:
        return None
    priority_terms = ("bhvr", "behavior", "behav", "action", "anno")
    for term in priority_terms:
        matches = [p for p in filtered if term in p.name.lower()]
        if matches:
            return matches[0]
    return filtered[0]


def convert_seq_to_mp4(
    seq_path: Path,
    mp4_path: Path,
    *,
    seek_mat_path: Path | None,
    fps: float,
    overwrite: bool,
) -> dict:
    """Convert a NorPix .seq file to MP4 using the local SeqReader."""
    if mp4_path.is_file() and not overwrite:
        cap = cv2.VideoCapture(str(mp4_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        return {
            "converted": False,
            "mp4_path": str(mp4_path),
            "frames": frame_count,
            "width": width,
            "height": height,
        }

    from utils.seq_reader import SeqReader

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with SeqReader(seq_path) as reader:
        reader.build_seek_table(str(seek_mat_path) if seek_mat_path else None)
        n_frames = len(reader.seek_table or [])
        if n_frames <= 0:
            n_frames = int(reader.num_frames)
        width = int(reader.width)
        height = int(reader.height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {mp4_path}")
        try:
            for idx in range(n_frames):
                frame = reader.read_frame(idx)
                if frame.ndim == 2:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
                if idx and idx % 5000 == 0:
                    print(f"[convert] {seq_path.name}: {idx}/{n_frames} frames", flush=True)
        finally:
            writer.release()

    return {
        "converted": True,
        "mp4_path": str(mp4_path),
        "frames": n_frames,
        "width": width,
        "height": height,
        "elapsed_s": round(time.time() - t0, 3),
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "yolo-temporal-full-video-v1"
BUILD_KIND = "full_video_sliding"
DEFAULT_FPS = 30.0


def build_class_maps(class_names: Sequence[str]) -> tuple[list[str], dict[str, int]]:
    """Return validated class names and a class_to_idx mapping.

    Full-video training needs an explicit background class because unannotated
    frames are labelled as "other". GUI exports append it automatically, but
    command-line users may provide custom class names directly.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in class_names:
        name = str(raw).strip()
        if not name:
            continue
        if name in seen:
            raise SystemExit(f"[full-video] duplicate class name: {name!r}")
        cleaned.append(name)
        seen.add(name)
    if not cleaned:
        raise SystemExit("[full-video] at least one class name is required")
    if "other" not in seen:
        cleaned.append("other")
    return cleaned, {name: idx for idx, name in enumerate(cleaned)}


def load_class_names(args: argparse.Namespace) -> list[str]:
    if args.class_names_file is not None:
        path = Path(args.class_names_file)
        if not path.is_file():
            raise SystemExit(f"[full-video] class names file not found: {path}")
        names = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return names
    return list(args.class_names)


def parse_annot_for_classes(path: Path, class_to_idx: dict[str, int]) -> tuple[dict, list[dict], set[str]]:
    """Parse BENTO-style .annot files using the active class map.

    Unknown labels are reported so users can catch class-name mismatches while
    still allowing annotation files that contain channels not used here.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    header = {"fps": 30.0, "stop_time": None}
    m = re.search(r"Annotation framerate:\s*([\d.]+)", text)
    if m:
        header["fps"] = float(m.group(1))
    m = re.search(r"Annotation stop time:\s*([\d.eE+-]+)", text)
    if m:
        header["stop_time"] = float(m.group(1))

    bouts: list[dict] = []
    unknown: set[str] = set()
    blocks = re.split(r"\n(Ch\w+)-+\n", text)
    for i in range(1, len(blocks), 2):
        channel = blocks[i].strip()
        body = blocks[i + 1] if i + 1 < len(blocks) else ""
        for match in re.finditer(
            r">(\S+)\s*\nStart\s+Stop\s+Duration\s*\n((?:[\d.\s]+\n?)+)",
            body,
        ):
            behavior = match.group(1).strip()
            if behavior not in class_to_idx:
                unknown.add(behavior)
                continue
            if behavior == "other":
                continue
            for row in match.group(2).strip().splitlines():
                parts = row.split()
                if len(parts) < 2:
                    continue
                try:
                    start_s = float(parts[0])
                    stop_s = float(parts[1])
                except ValueError:
                    continue
                if stop_s <= start_s:
                    continue
                bouts.append(
                    {
                        "channel": channel,
                        "behavior": behavior,
                        "start_s": start_s,
                        "stop_s": stop_s,
                    }
                )
    return header, bouts, unknown


def expand_gt_to_frames_for_classes(
    bouts: Sequence[dict],
    n_frames: int,
    fps: float,
    class_to_idx: dict[str, int],
    class_names: Sequence[str],
) -> np.ndarray:
    out = np.full(n_frames, class_to_idx["other"], dtype=np.int16)
    # Write non-background classes in class order. Later classes overwrite
    # earlier classes when annotations overlap.
    for behavior in [name for name in class_names if name != "other"]:
        idx = class_to_idx[behavior]
        for bout in bouts:
            if bout["behavior"] != behavior:
                continue
            start = max(0, int(round(float(bout["start_s"]) * fps)))
            end = min(n_frames, int(round(float(bout["stop_s"]) * fps)))
            if end > start:
                out[start:end] = idx
    return out


# ---------------------------------------------------------------------------
# Async NPZ writer used by the full-video preparation pipeline.
# ---------------------------------------------------------------------------
class _AsyncNPZWriter:
    def __init__(
        self,
        num_workers: int = 3,
        *,
        compression: str = "deflated",
        compresslevel: int = 1,
    ) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._error: Optional[Exception] = None
        self._compression = str(compression)
        self._compresslevel = int(compresslevel)
        self._workers = [
            threading.Thread(target=self._run, daemon=True, name=f"npz-writer-{i}")
            for i in range(max(1, num_workers))
        ]
        for w in self._workers:
            w.start()

    def submit(self, path: Path, payload: dict) -> None:
        if self._error is not None:
            raise self._error
        self._q.put((path, payload))

    def flush(self) -> None:
        self._q.join()
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self.flush()
        for _ in self._workers:
            self._q.put(None)
        for w in self._workers:
            w.join()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                path, payload = item
                path.parent.mkdir(parents=True, exist_ok=True)
                save_npz_payload(
                    path,
                    payload,
                    compression=self._compression,
                    compresslevel=self._compresslevel,
                )
            except Exception as exc:  # noqa: BLE001
                self._error = exc
            finally:
                self._q.task_done()


def save_npz_payload(
    path: Path,
    payload: dict,
    *,
    compression: str = "deflated",
    compresslevel: int = 1,
) -> None:
    """Write an NPZ payload with explicit compression control.

    `np.savez_compressed` uses zlib's default compression, which is often too
    slow for hundreds of thousands of small windows. Deflate level 1 keeps
    files compressed while reducing wall time substantially on typical CPUs.
    """
    path = Path(path)
    if compression == "stored":
        np.savez(path, **payload)
        return
    zip_compression = zipfile.ZIP_DEFLATED
    level = max(0, min(9, int(compresslevel)))
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zip_compression,
        compresslevel=level,
        allowZip64=True,
    ) as zf:
        for key, value in payload.items():
            if not re.match(r"^[A-Za-z0-9_]+$", str(key)):
                raise ValueError(f"Invalid NPZ key: {key!r}")
            with zf.open(f"{key}.npy", mode="w", force_zip64=True) as fh:
                np_format.write_array(fh, np.asanyarray(value), allow_pickle=False)


class _HDF5VideoStore:
    def __init__(self, path: Path, args, *, class_names: Sequence[str], fps: float) -> None:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - dependency should be installed
            raise RuntimeError("h5py is required for --storage_mode hdf5.") from exc
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self.h5 = h5py.File(self.path, "w")
        self.h5.attrs["schema_version"] = SCHEMA_VERSION
        self.h5.attrs["storage_format"] = "hdf5"
        self.h5.attrs["body_length_px"] = float(args.body_length_px or 0.0)
        self.h5.attrs["n_animals"] = int(args.n_animals)
        self.h5.attrs["fps"] = float(fps)
        self.h5.attrs["class_names_json"] = json.dumps(list(class_names))
        self.count = 0
        self.capacity = 0
        self.block_size = 512
        self.datasets: dict[str, object] = {}
        self.compression = None if str(args.hdf5_compression).lower() == "none" else str(args.hdf5_compression)
        self.compression_opts = None
        if self.compression == "gzip":
            self.compression_opts = max(0, min(9, int(args.hdf5_compresslevel)))

    def _frame_payload(self, det: NCropDetection) -> dict[str, np.ndarray]:
        pose_conf = np.asarray(det.pose_conf, dtype=np.float32)
        return {
            "animal_keypoints": np.asarray(det.keypoints_crop_norm, dtype=np.float32),
            "animal_keypoints_raw": np.asarray(det.keypoints_xy_raw, dtype=np.float32),
            "animal_keypoints_conf": np.asarray(det.keypoints_conf_raw, dtype=np.float32),
            "animal_mask": np.asarray(det.animal_mask, dtype=bool),
            "pose_mask": np.asarray(det.pose_mask, dtype=bool),
            "pose_conf": pose_conf,
            "crop_conf": np.asarray(det.crop_conf, dtype=np.float32),
            "track_conf": np.asarray(det.track_conf, dtype=np.float32),
            "track_age": np.asarray(det.track_age, dtype=np.float32),
            "relation_features": np.asarray(det.relation_features, dtype=np.float32),
            "relation_pose_mask": np.asarray(det.relation_pose_mask, dtype=bool),
            "relation_pose_conf": (pose_conf[:, None] * pose_conf[None, :]).astype(np.float32),
            "relation_present": np.asarray(det.relation_present, dtype=bool),
            "bbox_xyxy_animals": np.asarray(det.bbox_xyxy_animals, dtype=np.int32),
            "bbox_xyxy_group": np.asarray(det.bbox_xyxy_group, dtype=np.int32),
            "crop_xyxy_animals": np.asarray(det.crop_xyxy_animals, dtype=np.int32),
            "crop_xyxy_group": np.asarray(det.crop_xyxy_group, dtype=np.int32),
            "track_ids": np.asarray(det.track_ids, dtype=np.int32),
        }

    def _ensure_capacity(self, payload: dict[str, np.ndarray]) -> None:
        if self.count < self.capacity:
            return
        new_capacity = max(self.block_size, self.capacity + self.block_size)
        for name, arr in payload.items():
            if name not in self.datasets:
                self.datasets[name] = self.h5.create_dataset(
                    name,
                    shape=(new_capacity, *arr.shape),
                    maxshape=(None, *arr.shape),
                    chunks=(1, *arr.shape),
                    dtype=arr.dtype,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                    shuffle=(self.compression is not None),
                )
            else:
                self.datasets[name].resize((new_capacity, *arr.shape))
        self.capacity = new_capacity

    def append(self, det: NCropDetection) -> None:
        payload = self._frame_payload(det)
        self._ensure_capacity(payload)
        for name, arr in payload.items():
            self.datasets[name][self.count] = arr
        self.count += 1

    def close(self) -> None:
        for ds in self.datasets.values():
            ds.resize((self.count, *ds.shape[1:]))
        self.h5.attrs["frame_count"] = int(self.count)
        self.h5.close()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
@dataclass
class VideoSource:
    split: str         # dataset split this video belongs to: 'train' or 'val'
    source_split: str  # which top-level dir it came from (e.g., 'train', 'validation')
    video_id: str      # source video identifier (folder name)
    video_dir: Path    # path to the source video folder
    seq_path: Optional[Path]     # path to .seq file (NorPix format) if applicable
    annot_path: Path   # path to .annot file
    source_path: Optional[Path] = None  # direct user video path from source_manifest_csv
    fps_override: Optional[float] = None
    mp4_path: Optional[Path] = None  # set after conversion


def discover_sources(
    data_root: Path,
    train_splits: Sequence[str],
    val_splits: Sequence[str],
    exclude_splits: Sequence[str],
    logger,
) -> List[VideoSource]:
    """Walk the dataset root and discover all (video, annot) pairs in the
    requested train + val split folders. Refuse to touch exclude_splits.

    Expected layout::

        data_root/
        ├── train/
        │   ├── video_001/   (contains .seq or .mp4 + .annot)
        │   └── video_002/
        └── validation/
            └── ...
    """
    sources: List[VideoSource] = []
    excluded_seen = 0
    excluded = {str(s).strip() for s in exclude_splits}

    for dataset_split, split_name_list in (("train", train_splits), ("val", val_splits)):
        for split_name in split_name_list:
            if str(split_name).strip() in excluded:
                raise SystemExit(
                    f"[discover] FATAL: requested split {split_name!r} is listed in "
                    f"--exclude_splits={sorted(excluded)!r}. Held-out splits cannot be "
                    "used for full-video NPZ construction."
                )
            split_dir = data_root / split_name
            if not split_dir.is_dir():
                logger.warning(f"[discover] split dir missing: {split_dir}")
                continue
            for vid_dir in sorted(split_dir.iterdir()):
                if not vid_dir.is_dir():
                    continue
                seq = find_seq(vid_dir)
                annot = find_annot(vid_dir)
                if seq is None or annot is None:
                    logger.info(f"[discover] skip {split_name}/{vid_dir.name}  "
                                f"seq={seq is not None} annot={annot is not None}")
                    continue
                sources.append(VideoSource(
                    split=dataset_split, source_split=split_name,
                    video_id=vid_dir.name, video_dir=vid_dir,
                    seq_path=seq, annot_path=annot,
                ))

    # Strict leakage guard: refuse to touch exclude_splits dirs
    for excl in exclude_splits:
        excl_dir = data_root / excl
        if excl_dir.is_dir():
            excluded_seen += sum(1 for _ in excl_dir.iterdir() if _.is_dir())

    # Strict source-grouping check: no source video should appear in both train
    # and val splits. (Possible if user passes overlapping
    # split lists like --train_splits train --val_splits train.)
    by_id = defaultdict(set)
    for s in sources:
        by_id[s.video_id].add(s.split)
    overlap = [vid for vid, splits in by_id.items() if len(splits) > 1]
    if overlap:
        raise SystemExit(
            f"[discover] FATAL: {len(overlap)} source videos appear in BOTH "
            f"train and val splits. Use disjoint --train_splits / --val_splits. "
            f"Examples: {overlap[:5]}"
        )

    n_train = sum(1 for s in sources if s.split == "train")
    n_val = sum(1 for s in sources if s.split == "val")
    logger.info(f"[discover] train sources: {n_train}  |  val sources: {n_val}  "
                f"|  excluded (held-out test) dirs intentionally untouched: {excluded_seen}")
    return sources


def discover_sources_from_manifest(
    manifest_csv: Path,
    exclude_splits: Sequence[str],
    logger,
) -> List[VideoSource]:
    """Load user-video sources from a CSV exported by the annotation GUI.

    Required columns: split, video_path, annot_path.
    Optional columns: video_id, source_split, fps.
    """
    manifest_csv = Path(manifest_csv)
    if not manifest_csv.is_file():
        raise SystemExit(f"[manifest] source manifest CSV not found: {manifest_csv}")
    excluded = {str(s).strip() for s in exclude_splits}
    sources: list[VideoSource] = []
    with manifest_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"split", "video_path", "annot_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"[manifest] {manifest_csv} missing required columns: {sorted(missing)}"
            )
        for row_i, row in enumerate(reader, start=2):
            split = str(row.get("split") or "").strip().lower()
            split_aliases = {"validation": "val", "valid": "val"}
            split = split_aliases.get(split, split)
            if split in excluded:
                raise SystemExit(
                    f"[manifest] row {row_i}: split {split!r} is excluded by --exclude_splits"
                )
            if split not in {"train", "val"}:
                raise SystemExit(
                    f"[manifest] row {row_i}: split must be 'train' or 'val', got {split!r}"
                )
            video_path = Path(str(row.get("video_path") or "").strip())
            annot_path = Path(str(row.get("annot_path") or "").strip())
            if not video_path.is_absolute():
                video_path = manifest_csv.parent / video_path
            if not annot_path.is_absolute():
                annot_path = manifest_csv.parent / annot_path
            if not video_path.is_file():
                raise SystemExit(f"[manifest] row {row_i}: video_path not found: {video_path}")
            if not annot_path.is_file():
                raise SystemExit(f"[manifest] row {row_i}: annot_path not found: {annot_path}")
            video_id = str(row.get("video_id") or video_path.stem).strip() or video_path.stem
            fps_override = None
            if str(row.get("fps") or "").strip():
                try:
                    fps_override = float(str(row.get("fps")).strip())
                except ValueError as exc:
                    raise SystemExit(
                        f"[manifest] row {row_i}: fps must be numeric, got {row.get('fps')!r}"
                    ) from exc
            sources.append(
                VideoSource(
                    split=split,
                    source_split=str(row.get("source_split") or split),
                    video_id=video_id,
                    video_dir=video_path.parent,
                    seq_path=None,
                    annot_path=annot_path,
                    source_path=video_path,
                    fps_override=fps_override,
                )
            )

    by_id = defaultdict(set)
    for s in sources:
        by_id[s.video_id].add(s.split)
    overlap = [vid for vid, splits in by_id.items() if len(splits) > 1]
    if overlap:
        raise SystemExit(
            f"[manifest] FATAL: {len(overlap)} source videos appear in BOTH train and val. "
            f"Examples: {overlap[:5]}"
        )
    logger.info(
        f"[manifest] loaded user-video sources from {manifest_csv}: "
        f"train={sum(1 for s in sources if s.split == 'train')} "
        f"val={sum(1 for s in sources if s.split == 'val')}"
    )
    return sources


# ---------------------------------------------------------------------------
# Window selection + labelling
# ---------------------------------------------------------------------------
def emit_window_starts(
    n_frames: int, window_size: int, window_stride: int,
) -> List[int]:
    """Sliding-window start indices [0, stride, 2*stride, ...] such that the
    last window ends at or before n_frames."""
    if n_frames < window_size:
        return []
    last_start = n_frames - window_size
    return list(range(0, last_start + 1, window_stride))


def window_label(
    label_vec: np.ndarray,
    t: int,
    window_size: int,
    min_dominance: float,
    class_names: Sequence[str],
) -> Optional[str]:
    """Return the dominant class for the window if its share of frames is
    >= min_dominance, else None (window will be skipped)."""
    seg = label_vec[t:t + window_size]
    if len(seg) == 0:
        return None
    counts = np.bincount(seg.astype(np.int64), minlength=len(class_names))
    top_count = int(counts.max())
    winners = np.where(counts == top_count)[0]
    if len(winners) > 1:
        # Do not manufacture a class label for exact boundary ties.
        return None
    top_idx = int(winners[0])
    top_share = top_count / len(seg)
    if top_share < min_dominance:
        return None
    return class_names[top_idx]


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------
def _safe_stem(text: str, max_len: int = 200) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    return (clean[:max_len] or "video")


def _build_payload(frames: List[NCropDetection], args) -> dict:
    """Stack a list of per-frame detections into the NPZ payload dict."""
    animal_keypoints = np.stack([e.keypoints_crop_norm for e in frames], axis=0)
    animal_keypoints_raw = np.stack([e.keypoints_xy_raw for e in frames], axis=0)
    animal_keypoints_conf = np.stack([e.keypoints_conf_raw for e in frames], axis=0)
    animal_mask = np.stack([e.animal_mask for e in frames], axis=0)
    pose_mask = np.stack([e.pose_mask for e in frames], axis=0)
    pose_conf = np.stack([e.pose_conf for e in frames], axis=0)
    crop_conf = np.stack([e.crop_conf for e in frames], axis=0)
    track_conf = np.stack([e.track_conf for e in frames], axis=0)
    track_age = np.stack([e.track_age for e in frames], axis=0)
    relation_features = np.stack([e.relation_features for e in frames], axis=0)
    relation_pose_mask = np.stack([e.relation_pose_mask for e in frames], axis=0)
    relation_present = np.stack([e.relation_present for e in frames], axis=0)
    bbox_xyxy_animals = np.stack([e.bbox_xyxy_animals for e in frames], axis=0)
    bbox_xyxy_group = np.asarray([e.bbox_xyxy_group for e in frames], dtype=np.int32)
    crop_xyxy_animals = np.stack([e.crop_xyxy_animals for e in frames], axis=0)
    crop_xyxy_group = np.asarray([e.crop_xyxy_group for e in frames], dtype=np.int32)
    track_ids = np.stack([e.track_ids for e in frames], axis=0)

    return {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "animal_keypoints": animal_keypoints.astype(np.float32),
        "animal_keypoints_raw": animal_keypoints_raw.astype(np.float32),
        "animal_keypoints_conf": animal_keypoints_conf.astype(np.float32),
        "animal_mask": animal_mask.astype(bool),
        "pose_mask": pose_mask.astype(bool),
        "pose_conf": pose_conf.astype(np.float32),
        "crop_conf": crop_conf.astype(np.float32),
        "track_conf": track_conf.astype(np.float32),
        "track_age": track_age.astype(np.float32),
        "relation_features": relation_features.astype(np.float32),
        "relation_pose_mask": relation_pose_mask.astype(bool),
        "relation_pose_conf": (pose_conf[:, :, None] * pose_conf[:, None, :]).astype(np.float32),
        "relation_present": relation_present.astype(bool),
        "bbox_xyxy_animals": bbox_xyxy_animals.astype(np.int32),
        "bbox_xyxy_group": bbox_xyxy_group.astype(np.int32),
        "crop_xyxy_animals": crop_xyxy_animals.astype(np.int32),
        "crop_xyxy_group": crop_xyxy_group.astype(np.int32),
        "track_ids": track_ids.astype(np.int32),
        "body_length_px": np.float32(args.body_length_px or 0.0),
        "n_animals": np.int32(args.n_animals),
    }


def _video_already_done(seq_dir: Path, safe_stem: str) -> int:
    """Return the count of NPZ files for this video already on disk (across
    all class subdirs). Used by --skip_existing as a fast per-video resume check."""
    n = 0
    if not seq_dir.exists():
        return 0
    for cls_dir in seq_dir.iterdir():
        if cls_dir.is_dir():
            n += sum(1 for _ in cls_dir.glob(f"{safe_stem}_*.npz"))
    return n


def _stable_video_seed(seed: int, video_id: str) -> int:
    digest = hashlib.blake2b(str(video_id).encode("utf-8"), digest_size=8).hexdigest()
    return (int(seed) + int(digest, 16)) % (2**32)


def _done_marker_path(output_root: Path, safe_stem: str) -> Path:
    return Path(output_root) / "build_state" / "completed_videos" / f"{safe_stem}.json"


def _video_build_signature(src: VideoSource, args, class_names: Sequence[str]) -> dict:
    source_path = src.source_path or src.seq_path
    return {
        "video_id": src.video_id,
        "source_path": str(Path(source_path).resolve()) if source_path else "",
        "annot_path": str(Path(src.annot_path).resolve()),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "label_min_dominance": float(args.label_min_dominance),
        "other_subsample": float(args.other_subsample),
        "seed": int(args.seed),
        "fps_override": None if src.fps_override is None else float(src.fps_override),
        "class_names": list(class_names),
        "n_animals": int(args.n_animals),
        "crop_size": int(args.crop_size),
        "animal_scale_factor": float(args.animal_scale_factor),
        "group_scale_factor": float(args.group_scale_factor),
        "body_length_px": None if args.body_length_px is None else float(args.body_length_px),
        "yolo_weights": str(Path(args.yolo_weights).resolve()),
        "yolo_conf": float(args.yolo_conf),
        "yolo_iou": float(args.yolo_iou),
        "yolo_imgsz": int(args.yolo_imgsz),
        "source_mode": str(args.source_mode),
        "pose_conf_threshold": float(args.pose_conf_threshold),
        "keep_last_box": bool(args.keep_last_box),
        "storage_mode": str(args.storage_mode),
        "storage_payload": "pose_social_only_v1",
    }


def _load_done_marker(
    output_root: Path,
    safe_stem: str,
    src: VideoSource,
    args,
    class_names: Sequence[str],
    logger,
) -> tuple[list[dict], Counter] | None:
    marker_path = _done_marker_path(output_root, safe_stem)
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"[{src.video_id}] ignoring unreadable done marker {marker_path}: {exc}")
        return None
    expected = _video_build_signature(src, args, class_names)
    if marker.get("signature") != expected:
        logger.info(f"[{src.video_id}] done marker exists but build parameters changed; rebuilding")
        return None
    samples = list(marker.get("samples") or [])
    base = Path(output_root)
    missing = []
    for sample in samples:
        rel = sample.get("sequence_h5") or sample.get("sequence_npz") or ""
        if not rel or not (base / str(rel)).is_file():
            missing.append(rel)
    if missing:
        logger.warning(
            f"[{src.video_id}] done marker has {len(missing)} missing sample-store files; rebuilding. "
            f"First missing: {missing[0]}"
        )
        return None
    cls_counts = Counter(s.get("class_name", "") for s in samples)
    cls_counts.pop("", None)
    return samples, cls_counts


def _write_done_marker(
    output_root: Path,
    safe_stem: str,
    src: VideoSource,
    args,
    class_names: Sequence[str],
    samples: list[dict],
    cls_counts: Counter,
) -> None:
    marker_path = _done_marker_path(output_root, safe_stem)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "behaviorscope-y-full-video-build-v1",
        "completed_at_unix": time.time(),
        "signature": _video_build_signature(src, args, class_names),
        "n_samples": len(samples),
        "class_counts": dict(cls_counts),
        "samples": samples,
    }
    marker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _delete_existing_video_npzs(seq_dir: Path, safe_stem: str) -> int:
    if not seq_dir.exists():
        return 0
    deleted = 0
    for cls_dir in seq_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        for npz in cls_dir.glob(f"{safe_stem}_*.npz"):
            npz.unlink()
            deleted += 1
    return deleted


def _load_existing_samples_for_video(
    seq_dir: Path, safe_stem: str, src: VideoSource, args, fps: float,
) -> Tuple[List[dict], Counter]:
    """When --skip_existing is on and a video already has NPZs on disk, rebuild
    the manifest entries from the filenames without re-running YOLO."""
    samples: List[dict] = []
    cls_counts: Counter = Counter()
    class_to_idx = dict(getattr(args, "class_to_idx_resolved", {}))
    if not seq_dir.exists():
        return samples, cls_counts
    pat = re.compile(r"^(?P<stem>.+)_f(?P<frame>\d+)\.npz$")
    for cls_dir in seq_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        for npz in cls_dir.glob(f"{safe_stem}_*.npz"):
            m = pat.match(npz.name)
            if not m:
                continue
            t = int(m.group("frame"))
            seq_name = npz.stem
            samples.append({
                "id": f"{cls_name}_{seq_name}",
                "label": int(class_to_idx[cls_name]),
                "class_name": cls_name,
                "sequence_npz": str(npz.relative_to(seq_dir.parent)).replace("\\", "/"),
                "start_frame": int(t),
                "end_frame": int(t + args.window_size - 1),
                "fps": float(fps),
                "source_video": src.video_id,
                "source_split": src.source_split,
                "dataset_split": src.split,
            })
            cls_counts[cls_name] += 1
    return samples, cls_counts


def process_video(
    src: VideoSource, model, args, output_root: Path, writer: _AsyncNPZWriter | None,
    logger,
) -> Tuple[List[dict], Counter]:
    """Process one source video end-to-end with **streaming windows**:
       * convert .seq -> .mp4 if needed
       * parse .annot -> per-frame label
       * stream YOLO detections; emit windows on the fly using a deque buffer
         of size `window_size` (so memory is O(window_size), not O(n_frames))
       * write NPZs asynchronously

    Memory: peak ~window_size frames buffered (e.g. 32 * 450 KB ≈ 14 MB),
    not the entire video.

    Returns (sample_records, class_counts)."""
    class_names = list(args.class_names_resolved)
    class_to_idx = dict(args.class_to_idx_resolved)
    rng = random.Random(_stable_video_seed(args.seed, src.video_id))
    fps = float(src.fps_override or args.fps)
    safe_stem = _safe_stem(src.video_id)
    storage_mode = str(args.storage_mode).lower()
    seq_dir = output_root / "sequence_npz"
    if storage_mode == "npz":
        seq_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume support: skip a video whose windows already exist on disk ----
    if args.skip_existing:
        existing = _load_done_marker(output_root, safe_stem, src, args, class_names, logger)
        if existing is not None:
            samples, cls_counts = existing
            logger.info(
                f"[{src.video_id}] skip_existing: completed marker found; "
                f"reusing {len(samples)} manifest entries"
            )
            return samples, cls_counts

    if storage_mode == "npz":
        n_existing = _video_already_done(seq_dir, safe_stem)
        if n_existing and args.clean_incomplete_existing:
            deleted = _delete_existing_video_npzs(seq_dir, safe_stem)
            logger.warning(
                f"[{src.video_id}] removed {deleted} stale NPZ files without a matching "
                "completed-video marker before rebuilding"
            )

    # ---- 1. ensure .mp4 exists ----
    if src.source_path is not None:
        src.mp4_path = src.source_path
        if src.mp4_path.suffix.lower() == ".seq":
            seq_path = src.mp4_path
            from utils.seq_reader import SeqReader

            with SeqReader(seq_path) as reader:
                n_frames_mp4 = int(reader.num_frames)
                probed_fps = float(reader.header.get("frame_rate") or 0.0)
            if src.fps_override is None and probed_fps > 1e-6:
                fps = probed_fps
            cache_mp4 = output_root / "source_video_cache" / f"{safe_stem}.mp4"
            info = convert_seq_to_mp4(
                seq_path,
                cache_mp4,
                seek_mat_path=None,
                fps=fps,
                overwrite=False,
            )
            src.mp4_path = cache_mp4
            n_frames_mp4 = int(info.get("frames", n_frames_mp4) or n_frames_mp4)
            logger.info(f"[{src.video_id}] converted/using cached MP4 for .seq source: {cache_mp4}")
        else:
            cap = cv2.VideoCapture(str(src.mp4_path))
            n_frames_mp4 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            probed_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
            if src.fps_override is None and probed_fps > 1e-6:
                fps = probed_fps
    elif args.source_mode == "mp4":
        if src.seq_path is None:
            raise RuntimeError(f"{src.video_id}: source_mode=mp4 requires a seq_path or source_path")
        mp4_path = src.video_dir / f"{src.seq_path.stem}.mp4"
        info = convert_seq_to_mp4(
            src.seq_path, mp4_path, seek_mat_path=None,
            fps=fps, overwrite=False,
        )
        src.mp4_path = mp4_path
        n_frames_mp4 = info.get("frames", 0) or 0
    else:
        if src.seq_path is None:
            raise RuntimeError(f"{src.video_id}: source_mode=seq requires seq_path")
        seq_path = src.seq_path
        from utils.seq_reader import SeqReader

        with SeqReader(seq_path) as reader:
            n_frames_mp4 = int(reader.num_frames)
            probed_fps = float(reader.header.get("frame_rate") or 0.0)
        if src.fps_override is None and probed_fps > 1e-6:
            fps = probed_fps
        cache_mp4 = output_root / "source_video_cache" / f"{safe_stem}.mp4"
        info = convert_seq_to_mp4(
            seq_path,
            cache_mp4,
            seek_mat_path=None,
            fps=fps,
            overwrite=False,
        )
        src.mp4_path = cache_mp4
        n_frames_mp4 = int(info.get("frames", n_frames_mp4) or n_frames_mp4)
        logger.info(f"[{src.video_id}] converted/using cached MP4 for .seq source: {cache_mp4}")

    if n_frames_mp4 <= 0:
        logger.warning(f"[{src.video_id}] no readable frames; skipping")
        return [], Counter()

    # ---- 2. parse .annot -> per-frame label vector ----
    header, bouts, unknown_labels = parse_annot_for_classes(src.annot_path, class_to_idx)
    if unknown_labels:
        logger.warning(
            f"[{src.video_id}] ignored annotation labels not in class_names: "
            f"{sorted(unknown_labels)}"
        )
    annot_fps = float(header.get("fps") or fps)
    label_vec = expand_gt_to_frames_for_classes(
        bouts, n_frames_mp4, annot_fps, class_to_idx, class_names
    )

    # ---- 3. STREAMING window emission ----
    # Maintain a small deque of the last `window_size` detections, indexed by
    # their frame_idx. Whenever we have window_size consecutive frames covering
    # [next_window_t, next_window_t + window_size - 1], emit/label/write the
    # window and advance next_window_t by window_stride. This keeps RAM
    # bounded to window_size frames regardless of total video length.
    detections_iter = iterate_yolo_n_crops(
        model=model,
        video_path=src.mp4_path,
        n_animals=int(args.n_animals),
        crop_size=int(args.crop_size),
        animal_scale_factor=float(args.animal_scale_factor),
        group_scale_factor=float(args.group_scale_factor),
        body_length_px=args.body_length_px,
        crop_strategy="fixed_scale",
        conf_threshold=float(args.yolo_conf),
        iou_threshold=float(args.yolo_iou),
        imgsz=int(args.yolo_imgsz),
        device=str(args.device),
        keep_last_box=bool(args.keep_last_box),
        pose_conf_threshold=float(args.pose_conf_threshold),
        fps=fps,
        batch_size=int(args.yolo_batch),
    )

    # Buffer holds (frame_idx, NCropDetection) for the most recent
    # window_size+stride frames so we can both emit the current window and
    # have enough lookahead for the next stride step.
    buffer: deque = deque()
    next_window_t = 0
    samples: List[dict] = []
    cls_counts: Counter = Counter()
    n_dropped_dominance = 0
    n_dropped_subsample = 0
    n_emitted = 0
    n_skipped_missing = 0
    h5_store: _HDF5VideoStore | None = None
    h5_rel = ""
    if storage_mode == "hdf5":
        h5_path = output_root / "sequence_h5" / f"{safe_stem}.h5"
        h5_store = _HDF5VideoStore(h5_path, args, class_names=class_names, fps=fps)
        h5_rel = str(h5_path.relative_to(output_root)).replace("\\", "/")

    def _try_emit_windows() -> None:
        """Emit any windows for which we now have all required frames buffered."""
        nonlocal next_window_t, n_emitted, n_dropped_dominance, n_dropped_subsample, n_skipped_missing
        while buffer:
            t = next_window_t
            t_end = t + args.window_size - 1
            # Window not yet complete
            if buffer[-1][0] < t_end:
                return
            # Drop frames before t (no longer needed)
            while buffer and buffer[0][0] < t:
                buffer.popleft()
            # Collect contiguous frames [t .. t_end]
            window_frames: List[NCropDetection] = []
            ok = True
            expected = t
            for fi, det in buffer:
                if fi > t_end:
                    break
                if fi != expected:
                    ok = False
                    break
                window_frames.append(det)
                expected += 1
            if not ok or len(window_frames) != args.window_size:
                n_skipped_missing += 1
                next_window_t += args.window_stride
                continue

            cls = window_label(
                label_vec,
                t,
                args.window_size,
                args.label_min_dominance,
                class_names,
            )
            if cls is None:
                n_dropped_dominance += 1
                next_window_t += args.window_stride
                continue
            if cls == "other" and rng.random() > args.other_subsample:
                n_dropped_subsample += 1
                next_window_t += args.window_stride
                continue

            seq_name = f"{safe_stem}_f{t:06d}"
            sample_record = {
                "id": f"{cls}_{seq_name}",
                "label": int(class_to_idx[cls]),
                "class_name": cls,
                "start_frame": int(t),
                "end_frame": int(t + args.window_size - 1),
                "fps": float(fps),
                "source_video": src.video_id,
                "source_split": src.source_split,
                "dataset_split": src.split,
            }
            if h5_store is not None:
                sample_record.update({
                    "storage_format": "hdf5",
                    "sequence_h5": h5_rel,
                })
            else:
                if writer is None:
                    raise RuntimeError("NPZ storage selected but NPZ writer is not available.")
                npz_path = seq_dir / cls / f"{seq_name}.npz"
                payload = _build_payload(window_frames, args)
                writer.submit(npz_path, payload)
                sample_record.update({
                    "storage_format": "npz",
                    "sequence_npz": str(npz_path.relative_to(output_root)).replace("\\", "/"),
                })
            samples.append(sample_record)
            cls_counts[cls] += 1
            n_emitted += 1
            next_window_t += args.window_stride

    # Stream + emit
    try:
        for det in detections_iter:
            if h5_store is not None:
                h5_store.append(det)
            buffer.append((int(det.frame_idx), det))
            # Trim buffer aggressively: only keep frames we still might need
            # (i.e. frame_idx >= next_window_t). Since stride <= window_size,
            # the buffer never needs more than window_size + stride frames.
            while buffer and buffer[0][0] < next_window_t:
                buffer.popleft()
            _try_emit_windows()

        # Drain any final windows that became complete with the last frames
        _try_emit_windows()
    finally:
        if h5_store is not None:
            h5_store.close()

    if not samples:
        logger.warning(
            f"[{src.video_id}] no windows emitted "
            f"(dropped_dom={n_dropped_dominance}, dropped_other_sub={n_dropped_subsample}, "
            f"missing={n_skipped_missing})"
        )
        return [], Counter()

    logger.info(
        f"[{src.video_id}] frames={n_frames_mp4}  emitted={n_emitted}  "
        f"dropped_dom={n_dropped_dominance}  dropped_other_sub={n_dropped_subsample}  "
        f"missing={n_skipped_missing}  per_cls={dict(cls_counts)}"
    )
    if writer is not None:
        writer.flush()
    _write_done_marker(output_root, safe_stem, src, args, class_names, samples, cls_counts)
    return samples, cls_counts


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------
def write_manifest(
    manifest_path: Path,
    splits: Dict[str, List[dict]],
    meta: dict,
    class_to_idx: dict[str, int],
) -> None:
    payload = {
        "class_to_idx": dict(class_to_idx),
        "idx_to_class": {idx: name for name, idx in class_to_idx.items()},
        "meta": meta,
        "splits": splits,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TandemYTC NPZ builder for full-video sliding windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Data source (mutually exclusive) ---
    source_group = p.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--data_root", type=Path, default=None,
        help="Root directory containing split sub-folders (e.g. data_root/train/video_001/).")
    source_group.add_argument(
        "--source_manifest_csv", type=Path, default=None,
        help="CSV with columns: split, video_path, annot_path. "
             "Each row points to one video and its annotation file.")

    p.add_argument("--train_splits", nargs="+", default=["train"],
                   help="Sub-folder names under --data_root to use for training.")
    p.add_argument("--val_splits", nargs="+", default=["validation"],
                   help="Sub-folder names under --data_root to use for validation.")
    p.add_argument("--exclude_splits", nargs="+", default=[],
                   help="Sub-folder names to refuse to touch (held-out leakage guard).")
    p.add_argument("--class_names", nargs="+", default=list(DEFAULT_CLASSES),
                   help="Behavior classes in model order. 'other' is appended if omitted.")
    p.add_argument("--class_names_file", type=Path, default=None,
                   help="Optional newline-delimited class names file; overrides --class_names.")
    p.add_argument("--yolo_weights", type=Path, required=True)
    p.add_argument("--output_root", type=Path,
                   default=Path("behaviorscope_npz_full_video"))
    p.add_argument("--manifest_path", type=Path, default=None,
                   help="Default: <output_root>/sequence_manifest.json")
    p.add_argument("--storage_mode", choices=["hdf5", "npz"], default="hdf5",
                   help="hdf5 writes one compact per-video store; npz writes one debug file per window.")

    p.add_argument("--source_mode", choices=["mp4", "seq"], default="mp4")
    p.add_argument("--window_size", type=int, default=32)
    p.add_argument("--window_stride", type=int, default=16)
    p.add_argument("--n_animals", type=int, default=2)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--animal_scale_factor", type=float, default=4.0)
    p.add_argument("--group_scale_factor", type=float, default=8.0)
    p.add_argument("--body_length_px", type=float, default=None)
    p.add_argument("--pose_conf_threshold", type=float, default=0.3)
    p.add_argument("--yolo_conf", type=float, default=0.25)
    p.add_argument("--yolo_iou", type=float, default=0.45)
    p.add_argument("--yolo_imgsz", type=int, default=640)
    p.add_argument("--yolo_batch", type=int, default=64)
    p.add_argument("--yolo_task", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--keep_last_box", action="store_true", default=True)
    p.add_argument("--fps", type=float, default=DEFAULT_FPS,
                   help="Default FPS used when annotation/video disagree.")

    p.add_argument("--label_min_dominance", type=float, default=0.5,
                   help="Window labelled by dominant class only if its frame share is >= this value.")
    p.add_argument("--other_subsample", type=float, default=0.30,
                   help="Keep this fraction of 'other' windows (1.0 = all, 0 = none).")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--npz_writers", type=int, default=3)
    p.add_argument("--npz_compression", choices=["deflated", "stored"], default="deflated",
                   help="NPZ compression mode. 'deflated' is much smaller; 'stored' is faster but huge.")
    p.add_argument("--npz_compresslevel", type=int, default=1,
                   help="Deflate compression level 0-9. Level 1 is faster than numpy's default.")
    p.add_argument("--hdf5_compression", choices=["gzip", "lzf", "none"], default="gzip",
                   help="Compression used for compact HDF5 per-video stores.")
    p.add_argument("--hdf5_compresslevel", type=int, default=1,
                   help="Gzip compression level 0-9 for HDF5 stores. Ignored for lzf/none.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip only videos with a completed-video marker matching current build settings.")
    p.add_argument("--clean_incomplete_existing", action=argparse.BooleanOptionalAction, default=True,
                   help="Delete stale per-video NPZ files when no matching completed-video marker exists.")
    p.add_argument("--dry_run", action="store_true",
                   help="Discover videos, parse annotations, report counts; do not run YOLO or write NPZ.")
    p.add_argument("--validate_manifest", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--validation_sample_limit", type=int, default=0,
                   help="Number of manifest samples to validate after build. 0 validates all samples.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.window_size) <= 0:
        raise SystemExit("[full-video] --window_size must be > 0")
    if int(args.window_stride) <= 0:
        raise SystemExit("[full-video] --window_stride must be > 0")
    if not (0.0 <= float(args.label_min_dominance) <= 1.0):
        raise SystemExit("[full-video] --label_min_dominance must be between 0 and 1")
    if not (0.0 <= float(args.other_subsample) <= 1.0):
        raise SystemExit("[full-video] --other_subsample must be between 0 and 1")
    if int(args.npz_writers) < 1:
        raise SystemExit("[full-video] --npz_writers must be >= 1")
    if not (0 <= int(args.npz_compresslevel) <= 9):
        raise SystemExit("[full-video] --npz_compresslevel must be between 0 and 9")
    if not (0 <= int(args.hdf5_compresslevel) <= 9):
        raise SystemExit("[full-video] --hdf5_compresslevel must be between 0 and 9")

    class_names, class_to_idx = build_class_maps(load_class_names(args))
    args.class_names_resolved = class_names
    args.class_to_idx_resolved = class_to_idx

    output_root = args.output_root.resolve()
    manifest_path = (args.manifest_path or (output_root / "sequence_manifest.json")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "prepare_full_video_npz.log"
    logger = setup_logger("PrepareFullVideoNPZ", log_path)

    logger.info(f"[full-video] data_root:       {args.data_root}")
    logger.info(f"[full-video] source_csv:      {args.source_manifest_csv}")
    logger.info(f"[full-video] train_splits:    {args.train_splits}")
    logger.info(f"[full-video] val_splits:      {args.val_splits}")
    logger.info(f"[full-video] exclude_splits:  {args.exclude_splits}")
    logger.info(f"[full-video] output_root:     {output_root}")
    logger.info(f"[full-video] window_size:     {args.window_size}  stride={args.window_stride}")
    logger.info(f"[full-video] label_min_dom:   {args.label_min_dominance}")
    logger.info(f"[full-video] other_subsample: {args.other_subsample}")
    logger.info(f"[full-video] class_names:     {class_names}")
    logger.info(f"[full-video] storage_mode:    {args.storage_mode}")
    logger.info(f"[full-video] npz_compression: {args.npz_compression} level={args.npz_compresslevel}")
    logger.info(f"[full-video] hdf5_compression:{args.hdf5_compression} level={args.hdf5_compresslevel}")

    if args.source_manifest_csv is not None:
        sources = discover_sources_from_manifest(
            args.source_manifest_csv, args.exclude_splits, logger
        )
    else:
        sources = discover_sources(
            args.data_root, args.train_splits, args.val_splits, args.exclude_splits, logger,
        )
    if not sources:
        raise SystemExit("[full-video] no sources discovered. Check --data_root / --source_manifest_csv and split names.")

    if args.dry_run:
        logger.info("[full-video] DRY RUN - parsing annotations and reporting counts only.")
        total_frames = 0
        total_windows_estimate = 0
        for s in sources:
            try:
                header, bouts, unknown_labels = parse_annot_for_classes(s.annot_path, class_to_idx)
            except Exception as exc:
                logger.warning(f"[dry] {s.video_id}: annot parse failed: {exc}")
                continue
            n_bouts = len(bouts)
            logger.info(
                f"[dry] {s.split}/{s.video_id}  bouts={n_bouts}  fps={header.get('fps')} "
                f"unknown_labels={sorted(unknown_labels)}"
            )
        logger.info("[full-video] dry run complete.")
        return 0

    logger.info(f"[full-video] loading YOLO weights: {args.yolo_weights}")
    model = load_yolo_model(args.yolo_weights, device=args.device, task=args.yolo_task)

    npz_writer = None
    if str(args.storage_mode).lower() == "npz":
        npz_writer = _AsyncNPZWriter(
            num_workers=int(args.npz_writers),
            compression=str(args.npz_compression),
            compresslevel=int(args.npz_compresslevel),
        )

    splits_out: Dict[str, List[dict]] = {"train": [], "val": []}
    class_counts_total: Counter = Counter()
    per_video_summary: List[dict] = []

    t0 = time.time()
    for i, src in enumerate(sources):
        logger.info(f"[full-video] [{i+1}/{len(sources)}] {src.split}/{src.video_id}")
        try:
            samples, cls_counts = process_video(src, model, args, output_root, npz_writer, logger)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.error(f"[full-video] {src.video_id}: process failed: {exc}", exc_info=True)
            per_video_summary.append({
                "split": src.split, "video_id": src.video_id,
                "status": "error", "error": str(exc),
            })
            continue
        splits_out[src.split].extend(samples)
        class_counts_total.update(cls_counts)
        per_video_summary.append({
            "split": src.split, "video_id": src.video_id,
            "status": "ok", "n_windows": len(samples),
            **{f"n_{c}": int(cls_counts.get(c, 0)) for c in class_names},
        })

    if npz_writer is not None:
        npz_writer.close()
    elapsed = time.time() - t0

    # Build + write manifest
    n_train = len(splits_out["train"]); n_val = len(splits_out["val"])
    if n_train == 0 or n_val == 0:
        raise SystemExit(f"[full-video] empty split: train={n_train}, val={n_val}")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "build_kind": BUILD_KIND,
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "n_animals": int(args.n_animals),
        "keypoints_per_animal": 7,
        "class_names": list(class_names),
        "label_min_dominance": float(args.label_min_dominance),
        "other_subsample": float(args.other_subsample),
        "storage_mode": str(args.storage_mode),
        "storage_payload": "pose_social_only_v1",
        "npz_compression": str(args.npz_compression),
        "npz_compresslevel": int(args.npz_compresslevel),
        "hdf5_compression": str(args.hdf5_compression),
        "hdf5_compresslevel": int(args.hdf5_compresslevel),
        "yolo_weights": str(args.yolo_weights),
        "yolo_imgsz": int(args.yolo_imgsz),
        "data_root": str(args.data_root) if args.data_root else None,
        "source_manifest_csv": str(args.source_manifest_csv) if args.source_manifest_csv else None,
        "train_splits": list(args.train_splits),
        "val_splits": list(args.val_splits),
        "exclude_splits": list(args.exclude_splits),
        "build_elapsed_s": round(elapsed, 1),
        "class_counts": dict(class_counts_total),
        "n_source_videos_train": len({s["source_video"] for s in splits_out["train"]}),
        "n_source_videos_val": len({s["source_video"] for s in splits_out["val"]}),
    }
    write_manifest(manifest_path, splits_out, meta, class_to_idx)
    logger.info(f"[full-video] manifest -> {manifest_path}")

    # Per-video summary
    pv_csv = output_root / "per_video_build_summary.csv"
    if per_video_summary:
        fieldnames = sorted({key for row in per_video_summary for key in row.keys()})
        with open(pv_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(per_video_summary)
        logger.info(f"[full-video] per-video summary -> {pv_csv}")

    # Summary JSON
    summary = {
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "elapsed_s": round(elapsed, 1),
        "windows_total": n_train + n_val,
        "windows_train": n_train,
        "windows_val": n_val,
        "class_counts": dict(class_counts_total),
        "n_source_videos_train": meta["n_source_videos_train"],
        "n_source_videos_val": meta["n_source_videos_val"],
    }
    if args.validate_manifest:
        # Light validation: schema check via load_n_manifest + payload check on a sample
        try:
            splits_loaded, _ci, _ic, _meta = load_n_manifest(manifest_path)
            n_check = (
                int(args.validation_sample_limit)
                if int(args.validation_sample_limit) > 0
                else n_train + n_val
            )
            all_samples = splits_loaded["train"] + splits_loaded["val"]
            random.seed(args.seed)
            sampled = random.sample(all_samples, min(n_check, len(all_samples)))
            for s in sampled:
                if s.storage_format == "hdf5" or s.sequence_h5 is not None:
                    data = load_hdf5_window_payload(s)
                    validate_n_npz_payload(
                        data, sample_id=s.id,
                        num_frames=int(args.window_size),
                        n_animals=int(args.n_animals),
                        num_keypoints=7,
                        rel_feature_dim=REL_FEATURE_DIM,
                        require_schema_version=True,
                    )
                else:
                    if s.sequence_npz is None:
                        raise ValueError(f"{s.id}: missing sequence_npz")
                    with np.load(s.sequence_npz, allow_pickle=False) as data:
                        validate_n_npz_payload(
                            data, sample_id=s.id,
                            num_frames=int(args.window_size),
                            n_animals=int(args.n_animals),
                            num_keypoints=7,
                            rel_feature_dim=REL_FEATURE_DIM,
                            require_schema_version=True,
                        )
            summary["manifest_validation"] = {
                "samples_total": n_train + n_val,
                "samples_validated": len(sampled),
                "ok": True,
            }
            logger.info(f"[full-video] [validate] ok  samples={n_train+n_val}  "
                        f"checked={len(sampled)}")
        except Exception as exc:
            summary["manifest_validation"] = {"ok": False, "error": str(exc)}
            logger.error(f"[full-video] [validate] FAILED: {exc}")
            summary_path = output_root / "prepare_full_video_npz_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            raise SystemExit(f"[full-video] manifest validation failed: {exc}") from exc

    summary_path = output_root / "prepare_full_video_npz_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"[full-video] summary -> {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"[full-video] ERROR: {exc}", file=sys.stderr)
        raise

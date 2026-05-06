from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


@dataclass(frozen=True)
class SequenceSample:
    """Metadata describing a pre-cropped behavior sequence."""

    id: str
    label: int
    class_name: str
    sequence_npz: Path | None
    frame_paths: Tuple[Path, ...]
    start_frame: int
    end_frame: int
    fps: float
    source_video: str


def _split_counts(n_items: int, ratios: Sequence[float]) -> List[int]:
    if n_items == 0:
        return [0 for _ in ratios]
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("At least one split ratio must be > 0.")
    scaled = [r / ratio_sum * n_items for r in ratios]
    counts = [int(math.floor(val)) for val in scaled]
    remainder = n_items - sum(counts)

    fractional = sorted(
        ((scaled[i] - counts[i], i) for i in range(len(ratios))),
        reverse=True,
    )
    for _ in range(remainder):
        _, idx = fractional.pop(0)
        counts[idx] += 1
    return counts


def stratified_split(
    samples: Sequence[SequenceSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[SequenceSample]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[SequenceSample]] = {}
    for sample in samples:
        by_class.setdefault(sample.label, []).append(sample)

    splits: Dict[str, List[SequenceSample]] = {"train": [], "val": [], "test": []}
    for class_samples in by_class.values():
        rng.shuffle(class_samples)
        train_count, val_count, test_count = _split_counts(
            len(class_samples), (train_ratio, val_ratio, test_ratio)
        )

        start = 0
        splits["train"].extend(class_samples[start : start + train_count])
        start += train_count
        splits["val"].extend(class_samples[start : start + val_count])
        start += val_count
        splits["test"].extend(class_samples[start : start + test_count])
        if start < len(class_samples):
            splits["train"].extend(class_samples[start:])
    return splits


def _relative_path(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def save_sequence_manifest(
    manifest_path: Path | str,
    splits: Dict[str, Sequence[SequenceSample]],
    class_to_idx: Dict[str, int],
    metadata: Dict | None = None,
):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    base = manifest_path.parent
    payload = {
        "class_to_idx": class_to_idx,
        "idx_to_class": {idx: name for name, idx in class_to_idx.items()},
        "meta": metadata or {},
        "splits": {},
    }
    for split_name, split_samples in splits.items():
        payload["splits"][split_name] = [
            {
                "id": sample.id,
                "label": sample.label,
                "class_name": sample.class_name,
                "sequence_npz": _relative_path(sample.sequence_npz, base),
                "frame_paths": [
                    _relative_path(path, base) for path in sample.frame_paths
                ],
                "start_frame": sample.start_frame,
                "end_frame": sample.end_frame,
                "fps": sample.fps,
                "source_video": sample.source_video,
            }
            for sample in split_samples
        ]
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_sequence_manifest(
    manifest_path: Path | str,
) -> Tuple[
    Dict[str, List[SequenceSample]],
    Dict[str, int],
    Dict[int, str],
    Dict,
]:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    def _normalize_mapping(data: Dict) -> Tuple[Dict[str, int], Dict[int, str]]:
        c2i_raw = data.get("class_to_idx") or {}
        i2c_raw = data.get("idx_to_class") or {}
        if c2i_raw and i2c_raw:
            c2i = {str(k): int(v) for k, v in c2i_raw.items()}
            i2c = {int(k): str(v) for k, v in i2c_raw.items()}
            return c2i, i2c

        class_names = set()
        for split in data.get("splits", {}).values():
            for sample in split:
                name = sample.get("class_name")
                if name is not None:
                    class_names.add(str(name))
        if not class_names:
            class_names = {"unknown"}
        class_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        return class_to_idx, idx_to_class

    class_to_idx, idx_to_class = _normalize_mapping(payload)

    base = manifest_path.parent
    splits: Dict[str, List[SequenceSample]] = {}
    for split_name, samples in payload["splits"].items():
        split_entries: List[SequenceSample] = []
        for sample in samples:
            npz_path = sample.get("sequence_npz")
            frame_paths = [base / Path(p) for p in sample.get("frame_paths", [])]
            class_name = sample.get("class_name", "")
            label_val = sample.get("label", None)
            if label_val is None and class_name in class_to_idx:
                label_val = class_to_idx[class_name]
            try:
                label_int = int(label_val) if label_val is not None else 0
            except (TypeError, ValueError):
                label_int = class_to_idx.get(class_name, 0)
            split_entries.append(
                SequenceSample(
                    id=sample["id"],
                    label=label_int,
                    class_name=class_name,
                    sequence_npz=base / Path(npz_path) if npz_path else None,
                    frame_paths=tuple(frame_paths),
                    start_frame=int(sample["start_frame"]),
                    end_frame=int(sample["end_frame"]),
                    fps=float(sample.get("fps", 30.0)),
                    source_video=sample.get("source_video", ""),
                )
            )
        splits[split_name] = split_entries
    metadata = payload.get("meta", {})
    return splits, class_to_idx, idx_to_class, metadata


def build_frame_transform(frame_size: int, augment: bool) -> T.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    ops: List = [T.ToPILImage()]
    if augment:
        ops.append(
            T.RandomResizedCrop(frame_size, scale=(0.8, 1.0), ratio=(0.9, 1.1))
        )
        ops.append(T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1))
        ops.append(T.RandomHorizontalFlip())
    else:
        ops.append(T.Resize((frame_size, frame_size)))
    ops.append(T.ToTensor())
    ops.append(T.Normalize(mean=mean, std=std))
    return T.Compose(ops)


def _load_sequence_frames(sample: SequenceSample) -> np.ndarray:
    if sample.sequence_npz and sample.sequence_npz.exists():
        data = np.load(sample.sequence_npz, allow_pickle=False)
        frames = data["frames"]
        return frames
    if sample.frame_paths:
        frames = []
        for path in sample.frame_paths:
            with Image.open(path) as img:
                frames.append(np.array(img.convert("RGB")))
        return np.stack(frames, axis=0)
    raise FileNotFoundError(
        f"Sample {sample.id} has no NPZ data or frame paths available."
    )


class SequenceDataset(Dataset):
    """Loads pre-cropped sequences stored as NPZ tensors or QA crops."""

    def __init__(
        self,
        samples: Sequence[SequenceSample],
        num_frames: int,
        frame_size: int,
        augment: bool = False,
    ):
        self.samples = list(samples)
        self.num_frames = num_frames
        self.transform = build_frame_transform(frame_size, augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        try:
            frames = _load_sequence_frames(sample)
            if frames.shape[0] != self.num_frames:
                raise ValueError(
                    f"Sequence length mismatch for {sample.id}: "
                    f"{frames.shape[0]} vs expected {self.num_frames}"
                )
            processed = [self.transform(frame) for frame in frames]
            clip_tensor = torch.stack(processed, dim=0)
            return clip_tensor, sample.label
        except Exception as exc:  # pragma: no cover - recovery path
            print(f"Warning: skipping {sample.id} ({exc})")
            return None


def collate_with_error_handling(batch: Sequence):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    clips, labels = zip(*batch)
    return torch.stack(clips, dim=0), torch.tensor(labels, dtype=torch.long)

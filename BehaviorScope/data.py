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
from torchvision.transforms import functional as TF


if __package__ is None or __package__ == "":  # pragma: no cover
    from config import AUGMENTATION_CONFIG, IMAGENET_MEAN, IMAGENET_STD
    from utils.pose_features import extract_rich_pose_features
else:
    from .config import AUGMENTATION_CONFIG, IMAGENET_MEAN, IMAGENET_STD
    from .utils.pose_features import extract_rich_pose_features


@dataclass(frozen=True)
class SequenceAugmentConfig:
    """Lightweight augmentations tuned for top-down behavior clips."""

    # Spatial (simulate crop/scale drift from YOLO)
    scale_range: Tuple[float, float] = AUGMENTATION_CONFIG["scale_range"]
    translate_frac: float = AUGMENTATION_CONFIG["translate_frac"]
    rotation_deg: float = AUGMENTATION_CONFIG["rotation_deg"]
    hflip_p: float = AUGMENTATION_CONFIG["hflip_p"]
    vflip_p: float = AUGMENTATION_CONFIG["vflip_p"]

    # Photometric (keep mild to preserve behavior cues)
    brightness: float = AUGMENTATION_CONFIG["brightness"]
    contrast: float = AUGMENTATION_CONFIG["contrast"]
    saturation: float = AUGMENTATION_CONFIG["saturation"]
    hue: float = AUGMENTATION_CONFIG["hue"]
    blur_p: float = AUGMENTATION_CONFIG["blur_p"]
    blur_sigma: Tuple[float, float] = AUGMENTATION_CONFIG["blur_sigma"]
    noise_p: float = AUGMENTATION_CONFIG["noise_p"]
    noise_std: float = AUGMENTATION_CONFIG["noise_std"]  # applied in [0, 1] pixel space before normalize

    # Temporal (kept mild via caps inside the dataset)
    start_jitter_min: int = AUGMENTATION_CONFIG["start_jitter_min"]
    start_jitter: int = AUGMENTATION_CONFIG["start_jitter"]
    frame_dropout: int = AUGMENTATION_CONFIG["frame_dropout"]


def _infer_zero_pad_width(path: Path) -> int:
    stem = path.stem
    return len(stem) if stem.isdigit() else 6


def _load_frames_from_paths(frame_paths: Sequence[Path]) -> np.ndarray:
    frames = []
    for path in frame_paths:
        with Image.open(path) as img:
            frames.append(np.array(img.convert("RGB")))
    return np.stack(frames, axis=0)


def _apply_temporal_shift_with_padding(frames: np.ndarray, offset: int) -> np.ndarray:
    if offset == 0:
        return frames
    num_frames = int(frames.shape[0])
    if num_frames <= 1:
        return frames
    if offset > 0:
        offset = min(offset, num_frames - 1)
        pad = np.repeat(frames[-1:, ...], offset, axis=0)
        return np.concatenate([frames[offset:, ...], pad], axis=0)
    offset = min(-offset, num_frames - 1)
    pad = np.repeat(frames[:1, ...], offset, axis=0)
    return np.concatenate([pad, frames[: num_frames - offset, ...]], axis=0)


def _apply_frame_dropout(frames: np.ndarray, dropout: int) -> np.ndarray:
    num_frames = int(frames.shape[0])
    if dropout <= 0 or num_frames <= 1:
        return frames
    dropout = min(dropout, num_frames - 1)
    indices = random.sample(range(num_frames), dropout)
    out = frames.copy()
    for idx in indices:
        if idx == 0:
            src = 1
        elif idx == num_frames - 1:
            src = num_frames - 2
        else:
            src = idx - 1 if random.random() < 0.5 else idx + 1
        out[idx] = frames[src]
    return out


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


def stratified_split_by_source_video(
    samples: Sequence[SequenceSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[SequenceSample]]:
    """
    Stratified split that keeps all windows from the same `source_video` together.

    The default `stratified_split` operates at the window/sample level, which can
    leak overlapping windows from a single video across train/val/test. This
    variant prevents that leakage while still trying to match the requested split
    ratios (approximately, by number of windows per class).
    """

    rng = random.Random(seed)
    by_class_and_video: Dict[int, Dict[str, List[SequenceSample]]] = {}
    for sample in samples:
        video_id = sample.source_video or sample.id
        by_class_and_video.setdefault(sample.label, {}).setdefault(video_id, []).append(
            sample
        )

    splits: Dict[str, List[SequenceSample]] = {"train": [], "val": [], "test": []}
    split_names = ("train", "val", "test")

    for videos in by_class_and_video.values():
        group_items = list(videos.items())
        rng.shuffle(group_items)
        group_items.sort(key=lambda item: -len(item[1]))

        total_samples = sum(len(group_samples) for _, group_samples in group_items)
        train_target, val_target, test_target = _split_counts(
            total_samples, (train_ratio, val_ratio, test_ratio)
        )
        targets = {"train": train_target, "val": val_target, "test": test_target}
        counts = {"train": 0, "val": 0, "test": 0}

        for _, group_samples in group_items:
            group_size = len(group_samples)

            def _score(candidate: str) -> Tuple[float, float]:
                if targets[candidate] == 0 and group_size > 0:
                    return (1e12, 0.0)

                next_counts = dict(counts)
                next_counts[candidate] += group_size
                err = sum(
                    abs(next_counts[name] - targets[name]) for name in split_names
                )
                remaining = targets[candidate] - counts[candidate]
                return (err, -float(remaining))

            chosen = min(split_names, key=_score)
            splits[chosen].extend(group_samples)
            counts[chosen] += group_size

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

    class_to_idx = {str(k): int(v) for k, v in payload["class_to_idx"].items()}
    idx_to_class = {int(k): v for k, v in payload["idx_to_class"].items()}

    base = manifest_path.parent
    splits: Dict[str, List[SequenceSample]] = {}
    for split_name, samples in payload["splits"].items():
        split_entries: List[SequenceSample] = []
        for sample in samples:
            npz_path = sample.get("sequence_npz")
            frame_paths = [base / Path(p) for p in sample.get("frame_paths", [])]
            split_entries.append(
                SequenceSample(
                    id=sample["id"],
                    label=int(sample["label"]),
                    class_name=sample["class_name"],
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
    ops.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(ops)


def _load_sequence_data(
    sample: SequenceSample, *, load_keypoints: bool = True
) -> Tuple[np.ndarray, np.ndarray | None]:
    if sample.sequence_npz and sample.sequence_npz.exists():
        data = np.load(sample.sequence_npz, allow_pickle=False)
        frames = data["frames"]
        keypoints = None
        if load_keypoints and "keypoints" in data:
            keypoints = data["keypoints"]
        return frames, keypoints
    if sample.frame_paths:
        return _load_frames_from_paths(sample.frame_paths), None
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
        augment_config: SequenceAugmentConfig | None = None,
        load_pose: bool = True,
    ):
        self.samples = list(samples)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment = augment
        self.augment_config = augment_config or SequenceAugmentConfig()
        self.load_pose = bool(load_pose)
        self.base_transform = build_frame_transform(frame_size, augment=False)

    def _process_augmented_clip(
        self,
        frames: np.ndarray,
        keypoints: np.ndarray | None = None,
        *,
        return_keypoints_px: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor | None] | Tuple[torch.Tensor, torch.Tensor | None, np.ndarray]:
        cfg = self.augment_config
        height, width = int(frames.shape[1]), int(frames.shape[2])

        do_hflip = random.random() < cfg.hflip_p
        do_vflip = random.random() < cfg.vflip_p

        angle = random.uniform(-cfg.rotation_deg, cfg.rotation_deg)
        max_dx = cfg.translate_frac * float(width)
        max_dy = cfg.translate_frac * float(height)
        translate = (
            int(round(random.uniform(-max_dx, max_dx))),
            int(round(random.uniform(-max_dy, max_dy))),
        )
        scale = random.uniform(cfg.scale_range[0], cfg.scale_range[1])

        color_jitter = T.ColorJitter(
            brightness=cfg.brightness,
            contrast=cfg.contrast,
            saturation=cfg.saturation,
            hue=cfg.hue,
        )
        fn_idx, b, c, s, h = T.ColorJitter.get_params(
            color_jitter.brightness,
            color_jitter.contrast,
            color_jitter.saturation,
            color_jitter.hue,
        )
        order = [int(v) for v in fn_idx.tolist()]

        blur_sigma = None
        if random.random() < cfg.blur_p:
            blur_sigma = random.uniform(cfg.blur_sigma[0], cfg.blur_sigma[1])

        processed: List[torch.Tensor] = []
        for frame in frames:
            img = Image.fromarray(frame)
            if do_hflip:
                img = TF.hflip(img)
            if do_vflip:
                img = TF.vflip(img)
            
            # Affine transform
            img = TF.affine(
                img,
                angle=angle,
                translate=list(translate),
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=T.InterpolationMode.BILINEAR,
                fill=0,
            )
            
            for fn_id in order:
                if fn_id == 0 and b is not None:
                    img = TF.adjust_brightness(img, b)
                elif fn_id == 1 and c is not None:
                    img = TF.adjust_contrast(img, c)
                elif fn_id == 2 and s is not None:
                    img = TF.adjust_saturation(img, s)
                elif fn_id == 3 and h is not None:
                    img = TF.adjust_hue(img, h)

            if blur_sigma is not None:
                img = TF.gaussian_blur(
                    img, kernel_size=[3, 3], sigma=[blur_sigma, blur_sigma]
                )

            img = TF.resize(
                img,
                [self.frame_size, self.frame_size],
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.to_tensor(img)
            if cfg.noise_std > 0 and random.random() < cfg.noise_p:
                tensor = (tensor + torch.randn_like(tensor) * cfg.noise_std).clamp(
                    0.0, 1.0
                )
            tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            processed.append(tensor)
        
        clip_tensor = torch.stack(processed, dim=0)
        
        # Process keypoints if present
        pose_tensor = None
        keypoints_px = None
        if keypoints is not None:
            # keypoints: (T, K, 3) -> (x, y, conf)
            # We need to apply geometric transforms: hflip, vflip, affine
            # Note: T.affine on images rotates around center. We must match that.
            
            T_k, K_k, _ = keypoints.shape
            kpts = keypoints.copy() # shape (T, K, 3)
            
            # Center of the original image
            cx, cy = width / 2.0, height / 2.0
            
            for t in range(T_k):
                for k in range(K_k):
                    x, y, conf = kpts[t, k]
                    
                    # 1. Flips
                    if do_hflip:
                        x = (width - 1) - x
                    if do_vflip:
                        y = (height - 1) - y
                    
                    # 2. Affine (Rotate + Scale + Translate)
                    # Coordinates are relative to top-left (0,0).
                    # Rotation is around center (cx, cy).
                    
                    # Translate to center
                    x -= cx
                    y -= cy
                    
                    # Rotate
                    # torchvision.transforms.functional.affine() uses clockwise-positive angles.
                    rad = math.radians(angle)
                    xr = x * math.cos(rad) - y * math.sin(rad)
                    yr = x * math.sin(rad) + y * math.cos(rad)
                    
                    # Scale
                    xr *= scale
                    yr *= scale
                    
                    # Translate back + shift
                    xr += cx + translate[0]
                    yr += cy + translate[1]
                    
                    # 3. Resize to target frame_size
                    # The image was resized from (width, height) to (frame_size, frame_size)
                    # BUT TF.affine keeps the original size canvas, THEN TF.resize scales it.
                    # So we normalize by original dims and scale to new dims.
                    
                    xr = xr / width * self.frame_size
                    yr = yr / height * self.frame_size
                    
                    kpts[t, k, 0] = xr
                    kpts[t, k, 1] = yr
            
            if return_keypoints_px:
                keypoints_px = kpts.copy()

            # Normalize to [0, 1]
            kpts[:, :, 0] /= self.frame_size
            kpts[:, :, 1] /= self.frame_size
            
            # Extract rich features (distances, speeds, etc.)
            feats = extract_rich_pose_features(kpts)
            pose_tensor = torch.from_numpy(feats).float()

        if return_keypoints_px:
            return clip_tensor, pose_tensor, keypoints_px
        return clip_tensor, pose_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        try:
            frames, keypoints = None, None
            # ... (Jitter logic for QA paths omitted for brevity as it doesn't support pose yet) ...
            # Loading frames
            if frames is None:
                frames, keypoints = _load_sequence_data(
                    sample, load_keypoints=self.load_pose
                )
            
            if frames.shape[0] != self.num_frames:
                raise ValueError(
                    f"Sequence length mismatch for {sample.id}: "
                    f"{frames.shape[0]} vs expected {self.num_frames}"
                )
            
            # Handle keypoints array slicing if frame dropout happens
            if self.augment:
                if self.augment_config.start_jitter > 0 and not sample.frame_paths:
                    jitter_cap = min(
                        int(self.augment_config.start_jitter),
                        int(self.num_frames // 4),
                    )
                    jitter_min = min(
                        max(0, int(self.augment_config.start_jitter_min)), jitter_cap
                    )
                    if jitter_cap > 0:
                        if jitter_min > 0:
                            magnitude = random.randint(jitter_min, jitter_cap)
                            offset = (
                                -magnitude if random.random() < 0.5 else magnitude
                            )
                        else:
                            offset = random.randint(-jitter_cap, jitter_cap)
                        frames = _apply_temporal_shift_with_padding(frames, offset)
                        if keypoints is not None:
                            keypoints = _apply_temporal_shift_with_padding(keypoints, offset)

                dropout_cap = min(
                    int(self.augment_config.frame_dropout),
                    int(self.num_frames // 3),
                )
                if dropout_cap > 0:
                    frames = _apply_frame_dropout(frames, dropout_cap)
                    if keypoints is not None:
                        keypoints = _apply_frame_dropout(keypoints, dropout_cap)

                clip_tensor, pose_tensor = self._process_augmented_clip(frames, keypoints)
            else:
                processed = [self.base_transform(frame) for frame in frames]
                clip_tensor = torch.stack(processed, dim=0)
                pose_tensor = None
                if self.load_pose and keypoints is not None:
                    # Resize/Normalize keypoints for consistency even if no augment
                    # Just scaling to [0, 1] relative to original image dims?
                    # The base_transform does Resize((frame_size)).
                    h, w = frames.shape[1], frames.shape[2]
                    kpts = keypoints.copy()
                    kpts[:, :, 0] *= (self.frame_size / w)
                    kpts[:, :, 1] *= (self.frame_size / h)
                    # Normalize to [0, 1]
                    kpts[:, :, 0] /= self.frame_size
                    kpts[:, :, 1] /= self.frame_size
                    
                    feats = extract_rich_pose_features(kpts)
                    pose_tensor = torch.from_numpy(feats).float()

            return (clip_tensor, pose_tensor), sample.label
        except Exception as exc:  # pragma: no cover - recovery path      
            print(f"Warning: skipping {sample.id} ({exc})")
            return None


def collate_with_error_handling(batch: Sequence):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    inputs, labels = zip(*batch)
    # inputs is a tuple of ((clip, pose), (clip, pose), ...)
    clips = [i[0] for i in inputs]
    poses = [i[1] for i in inputs]
    
    clip_batch = torch.stack(clips, dim=0)
    
    # Handle mixed pose/no-pose in batch
    has_pose = any(p is not None for p in poses)
    pose_batch = None
    pose_present = torch.tensor([p is not None for p in poses], dtype=torch.bool)
    if has_pose:
        # If some samples have pose and others don't, we must pad the missing ones
        # to maintain batch alignment.
        # 1. Find the shape of a valid pose tensor
        valid_sample = next(p for p in poses if p is not None)
        # shape: [T, FeatureDim]
        
        padded_poses = []
        for p in poses:
            if p is not None:
                padded_poses.append(p)
            else:
                padded_poses.append(torch.zeros_like(valid_sample))
        
        pose_batch = torch.stack(padded_poses, dim=0)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    if not has_pose:
        return (clip_batch, None), labels_tensor
    if bool(pose_present.all()):
        return (clip_batch, pose_batch), labels_tensor
    return (clip_batch, pose_batch, pose_present), labels_tensor

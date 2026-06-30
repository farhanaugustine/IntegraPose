"""TandemYTC multi-animal pose/social data loader."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from utils.pose_features_y import extract_per_animal_pose_features
except Exception:  # pragma: no cover
    from .utils.pose_features_y import extract_per_animal_pose_features


@dataclass
class MultiAnimalSequenceSample:
    id: str
    label: int
    class_name: str
    sequence_npz: Path | None
    start_frame: int
    end_frame: int
    fps: float
    source_video: str
    sequence_h5: Path | None = None
    storage_format: str = "npz"


REQUIRED_SCHEMA_VERSION = "yolo-temporal-full-video-v1"
COMPATIBLE_SCHEMA_VERSIONS = {
    REQUIRED_SCHEMA_VERSION,
    # BehaviorScope-Y full-video caches can carry additional arrays; TandemYTC
    # only consumes the shared pose/social tensor contract.
    "behaviorscope-n-v1",
}


def _shape_str(arr: np.ndarray) -> str:
    return "x".join(str(v) for v in arr.shape)


def validate_n_npz_payload(
    data,
    *,
    sample_id: str,
    num_frames: int,
    n_animals: int,
    num_keypoints: int,
    rel_feature_dim: int,
    require_schema_version: bool = False,
    require_visual: bool | None = None,
) -> None:
    """Validate the pose/social tensor contract used by TandemYTC."""
    required = {
        "animal_keypoints": (num_frames, n_animals, num_keypoints, 3),
        "animal_keypoints_raw": (num_frames, n_animals, num_keypoints, 2),
        "animal_keypoints_conf": (num_frames, n_animals, num_keypoints),
        "animal_mask": (num_frames, n_animals),
        "pose_mask": (num_frames, n_animals),
        "pose_conf": (num_frames, n_animals),
        "crop_conf": (num_frames, n_animals),
        "track_conf": (num_frames, n_animals),
        "track_age": (num_frames, n_animals),
        "relation_features": (num_frames, n_animals, n_animals, rel_feature_dim),
        "relation_pose_mask": (num_frames, n_animals, n_animals),
        "relation_pose_conf": (num_frames, n_animals, n_animals),
        "relation_present": (num_frames, n_animals, n_animals),
        "track_ids": (num_frames, n_animals),
    }
    missing = [name for name in required if name not in data.files]
    if missing:
        raise ValueError(f"{sample_id}: missing required keys: {missing}")

    if require_schema_version:
        if "schema_version" not in data.files:
            raise ValueError(f"{sample_id}: missing schema_version")
        schema = str(np.asarray(data["schema_version"]).item())
        if schema not in COMPATIBLE_SCHEMA_VERSIONS:
            raise ValueError(
                f"{sample_id}: schema_version={schema!r}, expected one of "
                f"{sorted(COMPATIBLE_SCHEMA_VERSIONS)!r}"
            )

    if "n_animals" in data.files:
        npz_n_animals = int(np.asarray(data["n_animals"]).item())
        if npz_n_animals != int(n_animals):
            raise ValueError(
                f"{sample_id}: n_animals={npz_n_animals}, expected {n_animals}"
            )

    for name, expected in required.items():
        arr = data[name]
        if arr.ndim != len(expected):
            raise ValueError(
                f"{sample_id}: {name} ndim={arr.ndim}, expected {len(expected)} "
                f"(shape={_shape_str(arr)})"
            )
        for axis, exp in enumerate(expected):
            if exp is not None and int(arr.shape[axis]) != int(exp):
                raise ValueError(
                    f"{sample_id}: {name} shape={_shape_str(arr)}, "
                    f"expected axis {axis} == {exp}"
                )

    animal_mask = data["animal_mask"].astype(bool)
    pose_mask = data["pose_mask"].astype(bool)
    if np.any(pose_mask & ~animal_mask):
        raise ValueError(f"{sample_id}: pose_mask contains True where animal_mask is False")

    rel_present = data["relation_present"].astype(bool)
    expected_rel_present = animal_mask[:, :, None] & animal_mask[:, None, :]
    diag = np.arange(n_animals)
    expected_rel_present[:, diag, diag] = False
    if not np.array_equal(rel_present, expected_rel_present):
        raise ValueError(f"{sample_id}: relation_present is inconsistent with animal_mask")


def load_n_manifest(manifest_path: Path | str):
    """Load a sequence_manifest.json."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")
    if manifest_path.suffix.lower() == ".csv":
        raise ValueError(
            "Train expects the JSON training manifest produced by Prepare Full Video, "
            f"but received a CSV file: {manifest_path}\n"
            "Use the Prepare Full Video tab first, then pass its Training manifest output "
            "JSON, usually sequence_manifest.json or the custom .json path you selected."
        )
    base = manifest_path.parent
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Training manifest is not valid JSON: {manifest_path}\n"
            "Train expects the JSON written by Prepare Full Video, not source_manifest.csv "
            "or another annotation-export file."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Training manifest must be a JSON object: {manifest_path}")
    missing = [name for name in ("class_to_idx", "splits") if name not in payload]
    if missing:
        raise ValueError(
            f"Training manifest is missing required field(s) {missing}: {manifest_path}\n"
            "Use the Prepare Full Video output JSON as the Train manifest."
        )

    class_to_idx = {str(k): int(v) for k, v in payload["class_to_idx"].items()}
    idx_payload = payload.get("idx_to_class")
    if isinstance(idx_payload, dict):
        idx_to_class = {int(k): str(v) for k, v in idx_payload.items()}
    else:
        idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
    meta = payload.get("meta", {})

    splits: Dict[str, List[MultiAnimalSequenceSample]] = {}
    for split_name, samples in payload["splits"].items():
        out: List[MultiAnimalSequenceSample] = []
        for s in samples:
            npz_rel = s.get("sequence_npz")
            h5_rel = s.get("sequence_h5") or s.get("feature_store_path")
            if not npz_rel and not h5_rel:
                continue
            npz_path = (base / Path(npz_rel)).resolve() if npz_rel else None
            h5_path = (base / Path(h5_rel)).resolve() if h5_rel else None
            storage_format = str(s.get("storage_format") or ("hdf5" if h5_rel else "npz")).lower()
            out.append(MultiAnimalSequenceSample(
                id=s["id"],
                label=int(s["label"]),
                class_name=s["class_name"],
                sequence_npz=npz_path,
                start_frame=int(s.get("start_frame", 0)),
                end_frame=int(s.get("end_frame", 0)),
                fps=float(s.get("fps", 30.0)),
                source_video=s.get("source_video", ""),
                sequence_h5=h5_path,
                storage_format=storage_format,
            ))
        splits[split_name] = out
    return splits, class_to_idx, idx_to_class, meta


class _ArrayPayload(dict):
    @property
    def files(self) -> list[str]:
        return list(self.keys())


def load_hdf5_window_payload(sample: MultiAnimalSequenceSample) -> _ArrayPayload:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("h5py is required to load TandemYTC HDF5 stores.") from exc
    if sample.sequence_h5 is None:
        raise ValueError(f"{sample.id}: missing sequence_h5 path")
    start = int(sample.start_frame)
    end = int(sample.end_frame) + 1
    payload = _ArrayPayload()
    with h5py.File(sample.sequence_h5, "r") as h5:
        payload["schema_version"] = np.asarray(h5.attrs.get("schema_version", REQUIRED_SCHEMA_VERSION))
        for name in (
            "animal_keypoints",
            "animal_keypoints_raw",
            "animal_keypoints_conf",
            "animal_mask",
            "pose_mask",
            "pose_conf",
            "crop_conf",
            "track_conf",
            "track_age",
            "relation_features",
            "relation_pose_mask",
            "relation_pose_conf",
            "relation_present",
            "bbox_xyxy_animals",
            "bbox_xyxy_group",
            "crop_xyxy_animals",
            "crop_xyxy_group",
            "track_ids",
        ):
            if name in h5:
                payload[name] = np.asarray(h5[name][start:end])
        payload["body_length_px"] = np.asarray(h5.attrs.get("body_length_px", 0.0), dtype=np.float32)
        payload["n_animals"] = np.asarray(h5.attrs.get("n_animals", 0), dtype=np.int32)
    return payload


class MultiAnimalSequenceDataset(Dataset):
    """Loads TandemYTC windows and returns pose/social tensors with masks."""

    def __init__(
        self,
        samples: Sequence[MultiAnimalSequenceSample],
        num_frames: int,
        n_animals: int,
        num_keypoints: int,
        rel_feature_dim: int = 11,
        pose_dropout_p_uniform: float = 0.0,
        strict_schema: bool = True,
        skip_invalid: bool = False,
        rng_seed: int = 0,
    ):
        self.samples = list(samples)
        self.num_frames = int(num_frames)
        self.n_animals = int(n_animals)
        self.num_keypoints = int(num_keypoints)
        self.rel_feature_dim = int(rel_feature_dim)
        self.pose_dropout_p_uniform = float(pose_dropout_p_uniform)
        self.strict_schema = bool(strict_schema)
        self.skip_invalid = bool(skip_invalid)
        self.rng_seed = int(rng_seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _rng_for_sample(self, idx: int, sample: MultiAnimalSequenceSample) -> np.random.Generator:
        key = f"{self.rng_seed}|{idx}|{sample.id}|{sample.start_frame}|{sample.end_frame}"
        seed = int(hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest(), 16)
        return np.random.default_rng(seed)

    def _apply_pose_dropout(
        self,
        pose_mask: np.ndarray,
        pose_conf: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.pose_dropout_p_uniform <= 0.0:
            return pose_mask, pose_conf
        p = float(rng.uniform(0.0, self.pose_dropout_p_uniform))
        if p <= 1e-9:
            return pose_mask, pose_conf
        drop = rng.random(pose_mask.shape) < p
        new_mask = pose_mask & ~drop
        new_conf = pose_conf.copy()
        new_conf[drop] = 0.0
        return new_mask, new_conf

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        try:
            if sample.storage_format == "hdf5" or sample.sequence_h5 is not None:
                data = load_hdf5_window_payload(sample)
            else:
                if sample.sequence_npz is None:
                    raise ValueError(f"{sample.id}: missing sequence_npz path")
                data = np.load(sample.sequence_npz, allow_pickle=False)

            if self.strict_schema:
                validate_n_npz_payload(
                    data,
                    sample_id=sample.id,
                    num_frames=self.num_frames,
                    n_animals=self.n_animals,
                    num_keypoints=self.num_keypoints,
                    rel_feature_dim=self.rel_feature_dim,
                    require_schema_version=True,
                )

            if "animal_keypoints" in data.files:
                t_clip = int(data["animal_keypoints"].shape[0])
            elif "animal_mask" in data.files:
                t_clip = int(data["animal_mask"].shape[0])
            else:
                t_clip = self.num_frames
            if t_clip != self.num_frames:
                raise ValueError(
                    f"window size mismatch for {sample.id}: {t_clip} vs {self.num_frames}"
                )

            n = self.n_animals

            def _take(name: str, fallback_shape, dtype):
                arr = data[name] if name in data.files else np.zeros(fallback_shape, dtype=dtype)
                if arr.ndim >= 2 and arr.shape[1] != n:
                    if arr.shape[1] < n:
                        pad_shape = list(arr.shape)
                        pad_shape[1] = n - arr.shape[1]
                        arr = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=1)
                    elif arr.shape[1] > n:
                        arr = arr[:, :n]
                return arr

            animal_keypoints = _take("animal_keypoints", (t_clip, n, self.num_keypoints, 3), np.float32)
            _take("animal_keypoints_raw", (t_clip, n, self.num_keypoints, 2), np.float32)
            _take("animal_keypoints_conf", (t_clip, n, self.num_keypoints), np.float32)

            animal_mask = _take("animal_mask", (t_clip, n), np.bool_).astype(bool)
            pose_mask = _take("pose_mask", (t_clip, n), np.bool_).astype(bool)
            pose_conf = _take("pose_conf", (t_clip, n), np.float32).astype(np.float32)
            crop_conf = _take("crop_conf", (t_clip, n), np.float32).astype(np.float32)
            track_conf = _take("track_conf", (t_clip, n), np.float32).astype(np.float32)
            track_age = _take("track_age", (t_clip, n), np.float32).astype(np.float32)

            rel = data["relation_features"] if "relation_features" in data.files else np.zeros((t_clip, n, n, self.rel_feature_dim), dtype=np.float32)
            rel_pose_mask = data["relation_pose_mask"] if "relation_pose_mask" in data.files else np.zeros((t_clip, n, n), dtype=bool)
            rel_pose_conf = data["relation_pose_conf"] if "relation_pose_conf" in data.files else np.zeros((t_clip, n, n), dtype=np.float32)
            rel_present = data["relation_present"] if "relation_present" in data.files else np.zeros((t_clip, n, n), dtype=bool)

            if rel.shape[1] != n or rel.shape[2] != n:
                trimmed = np.zeros((t_clip, n, n, rel.shape[-1]), dtype=rel.dtype)
                copy_n = min(n, rel.shape[1])
                trimmed[:, :copy_n, :copy_n, :] = rel[:, :copy_n, :copy_n, :]
                rel = trimmed
            for arr_name, arr_in in (
                ("rel_pose_mask", rel_pose_mask),
                ("rel_pose_conf", rel_pose_conf),
                ("rel_present", rel_present),
            ):
                if arr_in.shape[1] != n or arr_in.shape[2] != n:
                    trimmed = np.zeros((t_clip, n, n), dtype=arr_in.dtype)
                    copy_n = min(n, arr_in.shape[1])
                    trimmed[:, :copy_n, :copy_n] = arr_in[:, :copy_n, :copy_n]
                    if arr_name == "rel_pose_mask":
                        rel_pose_mask = trimmed.astype(bool)
                    elif arr_name == "rel_pose_conf":
                        rel_pose_conf = trimmed.astype(np.float32)
                    else:
                        rel_present = trimmed.astype(bool)

            try:
                pose_self_feats = extract_per_animal_pose_features(animal_keypoints)
            except Exception:
                k = self.num_keypoints
                d_pose = k * 4 + k * (k - 1) // 2
                pose_self_feats = np.zeros((t_clip, n, d_pose), dtype=np.float32)

            rng = self._rng_for_sample(idx, sample)
            pose_mask, pose_conf = self._apply_pose_dropout(pose_mask, pose_conf, rng)

            out = {
                "animal_mask": torch.from_numpy(animal_mask),
                "pose_mask": torch.from_numpy(pose_mask),
                "pose_conf": torch.from_numpy(pose_conf),
                "crop_conf": torch.from_numpy(crop_conf),
                "track_conf": torch.from_numpy(track_conf),
                "track_age": torch.from_numpy(track_age),
                "pose_self": torch.from_numpy(pose_self_feats),
                "relation_features": torch.from_numpy(rel.astype(np.float32)),
                "relation_pose_mask": torch.from_numpy(rel_pose_mask),
                "relation_pose_conf": torch.from_numpy(rel_pose_conf.astype(np.float32)),
                "relation_present": torch.from_numpy(rel_present),
            }
            label = torch.tensor(int(sample.label), dtype=torch.long)
            return out, label
        except Exception as exc:
            if self.skip_invalid:
                print(f"[data_y] skipping {sample.id}: {exc}")
                return None
            raise


def collate_multi_animal(batch):
    """Stack a batch of dict samples. Drops failed samples (None)."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    inputs = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch], dim=0)
    out: Dict[str, torch.Tensor] = {}
    for key in inputs[0].keys():
        out[key] = torch.stack([s[key] for s in inputs], dim=0)
    return out, labels


def compute_class_weights(
    samples: Sequence[MultiAnimalSequenceSample],
    num_classes: int,
    strategy: str = "sqrt_inverse",
    clamp_max: float | None = 2.0,
) -> torch.Tensor:
    """Compute per-class loss weights from the training samples."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for s in samples:
        if 0 <= int(s.label) < num_classes:
            counts[int(s.label)] += 1.0
    counts = np.maximum(counts, 1.0)
    if strategy == "inverse":
        w = 1.0 / counts
    elif strategy == "sqrt_inverse":
        w = 1.0 / np.sqrt(counts)
    elif strategy in ("none", "uniform"):
        w = np.ones_like(counts)
    else:
        raise ValueError(f"Unknown class_weighting strategy: {strategy}")
    w = w / w.min()
    if clamp_max is not None and clamp_max > 0:
        w = np.minimum(w, float(clamp_max))
    return torch.from_numpy(w.astype(np.float32))


def make_weighted_sampler(
    samples: Sequence[MultiAnimalSequenceSample],
    class_weights: torch.Tensor,
    generator: torch.Generator | None = None,
):
    """Return a sampler that draws each sample with weight class_weights[label]."""
    from torch.utils.data import WeightedRandomSampler

    sample_weights = np.array(
        [float(class_weights[int(s.label)]) for s in samples],
        dtype=np.float64,
    )
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )

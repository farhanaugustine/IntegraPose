from __future__ import annotations

import csv
import inspect
import json
import random
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import cv2
import numpy as np


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BBOX_EPSILON = 1e-6


class DatasetAugmentorCoreError(RuntimeError):
    """Raised when the augmentation engine cannot complete a requested run."""


@dataclass(frozen=True)
class DomainShiftSettings:
    white_mouse: bool = False
    invert_intensity: float = 1.0
    red_light: bool = False
    red_boost: float = 1.5
    bg_suppress: float = 0.2
    bw: bool = False
    bw_prob: float = 0.5


@dataclass(frozen=True)
class ArtifactSettings:
    frame_noise_prob: float = 0.0
    frame_noise_std: float = 12.0
    frame_banding_prob: float = 0.0
    frame_banding_amp: float = 12.0
    regional_noise_prob: float = 0.0
    regional_noise_std: float = 28.0
    occlusion_prob: float = 0.0
    occlusion_count: int = 2
    occlusion_min_frac: float = 0.04
    occlusion_max_frac: float = 0.18
    occlusion_mode: str = "random"


@dataclass(frozen=True)
class TransformSettings:
    horizontal_flip_prob: float = 0.5
    affine_prob: float = 0.9
    scale_min: float = 0.90
    scale_max: float = 1.10
    translate_percent: float = 0.08
    rotate_degrees: float = 12.0
    shear_degrees: float = 6.0
    brightness_contrast_prob: float = 0.5
    hue_saturation_prob: float = 0.4
    blur_prob: float = 0.15
    gauss_noise_prob: float = 0.2


@dataclass(frozen=True)
class AugmentationRecipe:
    dataset_root: Path
    output_root: Optional[Path] = None
    split: str = "all"
    copy_originals: bool = True
    seed: int = 42
    plan: str = "add"
    add_count: int = 1000
    target_ratio: float = 1.0
    add_per_class: int = 0
    multiplier: float = 1.0
    include_classes: str = "all"
    exclude_classes: str = ""
    preview_count: int = 50
    max_total_augs: int = 20000
    max_augs_per_source_image: int = 200
    min_visibility: float = 0.25
    domain_shift: DomainShiftSettings = DomainShiftSettings()
    artifacts: ArtifactSettings = ArtifactSettings()
    transforms: TransformSettings = TransformSettings()


@dataclass(frozen=True)
class LabelObject:
    class_id: int
    bbox: list[float]
    keypoints: list[tuple[float, float, int]]


@dataclass(frozen=True)
class SourceSample:
    image_path: Path
    label_path: Path
    split: str
    objects: list[LabelObject]


@dataclass(frozen=True)
class AugmentationResult:
    output_root: Path
    augmented_samples: int
    original_counts: dict[int, int]
    target_counts: dict[int, int]
    remaining_need: dict[int, int]
    manifest_path: Path
    recipe_path: Path
    preview_dir: Path


ProgressCallback = Callable[[str], None]


def parse_class_set(value: str) -> Optional[set[int]]:
    raw = str(value or "").strip().lower()
    if raw in {"", "none"}:
        return set()
    if raw == "all":
        return None
    try:
        return {int(part.strip()) for part in raw.split(",") if part.strip()}
    except ValueError as exc:
        raise DatasetAugmentorCoreError(
            "Class filters must be 'all', blank, or comma-separated integer IDs."
        ) from exc


def sanitize_yolo_bbox(cx: float, cy: float, bw: float, bh: float) -> list[float]:
    eps = BBOX_EPSILON
    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    bw = float(np.clip(bw, 0.0, 1.0))
    bh = float(np.clip(bh, 0.0, 1.0))

    x1 = max(0.0, min(1.0, cx - (bw / 2.0)))
    y1 = max(0.0, min(1.0, cy - (bh / 2.0)))
    x2 = max(0.0, min(1.0, cx + (bw / 2.0)))
    y2 = max(0.0, min(1.0, cy + (bh / 2.0)))

    if x2 <= x1:
        midpoint = float(np.clip((x1 + x2) / 2.0, eps, 1.0 - eps))
        x1 = max(0.0, midpoint - eps)
        x2 = min(1.0, midpoint + eps)
    if y2 <= y1:
        midpoint = float(np.clip((y1 + y2) / 2.0, eps, 1.0 - eps))
        y1 = max(0.0, midpoint - eps)
        y2 = min(1.0, midpoint + eps)

    if x1 <= 0.0:
        x1 = eps
    if y1 <= 0.0:
        y1 = eps
    if x2 >= 1.0:
        x2 = 1.0 - eps
    if y2 >= 1.0:
        y2 = 1.0 - eps

    x1 = min(x1, x2 - eps)
    y1 = min(y1, y2 - eps)
    return [
        float(np.clip((x1 + x2) / 2.0, 0.0, 1.0)),
        float(np.clip((y1 + y2) / 2.0, 0.0, 1.0)),
        float(np.clip(x2 - x1, eps, 1.0)),
        float(np.clip(y2 - y1, eps, 1.0)),
    ]


def read_label_objects(label_path: Path, *, img_w: int, img_h: int) -> list[LabelObject]:
    objects: list[LabelObject] = []
    if not label_path.exists():
        return objects
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
        except Exception:
            continue
        keypoints: list[tuple[float, float, int]] = []
        kp_payload = parts[5:]
        if kp_payload and len(kp_payload) % 3 == 0:
            for idx in range(0, len(kp_payload), 3):
                try:
                    kx = float(kp_payload[idx]) * img_w
                    ky = float(kp_payload[idx + 1]) * img_h
                    kv = int(float(kp_payload[idx + 2]))
                except Exception:
                    kx, ky, kv = 0.0, 0.0, 0
                keypoints.append((kx, ky, kv))
        objects.append(LabelObject(class_id, sanitize_yolo_bbox(cx, cy, bw, bh), keypoints))
    return objects


def write_label_objects(
    output_path: Path,
    *,
    boxes: Sequence[Sequence[float]],
    classes: Sequence[int],
    keypoints: Sequence[Sequence[float]],
    vis_flat: Sequence[int],
    keypoint_counts: Sequence[int],
    img_w: int,
    img_h: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kp_idx = 0
    vis_idx = 0
    lines: list[str] = []
    for obj_idx, box in enumerate(boxes):
        cls = int(float(classes[obj_idx])) if obj_idx < len(classes) else 0
        cx, cy, bw, bh = sanitize_yolo_bbox(*[float(v) for v in box[:4]])
        row = [f"{cls}", f"{cx:.8f}", f"{cy:.8f}", f"{bw:.8f}", f"{bh:.8f}"]
        kp_count = int(keypoint_counts[obj_idx]) if obj_idx < len(keypoint_counts) else 0
        for _ in range(max(0, kp_count)):
            if kp_idx < len(keypoints):
                kx_px, ky_px = keypoints[kp_idx]
                kx = max(0.0, min(1.0, float(kx_px) / max(img_w, 1)))
                ky = max(0.0, min(1.0, float(ky_px) / max(img_h, 1)))
                kv = int(vis_flat[vis_idx]) if vis_idx < len(vis_flat) else 0
                row.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])
                kp_idx += 1
                vis_idx += 1
            else:
                row.extend(["0.000000", "0.000000", "0"])
        lines.append(" ".join(row))
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def discover_samples(dataset_root: Path, split: str = "all") -> list[SourceSample]:
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    if not images_root.exists() or not labels_root.exists():
        return []

    split_filter = str(split or "all").strip()
    if split_filter.lower() == "all":
        split_dirs = [p for p in sorted(images_root.iterdir()) if p.is_dir()]
    else:
        split_dirs = [images_root / split_filter]

    samples: list[SourceSample] = []
    for split_dir in split_dirs:
        if not split_dir.is_dir():
            continue
        label_split_dir = labels_root / split_dir.name
        for img_path in _find_images(split_dir):
            label_path = label_split_dir / f"{img_path.stem}.txt"
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            objects = read_label_objects(label_path, img_w=w, img_h=h)
            samples.append(SourceSample(img_path, label_path, split_dir.name, objects))
    return samples


def compute_instance_counts(samples: Iterable[SourceSample]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for sample in samples:
        for obj in sample.objects:
            counts[obj.class_id] += 1
    return dict(sorted(counts.items()))


def compute_class_plan(
    counts: dict[int, int],
    *,
    plan: str,
    add_count: int,
    target_ratio: float,
    add_per_class: int,
    multiplier: float,
    include_classes: str,
    exclude_classes: str,
) -> tuple[dict[int, int], dict[int, int], set[int]]:
    if not counts:
        return {}, {}, set()
    classes = sorted(counts)
    include = parse_class_set(include_classes)
    exclude = parse_class_set(exclude_classes)
    augment_classes = set(classes) if include is None else set(include)
    augment_classes = {c for c in augment_classes - set(exclude or set()) if c in counts}
    if not augment_classes:
        raise DatasetAugmentorCoreError(
            "After include/exclude filtering, there are no labeled classes left to augment."
        )

    max_count = max(counts.values())

    def need_balance() -> tuple[dict[int, int], dict[int, int]]:
        if not (0.0 < float(target_ratio) <= 1.0):
            raise DatasetAugmentorCoreError("target_ratio must be in (0, 1].")
        target = {c: int(round(max_count * float(target_ratio))) for c in classes}
        need = {
            c: max(0, target[c] - counts[c]) if c in augment_classes else 0
            for c in classes
        }
        return need, target

    def need_add() -> tuple[dict[int, int], dict[int, int]]:
        if int(add_per_class) < 0:
            raise DatasetAugmentorCoreError("add_per_class must be >= 0.")
        target = {
            c: counts[c] + (int(add_per_class) if c in augment_classes else 0)
            for c in classes
        }
        need = {c: int(add_per_class) if c in augment_classes else 0 for c in classes}
        return need, target

    def need_scale() -> tuple[dict[int, int], dict[int, int]]:
        if float(multiplier) < 1.0:
            raise DatasetAugmentorCoreError("multiplier must be >= 1.0.")
        target = {
            c: int(round(counts[c] * float(multiplier))) if c in augment_classes else counts[c]
            for c in classes
        }
        need = {
            c: max(0, target[c] - counts[c]) if c in augment_classes else 0
            for c in classes
        }
        return need, target

    if plan == "random":
        total = max(1, int(add_count))
        per_class = max(1, int(np.ceil(total / max(1, len(augment_classes)))))
        need = {c: per_class if c in augment_classes else 0 for c in classes}
        target = {c: counts[c] + need[c] for c in classes}
    elif plan == "balance":
        need, target = need_balance()
    elif plan == "add":
        need, target = need_add()
    elif plan == "scale":
        need, target = need_scale()
    elif plan == "balance_add":
        need_b, _target_b = need_balance()
        need_a, _target_a = need_add()
        need = {c: need_b[c] + need_a[c] for c in classes}
        target = {c: counts[c] + need[c] for c in classes}
    elif plan == "balance_scale":
        need_b, _target_b = need_balance()
        need_s, _target_s = need_scale()
        need = {c: max(need_b[c], need_s[c]) for c in classes}
        target = {c: counts[c] + need[c] for c in classes}
    else:
        raise DatasetAugmentorCoreError(
            "plan must be one of: random, balance, add, scale, balance_add, balance_scale."
        )
    return dict(sorted(need.items())), dict(sorted(target.items())), augment_classes


def run_augmentation(
    recipe: AugmentationRecipe,
    *,
    albumentations_module=None,
    log: Optional[ProgressCallback] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> AugmentationResult:
    try:
        import albumentations as A
    except Exception as exc:
        if albumentations_module is None:
            raise DatasetAugmentorCoreError(
                "Albumentations is required for augmentation. Install it with: "
                "python tools/install_albumentations_gui.py "
                "(or manually: python -m pip install --no-deps -r requirements-albumentations-gui.txt)"
            ) from exc
        A = albumentations_module
    if albumentations_module is not None:
        A = albumentations_module

    dataset_root = recipe.dataset_root
    if not dataset_root.exists():
        raise DatasetAugmentorCoreError(f"Dataset root not found: {dataset_root}")

    output_root = recipe.output_root or dataset_root / time.strftime("aug_run_%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)
    preview_dir = output_root / "aug_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    random.seed(recipe.seed)
    np.random.seed(recipe.seed)

    samples = discover_samples(dataset_root, recipe.split)
    if not samples:
        raise DatasetAugmentorCoreError(
            f"No samples found in {dataset_root}. Expected images/<split>/ and labels/<split>/ layout."
        )
    labeled_samples = [sample for sample in samples if sample.objects]
    counts = compute_instance_counts(labeled_samples)
    if not counts:
        raise DatasetAugmentorCoreError("No labeled boxes found in the selected split(s).")

    need, target, augment_classes = compute_class_plan(
        counts,
        plan=recipe.plan,
        add_count=recipe.add_count,
        target_ratio=recipe.target_ratio,
        add_per_class=recipe.add_per_class,
        multiplier=recipe.multiplier,
        include_classes=recipe.include_classes,
        exclude_classes=recipe.exclude_classes,
    )
    total_requested = min(max(1, int(recipe.max_total_augs)), max(1, sum(need.values())))
    _emit(log, f"Discovered {len(samples)} source samples; {len(labeled_samples)} contain labels.")
    _emit(log, _format_plan_summary(counts, target, need, augment_classes, recipe.plan))

    transform = build_transform(A, recipe.transforms, recipe.min_visibility)

    if recipe.copy_originals:
        _copy_originals(samples, output_root)
        _emit(log, "Copied original samples to output root.")

    recipe_path = output_root / "augmentation_recipe.json"
    _write_recipe(recipe_path, recipe, output_root)
    manifest_path = output_root / "aug_manifest.csv"
    augmented_image_paths: list[Path] = []
    per_source_aug_count: dict[Path, int] = defaultdict(int)
    aug_counter = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        manifest = csv.writer(manifest_file)
        manifest.writerow(["aug_image", "aug_label", "split", "source_image", "source_label"])

        while aug_counter < recipe.max_total_augs and sum(need.values()) > 0:
            sample = _pick_source(labeled_samples, need, augment_classes, per_source_aug_count, recipe)
            if sample is None:
                _emit(log, "No eligible source images remain for the requested class plan.")
                break

            image = cv2.imread(str(sample.image_path))
            if image is None:
                per_source_aug_count[sample.image_path] += 1
                continue

            transformed = _transform_sample(transform, image, sample)
            if transformed is None:
                per_source_aug_count[sample.image_path] += 1
                continue

            aug_img, aug_boxes, aug_classes, keypoints_flat, vis_flat, keypoint_counts = transformed
            aug_img = apply_domain_shift(aug_img, recipe.domain_shift)
            aug_img = apply_frame_and_patch_artifacts(aug_img, recipe.artifacts)

            per_class_gain: dict[int, int] = defaultdict(int)
            for class_id in aug_classes:
                if class_id in augment_classes and need.get(class_id, 0) > 0:
                    per_class_gain[class_id] += 1
            if sum(per_class_gain.values()) == 0:
                per_source_aug_count[sample.image_path] += 1
                continue

            aug_counter += 1
            per_source_aug_count[sample.image_path] += 1
            out_img_dir = output_root / "images" / sample.split
            out_lbl_dir = output_root / "labels" / sample.split
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            new_stem = f"{sample.image_path.stem}_aug_{aug_counter:06d}"
            out_img_path = out_img_dir / f"{new_stem}{sample.image_path.suffix.lower()}"
            out_lbl_path = out_lbl_dir / f"{new_stem}.txt"
            cv2.imwrite(str(out_img_path), aug_img)
            write_label_objects(
                out_lbl_path,
                boxes=aug_boxes,
                classes=aug_classes,
                keypoints=keypoints_flat,
                vis_flat=vis_flat,
                keypoint_counts=keypoint_counts,
                img_w=aug_img.shape[1],
                img_h=aug_img.shape[0],
            )
            manifest.writerow(
                [
                    out_img_path.name,
                    out_lbl_path.name,
                    sample.split,
                    sample.image_path.name,
                    sample.label_path.name,
                ]
            )
            augmented_image_paths.append(out_img_path)

            for class_id, gain in per_class_gain.items():
                need[class_id] = max(0, need[class_id] - gain)

            if progress is not None and (aug_counter % 25 == 0 or aug_counter == total_requested):
                progress(aug_counter, total_requested)
            if aug_counter % 200 == 0:
                _emit(log, f"Saved {aug_counter} augmentations; remaining needed instances: {sum(need.values())}")

    _write_previews(augmented_image_paths, output_root, preview_dir, recipe.preview_count)
    _emit(log, f"Augmentation complete. Created {aug_counter} samples. Output: {output_root}")
    return AugmentationResult(
        output_root=output_root,
        augmented_samples=aug_counter,
        original_counts=counts,
        target_counts=target,
        remaining_need=dict(sorted(need.items())),
        manifest_path=manifest_path,
        recipe_path=recipe_path,
        preview_dir=preview_dir,
    )


def build_transform(A, settings: TransformSettings, min_visibility: float):
    transforms = []
    if settings.horizontal_flip_prob > 0:
        transforms.append(A.HorizontalFlip(p=_prob(settings.horizontal_flip_prob)))
    if settings.affine_prob > 0:
        transforms.append(
            A.Affine(
                scale=(float(settings.scale_min), float(settings.scale_max)),
                translate_percent=(0.0, max(0.0, float(settings.translate_percent))),
                rotate=(-abs(float(settings.rotate_degrees)), abs(float(settings.rotate_degrees))),
                shear=(-abs(float(settings.shear_degrees)), abs(float(settings.shear_degrees))),
                p=_prob(settings.affine_prob),
            )
        )
    if settings.brightness_contrast_prob > 0:
        transforms.append(A.RandomBrightnessContrast(p=_prob(settings.brightness_contrast_prob)))
    if settings.hue_saturation_prob > 0:
        transforms.append(A.HueSaturationValue(p=_prob(settings.hue_saturation_prob)))
    if settings.blur_prob > 0:
        transforms.append(A.GaussianBlur(blur_limit=(3, 5), p=_prob(settings.blur_prob)))
    if settings.gauss_noise_prob > 0:
        transforms.append(_make_gauss_noise(A, p=_prob(settings.gauss_noise_prob)))

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels", "box_indices"],
            min_visibility=float(min_visibility),
        ),
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def apply_domain_shift(image: np.ndarray, settings: DomainShiftSettings) -> np.ndarray:
    out = image
    if settings.white_mouse:
        intensity = float(settings.invert_intensity)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        inv_v = 255 - v
        v_final = cv2.addWeighted(inv_v, intensity, v, 1.0 - intensity, 0) if intensity < 1.0 else inv_v
        out = cv2.cvtColor(cv2.merge([h, s, v_final]), cv2.COLOR_HSV2BGR)
    if settings.red_light:
        red_boost = float(settings.red_boost)
        bg_suppress = float(settings.bg_suppress)
        b, g, r = cv2.split(out)
        r = np.clip(r.astype(np.float32) * red_boost, 0, 255).astype(np.uint8)
        g = np.clip(g.astype(np.float32) * bg_suppress, 0, 255).astype(np.uint8)
        b = np.clip(b.astype(np.float32) * bg_suppress, 0, 255).astype(np.uint8)
        out = cv2.merge([b, g, r])
    if settings.bw and random.random() < _prob(settings.bw_prob):
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return out


def apply_frame_and_patch_artifacts(image: np.ndarray, settings: ArtifactSettings) -> np.ndarray:
    out = image
    if random.random() < _prob(settings.frame_noise_prob):
        out = _apply_whole_frame_noise(out, settings.frame_noise_std)
    if random.random() < _prob(settings.frame_banding_prob):
        out = _apply_frame_banding(out, settings.frame_banding_amp)
    if random.random() < _prob(settings.regional_noise_prob):
        out = _apply_regional_noise(
            out,
            settings.occlusion_count,
            settings.occlusion_min_frac,
            settings.occlusion_max_frac,
            settings.regional_noise_std,
        )
    if random.random() < _prob(settings.occlusion_prob):
        out = _apply_occlusions(
            out,
            settings.occlusion_count,
            settings.occlusion_min_frac,
            settings.occlusion_max_frac,
            settings.occlusion_mode,
        )
    return out


def _transform_sample(transform, image: np.ndarray, sample: SourceSample):
    objects = sample.objects
    bboxes = [obj.bbox for obj in objects]
    classes = [obj.class_id for obj in objects]
    original_keypoint_counts = [len(obj.keypoints) for obj in objects]
    original_vis: list[int] = []
    original_keypoints: list[tuple[float, float]] = []
    object_kp_offsets: list[int] = []
    running_offset = 0
    for obj in objects:
        object_kp_offsets.append(running_offset)
        for kx, ky, kv in obj.keypoints:
            original_keypoints.append((kx, ky))
            original_vis.append(kv)
        running_offset += len(obj.keypoints)

    box_indices = list(range(len(bboxes)))
    result = transform(
        image=image,
        bboxes=bboxes,
        class_labels=classes,
        box_indices=box_indices,
        keypoints=original_keypoints,
    )
    aug_img = result["image"]
    aug_boxes = [list(box) for box in result["bboxes"]]
    aug_classes = [int(float(c)) for c in result["class_labels"]]
    aug_keypoints = list(result["keypoints"])
    surviving_box_indices = [int(idx) for idx in result.get("box_indices", [])]
    if not aug_boxes:
        return None

    aug_h, aug_w = aug_img.shape[:2]
    keypoint_counts: list[int] = []
    vis_flat: list[int] = []
    keypoints_flat: list[tuple[float, float]] = []
    for orig_idx in surviving_box_indices:
        if orig_idx < 0 or orig_idx >= len(original_keypoint_counts):
            continue
        start = object_kp_offsets[orig_idx]
        count = original_keypoint_counts[orig_idx]
        keypoint_counts.append(int(count))
        for k in range(count):
            flat_idx = start + k
            if flat_idx >= len(aug_keypoints):
                keypoints_flat.append((0.0, 0.0))
                vis_flat.append(0)
                continue
            kx_px, ky_px = aug_keypoints[flat_idx]
            original_visibility = original_vis[flat_idx] if flat_idx < len(original_vis) else 0
            in_frame = (0.0 <= float(kx_px) <= float(aug_w)) and (0.0 <= float(ky_px) <= float(aug_h))
            vis_flat.append(0 if int(original_visibility) == 0 or not in_frame else int(original_visibility))
            keypoints_flat.append((float(kx_px), float(ky_px)))
    return aug_img, aug_boxes, aug_classes, keypoints_flat, vis_flat, keypoint_counts


def _copy_originals(samples: Sequence[SourceSample], output_root: Path) -> None:
    for sample in samples:
        out_img_dir = output_root / "images" / sample.split
        out_lbl_dir = output_root / "labels" / sample.split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image_path, out_img_dir / sample.image_path.name)
        if sample.label_path.exists():
            shutil.copy2(sample.label_path, out_lbl_dir / sample.label_path.name)


def _find_images(img_dir: Path) -> list[Path]:
    out: list[Path] = []
    for ext in IMG_EXTS:
        out.extend(sorted(img_dir.glob(f"*{ext}")))
    return out


def _pick_source(
    samples: Sequence[SourceSample],
    need: dict[int, int],
    augment_classes: set[int],
    per_source_aug_count: dict[Path, int],
    recipe: AugmentationRecipe,
) -> Optional[SourceSample]:
    candidates: list[SourceSample] = []
    weights: list[int] = []
    for sample in samples:
        if per_source_aug_count[sample.image_path] >= int(recipe.max_augs_per_source_image):
            continue
        present = {obj.class_id for obj in sample.objects}.intersection(augment_classes)
        weight = sum(need.get(class_id, 0) for class_id in present)
        if weight > 0:
            candidates.append(sample)
            weights.append(weight)
    if not candidates:
        return None
    return random.choices(candidates, weights=weights, k=1)[0]


def _write_recipe(recipe_path: Path, recipe: AugmentationRecipe, output_root: Path) -> None:
    payload = asdict(recipe)
    payload["dataset_root"] = str(recipe.dataset_root)
    payload["output_root"] = str(output_root)
    recipe_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_previews(
    augmented_image_paths: Sequence[Path],
    output_root: Path,
    preview_dir: Path,
    preview_count: int,
) -> None:
    if not augmented_image_paths or int(preview_count) <= 0:
        return
    chosen = random.sample(list(augmented_image_paths), k=min(int(preview_count), len(augmented_image_paths)))
    for image_path in chosen:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        rel_parts = image_path.relative_to(output_root).parts
        split = rel_parts[1] if len(rel_parts) >= 3 else "train"
        label_path = output_root / "labels" / split / f"{image_path.stem}.txt"
        h, w = image.shape[:2]
        objects = read_label_objects(label_path, img_w=w, img_h=h)
        boxes = [obj.bbox for obj in objects]
        classes = [obj.class_id for obj in objects]
        preview = _draw_boxes(image, boxes, classes)
        cv2.imwrite(str(preview_dir / f"{image_path.stem}_preview.jpg"), preview)


def _draw_boxes(image_bgr: np.ndarray, boxes: Sequence[Sequence[float]], classes: Sequence[int]) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    for box, class_id in zip(boxes, classes):
        x1, y1, x2, y2 = _yolo_to_xyxy(box, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            str(int(class_id)),
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def _yolo_to_xyxy(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = [float(v) for v in box[:4]]
    x1 = int((cx - bw / 2.0) * width)
    y1 = int((cy - bh / 2.0) * height)
    x2 = int((cx + bw / 2.0) * width)
    y2 = int((cy + bh / 2.0) * height)
    return (
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(0, min(x2, width - 1)),
        max(0, min(y2, height - 1)),
    )


def _make_gauss_noise(A, p: float):
    params = inspect.signature(A.GaussNoise).parameters
    if "var_limit" in params:
        return A.GaussNoise(var_limit=(5.0, 25.0), p=p)
    if "std_range" in params:
        return A.GaussNoise(std_range=(0.01, 0.05), p=p)
    return A.GaussNoise(p=p)


def _clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def _patch_bounds(image_shape, min_frac: float, max_frac: float) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    min_frac = max(0.001, min(float(min_frac), 1.0))
    max_frac = max(min_frac, min(float(max_frac), 1.0))
    patch_w = random.randint(max(1, int(w * min_frac)), max(1, int(w * max_frac)))
    patch_h = random.randint(max(1, int(h * min_frac)), max(1, int(h * max_frac)))
    x1 = random.randint(0, max(0, w - patch_w))
    y1 = random.randint(0, max(0, h - patch_h))
    return x1, y1, x1 + patch_w, y1 + patch_h


def _apply_whole_frame_noise(image: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.normal(0.0, float(std), image.shape).astype(np.float32)
    return _clip_uint8(image.astype(np.float32) + noise)


def _apply_frame_banding(image: np.ndarray, amplitude: float) -> np.ndarray:
    h = image.shape[0]
    period = random.uniform(8.0, 28.0)
    phase = random.uniform(0.0, 2.0 * np.pi)
    y = np.arange(h, dtype=np.float32)
    band = np.sin((2.0 * np.pi * y / period) + phase) * float(amplitude)
    return _clip_uint8(image.astype(np.float32) + band.reshape(h, 1, 1))


def _apply_regional_noise(
    image: np.ndarray,
    count: int,
    min_frac: float,
    max_frac: float,
    std: float,
) -> np.ndarray:
    out = image.copy()
    for _ in range(max(1, int(count))):
        x1, y1, x2, y2 = _patch_bounds(out.shape, min_frac, max_frac)
        patch = out[y1:y2, x1:x2].astype(np.float32)
        noise = np.random.normal(0.0, float(std), patch.shape).astype(np.float32)
        out[y1:y2, x1:x2] = _clip_uint8(patch + noise)
    return out


def _apply_occlusions(
    image: np.ndarray,
    count: int,
    min_frac: float,
    max_frac: float,
    mode: str,
) -> np.ndarray:
    out = image.copy()
    modes = ["black", "gray", "noise"] if mode == "random" else [mode]
    for _ in range(max(1, int(count))):
        x1, y1, x2, y2 = _patch_bounds(out.shape, min_frac, max_frac)
        selected = random.choice(modes)
        if selected == "black":
            fill = np.zeros_like(out[y1:y2, x1:x2])
        elif selected == "gray":
            fill = np.full_like(out[y1:y2, x1:x2], random.randint(35, 210))
        else:
            fill = np.random.randint(0, 256, out[y1:y2, x1:x2].shape, dtype=np.uint8)
        out[y1:y2, x1:x2] = fill
    return out


def _format_plan_summary(
    counts: dict[int, int],
    target: dict[int, int],
    need: dict[int, int],
    augment_classes: set[int],
    plan: str,
) -> str:
    lines = [f"Class-aware plan: {plan}"]
    for class_id in sorted(counts):
        flag = " (augment)" if class_id in augment_classes else ""
        lines.append(
            f"  class {class_id}: {counts[class_id]} -> target {target[class_id]} "
            f"(need +{need[class_id]}){flag}"
        )
    return "\n".join(lines)


def _prob(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _emit(log: Optional[ProgressCallback], message: str) -> None:
    if log is not None:
        log(message)


__all__ = [
    "ArtifactSettings",
    "AugmentationRecipe",
    "AugmentationResult",
    "DatasetAugmentorCoreError",
    "DomainShiftSettings",
    "TransformSettings",
    "apply_domain_shift",
    "apply_frame_and_patch_artifacts",
    "compute_class_plan",
    "compute_instance_counts",
    "discover_samples",
    "parse_class_set",
    "read_label_objects",
    "run_augmentation",
    "sanitize_yolo_bbox",
    "write_label_objects",
]

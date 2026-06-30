from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from integra_pose.plugins.plugin_dataset_augmentor_lab.core import (
    AugmentationRecipe,
    SourceSample,
    LabelObject,
    TransformSettings,
    compute_class_plan,
    run_augmentation,
    write_label_objects,
)
from integra_pose.plugins.plugin_dataset_augmentor_lab import core


class _NoOpTransform:
    def __call__(self, *, image, bboxes, class_labels, box_indices, keypoints):
        return {
            "image": image,
            "bboxes": bboxes,
            "class_labels": class_labels,
            "box_indices": box_indices,
            "keypoints": keypoints,
        }


class _FakeAlbumentations:
    class BboxParams:
        def __init__(self, **_kwargs):
            pass

    class KeypointParams:
        def __init__(self, **_kwargs):
            pass

    class Compose:
        def __init__(self, _transforms, **_kwargs):
            pass

        def __call__(self, **kwargs):
            return _NoOpTransform()(**kwargs)

    class HorizontalFlip:
        def __init__(self, **_kwargs):
            pass

    class Affine:
        def __init__(self, **_kwargs):
            pass

    class RandomBrightnessContrast:
        def __init__(self, **_kwargs):
            pass

    class HueSaturationValue:
        def __init__(self, **_kwargs):
            pass

    class GaussianBlur:
        def __init__(self, **_kwargs):
            pass

    class GaussNoise:
        def __init__(self, p=0.0):
            pass


def test_compute_class_plan_balance_and_filters() -> None:
    need, target, augment_classes = compute_class_plan(
        {0: 10, 1: 4, 2: 2},
        plan="balance",
        add_count=100,
        target_ratio=0.7,
        add_per_class=0,
        multiplier=1.0,
        include_classes="all",
        exclude_classes="0",
    )

    assert augment_classes == {1, 2}
    assert target == {0: 7, 1: 7, 2: 7}
    assert need == {0: 0, 1: 3, 2: 5}


def test_dropped_bbox_keeps_surviving_keypoints_aligned(tmp_path: Path) -> None:
    sample = SourceSample(
        image_path=tmp_path / "frame.jpg",
        label_path=tmp_path / "frame.txt",
        split="train",
        objects=[
            LabelObject(0, [0.25, 0.5, 0.2, 0.2], [(10.0, 20.0, 2), (11.0, 21.0, 2)]),
            LabelObject(1, [0.75, 0.5, 0.2, 0.2], [(70.0, 20.0, 2), (71.0, 21.0, 2)]),
        ],
    )

    def drop_first_transform(**_kwargs):
        return {
            "image": np.zeros((100, 100, 3), dtype=np.uint8),
            "bboxes": [[0.75, 0.5, 0.2, 0.2]],
            "class_labels": [1],
            "box_indices": [1],
            "keypoints": [(10.0, 20.0), (11.0, 21.0), (70.0, 20.0), (71.0, 21.0)],
        }

    aug_img, boxes, classes, keypoints, vis, counts = core._transform_sample(
        drop_first_transform,
        np.zeros((100, 100, 3), dtype=np.uint8),
        sample,
    )
    out_label = tmp_path / "out.txt"
    write_label_objects(
        out_label,
        boxes=boxes,
        classes=classes,
        keypoints=keypoints,
        vis_flat=vis,
        keypoint_counts=counts,
        img_w=aug_img.shape[1],
        img_h=aug_img.shape[0],
    )

    parts = out_label.read_text(encoding="utf-8").strip().split()
    assert parts[0] == "1"
    assert parts[5:11] == ["0.700000", "0.200000", "2", "0.710000", "0.210000", "2"]


def test_run_augmentation_writes_manifest_and_recipe(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    image_dir = dataset / "images" / "train"
    label_dir = dataset / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image_path = image_dir / "frame.jpg"
    cv2.imwrite(str(image_path), np.full((48, 64, 3), 128, dtype=np.uint8))
    (label_dir / "frame.txt").write_text(
        "0 0.300000 0.500000 0.200000 0.200000\n"
        "1 0.700000 0.500000 0.200000 0.200000\n",
        encoding="utf-8",
    )

    output = tmp_path / "augmented"
    result = run_augmentation(
        AugmentationRecipe(
            dataset_root=dataset,
            output_root=output,
            plan="add",
            add_per_class=1,
            preview_count=0,
            copy_originals=False,
            transforms=TransformSettings(
                horizontal_flip_prob=0.0,
                affine_prob=0.0,
                brightness_contrast_prob=0.0,
                hue_saturation_prob=0.0,
                blur_prob=0.0,
                gauss_noise_prob=0.0,
            ),
        ),
        albumentations_module=_FakeAlbumentations,
    )

    assert result.augmented_samples == 1
    assert result.manifest_path.exists()
    assert result.recipe_path.exists()
    assert (output / "images" / "train" / "frame_aug_000001.jpg").exists()
    assert (output / "labels" / "train" / "frame_aug_000001.txt").exists()
    assert "aug_image,aug_label,split,source_image,source_label" in result.manifest_path.read_text(encoding="utf-8")

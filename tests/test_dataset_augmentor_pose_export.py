from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from integra_pose.plugins.plugin_dataset_augmentor_lab import core
from integra_pose.plugins.plugin_dataset_augmentor_lab.dataset_io import load_metadata


class Identity:
    def __call__(self, **kwargs):
        return kwargs


def dataset(tmp_path):
    root = tmp_path / "dataset"
    for split, value in (("train", 40), ("val", 80), ("test", 120)):
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        cv2.imwrite(str(images / "mouse.png"), np.full((40, 60, 3), value, np.uint8))
        (labels / "mouse.txt").write_text("0 0.5 0.5 0.8 0.8 0.2 0.3 2 0.8 0.4 1\n")
    cv2.imwrite(str(root / "images/train/background.png"), np.full((40, 60, 3), 10, np.uint8))
    (root / "labels/train/background.txt").write_text("")
    (root / "data.yaml").write_text(yaml.safe_dump(dict(names={0: "Digging"}, kpt_shape=[2, 3],
                                                       flip_idx=[1, 0], train="images/train",
                                                       val="images/val", test="images/test",
                                                       kpt_names={0: ["Left", "Right"]}, skeleton=[[0, 1]])))
    return root


def recipe(root, output, **kwargs):
    return core.AugmentationRecipe(dataset_root=root, output_root=output,
                                   plan="target_total", target_total_images=5,
                                   preview_count=0,
                                   transforms=core.TransformSettings(horizontal_flip_prob=0), **kwargs)


def identity_engine(monkeypatch):
    monkeypatch.setattr(core, "build_transform", lambda *_: Identity())


def test_exact_target_keeps_backgrounds_and_holdouts(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    identity_engine(monkeypatch)
    output = tmp_path / "output"
    result = core.run_augmentation(recipe(root, output), albumentations_module=object())
    assert result.augmented_samples == 3
    assert len(list((output / "images/train").iterdir())) == 5
    assert (output / "labels/train/background.txt").read_bytes() == b""
    for split in ("val", "test"):
        for kind, suffix in (("images", "png"), ("labels", "txt")):
            rel = f"{kind}/{split}/mouse.{suffix}"
            assert (output / rel).read_bytes() == (root / rel).read_bytes()
    assert yaml.safe_load((output / "data.yaml").read_text())["kpt_shape"] == [2, 3]
    assert not (output / "EXPORT_INCOMPLETE.txt").exists()


@pytest.mark.parametrize("split, target, expected_new", [("val", 3, 2), ("test", 3, 2), ("all", 7, 3)])
def test_other_splits_can_be_selected_explicitly(tmp_path, monkeypatch, split, target, expected_new):
    root = dataset(tmp_path)
    identity_engine(monkeypatch)
    output = tmp_path / "output"
    result = core.run_augmentation(replace(recipe(root, output), split=split, target_total_images=target),
                                   albumentations_module=object())
    assert result.augmented_samples == expected_new
    selected = ("train", "val", "test") if split == "all" else (split,)
    assert sum(len(list((output / "images" / s).iterdir())) for s in selected) == target
    if split != "all":
        assert not list((output / "images/train").glob("*_aug_*"))


def test_mirror_swaps_coordinates_and_visibility(tmp_path):
    sample = core.SourceSample(tmp_path / "a.png", tmp_path / "a.txt", "train", [
        core.LabelObject(0, [0.4, 0.5, 0.2, 0.2], [(10, 20, 2), (70, 30, 1), (0, 0, 0)])])
    image = np.zeros((80, 100, 3), np.uint8)
    image[:, :20] = 255
    result = core._transform_sample(Identity(), image, sample, horizontal_flip_prob=1, flip_idx=[1, 0, 2])
    assert result[3] == [(29.0, 30.0), (89.0, 20.0), (99.0, 0.0)]
    assert result[4] == [1, 2, 0]
    assert result[1][0][0] == pytest.approx(0.6)
    assert np.all(result[0][:, -20:] == 255)


def test_outside_points_keep_slots_and_write_zero_triples(tmp_path):
    sample = core.SourceSample(tmp_path / "a.png", tmp_path / "a.txt", "train", [
        core.LabelObject(0, [0.5, 0.5, 0.8, 0.8], [(10, 20, 2), (20, 30, 1)])])
    def outside(**kwargs):
        kwargs["keypoints"] = [(-1, 20), (100, 30)]
        return kwargs
    result = core._transform_sample(outside, np.zeros((80, 100, 3), np.uint8), sample)
    assert result[4] == [0, 0] and result[5] == [2]
    output = tmp_path / "pose.txt"
    core.write_label_objects(output, boxes=result[1], classes=result[2], keypoints=result[3],
                             vis_flat=result[4], keypoint_counts=result[5], img_w=100, img_h=80)
    assert list(map(float, output.read_text().split()[5:])) == [0, 0, 0, 0, 0, 0]


def test_occluder_marks_visible_points_occluded(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    identity_engine(monkeypatch)
    monkeypatch.setattr(core, "_patch_bounds", lambda *_: (0, 0, 60, 40))
    output = tmp_path / "out"
    spec = replace(recipe(root, output), artifacts=core.ArtifactSettings(occlusion_prob=1, occlusion_mode="black"))
    core.run_augmentation(spec, albumentations_module=object())
    for path in (output / "labels/train").glob("*_aug_*"):
        assert path.read_text().split()[7::3] == ["1", "1"]


def test_missing_mapping_blocks_pose_flip(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    metadata = yaml.safe_load((root / "data.yaml").read_text())
    metadata.pop("flip_idx")
    (root / "data.yaml").write_text(yaml.safe_dump(metadata))
    spec = replace(recipe(root, tmp_path / "out"), transforms=core.TransformSettings(horizontal_flip_prob=1))
    with pytest.raises(core.DatasetAugmentorCoreError, match="flip_idx"):
        core.run_augmentation(spec, albumentations_module=object())
    assert not spec.output_root.exists()


def test_bad_keypoint_count_fails_before_writing(tmp_path):
    root = dataset(tmp_path)
    (root / "labels/train/mouse.txt").write_text("0 0.5 0.5 0.8 0.8 0.2 0.3 2\n")
    with pytest.raises(ValueError, match="same keypoint count"):
        core.run_augmentation(recipe(root, tmp_path / "out"), albumentations_module=object())
    assert not (tmp_path / "out").exists()


def test_failed_write_leaves_incomplete_marker(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    identity_engine(monkeypatch)
    monkeypatch.setattr(core.cv2, "imwrite", lambda *_: False)
    with pytest.raises(core.DatasetAugmentorCoreError, match="write failed"):
        core.run_augmentation(recipe(root, tmp_path / "out"), albumentations_module=object())
    assert (tmp_path / "out/EXPORT_INCOMPLETE.txt").exists()


def test_existing_output_is_never_overwritten(tmp_path):
    root = dataset(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("unchanged")
    with pytest.raises(core.DatasetAugmentorCoreError, match="never overwritten"):
        core.run_augmentation(recipe(root, output), albumentations_module=object())
    assert (output / "keep.txt").read_text() == "unchanged"


def test_random_plan_does_not_round_up():
    need, _, _ = core.compute_class_plan({i: 100 for i in range(6)}, plan="random", add_count=6730,
                                        target_ratio=1, add_per_class=0, multiplier=1,
                                        include_classes="all", exclude_classes="")
    assert sum(need.values()) == 6730


def test_real_albumentations_seed_repeats(tmp_path):
    A = pytest.importorskip("albumentations")
    root = dataset(tmp_path)
    generated = []
    for name in ("one", "two"):
        output = tmp_path / name
        core.run_augmentation(replace(recipe(root, output), preview_count=1), albumentations_module=A)
        generated.append([(p.name, p.read_bytes()) for p in sorted((output / "images/train").glob("*_aug_*"))])
        assert len(list((output / "aug_preview").iterdir())) == 1
    assert generated[0] == generated[1]


def test_bad_flip_mapping_is_rejected(tmp_path):
    root = dataset(tmp_path)
    meta = yaml.safe_load((root / "data.yaml").read_text())
    meta["flip_idx"] = [0, 0]
    (root / "data.yaml").write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match="flip_idx"):
        load_metadata(root)


def test_recording_overlap_is_rejected(tmp_path):
    root = dataset(tmp_path)
    (root / "manifest.csv").write_text("image,split,source_group\nimages/train/mouse.png,train,recording1\nimages/val/mouse.png,val,recording1\n")
    with pytest.raises(ValueError, match="Recording group crosses"):
        core.run_augmentation(recipe(root, tmp_path / "out"), albumentations_module=object())
    assert not (tmp_path / "out").exists()


def test_behavior_filter_controls_new_images_only(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    meta = yaml.safe_load((root / "data.yaml").read_text())
    meta["names"][1] = "Sniffing"
    (root / "data.yaml").write_text(yaml.safe_dump(meta))
    cv2.imwrite(str(root / "images/train/sniff.png"), np.full((40, 60, 3), 160, np.uint8))
    (root / "labels/train/sniff.txt").write_text("1 0.5 0.5 0.8 0.8 0.2 0.3 2 0.8 0.4 1\n")
    identity_engine(monkeypatch)
    output = tmp_path / "out"
    core.run_augmentation(replace(recipe(root, output), include_classes="1"), albumentations_module=object())
    assert (output / "images/train/mouse.png").exists()
    assert len(list((output / "images/train").iterdir())) == 5
    assert all(p.read_text().split()[0] == "1" for p in (output / "labels/train").glob("*_aug_*"))


def test_exact_plan_counts_images_not_objects(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    path = root / "labels/train/mouse.txt"
    path.write_text(path.read_text() * 2)
    identity_engine(monkeypatch)
    result = core.run_augmentation(recipe(root, tmp_path / "out"), albumentations_module=object())
    assert result.augmented_samples == 3


def test_gui_behavior_names_and_target_controls(tmp_path):
    import tkinter as tk
    from integra_pose.plugins.plugin_dataset_augmentor_lab.ui import DatasetAugmentorLabWindow
    try:
        owner = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display unavailable")
    owner.withdraw()
    try:
        window = DatasetAugmentorLabWindow(None, parent=owner)
        window.withdraw()
        window._dataset_root_var.set(str(dataset(tmp_path)))
        assert window._split_var.get() == "train"
        window._plan_var.set("target_total")
        window._target_total_var.set("40000")
        assert window._collect_recipe().target_total_images == 40000
        window._choose_behaviors()
        owner.update()
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        checkboxes = [w for w in descendants(window) if w.winfo_class() == "TCheckbutton"]
        assert any(w.cget("text") == "0: Digging" for w in checkboxes)
        window.destroy()
    finally:
        owner.destroy()

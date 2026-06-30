import json
from pathlib import Path

import numpy as np
import pytest

from integra_pose.plugins.plugin_tandem_yolo_toolkit.yolo_temporal_classifier.data_y import (
    load_hdf5_window_payload,
    load_n_manifest,
    validate_n_npz_payload,
)

h5py = pytest.importorskip("h5py")


def test_hdf5_manifest_window_slice_validates(tmp_path: Path) -> None:
    h5_path = tmp_path / "sequence_h5" / "video_a.h5"
    h5_path.parent.mkdir()
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["schema_version"] = "yolo-temporal-full-video-v1"
        h5.attrs["storage_format"] = "hdf5"
        h5.attrs["body_length_px"] = 20.0
        h5.attrs["n_animals"] = 2
        h5.create_dataset("animal_keypoints", data=np.zeros((5, 2, 7, 3), dtype=np.float32))
        h5.create_dataset("animal_keypoints_raw", data=np.zeros((5, 2, 7, 2), dtype=np.float32))
        h5.create_dataset("animal_keypoints_conf", data=np.ones((5, 2, 7), dtype=np.float32))
        h5.create_dataset("animal_mask", data=np.ones((5, 2), dtype=bool))
        h5.create_dataset("pose_mask", data=np.ones((5, 2), dtype=bool))
        h5.create_dataset("pose_conf", data=np.ones((5, 2), dtype=np.float32))
        h5.create_dataset("crop_conf", data=np.ones((5, 2), dtype=np.float32))
        h5.create_dataset("track_conf", data=np.ones((5, 2), dtype=np.float32))
        h5.create_dataset("track_age", data=np.zeros((5, 2), dtype=np.float32))
        rel = np.zeros((5, 2, 2, 11), dtype=np.float32)
        rel[:, :, :, 0] = np.arange(5, dtype=np.float32)[:, None, None]
        h5.create_dataset("relation_features", data=rel)
        h5.create_dataset("relation_pose_mask", data=np.ones((5, 2, 2), dtype=bool))
        h5.create_dataset("relation_pose_conf", data=np.ones((5, 2, 2), dtype=np.float32))
        relation_present = np.ones((5, 2, 2), dtype=bool)
        relation_present[:, 0, 0] = False
        relation_present[:, 1, 1] = False
        h5.create_dataset("relation_present", data=relation_present)
        h5.create_dataset("bbox_xyxy_animals", data=np.zeros((5, 2, 4), dtype=np.int32))
        h5.create_dataset("bbox_xyxy_group", data=np.zeros((5, 4), dtype=np.int32))
        h5.create_dataset("crop_xyxy_animals", data=np.zeros((5, 2, 4), dtype=np.int32))
        h5.create_dataset("crop_xyxy_group", data=np.zeros((5, 4), dtype=np.int32))
        h5.create_dataset("track_ids", data=np.zeros((5, 2), dtype=np.int32))

    manifest = {
        "class_to_idx": {"other": 0},
        "idx_to_class": {"0": "other"},
        "meta": {"schema_version": "yolo-temporal-full-video-v1"},
        "splits": {
            "train": [
                {
                    "id": "other_video_a_f000001",
                    "label": 0,
                    "class_name": "other",
                    "storage_format": "hdf5",
                    "sequence_h5": "sequence_h5/video_a.h5",
                    "start_frame": 1,
                    "end_frame": 3,
                    "fps": 30.0,
                    "source_video": "video_a",
                }
            ],
            "val": [],
        },
    }
    manifest_path = tmp_path / "sequence_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    splits, _class_to_idx, _idx_to_class, _meta = load_n_manifest(manifest_path)
    sample = splits["train"][0]
    payload = load_hdf5_window_payload(sample)

    assert "group_frames" not in payload
    assert "animal_frames" not in payload
    assert payload["relation_features"][:, 0, 1, 0].tolist() == [1.0, 2.0, 3.0]
    validate_n_npz_payload(
        payload,
        sample_id=sample.id,
        num_frames=3,
        n_animals=2,
        num_keypoints=7,
        rel_feature_dim=11,
        require_schema_version=True,
        require_visual=False,
    )


def test_training_manifest_rejects_source_manifest_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "source_manifest.csv"
    csv_path.write_text("split,video_path,annotation_path\ntrain,video.mp4,video.annot\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Prepare Full Video"):
        load_n_manifest(csv_path)

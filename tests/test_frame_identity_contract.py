from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from integra_pose.utils.detection_contract import (
    DetectionContractError,
    enforce_ultralytics_max_det,
)
from integra_pose.utils.frame_identity import (
    FRAME_LABEL_MANIFEST_FILENAME,
    FrameIdentityError,
    extract_model_class_metadata,
    frame_label_filename,
    load_frame_label_class_metadata,
    load_frame_label_class_names,
    load_frame_label_manifest,
    normalize_class_name_metadata,
    resolve_frame_label_indices,
    write_frame_label_manifest,
)


class _Boxes:
    def __init__(self, confidence) -> None:
        self.conf = confidence

    def __len__(self) -> int:
        return len(self.conf)


class _Result:
    def __init__(self, confidence, row_ids=None) -> None:
        self.boxes = _Boxes(confidence)
        self.row_ids = list(range(len(confidence))) if row_ids is None else list(row_ids)

    def __getitem__(self, indices):
        return _Result(
            [self.boxes.conf[index] for index in indices],
            [self.row_ids[index] for index in indices],
        )


def test_canonical_names_are_zero_based_for_numeric_source_stems() -> None:
    assert frame_label_filename("mouse_1.mp4", 0) == "mouse_1_frame_000000.txt"
    assert frame_label_filename("mouse_1.mp4", 1) == "mouse_1_frame_000001.txt"


def test_legacy_unindexed_first_frame_resolves_without_numeric_stem_alias() -> None:
    resolved = resolve_frame_label_indices(
        ["mouse_1.txt", "mouse_1_000001.txt", "mouse_1_000010.txt"],
        source="mouse_1.mp4",
    )
    assert resolved == {
        "mouse_1.txt": 0,
        "mouse_1_000001.txt": 1,
        "mouse_1_000010.txt": 10,
    }


def test_native_ultralytics_video_suffixes_are_normalized_to_zero_based() -> None:
    resolved = resolve_frame_label_indices(
        ["mouse_1_1.txt", "mouse_1_5.txt"],
        source="mouse_1.mp4",
    )
    assert resolved == {"mouse_1_1.txt": 0, "mouse_1_5.txt": 4}


def test_source_scope_excludes_similarly_named_and_other_video_files() -> None:
    resolved = resolve_frame_label_indices(
        [
            "mouse_1_frame_000000.txt",
            "mouse_10_frame_000001.txt",
            "other_frame_000002.txt",
        ],
        source="mouse_1.mp4",
    )

    assert resolved == {"mouse_1_frame_000000.txt": 0}


def test_duplicate_frame_identities_fail_instead_of_merging() -> None:
    with pytest.raises(FrameIdentityError, match="same frame"):
        resolve_frame_label_indices(
            ["mouse_frame_000001.txt", "other_frame_000001.txt"]
        )


def test_frame_label_manifest_round_trip(tmp_path: Path) -> None:
    path = write_frame_label_manifest(
        tmp_path,
        source="mouse_1.mp4",
        max_det=1,
        class_names={0: "Sniffing", 1: "Wall-Rearing", 2: "Ambulatory"},
        class_names_source="model.names",
        model_task="detect",
    )
    assert path.name == FRAME_LABEL_MANIFEST_FILENAME
    payload = load_frame_label_manifest(tmp_path)
    assert payload["frame_index_base"] == 0
    assert payload["source_stem"] == "mouse_1"
    assert payload["max_det"] == 1
    assert payload["class_count"] == 3
    assert payload["class_names"] == {
        "0": "Sniffing",
        "1": "Wall-Rearing",
        "2": "Ambulatory",
    }
    assert load_frame_label_class_names(tmp_path) == [
        "Sniffing",
        "Wall-Rearing",
        "Ambulatory",
    ]
    metadata = load_frame_label_class_metadata(tmp_path)
    assert metadata["class_names_source"] == "model.names"
    assert metadata["model_task"] == "detect"


def test_legacy_frame_label_manifest_has_no_class_metadata(tmp_path: Path) -> None:
    write_frame_label_manifest(tmp_path, source="legacy.mp4", max_det=2)

    assert load_frame_label_class_metadata(tmp_path) == {}


def test_class_metadata_requires_dense_unique_names() -> None:
    with pytest.raises(FrameIdentityError, match="missing IDs: 1"):
        normalize_class_name_metadata({0: "Sniffing", 2: "Ambulatory"})
    with pytest.raises(FrameIdentityError, match="unique name"):
        normalize_class_name_metadata(["Sniffing", "sniffing"])
    with pytest.raises(FrameIdentityError, match="empty name"):
        normalize_class_name_metadata(["Sniffing", ""])


@pytest.mark.parametrize("task", ["detect", "pose"])
def test_model_class_metadata_supports_detection_and_pose(task: str) -> None:
    model = SimpleNamespace(
        names={0: "Sniffing", 1: "Wall-Rearing", 2: "Ambulatory"},
        task=task,
        model=SimpleNamespace(nc=3),
    )

    names, resolved_task = extract_model_class_metadata(model)

    assert names == ["Sniffing", "Wall-Rearing", "Ambulatory"]
    assert resolved_task == task


def test_model_class_metadata_rejects_name_count_mismatch() -> None:
    model = SimpleNamespace(
        names={0: "Sniffing", 1: "Wall-Rearing"},
        task="detect",
        model=SimpleNamespace(nc=3),
    )

    with pytest.raises(FrameIdentityError, match="declares 3 classes"):
        extract_model_class_metadata(model)


def test_detection_contract_retains_highest_confidence_rows() -> None:
    outcome = enforce_ultralytics_max_det(_Result([0.2, 0.95, 0.7]), 1)
    assert outcome.original_count == 3
    assert outcome.retained_count == 1
    assert outcome.dropped_count == 2
    assert outcome.result.row_ids == [1]


def test_detection_contract_rejects_arbitrary_cap_without_confidence() -> None:
    result = _Result([0.2, 0.95])
    result.boxes = type(
        "BoxesWithoutConfidence",
        (),
        {"conf": None, "__len__": lambda _self: 2},
    )()

    with pytest.raises(DetectionContractError, match="cannot be capped objectively"):
        enforce_ultralytics_max_det(result, 1)

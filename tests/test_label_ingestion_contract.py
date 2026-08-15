from __future__ import annotations

from pathlib import Path

import pytest

from integra_pose.utils.bout_analyzer import BoutAnalysisError, load_and_preprocess_data


VALID_ROW = "0 0.500000 0.500000 0.200000 0.200000 7\n"


def _load(labels: Path, *, source: str = "mouse_1.mp4", single: bool = False):
    return load_and_preprocess_data(
        None,
        str(labels),
        single_animal_mode=single,
        class_names={0: "behavior"},
        source_video=source,
    )[0]


def test_legacy_numeric_video_stem_maps_base_and_next_frame_uniquely(tmp_path: Path) -> None:
    (tmp_path / "mouse_1.txt").write_text(VALID_ROW, encoding="utf-8")
    (tmp_path / "mouse_1_000001.txt").write_text(VALID_ROW, encoding="utf-8")

    frame_df = _load(tmp_path)

    assert frame_df["frame"].tolist() == [0, 1]
    assert frame_df["source_label_file"].tolist() == ["mouse_1.txt", "mouse_1_000001.txt"]


def test_native_one_based_ultralytics_names_normalize_with_source_context(tmp_path: Path) -> None:
    (tmp_path / "mouse_1.txt").write_text(VALID_ROW, encoding="utf-8")
    (tmp_path / "mouse_2.txt").write_text(VALID_ROW, encoding="utf-8")

    frame_df = _load(tmp_path, source="mouse.mp4")

    assert frame_df["frame"].tolist() == [0, 1]


def test_duplicate_frame_files_fail_before_rows_are_merged(tmp_path: Path) -> None:
    (tmp_path / "mouse_1_frame_000001.txt").write_text(VALID_ROW, encoding="utf-8")
    (tmp_path / "mouse_1_img_000001.txt").write_text(VALID_ROW, encoding="utf-8")

    with pytest.raises(BoutAnalysisError, match="same frame"):
        _load(tmp_path)


def test_auxiliary_txt_is_ignored_but_cannot_qualify_a_folder(tmp_path: Path) -> None:
    (tmp_path / "classes.txt").write_text("behavior\n", encoding="utf-8")
    with pytest.raises(BoutAnalysisError, match="No frame-indexed"):
        _load(tmp_path)

    (tmp_path / "mouse_1_frame_000000.txt").write_text(VALID_ROW, encoding="utf-8")
    frame_df = _load(tmp_path)
    assert len(frame_df) == 1


def test_malformed_declared_detection_fails_instead_of_partial_success(tmp_path: Path) -> None:
    (tmp_path / "mouse_1_frame_000000.txt").write_text(
        "0 0.5 0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(BoutAnalysisError, match="expected at least 5 columns"):
        _load(tmp_path)


def test_txt_and_labels_csv_geometry_mismatch_fails(tmp_path: Path) -> None:
    (tmp_path / "mouse_1_frame_000000.txt").write_text(VALID_ROW, encoding="utf-8")
    (tmp_path / "labels.csv").write_text(
        "frame,class_id,x_center_n,y_center_n,width_n,height_n,bbox_conf,track_id\n"
        "0,0,0.4,0.5,0.2,0.2,0.9,7\n",
        encoding="utf-8",
    )

    with pytest.raises(BoutAnalysisError, match="TXT/CSV x_center mismatch"):
        _load(tmp_path)


def test_duplicate_track_identity_within_frame_fails(tmp_path: Path) -> None:
    (tmp_path / "mouse_1_frame_000000.txt").write_text(
        VALID_ROW + "1 0.600000 0.500000 0.200000 0.200000 7\n",
        encoding="utf-8",
    )

    with pytest.raises(BoutAnalysisError, match="same track ID and frame"):
        _load(tmp_path)

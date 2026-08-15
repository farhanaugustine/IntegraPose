from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from integra_pose.plugins.plugin_assisted_pose_curation.core import (
    ACTIVE_LEARNING_SCORING_MAX_DET,
    ActiveLearningCandidate,
    active_learning_frame_filename,
    source_video_identity,
)
from integra_pose.plugins.plugin_assisted_pose_curation.ui import (
    AnnotationIntegrityError,
    AssistedPoseCurationWindow,
)


class _Var:
    def __init__(self, value="") -> None:
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


def _window_without_tk() -> AssistedPoseCurationWindow:
    return object.__new__(AssistedPoseCurationWindow)


def test_same_stem_videos_get_distinct_source_ids_and_frame_names(tmp_path: Path) -> None:
    source_a = tmp_path / "session_a" / "trial.mp4"
    source_b = tmp_path / "session_b" / "trial.mp4"
    source_a.parent.mkdir()
    source_b.parent.mkdir()
    source_a.write_bytes(b"source-a")
    source_b.write_bytes(b"source-b")

    source_a_id = source_video_identity(source_a)
    source_b_id = source_video_identity(source_b)
    name_a = active_learning_frame_filename(12, video_path=source_a, source_id=source_a_id)
    name_b = active_learning_frame_filename(12, video_path=source_b, source_id=source_b_id)

    assert source_a_id != source_b_id
    assert name_a != name_b
    assert name_a.startswith(f"trial__src_{source_a_id}__frame_000012")
    assert name_b.startswith(f"trial__src_{source_b_id}__frame_000012")


def test_malformed_annotation_row_raises_instead_of_returning_partial_instances(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "frame.jpg"
    label_dir = tmp_path / "labels"
    image_path.parent.mkdir()
    label_dir.mkdir()
    image_path.write_bytes(b"image placeholder")
    (label_dir / "frame.txt").write_text(
        "0 0.5 0.5 0.2 0.2 0.4 0.4 2\n"
        "0 0.5 0.5 0.2 0.2 malformed 0.4 2\n",
        encoding="utf-8",
    )
    window = _window_without_tk()
    window._label_dir_var = _Var(str(label_dir))
    window._keypoint_names = ["Nose"]
    window._class_names = ["mouse"]

    with pytest.raises(AnnotationIntegrityError, match="line 2"):
        window._load_instances_for_path(image_path, 100, 100)


def test_save_is_blocked_and_existing_malformed_label_is_unchanged(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    label_path = label_dir / "frame.txt"
    original = "this row is malformed\n"
    label_path.write_text(original, encoding="utf-8")

    window = _window_without_tk()
    window._current_path = image_path
    window._current_image = np.zeros((32, 32, 3), dtype=np.uint8)
    window._label_integrity_error = "Malformed annotation in frame.txt, line 1"
    window._status_var = _Var()
    window._append_error = lambda _message: None

    with patch(
        "integra_pose.plugins.plugin_assisted_pose_curation.ui.messagebox.showerror"
    ) as showerror:
        saved = window._save_current_pose()

    assert saved is False
    assert label_path.read_text(encoding="utf-8") == original
    showerror.assert_called_once()


def test_assist_is_blocked_for_frame_with_annotation_integrity_error(tmp_path: Path) -> None:
    window = _window_without_tk()
    window._current_path = tmp_path / "frame.jpg"
    window._current_image = np.zeros((32, 32, 3), dtype=np.uint8)
    window._label_integrity_error = "Malformed annotation in frame.txt, line 1"
    window._status_var = _Var()
    window._append_error = lambda _message: None
    window._predict_seq = 7

    window._schedule_assist_for_current_image()

    assert window._predict_seq == 7
    assert "blocked" in window._status_var.get().lower()


class _SingleFrameCapture:
    def __init__(self, path: str, *, colors: dict[str, int] | None = None) -> None:
        self.path = str(path)
        self.colors = colors or {}
        self.read_count = 0

    def isOpened(self) -> bool:
        return True

    def get(self, _prop) -> float:
        return 1.0

    def set(self, _prop, _value) -> bool:
        return True

    def read(self):
        if self.read_count:
            return False, None
        self.read_count += 1
        value = int(self.colors.get(self.path, 32))
        return True, np.full((24, 24, 3), value, dtype=np.uint8)

    def release(self) -> None:
        return None


def test_active_learning_scoring_forces_one_detection_per_frame(tmp_path: Path) -> None:
    source = tmp_path / "trial.mp4"
    source.write_bytes(b"video")
    window = _window_without_tk()
    window._keypoint_names = ["Nose"]
    window._reviewed_memory_reference_items = lambda: []
    observed_max_det: list[int] = []

    def predict(_model, images, _names, *, conf, max_det, batch_size):
        assert conf == pytest.approx(0.25)
        assert batch_size == 1
        observed_max_det.append(max_det)
        return [None for _ in images]

    window._predict_pose_batch = predict
    window._compute_embeddings_batch = lambda _model, images, **_kwargs: [None for _ in images]
    window.after = lambda _delay, _callback: None

    with patch(
        "integra_pose.plugins.plugin_assisted_pose_curation.ui.cv2.VideoCapture",
        side_effect=lambda path: _SingleFrameCapture(path),
    ):
        candidates = window._audit_video_candidates(
            pose_model=object(),
            embed_model=object(),
            video_path=source,
            stride=1,
            conf=0.25,
        )

    assert observed_max_det == [ACTIVE_LEARNING_SCORING_MAX_DET] == [1]
    assert len(candidates) == 1
    assert candidates[0].source_video_id == source_video_identity(source)
    assert candidates[0].image_name == active_learning_frame_filename(0, video_path=source)


def test_selected_same_stem_frames_are_saved_separately_and_never_overwritten(tmp_path: Path) -> None:
    source_a = tmp_path / "a" / "trial.mp4"
    source_b = tmp_path / "b" / "trial.mp4"
    source_a.parent.mkdir()
    source_b.parent.mkdir()
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")
    colors = {str(source_a): 32, str(source_b): 224}
    candidates = []
    for source in (source_a, source_b):
        source_id = source_video_identity(source)
        candidates.append(
            ActiveLearningCandidate(
                frame_index=0,
                source_video_path=str(source),
                source_video_id=source_id,
                image_name=active_learning_frame_filename(
                    0,
                    video_path=source,
                    source_id=source_id,
                ),
                selected=True,
            )
        )

    image_dir = tmp_path / "images"
    window = _window_without_tk()
    window.after = lambda _delay, _callback: None
    with patch(
        "integra_pose.plugins.plugin_assisted_pose_curation.ui.cv2.VideoCapture",
        side_effect=lambda path: _SingleFrameCapture(path, colors=colors),
    ):
        saved = window._save_selected_audit_frames(image_dir=image_dir, candidates=candidates)

    assert len(saved) == 2
    assert len(set(saved)) == 2
    hashes_before = {
        name: hashlib.sha256((image_dir / name).read_bytes()).hexdigest()
        for name in saved
    }
    assert all(cv2.imread(str(image_dir / name)) is not None for name in saved)

    changed_colors = {str(source_a): 100, str(source_b): 100}
    with patch(
        "integra_pose.plugins.plugin_assisted_pose_curation.ui.cv2.VideoCapture",
        side_effect=lambda path: _SingleFrameCapture(path, colors=changed_colors),
    ):
        saved_again = window._save_selected_audit_frames(image_dir=image_dir, candidates=candidates)

    assert saved_again == saved
    assert {
        name: hashlib.sha256((image_dir / name).read_bytes()).hexdigest()
        for name in saved
    } == hashes_before

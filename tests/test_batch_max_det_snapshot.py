from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from integra_pose.gui.services.batch_processing_service import BatchProcessingService
from integra_pose.logic.batch_pipeline import BatchPipeline
from integra_pose.utils.batch_session import (
    BatchModelCapabilities,
    BatchSession,
    CURRENT_SCHEMA_VERSION,
)


class _Var:
    def __init__(self, value) -> None:
        self.value = value

    def get(self):
        return self.value


class _ExplodingVar:
    def get(self):
        raise AssertionError("Batch settings must not read max_det live from Tab 2.")


def _pipeline_app(live_tab2_max_det=999):
    inference = SimpleNamespace(
        infer_hide_labels_var=_Var(False),
        infer_hide_conf_var=_Var(False),
        tracker_config_path=_Var(""),
        conf_thres_var=_Var("0.25"),
        iou_thres_var=_Var("0.45"),
        infer_imgsz_var=_Var("640"),
        infer_device_var=_Var("cpu"),
        infer_max_det_var=_ExplodingVar() if live_tab2_max_det is None else _Var(live_tab2_max_det),
        infer_augment_var=_Var(False),
        motion_direction_threshold_var=_Var("15"),
        motion_velocity_threshold_var=_Var("0"),
    )
    config = SimpleNamespace(
        inference=inference,
        setup=SimpleNamespace(keypoint_names_str=_Var("")),
        pose_clustering=SimpleNamespace(skeleton_connections=[]),
    )
    return SimpleNamespace(config=config, log_message=lambda *_args, **_kwargs: None)


def test_batch_session_round_trips_max_det_and_legacy_default(tmp_path: Path) -> None:
    session = BatchProcessingService.build_session(
        source_path="",
        roi_strategy="single",
        model_path="model.pt",
        output_path=str(tmp_path / "outputs"),
        inference_device="cpu",
        videos=[],
        max_det=17,
        behavior_bout_class_mode="multi_label",
    )

    assert session.max_det == 17
    assert session.behavior_bout_class_mode == "multi_label"
    assert session.to_dict()["max_det"] == 17
    assert session.to_dict()["behavior_bout_class_mode"] == "multi_label"
    assert BatchSession.from_dict(session.to_dict()).max_det == 17
    assert BatchSession.from_dict({}).max_det == 300

    session_path = session.save_json(tmp_path / "batch_session.json")
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CURRENT_SCHEMA_VERSION == 8
    assert payload["max_det"] == 17
    assert BatchSession.load_json(session_path).max_det == 17


def test_batch_session_drops_gui_keypoint_names_for_detection_model(tmp_path: Path) -> None:
    session = BatchProcessingService.build_session(
        source_path="",
        roi_strategy="single",
        model_path="detect.pt",
        output_path=str(tmp_path / "outputs"),
        videos=[],
        model_capabilities=BatchModelCapabilities(
            task="detect",
            has_keypoints=False,
            keypoint_count=0,
        ),
        keypoint_names=["Nose", "Thorax", "TailBase"],
        keypoint_names_source="gui_config",
    )

    assert session.keypoint_names == []
    assert session.keypoint_names_source == "not_applicable"
    assert session.keypoint_schema_path == ""


def test_batch_session_rejects_invalid_detection_caps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        BatchSession.from_dict({"max_det": 0})

    with pytest.raises(ValueError, match="at least 1"):
        BatchProcessingService.build_session(
            source_path="",
            roi_strategy="single",
            model_path="model.pt",
            output_path=str(tmp_path / "outputs"),
            inference_device="cpu",
            videos=[],
            max_det=0,
        )


def test_batch_session_rejects_unknown_statistics_correction(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="stats_correction"):
        BatchProcessingService.build_session(
            source_path="",
            roi_strategy="single",
            model_path="model.pt",
            output_path=str(tmp_path / "outputs"),
            inference_device="cpu",
            videos=[],
            stats_correction="not-a-method",
        )


def test_pipeline_uses_session_max_det_not_live_inference_tab(tmp_path: Path) -> None:
    app = _pipeline_app(live_tab2_max_det=None)
    pipeline = BatchPipeline(app)
    session = BatchSession.create()
    session.model_path = str(tmp_path / "model.pt")
    session.inference_device = "cpu"
    session.max_det = 7
    session.tracker_enabled = False
    session.save_annotated_video = False
    session.single_animal_mode = False
    session.inference_batch_size = 2
    session.video_fps = 30.0

    settings = pipeline._build_inference_settings(
        session,
        str(tmp_path / "video.mp4"),
        tmp_path / "runs",
        "infer",
    )

    assert settings.max_det == 7

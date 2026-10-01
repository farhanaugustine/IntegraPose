"""Human/mouse regression coverage with real label files and decoded video."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.profiles import (
    PRESET_NAMES, preset_config, validate_config,
)
from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.selection import select_subject
from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.analysis import calculate_original_gait_metrics
from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.data_loader import _ensure_track_ids
from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.main import run
from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.review import build_bouts, _clip
from integra_pose.utils.yolo_pose_labels import YoloPoseLabelSchema, write_pose_label_schema


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_preset_round_trip_and_optional_pose_metrics(name):
    config = validate_config(json.loads(json.dumps(preset_config(name))))
    assert config["GAIT_ANALYSIS"]["OPPOSING_PAW"] in config["GAIT_ANALYSIS"]["GAIT_PAWS"]
    assert config["GENERAL_PARAMS"]["TARGET_TRACK_ID"] is None
    if name == PRESET_NAMES[1]:
        assert len(config["DATASET"]["KEYPOINT_ORDER"]) == 17
        assert config["DATASET"]["BEHAVIOR_CLASSES"] == {0: "Person"}
        assert not config["POSE_METRICS"]["ELONGATION_CONNECTION"]


@pytest.mark.parametrize("section,key,value", [
    ("GAIT_ANALYSIS", "OPPOSING_PAW", "Nose"),
    ("GENERAL_PARAMS", "KEYPOINT_CONF_THRESHOLD", 1.1),
    ("GENERAL_PARAMS", "START_FRAME", 1.5),
    ("GAIT_ANALYSIS", "PAW_SPEED_THRESHOLD_PX_PER_FRAME", float("nan")),
    ("POSE_METRICS", "BODY_ANGLE_CONNECTION", ["Absent", "Nose"]),
])
def test_invalid_config_rejected(section, key, value):
    config = preset_config()
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def _rows(config, tracks=(2, 9)):
    data = pd.DataFrame({"frame": [0, 1, 0, 1], "track_id": [tracks[0]] * 2 + [tracks[-1]] * 2})
    for point in config["DATASET"]["KEYPOINT_ORDER"]:
        data[f"{point}_x"] = 10.
        data[f"{point}_y"] = 20.
        data[f"{point}_conf"] = .9
    return data


def test_multiple_subjects_require_selection_and_low_confidence_is_missing():
    config = validate_config(preset_config(PRESET_NAMES[1]))
    rows = _rows(config)
    with pytest.raises(ValueError, match="target track ID"):
        select_subject(rows, config, 10)
    config["GENERAL_PARAMS"]["TARGET_TRACK_ID"] = 9
    rows.loc[3, "Left Ankle_conf"] = .1
    result = select_subject(rows, config, 10)
    assert result.track_id.tolist() == [9, 9]
    assert np.isnan(result.loc[1, "Left Ankle_x"])
    assert result.loc[1, "Right Ankle_x"] == 10


def test_frame_range_and_duplicate_subject_rows_rejected():
    config = validate_config(preset_config())
    config["GENERAL_PARAMS"]["TARGET_TRACK_ID"] = 2
    rows = _rows(config)
    with pytest.raises(ValueError, match="frame range"):
        select_subject(rows, config, 1)
    with pytest.raises(ValueError, match="multiple detections"):
        select_subject(pd.concat([rows, rows.iloc[[0]]]), config, 10)


def test_explicit_human_opposite_landmark_computes_step_geometry():
    config = preset_config(PRESET_NAMES[1])
    events = pd.DataFrame([
        {"track_id": 2, "frame": 10, "paw": "Left Ankle", "event": "foot_strike", "x": 0., "y": 0.},
        {"track_id": 2, "frame": 13, "paw": "Right Ankle", "event": "foot_strike", "x": 6., "y": 3.},
        {"track_id": 2, "frame": 16, "paw": "Left Ankle", "event": "foot_strike", "x": 12., "y": 0.},
    ])
    rows = pd.DataFrame({"frame": range(10, 17), "track_id": 2, "Left Ankle_x": np.arange(7) * 2.,
                         "Left Ankle_y": 0., "Left Ankle_speed_px_per_frame": 2.,
                         "speed_px_per_frame": 2., "speed_px_per_s": 60.})
    result = calculate_original_gait_metrics(events, rows, config, fps=30.)
    assert result.loc[0, "step_width"] == pytest.approx(3.)
    assert result.loc[0, "step_length"] == pytest.approx(np.sqrt(45))
    assert result.loc[0, "stride_duration_s"] == .2


def test_inferred_track_never_reuses_an_id_already_present_in_same_frame():
    detections = [
        {"frame": 0, "track_id": 7, "bbox": [20, 20, 30, 30], "video_width": 100, "video_height": 100},
        {"frame": 0, "track_id": None, "bbox": [21, 20, 30, 30], "video_width": 100, "video_height": 100},
    ]
    _ensure_track_ids(detections)
    assert detections[0]["track_id"] != detections[1]["track_id"]


def test_bouts_do_not_bridge_missing_intervals_or_subjects():
    strides = pd.DataFrame({"track_id": [1, 1, 1, 2], "start_frame": [10, 16, 30, 10], "end_frame": [16, 22, 36, 16]})
    details, bouts = build_bouts(strides, 30)
    assert bouts.stride_count.tolist() == [2, 1, 1]
    assert bouts.iloc[0].duration_s == .4
    assert details.bout_id.tolist() == ["B001", "B001", "B002", "B003"]


def make_recording(tmp_path, config, *, still=False):
    video = tmp_path / "walking.avi"
    labels = tmp_path / "labels"
    labels.mkdir()
    points = config["DATASET"]["KEYPOINT_ORDER"]
    write_pose_label_schema(labels, YoloPoseLabelSchema(len(points), 3, include_track_id=True))
    # Names exported by IntegraPose inference, in actual model order.
    from integra_pose.utils.keypoint_schema import safe_keypoint_token
    (labels / 'labels.csv').write_text(','.join(f'kp_{safe_keypoint_token(p)}_x_n' for p in points) + '\n', encoding='utf-8')
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    assert writer.isOpened()
    position = 50.
    for frame in range(75):
        speed = 0 if still else (8 if frame % 12 in (3, 4, 5, 6) else 1)
        position += speed
        image = np.full((240, 320, 3), 60, dtype=np.uint8)
        cv2.putText(image, str(frame), (130, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(image)
        tokens = [0, .4 + frame / 1000, .5, .6, .7]
        for index, _point in enumerate(points):
            tokens += [(position + index * .1) / 320, .6, .95]
        tokens += [7]
        (labels / f"frame_{frame:06d}.txt").write_text(" ".join(map(str, tokens)), encoding="utf-8")
    writer.release()
    return SimpleNamespace(video_path=str(video), yolo_dir=str(labels), output_dir=str(tmp_path / "results"))


@pytest.mark.parametrize("name", PRESET_NAMES)
@pytest.mark.parametrize("method", ["Original", "Peak-Based (Advanced)"])
def test_video_label_pipeline_and_html_for_both_species(tmp_path, name, method):
    config = preset_config(name)
    config["GAIT_ANALYSIS"]["GAIT_DETECTION_METHOD"] = method
    config["REVIEW"]["EXPORT_CLIPS"] = False
    config["REVIEW"]["EXPORT_COORDINATION"] = True
    args = make_recording(tmp_path, config)
    result = run(args, config)
    assert result.succeeded, result
    report = (tmp_path / "results" / "gait_review_report.html").read_text(encoding="utf-8")
    assert "Limb motion timeline" in report and "<svg" in report
    assert "__TITLE__" not in report
    assert result.artifacts["stride_count"] > 0
    table = pd.read_csv(result.artifacts["gait_summary"])
    assert np.allclose(table.stride_duration_s, (table.end_frame - table.start_frame) / 30)
    assert table.track_id.unique().tolist() == [7]
    assert not result.artifacts['review_warnings'], result.artifacts['review_warnings']
    motion = pd.read_csv(tmp_path/'results'/'limb_motion_series.csv')
    assert set(motion.subject_type) == {config['SUBJECT_TYPE']}
    assert motion.frame.tolist() == list(range(75))
    assert 'motion_review' in result.artifacts
    assert ('Left_knee_bend_deg' in motion) == (name == PRESET_NAMES[1])
    if name == PRESET_NAMES[1]:
        frame_table = pd.read_csv(result.artifacts["final_analysis"])
        assert frame_table.elongation.isna().all()
        assert frame_table.behavior_name.unique().tolist() == ["Person"]


def test_empty_stride_run_replaces_previous_report(tmp_path):
    config = preset_config(PRESET_NAMES[1])
    config["REVIEW"]["EXPORT_CLIPS"] = False
    args = make_recording(tmp_path, config, still=True)
    result = run(args, config)
    assert result.succeeded
    assert result.artifacts["stride_count"] == 0
    assert "No accepted strides" in (tmp_path / "results" / "gait_review_report.html").read_text(encoding="utf-8")
    assert pd.read_csv(result.artifacts["stride_details"]).empty


def test_clip_export_has_exact_inclusive_frame_count(tmp_path, monkeypatch):
    config = preset_config()
    args = make_recording(tmp_path, config)
    monkeypatch.setattr("integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.review.shutil.which", lambda _: None)
    path = _clip(args.video_path, tmp_path / "clip.avi", 10, 16, 30, pd.DataFrame({"frame": []}), [])
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 7
        assert cap.get(cv2.CAP_PROP_FPS) == 30
    finally:
        cap.release()

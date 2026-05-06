import importlib.util
from pathlib import Path
import sys

import pytest
import pandas as pd


def _load_module(rel_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_bout_analyzer = _load_module(
    "integra_pose/utils/bout_analyzer.py",
    "test_bout_analyzer_module",
)

BoutAnalysisError = _bout_analyzer.BoutAnalysisError
BoutAnalysisCancelledError = _bout_analyzer.BoutAnalysisCancelledError
analyze_bouts = _bout_analyzer.analyze_bouts
assign_roi_membership = _bout_analyzer.assign_roi_membership
compute_object_interactions = _bout_analyzer.compute_object_interactions
compute_roi_metrics = _bout_analyzer.compute_roi_metrics
load_and_preprocess_data = _bout_analyzer.load_and_preprocess_data
save_analysis_outputs = _bout_analyzer.save_analysis_outputs


def test_analyze_bouts_does_not_split_on_roi_changes():
    per_frame_df = pd.DataFrame(
        {
            "track_id": [0] * 7,
            "frame": [0, 1, 2, 3, 4, 5, 6],
            "class_id": [0] * 7,
            "ROI Name": ["roiA", "roiA", "roiA", "", "", "roiA", "roiA"],
        }
    )

    detailed, summary = analyze_bouts(
        per_frame_df,
        class_names={0: "TestBehavior"},
        max_gap_frames=10,
        min_bout_frames=1,
        fps=30,
        roi_column="ROI Name",
    )

    assert detailed.shape[0] == 1
    assert detailed.loc[0, "Start Frame"] == 0
    assert detailed.loc[0, "End Frame"] == 6
    assert detailed.loc[0, "ROI Name"] == "roiA"
    assert detailed.loc[0, "ROI Memberships"] == ("roiA",)

    summary_roi_a = summary[summary["ROI Name"] == "roiA"].reset_index(drop=True)
    assert summary_roi_a.shape[0] == 1
    assert int(summary_roi_a.loc[0, "Bout_Count"]) == 1


def test_analyze_bouts_bridges_small_gaps_when_roi_constant():
    per_frame_df = pd.DataFrame(
        {
            "track_id": [0, 0, 0],
            "frame": [0, 2, 3],
            "class_id": [0, 0, 0],
            "ROI Name": ["roiA", "roiA", "roiA"],
        }
    )

    detailed, _summary = analyze_bouts(
        per_frame_df,
        class_names={0: "TestBehavior"},
        max_gap_frames=2,
        min_bout_frames=1,
        fps=30,
        roi_column="ROI Name",
    )

    roi_a = detailed[detailed["ROI Name"] == "roiA"].reset_index(drop=True)
    assert roi_a.shape[0] == 1
    assert roi_a.loc[0, "Start Frame"] == 0
    assert roi_a.loc[0, "End Frame"] == 3


def test_compute_roi_metrics_dedupes_duplicate_frames():
    per_frame_df = pd.DataFrame(
        [
            {"track_id": 0, "frame": 0, "class_id": 0, "ROI Name": "roiA", "confidence": 0.2},
            {"track_id": 0, "frame": 0, "class_id": 0, "ROI Name": "roiA", "confidence": 0.9},
            {"track_id": 0, "frame": 1, "class_id": 0, "ROI Name": "roiA", "confidence": 0.8},
        ]
    )

    roi_metrics = compute_roi_metrics(per_frame_df, roi_column="ROI Name", fps=30, class_names={0: "Beh"})
    assert roi_metrics is not None
    overview = roi_metrics["entries_exits"]
    row = overview.loc[overview["ROI Name"] == "roiA"].iloc[0]
    assert int(row["Frames in ROI"]) == 2
    assert int(row["Entries"]) == 1
    assert int(row["Exits"]) == 1


def test_compute_roi_metrics_bridges_short_membership_flicker():
    per_frame_df = pd.DataFrame(
        [
            {"track_id": 1, "frame": 0, "class_id": 0, "ROI Name": "roiA", "ROI Memberships": ("roiA",)},
            {"track_id": 1, "frame": 1, "class_id": 0, "ROI Name": "roiA", "ROI Memberships": ("roiA",)},
            {"track_id": 1, "frame": 2, "class_id": 0, "ROI Name": "roiA", "ROI Memberships": ("roiA",)},
            {"track_id": 1, "frame": 3, "class_id": 0, "ROI Name": "", "ROI Memberships": ()},
            {"track_id": 1, "frame": 4, "class_id": 0, "ROI Name": "roiA", "ROI Memberships": ("roiA",)},
            {"track_id": 1, "frame": 5, "class_id": 0, "ROI Name": "roiA", "ROI Memberships": ("roiA",)},
            {"track_id": 1, "frame": 6, "class_id": 0, "ROI Name": "roiA", "ROI Memberships": ("roiA",)},
        ]
    )

    roi_metrics = compute_roi_metrics(
        per_frame_df,
        roi_column="ROI Name",
        fps=30,
        class_names={0: "Beh"},
        max_gap_frames=1,
        min_dwell_frames=5,
    )

    row = roi_metrics["entries_exits"].loc[roi_metrics["entries_exits"]["ROI Name"] == "roiA"].iloc[0]
    assert int(row["Entries"]) == 1
    assert int(row["Exits"]) == 1
    assert int(row["Dwell Events"]) == 1


def test_assign_roi_membership_event_log_bridges_short_flicker():
    per_frame_df = pd.DataFrame(
        [
            {"track_id": 1, "frame": 0, "x_center": 0.2, "y_center": 0.2, "w": 0.1, "h": 0.1},
            {"track_id": 1, "frame": 1, "x_center": 0.2, "y_center": 0.2, "w": 0.1, "h": 0.1},
            {"track_id": 1, "frame": 2, "x_center": 0.8, "y_center": 0.8, "w": 0.1, "h": 0.1},
            {"track_id": 1, "frame": 3, "x_center": 0.2, "y_center": 0.2, "w": 0.1, "h": 0.1},
            {"track_id": 1, "frame": 4, "x_center": 0.2, "y_center": 0.2, "w": 0.1, "h": 0.1},
            {"track_id": 1, "frame": 5, "x_center": 0.2, "y_center": 0.2, "w": 0.1, "h": 0.1},
        ]
    )
    roi_polygons = {"roiA": [[(0, 0), (0, 40), (40, 40), (40, 0)]]}

    _df_result, event_log = assign_roi_membership(
        per_frame_df,
        roi_polygons=roi_polygons,
        video_width=100,
        video_height=100,
        entry_threshold=0.75,
        exit_threshold=0.25,
        max_gap_frames=1,
        min_dwell_frames=5,
    )

    assert event_log["entries"] == [{"frame": 0, "track_id": 1, "roi_name": "roiA"}]
    assert event_log["exits"] == [{"frame": 5, "track_id": 1, "roi_name": "roiA"}]


def test_assign_roi_membership_event_log_ignores_brief_presence_below_min_dwell():
    per_frame_df = pd.DataFrame(
        [
            {"track_id": 1, "frame": 0, "x_center": 0.2, "y_center": 0.2, "w": 0.1, "h": 0.1},
            {"track_id": 1, "frame": 1, "x_center": 0.2, "y_center": 0.2, "w": 0.1, "h": 0.1},
            {"track_id": 1, "frame": 2, "x_center": 0.2, "y_center": 0.2, "w": 0.1, "h": 0.1},
        ]
    )
    roi_polygons = {"roiA": [[(0, 0), (0, 40), (40, 40), (40, 0)]]}

    _df_result, event_log = assign_roi_membership(
        per_frame_df,
        roi_polygons=roi_polygons,
        video_width=100,
        video_height=100,
        entry_threshold=0.75,
        exit_threshold=0.25,
        max_gap_frames=1,
        min_dwell_frames=5,
    )

    assert event_log["entries"] == []
    assert event_log["exits"] == []


def test_compute_object_interactions_bridges_short_flicker():
    per_frame_df = pd.DataFrame(
        [
            {"track_id": 1, "frame": 0, "x_center": 50, "y_center": 50, "w": 10, "h": 10, "keypoints": [(50, 50, 1.0)]},
            {"track_id": 1, "frame": 1, "x_center": 50, "y_center": 50, "w": 10, "h": 10, "keypoints": [(50, 50, 1.0)]},
            {"track_id": 1, "frame": 2, "x_center": 90, "y_center": 50, "w": 10, "h": 10, "keypoints": [(90, 50, 1.0)]},
            {"track_id": 1, "frame": 3, "x_center": 50, "y_center": 50, "w": 10, "h": 10, "keypoints": [(50, 50, 1.0)]},
            {"track_id": 1, "frame": 4, "x_center": 50, "y_center": 50, "w": 10, "h": 10, "keypoints": [(50, 50, 1.0)]},
            {"track_id": 1, "frame": 5, "x_center": 50, "y_center": 50, "w": 10, "h": 10, "keypoints": [(50, 50, 1.0)]},
        ]
    )
    object_rois = {"obj": [[(40, 40), (60, 40), (60, 60), (40, 60)]]}

    result = compute_object_interactions(
        per_frame_df,
        object_rois,
        100,
        100,
        keypoint_index=0,
        distance_threshold_px=0,
        fps=30,
        max_gap_frames=1,
        min_dwell_frames=5,
    )

    row = result["summary"].iloc[0]
    assert int(row["Entries"]) == 1
    assert int(row["Exits"]) == 1
    assert int(row["Dwell Events"]) == 1
    assert result["events"]["entries"] == [{"frame": 0, "track_id": 1, "object_roi": "obj"}]
    assert result["events"]["exits"] == [{"frame": 5, "track_id": 1, "object_roi": "obj"}]


def test_compute_roi_metrics_filters_brief_entry_exit_by_min_dwell():
    per_frame_df = pd.DataFrame(
        [
            {"track_id": 1, "frame": 0, "class_id": 0, "ROI Name": "roiA", "ROI Memberships": ("roiA",)},
            {"track_id": 1, "frame": 1, "class_id": 0, "ROI Name": "roiA", "ROI Memberships": ("roiA",)},
            {"track_id": 1, "frame": 2, "class_id": 0, "ROI Name": "roiA", "ROI Memberships": ("roiA",)},
        ]
    )

    roi_metrics = compute_roi_metrics(
        per_frame_df,
        roi_column="ROI Name",
        fps=30,
        class_names={0: "Beh"},
        max_gap_frames=5,
        min_dwell_frames=5,
    )

    row = roi_metrics["entries_exits"].loc[roi_metrics["entries_exits"]["ROI Name"] == "roiA"].iloc[0]
    assert int(row["Entries"]) == 0
    assert int(row["Exits"]) == 0
    assert int(row["Dwell Events"]) == 0


def test_compute_object_interactions_filters_brief_entry_exit_by_min_dwell():
    per_frame_df = pd.DataFrame(
        [
            {"track_id": 1, "frame": 0, "x_center": 50, "y_center": 50, "w": 10, "h": 10, "keypoints": [(50, 50, 1.0)]},
            {"track_id": 1, "frame": 1, "x_center": 50, "y_center": 50, "w": 10, "h": 10, "keypoints": [(50, 50, 1.0)]},
            {"track_id": 1, "frame": 2, "x_center": 50, "y_center": 50, "w": 10, "h": 10, "keypoints": [(50, 50, 1.0)]},
        ]
    )
    object_rois = {"obj": [[(40, 40), (60, 40), (60, 60), (40, 60)]]}

    result = compute_object_interactions(
        per_frame_df,
        object_rois,
        100,
        100,
        keypoint_index=0,
        distance_threshold_px=0,
        fps=30,
        max_gap_frames=5,
        min_dwell_frames=5,
    )

    row = result["summary"].iloc[0]
    assert int(row["Entries"]) == 0
    assert int(row["Exits"]) == 0
    assert int(row["Dwell Events"]) == 0
    assert result["events"]["entries"] == []
    assert result["events"]["exits"] == []


def test_compute_object_interactions_does_not_fallback_to_bbox_center_when_keypoint_missing():
    per_frame_df = pd.DataFrame(
        [
            {"track_id": 1, "frame": 0, "x_center": 50, "y_center": 50, "w": 10, "h": 10, "keypoints": [(50, 50, 1.0)]},
            {"track_id": 1, "frame": 1, "x_center": 50, "y_center": 50, "w": 10, "h": 10, "keypoints": [(50, 50, 0.0)]},
        ]
    )
    object_rois = {"obj": [[(40, 40), (60, 40), (60, 60), (40, 60)]]}

    result = compute_object_interactions(
        per_frame_df,
        object_rois,
        100,
        100,
        keypoint_index=0,
        distance_threshold_px=0,
        fps=30,
        max_gap_frames=0,
    )

    row = result["summary"].iloc[0]
    assert int(row["Entries"]) == 1
    assert int(row["Exits"]) == 1
    assert int(row["Frames Interacting"]) == 1
    assert result["per_frame_df"].loc[1, "Object Interaction ROI"] == ""
    assert result["per_frame_df"].loc[1, "Object Interaction State"] == ""


def test_compute_object_interactions_carries_last_keypoint_within_gap_tolerance():
    per_frame_df = pd.DataFrame(
        [
            {"track_id": 1, "frame": 0, "x_center": 90, "y_center": 50, "w": 10, "h": 10, "keypoints": [(50, 50, 1.0)]},
            {"track_id": 1, "frame": 1, "x_center": 90, "y_center": 50, "w": 10, "h": 10, "keypoints": [(50, 50, 0.0)]},
        ]
    )
    object_rois = {"obj": [[(40, 40), (60, 40), (60, 60), (40, 60)]]}

    result = compute_object_interactions(
        per_frame_df,
        object_rois,
        100,
        100,
        keypoint_index=0,
        distance_threshold_px=0,
        fps=30,
        max_gap_frames=1,
    )

    row = result["summary"].iloc[0]
    assert int(row["Entries"]) == 1
    assert int(row["Exits"]) == 1
    assert int(row["Frames Interacting"]) == 2
    assert result["per_frame_df"].loc[1, "Object Interaction ROI"] == "obj"
    assert result["per_frame_df"].loc[1, "Object Interaction State"] == "contact"


def test_loader_requires_track_ids_for_multi_detection_frames(tmp_path):
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("names:\n  0: behavior\nrois: {}\n", encoding="utf-8")

    yolo_dir = tmp_path / "yolo"
    yolo_dir.mkdir()
    (yolo_dir / "frame_0.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n0 0.6 0.6 0.1 0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(BoutAnalysisError):
        load_and_preprocess_data(str(yaml_path), str(yolo_dir), single_animal_mode=False)

    df, _rois, _names = load_and_preprocess_data(str(yaml_path), str(yolo_dir), single_animal_mode=True)
    assert df.groupby("frame").size().max() == 1


def test_assign_roi_membership_honors_stop_check():
    per_frame_df = pd.DataFrame(
        [{"track_id": 1, "frame": idx, "x_center": 0.2, "y_center": 0.2, "w": 0.1, "h": 0.1} for idx in range(600)]
    )
    roi_polygons = {"roiA": [[(0, 0), (0, 40), (40, 40), (40, 0)]]}
    stop_state = {"count": 0}

    def _stop_check():
        stop_state["count"] += 1
        return stop_state["count"] >= 2

    with pytest.raises(BoutAnalysisCancelledError):
        assign_roi_membership(
            per_frame_df,
            roi_polygons=roi_polygons,
            video_width=100,
            video_height=100,
            stop_check=_stop_check,
        )


def test_compute_object_interactions_honors_stop_check():
    per_frame_df = pd.DataFrame(
        [
            {"track_id": 1, "frame": idx, "x_center": 50, "y_center": 50, "w": 10, "h": 10, "keypoints": [(50, 50, 1.0)]}
            for idx in range(600)
        ]
    )
    object_rois = {"obj": [[(40, 40), (60, 40), (60, 60), (40, 60)]]}
    stop_state = {"count": 0}

    def _stop_check():
        stop_state["count"] += 1
        return stop_state["count"] >= 2

    with pytest.raises(BoutAnalysisCancelledError):
        compute_object_interactions(
            per_frame_df,
            object_rois,
            100,
            100,
            keypoint_index=0,
            stop_check=_stop_check,
        )


def test_save_analysis_outputs_keeps_roi_events_when_bout_threshold_is_strict(tmp_path):
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("names:\n  0: behavior\nrois: {}\n", encoding="utf-8")
    output_dir = tmp_path / "outputs"

    per_frame_df = pd.DataFrame(
        [
            {"track_id": 1, "frame": 0, "class_id": 0, "ROI Name": "roiA", "ROI Memberships": ("roiA",)},
            {"track_id": 1, "frame": 1, "class_id": 0, "ROI Name": "roiA", "ROI Memberships": ("roiA",)},
            {"track_id": 1, "frame": 2, "class_id": 0, "ROI Name": "roiA", "ROI Memberships": ("roiA",)},
        ]
    )

    detailed_bouts_df, _summary_df, _excel_path, roi_metrics, _module_outputs = save_analysis_outputs(
        per_frame_df,
        str(yaml_path),
        str(output_dir),
        max_gap_frames=500,
        min_bout_frames=500,
        fps=30,
        video_name="roi_events",
        roi_column="ROI Name",
        class_names={0: "behavior"},
        roi_max_gap_frames=0,
        roi_min_dwell_frames=1,
    )

    assert detailed_bouts_df.empty
    assert roi_metrics is not None
    row = roi_metrics["entries_exits"].loc[roi_metrics["entries_exits"]["ROI Name"] == "roiA"].iloc[0]
    assert int(row["Entries"]) == 1
    assert int(row["Exits"]) == 1
    assert int(row["Dwell Events"]) == 1

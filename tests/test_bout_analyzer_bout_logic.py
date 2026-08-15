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
AnalyticsModuleContext = _bout_analyzer.AnalyticsModuleContext
_module_latency_metrics = _bout_analyzer._module_latency_metrics
_module_normalization_summary = _bout_analyzer._module_normalization_summary
_module_object_transition_analysis = _bout_analyzer._module_object_transition_analysis
_module_roi_time_heatmap = _bout_analyzer._module_roi_time_heatmap


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


def test_analyze_bouts_never_bridges_across_an_explicit_other_behavior():
    per_frame_df = pd.DataFrame(
        {
            "track_id": [0, 0, 0, 0, 0],
            "frame": [0, 1, 2, 3, 4],
            "class_id": [0, 0, 1, 0, 0],
        }
    )

    detailed, _summary = analyze_bouts(
        per_frame_df,
        class_names={0: "A", 1: "B"},
        max_gap_frames=10,
        min_bout_frames=1,
        fps=30,
    )

    a_bouts = detailed[detailed["Behavior"] == "A"].reset_index(drop=True)
    assert a_bouts[["Start Frame", "End Frame"]].to_dict("records") == [
        {"Start Frame": 0, "End Frame": 1},
        {"Start Frame": 3, "End Frame": 4},
    ]
    assert not any(
        int(left["Start Frame"]) <= int(right["End Frame"])
        and int(right["Start Frame"]) <= int(left["End Frame"])
        for left_index, left in detailed.iterrows()
        for right_index, right in detailed.iterrows()
        if left_index < right_index and int(left["Track ID"]) == int(right["Track ID"])
    )


def test_analyze_bouts_reports_observed_and_bridged_support():
    per_frame_df = pd.DataFrame(
        {
            "track_id": [0, 0],
            "frame": [0, 4],
            "class_id": [0, 0],
        }
    )

    detailed, _summary = analyze_bouts(
        per_frame_df,
        class_names={0: "A"},
        max_gap_frames=4,
        min_bout_frames=5,
        fps=10,
    )

    assert detailed.shape[0] == 1
    row = detailed.iloc[0]
    assert int(row["Duration (Frames)"]) == 5
    assert int(row["Observed Frames"]) == 2
    assert int(row["Bridged Frames"]) == 3
    assert int(row["Maximum Bridged Gap (Frames)"]) == 3
    assert float(row["Observed Fraction"]) == pytest.approx(0.4)


def test_analyze_bouts_resolves_same_frame_class_conflicts_by_confidence():
    per_frame_df = pd.DataFrame(
        [
            {"track_id": 0, "frame": 0, "class_id": 0, "confidence": 0.2},
            {"track_id": 0, "frame": 0, "class_id": 1, "confidence": 0.9},
            {"track_id": 0, "frame": 1, "class_id": 1, "confidence": 0.8},
        ]
    )

    detailed, _summary = analyze_bouts(
        per_frame_df,
        class_names={0: "A", 1: "B"},
        max_gap_frames=0,
        min_bout_frames=1,
        fps=30,
    )

    assert detailed[["Behavior", "Start Frame", "End Frame"]].to_dict("records") == [
        {"Behavior": "B", "Start Frame": 0, "End Frame": 1}
    ]
    assert int(detailed.iloc[0]["Class ID"]) == 1
    assert int(detailed.iloc[0]["Resolved Class-Conflict Frames"]) == 1


def test_analyze_bouts_multi_label_mode_preserves_overlapping_class_channels():
    per_frame_df = pd.DataFrame(
        [
            {"track_id": 3, "frame": 0, "class_id": 0, "confidence": 0.9},
            {"track_id": 3, "frame": 1, "class_id": 0, "confidence": 0.9},
            {"track_id": 3, "frame": 1, "class_id": 1, "confidence": 0.8},
            {"track_id": 3, "frame": 2, "class_id": 0, "confidence": 0.9},
            {"track_id": 3, "frame": 2, "class_id": 1, "confidence": 0.8},
            {"track_id": 3, "frame": 3, "class_id": 1, "confidence": 0.8},
        ]
    )

    detailed, summary = analyze_bouts(
        per_frame_df,
        class_names={0: "rear", 1: "wall_rear"},
        max_gap_frames=0,
        min_bout_frames=1,
        fps=30,
        bout_class_mode="multi_label",
    )

    assert detailed[
        ["Track ID", "Class ID", "Behavior", "Start Frame", "End Frame"]
    ].to_dict("records") == [
        {
            "Track ID": 3,
            "Class ID": 0,
            "Behavior": "rear",
            "Start Frame": 0,
            "End Frame": 2,
        },
        {
            "Track ID": 3,
            "Class ID": 1,
            "Behavior": "wall_rear",
            "Start Frame": 1,
            "End Frame": 3,
        },
    ]
    assert detailed["Resolved Class-Conflict Frames"].eq(0).all()
    assert detailed["Concurrent Class Frames"].tolist() == [2, 2]
    assert set(summary["Class ID"].astype(int)) == {0, 1}


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


def test_compute_roi_transitions_apply_gap_and_min_dwell_to_primary_visits():
    per_frame_df = pd.DataFrame(
        [
            {"track_id": 1, "frame": 0, "class_id": 0, "ROI Name": "A", "ROI Memberships": ("A",)},
            {"track_id": 1, "frame": 1, "class_id": 0, "ROI Name": "A", "ROI Memberships": ("A",)},
            # One-frame B flicker is below the two-frame minimum and must not
            # create A->B or B->A transitions.
            {"track_id": 1, "frame": 2, "class_id": 0, "ROI Name": "B", "ROI Memberships": ("B",)},
            {"track_id": 1, "frame": 3, "class_id": 0, "ROI Name": "A", "ROI Memberships": ("A",)},
            {"track_id": 1, "frame": 4, "class_id": 0, "ROI Name": "A", "ROI Memberships": ("A",)},
            {"track_id": 1, "frame": 5, "class_id": 0, "ROI Name": "", "ROI Memberships": ()},
            {"track_id": 1, "frame": 6, "class_id": 0, "ROI Name": "C", "ROI Memberships": ("C",)},
            {"track_id": 1, "frame": 7, "class_id": 0, "ROI Name": "C", "ROI Memberships": ("C",)},
        ]
    )

    metrics = compute_roi_metrics(
        per_frame_df,
        roi_column="ROI Name",
        fps=30,
        class_names={0: "Beh"},
        max_gap_frames=1,
        min_dwell_frames=2,
    )

    transitions = metrics["transitions"]
    assert transitions.to_dict("records") == [
        {"From ROI": "A", "To ROI": "C", "Transition Count": 1}
    ]


def test_compute_roi_metrics_keeps_nested_occupancy_concurrent():
    per_frame_df = pd.DataFrame(
        [
            {
                "track_id": 2,
                "frame": frame,
                "class_id": 0,
                "ROI Name": "center",
                "ROI Memberships": ("center", "arena"),
            }
            for frame in range(3)
        ]
    )

    metrics = compute_roi_metrics(
        per_frame_df,
        roi_column="ROI Name",
        fps=30,
        class_names={0: "Beh"},
    )

    overview = metrics["entries_exits"].set_index("ROI Name")
    assert int(overview.loc["arena", "Frames in ROI"]) == 3
    assert int(overview.loc["center", "Frames in ROI"]) == 3
    assert metrics["transitions"].empty
    assert set(overview["Occupancy Semantics"]) == {"concurrent_membership"}

    exclusive = metrics["exclusive_entries_exits"].set_index("ROI Name")
    assert list(exclusive.index) == ["center"]
    assert int(exclusive.loc["center", "Frames in ROI"]) == 3
    assert int(exclusive["Frames in ROI"].sum()) == 3
    assert set(exclusive["Occupancy Semantics"]) == {"exclusive_primary"}


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
    assert int(row["Raw Occupancy Frames"]) == 3
    assert int(row["Qualified Dwell Frames"]) == 0
    assert roi_metrics["qualified_memberships_by_track_frame"] == {}


def test_saved_behavior_bout_uses_dwell_qualified_roi_context(tmp_path):
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("names:\n  0: behavior\nrois: {}\n", encoding="utf-8")
    per_frame_df = pd.DataFrame(
        [
            {
                "track_id": 1,
                "frame": frame,
                "class_id": 0,
                "ROI Name": "roiA",
                "ROI Memberships": ("roiA",),
            }
            for frame in range(3)
        ]
    )

    detailed, _summary, *_rest = save_analysis_outputs(
        per_frame_df,
        str(yaml_path),
        str(tmp_path / "outputs"),
        max_gap_frames=0,
        min_bout_frames=1,
        fps=30,
        video_name="qualified_context",
        roi_column="ROI Name",
        class_names={0: "behavior"},
        roi_max_gap_frames=0,
        roi_min_dwell_frames=5,
    )

    row = detailed.iloc[0]
    assert row["ROI Name"] == ""
    assert row["ROI Memberships"] == ()
    assert row["Raw ROI Name"] == "roiA"
    assert row["Raw ROI Memberships"] == ("roiA",)
    assert row["ROI Context Semantics"] == "dwell_qualified"


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
    assert int(row["Raw Interaction Frames"]) == 3
    assert int(row["Qualified Interaction Frames"]) == 0
    assert float(row["Raw Interaction Time (s)"]) == pytest.approx(0.1)
    assert float(row["Qualified Interaction Time (s)"]) == pytest.approx(0.0)
    assert result["per_frame_df"]["Qualified Object Interaction ROI"].eq("").all()
    assert all(
        memberships == ()
        for memberships in result["per_frame_df"]["Qualified Object Interaction Memberships"]
    )
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


def test_object_approach_rate_is_normalized_by_frame_delta():
    per_frame_df = pd.DataFrame(
        [
            {"track_id": 1, "frame": 0, "keypoints": [(90, 50, 1.0)]},
            # One missing frame: distance changes by 20 px over two frame
            # intervals, so the rate is 10 px/frame rather than 20.
            {"track_id": 1, "frame": 2, "keypoints": [(70, 50, 1.0)]},
        ]
    )
    object_rois = {"obj": [[(40, 40), (60, 40), (60, 60), (40, 60)]]}

    result = compute_object_interactions(
        per_frame_df,
        object_rois,
        100,
        100,
        keypoint_index=0,
        fps=30,
        max_gap_frames=1,
    )

    by_track = result["approach_retreat_by_track"].iloc[0]
    event = result["approach_retreat_events"].iloc[0]
    assert int(by_track["Approach Frames"]) == 2
    assert float(by_track["Mean Approach Rate (px/frame)"]) == pytest.approx(10.0)
    assert float(by_track["Net Distance Change (px)"]) == pytest.approx(-20.0)
    assert int(event["Start Frame"]) == 0
    assert int(event["End Frame"]) == 2
    assert int(event["Duration (Frames)"]) == 2


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
    saved_empty_bouts = pd.read_csv(output_dir / "roi_events_detailed_bouts.csv")
    saved_empty_summary = pd.read_csv(output_dir / "roi_events_summary.csv")
    assert saved_empty_bouts.empty
    assert saved_empty_summary.empty
    assert "Detection Min Bout (Frames)" in saved_empty_bouts.columns
    assert "Bout Construction Semantics" in saved_empty_bouts.columns
    assert roi_metrics is not None
    row = roi_metrics["entries_exits"].loc[roi_metrics["entries_exits"]["ROI Name"] == "roiA"].iloc[0]
    assert int(row["Entries"]) == 1
    assert int(row["Exits"]) == 1
    assert int(row["Dwell Events"]) == 1


def test_save_analysis_outputs_records_bout_construction_provenance(tmp_path):
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("names:\n  0: behavior\nrois: {}\n", encoding="utf-8")
    per_frame_df = pd.DataFrame(
        [
            {"track_id": 1, "frame": 4, "class_id": 0},
            {"track_id": 1, "frame": 5, "class_id": 0},
        ]
    )

    detailed, summary, *_rest = save_analysis_outputs(
        per_frame_df,
        str(yaml_path),
        str(tmp_path / "outputs"),
        max_gap_frames=7,
        min_bout_frames=2,
        fps=20,
        video_name="provenance",
        class_names={0: "behavior"},
    )

    assert detailed.iloc[0]["Interval Semantics"] == "inclusive_start_and_end_frames"
    assert int(detailed.iloc[0]["Detection Max Gap (Frames)"]) == 7
    assert int(detailed.iloc[0]["Detection Min Bout (Frames)"]) == 2
    assert float(detailed.iloc[0]["Analysis FPS"]) == 20.0
    assert int(summary.iloc[0]["Detection Max Gap (Frames)"]) == 7


def test_latency_metrics_pair_entries_and_exits_within_track(tmp_path):
    roi_events = {
        "entries": [
            {"frame": 0, "track_id": 1, "roi_name": "Center"},
            {"frame": 10, "track_id": 1, "roi_name": "Center"},
            {"frame": 2, "track_id": 2, "roi_name": "Center"},
        ],
        "exits": [
            {"frame": 5, "track_id": 1, "roi_name": "Center"},
            {"frame": 15, "track_id": 1, "roi_name": "Center"},
            {"frame": 8, "track_id": 2, "roi_name": "Center"},
        ],
    }
    context = AnalyticsModuleContext(
        per_frame_df=pd.DataFrame(),
        detailed_bouts_df=pd.DataFrame(),
        summary_df=pd.DataFrame(),
        roi_metrics={"events": roi_events},
        object_metrics=None,
        fps=10,
        class_names={},
        output_folder=str(tmp_path),
        video_name="multi_track",
        roi_column=None,
    )

    outputs = _module_latency_metrics(context)
    latency = pd.read_csv(outputs["files"]["zone_latency"])

    assert latency[["Track ID", "Target Name"]].to_dict("records") == [
        {"Track ID": 1, "Target Name": "Center"},
        {"Track ID": 2, "Target Name": "Center"},
    ]
    track_one = latency.loc[latency["Track ID"] == 1].iloc[0]
    track_two = latency.loc[latency["Track ID"] == 2].iloc[0]
    assert float(track_one["Re-entry Delay (s)"]) == pytest.approx(0.5)
    assert pd.isna(track_two["Re-entry Delay (s)"])


def test_normalized_roi_summary_uses_available_track_time(tmp_path):
    per_frame = pd.DataFrame(
        [
            {"track_id": track_id, "frame": frame, "ROI Name": "Center"}
            for track_id in (1, 2)
            for frame in range(10)
        ]
    )
    aggregate = pd.DataFrame(
        [{"ROI Name": "Center", "Entries": 2, "Time in ROI (s)": 2.0}]
    )
    per_track = pd.DataFrame(
        [
            {"Track ID": 1, "ROI Name": "Center", "Entries": 1, "Time in ROI (s)": 1.0},
            {"Track ID": 2, "ROI Name": "Center", "Entries": 1, "Time in ROI (s)": 1.0},
        ]
    )
    context = AnalyticsModuleContext(
        per_frame_df=per_frame,
        detailed_bouts_df=pd.DataFrame(),
        summary_df=pd.DataFrame(),
        roi_metrics={"entries_exits": aggregate, "entries_exits_by_track": per_track},
        object_metrics=None,
        fps=10,
        class_names={},
        output_folder=str(tmp_path),
        video_name="multi_track",
        roi_column="ROI Name",
    )

    outputs = _module_normalization_summary(context)
    pooled = pd.read_csv(outputs["files"]["zone_summary"]).iloc[0]
    by_track = pd.read_csv(outputs["files"]["zone_by_track"])

    assert int(pooled["Tracks in Analysis"]) == 2
    assert float(pooled["Available Track-Time (s)"]) == pytest.approx(2.0)
    assert float(pooled["Percent of Session"]) == pytest.approx(1.0)
    assert float(pooled["Entries per Minute"]) == pytest.approx(60.0)
    assert by_track["Percent of Session"].tolist() == pytest.approx([1.0, 1.0])


def test_roi_heatmap_counts_all_nested_memberships(tmp_path):
    per_frame = pd.DataFrame(
        [
            {
                "track_id": 1,
                "frame": frame,
                "ROI Name": "center",
                "ROI Memberships": ("center", "arena"),
            }
            for frame in range(3)
        ]
    )
    context = AnalyticsModuleContext(
        per_frame_df=per_frame,
        detailed_bouts_df=pd.DataFrame(),
        summary_df=pd.DataFrame(),
        roi_metrics={},
        object_metrics=None,
        fps=1,
        class_names={},
        output_folder=str(tmp_path),
        video_name="nested",
        roi_column="ROI Name",
    )

    outputs = _module_roi_time_heatmap(context, bin_size_seconds=60)
    occupancy = pd.read_csv(outputs["files"]["occupancy"]).set_index("ROI Name")

    assert float(occupancy.loc["arena", "Time in ROI (s)"]) == pytest.approx(3.0)
    assert float(occupancy.loc["center", "Time in ROI (s)"]) == pytest.approx(3.0)


def test_object_transition_sequence_preserves_revisits(tmp_path):
    object_events = {
        "entries": [
            {"frame": 0, "track_id": 1, "object_roi": "A"},
            {"frame": 5, "track_id": 1, "object_roi": "A"},
            {"frame": 10, "track_id": 1, "object_roi": "B"},
        ],
        "exits": [],
    }
    context = AnalyticsModuleContext(
        per_frame_df=pd.DataFrame(),
        detailed_bouts_df=pd.DataFrame(),
        summary_df=pd.DataFrame(),
        roi_metrics=None,
        object_metrics={"events": object_events},
        fps=10,
        class_names={},
        output_folder=str(tmp_path),
        video_name="revisits",
        roi_column=None,
    )

    outputs = _module_object_transition_analysis(context)
    sequence = pd.read_csv(outputs["files"]["sequence"])
    transitions = pd.read_csv(outputs["files"]["matrix"])

    assert sequence["Object ROI"].tolist() == ["A", "A", "B"]
    assert sequence["Is Revisit"].tolist() == [False, True, False]
    assert transitions[["From Object", "To Object", "Transition_Count"]].to_dict("records") == [
        {"From Object": "A", "To Object": "A", "Transition_Count": 1},
        {"From Object": "A", "To Object": "B", "Transition_Count": 1},
    ]

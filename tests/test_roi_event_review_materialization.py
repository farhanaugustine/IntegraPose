import json

import pandas as pd
import pytest

from integra_pose.utils.roi_event_review import (
    ROIReviewValidationError,
    materialize_reviewed_roi_events,
    register_authoritative_roi_review_in_manifest,
    save_reviewed_roi_bundle,
)


def _event(
    event_id: int,
    event_type: str,
    frame: int,
    *,
    status: str = "confirmed",
    target: str = "Center",
    track_id: int = 1,
) -> dict:
    return {
        "Event ID": event_id,
        "Source": "zone",
        "Event Type": event_type,
        "Target Name": target,
        "Track ID": track_id,
        "Frame": frame,
        "Review Status": status,
        "Corrected Event Type": event_type,
        "Corrected Target Name": target,
        "Corrected Track ID": track_id,
        "Corrected Frame": frame,
        "Reviewer Notes": "",
        "Corrected Manually": False,
    }


def test_materialize_reviewed_roi_events_applies_minimum_dwell() -> None:
    review = pd.DataFrame(
        [
            _event(1, "entry", 0),
            _event(2, "exit", 2),
            _event(3, "entry", 10),
            _event(4, "exit", 15),
        ]
    )

    result = materialize_reviewed_roi_events(
        review,
        fps=10,
        min_dwell_frames=5,
        max_gap_frames=1,
    )

    assert len(result["all_visits"]) == 2
    assert len(result["qualified_visits"]) == 1
    assert int(result["qualified_visits"].iloc[0]["Duration (Frames)"]) == 6
    overview = result["overview"].iloc[0]
    assert int(overview["Reviewed Visit Pairs"]) == 2
    assert int(overview["Below Minimum Dwell Visits"]) == 1
    assert int(overview["Entries"]) == 1
    assert int(overview["Qualified Dwell Frames"]) == 6


def test_materialize_reviewed_roi_events_rejects_unpaired_or_unreviewed_events() -> None:
    with pytest.raises(ROIReviewValidationError, match="still unreviewed"):
        materialize_reviewed_roi_events(
            pd.DataFrame([_event(1, "entry", 0, status="detected")]),
            fps=30,
            min_dwell_frames=1,
        )

    with pytest.raises(ROIReviewValidationError, match="unclosed entry"):
        materialize_reviewed_roi_events(
            pd.DataFrame([_event(1, "entry", 0)]),
            fps=30,
            min_dwell_frames=1,
        )


def test_save_and_register_reviewed_roi_bundle_preserves_raw_occupancy(tmp_path) -> None:
    review = pd.DataFrame([_event(1, "entry", 5), _event(2, "exit", 9)])
    raw_overview = pd.DataFrame(
        [{"ROI Name": "Center", "Frames in ROI": 20, "Time in ROI (s)": 2.0}]
    )
    raw_track = pd.DataFrame(
        [
            {
                "Track ID": 1,
                "ROI Name": "Center",
                "Frames in ROI": 20,
                "Time in ROI (s)": 2.0,
            }
        ]
    )

    artifacts = save_reviewed_roi_bundle(
        review,
        output_dir=tmp_path,
        video_name="video",
        fps=10,
        min_dwell_frames=5,
        max_gap_frames=1,
        raw_overview_df=raw_overview,
        raw_per_track_df=raw_track,
    )
    overview = pd.read_csv(artifacts["overview_path"])
    assert float(overview.iloc[0]["Raw Occupancy Time (s)"]) == 2.0
    assert float(overview.iloc[0]["Qualified Dwell Time (s)"]) == 0.5
    assert json.loads(
        (tmp_path / "video_reviewed_roi_validation.json").read_text(encoding="utf-8")
    )["status"] == "valid"

    updated = register_authoritative_roi_review_in_manifest(
        {
            "outputs": {
                "roi_events_csv": "raw_events.csv",
                "roi_metrics_files": {"entries_exits": "raw_overview.csv"},
                "modules": {
                    "event_aligned_windows": {"files": {"summary": "stale.csv"}},
                    "detection_quality": {"files": {"quality": "keep.csv"}},
                },
            }
        },
        artifacts,
        review_workspace_path="workspace.csv",
    )
    outputs = updated["outputs"]
    assert outputs["raw_roi_events_csv"] == "raw_events.csv"
    assert outputs["roi_events_csv"] == artifacts["events_path"]
    assert outputs["roi_metrics_files"]["entries_exits"] == artifacts["overview_path"]
    assert "event_aligned_windows" not in outputs["modules"]
    assert "detection_quality" in outputs["modules"]


def test_manifest_registration_keeps_concurrent_and_exclusive_roi_outputs_separate(
    tmp_path,
) -> None:
    review = pd.DataFrame([_event(1, "entry", 5), _event(2, "exit", 9)])
    concurrent = save_reviewed_roi_bundle(
        review,
        output_dir=tmp_path / "concurrent",
        video_name="video",
        fps=10,
        min_dwell_frames=1,
        max_gap_frames=0,
    )
    exclusive = save_reviewed_roi_bundle(
        review.assign(**{"Corrected Target Name": "Primary"}),
        output_dir=tmp_path / "exclusive",
        video_name="video_exclusive",
        fps=10,
        min_dwell_frames=1,
        max_gap_frames=0,
    )

    updated = register_authoritative_roi_review_in_manifest(
        {"outputs": {"roi_metrics_files": {}}},
        concurrent,
        review_workspace_path="review.sqlite3",
        exclusive_artifacts=exclusive,
    )

    outputs = updated["outputs"]
    assert (
        outputs["roi_metrics_files"]["dwell_events"]
        == concurrent["dwell_events_path"]
    )
    assert (
        outputs["roi_metrics_files"]["exclusive_dwell_events"]
        == exclusive["dwell_events_path"]
    )
    assert (
        outputs["roi_metrics_files"]["exclusive_entries_exits"]
        == exclusive["overview_path"]
    )
    assert (
        updated["notes"]["roi_review"]["occupancy_semantics"]
        == "separate_concurrent_and_exclusive_authoritative_bundles"
    )


def test_all_null_review_keeps_raw_occupancy_but_zeroes_qualified_dwell(
    tmp_path,
) -> None:
    review = pd.DataFrame([_event(1, "entry", 5), _event(2, "exit", 9)])
    review["Corrected Event Type"] = "null"
    raw_overview = pd.DataFrame(
        [{"ROI Name": "Center", "Frames in ROI": 20, "Time in ROI (s)": 2.0}]
    )
    raw_track = pd.DataFrame(
        [
            {
                "Track ID": 1,
                "ROI Name": "Center",
                "Frames in ROI": 20,
                "Time in ROI (s)": 2.0,
            }
        ]
    )

    artifacts = save_reviewed_roi_bundle(
        review,
        output_dir=tmp_path,
        video_name="video",
        fps=10,
        min_dwell_frames=5,
        max_gap_frames=1,
        raw_overview_df=raw_overview,
        raw_per_track_df=raw_track,
    )

    overview = pd.read_csv(artifacts["overview_path"])
    assert len(overview) == 1
    assert float(overview.iloc[0]["Raw Occupancy Time (s)"]) == 2.0
    assert float(overview.iloc[0]["Qualified Dwell Time (s)"]) == 0.0
    assert int(overview.iloc[0]["Entries"]) == 0
    assert artifacts["status"] == "valid_empty"

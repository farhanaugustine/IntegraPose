from pathlib import Path

import pandas as pd

from integra_pose.utils.batch_exporter import (
    build_analysis_coverage,
    build_video_summary,
    collect_batch_frames,
    collect_batch_module_tables,
)
from integra_pose.utils.bout_review import register_authoritative_review_in_manifest
from integra_pose.utils.safe_io import safe_write_json


def test_batch_exports_prefer_authoritative_reviewed_bouts(tmp_path: Path) -> None:
    raw_path = tmp_path / "video_detailed_bouts.csv"
    reviewed_path = tmp_path / "video_reviewed_bouts.csv"
    pd.DataFrame(
        [
            {"Bout ID": "a", "Track ID": 0, "Behavior": "walk", "Duration (s)": 1.0},
            {"Bout ID": "b", "Track ID": 0, "Behavior": "groom", "Duration (s)": 2.0},
        ]
    ).to_csv(raw_path, index=False)
    pd.DataFrame(
        [{"Bout ID": "a", "Track ID": 0, "Behavior": "walk", "Duration (s)": 1.0, "Review Status": "confirmed"}]
    ).to_csv(reviewed_path, index=False)
    result = {
        "video_id": "video-1",
        "video_name": "video",
        "video_path": str(tmp_path / "video.mp4"),
        "analytics_output_dir": str(tmp_path),
        "detailed_bouts_csv": str(raw_path),
        "reviewed_bouts_csv": str(reviewed_path),
    }

    _, _, bouts = collect_batch_frames([result])
    summary = build_video_summary([result])

    assert bouts["Bout ID"].tolist() == ["a"]
    assert summary.iloc[0]["bout_count"] == 1
    assert summary.iloc[0]["bout_duration_total_s"] == 1.0


def test_batch_exports_fall_back_to_raw_when_review_is_not_authoritative(tmp_path: Path) -> None:
    raw_path = tmp_path / "video_detailed_bouts.csv"
    pd.DataFrame(
        [{"Bout ID": "a", "Track ID": 0, "Behavior": "walk", "Duration (s)": 1.0}]
    ).to_csv(raw_path, index=False)
    result = {
        "video_id": "video-1",
        "video_name": "video",
        "video_path": str(tmp_path / "video.mp4"),
        "analytics_output_dir": str(tmp_path),
        "detailed_bouts_csv": str(raw_path),
        "reviewed_bouts_csv": str(tmp_path / "missing_reviewed_bouts.csv"),
    }

    _, _, bouts = collect_batch_frames([result])

    assert bouts["Bout ID"].tolist() == ["a"]


def test_unregistered_review_filename_never_overrides_manifest_active_raw_bouts(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "video_detailed_bouts.csv"
    stale_review = tmp_path / "video_reviewed_bouts.csv"
    pd.DataFrame(
        [{"Bout ID": "raw", "Behavior": "walk", "Duration (s)": 1.0}]
    ).to_csv(raw_path, index=False)
    pd.DataFrame(
        [{"Bout ID": "stale", "Behavior": "groom", "Duration (s)": 99.0}]
    ).to_csv(stale_review, index=False)
    result = {
        "video_id": "video-1",
        "video_name": "video",
        "video_path": str(tmp_path / "video.mp4"),
        "analytics_output_dir": str(tmp_path),
        "detailed_bouts_csv": str(raw_path),
    }

    _, _, bouts = collect_batch_frames([result])

    assert bouts["Bout ID"].tolist() == ["raw"]


def test_video_summary_separates_raw_occupancy_from_qualified_dwell(tmp_path: Path) -> None:
    roi_path = tmp_path / "roi.csv"
    object_path = tmp_path / "objects.csv"
    pd.DataFrame(
        [
            {
                "ROI Name": "Center",
                "Entries": 0,
                "Exits": 0,
                "Raw Occupancy Time (s)": 0.1,
                "Qualified Dwell Time (s)": 0.0,
            }
        ]
    ).to_csv(roi_path, index=False)
    pd.DataFrame(
        [
            {
                "Object ROI": "Object A",
                "Entries": 0,
                "Raw Interaction Time (s)": 0.1,
                "Qualified Interaction Time (s)": 0.0,
            }
        ]
    ).to_csv(object_path, index=False)

    summary = build_video_summary(
        [
            {
                "video_id": "video-1",
                "video_name": "video",
                "video_path": str(tmp_path / "video.mp4"),
                "roi_overview_csv": str(roi_path),
                "object_interactions_csv": str(object_path),
            }
        ]
    ).iloc[0]

    assert float(summary["roi_raw_occupancy_total_s"]) == 0.1
    assert float(summary["roi_qualified_dwell_total_s"]) == 0.0
    assert float(summary["roi_dwell_total_s"]) == 0.0
    assert float(summary["object_raw_interaction_total_s"]) == 0.1
    assert float(summary["object_qualified_interaction_total_s"]) == 0.0
    assert float(summary["object_interaction_total_s"]) == 0.0


def test_batch_module_collection_ignores_invalidated_raw_bout_modules(tmp_path: Path) -> None:
    raw_transition = tmp_path / "raw_transitions.csv"
    pd.DataFrame([{"From Behavior": "walk", "To Behavior": "groom"}]).to_csv(raw_transition, index=False)
    manifest = register_authoritative_review_in_manifest(
        {
            "outputs": {
                "detailed_bouts_csv": "raw.csv",
                "summary_csv": "raw_summary.csv",
                "modules": {
                    "behavior_transitions": {"files": {"summary": str(raw_transition)}}
                },
            }
        },
        {
            "raw_detected_path": "raw_snapshot.csv",
            "decisions_path": "decisions.csv",
            "workspace_path": "workspace.csv",
            "authoritative_path": "reviewed.csv",
            "summary_path": "reviewed_summary.csv",
        },
    )
    manifest_path = safe_write_json(tmp_path / "run_manifest.json", manifest)

    bundle = collect_batch_module_tables([{"run_manifest_json": str(manifest_path)}])
    coverage = build_analysis_coverage(
        [{"run_manifest_json": str(manifest_path), "video_id": "video-1"}]
    )

    assert bundle.file_index_df.empty
    assert bundle.tables == {}
    transition_row = coverage[
        coverage["analysis_key"] == "behavior_transitions"
    ].iloc[0]
    assert transition_row["status"] == "invalidated_after_review"

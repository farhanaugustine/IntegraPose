from integra_pose.logic.batch_preflight import (
    AnalysisPreflightConfig,
    PreflightVideoState,
    build_analysis_preflight_rows,
)


def _make_config(**overrides):
    payload = {
        "roi_strategy": "per_video",
        "use_existing_labels": False,
        "model_path": "",
        "labels_root": "",
        "object_interaction_enabled": True,
        "object_count": 2,
        "include_kpss": False,
        "class_count": 0,
        "enabled_metrics": set(),
        "metric_specs": (),
    }
    payload.update(overrides)
    return AnalysisPreflightConfig(**payload)


def test_preflight_lists_missing_roi_and_object_roi_videos() -> None:
    rows = build_analysis_preflight_rows(
        _make_config(),
        [
            PreflightVideoState(
                video_id="video-1",
                video_name="video_one.mp4",
                video_path="C:/videos/video_one.mp4",
                has_rois=True,
                has_object_rois=True,
                group="Control",
                subject_id="MouseA",
                time_point="Day0",
            ),
            PreflightVideoState(
                video_id="video-2",
                video_name="video_two.mp4",
                video_path="C:/videos/video_two.mp4",
                has_rois=False,
                has_object_rois=True,
                group="Control",
                subject_id="MouseB",
                time_point="Day0",
            ),
            PreflightVideoState(
                video_id="video-3",
                video_name="video_three.mp4",
                video_path="C:/videos/video_three.mp4",
                has_rois=True,
                has_object_rois=False,
                group="Treatment",
                subject_id="MouseC",
                time_point="Day7",
            ),
        ],
    )

    missing_roi_rows = [row for row in rows if row["analysis"] == "Missing arena ROI"]
    missing_object_rows = [row for row in rows if row["analysis"] == "Missing object ROI"]

    assert missing_roi_rows == [
        {
            "analysis": "Missing arena ROI",
            "will_run": "Fix",
            "scope": "video-2 | video_two.mp4",
            "variables": "Group=Control | Subject=MouseB | Time=Day0",
            "reason": "Draw arena ROI(s) for this video before batch run.",
        }
    ]
    assert missing_object_rows == [
        {
            "analysis": "Missing object ROI",
            "will_run": "Fix",
            "scope": "video-3 | video_three.mp4",
            "variables": "Group=Treatment | Subject=MouseC | Time=Day7",
            "reason": "Place object ROI(s) for this video before object interaction analytics.",
        }
    ]


def test_preflight_reports_missing_shared_sets_in_single_mode() -> None:
    rows = build_analysis_preflight_rows(
        _make_config(roi_strategy="single"),
        [
            PreflightVideoState(
                video_id="video-1",
                video_name="video_one.mp4",
                video_path="C:/videos/video_one.mp4",
                has_rois=False,
                has_object_rois=False,
            )
        ],
        shared_has_rois=False,
        shared_has_object_rois=False,
    )

    assert any(
        row["analysis"] == "Missing arena ROI setup" and row["scope"] == "Shared ROI set"
        for row in rows
    )
    assert any(
        row["analysis"] == "Missing object ROI setup" and row["scope"] == "Shared object ROI set"
        for row in rows
    )

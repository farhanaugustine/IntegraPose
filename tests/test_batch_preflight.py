from pathlib import Path

from integra_pose.logic.batch_preflight import (
    AnalysisPreflightConfig,
    PreflightVideoState,
    _default_find_labels_dir,
    _default_index_label_dirs,
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


def test_single_arena_strategy_accepts_per_video_object_placements(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "pose_model.pt"
    model_path.write_bytes(b"test")
    rows = build_analysis_preflight_rows(
        _make_config(roi_strategy="single", model_path=str(model_path)),
        [
            PreflightVideoState(
                video_id="video-1",
                video_name="video_one.mp4",
                video_path="C:/videos/video_one.mp4",
                has_rois=False,
                has_object_rois=True,
            ),
            PreflightVideoState(
                video_id="video-2",
                video_name="video_two.mp4",
                video_path="C:/videos/video_two.mp4",
                has_rois=False,
                has_object_rois=True,
            ),
        ],
        shared_has_rois=False,
        shared_has_object_rois=False,
    )

    object_row = next(
        row for row in rows
        if row["analysis"] == "Object interaction summaries"
    )
    assert object_row["will_run"] == "Yes"
    assert object_row["scope"] == "2/2 video(s)"
    assert not any(
        row["analysis"] in {"Missing object ROI", "Missing object ROI setup"}
        for row in rows
    )


def test_label_index_ignores_auxiliary_text_only_directories(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "classes.txt").write_text("mouse\n", encoding="utf-8")
    (labels_dir / "notes.txt").write_text("reviewed\n", encoding="utf-8")

    assert _default_index_label_dirs(tmp_path) == []


def test_existing_label_lookup_uses_exact_source_boundary(tmp_path: Path) -> None:
    mouse_10 = tmp_path / "run_mouse10" / "labels"
    mouse_1 = tmp_path / "run_mouse1" / "labels"
    mouse_10.mkdir(parents=True)
    mouse_1.mkdir(parents=True)
    (mouse_10 / "mouse10_frame_000000.txt").write_text("", encoding="utf-8")
    (mouse_1 / "mouse1_frame_000000.txt").write_text("", encoding="utf-8")
    indexed = _default_index_label_dirs(tmp_path)

    resolved = _default_find_labels_dir(
        video_stem="mouse1",
        preferred_dir=None,
        labels_root=tmp_path,
        indexed_dirs=indexed,
    )

    assert resolved == mouse_1.resolve()


def test_existing_label_lookup_does_not_treat_mouse10_as_mouse1(tmp_path: Path) -> None:
    mouse_10 = tmp_path / "run_mouse10" / "labels"
    mouse_10.mkdir(parents=True)
    (mouse_10 / "mouse10_frame_000000.txt").write_text("", encoding="utf-8")

    resolved = _default_find_labels_dir(
        video_stem="mouse1",
        preferred_dir=None,
        labels_root=tmp_path,
        indexed_dirs=_default_index_label_dirs(tmp_path),
    )

    assert resolved is None


def test_existing_label_lookup_rejects_equal_scoring_directories(tmp_path: Path) -> None:
    first = tmp_path / "run_a" / "labels"
    second = tmp_path / "run_b" / "labels"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "mouse1_frame_000000.txt").write_text("", encoding="utf-8")
    (second / "mouse1_frame_000000.txt").write_text("", encoding="utf-8")

    resolved = _default_find_labels_dir(
        video_stem="mouse1",
        preferred_dir=None,
        labels_root=tmp_path,
        indexed_dirs=_default_index_label_dirs(tmp_path),
    )

    assert resolved is None


def test_preflight_flags_each_missing_design_value_by_video(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"test")
    videos = [
        PreflightVideoState(
            video_id="video-1",
            video_name="one.mp4",
            video_path="C:/videos/one.mp4",
            has_rois=False,
            has_object_rois=False,
            group="Control",
            subject_id="",
            time_point="Day0",
        ),
        PreflightVideoState(
            video_id="video-2",
            video_name="two.mp4",
            video_path="C:/videos/two.mp4",
            has_rois=False,
            has_object_rois=False,
            group="",
            subject_id="Mouse02",
            time_point="",
        ),
    ]

    rows = build_analysis_preflight_rows(
        _make_config(
            model_path=str(model_path),
            object_interaction_enabled=False,
        ),
        videos,
    )

    fixes = {
        (row["analysis"], row["scope"], row["variables"])
        for row in rows
        if row["will_run"] == "Fix"
        and row["analysis"].startswith("Missing ")
    }
    assert any(
        analysis == "Missing Subject ID"
        and scope == "video-1 | one.mp4"
        and "Missing=Subject ID" in variables
        for analysis, scope, variables in fixes
    )
    assert any(
        analysis == "Missing design metadata"
        and scope == "video-2 | two.mp4"
        and "Missing=Group, Time Point" in variables
        for analysis, scope, variables in fixes
    )


def test_preflight_separates_mixed_effects_and_kpss_readiness(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"test")
    videos = [
        PreflightVideoState(
            video_id=f"{group}-{subject}-{day}",
            video_name=f"{group}_{subject}_Day{day}.mp4",
            video_path=f"C:/videos/{group}_{subject}_Day{day}.mp4",
            has_rois=False,
            has_object_rois=False,
            group=group,
            subject_id=subject,
            time_point=f"Day{day}",
        )
        for group, subjects in (
            ("Control", ("C1", "C2")),
            ("Treatment", ("T1", "T2")),
        )
        for subject in subjects
        for day in (0, 1, 2, 3, 4)
    ]

    rows = build_analysis_preflight_rows(
        _make_config(
            model_path=str(model_path),
            object_interaction_enabled=False,
            include_mixed_effects=True,
            include_kpss=True,
        ),
        videos,
    )

    mixed = next(row for row in rows if row["analysis"] == "Mixed-effects models")
    kpss = next(
        row
        for row in rows
        if row["analysis"] == "KPSS stationarity diagnostic"
    )
    assert mixed["will_run"] == "Yes"
    assert kpss["will_run"] == "Yes"
    assert "Control" in kpss["scope"]
    assert "Treatment" in kpss["scope"]


def test_group_preflight_counts_independent_subjects_not_video_rows(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"test")
    videos = [
        PreflightVideoState(
            video_id=f"{group}-{day}",
            video_name=f"{group}_Day{day}.mp4",
            video_path=f"C:/videos/{group}_Day{day}.mp4",
            has_rois=False,
            has_object_rois=False,
            group=group,
            subject_id=subject,
            time_point=f"Day{day}",
        )
        for group, subject in (("Control", "C1"), ("Treatment", "T1"))
        for day in (0, 1, 2, 3, 4)
    ]

    rows = build_analysis_preflight_rows(
        _make_config(
            model_path=str(model_path),
            object_interaction_enabled=False,
        ),
        videos,
    )

    group_row = next(
        row
        for row in rows
        if row["analysis"] == "Group comparison statistics"
    )
    assert group_row["will_run"] == "No"
    assert "independent subject/video units" in group_row["reason"]
    assert "smallest group has 1" in group_row["reason"]


def test_preflight_flags_unavailable_additional_factors() -> None:
    rows = build_analysis_preflight_rows(
        _make_config(
            object_interaction_enabled=False,
            categorical_factors=("housing_condition",),
        ),
        [
            PreflightVideoState(
                video_id="video-1",
                video_name="one.mp4",
                video_path="C:/videos/one.mp4",
                has_rois=False,
                has_object_rois=False,
            )
        ],
    )

    factor_row = next(
        row
        for row in rows
        if row["analysis"] == "Unavailable categorical factor"
    )
    assert factor_row["will_run"] == "Fix"
    assert factor_row["scope"] == "housing_condition"


def test_preflight_does_not_mark_an_empty_builtin_factor_as_ready() -> None:
    rows = build_analysis_preflight_rows(
        _make_config(
            object_interaction_enabled=False,
            categorical_factors=("time",),
        ),
        [
            PreflightVideoState(
                video_id="video-1",
                video_name="one.mp4",
                video_path="C:/videos/one.mp4",
                has_rois=False,
                has_object_rois=False,
                group="Control",
                subject_id="Mouse01",
                time_point="",
            )
        ],
    )

    factor_row = next(
        row
        for row in rows
        if row["analysis"] == "Configured factor: time_point"
    )
    assert factor_row["will_run"] == "Fix"
    assert factor_row["scope"] == "0/1 video(s) | 0 level(s)"


def test_model_load_error_blocks_core_but_keeps_full_preflight_rows(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "broken.pt"
    model_path.write_bytes(b"not-a-model")
    rows = build_analysis_preflight_rows(
        _make_config(
            model_path=str(model_path),
            model_preflight_error="checkpoint could not be loaded",
            object_interaction_enabled=False,
        ),
        [
            PreflightVideoState(
                video_id="video-1",
                video_name="one.mp4",
                video_path="C:/videos/one.mp4",
                has_rois=False,
                has_object_rois=False,
            )
        ],
    )

    inference = next(
        row for row in rows if row["analysis"] == "Run model inference"
    )
    core = next(
        row for row in rows if row["analysis"] == "Behavior bout summaries"
    )
    assert inference["will_run"] == "No"
    assert "checkpoint could not be loaded" in inference["reason"]
    assert core["will_run"] == "No"
    assert any(row["analysis"] == "Missing design metadata" for row in rows)

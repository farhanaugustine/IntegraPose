from __future__ import annotations

import json
import importlib
from pathlib import Path
import sys

import pandas as pd
import pytest

# A few legacy test modules import an already-installed IntegraPose package
# during collection. Ensure this new source-only subpackage is discoverable
# from the checkout as well, regardless of collection order.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import integra_pose

_LOCAL_PACKAGE = str(_REPO_ROOT / "integra_pose")
if _LOCAL_PACKAGE not in integra_pose.__path__:
    integra_pose.__path__.insert(0, _LOCAL_PACKAGE)
importlib.invalidate_caches()

from integra_pose.bout_reviewer import manifest_project
from integra_pose.bout_reviewer.analytics import behavior_correction_rows
from integra_pose.bout_reviewer.exporter import export_review
from integra_pose.bout_reviewer.integration import (
    materialize_integrapose_review,
)
from integra_pose.bout_reviewer.models import (
    BEHAVIOR,
    OBJECT_INTERACTION,
    ROI_CONCURRENT,
    ROI_EXCLUSIVE,
    ReviewError,
)
from integra_pose.bout_reviewer.scoring import score_store_sweep
from integra_pose.bout_reviewer.store import ReviewStore


def _write_interval_table(
    path: Path,
    *,
    label_column: str,
    rows: list[tuple[str, int, int, int]],
) -> None:
    pd.DataFrame(
        [
            {
                label_column: label,
                "Track ID": track_id,
                "Start Frame": start,
                "End Frame": end,
                "Duration (Frames)": end - start + 1,
            }
            for label, track_id, start, end in rows
        ]
    ).to_csv(path, index=False)


def _run_manifest_project(
    tmp_path: Path,
    monkeypatch,
):
    run_root = tmp_path / "run"
    run_root.mkdir()
    behavior = run_root / "mouse_detailed_bouts.csv"
    pd.DataFrame(
        [
            {
                "Run ID": "run-123",
                "Bout ID": "bout-rear",
                "Track ID": 7,
                "Class ID": 0,
                "Behavior": "rear",
                "Start Frame": 10,
                "End Frame": 20,
                "Duration (Frames)": 11,
            },
            {
                "Run ID": "run-123",
                "Bout ID": "bout-wall-rear",
                "Track ID": 7,
                "Class ID": 1,
                "Behavior": "wall_rear",
                "Start Frame": 15,
                "End Frame": 25,
                "Duration (Frames)": 11,
            },
        ]
    ).to_csv(behavior, index=False)

    concurrent = run_root / "mouse_roi_dwell_events.csv"
    exclusive = run_root / "mouse_roi_exclusive_dwell_events.csv"
    objects = run_root / "mouse_object_interactions_dwell_events.csv"
    _write_interval_table(
        concurrent,
        label_column="ROI Name",
        rows=[("Center", 7, 2, 8)],
    )
    _write_interval_table(
        exclusive,
        label_column="ROI Name",
        rows=[("Center", 7, 2, 8)],
    )
    _write_interval_table(
        objects,
        label_column="Object ROI",
        rows=[("Novel object", 7, 40, 47)],
    )

    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-123",
                "video": {"base_name": "mouse"},
                "inputs": {
                    "video_file": "external_drive/source/mouse.mp4",
                    "yolo_folder": "external_drive/lost_labels",
                    "behavior_names": ["rear", "wall_rear"],
                    "roi_polygons": {"Center": []},
                    "object_roi_polygons": {"Novel object": []},
                },
                "parameters": {
                    "fps": 30.0,
                    "single_animal_mode": False,
                    "max_gap_frames": 3,
                    "min_bout_frames": 5,
                    "behavior_bout_class_mode": "multi_label",
                },
                "outputs": {
                    "detailed_bouts_csv": str(behavior),
                    "roi_metrics_files": {
                        "dwell_events": str(concurrent),
                        "exclusive_dwell_events": str(exclusive),
                    },
                    "object_interaction_files": {
                        "dwell_events": str(objects),
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        manifest_project,
        "_select_display_video",
        lambda **_kwargs: (
            manifest_path,
            "analytics_annotated",
            (100, 30.0, 640, 480),
            [],
        ),
    )
    return manifest_project.load_run_manifest_project(manifest_path)


def test_manifest_loader_uses_analytics_tables_without_yolo_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _run_manifest_project(tmp_path, monkeypatch)
    video = project.videos[0]

    assert video.single_animal_mode is False
    assert video.behavior_classes == {0: "rear", 1: "wall_rear"}
    assert video.behavior_settings["behavior_bout_class_mode"] == "multi_label"
    assert {
        prediction.event_kind for prediction in video.predictions
    } == {
        BEHAVIOR,
        ROI_CONCURRENT,
        ROI_EXCLUSIVE,
        OBJECT_INTERACTION,
    }
    behavior = [
        prediction
        for prediction in video.predictions
        if prediction.event_kind == BEHAVIOR
    ]
    assert [(bout.class_id, bout.start_frame, bout.end_frame) for bout in behavior] == [
        (0, 10, 20),
        (1, 15, 25),
    ]
    assert (
        video.path_provenance["yolo_folder_recorded_for_provenance_only"]
        == "external_drive/lost_labels"
    )


def test_behavior_store_allows_cooccurrence_but_merges_only_same_class(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _run_manifest_project(tmp_path, monkeypatch)
    store = ReviewStore(":memory:")
    try:
        store.sync_project(project)
        bouts = store.list_review_bouts(
            project.videos[0].video_id,
            BEHAVIOR,
            include_inactive=False,
        )
        assert len(store.behavior_overlap_rows()) == 1
        assert store.behavior_overlap_rows()[0]["same_class"] == 0

        with pytest.raises(ReviewError, match="share video, event type, class"):
            store.merge_bouts([bout.review_id for bout in bouts], "Reviewer")

        wall_rear = next(bout for bout in bouts if bout.class_id == 1)
        corrected = store.update_bout(
            wall_rear.review_id,
            event_kind=BEHAVIOR,
            class_id=0,
            label="rear",
            track_id=7,
            start_frame=wall_rear.start_frame,
            end_frame=wall_rear.end_frame,
            note="Corrected class after video review.",
            reviewer="Reviewer",
        )
        rear = next(bout for bout in bouts if bout.class_id == 0)
        merged = store.merge_bouts(
            [rear.review_id, corrected.review_id],
            "Reviewer",
        )

        assert merged.class_id == 0
        assert merged.label == "rear"
        assert merged.start_frame == 10
        assert merged.end_frame == 25
        assert set(merged.origin_prediction_ids) == {
            rear.origin_prediction_ids[0],
            wall_rear.origin_prediction_ids[0],
        }
    finally:
        store.close()


def test_behavior_class_and_track_corrections_are_audited(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _run_manifest_project(tmp_path, monkeypatch)
    store = ReviewStore(":memory:")
    try:
        store.sync_project(project)
        rear = next(
            bout
            for bout in store.list_review_bouts(event_kind=BEHAVIOR)
            if bout.class_id == 0
        )
        store.update_bout(
            rear.review_id,
            event_kind=BEHAVIOR,
            class_id=1,
            label="wall_rear",
            track_id=8,
            start_frame=rear.start_frame,
            end_frame=rear.end_frame,
            note="Class and tracklet corrected from the video.",
            reviewer="Reviewer",
        )

        payload = json.loads(store.action_rows()[-1]["payload_json"])
        assert set(payload["change_types"]) == {"class", "track", "note"}
        rows = behavior_correction_rows(store)
        original = next(
            row
            for row in rows
            if row.scope == "video_behavior_class" and row.class_id == "0"
        )
        destination = next(
            row
            for row in rows
            if row.scope == "video_behavior_class" and row.class_id == "1"
        )
        assert original.changed_unique_predictions == 1
        assert original.reclassified_from == 1
        assert original.track_corrected == 1
        assert destination.reclassified_into == 1
    finally:
        store.close()


def test_complete_review_materializes_separate_authoritative_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _run_manifest_project(tmp_path, monkeypatch)
    video = project.videos[0]
    database = project.root / "bout_review_workspace" / "review.sqlite3"
    store = ReviewStore(database)
    try:
        store.sync_project(project)
        review_bouts = store.list_review_bouts(include_inactive=False)
        store.accept(
            [bout.review_id for bout in review_bouts],
            "Reviewer",
        )
        for track_id in store.behavior_track_ids(video.video_id):
            store.mark_scope(
                video.video_id,
                BEHAVIOR,
                True,
                "Reviewer",
                track_id=track_id,
            )
        for event_kind in (
            ROI_CONCURRENT,
            ROI_EXCLUSIVE,
            OBJECT_INTERACTION,
        ):
            store.mark_scope(
                video.video_id,
                event_kind,
                True,
                "Reviewer",
            )

        result = materialize_integrapose_review(
            project,
            store,
            project.root / "export",
        )
    finally:
        store.close()

    assert result["manifest_updated"] is True
    assert result["behavior"] == "complete"
    assert result["roi"] == "complete"
    assert result["object_interaction"] == "complete"
    assert result["spatial"] == "complete"

    manifest = json.loads(
        project.session_path.read_text(encoding="utf-8")
    )
    outputs = manifest["outputs"]
    roi_files = outputs["roi_metrics_files"]
    assert outputs["bout_review_database"] == (
        "bout_review_workspace/review.sqlite3"
    )
    assert outputs["bout_review_status_json"] == (
        "bout_review_workspace/last_review_status.json"
    )
    portable_paths = outputs["bout_review_portable_paths"]
    assert portable_paths["path_policy"] == (
        "run_manifest_parent_relative_v1"
    )
    assert not Path(
        portable_paths["behavior_artifacts"]["authoritative_path"]
    ).is_absolute()
    assert roi_files["dwell_events"] != roi_files["exclusive_dwell_events"]
    assert (
        Path(roi_files["dwell_events"]).parent.name == "Concurrent"
    )
    assert (
        Path(roi_files["exclusive_dwell_events"]).parent.name == "Exclusive"
    )
    reviewed_behavior = pd.read_csv(outputs["reviewed_bouts_csv"])
    assert set(reviewed_behavior["Class ID"].astype(int)) == {0, 1}
    assert Path(outputs["object_interaction_files"]["dwell_events"]).is_file()
    assert (
        project.root
        / "bout_review_workspace"
        / "last_review_status.json"
    ).is_file()


def test_all_rejected_review_materializes_readable_empty_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _run_manifest_project(tmp_path, monkeypatch)
    video = project.videos[0]
    store = ReviewStore(
        project.root / "bout_review_workspace" / "review.sqlite3"
    )
    try:
        store.sync_project(project)
        store.reject(
            [
                bout.review_id
                for bout in store.list_review_bouts(include_inactive=False)
            ],
            "Reviewer",
        )
        for track_id in store.behavior_track_ids(video.video_id):
            store.mark_scope(
                video.video_id,
                BEHAVIOR,
                True,
                "Reviewer",
                track_id=track_id,
            )
        for event_kind in (
            ROI_CONCURRENT,
            ROI_EXCLUSIVE,
            OBJECT_INTERACTION,
        ):
            store.mark_scope(
                video.video_id,
                event_kind,
                True,
                "Reviewer",
            )
        result = materialize_integrapose_review(
            project,
            store,
            project.root / "all_rejected_export",
        )
    finally:
        store.close()

    behavior = pd.read_csv(
        result["behavior_artifacts"]["authoritative_path"]
    )
    object_dwell = pd.read_csv(
        result["object_interaction_artifacts"]["dwell_events"]
    )
    assert behavior.empty
    assert {"Class ID", "Review Decision"}.issubset(behavior.columns)
    assert object_dwell.empty
    assert {"Object ROI", "Review Decision"}.issubset(object_dwell.columns)
    assert result["roi_concurrent_artifacts"]["status"] == "valid_empty"
    assert result["roi_exclusive_artifacts"]["status"] == "valid_empty"


def test_review_export_keeps_mode_families_and_advanced_tiou_separate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _run_manifest_project(tmp_path, monkeypatch)
    video = project.videos[0]
    store = ReviewStore(":memory:")
    try:
        store.sync_project(project)
        store.accept(
            [
                bout.review_id
                for bout in store.list_review_bouts(include_inactive=False)
            ],
            "Reviewer",
        )
        for track_id in store.behavior_track_ids(video.video_id):
            store.mark_scope(
                video.video_id,
                BEHAVIOR,
                True,
                "Reviewer",
                track_id=track_id,
            )
        for event_kind in (
            ROI_CONCURRENT,
            ROI_EXCLUSIVE,
            OBJECT_INTERACTION,
        ):
            store.mark_scope(
                video.video_id,
                event_kind,
                True,
                "Reviewer",
            )
        output = export_review(
            project,
            store,
            score_store_sweep(store, advanced=True),
            event_iou_thresholds=[0.25, 0.5, 0.75, 0.95],
        )
    finally:
        store.close()

    assert {
        path.name
        for path in output.iterdir()
        if path.is_dir()
    }.issuperset(
        {
            "Behavior_Bouts",
            "ROI_Bouts",
            "Object_Interactions",
            "Shared_Audit",
        }
    )
    for category in (
        "Behavior_Bouts",
        "ROI_Bouts",
        "Object_Interactions",
    ):
        assert len(list((output / category / "Figures").glob("*.png"))) == 4
    export_manifest = json.loads(
        (output / "review_export_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert export_manifest["advanced_tiou_sweep"] is True
    assert export_manifest["temporal_event_iou_thresholds"] == [
        0.25,
        0.5,
        0.75,
        0.95,
    ]


def test_review_window_keeps_tooltips_and_integrapose_palette(
    tmp_path: Path,
) -> None:
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from integra_pose.bout_reviewer.app import MainWindow

    application = QApplication.instance() or QApplication([])
    settings = QSettings(
        str(tmp_path / "reviewer-layout.ini"),
        QSettings.Format.IniFormat,
    )
    settings.clear()
    window = MainWindow(initial_root=None, settings=settings)
    try:
        controls = (
            window.video_combo,
            window.frame_slider,
            window.play_button,
            window.review_mode,
            window.event_table,
            window.add_button,
            window.split_button,
            window.merge_button,
            window.acknowledge_overlap_button,
            window.iou_threshold,
            window.advanced_iou_sweep,
            window.score_table,
            window.correction_table,
            window.overlap_table,
            window.timeline,
            window.timeline.canvas,
            window.timeline.zoom,
        )
        assert all(control.toolTip().strip() for control in controls)
        assert window.open_action.toolTip().strip()
        assert window.source_videos_action.toolTip().strip()
        assert window.provenance_action.toolTip().strip()
        assert window.show_video_action.toolTip().strip()
        assert window.show_review_action.toolTip().strip()
        assert window.show_timeline_action.toolTip().strip()
        assert window.reset_layout_action.toolTip().strip()
        assert window.iou_threshold.value() == pytest.approx(0.5)
        assert window.advanced_iou_sweep.isChecked() is False
        assert "#6ee7b7" in window.styleSheet().casefold()
        assert "integraPose".casefold() in window.windowTitle().casefold()
    finally:
        window.close()
        application.processEvents()


def test_review_window_layout_is_resizable_persistent_and_recoverable(
    tmp_path: Path,
) -> None:
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QHeaderView

    from integra_pose.bout_reviewer.app import MainWindow

    application = QApplication.instance() or QApplication([])
    settings = QSettings(
        str(tmp_path / "persistent-layout.ini"),
        QSettings.Format.IniFormat,
    )
    settings.clear()

    window = MainWindow(initial_root=None, settings=settings)
    window.show()
    application.processEvents()
    try:
        assert (window.minimumWidth(), window.minimumHeight()) == (760, 480)
        assert (
            window.video_display.minimumWidth(),
            window.video_display.minimumHeight(),
        ) == (320, 180)
        assert window.upper_splitter.handleWidth() == 8
        assert window.main_splitter.handleWidth() == 8
        assert window.timeline.scroll_area.maximumHeight() > 10_000

        event_header = window.event_table.horizontalHeader()
        assert event_header.sectionsMovable() is True
        assert event_header.sectionResizeMode(0) == (
            QHeaderView.ResizeMode.Interactive
        )

        window.resize(780, 500)
        window.upper_splitter.setSizes([390, 330])
        window.main_splitter.setSizes([330, 130])
        event_header.resizeSection(0, 137)
        window.show_timeline_action.setChecked(False)
        application.processEvents()
        assert window.timeline.isHidden() is True
        window.move(5000, 5000)
    finally:
        window.close()
        application.processEvents()

    restored = MainWindow(initial_root=None, settings=settings)
    restored.show()
    restored._ensure_window_on_screen()
    application.processEvents()
    try:
        assert restored.show_timeline_action.isChecked() is False
        assert restored.timeline.isHidden() is True
        assert restored.event_table.horizontalHeader().sectionSize(0) == 137
        assert any(
            screen.availableGeometry().intersects(restored.frameGeometry())
            for screen in QApplication.screens()
        )

        restored.reset_layout()
        application.processEvents()
        assert restored.show_video_action.isChecked() is True
        assert restored.show_review_action.isChecked() is True
        assert restored.show_timeline_action.isChecked() is True
        assert restored.timeline.isHidden() is False
        assert restored.event_table.horizontalHeader().sectionSize(0) == 92
        assert settings.value("window_layout/version") is None
    finally:
        restored.close()
        application.processEvents()

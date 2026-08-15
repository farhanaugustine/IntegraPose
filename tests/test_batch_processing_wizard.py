from unittest.mock import Mock
from types import SimpleNamespace
from pathlib import Path
import importlib.util
import json
import sys
import tkinter as tk
import numpy as np
import pandas as pd
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from integra_pose.gui import batch_processing_wizard as wizard_mod


def _load_module(rel_path: str, module_name: str):
    module_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_batch_session_mod = _load_module(
    "integra_pose/utils/batch_session.py",
    "test_batch_session_module",
)
BatchSession = _batch_session_mod.BatchSession
BatchVideoItem = _batch_session_mod.BatchVideoItem


class _FakeTree:
    def __init__(self, *selected_ids: str) -> None:
        self._selected_ids = tuple(selected_ids)
        self.rows = []

    def selection(self):
        return self._selected_ids

    def delete(self, *items):
        self.rows = []

    def get_children(self):
        return tuple(row["iid"] for row in self.rows)

    def insert(self, parent, index, iid=None, values=(), tags=()):
        self.rows.append({"iid": iid, "values": values, "tags": tuple(tags)})

    def selection_set(self, item_ids):
        if isinstance(item_ids, str):
            self._selected_ids = (item_ids,)
            return
        self._selected_ids = tuple(item_ids)


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def _make_item(
    video_id: str,
    video_name: str,
    *,
    group: str = "",
    subject_id: str = "",
    time_point: str = "",
    excluded: bool = False,
) -> BatchVideoItem:
    return BatchVideoItem(
        video_id=video_id,
        video_name=video_name,
        video_path=f"C:/videos/{video_name}.mp4",
        excluded=excluded,
        group=group,
        subject_id=subject_id,
        time_point=time_point,
    )


def _make_wizard(*selected_ids: str, items: list[BatchVideoItem]):
    wizard = wizard_mod.BatchProcessingWizard.__new__(wizard_mod.BatchProcessingWizard)
    wizard.queue_items = items
    wizard.queue_tree = _FakeTree(*selected_ids)
    refresh_queue_tree = wizard_mod.BatchProcessingWizard._refresh_queue_tree.__get__(wizard, wizard_mod.BatchProcessingWizard)
    wizard._refresh_queue_tree = Mock(wraps=refresh_queue_tree)
    wizard._refresh_roi_summary = Mock()
    wizard.progress_text_var = _FakeVar("")
    wizard.roi_name_templates_var = _FakeVar("")
    wizard.object_name_templates_var = _FakeVar("")
    wizard.roi_strategy_var = _FakeVar("per_video")
    wizard.queue_filter_var = _FakeVar("")
    wizard.session_status_var = _FakeVar("")
    wizard.shared_rois = {}
    wizard.shared_object_rois = {}
    wizard._queue_sort_column = "video_name"
    wizard._queue_sort_desc = False
    wizard._suspend_dirty_tracking = False
    wizard._has_unsaved_changes = False
    wizard._session_file_path = ""
    wizard._autosave_after_id = None
    return wizard


def test_batch_finish_classification_uses_explicit_status_not_message_prefix() -> None:
    classify = wizard_mod.BatchProcessingWizard._classify_batch_finish

    assert classify(False, "Everything is fine", {"status": "partial"}) == "partial"
    assert classify(False, "Neutral message", {"status": "failed"}) == "failed"
    assert classify(True, "Neutral message", {"status": "success"}) == "cancelled"
    assert classify(False, "Batch run failed: legacy text", None) == "success"


def test_partial_batch_finish_uses_warning_dialog_not_success_dialog(monkeypatch) -> None:
    wizard = wizard_mod.BatchProcessingWizard.__new__(wizard_mod.BatchProcessingWizard)
    wizard.queue_items = []
    wizard._current_session = None
    wizard._last_run_payload = None
    wizard._refresh_queue_tree = Mock()
    wizard._mark_session_clean = Mock()
    wizard.progress_text_var = _FakeVar("")
    wizard.run_button = Mock()
    wizard.resume_button = Mock()
    wizard.stop_button = Mock()
    wizard.app = SimpleNamespace(log_message=Mock())
    warnings = Mock()
    errors = Mock()
    infos = Mock()
    monkeypatch.setattr(wizard_mod.messagebox, "showwarning", warnings)
    monkeypatch.setattr(wizard_mod.messagebox, "showerror", errors)
    monkeypatch.setattr(wizard_mod.messagebox, "showinfo", infos)
    session = SimpleNamespace(review_policy="skip")

    wizard._on_batch_finish(
        session,
        False,
        "Batch run partially completed: 1/2 video(s) succeeded; 1 failed.",
        {
            "status": "partial",
            "workbook_path": "batch.xlsx",
            "session_json_path": "batch_session.json",
        },
    )

    warnings.assert_called_once()
    errors.assert_not_called()
    infos.assert_not_called()
    wizard.app.log_message.assert_called_once_with(
        "Batch run partially completed: 1/2 video(s) succeeded; 1 failed.",
        "WARNING",
    )


def test_failed_batch_finish_never_uses_success_dialog(monkeypatch) -> None:
    wizard = wizard_mod.BatchProcessingWizard.__new__(wizard_mod.BatchProcessingWizard)
    wizard.queue_items = []
    wizard._refresh_queue_tree = Mock()
    wizard._mark_session_clean = Mock()
    wizard.progress_text_var = _FakeVar("")
    wizard.run_button = Mock()
    wizard.resume_button = Mock()
    wizard.stop_button = Mock()
    wizard.app = SimpleNamespace(log_message=Mock())
    warnings = Mock()
    errors = Mock()
    infos = Mock()
    monkeypatch.setattr(wizard_mod.messagebox, "showwarning", warnings)
    monkeypatch.setattr(wizard_mod.messagebox, "showerror", errors)
    monkeypatch.setattr(wizard_mod.messagebox, "showinfo", infos)
    session = SimpleNamespace(review_policy="skip")

    wizard._on_batch_finish(
        session,
        False,
        "No videos completed.",
        {"status": "failed", "workbook_path": "", "session_json_path": ""},
    )

    errors.assert_called_once()
    warnings.assert_not_called()
    infos.assert_not_called()


def test_assign_group_applies_to_all_selected_videos() -> None:
    items = [
        _make_item("video-1", "video_one", group="Control"),
        _make_item("video-2", "video_two", group="Control"),
        _make_item("video-3", "video_three", group="Vehicle"),
    ]
    wizard = _make_wizard("video-1", "video-2", items=items)
    wizard._prompt_roi_name = Mock(return_value="Treatment")

    wizard._assign_group_to_selected()

    assert items[0].group == "Treatment"
    assert items[1].group == "Treatment"
    assert items[2].group == "Vehicle"
    wizard._refresh_queue_tree.assert_called_once()
    assert wizard._prompt_roi_name.call_args.kwargs["prompt"] == "Group for 2 selected videos:"


def test_assign_time_point_applies_to_all_selected_videos() -> None:
    items = [
        _make_item("video-1", "video_one", time_point="Baseline"),
        _make_item("video-2", "video_two", time_point="Baseline"),
        _make_item("video-3", "video_three", time_point="Week2"),
    ]
    wizard = _make_wizard("video-1", "video-2", items=items)
    wizard._prompt_roi_name = Mock(return_value="Week1")

    wizard._assign_time_point_to_selected()

    assert items[0].time_point == "Week1"
    assert items[1].time_point == "Week1"
    assert items[2].time_point == "Week2"
    wizard._refresh_queue_tree.assert_called_once()
    assert wizard._prompt_roi_name.call_args.kwargs["prompt"] == "Time point for 2 selected videos:"


def test_assign_subject_applies_to_all_selected_videos() -> None:
    items = [
        _make_item("video-1", "video_one", subject_id="MouseA", time_point="Baseline"),
        _make_item("video-2", "video_two", subject_id="MouseB", time_point="Baseline"),
        _make_item("video-3", "video_three", subject_id="MouseC", time_point="Week1"),
    ]
    wizard = _make_wizard("video-1", "video-2", items=items)
    wizard._prompt_roi_name = Mock(return_value="CohortSubject")

    wizard._assign_subject_to_selected()

    assert items[0].subject_id == "CohortSubject"
    assert items[1].subject_id == "CohortSubject"
    assert items[2].subject_id == "MouseC"
    assert wizard._prompt_roi_name.call_args.kwargs["prompt"] == "Subject ID for 2 selected videos:"


def test_edit_metadata_applies_distinct_values_to_multiple_selected_videos() -> None:
    items = [
        _make_item(
            "video-1",
            "video_one",
            group="Control",
            subject_id="MouseA",
            time_point="Baseline",
        ),
        _make_item(
            "video-2",
            "video_two",
            group="Control",
            subject_id="MouseB",
            time_point="Baseline",
        ),
        _make_item(
            "video-3",
            "video_three",
            group="Vehicle",
            subject_id="MouseC",
            time_point="Week2",
        ),
    ]
    wizard = _make_wizard("video-1", "video-2", items=items)
    wizard._prompt_metadata_assignments = Mock(
        return_value=[
            {
                "video_id": "video-1",
                "group": "Treatment",
                "subject_id": "Rat01",
                "time_point": "Day1",
            },
            {
                "video_id": "video-2",
                "group": "Treatment",
                "subject_id": "Rat02",
                "time_point": "Day7",
            },
        ]
    )

    wizard._edit_metadata_for_selected()

    assert (items[0].group, items[0].subject_id, items[0].time_point) == (
        "Treatment",
        "Rat01",
        "Day1",
    )
    assert (items[1].group, items[1].subject_id, items[1].time_point) == (
        "Treatment",
        "Rat02",
        "Day7",
    )
    assert (items[2].group, items[2].subject_id, items[2].time_point) == (
        "Vehicle",
        "MouseC",
        "Week2",
    )
    assert wizard.progress_text_var.get() == "Updated design metadata for 2 selected video(s)."


def test_edit_all_metadata_includes_every_nonexcluded_video() -> None:
    items = [
        _make_item("video-1", "video_one"),
        _make_item("video-2", "video_two"),
        _make_item("video-3", "video_three", excluded=True),
    ]
    wizard = _make_wizard(items=items)
    wizard._edit_metadata_for_items = Mock()

    wizard._edit_metadata_for_all_included()

    selected_items = wizard._edit_metadata_for_items.call_args.args[0]
    assert [item.video_id for item in selected_items] == ["video-1", "video-2"]


def test_exclude_and_include_selected_rows_updates_queue_state() -> None:
    items = [
        _make_item("video-1", "video_one"),
        _make_item("video-2", "video_two"),
    ]
    wizard = _make_wizard("video-1", items=items)

    wizard._exclude_selected_from_queue()

    assert items[0].excluded is True
    assert wizard.progress_text_var.get() == "Excluded 1 video(s) from the batch run."
    wizard.queue_tree._selected_ids = ("video-1",)

    wizard._include_selected_in_queue()

    assert items[0].excluded is False
    assert wizard.progress_text_var.get() == "Included 1 video(s) in the batch run."


def test_remove_selected_rows_prunes_queue() -> None:
    items = [
        _make_item("video-1", "video_one"),
        _make_item("video-2", "video_two"),
        _make_item("video-3", "video_three"),
    ]
    wizard = _make_wizard("video-1", "video-3", items=items)

    wizard._remove_selected_from_queue()

    assert [item.video_id for item in wizard.queue_items] == ["video-2"]
    assert wizard.progress_text_var.get() == "Removed 2 video(s) from the queue."


def test_discover_queue_merges_existing_rows_by_video_path() -> None:
    existing = _make_item("persisted_id", "video_one", group="Control", subject_id="MouseA", time_point="Day1")
    existing.rois = {"Center": {"polygon": [(0, 0), (1, 0), (1, 1)]}}
    wizard = _make_wizard(items=[existing])
    wizard.service = SimpleNamespace(
        discover_queue=Mock(
            return_value=[
                _make_item("new_id_one", "video_one"),
                _make_item("new_id_two", "video_two"),
            ]
        )
    )
    wizard.source_path_var = _FakeVar("C:/videos")
    wizard.recursive_scan_var = _FakeVar(False)

    wizard._discover_queue()

    assert len(wizard.queue_items) == 2
    assert wizard.queue_items[0] is existing
    assert wizard.queue_items[0].group == "Control"
    assert wizard.queue_items[0].rois
    assert wizard.queue_items[1].video_name == "video_two"
    assert wizard.progress_text_var.get() == "Discovered 2 video(s). Added 1 new row(s); queue now has 2 total."


def test_rediscovery_fills_blank_metadata_without_replacing_existing_rows() -> None:
    existing = _make_item("persisted_id", "video_one")
    existing.rois = {"Center": {"polygon": [(0, 0), (1, 0), (1, 1)]}}
    candidate = _make_item(
        "new_id",
        "video_one",
        group="Control",
        subject_id="Mouse01",
        time_point="Day7",
    )
    candidate.metadata_sources = {
        "group": "filename",
        "subject_id": "filename",
        "time_point": "filename",
    }
    wizard = _make_wizard(items=[existing])

    added = wizard._merge_discovered_items([candidate])

    assert added == 0
    assert wizard.queue_items == [existing]
    assert existing.video_id == "persisted_id"
    assert existing.group == "Control"
    assert existing.subject_id == "Mouse01"
    assert existing.time_point == "Day7"
    assert existing.metadata_sources["group"] == "filename"
    assert existing.rois


def test_parse_name_templates_trims_and_deduplicates() -> None:
    parsed = wizard_mod.BatchProcessingWizard._parse_name_templates(" Center , Start\n, center, Goal , ")

    assert parsed == ["Center", "Start", "Goal"]


def test_guided_roi_assignment_auto_advances_across_selected_videos(monkeypatch) -> None:
    items = [
        _make_item("video-1", "video_one"),
        _make_item("video-2", "video_two"),
        _make_item("video-3", "video_three"),
    ]
    wizard = _make_wizard("video-1", "video-2", items=items)
    wizard.roi_name_templates_var.set("Center, Edge")
    wizard._read_first_frame = Mock(return_value="frame")
    polygons = [
        [(1, 1), (2, 1), (2, 2)],
        [(3, 3), (4, 3), (4, 4)],
        [(5, 5), (6, 5), (6, 6)],
        [(7, 7), (8, 7), (8, 8)],
    ]
    draw_mock = Mock(side_effect=polygons)
    monkeypatch.setattr(wizard_mod, "draw_roi", draw_mock)

    wizard._draw_rois_for_selected_video()

    assert list(items[0].rois.keys()) == ["Center", "Edge"]
    assert list(items[1].rois.keys()) == ["Center", "Edge"]
    assert items[2].rois == {}
    assert items[0].roi_status == "done"
    assert items[1].roi_status == "done"
    assert len(wizard.queue_tree.rows) == 3
    wizard._refresh_roi_summary.assert_called_once()
    assert draw_mock.call_count == 4
    assert wizard.progress_text_var.get() == "Guided ROI drawing completed for 2 video(s) with 2 ROI name(s)."


def test_object_queue_uses_configured_template_and_every_included_video(monkeypatch) -> None:
    items = [
        _make_item("video-1", "video_one"),
        _make_item("video-2", "video_two"),
        _make_item("video-3", "video_three", excluded=True),
    ]
    wizard = _make_wizard(items=items)
    wizard.object_count_var = _FakeVar("2")
    wizard.object_roi_size_px_var = _FakeVar("48")
    wizard.object_roi_shape_var = _FakeVar("circle")
    wizard.object_distance_px_var = _FakeVar("12")
    wizard.object_name_templates_var = _FakeVar("Food, Novel object")
    captured = {}

    class _FakeController:
        def __init__(self, parent, **kwargs):
            captured["controller_parent"] = parent
            captured["controller_kwargs"] = kwargs

        def start(self):
            captured["started"] = True

    class _FakeBuilder:
        def __init__(self, parent, **kwargs):
            captured["builder_parent"] = parent
            captured["builder_kwargs"] = kwargs
            kwargs["on_confirm"](kwargs["existing_rows"])

    monkeypatch.setattr(wizard_mod, "_BatchRoiLoopController", _FakeController)
    monkeypatch.setattr(wizard_mod, "RoiBuilderDialog", _FakeBuilder)

    wizard._start_object_roi_queue_loop()

    rows = captured["builder_kwargs"]["existing_rows"]
    assert [row.name for row in rows] == ["Food", "Novel object"]
    assert [row.shape for row in rows] == ["circle", "circle"]
    assert [row.size_px for row in rows] == [48, 48]
    assert captured["builder_kwargs"]["existing_roi_names"] == ()
    controller_kwargs = captured["controller_kwargs"]
    assert [item.video_id for item in controller_kwargs["items"]] == ["video-1", "video-2"]
    assert controller_kwargs["object_mode"] is True
    assert controller_kwargs["template_rows"] == rows
    assert captured["started"] is True


def test_object_queue_save_automatically_advances_to_next_video(monkeypatch) -> None:
    items = [
        _make_item("video-1", "video_one"),
        _make_item("video-2", "video_two"),
    ]
    opened_positions = []
    committed_ids = []

    def _fake_editor(_parent, **kwargs):
        opened_positions.append(kwargs["loop_position"])
        kwargs["on_save"](kwargs["rois"])

    wizard = SimpleNamespace(
        _collect_video_rois_for_editor=lambda _item, object_mode=False: [],
        _capture_video_frame_for_loop=lambda _path, preferred_frame_index=None: (
            np.zeros((120, 160, 3), dtype=np.uint8),
            0,
        ),
        _commit_loop_save_to_video_store=lambda item, _edited, object_mode=False: committed_ids.append(item.video_id),
        _parse_nonnegative_float=lambda value, _label, _fallback: float(value),
        object_distance_px_var=_FakeVar("12"),
        app=SimpleNamespace(log_message=Mock()),
        _refresh_queue_tree=Mock(),
        _refresh_roi_summary=Mock(),
        _mark_session_dirty=Mock(),
        _show_loop_summary=Mock(),
    )
    monkeypatch.setattr(wizard_mod, "RoiEditor", _fake_editor)
    controller = wizard_mod._BatchRoiLoopController(
        wizard,
        items=items,
        template_rows=[
            wizard_mod.RoiBuilderRow(name="Object_1", shape="circle", size_px=32)
        ],
        object_mode=True,
    )

    controller.start()

    assert opened_positions == [(1, 2), (2, 2)]
    assert committed_ids == ["video-1", "video-2"]
    wizard._show_loop_summary.assert_called_once()
    assert wizard._show_loop_summary.call_args.kwargs["saved"] == ["video-1", "video-2"]


def test_object_distance_help_explains_edge_and_keypoint_semantics() -> None:
    help_text = wizard_mod.OBJECT_DISTANCE_HELP_TEXT.lower()

    assert "nearest edge" in help_text
    assert "not measured from the object's center" in help_text
    assert "selected object-interaction keypoint" in help_text
    assert "detection-only models" in help_text


def test_metadata_dialog_bulk_fill_and_per_video_override() -> None:
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    items = [
        _make_item("video-1", "video_one", group="Control", subject_id="MouseA"),
        _make_item("video-2", "video_two", group="Control", subject_id="MouseB"),
    ]
    dialog = None
    try:
        dialog = wizard_mod._BatchMetadataEditorDialog(
            root,
            items=items,
            known_values={
                "group": ["Control", "Treatment"],
                "subject_id": ["MouseA", "MouseB"],
                "time_point": ["Day1", "Day7"],
            },
        )
        dialog.withdraw()
        dialog._bulk_vars["group"].set("Treatment")
        dialog._bulk_vars["time_point"].set("Day1")
        dialog._fill_nonblank_columns()
        dialog._row_vars[1][1]["subject_id"].set("MouseZ")
        dialog._row_vars[1][1]["time_point"].set("Day7")

        dialog._save()

        assert dialog.result == [
            {
                "video_id": "video-1",
                "group": "Treatment",
                "subject_id": "MouseA",
                "time_point": "Day1",
            },
            {
                "video_id": "video-2",
                "group": "Treatment",
                "subject_id": "MouseZ",
                "time_point": "Day7",
            },
        ]
    finally:
        if dialog is not None:
            try:
                dialog.destroy()
            except tk.TclError:
                pass
        root.destroy()


def test_batch_session_round_trips_guided_name_templates() -> None:
    session = BatchSession.create()
    session.roi_name_templates = ["Center", "Edge"]
    session.object_name_templates = ["Cup_A", "Cup_B"]
    session.tracker_config_path = "bytetrack.yaml"
    session.figure_export_mode = "assay_shortlist"
    session.generate_video_quicklooks = False
    session.keypoint_names = ["Nose", "Thorax", "TailBase"]
    session.keypoint_names_source = "dataset_yaml"
    session.keypoint_schema_path = "C:/models/dataset.yaml"
    session.model_capabilities.keypoint_names = ["Nose", "Thorax", "TailBase"]
    session.model_capabilities.keypoint_names_source = "dataset_yaml"
    session.model_capabilities.keypoint_schema_path = "C:/models/dataset.yaml"
    session.roi_max_gap_frames = 17
    session.roi_min_dwell_frames = 9
    session.videos = [
        BatchVideoItem(video_id="video-1", video_name="video_one", video_path="C:/videos/video_one.mp4", excluded=True)
    ]

    restored = BatchSession.from_dict(session.to_dict())

    assert restored.roi_name_templates == ["Center", "Edge"]
    assert restored.object_name_templates == ["Cup_A", "Cup_B"]
    assert restored.tracker_config_path == "bytetrack.yaml"
    assert restored.figure_export_mode == "assay_shortlist"
    assert restored.generate_video_quicklooks is False
    assert restored.keypoint_names == ["Nose", "Thorax", "TailBase"]
    assert restored.keypoint_names_source == "dataset_yaml"
    assert restored.keypoint_schema_path == "C:/models/dataset.yaml"
    assert restored.model_capabilities.keypoint_names == ["Nose", "Thorax", "TailBase"]
    assert restored.roi_max_gap_frames == 17
    assert restored.roi_min_dwell_frames == 9
    assert restored.videos[0].excluded is True


def test_batch_session_round_trips_advanced_stats_and_metadata_provenance() -> None:
    session = BatchSession.create()
    session.stats_auto_detect_design = False
    session.include_mixed_effects = False
    session.include_kpss = True
    session.videos = [
        BatchVideoItem(
            video_id="video-1",
            video_name="Control_Mouse01_Day7.mp4",
            video_path="C:/videos/Control_Mouse01_Day7.mp4",
            group="Control",
            subject_id="Mouse01",
            time_point="Day7",
            metadata_sources={
                "group": "filename",
                "subject_id": "filename",
                "time_point": "filename",
            },
            metadata_warnings=["review me"],
        )
    ]

    restored = BatchSession.from_dict(session.to_dict())

    assert restored.stats_auto_detect_design is False
    assert restored.include_mixed_effects is False
    assert restored.include_kpss is True
    assert restored.videos[0].metadata_sources["group"] == "filename"
    assert restored.videos[0].metadata_warnings == ["review me"]


def test_legacy_combined_trend_setting_remains_backward_compatible() -> None:
    restored = BatchSession.from_dict(
        {
            "schema_version": 7,
            "include_kpss": True,
        }
    )

    assert restored.include_mixed_effects is True
    assert restored.include_kpss is True


def test_new_advanced_stats_defaults_enable_mixed_but_not_kpss() -> None:
    restored = BatchSession.from_dict(
        {
            "schema_version": 8,
        }
    )

    assert restored.stats_auto_detect_design is True
    assert restored.include_mixed_effects is True
    assert restored.include_kpss is False


def test_advanced_factor_aliases_are_normalized_and_validated() -> None:
    parse = wizard_mod.BatchProcessingWizard._parse_stats_categorical_factors
    validate = (
        wizard_mod.BatchProcessingWizard._validated_stats_categorical_factors
    )

    assert parse("Cohort, time-point, animal_id, cohort") == [
        "group",
        "time_point",
        "subject_id",
    ]
    assert validate("condition, visit, subject") == ["group", "time_point"]
    with pytest.raises(ValueError, match="housing_condition"):
        validate("housing_condition")


def test_manual_metadata_assignment_updates_provenance_and_clears_stale_warning() -> None:
    item = _make_item("video-1", "video_one")
    item.metadata_sources = {"group": "folder:Control"}
    item.metadata_warnings = [
        "Conflicting group candidates were found; assign this field manually.",
        "Conflicting time point candidates were found; assign this field manually.",
    ]

    wizard_mod.BatchProcessingWizard._set_design_metadata_value(
        item,
        "group",
        "ReviewedControl",
    )

    assert item.group == "ReviewedControl"
    assert item.metadata_sources["group"] == "manual"
    assert not any("group" in warning.casefold() for warning in item.metadata_warnings)
    assert any("time point" in warning.casefold() for warning in item.metadata_warnings)


def test_full_preflight_still_reports_design_checks_without_a_model(
    monkeypatch,
) -> None:
    wizard = wizard_mod.BatchProcessingWizard.__new__(
        wizard_mod.BatchProcessingWizard
    )
    wizard.model_path_var = _FakeVar("")
    wizard.use_existing_labels_var = _FakeVar(False)
    wizard.model_capabilities = wizard_mod.BatchModelCapabilities()
    wizard.analysis_preflight_var = _FakeVar("")
    wizard.progress_text_var = _FakeVar("")
    wizard._configured_keypoint_names = Mock(return_value=[])
    wizard._update_model_capability_text = Mock()
    wizard._build_analysis_preflight_rows = Mock(
        return_value=[
            {
                "analysis": "Missing subject ID",
                "will_run": "Fix",
                "scope": "video-1",
                "variables": "Subject=-",
                "reason": "Assign a Subject ID.",
            }
        ]
    )
    wizard._show_analysis_preflight_dialog = Mock()
    wizard._mark_session_dirty = Mock()
    warning = Mock()
    monkeypatch.setattr(wizard_mod.messagebox, "showwarning", warning)

    wizard._preflight_model()

    warning.assert_not_called()
    wizard._show_analysis_preflight_dialog.assert_called_once()
    assert "Fix=1" in wizard.analysis_preflight_var.get()
    assert "No model path is selected" in wizard.model_capabilities.warnings[0]


def test_batch_session_defaults_bbox_only_roi_entry_mode() -> None:
    restored = BatchSession.from_dict({})

    assert restored.roi_event_mode == "bbox_only"
    assert restored.roi_max_gap_frames == 5
    assert restored.roi_min_dwell_frames == 3
    assert restored.temporal_threshold_unit == "frames"
    assert restored.behavior_bout_class_mode == "mutually_exclusive"


def test_new_batch_sessions_default_to_seconds_without_changing_legacy_sessions() -> None:
    created = BatchSession.create()
    restored_new = BatchSession.from_dict(created.to_dict())
    restored_legacy = BatchSession.from_dict(
        {"min_bout_frames": 7, "max_gap_frames": 2}
    )

    assert created.temporal_threshold_unit == "seconds"
    assert restored_new.temporal_threshold_unit == "seconds"
    assert restored_new.min_bout_seconds == 0.10
    assert restored_legacy.temporal_threshold_unit == "frames"
    assert restored_legacy.min_bout_frames == 7


def test_batch_session_preserves_zero_gap_settings() -> None:
    restored = BatchSession.from_dict(
        {
            "max_gap_frames": 0,
            "roi_max_gap_frames": 0,
            "roi_entry_threshold": 0,
            "roi_exit_threshold": 0,
            "keypoint_entry_ratio_threshold": 0,
        }
    )

    assert restored.max_gap_frames == 0
    assert restored.roi_max_gap_frames == 0
    assert restored.roi_entry_threshold == 0
    assert restored.roi_exit_threshold == 0
    assert restored.keypoint_entry_ratio_threshold == 0


def test_batch_session_round_trips_multi_label_behavior_bouts() -> None:
    session = BatchSession.create()
    session.behavior_bout_class_mode = "multi_label"

    restored = BatchSession.from_dict(session.to_dict())

    assert restored.behavior_bout_class_mode == "multi_label"


def test_temporal_unit_switch_converts_minima_and_gap_limits() -> None:
    wizard = wizard_mod.BatchProcessingWizard.__new__(
        wizard_mod.BatchProcessingWizard
    )
    wizard._last_temporal_threshold_unit = "seconds"
    wizard.temporal_threshold_unit_var = _FakeVar("frames")
    wizard.video_fps_var = _FakeVar("30")
    wizard.max_gap_frames_var = _FakeVar("0.17")
    wizard.min_bout_frames_var = _FakeVar("0.10")
    wizard.roi_max_gap_frames_var = _FakeVar("0.05")
    wizard.roi_min_dwell_frames_var = _FakeVar("0.20")
    wizard.progress_text_var = _FakeVar("")

    wizard._on_temporal_threshold_unit_changed()

    assert wizard.max_gap_frames_var.get() == "5"
    assert wizard.min_bout_frames_var.get() == "3"
    assert wizard.roi_max_gap_frames_var.get() == "1"
    assert wizard.roi_min_dwell_frames_var.get() == "6"


def test_results_status_banner_distinguishes_draft_and_finalized(
    tmp_path,
) -> None:
    wizard = wizard_mod.BatchProcessingWizard.__new__(
        wizard_mod.BatchProcessingWizard
    )
    wizard.output_path_var = _FakeVar(str(tmp_path))
    wizard.results_status_var = _FakeVar("")
    wizard.service = SimpleNamespace(is_running=lambda: False)
    status_path = tmp_path / "batch_results_status.json"
    status_path.write_text(
        '{"status":"draft","source_kind":"automatic_pre_review_outputs"}',
        encoding="utf-8",
    )

    wizard._refresh_results_status()
    assert wizard.results_status_var.get().startswith("DRAFT:")

    status_path.write_text(
        '{"status":"finalized","source_kind":"authoritative_reviewed_outputs"}',
        encoding="utf-8",
    )
    wizard._refresh_results_status()
    assert wizard.results_status_var.get().startswith("FINALIZED:")


def test_session_keypoint_schema_prefers_validated_model_names_over_gui_defaults() -> None:
    wizard = wizard_mod.BatchProcessingWizard.__new__(wizard_mod.BatchProcessingWizard)
    wizard.app = SimpleNamespace(
        config=SimpleNamespace(
            setup=SimpleNamespace(keypoint_names_str=_FakeVar("Nose,Left_Ear,Right_Ear,TailBase"))
        ),
        log_message=Mock(),
    )
    wizard.use_existing_labels_var = _FakeVar(False)
    capabilities = wizard_mod.BatchModelCapabilities(
        has_keypoints=True,
        keypoint_count=3,
        keypoint_names=["Nose", "Thorax", "TailBase"],
        keypoint_names_source="dataset_yaml",
        keypoint_schema_path="C:/models/dataset.yaml",
    )

    names, source, schema_path = wizard._resolve_session_keypoint_schema(capabilities)

    assert names == ["Nose", "Thorax", "TailBase"]
    assert source == "dataset_yaml"
    assert schema_path == "C:/models/dataset.yaml"
    wizard.app.log_message.assert_not_called()


def test_session_keypoint_schema_ignores_gui_names_for_detection_model() -> None:
    wizard = wizard_mod.BatchProcessingWizard.__new__(wizard_mod.BatchProcessingWizard)
    wizard.app = SimpleNamespace(
        config=SimpleNamespace(
            setup=SimpleNamespace(keypoint_names_str=_FakeVar("Nose,Thorax,TailBase"))
        ),
        log_message=Mock(),
    )
    wizard.use_existing_labels_var = _FakeVar(False)
    capabilities = wizard_mod.BatchModelCapabilities(
        task="detect",
        has_keypoints=False,
        keypoint_count=0,
    )

    assert wizard._resolve_session_keypoint_schema(capabilities) == (
        [],
        "not_applicable",
        "",
    )


def test_session_keypoint_schema_uses_generic_names_instead_of_truncating_gui_defaults() -> None:
    wizard = wizard_mod.BatchProcessingWizard.__new__(wizard_mod.BatchProcessingWizard)
    wizard.app = SimpleNamespace(
        config=SimpleNamespace(
            setup=SimpleNamespace(keypoint_names_str=_FakeVar("Nose,Left_Ear,Right_Ear,TailBase"))
        ),
        log_message=Mock(),
    )
    wizard.use_existing_labels_var = _FakeVar(False)
    capabilities = wizard_mod.BatchModelCapabilities(has_keypoints=True, keypoint_count=3)

    names, source, schema_path = wizard._resolve_session_keypoint_schema(capabilities)

    assert names == ["kp0", "kp1", "kp2"]
    assert source == "generated_generic"
    assert schema_path == ""
    wizard.app.log_message.assert_called_once()


def test_roi_entry_mode_toggle_maps_to_clear_backend_modes() -> None:
    assert wizard_mod.BatchProcessingWizard._roi_event_mode_from_toggle(False) == "bbox_only"
    assert wizard_mod.BatchProcessingWizard._roi_event_mode_from_toggle(True) == "keypoint_index"
    assert wizard_mod.BatchProcessingWizard._uses_keypoint_roi_entry("bbox_only") is False
    assert wizard_mod.BatchProcessingWizard._uses_keypoint_roi_entry("tab6_hybrid") is False
    assert wizard_mod.BatchProcessingWizard._uses_keypoint_roi_entry("keypoint_index") is True
    assert wizard_mod.BatchProcessingWizard._uses_keypoint_roi_entry("keypoint_ratio") is True


def test_object_required_metrics_enable_batch_object_toggle() -> None:
    wizard = wizard_mod.BatchProcessingWizard.__new__(wizard_mod.BatchProcessingWizard)
    analytics_cfg = SimpleNamespace()
    for spec in wizard_mod.BatchProcessingWizard.METRIC_SPECS:
        setattr(analytics_cfg, spec.var_attr, _FakeVar(spec.key == "object_transition_analysis"))
    wizard.app = SimpleNamespace(config=SimpleNamespace(analytics=analytics_cfg))
    wizard.METRIC_SPECS = wizard_mod.BatchProcessingWizard.METRIC_SPECS
    wizard.object_interaction_enabled_var = _FakeVar(False)
    wizard._refresh_roi_summary = Mock()

    wizard._ensure_object_interaction_toggle_for_selected_metrics()

    assert wizard.object_interaction_enabled_var.get() is True
    wizard._refresh_roi_summary.assert_called_once()


def test_refresh_item_review_status_marks_completed_when_bout_and_roi_reviews_done(tmp_path) -> None:
    wizard = wizard_mod.BatchProcessingWizard.__new__(wizard_mod.BatchProcessingWizard)
    item = _make_item("video-1", "video_one")
    item.analytics_output_dir = str(tmp_path)
    detailed_csv = tmp_path / "video_one_detailed_bouts.csv"
    pd.DataFrame(
        [{"Track ID": 0, "Behavior": "Standing", "Start Frame": 1, "End Frame": 5}]
    ).to_csv(detailed_csv, index=False)
    item.detailed_bouts_csv = str(detailed_csv)
    pd.DataFrame(
        [{"Source": "zone", "Event Type": "entry", "Target Name": "Center", "Track ID": 0, "Frame": 12}]
    ).to_csv(tmp_path / "video_one_roi_events.csv", index=False)
    pd.DataFrame(
        [{"Behavior": "Walking", "status": "corrected", "Original Behavior": "Standing", "Corrected Behavior": "Walking", "Review Status": "corrected", "Corrected Manually": True}]
    ).to_csv(tmp_path / "video_one_reviewed_bouts.csv", index=False)
    pd.DataFrame(
        [{"Event Type": "entry", "Target Name": "Center", "Frame": 12, "Review Status": "confirmed"}]
    ).to_csv(tmp_path / "video_one_roi_event_review.csv", index=False)
    (tmp_path / "video_one_reviewed_roi_validation.json").write_text(
        '{"status": "valid"}',
        encoding="utf-8",
    )

    wizard._refresh_item_review_status(item)

    assert item.review_status == "completed"
    assert item.bout_review_status == "completed"
    assert item.roi_review_status == "completed"


def test_refresh_item_review_status_requires_materialized_roi_review(tmp_path) -> None:
    wizard = wizard_mod.BatchProcessingWizard.__new__(wizard_mod.BatchProcessingWizard)
    item = _make_item("video-1", "video_one")
    item.analytics_output_dir = str(tmp_path)
    pd.DataFrame(
        [{"Source": "zone", "Event Type": "entry", "Target Name": "Center", "Track ID": 0, "Frame": 12}]
    ).to_csv(tmp_path / "video_one_roi_events.csv", index=False)
    pd.DataFrame(
        [{"Event Type": "entry", "Target Name": "Center", "Frame": 12, "Review Status": "confirmed"}]
    ).to_csv(tmp_path / "video_one_roi_event_review.csv", index=False)

    wizard._refresh_item_review_status(item)

    assert item.review_status == "in_progress"
    assert item.roi_review_status == "in_progress"


def test_refresh_item_review_status_keeps_queued_when_roi_review_missing(tmp_path) -> None:
    wizard = wizard_mod.BatchProcessingWizard.__new__(wizard_mod.BatchProcessingWizard)
    item = _make_item("video-1", "video_one")
    item.analytics_output_dir = str(tmp_path)
    pd.DataFrame(
        [{"Source": "zone", "Event Type": "entry", "Target Name": "Center", "Track ID": 0, "Frame": 12}]
    ).to_csv(tmp_path / "video_one_roi_events.csv", index=False)

    wizard._refresh_item_review_status(item)

    assert item.review_status == "queued"
    assert item.bout_review_status == "not_required"
    assert item.roi_review_status == "queued"


def test_refresh_item_review_status_uses_manifest_relative_behavior_source(
    tmp_path,
) -> None:
    wizard = wizard_mod.BatchProcessingWizard.__new__(
        wizard_mod.BatchProcessingWizard
    )
    item = _make_item("video-1", "video_one")
    item.analytics_output_dir = str(tmp_path)
    item.detailed_bouts_csv = "Z:/old_computer/video_one_detailed_bouts.csv"
    pd.DataFrame(
        [
            {
                "Track ID": 0,
                "Class ID": 1,
                "Behavior": "rear",
                "Start Frame": 1,
                "End Frame": 5,
            }
        ]
    ).to_csv(tmp_path / "video_one_detailed_bouts.csv", index=False)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "video": {"base_name": "video_one"},
                "outputs": {
                    "raw_detailed_bouts_csv": (
                        "video_one_detailed_bouts.csv"
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    status_dir = tmp_path / "bout_review_workspace"
    status_dir.mkdir()
    (status_dir / "last_review_status.json").write_text(
        json.dumps(
            {
                "behavior": "complete",
                "spatial": "not_applicable",
            }
        ),
        encoding="utf-8",
    )

    wizard._refresh_item_review_status(item)

    assert item.bout_review_status == "completed"
    assert item.roi_review_status == "not_required"
    assert item.review_status == "completed"


def test_behavior_review_prefers_manifest_before_stale_queue_csv() -> None:
    wizard = wizard_mod.BatchProcessingWizard.__new__(
        wizard_mod.BatchProcessingWizard
    )
    wizard._launch_integrated_review_for_item = Mock(return_value=True)
    item = _make_item("video-1", "video_one")
    item.detailed_bouts_csv = "Z:/old_computer/missing.csv"

    opened = wizard._open_video_review_for_item(item)

    assert opened is True
    wizard._launch_integrated_review_for_item.assert_called_once()


def test_batch_session_item_backfills_split_review_status_from_legacy_field() -> None:
    item = BatchVideoItem.from_dict(
        {
            "video_id": "video-1",
            "video_name": "video_one",
            "video_path": "C:/videos/video_one.mp4",
            "review_status": "completed",
        }
    )

    assert item.review_status == "completed"
    assert item.bout_review_status == "completed"
    assert item.roi_review_status == "completed"


def test_refresh_queue_tree_shows_separate_arena_and_object_statuses_for_per_video_mode() -> None:
    items = [
        _make_item("video-1", "video_one"),
        _make_item("video-2", "video_two"),
    ]
    items[0].rois = {"Center": {"polygon": [(0, 0), (1, 0), (1, 1)]}}
    items[0].object_rois = {"Cup_A": {"polygon": [(2, 2), (3, 2), (3, 3)]}}
    wizard = _make_wizard(items=items)

    wizard._refresh_queue_tree()

    first_row = wizard.queue_tree.rows[0]["values"]
    second_row = wizard.queue_tree.rows[1]["values"]
    assert first_row[2] == "included"
    assert first_row[6] == "done"
    assert first_row[7] == "done"
    assert second_row[6] == "pending"
    assert second_row[7] == "pending"


def test_refresh_queue_tree_shows_shared_assignments_for_single_strategy() -> None:
    items = [
        _make_item("video-1", "video_one"),
        _make_item("video-2", "video_two"),
    ]
    wizard = _make_wizard(items=items)
    wizard.roi_strategy_var.set("single")
    wizard.shared_rois = {"Center": {"polygon": [(0, 0), (1, 0), (1, 1)]}}
    wizard.shared_object_rois = {"Cup_A": {"polygon": [(2, 2), (3, 2), (3, 3)]}}

    wizard._refresh_queue_tree()

    for row in wizard.queue_tree.rows:
        values = row["values"]
        assert values[6] == "shared"
        assert values[7] == "shared"


def test_single_arena_strategy_shows_per_video_objects_as_done_without_shared_objects() -> None:
    items = [
        _make_item("video-1", "video_one"),
        _make_item("video-2", "video_two"),
    ]
    items[0].object_rois = {
        "Cup_A": {"polygons": [[(2, 2), (3, 2), (3, 3)]]}
    }
    wizard = _make_wizard(items=items)
    wizard.roi_strategy_var.set("single")

    wizard._refresh_queue_tree()

    assert wizard.queue_tree.rows[0]["values"][7] == "done"
    assert wizard.queue_tree.rows[1]["values"][7] == "pending"


def test_refresh_queue_tree_marks_excluded_rows() -> None:
    items = [
        _make_item("video-1", "video_one", excluded=True),
        _make_item("video-2", "video_two"),
    ]
    wizard = _make_wizard(items=items)

    wizard._refresh_queue_tree()

    assert wizard.queue_tree.rows[0]["values"][2] == "excluded"
    assert wizard.queue_tree.rows[0]["tags"] == ("excluded",)
    assert wizard.queue_tree.rows[1]["values"][2] == "included"


def test_refresh_queue_tree_shows_override_status_when_single_strategy_uses_per_video_override() -> None:
    items = [
        _make_item("video-1", "video_one"),
        _make_item("video-2", "video_two"),
    ]
    wizard = _make_wizard(items=items)
    wizard.roi_strategy_var.set("single")
    wizard.shared_rois = {"Center": {"polygons": [[(0, 0), (1, 0), (1, 1)]]}}
    items[0].rois = {"Center": {"polygons": [[(10, 10), (11, 10), (11, 11)]]}}

    wizard._refresh_queue_tree()

    assert wizard.queue_tree.rows[0]["values"][6] == "override"
    assert wizard.queue_tree.rows[1]["values"][6] == "shared"


def test_batch_session_to_dict_keeps_assay_shortlist_default() -> None:
    session = BatchSession.create()

    payload = session.to_dict()

    assert payload["figure_export_mode"] == "assay_shortlist"

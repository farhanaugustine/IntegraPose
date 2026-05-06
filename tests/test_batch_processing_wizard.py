from unittest.mock import Mock
from types import SimpleNamespace
from pathlib import Path
import importlib.util
import sys
import pandas as pd

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


def test_edit_metadata_requires_exactly_one_selected_video(monkeypatch) -> None:
    warnings = []
    items = [
        _make_item("video-1", "video_one", subject_id="MouseA", time_point="Baseline"),
        _make_item("video-2", "video_two", subject_id="MouseB", time_point="Baseline"),
    ]
    wizard = _make_wizard("video-1", "video-2", items=items)
    wizard._prompt_roi_name = Mock()
    monkeypatch.setattr(
        wizard_mod.messagebox,
        "showwarning",
        lambda title, message, parent=None: warnings.append((title, message)),
    )

    wizard._edit_metadata_for_selected()

    assert warnings == [("Batch Queue", "Select exactly one video row for editing subject and time metadata.")]
    wizard._prompt_roi_name.assert_not_called()


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


def test_batch_session_round_trips_guided_name_templates() -> None:
    session = BatchSession.create()
    session.roi_name_templates = ["Center", "Edge"]
    session.object_name_templates = ["Cup_A", "Cup_B"]
    session.tracker_config_path = "bytetrack.yaml"
    session.figure_export_mode = "assay_shortlist"
    session.generate_video_quicklooks = False
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
    assert restored.roi_max_gap_frames == 17
    assert restored.roi_min_dwell_frames == 9
    assert restored.videos[0].excluded is True


def test_batch_session_defaults_bbox_only_roi_entry_mode() -> None:
    restored = BatchSession.from_dict({})

    assert restored.roi_event_mode == "bbox_only"
    assert restored.roi_max_gap_frames == 5
    assert restored.roi_min_dwell_frames == 3


def test_batch_session_preserves_zero_gap_settings() -> None:
    restored = BatchSession.from_dict({"max_gap_frames": 0, "roi_max_gap_frames": 0})

    assert restored.max_gap_frames == 0
    assert restored.roi_max_gap_frames == 0


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
    pd.DataFrame([{"Behavior": "Walking", "Start Frame": 1, "End Frame": 5}]).to_csv(detailed_csv, index=False)
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

    wizard._refresh_item_review_status(item)

    assert item.review_status == "completed"
    assert item.bout_review_status == "completed"
    assert item.roi_review_status == "completed"


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

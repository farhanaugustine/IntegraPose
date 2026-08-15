from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from integra_pose.logic import batch_pipeline as batch_pipeline_mod
from integra_pose.logic.batch_pipeline import BatchPipeline
from integra_pose.utils.batch_session import BatchSession, BatchVideoItem
from integra_pose.utils.operation_result import OperationStatus


class _FakeApp:
    def __init__(self) -> None:
        self.config = SimpleNamespace()
        self.logs: list[tuple[str, str]] = []

    def log_message(self, message: str, level: str = "INFO") -> None:
        self.logs.append((level, message))


def _completed_item(root: Path, video_id: str) -> BatchVideoItem:
    video_stem = f"video_{video_id}"
    run_dir = root / "videos" / video_id / "inference" / "infer"
    labels_dir = run_dir / "labels"
    analytics_dir = root / "videos" / video_id / "analytics"
    labels_dir.mkdir(parents=True)
    analytics_dir.mkdir(parents=True)

    detailed = analytics_dir / f"{video_stem}_detailed_bouts.csv"
    summary = analytics_dir / f"{video_stem}_summary.csv"
    labels = labels_dir / "labels.csv"
    metrics = run_dir / "metrics.csv"
    detailed.write_text("Behavior,Start Frame,End Frame\nrest,0,2\n", encoding="utf-8")
    summary.write_text("Behavior,Bout Count\nrest,1\n", encoding="utf-8")
    labels.write_text("frame,class_id\n0,0\n", encoding="utf-8")
    metrics.write_text("frame,object_id\n0,0\n", encoding="utf-8")
    manifest = {
        "outputs": {
            "detailed_bouts_csv": str(detailed),
            "summary_csv": str(summary),
            "roi_metrics_files": {},
            "object_interaction_files": {},
            "modules": {},
        }
    }
    (analytics_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    return BatchVideoItem(
        video_id=video_id,
        video_name=f"{video_stem}.mp4",
        video_path=str(root / f"{video_stem}.mp4"),
        inference_status="completed",
        analytics_status="completed",
        run_output_dir=str(run_dir),
        yolo_output_dir=str(labels_dir),
        analytics_output_dir=str(analytics_dir),
        detailed_bouts_csv=str(detailed),
        summary_bouts_csv=str(summary),
        metrics_csv=str(metrics),
    )


def _patch_batch_exports(monkeypatch, captured: list[list[dict]]) -> None:
    monkeypatch.setattr("integra_pose.utils.seeds.apply_global_seed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(batch_pipeline_mod, "check_fps_consistency", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        batch_pipeline_mod,
        "collect_batch_frames",
        lambda rows: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
    )

    def _summary(rows):
        captured.append(list(rows))
        return pd.DataFrame(
            [{"video_id": row["video_id"], "group": row.get("group", "")} for row in rows]
        )

    monkeypatch.setattr(batch_pipeline_mod, "build_video_summary", _summary)
    monkeypatch.setattr(batch_pipeline_mod, "build_analysis_coverage", lambda _rows: pd.DataFrame())
    monkeypatch.setattr(
        batch_pipeline_mod,
        "collect_batch_module_tables",
        lambda _rows: SimpleNamespace(tables={}, file_index_df=pd.DataFrame()),
    )
    monkeypatch.setattr(
        batch_pipeline_mod,
        "export_batch_module_tables",
        lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()),
    )

    def _workbook(*, workbook_path, **_kwargs):
        Path(workbook_path).write_text("workbook", encoding="utf-8")
        return str(workbook_path)

    monkeypatch.setattr(batch_pipeline_mod, "write_batch_workbook", _workbook)
    monkeypatch.setattr(
        "integra_pose.logic.group_stats.export_group_stats_bundle",
        lambda *_args, **_kwargs: (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {},
        ),
    )


def _session(root: Path, items: list[BatchVideoItem]) -> BatchSession:
    session = BatchSession.create()
    session.output_path = str(root)
    session.videos = items
    session.export_publication_figures = False
    session.export_batch_dashboard = False
    session.export_group_stats_overview = False
    session.export_individual_profiles = False
    session.export_module_archive = False
    session.generate_video_quicklooks = False
    return session


def test_resume_rehydrates_all_completed_videos_before_aggregate_export(tmp_path, monkeypatch) -> None:
    items = [_completed_item(tmp_path, "one"), _completed_item(tmp_path, "two")]
    reviewed = Path(items[0].analytics_output_dir) / "video_one_reviewed_bouts.csv"
    reviewed.write_text("Behavior,Start Frame,End Frame\nrest,0,2\n", encoding="utf-8")
    manifest_path = Path(items[0].analytics_output_dir) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["reviewed_bouts_csv"] = str(reviewed)
    manifest["notes"] = {"bout_review": {"status": "complete"}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    captured: list[list[dict]] = []
    _patch_batch_exports(monkeypatch, captured)

    result = BatchPipeline(_FakeApp()).run(
        _session(tmp_path, items),
        stop_event=threading.Event(),
        resume=True,
    )

    assert result.status is OperationStatus.SUCCESS
    assert result.completed_count == 2
    assert result.resumed_count == 2
    assert {row["video_id"] for row in result.video_results or []} == {"one", "two"}
    restored_one = next(row for row in result.video_results or [] if row["video_id"] == "one")
    assert restored_one["reviewed_bouts_csv"] == str(reviewed.resolve())
    assert len(captured) == 1
    assert {row["video_id"] for row in captured[0]} == {"one", "two"}


def test_resume_reprocesses_instead_of_skipping_when_required_artifact_is_missing(tmp_path, monkeypatch) -> None:
    item = _completed_item(tmp_path, "one")
    Path(item.summary_bouts_csv).unlink()
    captured: list[list[dict]] = []
    _patch_batch_exports(monkeypatch, captured)
    pipeline = BatchPipeline(_FakeApp())
    attempted: list[str] = []
    monkeypatch.setattr(pipeline, "_build_inference_settings", lambda *_args, **_kwargs: object())

    def _fail_inference(*_args, **_kwargs):
        attempted.append(item.video_id)
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(pipeline, "_run_native_inference", _fail_inference)

    result = pipeline.run(
        _session(tmp_path, [item]),
        stop_event=threading.Event(),
        resume=True,
    )

    assert attempted == ["one"]
    assert result.status is OperationStatus.FAILED
    assert result.resumed_count == 0
    assert result.completed_count == 0
    assert result.failed_count == 1
    assert result.video_results == []


def test_no_included_videos_is_an_explicit_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("integra_pose.utils.seeds.apply_global_seed", lambda *_args, **_kwargs: None)
    result = BatchPipeline(_FakeApp()).run(
        _session(tmp_path, []),
        stop_event=threading.Event(),
    )

    assert result.status is OperationStatus.FAILED
    assert result.total_count == 0
    assert result.cancelled is False


def test_runtime_device_provenance_uses_predictor_device_not_auto_sentinel() -> None:
    model = SimpleNamespace(
        predictor=SimpleNamespace(device="cuda:1"),
        model=SimpleNamespace(device="cuda:0"),
    )

    assert BatchPipeline._resolve_runtime_device(model, "-1") == "cuda:1"
    assert BatchPipeline._resolve_runtime_device(SimpleNamespace(), "-1") == "unknown"
    assert BatchPipeline._resolve_runtime_device(SimpleNamespace(), "cpu") == "cpu"


def test_finalize_refuses_nonempty_unreviewed_bouts(tmp_path) -> None:
    item = _completed_item(tmp_path, "one")
    session = _session(tmp_path, [item])
    session.review_policy = "after_all"

    result = BatchPipeline(_FakeApp()).finalize_reviewed_results(
        session, stop_event=threading.Event()
    )

    assert result.status is OperationStatus.FAILED
    assert "bout review is not finalized" in result.message
    marker = json.loads(
        (tmp_path / "batch_results_status.json").read_text(encoding="utf-8")
    )
    assert marker["status"] == "needs_rebuild"


def test_finalize_rebuilds_from_reviewed_bouts_and_roi_outputs(
    tmp_path, monkeypatch
) -> None:
    item = _completed_item(tmp_path, "one")
    analytics_dir = Path(item.analytics_output_dir)
    manifest_path = analytics_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_roi_events = analytics_dir / "video_one_roi_events.csv"
    reviewed_roi_events = analytics_dir / "video_one_reviewed_roi_events.csv"
    reviewed_roi_overview = analytics_dir / "video_one_reviewed_roi_overview.csv"
    reviewed_roi_dwell = analytics_dir / "video_one_reviewed_roi_dwell_events.csv"
    reviewed_roi_validation = analytics_dir / "video_one_reviewed_roi_validation.json"
    reviewed_bouts = analytics_dir / "video_one_reviewed_bouts.csv"
    reviewed_bout_summary = analytics_dir / "video_one_reviewed_bouts_summary.csv"
    raw_roi_events.write_text(
        "Event ID,Event Type,ROI Name,Track ID,Frame\n"
        "e1,entry,Center,0,0\n"
        "e2,exit,Center,0,4\n",
        encoding="utf-8",
    )
    reviewed_roi_events.write_text(raw_roi_events.read_text(encoding="utf-8"), encoding="utf-8")
    reviewed_roi_overview.write_text(
        "ROI Name,Entries,Exits,Qualified Dwell Time (s)\nCenter,1,1,0.5\n",
        encoding="utf-8",
    )
    reviewed_roi_dwell.write_text(
        "ROI Name,Track ID,Start Frame,End Frame,Duration (s)\nCenter,0,0,4,0.5\n",
        encoding="utf-8",
    )
    reviewed_roi_validation.write_text(
        json.dumps({"status": "valid"}), encoding="utf-8"
    )
    reviewed_bouts.write_text(
        "Behavior,Start Frame,End Frame,Duration (s)\nrest,0,2,0.3\n",
        encoding="utf-8",
    )
    reviewed_bout_summary.write_text(
        "Behavior,Bout Count\nrest,1\n", encoding="utf-8"
    )
    manifest["outputs"].update(
        {
            "raw_detailed_bouts_csv": item.detailed_bouts_csv,
            "raw_summary_csv": item.summary_bouts_csv,
            "reviewed_bouts_csv": str(reviewed_bouts),
            "reviewed_bouts_summary_csv": str(reviewed_bout_summary),
            "raw_roi_events_csv": str(raw_roi_events),
            "reviewed_roi_events_csv": str(reviewed_roi_events),
            "reviewed_roi_overview_csv": str(reviewed_roi_overview),
            "reviewed_roi_dwell_events_csv": str(reviewed_roi_dwell),
            "reviewed_roi_validation_json": str(reviewed_roi_validation),
            "roi_events_csv": str(reviewed_roi_events),
            "roi_metrics_files": {
                "exclusive_entries_exits": str(reviewed_roi_overview),
                "exclusive_dwell_events": str(reviewed_roi_dwell),
            },
        }
    )
    manifest["notes"] = {
        "bout_review": {"status": "complete"},
        "roi_review": {"status": "complete"},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    captured: list[list[dict]] = []
    _patch_batch_exports(monkeypatch, captured)
    session = _session(tmp_path, [item])
    session.review_policy = "after_all"

    result = BatchPipeline(_FakeApp()).finalize_reviewed_results(
        session, stop_event=threading.Event()
    )

    assert result.status is OperationStatus.SUCCESS
    assert len(captured) == 1
    restored = captured[0][0]
    assert restored["reviewed_bouts_csv"] == str(reviewed_bouts.resolve())
    assert restored["roi_events_csv"] == str(reviewed_roi_events.resolve())
    assert restored["roi_overview_csv"] == str(reviewed_roi_overview.resolve())
    assert restored["roi_dwell_events_csv"] == str(reviewed_roi_dwell.resolve())
    marker = json.loads(
        (tmp_path / "batch_results_status.json").read_text(encoding="utf-8")
    )
    assert marker["status"] == "finalized"
    assert marker["source_kind"] == "authoritative_reviewed_outputs"


def test_finalize_writes_real_workbook_coverage_and_status_bundle(tmp_path) -> None:
    item = _completed_item(tmp_path, "one")
    session = _session(tmp_path, [item])
    session.review_policy = "skip"

    result = BatchPipeline(_FakeApp()).finalize_reviewed_results(
        session, stop_event=threading.Event()
    )

    assert result.status is OperationStatus.SUCCESS
    assert Path(result.workbook_path).is_file()
    assert (tmp_path / "analysis_coverage_table.csv").is_file()
    assert (tmp_path / "module_tables" / "module_file_index.csv").is_file()
    marker = json.loads(
        (tmp_path / "batch_results_status.json").read_text(encoding="utf-8")
    )
    assert marker["status"] == "finalized"
    assert marker["source_kind"] == "automatic_outputs_review_explicitly_skipped"

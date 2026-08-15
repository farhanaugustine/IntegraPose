from pathlib import Path

import pandas as pd

from integra_pose.utils.bout_review import (
    BoutReviewPaths,
    IncompleteBoutReviewError,
    append_review_decision,
    build_review_workspace,
    ethogram_window,
    inclusive_duration_frames,
    materialize_reviewed_bouts,
    normalize_detected_bouts,
    normalize_review_decisions,
    register_authoritative_review_in_manifest,
    save_review_bundle,
)
from integra_pose.utils.operation_result import OperationStatus
import integra_pose.utils.bout_review as bout_review_mod


def _raw_bouts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Run ID": "run-1",
                "Bout ID": "bout-a",
                "Track ID": 1,
                "Behavior": "walk",
                "Start Frame": 10,
                "End Frame": 10,
                "Detection Max Gap (Frames)": 5,
                "Detection Min Bout (Frames)": 3,
            },
            {
                "Run ID": "run-1",
                "Bout ID": "bout-b",
                "Track ID": 1,
                "Behavior": "groom",
                "Start Frame": 12,
                "End Frame": 20,
                "Detection Max Gap (Frames)": 5,
                "Detection Min Bout (Frames)": 3,
            },
            {
                "Run ID": "run-1",
                "Bout ID": "bout-c",
                "Track ID": 2,
                "Behavior": "rear",
                "Start Frame": 17,
                "End Frame": 25,
                "Detection Max Gap (Frames)": 5,
                "Detection Min Bout (Frames)": 3,
            },
        ]
    )


def test_inclusive_contract_accepts_single_frame_bout() -> None:
    assert inclusive_duration_frames(10, 10) == 1
    normalized = normalize_detected_bouts(_raw_bouts(), source_video="video.mp4", fps=20.0)
    assert normalized.loc[0, "Duration (Frames)"] == 1
    assert normalized.loc[0, "Duration (s)"] == 0.05


def test_review_decisions_do_not_mutate_raw_or_reconstruct_bouts() -> None:
    source = _raw_bouts()
    raw = normalize_detected_bouts(source, source_video="video.mp4", fps=10.0)
    decisions = append_review_decision(None, raw.iloc[0], "corrected", corrected_behavior="pause")
    workspace = build_review_workspace(raw, decisions, fps=10.0)

    assert source.loc[0, "Behavior"] == "walk"
    assert raw.loc[0, "Behavior"] == "walk"
    assert workspace.loc[0, "Behavior"] == "pause"
    assert workspace.loc[0, "status"] == "corrected"
    assert workspace.loc[0, "Detection Max Gap (Frames)"] == 5
    assert workspace.loc[0, "Detection Min Bout (Frames)"] == 3


def test_decision_history_is_append_only_and_latest_decision_materializes() -> None:
    raw = normalize_detected_bouts(_raw_bouts(), source_video="video.mp4", fps=10.0)
    decisions = append_review_decision(None, raw.iloc[0], "confirmed")
    decisions = append_review_decision(
        decisions,
        raw.iloc[0],
        "corrected",
        corrected_behavior="pause",
    )

    workspace = build_review_workspace(raw, decisions, fps=10.0)

    assert len(decisions) == 2
    assert decisions["Decision Sequence"].tolist() == [1, 2]
    assert workspace.loc[0, "status"] == "corrected"
    assert workspace.loc[0, "Behavior"] == "pause"


def test_authoritative_materialization_requires_all_decisions_and_excludes_rejected() -> None:
    raw = normalize_detected_bouts(_raw_bouts(), source_video="video.mp4", fps=10.0)
    decisions = append_review_decision(None, raw.iloc[0], "confirmed")
    try:
        materialize_reviewed_bouts(raw, decisions, fps=10.0)
    except IncompleteBoutReviewError as exc:
        assert "2 detected bout(s)" in str(exc)
    else:
        raise AssertionError("Incomplete review should not produce an authoritative table")

    decisions = append_review_decision(decisions, raw.iloc[1], "rejected")
    decisions = append_review_decision(
        decisions,
        raw.iloc[2],
        "corrected",
        corrected_behavior="investigate",
        corrected_start_frame=18,
        corrected_end_frame=20,
    )
    reviewed = materialize_reviewed_bouts(raw, decisions, fps=10.0)

    assert reviewed["Bout ID"].tolist() == ["bout-a", "bout-c"]
    corrected = reviewed[reviewed["Bout ID"] == "bout-c"].iloc[0]
    assert corrected["Behavior"] == "investigate"
    assert corrected["Duration (Frames)"] == 3
    assert corrected["Duration (s)"] == 0.3


def test_corrected_bouts_must_respect_minimum_duration_and_not_overlap() -> None:
    raw = normalize_detected_bouts(_raw_bouts(), source_video="video.mp4", fps=10.0)
    decisions = append_review_decision(
        None,
        raw.iloc[0],
        "corrected",
        corrected_start_frame=10,
        corrected_end_frame=11,
    )
    decisions = append_review_decision(decisions, raw.iloc[1], "confirmed")
    decisions = append_review_decision(decisions, raw.iloc[2], "confirmed")

    try:
        materialize_reviewed_bouts(raw, decisions, fps=10.0)
    except ValueError as exc:
        assert "below its configured minimum" in str(exc)
    else:
        raise AssertionError("A corrected sub-threshold bout must be rejected")

    decisions = append_review_decision(
        decisions,
        raw.iloc[0],
        "corrected",
        corrected_start_frame=9,
        corrected_end_frame=13,
    )
    try:
        materialize_reviewed_bouts(raw, decisions, fps=10.0)
    except ValueError as exc:
        assert "overlap for track 1" in str(exc)
    else:
        raise AssertionError("Overlapping reviewed bouts must be rejected")


def test_save_bundle_is_partial_until_complete_then_writes_authoritative_outputs(tmp_path: Path) -> None:
    authoritative = tmp_path / "video_reviewed_bouts.csv"
    raw = normalize_detected_bouts(_raw_bouts(), source_video="video.mp4", fps=30.0)
    decisions = append_review_decision(None, raw.iloc[0], "confirmed")

    partial = save_review_bundle(raw, decisions, authoritative_path=authoritative, fps=30.0)
    paths = BoutReviewPaths.from_authoritative(authoritative)
    assert partial.status is OperationStatus.PARTIAL
    assert paths.raw_detected.is_file()
    assert paths.decisions.is_file()
    assert paths.workspace.is_file()
    assert not paths.authoritative.exists()

    decisions = append_review_decision(decisions, raw.iloc[1], "rejected")
    decisions = append_review_decision(decisions, raw.iloc[2], "confirmed")
    complete = save_review_bundle(raw, decisions, authoritative_path=authoritative, fps=30.0)

    assert complete.succeeded
    assert paths.authoritative.is_file()
    assert paths.summary.is_file()
    assert len(pd.read_csv(paths.authoritative)) == 2


def test_all_rejected_review_writes_readable_empty_authoritative_tables(tmp_path: Path) -> None:
    authoritative = tmp_path / "all_rejected_reviewed_bouts.csv"
    raw = normalize_detected_bouts(_raw_bouts(), source_video="video.mp4", fps=30.0)
    decisions = None
    for _, row in raw.iterrows():
        decisions = append_review_decision(decisions, row, "rejected")

    result = save_review_bundle(raw, decisions, authoritative_path=authoritative, fps=30.0)
    paths = BoutReviewPaths.from_authoritative(authoritative)

    assert result.succeeded
    reviewed = pd.read_csv(paths.authoritative)
    summary = pd.read_csv(paths.summary)
    assert reviewed.empty
    assert "Bout ID" in reviewed.columns
    assert summary.empty
    assert "Bout_Count" in summary.columns


def test_immutable_raw_snapshot_rejects_a_different_detection_set(tmp_path: Path) -> None:
    authoritative = tmp_path / "video_reviewed_bouts.csv"
    raw = normalize_detected_bouts(_raw_bouts(), source_video="video.mp4", fps=30.0)
    first = save_review_bundle(raw, None, authoritative_path=authoritative, fps=30.0)
    assert first.status is OperationStatus.PARTIAL

    changed = raw.copy()
    changed.loc[0, "End Frame"] = 11
    changed.loc[0, "Original End Frame"] = 11
    second = save_review_bundle(changed, None, authoritative_path=authoritative, fps=30.0)

    assert second.failed
    assert "immutable" in second.message.lower()


def test_bundle_commit_failure_rolls_back_existing_review_files(tmp_path: Path, monkeypatch) -> None:
    authoritative = tmp_path / "video_reviewed_bouts.csv"
    raw = normalize_detected_bouts(_raw_bouts(), source_video="video.mp4", fps=30.0)
    decisions = append_review_decision(None, raw.iloc[0], "confirmed")
    assert save_review_bundle(raw, decisions, authoritative_path=authoritative, fps=30.0).status is OperationStatus.PARTIAL
    paths = BoutReviewPaths.from_authoritative(authoritative)
    before_decisions = paths.decisions.read_bytes()
    before_workspace = paths.workspace.read_bytes()
    decisions = append_review_decision(decisions, raw.iloc[1], "rejected")

    real_replace = bout_review_mod.os.replace
    failed_once = {"value": False}

    def _replace_with_one_failure(source, destination):
        if Path(destination) == paths.workspace and not failed_once["value"]:
            failed_once["value"] = True
            raise OSError("simulated commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr(bout_review_mod.os, "replace", _replace_with_one_failure)
    result = save_review_bundle(raw, decisions, authoritative_path=authoritative, fps=30.0)

    assert result.failed
    assert paths.decisions.read_bytes() == before_decisions
    assert paths.workspace.read_bytes() == before_workspace


def test_orphan_and_ambiguous_decisions_are_rejected(tmp_path: Path) -> None:
    raw = normalize_detected_bouts(_raw_bouts(), source_video="video.mp4", fps=30.0)
    decisions = append_review_decision(None, raw.iloc[0], "confirmed")
    orphaned = decisions.copy()
    orphaned.loc[0, "Bout ID"] = "unknown-bout"
    orphan_result = save_review_bundle(
        raw,
        orphaned,
        authoritative_path=tmp_path / "orphan_reviewed_bouts.csv",
        fps=30.0,
    )
    assert orphan_result.failed
    assert "unknown Bout ID" in orphan_result.error

    duplicated = pd.concat([decisions, decisions], ignore_index=True)
    try:
        normalize_review_decisions(duplicated)
    except ValueError as exc:
        assert "Decision IDs must be unique" in str(exc)
    else:
        raise AssertionError("Duplicate decision identities must be rejected")

    duplicate_sequence = pd.concat(
        [decisions, append_review_decision(decisions, raw.iloc[1], "rejected").iloc[[-1]]],
        ignore_index=True,
    )
    duplicate_sequence.loc[1, "Decision Sequence"] = duplicate_sequence.loc[0, "Decision Sequence"]
    try:
        normalize_review_decisions(duplicate_sequence)
    except ValueError as exc:
        assert "Sequence values must be unique" in str(exc)
    else:
        raise AssertionError("Duplicate decision sequence values must be rejected")


def test_immutable_snapshot_detects_changed_roi_even_with_reused_bout_id(tmp_path: Path) -> None:
    authoritative = tmp_path / "roi_reviewed_bouts.csv"
    source = _raw_bouts()
    source["ROI Name"] = ["Left", "Center", "Right"]
    raw = normalize_detected_bouts(source, source_video="video.mp4", fps=30.0)
    assert save_review_bundle(raw, None, authoritative_path=authoritative, fps=30.0).status is OperationStatus.PARTIAL

    changed = raw.copy(deep=True)
    changed.loc[0, "ROI Name"] = "Right"
    result = save_review_bundle(changed, None, authoritative_path=authoritative, fps=30.0)

    assert result.failed
    assert "immutable" in result.message.lower()


def test_zero_bout_review_is_complete_and_writes_readable_empty_outputs(tmp_path: Path) -> None:
    authoritative = tmp_path / "empty_reviewed_bouts.csv"
    empty_raw = pd.DataFrame(columns=["Track ID", "Behavior", "Start Frame", "End Frame"])

    result = save_review_bundle(empty_raw, None, authoritative_path=authoritative, fps=30.0)
    paths = BoutReviewPaths.from_authoritative(authoritative)

    assert result.succeeded
    assert result.artifacts["review_not_required"] is True
    assert pd.read_csv(paths.authoritative).empty
    assert pd.read_csv(paths.summary).empty


def test_manifest_registration_invalidates_raw_bout_dependent_outputs() -> None:
    manifest = {
        "outputs": {
            "detailed_bouts_csv": "raw_detailed.csv",
            "summary_csv": "raw_summary.csv",
            "excel_summary": "raw_summary.xlsx",
            "modules": {
                "behavior_transitions": {"files": {"summary": "raw_transitions.csv"}},
                "latency_metrics": {"files": {"summary": "latency.csv"}},
            },
        }
    }
    artifacts = {
        "raw_detected_path": "raw_snapshot.csv",
        "decisions_path": "decisions.csv",
        "workspace_path": "workspace.csv",
        "authoritative_path": "reviewed.csv",
        "summary_path": "reviewed_summary.csv",
        "accepted_bout_count": 2,
        "rejected_bout_count": 1,
    }

    updated = register_authoritative_review_in_manifest(manifest, artifacts)

    assert updated["outputs"]["detailed_bouts_csv"] == "reviewed.csv"
    assert updated["outputs"]["raw_detailed_bouts_csv"] == "raw_detailed.csv"
    assert "behavior_transitions" not in updated["outputs"]["modules"]
    assert "latency_metrics" in updated["outputs"]["modules"]
    assert "behavior_transitions" in updated["outputs"]["invalidated_raw_bout_modules"]
    assert updated["notes"]["bout_review"]["invalidated_raw_bout_modules"] == ["behavior_transitions"]
    updated_again = register_authoritative_review_in_manifest(updated, artifacts)
    assert updated_again["notes"]["bout_review"]["invalidated_raw_bout_modules"] == ["behavior_transitions"]


def test_ethogram_window_includes_neighboring_and_overlapping_behaviors() -> None:
    raw = normalize_detected_bouts(_raw_bouts(), source_video="video.mp4", fps=30.0)
    start, end, visible = ethogram_window(
        raw,
        focus_bout_id="bout-b",
        context_frames=4,
        total_frames=100,
    )

    assert (start, end) == (8, 24)
    assert set(visible["Bout ID"]) == {"bout-a", "bout-b", "bout-c"}

from pathlib import Path

import pandas as pd

from integra_pose.gui.services.batch_processing_service import BatchProcessingService
from integra_pose.logic.batch_design import (
    apply_design_metadata_inference,
    infer_design_metadata,
    parse_time_point_numeric,
)
from integra_pose.logic.batch_pipeline import BatchPipeline
from integra_pose.utils.batch_session import BatchVideoItem


def _guess_for(path: Path, *, source_path: Path, cohort: list[Path] | None = None):
    paths = cohort or [path]
    guesses = infer_design_metadata(paths, source_path=source_path)
    return guesses[str(path.resolve()).casefold()]


def test_filename_metadata_is_inferred_with_provenance(tmp_path: Path) -> None:
    video = tmp_path / "Control_Mouse01_Day7.mp4"
    video.write_bytes(b"")

    guess = _guess_for(video, source_path=tmp_path)

    assert guess.group == "Control"
    assert guess.subject_id == "Mouse01"
    assert guess.time_point == "Day7"
    assert guess.sources == {
        "group": "filename",
        "subject_id": "filename",
        "time_point": "filename",
    }
    assert guess.warnings == ()


def test_batch_service_discovers_and_assigns_filename_metadata(
    tmp_path: Path,
) -> None:
    video = tmp_path / "Treatment_Mouse02_Day14.mp4"
    video.write_bytes(b"")
    service = object.__new__(BatchProcessingService)

    items = service.discover_queue(str(tmp_path))

    assert len(items) == 1
    assert items[0].group == "Treatment"
    assert items[0].subject_id == "Mouse02"
    assert items[0].time_point == "Day14"
    assert items[0].metadata_sources["group"] == "filename"


def test_selected_known_group_folder_is_used_as_group(tmp_path: Path) -> None:
    group_root = tmp_path / "Control"
    video = group_root / "Mouse03" / "Day0" / "recording.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")

    guess = _guess_for(video, source_path=group_root)

    assert guess.group == "Control"
    assert guess.sources["group"] == "source_folder:Control"
    assert guess.subject_id == "Mouse03"
    assert guess.time_point == "Day0"


def test_selected_subject_folder_is_not_lost_from_relative_path(
    tmp_path: Path,
) -> None:
    subject_root = tmp_path / "Mouse04"
    video = subject_root / "Day7" / "recording.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")

    guess = _guess_for(video, source_path=subject_root)

    assert guess.subject_id == "Mouse04"
    assert guess.sources["subject_id"] == "source_folder:Mouse04"
    assert guess.time_point == "Day7"


def test_cohort_folder_layout_assigns_group_subject_and_time(tmp_path: Path) -> None:
    first = tmp_path / "DrugA" / "Mouse01" / "Day0" / "recording.mp4"
    second = tmp_path / "DrugB" / "Mouse02" / "Day7" / "recording.mp4"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"")
    second.write_bytes(b"")

    guess = _guess_for(
        first,
        source_path=tmp_path,
        cohort=[first, second],
    )

    assert guess.group == "DrugA"
    assert guess.subject_id == "Mouse01"
    assert guess.time_point == "Day0"
    assert guess.sources["group"] == "folder:DrugA"
    assert guess.sources["subject_id"] == "folder:Mouse01"
    assert guess.sources["time_point"] == "folder:Day0"


def test_conflicting_group_candidates_stay_blank_and_are_flagged(
    tmp_path: Path,
) -> None:
    first = tmp_path / "DrugA" / "Control_Mouse01_Day0.mp4"
    second = tmp_path / "DrugB" / "Mouse02_Day0.mp4"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"")
    second.write_bytes(b"")

    guess = _guess_for(
        first,
        source_path=tmp_path,
        cohort=[first, second],
    )

    assert guess.group == ""
    assert any("Conflicting group candidates" in warning for warning in guess.warnings)


def test_inference_preserves_manual_values_and_fills_only_blanks(
    tmp_path: Path,
) -> None:
    video = tmp_path / "Control_Mouse01_Day7.mp4"
    video.write_bytes(b"")
    item = BatchVideoItem(
        video_id="video-1",
        video_name=video.name,
        video_path=str(video),
        group="ReviewedGroup",
        metadata_sources={"group": "manual"},
    )

    counts = apply_design_metadata_inference(
        [item],
        source_path=tmp_path,
        overwrite=False,
    )

    assert item.group == "ReviewedGroup"
    assert item.subject_id == "Mouse01"
    assert item.time_point == "Day7"
    assert item.metadata_sources["group"] == "manual"
    assert counts == {
        "group": 0,
        "subject_id": 1,
        "time_point": 1,
        "ambiguous": 0,
    }


def test_manual_value_suppresses_a_resolved_inference_conflict(
    tmp_path: Path,
) -> None:
    first = tmp_path / "DrugA" / "Control_Mouse01_Day0.mp4"
    second = tmp_path / "DrugB" / "Mouse02_Day0.mp4"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"")
    second.write_bytes(b"")
    first_item = BatchVideoItem(
        video_id="one",
        video_name=first.name,
        video_path=str(first),
        group="ReviewedGroup",
        metadata_sources={"group": "manual"},
    )
    second_item = BatchVideoItem(
        video_id="two",
        video_name=second.name,
        video_path=str(second),
    )

    counts = apply_design_metadata_inference(
        [first_item, second_item],
        source_path=tmp_path,
    )

    assert first_item.group == "ReviewedGroup"
    assert first_item.metadata_warnings == []
    assert counts["ambiguous"] == 0


def test_common_time_labels_are_converted_to_ordered_day_scale() -> None:
    assert parse_time_point_numeric("Baseline") == 0.0
    assert parse_time_point_numeric("Day7") == 7.0
    assert parse_time_point_numeric("Week2") == 14.0
    assert parse_time_point_numeric("Hour24") == 1.0
    assert parse_time_point_numeric("Minute1440") == 1.0
    assert parse_time_point_numeric("Visit3") == 3.0
    assert parse_time_point_numeric("unknown") is None


def test_generic_video_id_is_not_mistaken_for_a_subject_id(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video_id_001_Day0.mp4"
    video.write_bytes(b"")

    guess = _guess_for(video, source_path=tmp_path)

    assert guess.subject_id == ""
    assert guess.time_point == "Day0"


def test_generic_time_series_phrase_is_not_assigned_as_a_time_point(
    tmp_path: Path,
) -> None:
    video = tmp_path / "mouse_behavior_time_series.mp4"
    video.write_bytes(b"")

    guess = _guess_for(video, source_path=tmp_path)

    assert guess.subject_id == ""
    assert guess.time_point == ""


def test_time_point_is_auto_discovered_as_a_factor_even_when_numeric() -> None:
    frame = pd.DataFrame(
        {
            "video_id": ["one", "two", "three", "four"],
            "group": ["Control", "Control", "Treatment", "Treatment"],
            "subject_id": ["C1", "C2", "T1", "T2"],
            "time_point": [0, 1, 0, 1],
            "response": [1.0, 2.0, 3.0, 4.0],
        }
    )

    discovered = BatchPipeline._discover_stats_categorical_factors(frame)

    assert discovered == ["time_point"]

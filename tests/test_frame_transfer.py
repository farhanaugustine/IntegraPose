from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from integra_pose.data_preprocessing.frame_transfer import (
    execute_frame_transfer,
    format_plan_preview,
    plan_frame_transfer,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake image")


def test_short_name_plan_uses_batch_folder_and_image_ids(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    _write_image(source / "Trial Alpha" / "very_long_original_name.jpg")
    _write_image(source / "Trial Beta" / "frame.png")

    plan = plan_frame_transfer(source, dest, operation="copy", shorten_names=True)

    names = [item.new_filename for item in plan.items]
    assert len(names) == 2
    assert all(name.startswith(plan.short_prefix + "_F") for name in names)
    assert names[0].endswith("_000001.jpg")
    assert names[1].endswith("_000001.png")
    assert plan.short_prefix.startswith("IMG")
    assert len(plan.short_prefix) == 7
    assert {item.folder_id for item in plan.items} == {"F001", "F002"}


def test_execute_copy_writes_manifest_and_keeps_sources(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    original = source / "folder" / "frame001.jpg"
    _write_image(original)

    result = execute_frame_transfer(source, dest, operation="copy", shorten_names=True)

    assert result.completed == 1
    assert original.exists()
    copied = Path(result.plan.items[0].dest_path)
    assert copied.exists()
    manifest = Path(result.manifest_path)
    assert manifest.exists()
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["original_filename"] == "frame001.jpg"
    assert rows[0]["new_filename"] == copied.name
    assert rows[0]["operation"] == "copy"


def test_execute_move_removes_source_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    original = source / "folder" / "frame001.jpg"
    _write_image(original)

    result = execute_frame_transfer(source, dest, operation="move", shorten_names=True)

    assert result.completed == 1
    assert not original.exists()
    assert Path(result.plan.items[0].dest_path).exists()


def test_dry_run_writes_manifest_without_copying(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    original = source / "folder" / "frame001.jpg"
    _write_image(original)

    result = execute_frame_transfer(source, dest, operation="copy", shorten_names=True, dry_run=True)

    assert result.completed == 0
    assert result.skipped == 1
    assert original.exists()
    assert not Path(result.plan.items[0].dest_path).exists()
    assert Path(result.manifest_path).exists()


def test_readable_mode_renames_collisions(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    _write_image(source / "A" / "frame.jpg")
    _write_image(source / "A" / "frame.png")
    _write_image(source / "B" / "frame.jpg")
    dest.mkdir()
    (dest / "A_frame.jpg").write_bytes(b"existing")

    plan = plan_frame_transfer(source, dest, operation="copy", shorten_names=False)

    assert plan.collision_renamed_count == 1
    assert "A_frame_2.jpg" in {item.new_filename for item in plan.items}
    assert "B_frame.jpg" in {item.new_filename for item in plan.items}


@pytest.mark.skipif(os.name != "nt", reason="Windows filesystems use case-insensitive collision semantics")
def test_readable_mode_renames_case_only_destination_collision(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    _write_image(source / "foo.jpg")
    dest.mkdir()
    (dest / "FOO.jpg").write_bytes(b"existing")

    plan = plan_frame_transfer(source, dest, operation="copy", shorten_names=False)

    assert plan.collision_renamed_count == 1
    assert plan.items[0].new_filename == "foo_2.jpg"
    assert (dest / "FOO.jpg").read_bytes() == b"existing"


def test_execute_refuses_late_destination_collision(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    original = source / "frame.jpg"
    _write_image(original)
    collision_contents = b"created after planning"

    def _create_collision(message: str) -> None:
        if not message.startswith("Planned "):
            return
        planned_destination = dest / "frame.jpg"
        planned_destination.parent.mkdir(parents=True, exist_ok=True)
        planned_destination.write_bytes(collision_contents)

    result = execute_frame_transfer(
        source,
        dest,
        operation="copy",
        shorten_names=False,
        on_progress=_create_collision,
    )

    assert result.completed == 0
    assert result.failed == 1
    assert original.exists()
    assert (dest / "frame.jpg").read_bytes() == collision_contents


def test_preview_reports_long_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    long_dest = tmp_path / ("very_long_destination_" * 10)
    _write_image(source / "folder" / ("very_long_image_name_" * 8 + ".jpg"))

    plan = plan_frame_transfer(source, long_dest, operation="copy", shorten_names=False)
    lines = format_plan_preview(plan)

    assert plan.long_path_count >= 1
    assert any("Warning:" in line for line in lines)


def test_preview_warns_when_destination_root_is_long(tmp_path: Path) -> None:
    source = tmp_path / "source"
    long_dest = tmp_path / ("destination_" + ("x" * 170))
    _write_image(source / "folder" / "frame.jpg")

    plan = plan_frame_transfer(source, long_dest, operation="copy", shorten_names=True)
    lines = format_plan_preview(plan)

    assert plan.dest_root_long is True
    assert any("destination folder path is already long" in line for line in lines)


def test_execute_blocks_unsafe_long_paths_before_copying(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / ("destination_" + ("x" * 80))
    original = source / "folder" / ("frame_" + ("y" * 150) + ".jpg")
    _write_image(original)

    result = execute_frame_transfer(source, dest, operation="copy", shorten_names=False)

    assert result.plan.blocked_path_count >= 1
    assert result.completed == 0
    assert result.failed == result.plan.blocked_path_count
    assert original.exists()
    assert not Path(result.plan.items[0].dest_path).exists()
    assert result.plan.items[0].status == "blocked_long_path"
    assert Path(result.manifest_path).exists()

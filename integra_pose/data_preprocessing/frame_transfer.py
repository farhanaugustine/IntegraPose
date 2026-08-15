"""Flatten frame folders safely, with short names and manifest output."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import shutil
from typing import Callable

FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_SHORT_PREFIX = "IMG"
DEST_ROOT_WARN_AT = 180
PATH_LENGTH_WARN_AT = 220
PATH_LENGTH_BLOCK_AT = 255


@dataclass(slots=True)
class FrameTransferItem:
    source_path: str
    dest_path: str
    new_filename: str
    original_filename: str
    source_folder: str
    source_relative_folder: str
    folder_id: str
    short_prefix: str
    image_index: int
    operation: str
    collision_renamed: bool = False
    path_length: int = 0
    status: str = "planned"
    error: str = ""


@dataclass(slots=True)
class FrameTransferPlan:
    source_root: str
    dest_root: str
    operation: str
    shorten_names: bool
    short_prefix: str
    dry_run: bool
    items: list[FrameTransferItem] = field(default_factory=list)
    skipped_hidden: int = 0
    skipped_non_frames: int = 0
    warning_count: int = 0

    @property
    def total_frames(self) -> int:
        return len(self.items)

    @property
    def collision_renamed_count(self) -> int:
        return sum(1 for item in self.items if item.collision_renamed)

    @property
    def max_path_length(self) -> int:
        return max((item.path_length for item in self.items), default=0)

    @property
    def dest_root_length(self) -> int:
        return len(str(Path(self.dest_root).absolute()))

    @property
    def dest_root_long(self) -> bool:
        return self.dest_root_length >= DEST_ROOT_WARN_AT

    @property
    def long_path_count(self) -> int:
        return sum(1 for item in self.items if item.path_length >= PATH_LENGTH_WARN_AT)

    @property
    def blocked_path_count(self) -> int:
        return sum(1 for item in self.items if item.path_length >= PATH_LENGTH_BLOCK_AT)

    @property
    def manifest_path(self) -> str:
        return str(Path(self.dest_root) / "frame_transfer_manifest.csv")

    @property
    def summary_path(self) -> str:
        return str(Path(self.dest_root) / "frame_transfer_summary.json")


@dataclass(slots=True)
class FrameTransferResult:
    plan: FrameTransferPlan
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    manifest_path: str = ""
    summary_path: str = ""


def is_frame_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in FRAME_EXTENSIONS


def plan_frame_transfer(
    source_root: str,
    dest_root: str,
    *,
    operation: str = "copy",
    shorten_names: bool = False,
    short_prefix: str | None = None,
    dry_run: bool = False,
) -> FrameTransferPlan:
    """Create a flattening plan without changing files."""

    source = Path(source_root).expanduser()
    dest = Path(dest_root).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if not str(dest).strip():
        raise ValueError("Destination directory is required.")

    operation = _normalize_operation(operation)
    prefix = _normalize_prefix(short_prefix) if shorten_names else ""
    if shorten_names and not prefix:
        prefix = _generate_short_prefix(dest)

    plan = FrameTransferPlan(
        source_root=str(source),
        dest_root=str(dest),
        operation=operation,
        shorten_names=bool(shorten_names),
        short_prefix=prefix,
        dry_run=bool(dry_run),
    )

    folders = _collect_source_folders(source, dest, plan)
    folder_ids = {folder: f"F{idx:03d}" for idx, folder in enumerate(folders, start=1)}
    reserved_names = _existing_dest_names(dest)
    next_index_by_folder: dict[Path, int] = {}

    for folder in folders:
        next_index_by_folder[folder] = 0
        for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
            if path.name.startswith(".") or _is_hidden(path):
                plan.skipped_hidden += 1
                continue
            if not path.is_file():
                continue
            if not is_frame_file(path.name):
                plan.skipped_non_frames += 1
                continue

            next_index_by_folder[folder] += 1
            rel_folder = _relative_folder(source, folder)
            folder_id = folder_ids[folder]
            image_index = next_index_by_folder[folder]
            candidate_name = _build_output_name(
                source=source,
                folder=folder,
                path=path,
                folder_id=folder_id,
                image_index=image_index,
                shorten_names=shorten_names,
                short_prefix=prefix,
            )
            unique_name, renamed = _unique_filename(candidate_name, reserved_names)
            dest_path = dest / unique_name
            path_length = len(str(dest_path.resolve() if dest.exists() else dest_path.absolute()))

            plan.items.append(
                FrameTransferItem(
                    source_path=str(path),
                    dest_path=str(dest_path),
                    new_filename=unique_name,
                    original_filename=path.name,
                    source_folder=str(folder),
                    source_relative_folder=rel_folder,
                    folder_id=folder_id,
                    short_prefix=prefix,
                    image_index=image_index,
                    operation=operation,
                    collision_renamed=renamed,
                    path_length=path_length,
                )
            )

    plan.warning_count = plan.long_path_count + plan.collision_renamed_count
    return plan


def execute_frame_transfer(
    source_root: str,
    dest_root: str,
    *,
    operation: str = "copy",
    shorten_names: bool = False,
    dry_run: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> FrameTransferResult:
    """Run a frame transfer plan and write manifest/summary files."""

    plan = plan_frame_transfer(
        source_root,
        dest_root,
        operation=operation,
        shorten_names=shorten_names,
        dry_run=dry_run,
    )
    dest = Path(plan.dest_root)
    dest.mkdir(parents=True, exist_ok=True)

    result = FrameTransferResult(plan=plan)
    if on_progress:
        on_progress(
            f"Planned {plan.total_frames} frame(s). "
            f"Max path length={plan.max_path_length}; blocked long paths={plan.blocked_path_count}."
        )

    if plan.blocked_path_count and not dry_run:
        message = (
            f"{plan.blocked_path_count} output path(s) are too long for safe Windows file operations "
            f"(>= {PATH_LENGTH_BLOCK_AT} characters). Enable short names or choose a shorter destination folder."
        )
        for item in plan.items:
            if item.path_length >= PATH_LENGTH_BLOCK_AT:
                item.status = "blocked_long_path"
                item.error = message
        result.failed = plan.blocked_path_count
        result.manifest_path = write_transfer_manifest(plan)
        result.summary_path = write_transfer_summary(result)
        if on_progress:
            on_progress(f"Blocked: {message}")
            on_progress(f"Manifest: {result.manifest_path}")
            on_progress(f"Summary: {result.summary_path}")
        return result

    action_word = "Would copy" if plan.operation == "copy" else "Would move"
    if not dry_run:
        action_word = "Copied" if plan.operation == "copy" else "Moved"

    for idx, item in enumerate(plan.items, start=1):
        try:
            if dry_run:
                item.status = "planned"
                result.skipped += 1
            else:
                Path(item.dest_path).parent.mkdir(parents=True, exist_ok=True)
                _copy_file_without_overwrite(
                    item.source_path,
                    item.dest_path,
                    remove_source=plan.operation == "move",
                )
                item.status = "completed"
                result.completed += 1
            if on_progress and (idx <= 10 or idx == plan.total_frames or idx % 100 == 0):
                on_progress(f"{idx}/{plan.total_frames} {action_word}: {item.source_path} -> {item.dest_path}")
        except Exception as exc:
            item.status = "failed"
            item.error = str(exc)
            result.failed += 1
            if on_progress:
                on_progress(f"Failed: {item.source_path} -> {item.dest_path}: {exc}")

    result.manifest_path = write_transfer_manifest(plan)
    result.summary_path = write_transfer_summary(result)

    if on_progress:
        on_progress(format_transfer_summary(result))
        on_progress(f"Manifest: {result.manifest_path}")
        on_progress(f"Summary: {result.summary_path}")

    return result


def transfer_frames(
    source_root: str,
    dest_root: str,
    *,
    on_progress: Callable[[str], None] | None = None,
    dry_run: bool = False,
    operation: str = "move",
    shorten_names: bool = False,
) -> int:
    """Backward-compatible transfer wrapper returning a frame count."""

    result = execute_frame_transfer(
        source_root,
        dest_root,
        operation=operation,
        shorten_names=shorten_names,
        dry_run=dry_run,
        on_progress=on_progress,
    )
    return result.plan.total_frames if dry_run else result.completed


def write_transfer_manifest(plan: FrameTransferPlan) -> str:
    path = Path(plan.manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "new_filename",
        "dest_path",
        "original_filename",
        "original_path",
        "source_folder",
        "source_relative_folder",
        "folder_id",
        "short_prefix",
        "image_index",
        "operation",
        "dry_run",
        "collision_renamed",
        "path_length",
        "status",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in plan.items:
            writer.writerow(
                {
                    "new_filename": item.new_filename,
                    "dest_path": item.dest_path,
                    "original_filename": item.original_filename,
                    "original_path": item.source_path,
                    "source_folder": item.source_folder,
                    "source_relative_folder": item.source_relative_folder,
                    "folder_id": item.folder_id,
                    "short_prefix": item.short_prefix,
                    "image_index": item.image_index,
                    "operation": item.operation,
                    "dry_run": bool(plan.dry_run),
                    "collision_renamed": bool(item.collision_renamed),
                    "path_length": int(item.path_length),
                    "status": item.status,
                    "error": item.error,
                }
            )
    return str(path)


def write_transfer_summary(result: FrameTransferResult) -> str:
    plan = result.plan
    path = Path(plan.summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": plan.source_root,
        "dest_root": plan.dest_root,
        "operation": plan.operation,
        "dry_run": bool(plan.dry_run),
        "shorten_names": bool(plan.shorten_names),
        "short_prefix": plan.short_prefix,
        "total_frames": plan.total_frames,
        "completed": result.completed,
        "failed": result.failed,
        "skipped": result.skipped,
        "collision_renamed_count": plan.collision_renamed_count,
        "dest_root_length": plan.dest_root_length,
        "dest_root_long": plan.dest_root_long,
        "long_path_count": plan.long_path_count,
        "blocked_path_count": plan.blocked_path_count,
        "max_path_length": plan.max_path_length,
        "skipped_hidden": plan.skipped_hidden,
        "skipped_non_frames": plan.skipped_non_frames,
        "manifest_path": result.manifest_path,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(path)


def format_transfer_summary(result: FrameTransferResult) -> str:
    plan = result.plan
    verb = "planned" if plan.dry_run else plan.operation + "ed"
    if plan.operation == "copy" and not plan.dry_run:
        verb = "copied"
    if plan.operation == "move" and not plan.dry_run:
        verb = "moved"
    return (
        f"Frame transfer {verb}: {plan.total_frames} frame(s); "
        f"completed={result.completed}, failed={result.failed}, "
        f"collisions renamed={plan.collision_renamed_count}, "
        f"long paths={plan.long_path_count}, blocked paths={plan.blocked_path_count}, "
        f"max path length={plan.max_path_length}."
    )


def format_plan_preview(plan: FrameTransferPlan, *, max_examples: int = 8) -> list[str]:
    lines = [
        (
            f"Preview: {plan.total_frames} frame(s), {len({item.source_relative_folder for item in plan.items})} "
            f"source folder(s), operation={plan.operation}, short_names={plan.shorten_names}."
        ),
        (
            f"Collisions needing rename: {plan.collision_renamed_count}; "
            f"long output paths: {plan.long_path_count}; blocked paths: {plan.blocked_path_count}; "
            f"max path length: {plan.max_path_length}; destination root length: {plan.dest_root_length}."
        ),
    ]
    if plan.dest_root_long:
        lines.append(
            "Warning: the destination folder path is already long. For faster, safer copying on Windows, "
            "choose a short folder such as C:\\IP_frames\\batch01."
        )
    if plan.shorten_names:
        lines.append(f"Short-name prefix for this run: {plan.short_prefix}")
    for item in plan.items[:max_examples]:
        lines.append(f"Example: {item.original_filename} -> {item.new_filename}")
    if plan.total_frames > max_examples:
        lines.append(f"... {plan.total_frames - max_examples} more frame(s).")
    if plan.long_path_count:
        lines.append("Warning: some output paths are long. Enable short names or choose a shorter destination path.")
    if plan.blocked_path_count:
        lines.append(
            f"Blocked: {plan.blocked_path_count} output path(s) are too long to copy safely. "
            "Enable short names or choose a shorter destination folder before running."
        )
    return lines


def _normalize_operation(operation: str) -> str:
    text = str(operation or "").strip().lower()
    if text not in {"copy", "move"}:
        return "copy"
    return text


def _normalize_prefix(prefix: str | None) -> str:
    raw = str(prefix or DEFAULT_SHORT_PREFIX).strip().upper()
    letters = "".join(ch for ch in raw if ch.isalpha())[:3]
    if len(letters) < 3:
        letters = DEFAULT_SHORT_PREFIX
    digits = "".join(ch for ch in raw if ch.isdigit())[:4]
    return f"{letters}{digits}" if len(digits) == 4 else ""


def _generate_short_prefix(dest: Path) -> str:
    existing = _existing_dest_names(dest)
    for _ in range(100):
        code = random.randint(0, 9999)
        prefix = f"{DEFAULT_SHORT_PREFIX}{code:04d}"
        prefix_key = _reservation_key(prefix + "_")
        if not any(name.startswith(prefix_key) for name in existing):
            return prefix
    return f"{DEFAULT_SHORT_PREFIX}{random.randint(0, 9999):04d}"


def _collect_source_folders(source: Path, dest: Path, plan: FrameTransferPlan) -> list[Path]:
    folders: list[Path] = []
    source_abs = source.resolve()
    dest_abs = dest.resolve() if dest.exists() else dest.absolute()
    for dirpath, dirnames, filenames in os.walk(source_abs):
        current = Path(dirpath)
        dirnames[:] = [
            name for name in dirnames
            if not name.startswith(".")
            and not _is_hidden(current / name)
            and not _is_same_or_inside(current / name, dest_abs)
        ]
        if current.name.startswith(".") or _is_hidden(current):
            plan.skipped_hidden += 1
            continue
        if any(is_frame_file(name) for name in filenames if not name.startswith(".")):
            folders.append(current)
    return sorted(folders, key=lambda p: str(p).lower())


def _existing_dest_names(dest: Path) -> set[str]:
    if not dest.exists():
        return set()
    try:
        return {_reservation_key(entry.name) for entry in dest.iterdir() if entry.is_file()}
    except Exception:
        return set()


def _build_output_name(
    *,
    source: Path,
    folder: Path,
    path: Path,
    folder_id: str,
    image_index: int,
    shorten_names: bool,
    short_prefix: str,
) -> str:
    ext = path.suffix.lower() or ".jpg"
    if shorten_names:
        return f"{short_prefix}_{folder_id}_{image_index:06d}{ext}"
    rel_folder = _relative_folder(source, folder)
    prefix = _slugify(rel_folder.replace(os.sep, "_")) if rel_folder != "." else ""
    stem = _slugify(path.stem) or "frame"
    return f"{prefix}_{stem}{ext}" if prefix else f"{stem}{ext}"


def _unique_filename(filename: str, reserved: set[str]) -> tuple[str, bool]:
    candidate = filename
    candidate_key = _reservation_key(candidate)
    if candidate_key not in reserved:
        reserved.add(candidate_key)
        return candidate, False
    stem = Path(filename).stem
    ext = Path(filename).suffix
    idx = 2
    while True:
        candidate = f"{stem}_{idx}{ext}"
        candidate_key = _reservation_key(candidate)
        if candidate_key not in reserved:
            reserved.add(candidate_key)
            return candidate, True
        idx += 1


def _reservation_key(filename: str) -> str:
    return os.path.normcase(str(filename))


def _copy_file_without_overwrite(source_path: str, dest_path: str, *, remove_source: bool) -> None:
    """Copy a frame with exclusive destination creation; optionally remove the source."""
    source = Path(source_path)
    destination = Path(dest_path)
    created = False
    try:
        with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
            created = True
            shutil.copyfileobj(source_handle, destination_handle)
        shutil.copystat(source, destination)
        if remove_source:
            try:
                source.unlink()
            except Exception:
                destination.unlink(missing_ok=True)
                raise
    except Exception:
        if created and not remove_source:
            destination.unlink(missing_ok=True)
        raise


def _relative_folder(source: Path, folder: Path) -> str:
    try:
        rel = folder.relative_to(source)
    except ValueError:
        return "."
    text = str(rel)
    return "." if text in {"", "."} else text


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "")).strip("._-")
    return cleaned or "folder"


def _is_hidden(path: Path) -> bool:
    try:
        return bool(path.stat().st_file_attributes & 2)
    except Exception:
        return False


def _is_same_or_inside(path: Path, parent: Path) -> bool:
    try:
        child = path.resolve() if path.exists() else path.absolute()
        parent_resolved = parent.resolve() if parent.exists() else parent.absolute()
        return child == parent_resolved or parent_resolved in child.parents
    except Exception:
        return False

"""Move frames from nested folders into a single destination, keeping names unique."""

from __future__ import annotations

import os
import shutil
from typing import Callable, Iterable

FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def is_frame_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in FRAME_EXTENSIONS


def get_unique_dest_path(dest_root: str, folder_name: str, original_filename: str) -> str:
    """
    Build a destination path in dest_root using the pattern:
        <folder_name>_<original_filename>
    and ensure it does not overwrite an existing file.
    """
    base, ext = os.path.splitext(original_filename)
    if folder_name:
        base = f"{folder_name}_{base}"
    filename = base + ext
    candidate = os.path.join(dest_root, filename)

    if not os.path.exists(candidate):
        return candidate

    idx = 1
    while True:
        new_name = f"{base}_{idx}{ext}"
        candidate = os.path.join(dest_root, new_name)
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def transfer_frames(
    source_root: str,
    dest_root: str,
    *,
    on_progress: Callable[[str], None] | None = None,
    dry_run: bool = False,
) -> int:
    """
    Move frame files from source_root (recursively) into dest_root.

    Args:
        source_root: Directory that contains frames in nested folders.
        dest_root: Destination directory where frames are collected.
        on_progress: Optional callback invoked with a status string per moved file.

    Returns:
        Number of frames moved.
    """
    if not os.path.isdir(source_root):
        raise FileNotFoundError(f"Source directory does not exist: {source_root}")

    os.makedirs(dest_root, exist_ok=True)
    moved_count = 0

    for dirpath, _, filenames in os.walk(source_root):
        for fname in filenames:
            if not is_frame_file(fname):
                continue

            src_path = os.path.join(dirpath, fname)
            folder_name = "" if os.path.abspath(dirpath) == os.path.abspath(source_root) else os.path.basename(dirpath.rstrip(os.sep))
            dest_path = get_unique_dest_path(dest_root, folder_name, fname)

            action = "Would move" if dry_run else "Moving"
            if not dry_run:
                shutil.move(src_path, dest_path)
            moved_count += 1
            if on_progress:
                on_progress(f"{action}: {src_path} -> {dest_path}")

    if on_progress:
        summary = f"{'Would transfer' if dry_run else 'Completed transfer'}: {moved_count} file(s)."
        on_progress(summary)
    return moved_count

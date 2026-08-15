import os
import threading
from typing import Any

from integra_pose.data_preprocessing import (
    crop_videos,
    execute_frame_transfer,
    extract_frames,
    plan_frame_transfer,
)
from integra_pose.data_preprocessing.frame_transfer import format_plan_preview, format_transfer_summary


class DataPreprocessingController:
    """Threaded actions for the Data Preprocessing tab."""

    def __init__(self, app: Any):
        self.app = app

    def run_frame_extraction(self) -> None:
        cfg = self.app.config.data_preprocessing
        source = cfg.video_path.get().strip() or cfg.video_folder.get().strip()
        output_root = cfg.output_dir.get().strip()
        mode = cfg.extraction_mode.get().strip().lower() or "stride"
        stride = max(1, int(cfg.stride.get() or 1))
        samples = int(cfg.sample_count.get() or 0)

        if not source:
            self.app._set_preprocessing_status("Select a source video or folder.")
            self.app._toast("Select a source video file or a folder with videos.", level="error")
            return

        video_exts = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
        videos: list[str] = []
        source_dir = ""
        if os.path.isfile(source):
            ext = os.path.splitext(source)[1].lower()
            if ext not in video_exts:
                self.app._set_preprocessing_status("Selected file is not a supported video.")
                self.app._toast("Select a supported video file (*.mp4, *.avi, *.mov, *.mkv, *.m4v).", level="error")
                return
            videos = [source]
            source_dir = os.path.dirname(source)
        elif os.path.isdir(source):
            source_dir = source
            try:
                entries = sorted(os.listdir(source), key=lambda n: n.lower())
            except Exception:
                entries = []
            for name in entries:
                if name.startswith("."):
                    continue
                path = os.path.join(source, name)
                if not os.path.isfile(path):
                    continue
                if os.path.splitext(name)[1].lower() in video_exts:
                    videos.append(path)
            if not videos:
                self.app._set_preprocessing_status("No supported videos found in folder.")
                self.app._toast("No videos found in the selected folder.", level="error")
                return
            if mode == "interactive":
                self.app._set_preprocessing_status("Interactive mode requires a single video file.")
                self.app._toast("Interactive extraction only supports a single video. Use a non-interactive mode for folders.", level="error")
                return
        else:
            self.app._set_preprocessing_status("Invalid source path.")
            self.app._toast("Select a valid video file or folder.", level="error")
            return

        if not output_root:
            output_root = os.path.join(source_dir, "frames")
            cfg.output_dir.set(output_root)

        invalid_chars = set('<>:"/\\\\|?*')

        def safe_folder_name(video_path: str, used: set[str]) -> str:
            stem = os.path.splitext(os.path.basename(video_path))[0]
            cleaned = "".join("_" if ch in invalid_chars else ch for ch in stem).strip()
            if not cleaned:
                cleaned = "video"
            name = cleaned
            if name in used:
                ext = os.path.splitext(video_path)[1].lstrip(".").lower()
                ext_clean = "".join("_" if ch in invalid_chars else ch for ch in ext).strip()
                if ext_clean:
                    name = f"{cleaned}_{ext_clean}"
            if name in used:
                idx = 2
                while f"{name}_{idx}" in used:
                    idx += 1
                name = f"{name}_{idx}"
            used.add(name)
            return name

        used_names: set[str] = set()
        tasks: list[tuple[str, str]] = []
        for video_path in videos:
            folder_name = safe_folder_name(video_path, used_names)
            tasks.append((video_path, os.path.join(output_root, folder_name)))

        self.app._show_data_preprocessing_tab()

        def worker():
            try:
                total_saved = 0
                ok_count = 0
                fail_count = 0
                total = len(tasks)
                self.app._set_preprocessing_status(f"Extracting frames ({total} video(s))...")
                self.app._set_job_status("Frame extraction running")
                for idx, (video_path, out_dir) in enumerate(tasks, start=1):
                    try:
                        label = os.path.basename(video_path)
                        self.app._set_preprocessing_status(f"Extracting {idx}/{total}: {label}")
                        self.app.log_message(
                            f"Starting frame extraction ({idx}/{total}): {video_path} -> {out_dir} (mode={mode})",
                            "INFO",
                        )
                        def progress(message: str, *, _label: str = label) -> None:
                            self.app.log_message(message, "INFO")
                            if (
                                message.startswith("Saved frame ")
                                or message.startswith("Scanning motion:")
                                or message.startswith("Warning:")
                                or message.startswith("Extraction complete:")
                            ):
                                self.app._set_preprocessing_status(f"{_label}: {message}"[:220])

                        result = extract_frames(
                            video_path,
                            out_dir,
                            mode=mode,
                            stride=stride,
                            total_to_save=samples,
                            on_progress=progress,
                        )
                        ok_count += 1
                        saved = result.get("saved")
                        if isinstance(saved, int):
                            total_saved += saved
                        self.app.log_message(f"Frame extraction complete: {result}", "INFO")
                    except Exception as exc:
                        fail_count += 1
                        self.app.log_message(f"Frame extraction failed for {video_path}: {exc}", "ERROR")
                        self.app._set_preprocessing_status(f"Extraction failed for {os.path.basename(video_path)}: {exc}"[:220])

                finished = f"Extraction finished: {ok_count} ok, {fail_count} failed. Total saved: {total_saved}."
                self.app._set_preprocessing_status(finished)
                self.app._set_job_status("No active jobs")
                if fail_count:
                    self.app._toast(finished, level="error")
                else:
                    self.app._toast(finished, level="info")
            except Exception as exc:  # pragma: no cover - UI thread runner
                self.app.log_message(f"Frame extraction failed: {exc}", "ERROR")
                self.app._set_preprocessing_status("Extraction failed.")
                self.app._set_job_status("Job failed")
                self.app._toast(f"Extraction failed: {exc}", level="error")

        threading.Thread(target=worker, daemon=True).start()

    def run_video_crop(self) -> None:
        cfg = self.app.config.data_preprocessing
        folder = cfg.crop_video_dir.get().strip()
        output_subdir = cfg.crop_output_subdir.get().strip() or "cropped"
        use_cuda = bool(cfg.crop_use_cuda.get())
        force_new_roi = bool(cfg.crop_force_new_roi.get())
        ffmpeg_bin = cfg.crop_ffmpeg_bin.get().strip() or "ffmpeg"

        if not folder or not os.path.isdir(folder):
            self.app._set_preprocessing_status("Select a valid folder with videos.")
            self.app._toast("Select a valid folder with videos.", level="error")
            return

        self.app._show_data_preprocessing_tab()

        def worker():
            try:
                self.app._set_preprocessing_status("Cropping videos...")
                self.app._set_job_status("Batch crop running")
                self.app.log_message(f"Starting batch crop: {folder} -> {output_subdir} (NVIDIA encoder={use_cuda})", "INFO")
                outputs = crop_videos(
                    folder,
                    output_subdir=output_subdir,
                    use_cuda=use_cuda,
                    force_new_roi=force_new_roi,
                    ffmpeg_bin=ffmpeg_bin,
                    on_progress=lambda msg: self.app.log_message(msg, "INFO"),
                )
                self.app._set_preprocessing_status(f"Cropped {len(outputs)} video(s).")
                self.app._set_job_status("No active jobs")
                self.app.log_message(f"Batch crop complete. Outputs: {outputs}", "INFO")
                self.app._toast(f"Cropped {len(outputs)} video(s).", level="info")
            except Exception as exc:  # pragma: no cover - UI thread runner
                self.app.log_message(f"Batch crop failed: {exc}", "ERROR")
                self.app._set_preprocessing_status("Batch crop failed.")
                self.app._set_job_status("Job failed")
                self.app._toast(f"Batch crop failed: {exc}", level="error")

        threading.Thread(target=worker, daemon=True).start()

    def run_frame_transfer(self) -> None:
        cfg = self.app.config.data_preprocessing
        source = cfg.transfer_source_root.get().strip()
        dest = cfg.transfer_dest_root.get().strip()
        dry_run = bool(cfg.transfer_dry_run.get())
        operation = str(getattr(cfg, "transfer_operation", None).get() if getattr(cfg, "transfer_operation", None) is not None else "copy").strip().lower()
        shorten_names = bool(getattr(getattr(cfg, "transfer_shorten_names", None), "get", lambda: True)())

        if not source or not os.path.isdir(source):
            self.app._set_preprocessing_status("Select a valid source root.")
            self.app._toast("Select a valid source root.", level="error")
            return
        if not dest:
            self.app._set_preprocessing_status("Select a destination folder.")
            self.app._toast("Select a destination folder.", level="error")
            return

        self.app._show_data_preprocessing_tab()

        def worker():
            try:
                self.app._set_preprocessing_status("Planning frame transfer...")
                self.app._set_job_status("Frame transfer running")
                self.app.log_message(
                    f"Starting frame transfer: {source} -> {dest} "
                    f"(operation={operation}, short_names={shorten_names}, dry_run={dry_run})",
                    "INFO",
                )
                if dry_run:
                    self.app.log_message(
                        "DRY RUN is enabled: no image files will be copied or moved. "
                        "Uncheck Dry run to perform the transfer.",
                        "WARNING",
                    )
                    self.app._set_preprocessing_status(
                        "DRY RUN: planning only; no image files will be copied or moved."
                    )

                def progress(message: str) -> None:
                    self.app.log_message(message, "INFO")
                    if message[:1].isdigit() and "/" in message:
                        step = message.split(" ", 1)[0]
                        if dry_run:
                            self.app._set_preprocessing_status(f"Dry run planning frame {step}; no files copied.")
                        else:
                            action = "Moving" if operation == "move" else "Copying"
                            self.app._set_preprocessing_status(f"{action} frame {step}...")
                    elif (
                        message.startswith("Planned ")
                        or message.startswith("Blocked:")
                        or message.startswith("Frame transfer ")
                    ):
                        self.app._set_preprocessing_status(message[:180])

                result = execute_frame_transfer(
                    source,
                    dest,
                    operation=operation,
                    shorten_names=shorten_names,
                    on_progress=progress,
                    dry_run=dry_run,
                )
                finished_text = format_transfer_summary(result)
                self.app._set_job_status("No active jobs")
                if result.plan.blocked_path_count:
                    persistent = (
                        "Blocked: output paths are too long. Enable short names or choose a shorter destination "
                        "such as C:\\IP_frames\\batch01."
                    )
                    self.app._set_preprocessing_status(persistent)
                    self.app.log_message(
                        "Frame transfer was blocked before copying because some output paths are too long. "
                        "Enable short names or choose a shorter destination folder.",
                        "ERROR",
                    )
                    self.app._toast(persistent, level="error", duration_ms=12000)
                elif dry_run:
                    persistent = (
                        f"Dry run complete: planned {result.plan.total_frames} frame(s), but copied 0. "
                        "Uncheck Dry run to copy files."
                    )
                    self.app._set_preprocessing_status(persistent)
                    if result.plan.long_path_count or result.plan.dest_root_long:
                        self.app.log_message(
                            "Dry run also found path warnings. Review the preview/log before unchecking Dry run.",
                            "WARNING",
                        )
                    self.app._toast(persistent, level="info", duration_ms=12000)
                elif result.plan.long_path_count:
                    persistent = (
                        "Finished with warning: some output paths are long. Use short names or a shorter "
                        "destination folder next time."
                    )
                    self.app._set_preprocessing_status(persistent)
                    self.app.log_message(
                        "Some output paths are long. Use short names and/or choose a shorter destination folder.",
                        "WARNING",
                    )
                    self.app._toast(persistent, level="info", duration_ms=9000)
                elif result.plan.dest_root_long:
                    persistent = (
                        "Finished with warning: destination folder is long. A shorter folder such as "
                        "C:\\IP_frames\\batch01 will be faster and safer."
                    )
                    self.app._set_preprocessing_status(persistent)
                    self.app.log_message(
                        "The destination folder path is long. Copying will be safer and faster from a shorter folder "
                        "such as C:\\IP_frames\\batch01.",
                        "WARNING",
                    )
                    self.app._toast(persistent, level="info", duration_ms=9000)
                else:
                    self.app._set_preprocessing_status(finished_text)
                    self.app._toast(finished_text, level="info", duration_ms=6000)
                self.app.log_message(f"Frame transfer manifest: {result.manifest_path}", "INFO")
                self.app.log_message(f"Frame transfer summary: {result.summary_path}", "INFO")
            except Exception as exc:  # pragma: no cover - UI thread runner
                self.app.log_message(f"Frame transfer failed: {exc}", "ERROR")
                self.app._set_preprocessing_status(f"Frame transfer failed: {exc}")
                self.app._set_job_status("Job failed")
                self.app._toast(f"Frame transfer failed: {exc}", level="error", duration_ms=12000)

        threading.Thread(target=worker, daemon=True).start()

    def preview_frame_transfer(self) -> None:
        cfg = self.app.config.data_preprocessing
        source = cfg.transfer_source_root.get().strip()
        dest = cfg.transfer_dest_root.get().strip()
        operation = str(getattr(cfg, "transfer_operation", None).get() if getattr(cfg, "transfer_operation", None) is not None else "copy").strip().lower()
        shorten_names = bool(getattr(getattr(cfg, "transfer_shorten_names", None), "get", lambda: True)())

        if not source or not os.path.isdir(source):
            self.app._set_preprocessing_status("Select a valid source root.")
            self.app._toast("Select a valid source root.", level="error")
            return
        if not dest:
            self.app._set_preprocessing_status("Select a destination folder.")
            self.app._toast("Select a destination folder.", level="error")
            return

        self.app._show_data_preprocessing_tab()

        def worker():
            try:
                self.app._set_preprocessing_status("Previewing frame transfer...")
                self.app._set_job_status("Frame transfer preview running")
                plan = plan_frame_transfer(
                    source,
                    dest,
                    operation=operation,
                    shorten_names=shorten_names,
                    dry_run=True,
                )
                for line in format_plan_preview(plan):
                    level = "WARNING" if line.startswith("Warning:") or line.startswith("Blocked:") else "INFO"
                    self.app.log_message(line, level)
                status = (
                    f"Preview: {plan.total_frames} frame(s), destination root length {plan.dest_root_length}, "
                    f"max path length {plan.max_path_length}, long paths {plan.long_path_count}, "
                    f"blocked paths {plan.blocked_path_count}."
                )
                self.app._set_preprocessing_status(status)
                self.app._set_job_status("No active jobs")
                if plan.blocked_path_count:
                    persistent = (
                        "Preview blocked: output paths are too long. Enable short names or choose a shorter "
                        "destination such as C:\\IP_frames\\batch01."
                    )
                    self.app._set_preprocessing_status(persistent)
                    self.app._toast(persistent, level="error", duration_ms=12000)
                elif plan.dest_root_long:
                    persistent = (
                        "Preview warning: destination folder is long. A shorter folder will be faster and safer."
                    )
                    self.app._set_preprocessing_status(persistent)
                    self.app._toast(persistent, level="info", duration_ms=9000)
                else:
                    self.app._toast(status, level="info", duration_ms=6000)
            except Exception as exc:
                self.app.log_message(f"Frame transfer preview failed: {exc}", "ERROR")
                self.app._set_preprocessing_status(f"Frame transfer preview failed: {exc}")
                self.app._set_job_status("Job failed")
                self.app._toast(f"Preview failed: {exc}", level="error", duration_ms=12000)

        threading.Thread(target=worker, daemon=True).start()

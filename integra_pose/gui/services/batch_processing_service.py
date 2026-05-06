"""Batch processing service (queue discovery, model preflight, and execution)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from integra_pose.logic.batch_pipeline import BatchPipeline
from integra_pose.utils.batch_session import (
    BatchModelCapabilities,
    BatchSession,
    discover_videos,
    make_batch_items,
)

ProgressCallback = Callable[[BatchSession, str, float], None]
FinishCallback = Callable[[BatchSession, bool, str, dict | None], None]
ReviewCallback = Callable[[BatchSession, Any], None]


class BatchProcessingService:
    """Owns non-UI batch operations for the standalone wizard."""

    def __init__(self, app) -> None:
        self.app = app
        self.pipeline = BatchPipeline(app)
        self._run_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.last_run_result: dict | None = None

    def discover_queue(self, source_path: str, *, recursive: bool = False):
        videos = discover_videos(source_path, recursive=recursive)
        return make_batch_items(videos)

    @staticmethod
    def _normalize_task(task: str | None) -> str:
        raw = (task or "").strip().lower()
        if raw in {"pose", "detect", "segment", "classify"}:
            return raw
        return "unknown"

    @staticmethod
    def _extract_class_names(names_obj: Any) -> list[str]:
        if isinstance(names_obj, dict):
            rows: list[tuple[int | None, str]] = []
            for key, value in names_obj.items():
                try:
                    key_idx: int | None = int(key)
                except Exception:
                    key_idx = None
                rows.append((key_idx, str(value)))
            rows.sort(key=lambda row: (row[0] is None, row[0] if row[0] is not None else 0))
            return [name for _idx, name in rows if name.strip()]
        if isinstance(names_obj, (list, tuple)):
            return [str(name) for name in names_obj if str(name).strip()]
        return []

    def preflight_model_capabilities(self, model_path: str) -> BatchModelCapabilities:
        path = str(model_path or "").strip()
        if not path:
            raise ValueError("Model path is required for preflight.")

        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"Model file does not exist: {resolved}")

        from ultralytics import YOLO

        model = YOLO(str(resolved))
        task = self._normalize_task(getattr(model, "task", None))

        names = self._extract_class_names(getattr(model, "names", None))
        class_count = len(names)
        if class_count <= 0:
            try:
                class_count = int(getattr(model.model, "nc", 0) or 0)
            except Exception:
                class_count = 0
            if class_count > 0 and not names:
                names = [f"class_{idx}" for idx in range(class_count)]

        kpt_shape = None
        try:
            kpt_shape = getattr(model.model, "kpt_shape", None)
        except Exception:
            kpt_shape = None

        keypoint_count = 0
        if isinstance(kpt_shape, (list, tuple)) and kpt_shape:
            try:
                keypoint_count = int(kpt_shape[0] or 0)
            except Exception:
                keypoint_count = 0

        has_keypoints = keypoint_count > 0
        if task == "unknown":
            task = "pose" if has_keypoints else "detect"

        warnings: list[str] = []
        if not has_keypoints:
            warnings.append(
                "No keypoints detected for this model. Batch analytics will use bbox/center fallback for ROI entry and kinematic limits."
            )
        if class_count <= 1:
            warnings.append(
                "Single-class model detected. Pairwise behavior comparisons and some KPSS/group tests may be skipped."
            )

        return BatchModelCapabilities(
            model_path=str(resolved),
            task=task,
            has_keypoints=has_keypoints,
            keypoint_count=max(0, keypoint_count),
            class_count=max(0, class_count),
            class_names=names,
            supports_multiclass_stats=class_count > 1,
            warnings=warnings,
        )

    @staticmethod
    def build_session(
        *,
        source_path: str,
        roi_strategy: str,
        model_path: str,
        output_path: str,
        inference_device: str = "",
        videos,
        shared_rois: dict[str, dict[str, Any]] | None = None,
        shared_object_rois: dict[str, dict[str, Any]] | None = None,
        model_capabilities: BatchModelCapabilities | None = None,
        roi_entry_threshold: float = 0.75,
        roi_exit_threshold: float = 0.25,
        roi_event_mode: str = "bbox_only",
        keypoint_entry_index: int = 0,
        keypoint_entry_indices: list[int] | None = None,
        keypoint_entry_ratio_threshold: float = 0.5,
        object_interaction_enabled: bool = False,
        object_count: int = 0,
        object_roi_size_px: int = 20,
        object_roi_shape: str = "circle",
        object_interaction_keypoint_index: int = 0,
        object_interaction_distance_px: float = 0.0,
        use_existing_labels: bool = False,
        existing_labels_root: str = "",
        inference_batch_size: int = 25,
        tracker_enabled: bool = True,
        tracker_config_path: str = "",
        min_bout_frames: int = 3,
        max_gap_frames: int = 5,
        roi_min_dwell_frames: int = 3,
        roi_max_gap_frames: int = 5,
        video_fps: float = 0.0,
        save_annotated_video: bool = False,
        save_confidence: bool = False,
        review_policy: str = "after_all",
        stats_correction: str = "fdr_bh",
        stats_categorical_factors: list[str] | None = None,
        analytics_assay_preset: str = "custom",
        analytics_enabled_metrics: list[str] | None = None,
        analytics_enabled_modules: list[str] | None = None,
        include_kpss: bool = True,
        figure_output_preset: str = "publication_dashboard",
        export_publication_figures: bool = True,
        export_batch_dashboard: bool = True,
        export_group_stats_overview: bool = True,
        export_individual_profiles: bool = False,
        export_module_archive: bool = False,
        figure_export_mode: str = "assay_shortlist",
        generate_video_quicklooks: bool = False,
        single_animal_mode: bool = False,
        yaml_path: str = "",
        keybind_profile_path: str = "",
        roi_name_templates: list[str] | None = None,
        object_name_templates: list[str] | None = None,
    ) -> BatchSession:
        session = BatchSession.create()
        session.source_path = str(source_path or "").strip()
        session.roi_strategy = str(roi_strategy or "single").strip() or "single"
        session.model_path = str(model_path or "").strip()
        session.inference_device = str(inference_device or "").strip()
        session.output_path = str(output_path or "").strip()
        session.videos = list(videos or [])
        session.shared_rois = dict(shared_rois or {})
        session.shared_object_rois = dict(shared_object_rois or {})
        session.model_capabilities = model_capabilities or BatchModelCapabilities()
        session.roi_entry_threshold = float(roi_entry_threshold)
        session.roi_exit_threshold = float(roi_exit_threshold)
        session.roi_event_mode = str(roi_event_mode or "bbox_only").strip() or "bbox_only"
        session.keypoint_entry_index = int(keypoint_entry_index or 0)
        session.keypoint_entry_indices = [int(idx) for idx in (keypoint_entry_indices or [])]
        session.keypoint_entry_ratio_threshold = float(keypoint_entry_ratio_threshold)
        session.object_interaction_enabled = bool(object_interaction_enabled)
        session.object_count = max(0, int(object_count or 0))
        session.object_roi_size_px = max(2, int(object_roi_size_px or 20))
        session.object_roi_shape = str(object_roi_shape or "circle").strip().lower() or "circle"
        session.object_interaction_keypoint_index = max(0, int(object_interaction_keypoint_index or 0))
        session.object_interaction_distance_px = max(0.0, float(object_interaction_distance_px or 0.0))
        session.use_existing_labels = bool(use_existing_labels)
        session.existing_labels_root = str(existing_labels_root or "").strip()
        session.inference_batch_size = max(1, int(inference_batch_size or 25))
        session.tracker_enabled = bool(tracker_enabled)
        session.tracker_config_path = str(tracker_config_path or "").strip()
        session.min_bout_frames = max(1, int(min_bout_frames or 3))
        session.max_gap_frames = max(0, int(5 if max_gap_frames is None else max_gap_frames))
        session.roi_min_dwell_frames = max(
            1,
            int(session.min_bout_frames if roi_min_dwell_frames is None else roi_min_dwell_frames),
        )
        session.roi_max_gap_frames = max(
            0,
            int(session.max_gap_frames if roi_max_gap_frames is None else roi_max_gap_frames),
        )
        # video_fps: 0.0 means "unset" — the batch pipeline will probe
        # each video for its own fps. Non-zero values override the probe
        # (useful when the encoded fps is wrong, e.g. variable-frame-rate
        # exports from some camera software).
        try:
            session.video_fps = max(0.0, float(video_fps or 0.0))
        except (TypeError, ValueError):
            session.video_fps = 0.0
        session.save_annotated_video = bool(save_annotated_video)
        session.save_confidence = False
        session.review_policy = str(review_policy or "after_all").strip() or "after_all"
        session.stats_correction = str(stats_correction or "fdr_bh").strip() or "fdr_bh"
        session.stats_categorical_factors = [str(name).strip() for name in (stats_categorical_factors or []) if str(name).strip()]
        session.analytics_assay_preset = str(analytics_assay_preset or "custom").strip() or "custom"
        session.analytics_enabled_metrics = [
            str(name).strip() for name in (analytics_enabled_metrics or []) if str(name).strip()
        ]
        session.analytics_enabled_modules = [
            str(name).strip() for name in (analytics_enabled_modules or []) if str(name).strip()
        ]
        session.include_kpss = bool(include_kpss)
        session.figure_output_preset = str(figure_output_preset or "publication_dashboard").strip() or "publication_dashboard"
        session.export_publication_figures = bool(export_publication_figures)
        session.export_batch_dashboard = bool(export_batch_dashboard)
        session.export_group_stats_overview = bool(export_group_stats_overview)
        session.export_individual_profiles = bool(export_individual_profiles)
        session.export_module_archive = bool(export_module_archive)
        session.figure_export_mode = str(figure_export_mode or "assay_shortlist").strip() or "assay_shortlist"
        session.generate_video_quicklooks = bool(generate_video_quicklooks)
        session.single_animal_mode = bool(single_animal_mode)
        session.yaml_path = str(yaml_path or "").strip()
        session.keybind_profile_path = str(keybind_profile_path or "").strip()
        session.roi_name_templates = [str(name).strip() for name in (roi_name_templates or []) if str(name).strip()]
        session.object_name_templates = [str(name).strip() for name in (object_name_templates or []) if str(name).strip()]
        return session

    def save_session(
        self,
        session: BatchSession,
        *,
        output_dir: str = "",
        file_path: str = "",
    ) -> Path:
        explicit_path = str(file_path or "").strip()
        if explicit_path:
            return session.save_json(explicit_path)
        root = Path(str(output_dir or "").strip()).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return session.save_json(root / "batch_session.json")

    def load_session(self, session_json_path: str) -> BatchSession:
        path = str(session_json_path or "").strip()
        if not path:
            raise ValueError("Session JSON path is required.")
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise ValueError(f"Session JSON not found: {target}")
        return BatchSession.load_json(target)

    def is_running(self) -> bool:
        return self._run_thread is not None and self._run_thread.is_alive()

    def stop(self) -> None:
        self._stop_event.set()

    def _safe_ui_call(self, callback: Callable[[], None]) -> bool:
        root = getattr(self.app, "root", None)
        if root is not None and hasattr(root, "after"):
            try:
                root.after(0, callback)
                return True
            except Exception:
                pass
        if threading.current_thread() is threading.main_thread():
            callback()
            return True
        return False

    def start_batch_run(
        self,
        session: BatchSession,
        *,
        on_progress: ProgressCallback | None = None,
        on_finish: FinishCallback | None = None,
        on_review_checkpoint: ReviewCallback | None = None,
        resume: bool = False,
    ) -> bool:
        if self.is_running():
            return False

        self._stop_event.clear()

        def _progress(message: str, ratio: float) -> None:
            if on_progress is None:
                return
            self._safe_ui_call(lambda: on_progress(session, message, ratio))

        def _finish(cancelled: bool, message: str, payload: dict | None = None) -> None:
            if on_finish is None:
                return
            self._safe_ui_call(lambda: on_finish(session, cancelled, message, payload))

        def _review(item: Any) -> None:
            if on_review_checkpoint is None:
                return
            done = threading.Event()
            raised: list[BaseException] = []

            def _invoke() -> None:
                try:
                    on_review_checkpoint(session, item)
                except BaseException as exc:  # pragma: no cover - surfaced to pipeline thread
                    raised.append(exc)
                finally:
                    done.set()

            if not self._safe_ui_call(_invoke):
                done.set()
            done.wait()
            if raised:
                raise raised[0]

        def _run() -> None:
            try:
                result = self.pipeline.run(
                    session,
                    stop_event=self._stop_event,
                    progress_callback=_progress,
                    review_callback=_review,
                    resume=resume,
                )
                payload = {
                    "workbook_path": result.workbook_path,
                    "session_json_path": result.session_json_path,
                    "video_results": result.video_results or [],
                }
                self.last_run_result = payload
                _finish(bool(result.cancelled), str(result.message), payload)
            except Exception as exc:
                _finish(False, f"Batch run failed: {exc}", None)
            finally:
                self._run_thread = None
                self._stop_event.clear()

        self._run_thread = threading.Thread(target=_run, daemon=True)
        self._run_thread.start()
        return True

    # Backward-compatible alias used by early scaffold wiring.
    def start_scaffold_run(
        self,
        session: BatchSession,
        *,
        on_progress: ProgressCallback | None = None,
        on_finish: FinishCallback | None = None,
    ) -> bool:
        return self.start_batch_run(session, on_progress=on_progress, on_finish=on_finish)

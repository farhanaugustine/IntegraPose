"""Execution pipeline for batch inference + analytics + export."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
import json
import math
import re
import time
import traceback

import cv2
import numpy as np
import pandas as pd

from integra_pose.logic.canonical_metrics import compile_metrics_from_labels
from integra_pose.logic.supervision_runner import (
    AdvancedOverlaySettings,
    AnnotationOptions,
    InferenceSettings,
    LabelsAggregateRecorder,
    SupervisionInferenceRunner,
)
from integra_pose.utils.batch_exporter import (
    build_analysis_coverage,
    build_video_summary,
    check_fps_consistency,
    collect_batch_module_tables,
    collect_batch_frames,
    export_batch_module_tables,
    write_batch_workbook,
)
from integra_pose.utils.bout_analyzer import BoutAnalysisCancelledError
from integra_pose.utils import video_creator
from integra_pose.utils.detection_contract import enforce_ultralytics_max_det
from integra_pose.utils.frame_identity import (
    FrameIdentityError,
    extract_model_class_metadata,
    frame_label_filename,
    load_frame_label_class_metadata,
    resolve_frame_label_indices,
    write_frame_label_manifest,
)
from integra_pose.utils.operation_result import OperationStatus
from integra_pose.utils.torch_backend import normalize_ultralytics_device
from integra_pose.utils.yolo_pose_labels import (
    YoloPoseLabelSchema,
    format_yolo_pose_label,
    write_pose_label_schema,
)

ProgressFn = Callable[[str, float], None]


@dataclass(slots=True)
class BatchRunResult:
    status: OperationStatus
    message: str
    workbook_path: str = ""
    session_json_path: str = ""
    video_results: list[dict] | None = None
    total_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    resumed_count: int = 0

    @property
    def cancelled(self) -> bool:
        """Compatibility view for callers using the original callback contract."""
        return self.status is OperationStatus.CANCELLED

    @property
    def succeeded(self) -> bool:
        return self.status is OperationStatus.SUCCESS

    @property
    def partial(self) -> bool:
        return self.status is OperationStatus.PARTIAL

    @property
    def failed(self) -> bool:
        return self.status is OperationStatus.FAILED


class BatchPipeline:
    """Run per-video inference and analytics for a batch session."""

    def __init__(self, app) -> None:
        self.app = app

    def _log(self, message: str, level: str = "INFO") -> None:
        try:
            self.app.log_message(message, level)
        except Exception:
            pass

    @staticmethod
    def _existing_file(*candidates: object) -> str:
        for candidate in candidates:
            text = str(candidate or "").strip()
            if not text:
                continue
            try:
                path = Path(text).expanduser()
                if path.is_file():
                    return str(path.resolve())
            except (OSError, RuntimeError):
                continue
        return ""

    def _rehydrate_completed_video_result(self, item) -> tuple[dict | None, str]:
        """Rebuild a completed video's aggregate payload from persisted artifacts.

        A resume may skip inference and analytics only when the prior core outputs
        still exist. Otherwise the caller reprocesses the item, preventing a
        silently incomplete batch workbook.
        """
        analytics_dir_text = str(getattr(item, "analytics_output_dir", "") or "").strip()
        if not analytics_dir_text:
            return None, "analytics output directory is not recorded"
        analytics_dir = Path(analytics_dir_text).expanduser()
        if not analytics_dir.is_dir():
            return None, f"analytics output directory is missing: {analytics_dir}"

        manifest_path = analytics_dir / "run_manifest.json"
        manifest: dict = {}
        if manifest_path.is_file():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest = loaded if isinstance(loaded, dict) else {}
            except (OSError, ValueError) as exc:
                return None, f"analytics manifest is unreadable: {exc}"
        else:
            return None, f"analytics manifest is missing: {manifest_path}"

        outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
        inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
        parameters = manifest.get("parameters") if isinstance(manifest.get("parameters"), dict) else {}
        video_stem = Path(str(getattr(item, "video_path", "") or "")).stem
        detailed_bouts_csv = self._existing_file(
            outputs.get("raw_detected_bouts_csv"),
            outputs.get("raw_detailed_bouts_csv"),
            getattr(item, "detailed_bouts_csv", ""),
            outputs.get("detailed_bouts_csv"),
            analytics_dir / f"{video_stem}_detailed_bouts.csv",
        )
        summary_bouts_csv = self._existing_file(
            outputs.get("raw_summary_csv"),
            getattr(item, "summary_bouts_csv", ""),
            outputs.get("summary_csv"),
            analytics_dir / f"{video_stem}_summary.csv",
        )
        if not detailed_bouts_csv or not summary_bouts_csv:
            missing = []
            if not detailed_bouts_csv:
                missing.append("detailed bouts CSV")
            if not summary_bouts_csv:
                missing.append("summary bouts CSV")
            return None, f"required prior artifact(s) missing: {', '.join(missing)}"

        labels_dir_text = str(getattr(item, "yolo_output_dir", "") or "").strip()
        run_dir_text = str(getattr(item, "run_output_dir", "") or "").strip()
        labels_dir = Path(labels_dir_text).expanduser() if labels_dir_text else None
        run_dir = Path(run_dir_text).expanduser() if run_dir_text else None

        def _under(directory: Path | None, filename: str) -> object:
            return directory / filename if directory is not None else ""

        roi_files = outputs.get("roi_metrics_files") if isinstance(outputs.get("roi_metrics_files"), dict) else {}
        object_files = (
            outputs.get("object_interaction_files")
            if isinstance(outputs.get("object_interaction_files"), dict)
            else {}
        )
        modules = outputs.get("modules") if isinstance(outputs.get("modules"), dict) else {}
        object_module = modules.get("object_interactions") if isinstance(modules.get("object_interactions"), dict) else {}
        object_module_files = object_module.get("files") if isinstance(object_module.get("files"), dict) else {}

        item.analytics_output_dir = str(analytics_dir.resolve())
        item.detailed_bouts_csv = detailed_bouts_csv
        item.summary_bouts_csv = summary_bouts_csv
        return {
            "video_id": str(getattr(item, "video_id", "") or ""),
            "video_name": str(getattr(item, "video_name", "") or ""),
            "video_path": str(getattr(item, "video_path", "") or ""),
            "group": str(getattr(item, "group", "") or ""),
            "subject_id": str(getattr(item, "subject_id", "") or ""),
            "time_point": str(getattr(item, "time_point", "") or ""),
            "run_output_dir": str(run_dir.resolve()) if run_dir is not None and run_dir.is_dir() else run_dir_text,
            "yolo_output_dir": str(labels_dir.resolve()) if labels_dir is not None and labels_dir.is_dir() else labels_dir_text,
            "analytics_output_dir": str(analytics_dir.resolve()),
            "labels_csv": self._existing_file(_under(labels_dir, "labels.csv"), _under(run_dir, "labels.csv")),
            "metrics_csv": self._existing_file(
                getattr(item, "metrics_csv", ""),
                _under(run_dir, "metrics.csv"),
                _under(labels_dir, "metrics.csv"),
            ),
            "metrics_summary_by_track_csv": self._existing_file(
                _under(run_dir, "metrics_summary_by_track.csv"),
                _under(labels_dir, "metrics_summary_by_track.csv"),
            ),
            "metrics_summary_by_frame_csv": self._existing_file(
                _under(run_dir, "metrics_summary_by_frame.csv"),
                _under(labels_dir, "metrics_summary_by_frame.csv"),
            ),
            "detailed_bouts_csv": detailed_bouts_csv,
            "reviewed_bouts_csv": self._existing_file(
                outputs.get("reviewed_bouts_csv"),
            ),
            "summary_bouts_csv": summary_bouts_csv,
            "roi_events_csv": self._existing_file(
                outputs.get("reviewed_roi_events_csv"),
                outputs.get("roi_events_csv"),
                analytics_dir / f"{video_stem}_roi_events.csv",
            ),
            "roi_overview_csv": self._existing_file(
                outputs.get("reviewed_roi_overview_csv"),
                roi_files.get("exclusive_entries_exits"),
                roi_files.get("entries_exits"),
            ),
            "roi_concurrent_overview_csv": self._existing_file(roi_files.get("entries_exits")),
            "roi_dwell_events_csv": self._existing_file(
                outputs.get("reviewed_roi_dwell_events_csv"),
                roi_files.get("exclusive_dwell_events"),
                roi_files.get("dwell_events"),
            ),
            "object_interactions_csv": self._existing_file(
                object_module_files.get("summary"),
                object_files.get("summary"),
                object_module_files.get("per_frame"),
                object_files.get("per_frame"),
            ),
            "object_dwell_events_csv": self._existing_file(
                object_module_files.get("dwell_events"),
                object_files.get("dwell_events"),
            ),
            "roi_polygons": dict(inputs.get("roi_polygons") or {}),
            "object_roi_polygons": dict(inputs.get("object_roi_polygons") or {}),
            "object_interaction_distance_px": float(parameters.get("object_interaction_distance_px", 0.0) or 0.0),
            "analytics_annotated_video_path": self._existing_file(
                analytics_dir / f"{video_stem}_annotated.mp4"
            ),
            "run_manifest_json": str(manifest_path.resolve()),
        }, ""

    @staticmethod
    def _run_counts(active_videos: list[object]) -> tuple[int, int]:
        completed = sum(
            1
            for item in active_videos
            if str(getattr(item, "inference_status", "")).strip() == "completed"
            and str(getattr(item, "analytics_status", "")).strip() == "completed"
        )
        failed = sum(
            1
            for item in active_videos
            if str(getattr(item, "inference_status", "")).strip() == "failed"
            or str(getattr(item, "analytics_status", "")).strip() == "failed"
        )
        return completed, failed

    @staticmethod
    def _safe_float(raw: str | float | int | None, default: float) -> float:
        try:
            if raw is None:
                return default
            return float(raw)
        except Exception:
            return default

    @staticmethod
    def _safe_int(raw: str | int | float | None, default: int) -> int:
        try:
            if raw is None:
                return default
            parsed = int(raw)
            return parsed if parsed > 0 else default
        except Exception:
            return default

    @staticmethod
    def _safe_nonnegative_int(raw: str | int | float | None, default: int) -> int:
        try:
            if raw is None:
                return default
            parsed = int(raw)
            return parsed if parsed >= 0 else default
        except Exception:
            return default

    @staticmethod
    def _probe_total_frames(video_path: str) -> int:
        """Best-effort frame count probe for progress reporting."""
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap or not cap.isOpened():
                return 0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            return total if total > 0 else 0
        except Exception:
            return 0
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    @staticmethod
    def _probe_video_metadata(video_path: str) -> tuple[float, int, int]:
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap or not cap.isOpened():
                return 0.0, 0, 0
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            return fps, width, height
        except Exception:
            return 0.0, 0, 0
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    @staticmethod
    def _next_available_run_dir(candidate: Path) -> Path:
        if not candidate.exists():
            return candidate
        counter = 1
        while True:
            next_candidate = candidate.parent / f"{candidate.name}_{counter}"
            if not next_candidate.exists():
                return next_candidate
            counter += 1

    @staticmethod
    def _to_numpy(value, *, dtype=None):
        if value is None:
            return None
        candidate = value
        if hasattr(candidate, "detach"):
            try:
                candidate = candidate.detach()
            except Exception:
                pass
        if hasattr(candidate, "cpu"):
            try:
                candidate = candidate.cpu()
            except Exception:
                pass
        if hasattr(candidate, "numpy"):
            try:
                candidate = candidate.numpy()
            except Exception:
                pass
        try:
            arr = np.asarray(candidate)
        except Exception:
            return None
        if dtype is not None:
            try:
                arr = arr.astype(dtype, copy=False)
            except Exception:
                arr = arr.astype(dtype)
        return arr

    @staticmethod
    def _normalize_device(value: str | None) -> str:
        """Normalize batch device input for CUDA, ROCm, MPS, or CPU."""
        return normalize_ultralytics_device(value, preserve_auto=True)

    @staticmethod
    def _resolve_runtime_device(model: object, fallback: str) -> str:
        """Read the device selected by Ultralytics after predictor startup."""
        predictor = getattr(model, "predictor", None)
        candidates = (
            getattr(predictor, "device", None),
            getattr(getattr(predictor, "model", None), "device", None),
            getattr(getattr(model, "model", None), "device", None),
            getattr(model, "device", None),
        )
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text and text.lower() not in {"none", "-1", "auto"}:
                return text
        fallback_text = str(fallback or "").strip()
        if fallback_text.lower() in {"", "-1", "auto"}:
            return "unknown"
        return fallback_text

    @staticmethod
    def _resolve_tracker_config(raw: str | Path | None) -> Path | None:
        text = str(raw or "").strip()
        if not text:
            return None
        candidate = Path(text).expanduser()
        if candidate.is_absolute() or candidate.is_file() or any(sep in text for sep in ("/", "\\")):
            return candidate.resolve()
        # Preserve built-in Ultralytics aliases such as botsort.yaml / bytetrack.yaml.
        return Path(text)

    @staticmethod
    def _load_json_payload(path: Path) -> dict | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _normalize_path_token(raw: str | Path | None) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        try:
            return str(Path(text).expanduser().resolve()).lower()
        except Exception:
            return str(Path(text).expanduser()).lower()

    def _validate_existing_labels_dir(
        self,
        *,
        labels_dir: Path,
        video_path: str,
        labels_root: Path | None,
        video_stem: str,
        preferred_dir: str | None,
    ) -> tuple[bool, str]:
        resolved_labels_dir = labels_dir.resolve()
        run_dir = resolved_labels_dir.parent if resolved_labels_dir.name.lower() == "labels" else resolved_labels_dir
        expected_video = self._normalize_path_token(video_path)

        metadata_candidates = [
            run_dir / "inference_metadata.json",
            resolved_labels_dir / "inference_metadata.json",
        ]
        for candidate in metadata_candidates:
            if not candidate.is_file():
                continue
            payload = self._load_json_payload(candidate)
            source_video = (payload or {}).get("source_video")
            if not source_video:
                continue
            if self._normalize_path_token(source_video) == expected_video:
                return True, ""
            return False, (
                f"Existing labels at {resolved_labels_dir} belong to '{source_video}', "
                f"not '{video_path}'."
            )

        manifest_candidates = [
            run_dir / "run_manifest.json",
            resolved_labels_dir / "run_manifest.json",
        ]
        for candidate in manifest_candidates:
            if not candidate.is_file():
                continue
            payload = self._load_json_payload(candidate) or {}
            inputs = payload.get("inputs")
            if not isinstance(inputs, dict):
                continue
            source_video = inputs.get("video_file")
            if not source_video:
                continue
            if self._normalize_path_token(source_video) == expected_video:
                return True, ""
            return False, (
                f"Existing labels manifest at {resolved_labels_dir} points to '{source_video}', "
                f"not '{video_path}'."
            )

        direct_candidates: set[str] = set()
        preferred = Path(str(preferred_dir or "").strip()).expanduser() if preferred_dir else None
        if preferred:
            direct_candidates.add(self._normalize_path_token(preferred))
        if labels_root is not None:
            for candidate in (
                labels_root / video_stem,
                labels_root / video_stem / "labels",
                labels_root / f"{video_stem}_labels",
            ):
                direct_candidates.add(self._normalize_path_token(candidate))

        if self._normalize_path_token(resolved_labels_dir) in direct_candidates:
            self._log(
                (
                    f"Existing labels for '{video_stem}' matched by direct folder path at {resolved_labels_dir}, "
                    "but no source-video metadata was found. Proceeding with caution."
                ),
                "WARNING",
            )
            return True, ""

        return False, (
            f"Existing labels at {resolved_labels_dir} could not be verified against source video '{video_path}'. "
            "Use a run folder that contains inference_metadata.json or run_manifest.json."
        )

    def _build_metrics_from_labels(
        self,
        *,
        settings: InferenceSettings,
        labels_csv_path: Path,
        run_dir: Path,
        frame_width: int,
        frame_height: int,
        fps: float,
    ) -> tuple[Path, Path, Path]:
        metrics_csv = run_dir / "metrics.csv"
        summary_track_csv = run_dir / "metrics_summary_by_track.csv"
        summary_frame_csv = run_dir / "metrics_summary_by_frame.csv"
        result = compile_metrics_from_labels(
            labels_csv_path=labels_csv_path,
            run_dir=run_dir,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            motion_direction_threshold_deg=settings.motion_direction_threshold_deg,
            motion_velocity_threshold_px=settings.motion_velocity_threshold_px,
            heading_indices=settings.heading_indices,
            anchor_keypoint_index=settings.anchor_keypoint_index,
            keypoint_names=list(settings.keypoint_names or []),
            log_fn=self._log,
        )
        if result.metrics_csv.is_file():
            try:
                renderer = SupervisionInferenceRunner(settings, stop_event=None, log_fn=self._log)
                renderer._render_metrics_dashboard(result.metrics_csv)
            except Exception as exc:
                self._log(f"Failed to render canonical metrics summaries/dashboard: {exc}", "WARNING")
        return metrics_csv, summary_track_csv, summary_frame_csv

    def _run_native_inference(
        self,
        settings: InferenceSettings,
        *,
        stop_event,
        progress_fn: Optional[Callable[[int], None]] = None,
    ) -> dict[str, object]:
        from ultralytics import YOLO

        model = YOLO(str(settings.model_path))
        model_class_names, model_task = extract_model_class_metadata(model)

        # Pass the user's device string straight through to Ultralytics.
        # We deliberately do NOT call ``model.to(...)`` here — torch's
        # ``.to()`` doesn't understand Ultralytics-specific values like
        # ``-1`` (auto-pick best idle GPU), so a pre-flight transfer
        # would crash for the project's default. Ultralytics'
        # ``model.track()`` / ``model.predict()`` handle the device
        # transfer internally, including ``-1`` resolution and CPU
        # fallback when no GPU is available.
        requested_raw = str(settings.device or "").strip()
        actual_device = self._normalize_device(settings.device)

        # If the user asked for a specific GPU index that doesn't exist
        # on this machine, ``_normalize_device`` already substituted CPU.
        # Surface the substitution — users almost always want to know.
        if (
            requested_raw
            and requested_raw.lower() not in ("cpu", "")
            and requested_raw not in ("-1", "auto")
            and actual_device == "cpu"
        ):
            self._log(
                f"Requested inference device {requested_raw!r} is not available on this "
                "machine; falling back to CPU. (Set the device to '-1' to let Ultralytics "
                "auto-pick, or pick a valid GPU index.)",
                "WARNING",
            )

        run_dir = self._next_available_run_dir(settings.project / settings.run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        labels_dir = run_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        source_path = Path(settings.source_path)
        requested_max_det = int(settings.max_det)
        if requested_max_det < 1:
            raise ValueError("max_det must be at least 1.")
        effective_max_det = 1 if bool(settings.single_animal_mode) else requested_max_det
        write_frame_label_manifest(
            labels_dir,
            source=source_path,
            max_det=effective_max_det,
            class_names=model_class_names,
            class_names_source="model.names",
            model_task=model_task,
        )
        labels_csv_path = labels_dir / "labels.csv"
        labels_csv = LabelsAggregateRecorder(
            labels_csv_path,
            self._log,
            include_confidence=True,
            keypoint_names=list(settings.keypoint_names or []),
            flush_interval_frames=max(1, int(getattr(settings, "labels_csv_flush_interval_frames", 60) or 60)),
        )

        video_writer = None
        video_writer_path = ""
        frame_width = 0
        frame_height = 0
        try:
            cap = cv2.VideoCapture(str(source_path))
            if cap.isOpened():
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
        except Exception:
            pass

        from integra_pose.utils.fps_resolver import resolve_fps, FpsUnavailableError

        user_fps = float(getattr(settings, "user_video_fps", 0.0) or 0.0)
        try:
            fps, fps_source = resolve_fps(
                source_path,
                user_fps if user_fps > 0 else "",
                log_fn=self._log,
                workflow_label="the Batch Wizard's Video FPS field",
            )
        except FpsUnavailableError as exc:
            self._log(str(exc), "ERROR")
            raise

        common_kwargs = dict(
            conf=settings.conf,
            iou=settings.iou,
            imgsz=settings.imgsz,
            batch=max(1, int(getattr(settings, "inference_batch_size", 25) or 25)),
            max_det=effective_max_det,
            device=actual_device,
            stream=True,
            augment=settings.augment,
            verbose=False,
        )
        if settings.use_tracker:
            track_kwargs = dict(common_kwargs)
            track_kwargs["persist"] = True
            # Pass the user's tracker config through verbatim. When the
            # field is empty (``settings.tracker_config is None``), we do
            # NOT supply ``tracker=`` — Ultralytics then resolves its own
            # bundled default (currently ``botsort.yaml``). This keeps
            # IntegraPose honest about not silently rewriting the user's
            # tracker choice and means any tracker config that's
            # compatible with Ultralytics works without IntegraPose-side
            # surgery. The default device of ``-1`` (auto-pick best
            # idle GPU) avoids the upstream Ultralytics bug where an
            # explicit ``device=cpu`` plus default-ReID BoT-SORT crashes
            # at ReID init; users who explicitly force CPU and want
            # multi-animal tracking should pick ``bytetrack.yaml``.
            if settings.tracker_config:
                track_kwargs["tracker"] = str(settings.tracker_config)
            iterator = model.track(source=str(settings.source_path), **track_kwargs)
        else:
            iterator = model.predict(source=str(settings.source_path), **common_kwargs)

        tracker_warning_emitted = False
        max_det_warning_emitted = False
        pose_label_schema: YoloPoseLabelSchema | None = None
        wrote_detection_only_labels = False
        for frame_index, result in enumerate(iterator):
            if stop_event.is_set():
                self._log("Stop signal received. Ending native batch inference loop.", "INFO")
                break
            if progress_fn is not None:
                try:
                    progress_fn(frame_index)
                except Exception:
                    pass

            limit_outcome = enforce_ultralytics_max_det(result, effective_max_det)
            result = limit_outcome.result
            if limit_outcome.dropped_count > 0 and not max_det_warning_emitted:
                max_det_warning_emitted = True
                self._log(
                    (
                        f"The model returned {limit_outcome.original_count} detections for frame {frame_index} "
                        f"with effective max_det={effective_max_det}; retained the {limit_outcome.retained_count} "
                        "highest-confidence detection(s)."
                    ),
                    "WARNING",
                )

            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                label_path = labels_dir / frame_label_filename(source_path, frame_index)
                if label_path.exists():
                    raise RuntimeError(
                        f"Duplicate label output for zero-based frame {frame_index}: {label_path}"
                    )
                label_path.write_text("", encoding="utf-8")
                if settings.save:
                    try:
                        plotted = result.plot()
                    except Exception:
                        plotted = None
                    if plotted is not None:
                        if video_writer is None:
                            h, w = plotted.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            video_writer_path = str(run_dir / f"{source_path.stem}_annotated.mp4")
                            video_writer = cv2.VideoWriter(video_writer_path, fourcc, fps, (w, h))
                        if video_writer is not None and video_writer.isOpened():
                            video_writer.write(plotted)
                continue

            xywhn = self._to_numpy(getattr(boxes, "xywhn", None), dtype=float)
            if xywhn is None or xywhn.ndim != 2 or xywhn.shape[0] == 0:
                raise ValueError(
                    f"Frame {frame_index} reported detections without a valid normalized xywh array."
                )
            cls = self._to_numpy(getattr(boxes, "cls", None))
            conf = self._to_numpy(getattr(boxes, "conf", None), dtype=float)
            track_raw = self._to_numpy(getattr(boxes, "id", None))
            if track_raw is not None:
                track_raw = np.asarray(track_raw).reshape(-1)
                if track_raw.size != xywhn.shape[0]:
                    track_raw = None

            keypoints = getattr(result, "keypoints", None)
            kp_xyn = self._to_numpy(getattr(keypoints, "xyn", None), dtype=float) if keypoints is not None else None
            kp_conf = self._to_numpy(getattr(keypoints, "conf", None), dtype=float) if keypoints is not None else None

            keep_indices = np.arange(xywhn.shape[0], dtype=int)
            if bool(settings.single_animal_mode):
                keep_idx = 0
                if conf is not None and conf.size > 0 and np.isfinite(conf).any():
                    try:
                        keep_idx = int(np.nanargmax(conf))
                    except Exception:
                        keep_idx = 0
                keep_idx = max(0, min(keep_idx, xywhn.shape[0] - 1))
                keep_indices = np.asarray([keep_idx], dtype=int)

            xywhn = xywhn[keep_indices]
            if cls is not None and cls.size >= keep_indices.max() + 1:
                cls = np.asarray(cls).reshape(-1)[keep_indices]
            if conf is not None and conf.size >= keep_indices.max() + 1:
                conf = np.asarray(conf).reshape(-1)[keep_indices]
            if kp_xyn is not None and kp_xyn.ndim == 3 and kp_xyn.shape[0] >= keep_indices.max() + 1:
                kp_xyn = kp_xyn[keep_indices, ...]
            if kp_conf is not None and kp_conf.ndim == 2 and kp_conf.shape[0] >= keep_indices.max() + 1:
                kp_conf = kp_conf[keep_indices, ...]
            if track_raw is not None and track_raw.size >= keep_indices.max() + 1:
                track_raw = track_raw[keep_indices]

            track_strings: list[str | None] = []
            has_any_track = False
            all_have_track = True
            for idx in range(xywhn.shape[0]):
                track_val = None
                if bool(settings.single_animal_mode):
                    track_val = "0"
                elif track_raw is not None and idx < len(track_raw):
                    try:
                        raw = track_raw[idx]
                        if isinstance(raw, (np.floating, float)) and np.isnan(float(raw)):
                            track_val = None
                        else:
                            track_val = str(int(raw))
                    except Exception:
                        track_val = None
                track_strings.append(track_val)
                if track_val is not None:
                    has_any_track = True
                else:
                    all_have_track = False

            emit_track_ids = has_any_track and all_have_track
            if settings.use_tracker and not emit_track_ids and not tracker_warning_emitted:
                tracker_warning_emitted = True
                self._log(
                    "Tracker IDs were unavailable or incomplete; label files emitted without track IDs.",
                    "WARNING",
                )

            pose_payload_available = bool(
                kp_xyn is not None
                and kp_xyn.ndim == 3
                and kp_xyn.shape[0] == xywhn.shape[0]
                and kp_xyn.shape[1] > 0
                and kp_xyn.shape[2] >= 2
            )
            keypoint_confidence_available = bool(
                pose_payload_available
                and kp_conf is not None
                and kp_conf.ndim == 2
                and kp_conf.shape[0] == xywhn.shape[0]
                and kp_conf.shape[1] >= kp_xyn.shape[1]
            )
            if pose_payload_available:
                candidate_schema = YoloPoseLabelSchema(
                    keypoint_count=int(kp_xyn.shape[1]),
                    keypoint_dimensions=3 if keypoint_confidence_available else 2,
                    include_bbox=True,
                    include_bbox_confidence=bool(settings.save_conf),
                    include_track_id=True,
                )
                if wrote_detection_only_labels:
                    raise ValueError(
                        "Pose keypoints appeared after detect-only label rows had already been written; "
                        "refusing to create a mixed label schema."
                    )
                if pose_label_schema is None:
                    pose_label_schema = candidate_schema
                    write_pose_label_schema(labels_dir, pose_label_schema)
                elif pose_label_schema != candidate_schema:
                    raise ValueError(
                        "Pose-label shape changed during batch inference; refusing to write a mixed schema."
                    )
            else:
                if pose_label_schema is not None:
                    raise ValueError(
                        "A pose-label frame was missing keypoints after the pose schema was established."
                    )
                wrote_detection_only_labels = True

            lines: list[str] = []
            for idx in range(xywhn.shape[0]):
                class_id = int(cls[idx]) if cls is not None and idx < len(cls) else 0
                bbox = [float(value) for value in xywhn[idx, :4]]
                bbox_confidence = None
                if bool(settings.save_conf):
                    try:
                        bbox_confidence = float(conf[idx]) if conf is not None and idx < len(conf) else 0.0
                    except (TypeError, ValueError):
                        bbox_confidence = 0.0
                    if not np.isfinite(bbox_confidence):
                        bbox_confidence = 0.0
                    bbox_confidence = max(0.0, min(1.0, bbox_confidence))
                track_id = int(track_strings[idx]) if emit_track_ids and track_strings[idx] is not None else None

                if pose_label_schema is not None and pose_payload_available:
                    row_keypoints: list[tuple[float, ...]] = []
                    for kp_idx in range(pose_label_schema.keypoint_count):
                        kx = float(kp_xyn[idx, kp_idx, 0])
                        ky = float(kp_xyn[idx, kp_idx, 1])
                        if pose_label_schema.keypoint_dimensions == 3:
                            row_keypoints.append((kx, ky, float(kp_conf[idx, kp_idx])))
                        else:
                            row_keypoints.append((kx, ky))
                    lines.append(
                        format_yolo_pose_label(
                            class_id=class_id,
                            bbox=bbox,
                            keypoints=row_keypoints,
                            bbox_confidence=bbox_confidence,
                            track_id=track_id,
                        )
                    )
                else:
                    fields = [str(class_id), *(f"{value:.6f}" for value in bbox)]
                    if bbox_confidence is not None:
                        fields.append(f"{bbox_confidence:.6f}")
                    if track_id is not None:
                        fields.append(str(track_id))
                    lines.append(" ".join(fields))

            label_path = labels_dir / frame_label_filename(source_path, frame_index)
            if label_path.exists():
                raise RuntimeError(
                    f"Duplicate label output for zero-based frame {frame_index}: {label_path}"
                )
            label_path.write_text("\n".join(lines), encoding="utf-8")
            labels_csv.record(
                frame_index,
                xywhn,
                cls,
                conf,
                kp_xyn,
                kp_conf,
                track_strings,
            )

            if settings.save:
                try:
                    plotted = result.plot()
                except Exception:
                    plotted = None
                if plotted is not None:
                    if video_writer is None:
                        h, w = plotted.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer_path = str(run_dir / f"{source_path.stem}_annotated.mp4")
                        video_writer = cv2.VideoWriter(video_writer_path, fourcc, fps, (w, h))
                    if video_writer is not None and video_writer.isOpened():
                        video_writer.write(plotted)

        labels_csv.close()
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass

        from integra_pose.utils.repro_metadata import build_inference_metadata
        from integra_pose.utils.safe_io import safe_write_json

        metadata_extra = {
            "max_det_requested": requested_max_det,
            "integrapose_source_file": str(Path(__file__).resolve()),
            "keypoint_names": list(getattr(settings, "keypoint_names", None) or []),
            "keypoint_names_source": str(getattr(settings, "keypoint_names_source", "") or "unresolved"),
            "keypoint_count": len(getattr(settings, "keypoint_names", None) or []),
            "heading_indices": list(getattr(settings, "heading_indices", None)) if getattr(settings, "heading_indices", None) else None,
            "heading_names": list(getattr(settings, "heading_names", None)) if getattr(settings, "heading_names", None) else None,
            "heading_source": str(getattr(settings, "heading_source", "") or "unavailable"),
            "anchor_keypoint_index": getattr(settings, "anchor_keypoint_index", None),
            "anchor_keypoint_name": str(getattr(settings, "anchor_keypoint_name", "") or ""),
        }
        seeds_snapshot = getattr(self, "_active_seeds_snapshot", None)
        if isinstance(seeds_snapshot, dict) and seeds_snapshot:
            metadata_extra["seeds"] = dict(seeds_snapshot)
        runtime_device = self._resolve_runtime_device(model, actual_device)
        metadata = build_inference_metadata(
            source_video=source_path,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            fps_source=fps_source,
            device_requested=requested_raw or actual_device,
            device_used=runtime_device,
            single_animal_mode=bool(settings.single_animal_mode),
            model_path=settings.model_path,
            conf_threshold=getattr(settings, "conf", None),
            iou_threshold=getattr(settings, "iou", None),
            max_det=effective_max_det,
            use_tracker=bool(settings.use_tracker),
            tracker_config=settings.tracker_config,
            extra=metadata_extra,
        )
        safe_write_json(run_dir / "inference_metadata.json", metadata, indent=2)
        return {
            "run_dir": run_dir,
            "labels_dir": labels_dir,
            "labels_csv": labels_csv_path,
            "annotated_video_path": video_writer_path,
        }

    def _build_post_inference_metrics_from_labels(
        self,
        *,
        labels_csv_path: Path,
        run_dir: Path,
        frame_width: int,
        frame_height: int,
        motion_direction_threshold_deg: float,
        motion_velocity_threshold_px: float,
    ) -> tuple[Path, Path, Path]:
        metrics_csv = run_dir / "metrics.csv"
        summary_track_csv = run_dir / "metrics_summary_by_track.csv"
        summary_frame_csv = run_dir / "metrics_summary_by_frame.csv"

        if not labels_csv_path.is_file():
            return metrics_csv, summary_track_csv, summary_frame_csv

        try:
            df = pd.read_csv(labels_csv_path)
        except Exception as exc:
            self._log(f"Failed to load labels CSV for post metrics: {exc}", "WARNING")
            return metrics_csv, summary_track_csv, summary_frame_csv

        required = {"frame", "x_center_n", "y_center_n"}
        if not required.issubset(set(df.columns)):
            self._log("Skipping post metrics: labels.csv missing required center columns.", "WARNING")
            return metrics_csv, summary_track_csv, summary_frame_csv

        work = df.copy()
        work["frame"] = pd.to_numeric(work["frame"], errors="coerce")
        work["x_center_n"] = pd.to_numeric(work["x_center_n"], errors="coerce")
        work["y_center_n"] = pd.to_numeric(work["y_center_n"], errors="coerce")
        if "track_id" not in work.columns:
            work["track_id"] = "0"
        work["track_id"] = work["track_id"].fillna("0").astype(str)
        work = work.dropna(subset=["frame", "x_center_n", "y_center_n"])
        if work.empty:
            return metrics_csv, summary_track_csv, summary_frame_csv

        fw = max(1.0, float(frame_width or 0.0))
        fh = max(1.0, float(frame_height or 0.0))
        work["anchor_x_px"] = work["x_center_n"] * fw
        work["anchor_y_px"] = work["y_center_n"] * fh
        work["frame"] = work["frame"].astype(int)
        work = work.sort_values(["track_id", "frame"]).reset_index(drop=True)

        threshold = max(0.0, float(motion_direction_threshold_deg))
        velocity_floor = max(0.0, float(motion_velocity_threshold_px))

        def _bearing(dx: float, dy: float) -> float:
            if dx == 0.0 and dy == 0.0:
                return 0.0
            angle = np.degrees(np.arctan2(dx, dy))
            return float((angle + 360.0) % 360.0)

        def _angular_delta(a: float, b: float) -> float:
            return float(abs((a - b + 180.0) % 360.0 - 180.0))

        rows: list[dict[str, object]] = []
        for track_id, group in work.groupby("track_id", dropna=False):
            group = group.sort_values("frame")
            prev_x = None
            prev_y = None
            prev_heading = None
            turn_count = 0
            total_path = 0.0
            for _, row in group.iterrows():
                x = float(row["anchor_x_px"])
                y = float(row["anchor_y_px"])
                speed = 0.0
                heading = np.nan
                if prev_x is not None and prev_y is not None:
                    dx = x - prev_x
                    dy = y - prev_y
                    speed = float(np.hypot(dx, dy))
                    total_path += speed
                    heading = _bearing(dx, -dy)
                    if prev_heading is not None and speed >= velocity_floor:
                        if _angular_delta(prev_heading, heading) >= threshold:
                            turn_count += 1
                    if speed >= velocity_floor and np.isfinite(heading):
                        prev_heading = heading
                else:
                    prev_heading = None
                prev_x, prev_y = x, y

                rows.append(
                    {
                        "frame": int(row["frame"]),
                        "object_id": str(track_id),
                        "class_id": int(row["class_id"]) if "class_id" in row and pd.notna(row["class_id"]) else 0,
                        "confidence": float(row["bbox_conf"]) if "bbox_conf" in row and pd.notna(row["bbox_conf"]) else np.nan,
                        "anchor_x_px": x,
                        "anchor_y_px": y,
                        "movement_heading_deg": heading,
                        "movement_speed_px_per_frame": speed,
                        "total_path_length_px": total_path,
                        "turn_count": turn_count,
                    }
                )

        metrics_df = pd.DataFrame(rows)
        if metrics_df.empty:
            return metrics_csv, summary_track_csv, summary_frame_csv

        try:
            metrics_df.to_csv(metrics_csv, index=False)
        except Exception as exc:
            self._log(f"Failed to write post metrics.csv: {exc}", "WARNING")

        try:
            summary_track = (
                metrics_df.groupby("object_id", as_index=False)
                .agg(
                    frames_observed=("frame", "count"),
                    first_frame=("frame", "min"),
                    last_frame=("frame", "max"),
                    mean_speed_px_per_frame=("movement_speed_px_per_frame", "mean"),
                    turn_count=("turn_count", "max"),
                    total_path_length_px=("total_path_length_px", "max"),
                )
                .rename(columns={"object_id": "track_id"})
            )
            summary_track.to_csv(summary_track_csv, index=False)
        except Exception as exc:
            self._log(f"Failed to write post metrics_summary_by_track.csv: {exc}", "WARNING")

        try:
            summary_frame = (
                metrics_df.groupby("frame", as_index=False)
                .agg(
                    objects_in_frame=("object_id", "nunique"),
                    mean_speed_px_per_frame=("movement_speed_px_per_frame", "mean"),
                )
            )
            summary_frame.to_csv(summary_frame_csv, index=False)
        except Exception as exc:
            self._log(f"Failed to write post metrics_summary_by_frame.csv: {exc}", "WARNING")

        return metrics_csv, summary_track_csv, summary_frame_csv

    def _resolve_yaml_path(self, session, output_root: Path) -> Path:
        if session.yaml_path:
            configured = Path(session.yaml_path).expanduser()
            if configured.is_file():
                return configured.resolve()

        try:
            cfg_path = str(self.app.config.analytics.roi_analytics_yaml_path_var.get() or "").strip()
        except Exception:
            cfg_path = ""
        if cfg_path:
            configured = Path(cfg_path).expanduser()
            if configured.is_file():
                return configured.resolve()

        class_names = list(session.model_capabilities.class_names or [])
        if not class_names:
            class_names = ["class_0"]
        names_map = {idx: name for idx, name in enumerate(class_names)}
        payload = {
            "path": str(output_root),
            "names": names_map,
            "rois": {},
        }
        keypoint_names, _source = self._get_keypoint_names(session)
        if keypoint_names:
            payload["kpt_shape"] = [len(keypoint_names), 3]
            payload["kpt_names"] = list(keypoint_names)
        yaml_path = output_root / "_batch_generated_dataset.yaml"
        try:
            import yaml

            yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        except Exception:
            # Fallback JSON syntax is valid enough for yaml.safe_load usage.
            yaml_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return yaml_path

    def _get_keypoint_names(self, session=None) -> tuple[list[str], str]:
        expected = 0
        if session is not None:
            capabilities = getattr(session, "model_capabilities", None)
            expected = max(0, int(getattr(capabilities, "keypoint_count", 0) or 0))
            task = str(getattr(capabilities, "task", "unknown") or "unknown").strip().lower()
            if task == "detect":
                return [], "not_applicable"
            session_names = [str(name).strip() for name in (getattr(session, "keypoint_names", []) or []) if str(name).strip()]
            if session_names and (not expected or len(session_names) == expected):
                return session_names, str(getattr(session, "keypoint_names_source", "session") or "session")
            capability_names = [str(name).strip() for name in (getattr(capabilities, "keypoint_names", []) or []) if str(name).strip()]
            if capability_names and (not expected or len(capability_names) == expected):
                return capability_names, str(getattr(capabilities, "keypoint_names_source", "model") or "model")

        names: list[str] = []
        try:
            raw = str(self.app.config.setup.keypoint_names_str.get() or "").strip()
        except Exception:
            raw = ""
        if raw:
            names = [token.strip() for token in raw.split(",") if token.strip()]
        if names and (not expected or len(names) == expected):
            return names, "gui_config"
        if expected:
            return [f"kp{idx}" for idx in range(expected)], "generated_generic"
        return names, "gui_config" if names else "unresolved"

    @staticmethod
    def _semantic_keypoint_index(names: list[str], aliases: set[str]) -> int | None:
        for idx, name in enumerate(names):
            token = re.sub(r"[^a-z0-9]+", "", str(name).casefold())
            if token in aliases:
                return idx
        return None

    def _build_inference_settings(self, session, video_path: str, infer_project_dir: Path, run_name: str) -> InferenceSettings:
        cfg = self.app.config.inference
        keypoint_names, keypoint_names_source = self._get_keypoint_names(session)
        grid_enabled_var = getattr(cfg, "grid_metrics_enabled_var", None)
        grid_size_var = getattr(cfg, "grid_size_px_var", None)
        grid_heatmaps_var = getattr(cfg, "grid_save_heatmaps_var", None)
        grid_bg_var = getattr(cfg, "grid_heatmap_use_frame_var", None)
        grid_enabled = bool(grid_enabled_var.get()) if hasattr(grid_enabled_var, "get") else False
        grid_size = self._safe_int(grid_size_var.get() if hasattr(grid_size_var, "get") else 50, 50)
        grid_save_heatmaps = bool(grid_heatmaps_var.get()) if hasattr(grid_heatmaps_var, "get") else False
        grid_heatmap_use_frame = bool(grid_bg_var.get()) if hasattr(grid_bg_var, "get") else True

        advanced = AdvancedOverlaySettings(
            halo_kernel=3,
            halo_opacity=0.35,
            blur_kernel=15,
            pixelate_size=12,
            heatmap_radius=30,
            heatmap_kernel=15,
            heatmap_opacity=0.50,
            heatmap_decay=0.97,
            heatmap_source="bbox",
            heatmap_anchor="CENTER",
            heatmap_keypoint_index=0,
            background_opacity=0.4,
            background_force_box=False,
            vertex_radius=4,
            edge_thickness=2,
            keypoint_palette="jet",
            edge_palette="jet",
            trace_length=20,
            trace_thickness=2,
            trace_opacity=0.85,
            trace_persistent=False,
            trace_source="bbox",
            trace_keypoint_index=0,
            trace_color="",
        )
        annotation = AnnotationOptions(
            use_boxes=False,
            use_labels=False,
            use_trace=False,
            use_tracker_ids=False,
            use_edges=False,
            use_vertices=False,
            use_heading_arrows=False,
            use_halo=False,
            use_blur=False,
            use_pixelate=False,
            use_background_overlay=False,
            use_heatmap=False,
            skeleton_source="config",
            skeleton_color="red",
            skeleton_file=None,
            config_skeleton=list(getattr(self.app.config.pose_clustering, "skeleton_connections", [])),
            hide_labels=bool(cfg.infer_hide_labels_var.get()),
            hide_conf=bool(cfg.infer_hide_conf_var.get()),
            tracker_enabled=bool(session.tracker_enabled),
            line_width=None,
            overlay_order=[],
            advanced=advanced,
            performance_safe=True,
            trace_source="bbox",
            trace_keypoint_index=None,
            trace_color="",
        )
        heading_from = str(getattr(cfg, "metrics_heading_from_var", "").get() if hasattr(getattr(cfg, "metrics_heading_from_var", None), "get") else "").strip()
        heading_to = str(getattr(cfg, "metrics_heading_to_var", "").get() if hasattr(getattr(cfg, "metrics_heading_to_var", None), "get") else "").strip()
        heading_indices: tuple[int, int] | None = None
        heading_names: tuple[str, str] | None = None
        heading_source = "unavailable"
        if heading_from and heading_to and heading_from != heading_to and heading_from in keypoint_names and heading_to in keypoint_names:
            heading_indices = (keypoint_names.index(heading_from), keypoint_names.index(heading_to))
            heading_names = (heading_from, heading_to)
            heading_source = "gui_config"
        else:
            tail_index = self._semantic_keypoint_index(
                keypoint_names,
                {"tailbase", "baseoftail", "tailroot", "basetail"},
            )
            nose_index = self._semantic_keypoint_index(keypoint_names, {"nose", "snout"})
            if tail_index is not None and nose_index is not None and tail_index != nose_index:
                heading_indices = (tail_index, nose_index)
                heading_names = (keypoint_names[tail_index], keypoint_names[nose_index])
                heading_source = "semantic_tailbase_to_nose"
                if heading_from or heading_to:
                    self._log(
                        f"Configured heading pair '{heading_from}' to '{heading_to}' is not present in the validated "
                        f"keypoint schema; using {heading_names[0]} to {heading_names[1]}.",
                        "WARNING",
                    )
        anchor_index = self._semantic_keypoint_index(
            keypoint_names,
            {"centerspine", "centrespine", "thorax", "bodycenter", "centrebody", "centerbody", "midback"},
        )
        anchor_name = keypoint_names[anchor_index] if anchor_index is not None else None
        tracker_config_raw = str(getattr(session, "tracker_config_path", "") or "").strip()
        if not tracker_config_raw:
            tracker_config_raw = str(cfg.tracker_config_path.get() or "").strip()
        return InferenceSettings(
            model_path=Path(session.model_path).expanduser().resolve(),
            source_path=Path(video_path).expanduser().resolve(),
            tracker_config=self._resolve_tracker_config(tracker_config_raw),
            use_tracker=bool(session.tracker_enabled),
            conf=self._safe_float(cfg.conf_thres_var.get(), 0.25),
            iou=self._safe_float(cfg.iou_thres_var.get(), 0.45),
            imgsz=self._safe_int(cfg.infer_imgsz_var.get(), 640),
            device=str(session.inference_device or cfg.infer_device_var.get() or "").strip(),
            max_det=max(1, self._safe_int(getattr(session, "max_det", 300), 300)),
            augment=bool(cfg.infer_augment_var.get()),
            show=False,
            preview_max_side=self._safe_nonnegative_int(
                getattr(getattr(cfg, "infer_preview_max_side_var", None), "get", lambda: "1280")(),
                1280,
            ),
            preview_frame_stride=self._safe_int(
                getattr(getattr(cfg, "infer_preview_stride_var", None), "get", lambda: "2")(),
                2,
            ),
            save=bool(session.save_annotated_video),
            async_video_save=bool(getattr(getattr(cfg, "infer_async_video_save_var", None), "get", lambda: False)()),
            async_video_queue_size=self._safe_int(
                getattr(getattr(cfg, "infer_async_video_queue_size_var", None), "get", lambda: "8")(),
                8,
            ),
            save_video_stride=self._safe_int(
                getattr(getattr(cfg, "infer_save_video_stride_var", None), "get", lambda: "1")(),
                1,
            ),
            save_txt=True,
            save_conf=False,
            save_crop=False,
            project=infer_project_dir,
            run_name=run_name,
            capture_metrics=True,
            grid_metrics_enabled=grid_enabled,
            grid_size_px=grid_size,
            grid_save_heatmaps=grid_save_heatmaps,
            grid_heatmap_use_frame=grid_heatmap_use_frame,
            heading_indices=heading_indices,
            heading_names=heading_names,
            model_keypoint_names=keypoint_names if keypoint_names else None,
            keypoint_names=keypoint_names if keypoint_names else None,
            keypoint_names_source=keypoint_names_source,
            anchor_keypoint_index=anchor_index,
            anchor_keypoint_name=anchor_name,
            heading_source=heading_source,
            annotation=annotation,
            motion_direction_threshold_deg=self._safe_float(cfg.motion_direction_threshold_var.get(), 15.0),
            motion_velocity_threshold_px=self._safe_float(cfg.motion_velocity_threshold_var.get(), 0.0),
            metrics_flush_interval_frames=self._safe_int(
                getattr(getattr(cfg, "metrics_flush_interval_frames_var", None), "get", lambda: "60")(),
                60,
            ),
            labels_csv_flush_interval_frames=self._safe_int(
                getattr(getattr(cfg, "labels_csv_flush_interval_frames_var", None), "get", lambda: "60")(),
                60,
            ),
            resource_guardrails_enabled=bool(
                getattr(getattr(cfg, "infer_resource_guardrails_enabled_var", None), "get", lambda: True)()
            ),
            min_free_disk_gb=self._safe_float(
                getattr(getattr(cfg, "infer_min_free_disk_gb_var", None), "get", lambda: "2.0")(),
                2.0,
            ),
            min_free_memory_mb=self._safe_nonnegative_int(
                getattr(getattr(cfg, "infer_min_free_memory_mb_var", None), "get", lambda: "1024")(),
                1024,
            ),
            single_animal_mode=bool(session.single_animal_mode),
            inference_batch_size=self._safe_int(
                getattr(session, "inference_batch_size", getattr(session, "ultralytics_batch_size", 25)),
                25,
            ),
            user_video_fps=float(getattr(session, "video_fps", 0.0) or 0.0),
        )

    @staticmethod
    def _choose_rois_for_video(session, item) -> dict:
        if getattr(item, "rois", None):
            return dict(item.rois)
        return dict(session.shared_rois or {})

    @staticmethod
    def _choose_object_rois_for_video(session, item) -> dict:
        if getattr(item, "object_rois", None):
            return dict(item.object_rois)
        return dict(getattr(session, "shared_object_rois", {}) or {})

    @staticmethod
    def _index_existing_label_dirs(root_dir: Path) -> list[tuple[Path, list[str]]]:
        grouped: dict[Path, list[str]] = {}
        if not root_dir.exists() or not root_dir.is_dir():
            return []
        for txt_file in root_dir.rglob("*.txt"):
            parent = txt_file.parent
            grouped.setdefault(parent, []).append(txt_file.name.lower())
        rows: list[tuple[Path, list[str]]] = []
        for parent, names in grouped.items():
            try:
                resolved = resolve_frame_label_indices(names)
            except FrameIdentityError:
                continue
            if resolved:
                rows.append((parent, sorted(resolved)))
        return rows

    @staticmethod
    def _find_existing_labels_dir(
        *,
        video_stem: str,
        preferred_dir: str | None,
        labels_root: Path | None,
        indexed_dirs: list[tuple[Path, list[str]]] | None,
    ) -> Path | None:
        stem = str(video_stem or "").strip().lower()
        if not stem:
            return None

        def _source_filename_match(name: str) -> bool:
            file_stem = Path(name).stem.casefold()
            return bool(
                file_stem == stem
                or re.match(rf"^{re.escape(stem)}__(?:frame|frm|image|img)_?\d+$", file_stem)
                or re.match(rf"^{re.escape(stem)}_(?:frame|frm|image|img)_?\d+$", file_stem)
                or re.match(rf"^{re.escape(stem)}_\d+$", file_stem)
            )

        def _dir_has_frames(path: Path, *, require_source_match: bool) -> bool:
            if not path.exists() or not path.is_dir():
                return False
            try:
                names = [item.name for item in path.glob("*.txt") if not item.name.startswith(".")]
                if require_source_match:
                    names = [name for name in names if _source_filename_match(name)]
                return bool(resolve_frame_label_indices(names, source=video_stem))
            except Exception:
                return False

        preferred = Path(str(preferred_dir or "").strip()).expanduser() if preferred_dir else None
        if preferred and _dir_has_frames(preferred, require_source_match=False):
            return preferred.resolve()

        if labels_root is not None:
            direct_candidates = [labels_root / video_stem, labels_root / video_stem / "labels", labels_root / f"{video_stem}_labels"]
            for candidate in direct_candidates:
                if _dir_has_frames(candidate, require_source_match=True):
                    return candidate.resolve()

        if not indexed_dirs:
            return None

        prefixes = (f"{stem}_frame", f"{stem}__frame")
        candidates: list[tuple[int, str, Path]] = []
        for parent, names in indexed_dirs:
            matching_names = [name for name in names if _source_filename_match(name)]
            prefix_hit = any(name.startswith(prefixes) for name in matching_names)
            stem_hit = bool(matching_names)
            parent_name = parent.name.lower()
            parent_hit = parent_name in {stem, f"{stem}_labels"}
            # Avoid false positives by requiring a concrete video-stem signal.
            if not (prefix_hit or stem_hit or parent_hit):
                continue
            score = 0
            if prefix_hit:
                score += 120
            if stem_hit:
                score += 60
            if parent_hit:
                score += 25
            if parent_name == "labels":
                score += 10
            score += min(len(matching_names), 20)
            candidates.append((score, str(parent).casefold(), parent))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            return None
        return candidates[0][2].resolve()

    @staticmethod
    def _normalize_stats_categorical_factors(
        configured_factors: list[str] | None,
        *,
        supports_multiclass_stats: bool,
        available_columns: set[str],
    ) -> tuple[list[str], list[str], list[str]]:
        raw_factors = list(configured_factors or [])
        normalized: list[str] = []
        removed_behavior: list[str] = []
        removed_unavailable: list[str] = []
        seen: set[str] = set()
        behavior_ban = {"behavior", "class", "class_id", "behavior_id", "class_name", "behavior_name"}
        available_map = {str(col).lower(): str(col) for col in available_columns}
        for factor in raw_factors:
            key = str(factor or "").strip()
            if not key:
                continue
            key_lower = key.lower()
            if key_lower in seen:
                continue
            seen.add(key_lower)
            if (not supports_multiclass_stats) and (key_lower in behavior_ban or key_lower.startswith("behavior_")):
                removed_behavior.append(key)
                continue
            mapped_key = available_map.get(key_lower)
            if available_columns and not mapped_key:
                removed_unavailable.append(key)
                continue
            normalized.append(mapped_key or key)
        return normalized, removed_behavior, removed_unavailable

    @staticmethod
    def _discover_stats_categorical_factors(
        video_summary_df: pd.DataFrame,
    ) -> list[str]:
        """Find replicated categorical design columns, excluding ID/unit fields."""

        if video_summary_df is None or video_summary_df.empty:
            return []
        excluded = {
            "group",
            "subject_id",
            "video_id",
            "video_name",
            "video_path",
        }
        discovered: list[str] = []
        row_count = max(1, int(len(video_summary_df.index)))
        max_levels = max(2, min(24, row_count))
        for column in video_summary_df.columns:
            column_name = str(column)
            column_key = column_name.casefold()
            if column_key in excluded:
                continue
            series = video_summary_df[column]
            if pd.api.types.is_numeric_dtype(series) and column_key != "time_point":
                continue
            cleaned = series.fillna("").astype(str).str.strip()
            cleaned = cleaned[cleaned != ""]
            level_count = int(cleaned.nunique(dropna=True))
            if 2 <= level_count <= max_levels:
                discovered.append(column_name)
        return discovered

    @staticmethod
    def _figure_export_options(session) -> dict[str, object]:
        export_mode = (
            str(getattr(session, "figure_export_mode", "full_bundle") or "full_bundle").strip()
            or "full_bundle"
        )
        return {
            "export_mode": export_mode,
            "export_publication_figures": bool(
                getattr(session, "export_publication_figures", export_mode != "disabled")
            ),
            "export_batch_dashboard": bool(
                getattr(session, "export_batch_dashboard", False)
            ),
            "export_group_stats_overview": bool(
                getattr(session, "export_group_stats_overview", False)
            ),
            "export_individual_profiles": bool(
                getattr(session, "export_individual_profiles", export_mode == "full_bundle")
            ),
            "export_module_archive": bool(
                getattr(session, "export_module_archive", export_mode == "full_bundle")
            ),
            "generate_video_quicklooks": bool(
                getattr(session, "generate_video_quicklooks", export_mode == "full_bundle")
            ),
        }

    def _export_batch_aggregates(
        self,
        *,
        session,
        video_results: list[dict],
        output_root: Path,
        progress_callback: Optional[ProgressFn] = None,
        precomputed_figure_records: list[dict[str, str]] | None = None,
    ) -> str:
        """Rebuild every batch-level derivative from active per-video outputs."""

        def _progress(message: str, ratio: float) -> None:
            if progress_callback is not None:
                progress_callback(message, ratio)

        fps_summary = check_fps_consistency(video_results, log_fn=self._log) or {}
        if (
            not bool(fps_summary.get("consistent", True))
            and str(
                getattr(session, "temporal_threshold_unit", "frames") or "frames"
            ).strip().lower()
            == "seconds"
        ):
            self._log(
                "Second-based bout and ROI thresholds were resolved separately "
                "for each video's FPS, so their real-time meaning is preserved.",
                "INFO",
            )
        keypoint_df, kinematic_df, bout_df = collect_batch_frames(video_results)
        video_summary_df = build_video_summary(video_results)
        analysis_coverage_df = build_analysis_coverage(video_results)
        module_bundle = collect_batch_module_tables(video_results)
        module_file_index_df, module_table_index_df = export_batch_module_tables(
            module_bundle,
            output_dir=output_root / "module_tables",
        )
        analysis_coverage_csv = output_root / "analysis_coverage_table.csv"
        (analysis_coverage_df if not analysis_coverage_df.empty else pd.DataFrame()).to_csv(
            analysis_coverage_csv,
            index=False,
        )

        class_capabilities = getattr(session, "model_capabilities", None)
        supports_multiclass = bool(
            getattr(class_capabilities, "supports_multiclass_stats", False)
        )
        class_names = list(getattr(class_capabilities, "class_names", []) or [])
        class_count = int(getattr(class_capabilities, "class_count", 0) or 0)
        if class_count <= 0 and class_names:
            class_count = len(class_names)
        if class_count > 1:
            supports_multiclass = True
        elif class_count == 1:
            supports_multiclass = False
        configured_stats_factors = list(
            getattr(session, "stats_categorical_factors", None) or []
        )
        if bool(getattr(session, "stats_auto_detect_design", True)):
            inferred_stats_factors = self._discover_stats_categorical_factors(
                video_summary_df
            )
            for factor in inferred_stats_factors:
                if factor.casefold() not in {
                    str(existing).casefold()
                    for existing in configured_stats_factors
                }:
                    configured_stats_factors.append(factor)
            if inferred_stats_factors:
                self._log(
                    "Automatically assigned categorical study-design factors: "
                    + ", ".join(inferred_stats_factors),
                    "INFO",
                )
        (
            stats_factors,
            removed_behavior_factors,
            removed_unavailable_factors,
        ) = self._normalize_stats_categorical_factors(
            configured_stats_factors,
            supports_multiclass_stats=supports_multiclass,
            available_columns=set(video_summary_df.columns),
        )
        if removed_behavior_factors:
            self._log(
                "Single-class mode detected; skipping behavior-like categorical "
                "factors in batch stats: "
                + ", ".join(removed_behavior_factors),
                "INFO",
            )
        if removed_unavailable_factors:
            self._log(
                "Batch stats ignored unavailable categorical factors: "
                + ", ".join(removed_unavailable_factors),
                "INFO",
            )

        _progress("Rebuilding group statistics...", 99.15)
        stats_output_dir = output_root / "group_stats"
        from integra_pose.logic.group_stats import export_group_stats_bundle

        omnibus_df, pairwise_df, kpss_df, effects_df, _stats_artifacts = (
            export_group_stats_bundle(
                video_summary_df,
                output_dir=stats_output_dir,
                correction_method=str(session.stats_correction or "fdr_bh"),
                include_kpss=bool(session.include_kpss),
                include_mixed_effects=bool(
                    getattr(session, "include_mixed_effects", True)
                ),
                categorical_factors=stats_factors,
                render_plots=False,
                log_fn=self._log,
            )
        )

        figure_manifest_df = pd.DataFrame()
        assay_figure_manifest_df = pd.DataFrame()
        figure_options = self._figure_export_options(session)
        figures_enabled = any(
            bool(figure_options[key])
            for key in (
                "export_publication_figures",
                "export_batch_dashboard",
                "export_group_stats_overview",
                "export_individual_profiles",
                "export_module_archive",
                "generate_video_quicklooks",
            )
        )
        if figures_enabled:
            try:
                _progress("Rendering batch figures from authoritative data...", 99.35)
                from integra_pose.logic.batch_figures import export_batch_figure_bundle

                figure_manifest_df, figure_artifacts = export_batch_figure_bundle(
                    video_summary_df=video_summary_df,
                    omnibus_df=omnibus_df,
                    pairwise_df=pairwise_df,
                    output_dir=output_root / "figures",
                    video_results=video_results,
                    module_tables=module_bundle.tables,
                    precomputed_figure_records=list(precomputed_figure_records or []),
                    assay_preset_key=str(
                        getattr(session, "analytics_assay_preset", "custom") or "custom"
                    ),
                    export_mode=str(figure_options["export_mode"]),
                    export_publication_figures=bool(
                        figure_options["export_publication_figures"]
                    ),
                    export_batch_dashboard=bool(
                        figure_options["export_batch_dashboard"]
                    ),
                    export_group_stats_overview=bool(
                        figure_options["export_group_stats_overview"]
                    ),
                    export_individual_profiles=bool(
                        figure_options["export_individual_profiles"]
                    ),
                    export_module_archive=bool(
                        figure_options["export_module_archive"]
                    ),
                    group_col="group",
                    time_col="time_point",
                )
                if getattr(figure_artifacts, "figure_count", 0):
                    self._log(
                        f"Saved {figure_artifacts.figure_count} batch figure "
                        f"artifacts to {figure_artifacts.output_dir}",
                        "INFO",
                    )
                assay_manifest_path = str(
                    getattr(figure_artifacts, "assay_manifest_csv", "") or ""
                ).strip()
                if assay_manifest_path:
                    assay_path = Path(assay_manifest_path).expanduser()
                    if assay_path.is_file():
                        assay_figure_manifest_df = pd.read_csv(assay_path)
            except Exception as exc:
                self._log(
                    f"Batch figure rendering failed: {exc}\n{traceback.format_exc()}",
                    "WARNING",
                )
        else:
            self._log("Publication figure export disabled for this batch session.", "INFO")

        _progress("Writing batch workbook...", 99.8)
        return write_batch_workbook(
            workbook_path=output_root / "batch_results.xlsx",
            keypoint_df=keypoint_df,
            kinematic_df=kinematic_df,
            bout_df=bout_df,
            video_summary_df=video_summary_df,
            omnibus_df=omnibus_df,
            pairwise_df=pairwise_df,
            kpss_df=kpss_df,
            effects_df=effects_df,
            analysis_coverage_df=analysis_coverage_df,
            figure_manifest_df=figure_manifest_df,
            assay_figure_manifest_df=assay_figure_manifest_df,
            module_file_index_df=module_file_index_df,
            module_table_index_df=module_table_index_df,
        )

    @staticmethod
    def _csv_row_state(path_value: object) -> tuple[str, str]:
        """Return (state, detail) for a review-source CSV."""

        text = str(path_value or "").strip()
        if not text:
            return "absent", ""
        path = Path(text).expanduser()
        if not path.is_file():
            return "missing", str(path)
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return "empty", str(path)
        except Exception as exc:
            return "unreadable", f"{path}: {exc}"
        return ("has_rows" if not frame.empty else "empty"), str(path)

    def _review_completion_issues(
        self,
        session,
        video_results: list[dict],
    ) -> list[str]:
        """Validate that every non-empty review source has authoritative output."""

        policy = str(getattr(session, "review_policy", "after_all") or "after_all").strip().lower()
        if policy == "skip":
            return []

        issues: list[str] = []
        for result in video_results:
            video_label = (
                str(result.get("video_name", "") or "").strip()
                or str(result.get("video_id", "") or "").strip()
                or "video"
            )
            manifest_path = str(result.get("run_manifest_json", "") or "").strip()
            try:
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            except Exception as exc:
                issues.append(f"{video_label}: run manifest is unreadable ({exc})")
                continue
            if not isinstance(manifest, dict):
                issues.append(f"{video_label}: run manifest is not an object")
                continue
            outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
            notes = manifest.get("notes") if isinstance(manifest.get("notes"), dict) else {}

            raw_bouts = (
                outputs.get("raw_detected_bouts_csv")
                or outputs.get("raw_detailed_bouts_csv")
                or result.get("detailed_bouts_csv")
            )
            bout_state, bout_detail = self._csv_row_state(raw_bouts)
            if bout_state in {"missing", "unreadable"}:
                issues.append(f"{video_label}: detected-bout source is {bout_state} ({bout_detail})")
            elif bout_state == "has_rows":
                bout_note = notes.get("bout_review") if isinstance(notes.get("bout_review"), dict) else {}
                reviewed_bouts = (
                    outputs.get("reviewed_bouts_csv")
                    or result.get("reviewed_bouts_csv")
                )
                reviewed_state = self._csv_row_state(reviewed_bouts)[0]
                if (
                    str(bout_note.get("status", "")).strip().lower() != "complete"
                    or reviewed_state
                    not in {"has_rows", "empty"}
                ):
                    issues.append(f"{video_label}: bout review is not finalized")

            raw_roi_events = (
                outputs.get("raw_roi_events_csv")
                or outputs.get("roi_events_csv")
                or result.get("roi_events_csv")
            )
            roi_state, roi_detail = self._csv_row_state(raw_roi_events)
            if roi_state in {"missing", "unreadable"}:
                issues.append(f"{video_label}: ROI-event source is {roi_state} ({roi_detail})")
            elif roi_state == "has_rows":
                roi_note = notes.get("roi_review") if isinstance(notes.get("roi_review"), dict) else {}
                validation_path = str(
                    outputs.get("reviewed_roi_validation_json") or ""
                ).strip()
                validation: dict = {}
                if validation_path and Path(validation_path).expanduser().is_file():
                    try:
                        loaded = json.loads(
                            Path(validation_path).expanduser().read_text(encoding="utf-8")
                        )
                        validation = loaded if isinstance(loaded, dict) else {}
                    except Exception:
                        validation = {}
                if (
                    str(roi_note.get("status", "")).strip().lower() != "complete"
                    or str(validation.get("status", "")).strip().lower()
                    not in {"valid", "valid_empty"}
                ):
                    issues.append(f"{video_label}: ROI review is not finalized")
        return issues

    @staticmethod
    def _invalidated_review_module_warnings(
        video_results: list[dict],
    ) -> list[str]:
        warnings: list[str] = []
        for result in video_results:
            manifest_path = str(result.get("run_manifest_json", "") or "").strip()
            try:
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            except Exception:
                continue
            outputs = (
                manifest.get("outputs")
                if isinstance(manifest, dict)
                and isinstance(manifest.get("outputs"), dict)
                else {}
            )
            invalidated: set[str] = set()
            for key in (
                "invalidated_raw_bout_modules",
                "invalidated_raw_roi_modules",
            ):
                payload = outputs.get(key)
                if isinstance(payload, dict):
                    invalidated.update(str(name) for name in payload)
            if invalidated:
                video_label = (
                    str(result.get("video_name", "") or "").strip()
                    or str(result.get("video_id", "") or "").strip()
                    or "video"
                )
                warnings.append(
                    f"{video_label}: review-dependent optional modules excluded "
                    f"({', '.join(sorted(invalidated))})"
                )
        return warnings

    @staticmethod
    def _finalized_source_kind(session, video_results: list[dict]) -> str:
        if (
            str(getattr(session, "review_policy", "") or "").strip().lower()
            == "skip"
        ):
            return "automatic_outputs_review_explicitly_skipped"
        for result in video_results:
            manifest_path = str(result.get("run_manifest_json", "") or "").strip()
            try:
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            except Exception:
                continue
            outputs = (
                manifest.get("outputs")
                if isinstance(manifest, dict)
                and isinstance(manifest.get("outputs"), dict)
                else {}
            )
            if outputs.get("reviewed_bouts_csv") or outputs.get(
                "reviewed_roi_events_csv"
            ):
                return "authoritative_reviewed_outputs"
        return "automatic_outputs_no_review_required"

    @staticmethod
    def _write_batch_results_status(
        *,
        output_root: Path,
        status: str,
        workbook_path: str = "",
        video_count: int = 0,
        review_policy: str = "",
        source_kind: str = "",
        review_issues: list[str] | None = None,
        warnings: list[str] | None = None,
        reason: str = "",
    ) -> str:
        from integra_pose.utils.safe_io import safe_write_json

        marker_path = output_root / "batch_results_status.json"
        payload = {
            "schema_version": 1,
            "status": str(status),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "workbook_path": str(workbook_path or ""),
            "video_count": int(video_count),
            "review_policy": str(review_policy or ""),
            "source_kind": str(source_kind or ""),
            "review_issues": list(review_issues or []),
            "warnings": list(warnings or []),
            "reason": str(reason or ""),
        }
        safe_write_json(marker_path, payload, indent=2)
        return str(marker_path)

    def mark_results_stale(self, session, *, reason: str) -> str:
        """Mark existing batch derivatives as needing an authoritative rebuild."""

        output_text = str(getattr(session, "output_path", "") or "").strip()
        if not output_text:
            return ""
        output_root = Path(output_text).expanduser().resolve()
        if not output_root.is_dir():
            return ""
        workbook_path = output_root / "batch_results.xlsx"
        marker_path = output_root / "batch_results_status.json"
        if not workbook_path.is_file() and not marker_path.is_file():
            return ""
        active_videos = [
            item
            for item in (getattr(session, "videos", None) or [])
            if not bool(getattr(item, "excluded", False))
        ]
        return self._write_batch_results_status(
            output_root=output_root,
            status="needs_rebuild",
            workbook_path=str(workbook_path) if workbook_path.is_file() else "",
            video_count=len(active_videos),
            review_policy=str(getattr(session, "review_policy", "") or ""),
            source_kind="reviewed_per_video_outputs",
            reason=reason,
        )

    def finalize_reviewed_results(
        self,
        session,
        *,
        stop_event=None,
        progress_callback: Optional[ProgressFn] = None,
    ) -> BatchRunResult:
        """Rebuild batch outputs only after all required reviews are complete."""

        active_videos = [
            item
            for item in (getattr(session, "videos", None) or [])
            if not bool(getattr(item, "excluded", False))
        ]
        total = len(active_videos)
        output_text = str(getattr(session, "output_path", "") or "").strip()
        if not output_text:
            return BatchRunResult(
                OperationStatus.FAILED,
                "Cannot finalize results: the batch output folder is not set.",
                total_count=total,
                failed_count=total,
            )
        output_root = Path(output_text).expanduser().resolve()
        if not output_root.is_dir():
            return BatchRunResult(
                OperationStatus.FAILED,
                f"Cannot finalize results: output folder does not exist: {output_root}",
                total_count=total,
                failed_count=total,
            )
        session_json_path = output_root / "batch_session.json"
        if not active_videos:
            return BatchRunResult(
                OperationStatus.FAILED,
                "Cannot finalize results: there are no included videos.",
                session_json_path=str(session_json_path),
                total_count=0,
            )

        if progress_callback is not None:
            progress_callback("Validating completed videos and reviews...", 5.0)
        video_results: list[dict] = []
        restoration_issues: list[str] = []
        for item in active_videos:
            if stop_event is not None and stop_event.is_set():
                return BatchRunResult(
                    OperationStatus.CANCELLED,
                    "Result finalization cancelled.",
                    session_json_path=str(session_json_path),
                    video_results=video_results,
                    total_count=total,
                    completed_count=len(video_results),
                )
            if (
                str(getattr(item, "inference_status", "")).strip() != "completed"
                or str(getattr(item, "analytics_status", "")).strip() != "completed"
            ):
                restoration_issues.append(
                    f"{getattr(item, 'video_name', 'video')}: inference and analytics are not complete"
                )
                continue
            result, error = self._rehydrate_completed_video_result(item)
            if result is None:
                restoration_issues.append(
                    f"{getattr(item, 'video_name', 'video')}: {error}"
                )
            else:
                video_results.append(result)
        if restoration_issues:
            return BatchRunResult(
                OperationStatus.FAILED,
                "Cannot finalize results:\n- " + "\n- ".join(restoration_issues),
                session_json_path=str(session_json_path),
                video_results=video_results,
                total_count=total,
                completed_count=len(video_results),
                failed_count=len(restoration_issues),
            )

        review_issues = self._review_completion_issues(session, video_results)
        if review_issues:
            self._write_batch_results_status(
                output_root=output_root,
                status="needs_rebuild",
                workbook_path=str(output_root / "batch_results.xlsx"),
                video_count=total,
                review_policy=str(getattr(session, "review_policy", "") or ""),
                source_kind="review_incomplete",
                review_issues=review_issues,
                reason="Required reviews must be completed before finalization.",
            )
            return BatchRunResult(
                OperationStatus.FAILED,
                "Cannot finalize results until required reviews are complete:\n- "
                + "\n- ".join(review_issues),
                workbook_path=str(output_root / "batch_results.xlsx"),
                session_json_path=str(session_json_path),
                video_results=video_results,
                total_count=total,
                completed_count=total,
            )

        try:
            if progress_callback is not None:
                progress_callback("Rebuilding batch results from authoritative outputs...", 20.0)
            workbook_path = self._export_batch_aggregates(
                session=session,
                video_results=video_results,
                output_root=output_root,
                progress_callback=progress_callback,
                precomputed_figure_records=[],
            )
            session.save_json(session_json_path)
            policy = str(getattr(session, "review_policy", "") or "")
            module_warnings = self._invalidated_review_module_warnings(video_results)
            self._write_batch_results_status(
                output_root=output_root,
                status="finalized",
                workbook_path=workbook_path,
                video_count=total,
                review_policy=policy,
                source_kind=self._finalized_source_kind(session, video_results),
                warnings=module_warnings,
            )
            if progress_callback is not None:
                progress_callback("Finalized batch results are ready.", 100.0)
            return BatchRunResult(
                OperationStatus.SUCCESS,
                f"Finalized batch results for {total} video(s).",
                workbook_path=workbook_path,
                session_json_path=str(session_json_path),
                video_results=video_results,
                total_count=total,
                completed_count=total,
            )
        except Exception as exc:
            self._log(
                f"Batch result finalization failed: {exc}\n{traceback.format_exc()}",
                "ERROR",
            )
            try:
                self._write_batch_results_status(
                    output_root=output_root,
                    status="needs_rebuild",
                    workbook_path=str(output_root / "batch_results.xlsx"),
                    video_count=total,
                    review_policy=str(
                        getattr(session, "review_policy", "") or ""
                    ),
                    source_kind="finalization_failed",
                    reason=f"Finalization failed: {exc}",
                )
            except Exception:
                pass
            return BatchRunResult(
                OperationStatus.FAILED,
                f"Batch result finalization failed: {exc}",
                session_json_path=str(session_json_path),
                video_results=video_results,
                total_count=total,
                completed_count=total,
            )

    def _time_threshold_frames(
        self,
        *,
        session,
        item,
        run_dir: Path,
    ) -> tuple[int, int, int, int, float]:
        """Resolve second-based settings against one video's actual FPS."""

        fps = 0.0
        try:
            fps = float(getattr(session, "video_fps", 0.0) or 0.0)
        except (TypeError, ValueError):
            fps = 0.0
        if fps <= 0:
            metadata_path = run_dir / "inference_metadata.json"
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    fps = float(metadata.get("fps") or 0.0)
                except Exception:
                    fps = 0.0
        if fps <= 0:
            fps, _width, _height = self._probe_video_metadata(item.video_path)
        if fps <= 0:
            raise ValueError(
                f"Cannot convert temporal settings from seconds for {item.video_name}: "
                "video FPS could not be resolved."
            )

        def _positive_frames(seconds_value: object, fallback: float) -> int:
            try:
                seconds = float(seconds_value)
            except (TypeError, ValueError):
                seconds = fallback
            if seconds <= 0:
                seconds = fallback
            return max(1, int(math.ceil((seconds * fps) - 1e-12)))

        def _gap_frames(seconds_value: object, fallback: float) -> int:
            try:
                seconds = float(seconds_value)
            except (TypeError, ValueError):
                seconds = fallback
            seconds = max(0.0, seconds)
            return max(0, int(math.floor((seconds * fps) + 1e-12)))

        return (
            _positive_frames(getattr(session, "min_bout_seconds", 0.10), 0.10),
            _gap_frames(getattr(session, "max_gap_seconds", 0.17), 0.17),
            _positive_frames(
                getattr(session, "roi_min_dwell_seconds", 0.10), 0.10
            ),
            _gap_frames(
                getattr(session, "roi_max_gap_seconds", 0.17), 0.17
            ),
            float(fps),
        )

    def _run_video_analytics(
        self,
        *,
        session,
        item,
        run_root: Path,
        run_dir: Path,
        labels_dir: Path,
        labels_csv: Path,
        metrics_csv: Path,
        metrics_track_csv: Path,
        metrics_frame_csv: Path,
        yaml_path: Path,
        labels_root: Path | None,
        use_existing_labels: bool,
        stop_event,
    ) -> dict | None:
        try:
            if stop_event.is_set():
                item.analytics_status = "cancelled"
                item.status_message = "Analytics cancelled."
                return None

            def _resolve_frame_threshold(
                session_attr: str,
                fallback: int,
                *,
                gui_attr: str = "",
            ) -> int:
                raw_value = getattr(session, session_attr, None)
                if raw_value is None and gui_attr:
                    try:
                        gui_var = getattr(self.app.config.analytics, gui_attr)
                        raw_value = gui_var.get()
                    except Exception:
                        raw_value = None
                if raw_value is None:
                    raw_value = fallback
                try:
                    return int(raw_value)
                except (TypeError, ValueError):
                    return int(fallback)

            min_bout_frames = _resolve_frame_threshold(
                "min_bout_frames",
                5,
                gui_attr="min_bout_duration_var",
            )
            max_gap_frames = _resolve_frame_threshold(
                "max_gap_frames",
                2,
                gui_attr="max_frame_gap_var",
            )
            roi_min_dwell_frames = _resolve_frame_threshold(
                "roi_min_dwell_frames",
                min_bout_frames,
            )
            roi_max_gap_frames = _resolve_frame_threshold(
                "roi_max_gap_frames",
                max_gap_frames,
            )
            temporal_threshold_unit = str(
                getattr(session, "temporal_threshold_unit", "frames") or "frames"
            ).strip().lower()
            temporal_threshold_fps = 0.0
            if temporal_threshold_unit == "seconds":
                (
                    min_bout_frames,
                    max_gap_frames,
                    roi_min_dwell_frames,
                    roi_max_gap_frames,
                    temporal_threshold_fps,
                ) = self._time_threshold_frames(
                    session=session,
                    item=item,
                    run_dir=run_dir,
                )
            else:
                temporal_threshold_unit = "frames"

            rois_for_video = self._choose_rois_for_video(session, item)
            object_rois_for_video = self._choose_object_rois_for_video(session, item)
            analytics_out_dir = run_root / "analytics"
            label_class_metadata = load_frame_label_class_metadata(labels_dir)
            saved_class_names = list(label_class_metadata.get("class_names") or [])
            if saved_class_names:
                behavior_names_override = saved_class_names
                behavior_names_source = "inference label metadata"
            else:
                behavior_names_override = list(
                    getattr(session.model_capabilities, "class_names", []) or []
                )
                behavior_names_source = (
                    "model class names" if behavior_names_override else "unresolved"
                )
            params = {
                "yolo_folder": str(labels_dir),
                "video_file": str(item.video_path),
                "yaml_file": str(yaml_path),
                "behavior_names_override": behavior_names_override,
                "behavior_names_source": behavior_names_source,
                "keypoint_names_override": list(getattr(session, "keypoint_names", []) or []),
                "keypoint_names_source": str(getattr(session, "keypoint_names_source", "unresolved") or "unresolved"),
                "keypoint_schema_path": str(getattr(session, "keypoint_schema_path", "") or ""),
                "min_bout_frames": min_bout_frames,
                "max_gap_frames": max_gap_frames,
                "behavior_bout_class_mode": str(
                    getattr(
                        session,
                        "behavior_bout_class_mode",
                        "mutually_exclusive",
                    )
                    or "mutually_exclusive"
                ),
                "roi_min_dwell_frames": max(1, int(roi_min_dwell_frames if roi_min_dwell_frames is not None else min_bout_frames)),
                "roi_max_gap_frames": max(0, int(roi_max_gap_frames if roi_max_gap_frames is not None else max_gap_frames)),
                "temporal_threshold_unit": temporal_threshold_unit,
                "configured_min_bout_seconds": float(
                    getattr(session, "min_bout_seconds", 0.0) or 0.0
                ),
                "configured_max_gap_seconds": float(
                    getattr(session, "max_gap_seconds", 0.0) or 0.0
                ),
                "configured_roi_min_dwell_seconds": float(
                    getattr(session, "roi_min_dwell_seconds", 0.0) or 0.0
                ),
                "configured_roi_max_gap_seconds": float(
                    getattr(session, "roi_max_gap_seconds", 0.0) or 0.0
                ),
                "temporal_threshold_fps": temporal_threshold_fps,
                "create_video": bool(session.save_annotated_video),
                "output_folder_override": str(analytics_out_dir),
                "use_rois_override": bool(rois_for_video),
                "roi_polygon_map_override": rois_for_video,
                "roi_event_mode": str(session.roi_event_mode or "bbox_only"),
                "roi_entry_threshold": float(session.roi_entry_threshold),
                "roi_exit_threshold": float(session.roi_exit_threshold),
                "keypoint_entry_index": int(session.keypoint_entry_index),
                "keypoint_entry_indices": [int(idx) for idx in (session.keypoint_entry_indices or [])],
                "keypoint_entry_ratio_threshold": float(session.keypoint_entry_ratio_threshold),
                "object_interaction_enabled": bool(getattr(session, "object_interaction_enabled", False)),
                "object_count": int(getattr(session, "object_count", 0) or 0),
                "object_roi_size_px": int(getattr(session, "object_roi_size_px", 20) or 20),
                "object_roi_shape": str(getattr(session, "object_roi_shape", "circle") or "circle"),
                "object_interaction_keypoint_index": int(getattr(session, "object_interaction_keypoint_index", 0) or 0),
                "object_interaction_distance_px": float(getattr(session, "object_interaction_distance_px", 0.0) or 0.0),
                # The wizard exposes the same debounce controls for all ROI
                # events, so object and zone visits must use identical temporal
                # semantics.
                "object_min_dwell_frames": max(1, int(roi_min_dwell_frames)),
                "object_max_gap_frames": max(0, int(roi_max_gap_frames)),
                "object_roi_polygon_map_override": object_rois_for_video,
                "use_existing_labels": bool(use_existing_labels),
                "existing_labels_root": str(labels_root or ""),
                "single_animal_mode_override": bool(session.single_animal_mode),
                "assay_preset_override": str(getattr(session, "analytics_assay_preset", "custom") or "custom"),
                "enabled_metrics_override": list(getattr(session, "analytics_enabled_metrics", []) or []),
                "enabled_modules_override": list(getattr(session, "analytics_enabled_modules", []) or []),
                "user_video_fps_override": float(getattr(session, "video_fps", 0.0) or 0.0),
                # Study provenance flows into the analytics manifest's
                # `provenance` block (schema v2). Empty strings outside batch
                # context. Tab 7 keys auto-split on these.
                "provenance_subject_id": str(getattr(item, "subject_id", "") or "").strip(),
                "provenance_group": str(getattr(item, "group", "") or "").strip(),
                "provenance_time_point": str(getattr(item, "time_point", "") or "").strip(),
                "stop_event": stop_event,
            }

            try:
                (
                    detailed_df,
                    _summary_df,
                    _excel_path,
                    roi_metrics,
                    output_folder,
                    _base_name,
                    _use_rois,
                    _roi_events,
                    _module_outputs,
                ) = self.app.analytics.run_analysis(params)
            except BoutAnalysisCancelledError:
                item.analytics_status = "cancelled"
                item.status_message = "Analytics cancelled."
                return None

            if stop_event.is_set():
                item.analytics_status = "cancelled"
                item.status_message = "Analytics cancelled."
                return None

            detailed_csv = Path(output_folder) / f"{Path(item.video_path).stem}_detailed_bouts.csv"
            summary_csv = Path(output_folder) / f"{Path(item.video_path).stem}_summary.csv"
            roi_overview_csv = ""
            roi_concurrent_overview_csv = ""
            roi_dwell_events_csv = ""
            object_interactions_csv = ""
            object_dwell_events_csv = ""
            if isinstance(roi_metrics, dict):
                files = roi_metrics.get("files")
                if isinstance(files, dict):
                    roi_overview_csv = str(
                        files.get("exclusive_entries_exits", "") or files.get("entries_exits", "") or ""
                    )
                    roi_concurrent_overview_csv = str(files.get("entries_exits", "") or "")
                    roi_dwell_events_csv = str(
                        files.get("exclusive_dwell_events", "") or files.get("dwell_events", "") or ""
                    )
            if isinstance(_module_outputs, dict):
                object_module = _module_outputs.get("object_interactions")
                if isinstance(object_module, dict):
                    files = object_module.get("files")
                    if isinstance(files, dict):
                        object_interactions_csv = str(files.get("summary", "") or files.get("per_frame", "") or "")
                        object_dwell_events_csv = str(files.get("dwell_events", "") or "")
            analytics_annotated_video_path = ""
            if params["create_video"] and not stop_event.is_set():
                analytics_video_path = Path(output_folder) / f"{_base_name}_annotated.mp4"
                object_metrics, object_events = video_creator.load_object_overlay_metrics(_module_outputs)
                try:
                    analytics_video_ok = bool(
                        video_creator.create_annotated_video(
                            str(item.video_path),
                            str(analytics_video_path),
                            str(labels_dir),
                            rois_for_video,
                            detailed_df,
                            roi_metrics=roi_metrics,
                            roi_events=_roi_events if _use_rois else None,
                            object_rois=object_rois_for_video,
                            object_metrics=object_metrics,
                            object_events=object_events,
                            single_animal_mode=bool(
                                params.get("single_animal_mode_override", False)
                            ),
                        )
                    )
                except Exception as exc:
                    analytics_video_ok = False
                    self._log(
                        f"Batch analytics video render failed for {item.video_name}: {exc}\n{traceback.format_exc()}",
                        "ERROR",
                    )
                if analytics_video_ok and analytics_video_path.is_file():
                    analytics_annotated_video_path = str(analytics_video_path)
                elif params["create_video"]:
                    self._log(
                        f"Batch analytics video render failed for {item.video_name}; CSV outputs were still saved.",
                        "WARNING",
                    )

            item.analytics_status = "completed"
            item.analytics_output_dir = str(output_folder)
            item.detailed_bouts_csv = str(detailed_csv) if detailed_csv.is_file() else ""
            item.summary_bouts_csv = str(summary_csv) if summary_csv.is_file() else ""
            roi_events_csv = Path(output_folder) / f"{Path(item.video_path).stem}_roi_events.csv"
            manifest_json = Path(output_folder) / "run_manifest.json"
            if session.review_policy == "skip":
                item.review_status = "skipped"
                item.bout_review_status = "skipped"
                item.roi_review_status = "skipped"
            else:
                bout_review_needed = not detailed_df.empty
                roi_review_needed = (
                    self._csv_row_state(roi_events_csv)[0] == "has_rows"
                )
                item.bout_review_status = "queued" if bout_review_needed else "not_required"
                item.roi_review_status = "queued" if roi_review_needed else "not_required"
                review_needed = bout_review_needed or roi_review_needed
                item.review_status = "queued" if review_needed else "not_required"
            item.status_message = "Completed"

            return {
                "video_id": item.video_id,
                "video_name": item.video_name,
                "video_path": item.video_path,
                "group": item.group,
                "subject_id": item.subject_id,
                "time_point": item.time_point,
                "run_output_dir": str(run_dir),
                "yolo_output_dir": str(labels_dir),
                "analytics_output_dir": str(output_folder),
                "labels_csv": str(labels_csv) if labels_csv.is_file() else "",
                "metrics_csv": str(metrics_csv) if metrics_csv.is_file() else "",
                "metrics_summary_by_track_csv": str(metrics_track_csv) if metrics_track_csv.is_file() else "",
                "metrics_summary_by_frame_csv": str(metrics_frame_csv) if metrics_frame_csv.is_file() else "",
                "detailed_bouts_csv": item.detailed_bouts_csv,
                "summary_bouts_csv": item.summary_bouts_csv,
                "roi_events_csv": str(roi_events_csv) if roi_events_csv.is_file() else "",
                "roi_overview_csv": roi_overview_csv,
                "roi_concurrent_overview_csv": roi_concurrent_overview_csv,
                "roi_dwell_events_csv": roi_dwell_events_csv,
                "object_interactions_csv": object_interactions_csv,
                "object_dwell_events_csv": object_dwell_events_csv,
                "roi_polygons": rois_for_video,
                "object_roi_polygons": object_rois_for_video,
                "object_interaction_distance_px": float(getattr(session, "object_interaction_distance_px", 0.0) or 0.0),
                "analytics_annotated_video_path": analytics_annotated_video_path,
                "run_manifest_json": str(manifest_json) if manifest_json.is_file() else "",
            }
        except Exception as exc:
            item.analytics_status = "failed"
            item.status_message = "Analytics failed."
            self._log(
                f"Batch video {item.video_name} analytics failed: {exc}\n{traceback.format_exc()}",
                "ERROR",
            )
            return None

    def run(
        self,
        session,
        *,
        stop_event,
        progress_callback: Optional[ProgressFn] = None,
        review_callback: Optional[Callable[[object], None]] = None,
        resume: bool = False,
    ) -> BatchRunResult:
        """Execute the per-video batch pipeline.

        ``resume=True`` skips any video whose ``inference_status`` and
        ``analytics_status`` are both ``"completed"`` from a prior run, so a
        50-video batch that crashed at video 30 picks up at video 30 instead
        of starting over. Failed / cancelled / pending videos are reprocessed
        normally. Pass ``resume=False`` for a clean re-run.
        """
        last_progress_value = 0.0

        def _progress(message: str, ratio: float) -> None:
            if progress_callback is None:
                return
            nonlocal last_progress_value
            safe_ratio = max(last_progress_value, float(ratio))
            last_progress_value = safe_ratio
            progress_callback(message, safe_ratio)

        output_root = Path(session.output_path).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        yaml_path = self._resolve_yaml_path(session, output_root)
        session.yaml_path = str(yaml_path)
        session_json_path = output_root / "batch_session.json"

        # Pin random seeds before any pipeline stage that touches RNGs
        # (model init, augmentation, tracker matching). Sourced from the
        # project's Seeds section so runs are reproducible across machines.
        try:
            seeds_config = self.app.config.seeds.to_seeds_config()
        except (AttributeError, TypeError):
            from integra_pose.utils.seeds import SeedsConfig as _SC

            seeds_config = _SC()
        from integra_pose.utils.seeds import apply_global_seed as _apply_global_seed

        _apply_global_seed(seeds_config, log_fn=self._log)
        self._active_seeds_snapshot = seeds_config.to_dict()
        use_existing_labels = bool(getattr(session, "use_existing_labels", False))
        labels_root_text = str(getattr(session, "existing_labels_root", "") or "").strip()
        labels_root = Path(labels_root_text).expanduser().resolve() if labels_root_text else None
        indexed_label_dirs: list[tuple[Path, list[str]]] | None = None
        if use_existing_labels:
            if labels_root is None or not labels_root.is_dir():
                raise ValueError("Existing labels mode is enabled but labels root directory is invalid or missing.")
            indexed_label_dirs = self._index_existing_label_dirs(labels_root)
            if not indexed_label_dirs:
                self._log(f"No label .txt files were found under existing labels root: {labels_root}", "WARNING")

        active_videos = [item for item in (session.videos or []) if not bool(getattr(item, "excluded", False))]
        total = max(1, len(active_videos))
        video_results: list[dict] = []
        cancelled = False
        figure_options = self._figure_export_options(session)
        quicklook_enabled = bool(figure_options["generate_video_quicklooks"])
        analytics_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch-analytics")
        analytics_futures: list[tuple[int, object, Future[dict | None]]] = []
        figure_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch-figures")
        figure_futures: list[Future[list[dict[str, str]]]] = []
        precomputed_figure_records: list[dict[str, str]] = []
        render_video_quicklook_bundle = None
        if quicklook_enabled:
            try:
                from integra_pose.logic.batch_figures import render_video_quicklook_bundle
            except Exception as exc:
                render_video_quicklook_bundle = None
                self._log(f"Batch per-video quicklook figure worker unavailable: {exc}", "WARNING")

        def _collect_done_figure_futures(*, wait_all: bool = False) -> None:
            nonlocal figure_futures, precomputed_figure_records
            pending: list[Future[list[dict[str, str]]]] = []
            for future in figure_futures:
                if not wait_all and not future.done():
                    pending.append(future)
                    continue
                try:
                    rows = future.result()
                except Exception as exc:
                    self._log(f"Batch per-video figure job failed: {exc}\n{traceback.format_exc()}", "WARNING")
                    continue
                if rows:
                    precomputed_figure_records.extend(rows)
            figure_futures = pending

        try:
            if not active_videos:
                session.save_json(session_json_path)
                return BatchRunResult(
                    status=OperationStatus.FAILED,
                    message="No included videos queued.",
                    session_json_path=str(session_json_path),
                    video_results=[],
                    total_count=0,
                )

            resumed_skip_count = 0
            for index, item in enumerate(active_videos, start=1):
                if stop_event.is_set():
                    cancelled = True
                    break

                if (
                    resume
                    and str(getattr(item, "inference_status", "")).strip() == "completed"
                    and str(getattr(item, "analytics_status", "")).strip() == "completed"
                ):
                    prior_result, resume_error = self._rehydrate_completed_video_result(item)
                    if prior_result is not None:
                        resumed_skip_count += 1
                        video_results.append(prior_result)
                        self._log(
                            f"Resume: reusing already-completed video {index}/{total}: "
                            f"{item.video_name}",
                            "INFO",
                        )
                        continue
                    self._log(
                        f"Resume: prior outputs for {item.video_name} cannot be reused "
                        f"({resume_error}); reprocessing this video.",
                        "WARNING",
                    )

                item.inference_status = "running"
                item.analytics_status = "pending"
                item.status_message = "Inference running"
                phase_start = ((index - 1) / total) * 100.0
                phase_end = ((index - 0.5) / total) * 100.0
                _progress(f"Processing video {index}/{total}: inference", phase_start)

                video_stem = Path(item.video_path).stem
                run_root = output_root / "videos" / f"{item.video_id}_{video_stem}"
                infer_project = run_root / "inference"
                infer_project.mkdir(parents=True, exist_ok=True)
                run_name = "infer"
                if use_existing_labels:
                    item.status_message = "Reusing existing labels"
                    _progress(f"Processing video {index}/{total}: resolving existing labels", phase_start)
                    labels_dir = self._find_existing_labels_dir(
                        video_stem=video_stem,
                        preferred_dir=str(item.yolo_output_dir or "").strip() or None,
                        labels_root=labels_root,
                        indexed_dirs=indexed_label_dirs,
                    )
                    if labels_dir is None or not labels_dir.is_dir():
                        item.inference_status = "failed"
                        item.analytics_status = "failed"
                        item.status_message = "Existing labels not found."
                        self._log(
                            f"Batch video {item.video_name} failed: could not locate existing labels for '{video_stem}' under {labels_root}",
                            "ERROR",
                        )
                        continue
                    labels_ok, labels_error = self._validate_existing_labels_dir(
                        labels_dir=labels_dir,
                        video_path=item.video_path,
                        labels_root=labels_root,
                        video_stem=video_stem,
                        preferred_dir=str(item.yolo_output_dir or "").strip() or None,
                    )
                    if not labels_ok:
                        item.inference_status = "failed"
                        item.analytics_status = "failed"
                        item.status_message = "Existing labels failed validation."
                        self._log(f"Batch video {item.video_name} failed: {labels_error}", "ERROR")
                        continue
                    run_dir = labels_dir.parent if labels_dir.name.lower() == "labels" else labels_dir
                    labels_csv = labels_dir / "labels.csv"
                    if not labels_csv.is_file():
                        alt_csv = run_dir / "labels.csv"
                        if alt_csv.is_file():
                            labels_csv = alt_csv
                    settings = self._build_inference_settings(session, item.video_path, infer_project, run_name)
                    metadata_path = run_dir / "inference_metadata.json"
                    frame_width = 0
                    frame_height = 0
                    fps = 0.0
                    if metadata_path.is_file():
                        try:
                            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                            frame_width = int(metadata.get("frame_width") or 0)
                            frame_height = int(metadata.get("frame_height") or 0)
                            fps = float(metadata.get("fps") or 0.0)
                        except Exception:
                            frame_width = 0
                            frame_height = 0
                            fps = 0.0
                    if not (frame_width and frame_height):
                        fps_probe, frame_width, frame_height = self._probe_video_metadata(item.video_path)
                        if fps <= 0:
                            fps = fps_probe
                    if fps <= 0:
                        from integra_pose.utils.fps_resolver import resolve_fps, FpsUnavailableError

                        user_fps = float(getattr(session, "video_fps", 0.0) or 0.0)
                        try:
                            fps, _fps_source = resolve_fps(
                                item.video_path,
                                user_fps if user_fps > 0 else "",
                                log_fn=self._log,
                                workflow_label="the Batch Wizard's Video FPS field",
                            )
                        except FpsUnavailableError as exc:
                            item.inference_status = "failed"
                            item.analytics_status = "failed"
                            item.status_message = str(exc)
                            self._log(str(exc), "ERROR")
                            continue
                    if stop_event.is_set():
                        cancelled = True
                        item.inference_status = "cancelled"
                        item.analytics_status = "cancelled"
                        item.status_message = "Batch stop requested."
                        break
                    if labels_csv.is_file():
                        metrics_csv, metrics_track_csv, metrics_frame_csv = self._build_metrics_from_labels(
                            settings=settings,
                            labels_csv_path=labels_csv,
                            run_dir=run_dir,
                            frame_width=frame_width,
                            frame_height=frame_height,
                            fps=fps,
                        )
                    else:
                        metrics_csv = run_dir / "metrics.csv"
                        if not metrics_csv.is_file():
                            metrics_csv = labels_dir / "metrics.csv"
                        metrics_track_csv = run_dir / "metrics_summary_by_track.csv"
                        if not metrics_track_csv.is_file():
                            metrics_track_csv = labels_dir / "metrics_summary_by_track.csv"
                        metrics_frame_csv = run_dir / "metrics_summary_by_frame.csv"
                        if not metrics_frame_csv.is_file():
                            metrics_frame_csv = labels_dir / "metrics_summary_by_frame.csv"
                    item.inference_status = "completed"
                    item.yolo_output_dir = str(labels_dir)
                    item.metrics_csv = str(metrics_csv) if metrics_csv.is_file() else ""
                    item.run_output_dir = str(run_dir)
                    if stop_event.is_set():
                        cancelled = True
                        item.analytics_status = "cancelled"
                        item.status_message = "Batch stop requested."
                        break
                    item.status_message = "Analytics running"
                    future = analytics_executor.submit(
                        self._run_video_analytics,
                        session=session,
                        item=item,
                        run_root=run_root,
                        run_dir=run_dir,
                        labels_dir=labels_dir,
                        labels_csv=labels_csv,
                        metrics_csv=metrics_csv,
                        metrics_track_csv=metrics_track_csv,
                        metrics_frame_csv=metrics_frame_csv,
                        yaml_path=yaml_path,
                        labels_root=labels_root,
                        use_existing_labels=use_existing_labels,
                        stop_event=stop_event,
                    )
                    analytics_futures.append((index, item, future))
                else:
                    settings = self._build_inference_settings(session, item.video_path, infer_project, run_name)
                    total_frames = self._probe_total_frames(item.video_path)
                    last_progress_emit = 0.0

                    def _native_progress(frame_index: int) -> None:
                        nonlocal last_progress_emit
                        now = time.perf_counter()
                        # Throttle UI updates to avoid flooding Tk callbacks.
                        if now - last_progress_emit < 0.4:
                            return
                        last_progress_emit = now

                        frame_number = frame_index + 1

                        if total_frames > 0:
                            frac = min(0.995, frame_number / float(total_frames))
                            ratio = phase_start + (phase_end - phase_start) * frac
                            _progress(
                                f"Processing video {index}/{total}: inference frame {frame_number:,}/{total_frames:,}",
                                ratio,
                            )
                        else:
                            _progress(
                                f"Processing video {index}/{total}: inference frame {frame_number:,}",
                                phase_start,
                            )

                    try:
                        native_result = self._run_native_inference(
                            settings,
                            stop_event=stop_event,
                            progress_fn=_native_progress,
                        )
                    except Exception as exc:
                        item.inference_status = "failed"
                        item.analytics_status = "failed"
                        item.status_message = f"Inference failed: {exc}"
                        self._log(
                            f"Batch video {item.video_name} inference failed: {exc}\n{traceback.format_exc()}",
                            "ERROR",
                        )
                        continue
                    run_dir = Path(native_result.get("run_dir") or infer_project)
                    labels_dir = run_dir / "labels"
                    labels_csv = labels_dir / "labels.csv"
                    metadata_path = run_dir / "inference_metadata.json"
                    frame_width = 0
                    frame_height = 0
                    if metadata_path.is_file():
                        try:
                            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                            fps = float(metadata.get("fps") or 0.0)
                            frame_width = int(metadata.get("frame_width") or 0)
                            frame_height = int(metadata.get("frame_height") or 0)
                        except Exception:
                            fps = 0.0
                            frame_width = 0
                            frame_height = 0
                    else:
                        fps = 0.0
                    if not (frame_width and frame_height):
                        fps_probe, frame_width, frame_height = self._probe_video_metadata(item.video_path)
                        if fps <= 0:
                            fps = fps_probe
                    if fps <= 0:
                        from integra_pose.utils.fps_resolver import resolve_fps, FpsUnavailableError

                        user_fps = float(getattr(session, "video_fps", 0.0) or 0.0)
                        try:
                            fps, _fps_source = resolve_fps(
                                item.video_path,
                                user_fps if user_fps > 0 else "",
                                log_fn=self._log,
                                workflow_label="the Batch Wizard's Video FPS field",
                            )
                        except FpsUnavailableError as exc:
                            item.inference_status = "failed"
                            item.analytics_status = "failed"
                            item.status_message = str(exc)
                            self._log(str(exc), "ERROR")
                            continue
                    if stop_event.is_set():
                        cancelled = True
                        item.inference_status = "cancelled"
                        item.analytics_status = "cancelled"
                        item.status_message = "Batch stop requested."
                        break
                    metrics_csv, metrics_track_csv, metrics_frame_csv = self._build_metrics_from_labels(
                        settings=settings,
                        labels_csv_path=labels_csv,
                        run_dir=run_dir,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        fps=fps,
                    )

                    if not labels_dir.exists():
                        item.inference_status = "failed"
                        item.analytics_status = "failed"
                        item.status_message = "Missing labels output from inference."
                        self._log(f"Batch video {item.video_name} failed: labels folder missing at {labels_dir}", "ERROR")
                        continue

                    item.inference_status = "completed"
                    item.yolo_output_dir = str(labels_dir)
                    item.metrics_csv = str(metrics_csv) if metrics_csv.is_file() else ""
                    item.run_output_dir = str(run_dir)
                    if stop_event.is_set():
                        cancelled = True
                        item.analytics_status = "cancelled"
                        item.status_message = "Batch stop requested."
                        break
                    item.status_message = "Analytics running"
                    future = analytics_executor.submit(
                        self._run_video_analytics,
                        session=session,
                        item=item,
                        run_root=run_root,
                        run_dir=run_dir,
                        labels_dir=labels_dir,
                        labels_csv=labels_csv,
                        metrics_csv=metrics_csv,
                        metrics_track_csv=metrics_track_csv,
                        metrics_frame_csv=metrics_frame_csv,
                        yaml_path=yaml_path,
                        labels_root=labels_root,
                        use_existing_labels=use_existing_labels,
                        stop_event=stop_event,
                    )
                    analytics_futures.append((index, item, future))

                pending_futures: list[tuple[int, object, Future[dict | None]]] = []
                for submitted_index, submitted_item, future in analytics_futures:
                    if not future.done():
                        pending_futures.append((submitted_index, submitted_item, future))
                        continue
                    result_payload = future.result()
                    if result_payload:
                        if str(getattr(session, "review_policy", "")).strip().lower() == "after_each" and review_callback is not None:
                            review_callback(submitted_item)
                            rehydrated, rehydrate_error = (
                                self._rehydrate_completed_video_result(submitted_item)
                            )
                            if rehydrated is not None:
                                result_payload = rehydrated
                            elif rehydrate_error:
                                self._log(
                                    f"Could not refresh reviewed outputs for "
                                    f"{submitted_item.video_name}: {rehydrate_error}",
                                    "WARNING",
                                )
                        video_results.append(result_payload)
                        session.save_json(session_json_path)
                        if render_video_quicklook_bundle is not None and not stop_event.is_set():
                            figure_futures.append(
                                figure_executor.submit(
                                    render_video_quicklook_bundle,
                                    result_payload,
                                    output_dir=output_root / "figures" / "individual_quicklook",
                                )
                            )
                    _progress(f"Completed analytics for video {submitted_index}/{total}", (submitted_index / total) * 100.0)
                analytics_futures = pending_futures
                _collect_done_figure_futures()
                if stop_event.is_set():
                    cancelled = True
                    break

            session.save_json(session_json_path)

            if stop_event.is_set():
                cancelled = True
            if analytics_futures:
                wait_message = "Waiting for running analytics job to stop..." if cancelled else "Waiting for queued analytics jobs to finish..."
                _progress(wait_message, max(last_progress_value, 99.0))
            for submitted_index, submitted_item, future in analytics_futures:
                if cancelled and not future.running():
                    future.cancel()
                    continue
                result_payload = future.result()
                if result_payload:
                    if str(getattr(session, "review_policy", "")).strip().lower() == "after_each" and review_callback is not None:
                        review_callback(submitted_item)
                        rehydrated, rehydrate_error = (
                            self._rehydrate_completed_video_result(submitted_item)
                        )
                        if rehydrated is not None:
                            result_payload = rehydrated
                        elif rehydrate_error:
                            self._log(
                                f"Could not refresh reviewed outputs for "
                                f"{submitted_item.video_name}: {rehydrate_error}",
                                "WARNING",
                            )
                    video_results.append(result_payload)
                    session.save_json(session_json_path)
                    if render_video_quicklook_bundle is not None and not stop_event.is_set():
                        figure_futures.append(
                            figure_executor.submit(
                                render_video_quicklook_bundle,
                                result_payload,
                                output_dir=output_root / "figures" / "individual_quicklook",
                            )
                        )
                _progress(f"Completed analytics for video {submitted_index}/{total}", (submitted_index / total) * 100.0)
            _collect_done_figure_futures()

            if stop_event.is_set():
                cancelled = True

            if any(
                str(getattr(item, "inference_status", "")).strip() == "cancelled"
                or str(getattr(item, "analytics_status", "")).strip() == "cancelled"
                for item in active_videos
            ):
                cancelled = True

            if cancelled:
                completed_count, failed_count = self._run_counts(active_videos)
                return BatchRunResult(
                    status=OperationStatus.CANCELLED,
                    message=(
                        f"Batch run cancelled after {completed_count}/{len(active_videos)} video(s) completed"
                        + (f"; {failed_count} failed." if failed_count else ".")
                    ),
                    session_json_path=str(session_json_path),
                    video_results=video_results,
                    total_count=len(active_videos),
                    completed_count=completed_count,
                    failed_count=failed_count,
                    resumed_count=resumed_skip_count,
                )

            # Every included video must have a terminal state before a
            # non-cancelled run can be classified. Treat an unexplained
            # unfinished state as a failure rather than allowing a success
            # result with missing cohort rows.
            for item in active_videos:
                inference_status = str(getattr(item, "inference_status", "")).strip()
                analytics_status = str(getattr(item, "analytics_status", "")).strip()
                if inference_status == "completed" and analytics_status == "completed":
                    continue
                if inference_status == "failed" or analytics_status == "failed":
                    continue
                if inference_status == "completed":
                    item.analytics_status = "failed"
                else:
                    item.inference_status = "failed"
                    item.analytics_status = "failed"
                item.status_message = "Batch processing did not reach a completed state."
                self._log(
                    f"Batch video {item.video_name} did not reach a terminal completed state; marking it failed.",
                    "ERROR",
                )

            if figure_futures:
                _progress("Finishing per-video figure jobs...", max(last_progress_value, 99.1))
                _collect_done_figure_futures(wait_all=True)

            workbook_file = self._export_batch_aggregates(
                session=session,
                video_results=video_results,
                output_root=output_root,
                progress_callback=_progress,
                precomputed_figure_records=precomputed_figure_records,
            )

            if resume and resumed_skip_count:
                self._log(
                    f"Resume: skipped {resumed_skip_count} already-completed video(s); "
                    f"processed {len(active_videos) - resumed_skip_count} new/incomplete video(s).",
                    "INFO",
                )
            completed_count, failed_count = self._run_counts(active_videos)
            if completed_count == len(active_videos) and failed_count == 0:
                run_status = OperationStatus.SUCCESS
                completion_message = f"Batch run completed: {completed_count}/{len(active_videos)} video(s) succeeded."
            elif completed_count > 0:
                run_status = OperationStatus.PARTIAL
                completion_message = (
                    f"Batch run partially completed: {completed_count}/{len(active_videos)} video(s) succeeded; "
                    f"{failed_count} failed."
                )
            else:
                run_status = OperationStatus.FAILED
                completion_message = (
                    f"Batch run failed: no videos completed; {failed_count}/{len(active_videos)} video(s) failed."
                )
            if resume and resumed_skip_count:
                completion_message = (
                    f"{completion_message.rstrip('.')} "
                    f"({resumed_skip_count} restored from prior completed outputs)."
                )
            review_issues = self._review_completion_issues(session, video_results)
            module_warnings = self._invalidated_review_module_warnings(video_results)
            aggregate_is_final = (
                run_status is OperationStatus.SUCCESS and not review_issues
            )
            self._write_batch_results_status(
                output_root=output_root,
                status="finalized" if aggregate_is_final else "draft",
                workbook_path=workbook_file,
                video_count=len(video_results),
                review_policy=str(getattr(session, "review_policy", "") or ""),
                source_kind=(
                    self._finalized_source_kind(session, video_results)
                    if aggregate_is_final
                    else "automatic_pre_review_outputs"
                ),
                review_issues=review_issues,
                warnings=module_warnings,
                reason=(
                    ""
                    if aggregate_is_final
                    else "Complete required reviews, then finalize to rebuild batch derivatives."
                ),
            )
            if run_status is OperationStatus.SUCCESS and review_issues:
                completion_message = (
                    f"{completion_message} The current workbook is a draft based on "
                    "automatic detections; complete reviews and choose Finalize "
                    "Reviewed Results to rebuild it."
                )
            return BatchRunResult(
                status=run_status,
                message=completion_message,
                workbook_path=workbook_file,
                session_json_path=str(session_json_path),
                video_results=video_results,
                total_count=len(active_videos),
                completed_count=completed_count,
                failed_count=failed_count,
                resumed_count=resumed_skip_count,
            )
        except Exception as exc:
            self._log(f"Batch pipeline failed: {exc}\n{traceback.format_exc()}", "ERROR")
            session.save_json(session_json_path)
            completed_count, failed_count = self._run_counts(active_videos)
            return BatchRunResult(
                status=OperationStatus.FAILED,
                message=f"Batch run failed: {exc}",
                workbook_path="",
                session_json_path=str(session_json_path),
                video_results=video_results,
                total_count=len(active_videos),
                completed_count=completed_count,
                failed_count=max(failed_count, len(active_videos) - completed_count),
                resumed_count=resumed_skip_count,
            )
        finally:
            analytics_executor.shutdown(wait=True, cancel_futures=cancelled)
            figure_executor.shutdown(wait=not cancelled, cancel_futures=cancelled)

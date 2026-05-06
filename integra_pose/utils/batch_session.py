"""Batch session models and serialization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Optional
import uuid

# PR 2B coherence-pass 2e: bumped from 3 → 4 when relative-path storage
# (project root anchored) was introduced. Old (v≤3) sessions load
# transparently because absolute paths pass through resolve_storage_path
# unchanged.
CURRENT_SCHEMA_VERSION = 4

# Top-level path-bearing fields on BatchSession. Extracted so save / load
# share the same source of truth — adding a new path field means adding
# its name here once, not patching three places.
_SESSION_PATH_FIELDS = (
    "source_path",
    "model_path",
    "output_path",
    "yaml_path",
    "tracker_config_path",
    "existing_labels_root",
    "keybind_profile_path",
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_videos(source_path: str, *, recursive: bool = False) -> list[Path]:
    """Discover videos from a file path or directory path."""
    source = Path(str(source_path or "").strip()).expanduser()
    if not source.exists():
        return []

    if source.is_file():
        return [source.resolve()] if source.suffix.lower() in VIDEO_EXTENSIONS else []

    if not source.is_dir():
        return []

    iterator = source.rglob("*") if recursive else source.glob("*")
    videos = [
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos, key=lambda p: str(p).lower())


@dataclass(slots=True)
class BatchVideoItem:
    """Queue row for one source video."""

    video_id: str
    video_name: str
    video_path: str
    excluded: bool = False
    group: str = ""
    subject_id: str = ""
    time_point: str = ""
    roi_status: str = "pending"
    inference_status: str = "pending"
    analytics_status: str = "pending"
    review_status: str = "pending"
    bout_review_status: str = "pending"
    roi_review_status: str = "pending"
    rois: dict[str, dict[str, Any]] = field(default_factory=dict)
    object_rois: dict[str, dict[str, Any]] = field(default_factory=dict)
    run_output_dir: str = ""
    yolo_output_dir: str = ""
    analytics_output_dir: str = ""
    detailed_bouts_csv: str = ""
    summary_bouts_csv: str = ""
    metrics_csv: str = ""
    status_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchVideoItem":
        review_status = str(payload.get("review_status", "pending")).strip() or "pending"
        bout_review_status = str(payload.get("bout_review_status", "")).strip()
        roi_review_status = str(payload.get("roi_review_status", "")).strip()
        if not bout_review_status:
            bout_review_status = review_status
        if not roi_review_status:
            roi_review_status = review_status
        return cls(
            video_id=str(payload.get("video_id", "")).strip(),
            video_name=str(payload.get("video_name", "")).strip(),
            video_path=str(payload.get("video_path", "")).strip(),
            excluded=bool(payload.get("excluded", False)),
            group=str(payload.get("group", "")).strip(),
            subject_id=str(payload.get("subject_id", "")).strip(),
            time_point=str(payload.get("time_point", "")).strip(),
            roi_status=str(payload.get("roi_status", "pending")).strip() or "pending",
            inference_status=str(payload.get("inference_status", "pending")).strip() or "pending",
            analytics_status=str(payload.get("analytics_status", "pending")).strip() or "pending",
            review_status=review_status,
            bout_review_status=bout_review_status,
            roi_review_status=roi_review_status,
            rois=dict(payload.get("rois") or {}),
            object_rois=dict(payload.get("object_rois") or {}),
            run_output_dir=str(payload.get("run_output_dir", "")).strip(),
            yolo_output_dir=str(payload.get("yolo_output_dir", "")).strip(),
            analytics_output_dir=str(payload.get("analytics_output_dir", "")).strip(),
            detailed_bouts_csv=str(payload.get("detailed_bouts_csv", "")).strip(),
            summary_bouts_csv=str(payload.get("summary_bouts_csv", "")).strip(),
            metrics_csv=str(payload.get("metrics_csv", "")).strip(),
            status_message=str(payload.get("status_message", "")).strip(),
        )


@dataclass(slots=True)
class BatchModelCapabilities:
    """Model summary used to gate optional analytics."""

    model_path: str = ""
    task: str = "unknown"
    has_keypoints: bool = False
    keypoint_count: int = 0
    class_count: int = 0
    class_names: list[str] = field(default_factory=list)
    supports_multiclass_stats: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchModelCapabilities":
        return cls(
            model_path=str(payload.get("model_path", "")).strip(),
            task=str(payload.get("task", "unknown")).strip() or "unknown",
            has_keypoints=bool(payload.get("has_keypoints", False)),
            keypoint_count=int(payload.get("keypoint_count", 0) or 0),
            class_count=int(payload.get("class_count", 0) or 0),
            class_names=[str(name) for name in (payload.get("class_names") or []) if str(name).strip()],
            supports_multiclass_stats=bool(payload.get("supports_multiclass_stats", False)),
            warnings=[str(item) for item in (payload.get("warnings") or []) if str(item).strip()],
        )


@dataclass(slots=True)
class BatchSession:
    """Serializable batch run settings and queue state."""

    schema_version: int
    session_id: str
    created_at: str
    source_path: str = ""
    roi_strategy: str = "single"
    roi_event_mode: str = "bbox_only"
    roi_entry_threshold: float = 0.75
    roi_exit_threshold: float = 0.25
    keypoint_entry_index: int = 0
    keypoint_entry_indices: list[int] = field(default_factory=list)
    keypoint_entry_ratio_threshold: float = 0.5
    object_interaction_enabled: bool = False
    object_count: int = 0
    object_roi_size_px: int = 20
    object_roi_shape: str = "circle"
    object_interaction_keypoint_index: int = 0
    object_interaction_distance_px: float = 0.0
    use_existing_labels: bool = False
    existing_labels_root: str = ""
    model_path: str = ""
    inference_device: str = ""
    inference_batch_size: int = 25
    output_path: str = ""
    tracker_enabled: bool = True
    tracker_config_path: str = ""
    min_bout_frames: int = 3
    max_gap_frames: int = 5
    roi_min_dwell_frames: int = 3
    roi_max_gap_frames: int = 5
    video_fps: float = 0.0
    save_annotated_video: bool = False
    save_confidence: bool = False
    review_policy: str = "after_all"
    stats_correction: str = "fdr_bh"
    stats_categorical_factors: list[str] = field(default_factory=list)
    analytics_assay_preset: str = "custom"
    analytics_enabled_metrics: list[str] = field(default_factory=list)
    analytics_enabled_modules: list[str] = field(default_factory=list)
    include_kpss: bool = True
    figure_output_preset: str = "publication_dashboard"
    export_publication_figures: bool = True
    export_batch_dashboard: bool = True
    export_group_stats_overview: bool = True
    export_individual_profiles: bool = False
    export_module_archive: bool = False
    figure_export_mode: str = "assay_shortlist"
    generate_video_quicklooks: bool = False
    single_animal_mode: bool = False
    yaml_path: str = ""
    keybind_profile_path: str = ""
    roi_name_templates: list[str] = field(default_factory=list)
    object_name_templates: list[str] = field(default_factory=list)
    model_capabilities: BatchModelCapabilities = field(default_factory=BatchModelCapabilities)
    shared_rois: dict[str, dict[str, Any]] = field(default_factory=dict)
    shared_object_rois: dict[str, dict[str, Any]] = field(default_factory=dict)
    videos: list[BatchVideoItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "session_id": str(self.session_id),
            "created_at": str(self.created_at),
            "source_path": str(self.source_path),
            "roi_strategy": str(self.roi_strategy),
            "roi_event_mode": str(self.roi_event_mode),
            "roi_entry_threshold": float(self.roi_entry_threshold),
            "roi_exit_threshold": float(self.roi_exit_threshold),
            "keypoint_entry_index": int(self.keypoint_entry_index),
            "keypoint_entry_indices": [int(idx) for idx in self.keypoint_entry_indices],
            "keypoint_entry_ratio_threshold": float(self.keypoint_entry_ratio_threshold),
            "object_interaction_enabled": bool(self.object_interaction_enabled),
            "object_count": int(self.object_count),
            "object_roi_size_px": int(self.object_roi_size_px),
            "object_roi_shape": str(self.object_roi_shape or "circle"),
            "object_interaction_keypoint_index": int(self.object_interaction_keypoint_index),
            "object_interaction_distance_px": float(self.object_interaction_distance_px),
            "use_existing_labels": bool(self.use_existing_labels),
            "existing_labels_root": str(self.existing_labels_root),
            "model_path": str(self.model_path),
            "inference_device": str(self.inference_device),
            "inference_batch_size": int(self.inference_batch_size),
            "output_path": str(self.output_path),
            "tracker_enabled": bool(self.tracker_enabled),
            "tracker_config_path": str(self.tracker_config_path),
            "min_bout_frames": int(self.min_bout_frames),
            "max_gap_frames": int(self.max_gap_frames),
            "roi_min_dwell_frames": int(self.roi_min_dwell_frames),
            "roi_max_gap_frames": int(self.roi_max_gap_frames),
            "video_fps": float(self.video_fps),
            "save_annotated_video": bool(self.save_annotated_video),
            "save_confidence": bool(self.save_confidence),
            "review_policy": str(self.review_policy),
            "stats_correction": str(self.stats_correction),
            "stats_categorical_factors": [str(name) for name in (self.stats_categorical_factors or [])],
            "analytics_assay_preset": str(self.analytics_assay_preset or "custom"),
            "analytics_enabled_metrics": [str(name) for name in (self.analytics_enabled_metrics or []) if str(name).strip()],
            "analytics_enabled_modules": [str(name) for name in (self.analytics_enabled_modules or []) if str(name).strip()],
            "include_kpss": bool(self.include_kpss),
            "figure_output_preset": str(self.figure_output_preset or "publication_dashboard"),
            "export_publication_figures": bool(self.export_publication_figures),
            "export_batch_dashboard": bool(self.export_batch_dashboard),
            "export_group_stats_overview": bool(self.export_group_stats_overview),
            "export_individual_profiles": bool(self.export_individual_profiles),
            "export_module_archive": bool(self.export_module_archive),
            "figure_export_mode": str(self.figure_export_mode or "assay_shortlist"),
            "generate_video_quicklooks": bool(self.generate_video_quicklooks),
            "single_animal_mode": bool(self.single_animal_mode),
            "yaml_path": str(self.yaml_path),
            "keybind_profile_path": str(self.keybind_profile_path),
            "roi_name_templates": [str(name).strip() for name in (self.roi_name_templates or []) if str(name).strip()],
            "object_name_templates": [str(name).strip() for name in (self.object_name_templates or []) if str(name).strip()],
            "model_capabilities": self.model_capabilities.to_dict(),
            "shared_rois": dict(self.shared_rois),
            "shared_object_rois": dict(self.shared_object_rois),
            "videos": [video.to_dict() for video in self.videos],
        }

    @classmethod
    def create(cls) -> "BatchSession":
        return cls(schema_version=1, session_id=uuid.uuid4().hex, created_at=_utc_now_iso())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchSession":
        video_items = [BatchVideoItem.from_dict(row) for row in (payload.get("videos") or []) if isinstance(row, dict)]
        caps_payload = payload.get("model_capabilities")
        if not isinstance(caps_payload, dict):
            caps_payload = {}
        raw_factors = payload.get("stats_categorical_factors", ())
        if isinstance(raw_factors, str):
            raw_factors = [token.strip() for token in raw_factors.split(",")] if raw_factors.strip() else []
        elif not isinstance(raw_factors, (list, tuple)):
            raw_factors = []
        raw_enabled_metrics = payload.get("analytics_enabled_metrics", ())
        if isinstance(raw_enabled_metrics, str):
            raw_enabled_metrics = (
                [token.strip() for token in raw_enabled_metrics.split(",")]
                if raw_enabled_metrics.strip()
                else []
            )
        elif not isinstance(raw_enabled_metrics, (list, tuple)):
            raw_enabled_metrics = []
        raw_enabled_modules = payload.get("analytics_enabled_modules", ())
        if isinstance(raw_enabled_modules, str):
            raw_enabled_modules = (
                [token.strip() for token in raw_enabled_modules.split(",")]
                if raw_enabled_modules.strip()
                else []
            )
        elif not isinstance(raw_enabled_modules, (list, tuple)):
            raw_enabled_modules = []
        raw_roi_templates = payload.get("roi_name_templates", ())
        if isinstance(raw_roi_templates, str):
            raw_roi_templates = [token.strip() for token in raw_roi_templates.split(",")] if raw_roi_templates.strip() else []
        elif not isinstance(raw_roi_templates, (list, tuple)):
            raw_roi_templates = []
        raw_object_templates = payload.get("object_name_templates", ())
        if isinstance(raw_object_templates, str):
            raw_object_templates = (
                [token.strip() for token in raw_object_templates.split(",")]
                if raw_object_templates.strip()
                else []
            )
        elif not isinstance(raw_object_templates, (list, tuple)):
            raw_object_templates = []
        raw_min_bout_frames = payload.get("min_bout_frames", 3)
        raw_max_gap_frames = payload.get("max_gap_frames", 5)
        raw_roi_min_dwell_frames = payload.get("roi_min_dwell_frames", raw_min_bout_frames)
        raw_roi_max_gap_frames = payload.get("roi_max_gap_frames", raw_max_gap_frames)
        legacy_figure_mode = str(payload.get("figure_export_mode", "assay_shortlist")).strip() or "assay_shortlist"
        raw_figure_preset = str(payload.get("figure_output_preset", "")).strip()
        if raw_figure_preset:
            figure_output_preset = raw_figure_preset
            export_publication_figures = bool(payload.get("export_publication_figures", True))
            export_batch_dashboard = bool(payload.get("export_batch_dashboard", True))
            export_group_stats_overview = bool(payload.get("export_group_stats_overview", True))
            export_individual_profiles = bool(payload.get("export_individual_profiles", False))
            export_module_archive = bool(payload.get("export_module_archive", False))
            generate_video_quicklooks = bool(payload.get("generate_video_quicklooks", False))
        elif legacy_figure_mode == "disabled":
            figure_output_preset = "custom"
            export_publication_figures = False
            export_batch_dashboard = False
            export_group_stats_overview = False
            export_individual_profiles = False
            export_module_archive = False
            generate_video_quicklooks = bool(payload.get("generate_video_quicklooks", False))
        elif legacy_figure_mode == "full_bundle":
            figure_output_preset = "full_analysis_archive"
            export_publication_figures = True
            export_batch_dashboard = True
            export_group_stats_overview = True
            export_individual_profiles = True
            export_module_archive = True
            generate_video_quicklooks = bool(payload.get("generate_video_quicklooks", True))
        else:
            figure_output_preset = "publication_dashboard"
            export_publication_figures = True
            export_batch_dashboard = True
            export_group_stats_overview = True
            export_individual_profiles = False
            export_module_archive = False
            generate_video_quicklooks = bool(payload.get("generate_video_quicklooks", False))

        return cls(
            schema_version=int(payload.get("schema_version", 1) or 1),
            session_id=str(payload.get("session_id", "")).strip() or uuid.uuid4().hex,
            created_at=str(payload.get("created_at", "")).strip() or _utc_now_iso(),
            source_path=str(payload.get("source_path", "")).strip(),
            roi_strategy=str(payload.get("roi_strategy", "single")).strip() or "single",
            roi_event_mode=str(payload.get("roi_event_mode", "bbox_only")).strip() or "bbox_only",
            roi_entry_threshold=float(payload.get("roi_entry_threshold", 0.75) or 0.75),
            roi_exit_threshold=float(payload.get("roi_exit_threshold", 0.25) or 0.25),
            keypoint_entry_index=int(payload.get("keypoint_entry_index", 0) or 0),
            keypoint_entry_indices=[int(idx) for idx in (payload.get("keypoint_entry_indices") or [])],
            keypoint_entry_ratio_threshold=float(payload.get("keypoint_entry_ratio_threshold", 0.5) or 0.5),
            object_interaction_enabled=bool(payload.get("object_interaction_enabled", False)),
            object_count=int(payload.get("object_count", 0) or 0),
            object_roi_size_px=int(payload.get("object_roi_size_px", 20) or 20),
            object_roi_shape=str(payload.get("object_roi_shape", "circle")).strip() or "circle",
            object_interaction_keypoint_index=int(payload.get("object_interaction_keypoint_index", 0) or 0),
            object_interaction_distance_px=float(payload.get("object_interaction_distance_px", 0.0) or 0.0),
            use_existing_labels=bool(payload.get("use_existing_labels", False)),
            existing_labels_root=str(payload.get("existing_labels_root", "")).strip(),
            model_path=str(payload.get("model_path", "")).strip(),
            inference_device=str(payload.get("inference_device", "")).strip(),
            inference_batch_size=int(payload.get("inference_batch_size", 25) or 25),
            output_path=str(payload.get("output_path", "")).strip(),
            tracker_enabled=bool(payload.get("tracker_enabled", True)),
            tracker_config_path=str(payload.get("tracker_config_path", "")).strip(),
            min_bout_frames=int(raw_min_bout_frames or 3),
            max_gap_frames=int(raw_max_gap_frames if raw_max_gap_frames is not None else 5),
            roi_min_dwell_frames=int(raw_roi_min_dwell_frames or 3),
            roi_max_gap_frames=int(raw_roi_max_gap_frames if raw_roi_max_gap_frames is not None else 5),
            video_fps=float(payload.get("video_fps", 0.0) or 0.0),
            save_annotated_video=bool(payload.get("save_annotated_video", False)),
            save_confidence=bool(payload.get("save_confidence", False)),
            review_policy=str(payload.get("review_policy", "after_all")).strip() or "after_all",
            stats_correction=str(payload.get("stats_correction", "fdr_bh")).strip() or "fdr_bh",
            stats_categorical_factors=[str(name).strip() for name in raw_factors if str(name).strip()],
            analytics_assay_preset=str(payload.get("analytics_assay_preset", "custom")).strip() or "custom",
            analytics_enabled_metrics=[str(name).strip() for name in raw_enabled_metrics if str(name).strip()],
            analytics_enabled_modules=[str(name).strip() for name in raw_enabled_modules if str(name).strip()],
            include_kpss=bool(payload.get("include_kpss", True)),
            figure_output_preset=figure_output_preset,
            export_publication_figures=export_publication_figures,
            export_batch_dashboard=export_batch_dashboard,
            export_group_stats_overview=export_group_stats_overview,
            export_individual_profiles=export_individual_profiles,
            export_module_archive=export_module_archive,
            figure_export_mode=legacy_figure_mode,
            generate_video_quicklooks=generate_video_quicklooks,
            single_animal_mode=bool(payload.get("single_animal_mode", False)),
            yaml_path=str(payload.get("yaml_path", "")).strip(),
            keybind_profile_path=str(payload.get("keybind_profile_path", "")).strip(),
            roi_name_templates=[str(name).strip() for name in raw_roi_templates if str(name).strip()],
            object_name_templates=[str(name).strip() for name in raw_object_templates if str(name).strip()],
            model_capabilities=BatchModelCapabilities.from_dict(caps_payload),
            shared_rois=dict(payload.get("shared_rois") or {}),
            shared_object_rois=dict(payload.get("shared_object_rois") or {}),
            videos=video_items,
        )

    def save_json(self, file_path: str | Path) -> Path:
        from integra_pose.utils.safe_io import safe_write_json
        from integra_pose.utils.path_resolver import to_storage_path

        target = Path(file_path).expanduser().resolve()
        project_root = target.parent

        # PR 2B coherence-pass 2e: write paths relative to the session
        # file's parent when they sit under it. Outside-the-root paths
        # stay absolute. Bumps schema_version to mark the new format.
        payload = self.to_dict()
        payload["schema_version"] = int(CURRENT_SCHEMA_VERSION)
        for field_name in _SESSION_PATH_FIELDS:
            value = payload.get(field_name)
            if isinstance(value, str) and value:
                payload[field_name] = to_storage_path(value, project_root)
        # Per-video paths in `videos[*].video_path` get the same treatment.
        videos = payload.get("videos") or []
        for entry in videos:
            if isinstance(entry, dict):
                vp = entry.get("video_path")
                if isinstance(vp, str) and vp:
                    entry["video_path"] = to_storage_path(vp, project_root)

        # safe_write_json writes to a sibling .tmp first and then os.replace's
        # it onto the target — atomic against power loss / crash mid-write.
        return safe_write_json(target, payload, indent=2)

    @classmethod
    def load_json(cls, file_path: str | Path) -> "BatchSession":
        from integra_pose.utils.safe_io import safe_read_json
        from integra_pose.utils.path_resolver import resolve_storage_path

        target = Path(file_path).expanduser().resolve()
        payload = safe_read_json(target)
        if not isinstance(payload, dict):
            raise ValueError("Batch session JSON must contain an object payload.")

        # PR 2B coherence-pass 2e: re-anchor any relative paths against the
        # session file's parent on the *current* machine. Absolute paths in
        # the payload pass through unchanged, so older (v≤3) sessions load
        # exactly as they did before.
        project_root = target.parent
        for field_name in _SESSION_PATH_FIELDS:
            value = payload.get(field_name)
            if isinstance(value, str) and value:
                payload[field_name] = resolve_storage_path(value, project_root)
        videos = payload.get("videos") or []
        for entry in videos:
            if isinstance(entry, dict):
                vp = entry.get("video_path")
                if isinstance(vp, str) and vp:
                    entry["video_path"] = resolve_storage_path(vp, project_root)

        return cls.from_dict(payload)


def make_batch_items(video_paths: list[Path]) -> list[BatchVideoItem]:
    items: list[BatchVideoItem] = []
    used_ids: set[str] = set()
    for path in video_paths:
        resolved = path.resolve()
        stem = resolved.stem or "video"
        slug = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_").lower() or "video"
        digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:8]
        video_id = f"{slug}_{digest}"
        suffix = 2
        while video_id in used_ids:
            video_id = f"{slug}_{digest}_{suffix}"
            suffix += 1
        used_ids.add(video_id)
        items.append(
            BatchVideoItem(
                video_id=video_id,
                video_name=resolved.name,
                video_path=str(resolved),
            )
        )
    return items

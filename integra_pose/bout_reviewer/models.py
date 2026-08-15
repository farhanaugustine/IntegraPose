from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


APP_VERSION = "1.3.0"
DATABASE_SCHEMA_VERSION = 2
FINGERPRINT_SCHEME = "portable-content-v3"

ROI_CONCURRENT = "roi_concurrent"
ROI_EXCLUSIVE = "roi_exclusive"
OBJECT_INTERACTION = "object_interaction"
BEHAVIOR = "behavior"
SPATIAL_EVENT_KINDS = (
    ROI_CONCURRENT,
    ROI_EXCLUSIVE,
    OBJECT_INTERACTION,
)
EVENT_KINDS = (*SPATIAL_EVENT_KINDS, BEHAVIOR)

EVENT_KIND_LABELS = {
    ROI_CONCURRENT: "ROI occupancy (concurrent)",
    ROI_EXCLUSIVE: "ROI occupancy (exclusive)",
    OBJECT_INTERACTION: "Object interaction",
    BEHAVIOR: "Behavior class",
}

UNREVIEWED = "unreviewed"
ACCEPTED = "accepted"
MODIFIED = "modified"
ADDED = "added"
REJECTED = "rejected"
SUPERSEDED_SPLIT = "superseded_split"
SUPERSEDED_MERGE = "superseded_merge"

ACTIVE_DECISIONS = {UNREVIEWED, ACCEPTED, MODIFIED, ADDED}
FINAL_DECISIONS = {
    ACCEPTED,
    MODIFIED,
    ADDED,
    REJECTED,
    SUPERSEDED_SPLIT,
    SUPERSEDED_MERGE,
}


class ReviewError(RuntimeError):
    """Actionable project, schema, or review-store error."""


@dataclass(frozen=True)
class PredictionBout:
    prediction_id: str
    video_id: str
    event_kind: str
    label: str
    track_id: int
    start_frame: int
    end_frame: int
    source_file: str
    source_row: int
    class_id: int | None = None

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewBout:
    review_id: str
    video_id: str
    event_kind: str
    label: str
    track_id: int
    start_frame: int
    end_frame: int
    decision: str
    active: bool
    origin_prediction_ids: list[str] = field(default_factory=list)
    parent_review_ids: list[str] = field(default_factory=list)
    note: str = ""
    reviewer: str = ""
    created_at: str = ""
    updated_at: str = ""
    class_id: int | None = None

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VideoRecord:
    video_id: str
    video_name: str
    video_stem: str
    subject_id: str
    group: str
    time_point: str
    display_video: Path
    display_video_relative: str
    analytics_dir: Path
    run_dir: Path
    fps: float
    frame_count: int
    width: int
    height: int
    source_fingerprint: str
    label_catalog: dict[str, list[str]]
    track_ids: list[int]
    predictions: list[PredictionBout]
    source_files: list[str]
    behavior_classes: dict[int, str] = field(default_factory=dict)
    single_animal_mode: bool = True
    behavior_settings: dict[str, Any] = field(default_factory=dict)
    display_video_role: str = "annotated"
    source_file_hashes: dict[str, str] = field(default_factory=dict)
    path_provenance: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps


@dataclass
class ProjectData:
    root: Path
    session_path: Path
    session_id: str
    project_label: str
    videos: list[VideoRecord]
    warnings: list[str] = field(default_factory=list)

    def video_by_id(self, video_id: str) -> VideoRecord:
        for video in self.videos:
            if video.video_id == video_id:
                return video
        raise KeyError(video_id)


@dataclass(frozen=True)
class ScoreRow:
    scope: str
    video_id: str
    event_kind: str
    label: str
    class_id: str
    track_id: str
    scope_complete: bool
    temporal_iou_threshold: float
    predicted_events: int
    reviewed_events: int
    true_positive_events: int
    false_positive_events: int
    false_negative_events: int
    event_precision: float | None
    event_recall: float | None
    event_f1: float | None
    mean_matched_iou: float | None
    mean_abs_start_error_frames: float | None
    mean_abs_end_error_frames: float | None
    predicted_positive_frames: int
    reviewed_positive_frames: int
    true_positive_frames: int
    false_positive_frames: int
    false_negative_frames: int
    frame_precision: float | None
    frame_recall: float | None
    frame_f1: float | None
    frame_iou: float | None
    true_negative_frames: int
    evaluated_channel_frames: int
    frame_specificity: float | None
    frame_accuracy: float | None
    frame_balanced_accuracy: float | None
    frame_cohen_kappa: float | None
    frame_mcc: float | None
    median_abs_start_error_frames: float | None
    median_abs_end_error_frames: float | None
    mean_abs_duration_error_frames: float | None
    median_abs_duration_error_frames: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BehaviorCorrectionRow:
    scope: str
    video_id: str
    class_id: str
    behavior: str
    predicted_bouts: int
    reviewed_predicted_bouts: int
    unreviewed_predicted_bouts: int
    accepted_unchanged: int
    changed_unique_predictions: int
    boundary_corrected: int
    reclassified_from: int
    reclassified_into: int
    track_corrected: int
    removed_from_reference: int
    split_source_bouts: int
    merged_source_bouts: int
    manually_added_bouts: int
    final_reference_bouts: int
    correct_review_ratio: float | None
    incorrect_review_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

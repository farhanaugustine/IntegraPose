from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AnnotationStatus(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    APPROVED = "approved"
    REJECTED = "rejected"


class AnnotationUiState(str, Enum):
    EMPTY = "empty"
    VIDEO_READY = "video_ready"
    DRAWING = "drawing"
    SPAN_SELECTED = "span_selected"
    REVIEWING = "reviewing"


@dataclass(frozen=True)
class ProjectRecord:
    id: int
    name: str
    db_path: str


@dataclass(frozen=True)
class VideoRecord:
    id: int
    project_id: int
    path: str
    filename: str
    sha256: str
    file_size: int
    fps: float
    total_frames: int
    duration_ms: int
    width: int
    height: int
    codec: str
    imported_at: str
    last_position_ms: int
    status: str
    annotation_count: int = 0
    ambiguous_count: int = 0
    approved_count: int = 0

    @property
    def duration_s(self) -> float:
        return float(self.duration_ms) / 1000.0


@dataclass(frozen=True)
class BehaviorRecord:
    id: int
    project_id: int
    name: str
    slug: str
    color: str
    definition: str
    hotkey: str | None
    sort_order: int
    is_active: bool


@dataclass(frozen=True)
class AnnotationRecord:
    id: int
    project_id: int
    video_id: int
    behavior_id: int
    behavior_name: str
    color: str
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int
    status: str
    confidence: float
    notes: str
    is_ambiguous: bool
    is_locked: bool
    created_at: str
    updated_at: str
    extracted_clip_path: str | None

    @property
    def duration_ms(self) -> int:
        return max(0, int(self.end_ms) - int(self.start_ms))

    @property
    def duration_frames(self) -> int:
        return max(0, int(self.end_frame) - int(self.start_frame) + 1)


@dataclass(frozen=True)
class HotkeyBinding:
    action: str
    key_sequence: str

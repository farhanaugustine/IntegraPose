from __future__ import annotations

from dataclasses import dataclass

from .models import AnnotationRecord, AnnotationUiState, VideoRecord


@dataclass
class AnnotationSessionState:
    ui_state: AnnotationUiState = AnnotationUiState.EMPTY
    current_video_id: int | None = None
    selected_behavior_id: int | None = None
    selected_annotation_id: int | None = None
    playhead_ms: int = 0
    pending_start_ms: int | None = None
    pending_end_ms: int | None = None

    def set_video(self, video: VideoRecord | None) -> None:
        if video is None:
            self.ui_state = AnnotationUiState.EMPTY
            self.current_video_id = None
            self.selected_annotation_id = None
            self.playhead_ms = 0
            self.pending_start_ms = None
            self.pending_end_ms = None
            return
        self.ui_state = AnnotationUiState.VIDEO_READY
        self.current_video_id = video.id
        self.selected_annotation_id = None
        self.pending_start_ms = None
        self.pending_end_ms = None

    def begin_drawing(self) -> None:
        self.ui_state = AnnotationUiState.DRAWING

    def finish_drawing(self) -> None:
        if self.current_video_id is None:
            self.ui_state = AnnotationUiState.EMPTY
        else:
            self.ui_state = AnnotationUiState.VIDEO_READY

    def select_annotation(self, annotation: AnnotationRecord | None) -> None:
        if annotation is None:
            self.selected_annotation_id = None
            self.ui_state = (
                AnnotationUiState.VIDEO_READY
                if self.current_video_id is not None
                else AnnotationUiState.EMPTY
            )
            return
        self.selected_annotation_id = annotation.id
        self.ui_state = AnnotationUiState.SPAN_SELECTED

    def set_reviewing(self) -> None:
        if self.current_video_id is not None:
            self.ui_state = AnnotationUiState.REVIEWING

    def set_pending_start(self, ms: int) -> None:
        self.pending_start_ms = int(ms)
        if self.pending_end_ms is not None and self.pending_end_ms < self.pending_start_ms:
            self.pending_end_ms = None

    def set_pending_end(self, ms: int) -> None:
        self.pending_end_ms = int(ms)
        if self.pending_start_ms is not None and self.pending_start_ms > self.pending_end_ms:
            self.pending_start_ms = None

    def clear_pending_boundaries(self) -> None:
        self.pending_start_ms = None
        self.pending_end_ms = None

    def has_pending_pair(self) -> bool:
        return self.pending_start_ms is not None and self.pending_end_ms is not None

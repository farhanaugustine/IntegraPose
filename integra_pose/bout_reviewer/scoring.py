from __future__ import annotations

import math
import statistics
from dataclasses import asdict
from typing import Iterable, Sequence

from .models import (
    ACCEPTED,
    ADDED,
    BEHAVIOR,
    EVENT_KINDS,
    MODIFIED,
    PredictionBout,
    ReviewBout,
    ScoreRow,
)
from .store import ReviewStore


REFERENCE_DECISIONS = {ACCEPTED, MODIFIED, ADDED}


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def interval_iou(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> float:
    intersection = max(
        0,
        min(first_end, second_end) - max(first_start, second_start) + 1,
    )
    union = (
        first_end
        - first_start
        + 1
        + second_end
        - second_start
        + 1
        - intersection
    )
    return intersection / union if union else 0.0


def _same_identity(prediction: PredictionBout, review: ReviewBout) -> bool:
    return (
        prediction.video_id == review.video_id
        and prediction.event_kind == review.event_kind
        and prediction.class_id == review.class_id
        and prediction.label == review.label
        and prediction.track_id == review.track_id
    )


def greedy_event_match(
    predictions: Sequence[PredictionBout],
    reviews: Sequence[ReviewBout],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        for review_index, review in enumerate(reviews):
            if not _same_identity(prediction, review):
                continue
            iou = interval_iou(
                prediction.start_frame,
                prediction.end_frame,
                review.start_frame,
                review.end_frame,
            )
            if iou >= iou_threshold:
                candidates.append((iou, prediction_index, review_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_predictions: set[int] = set()
    used_reviews: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, prediction_index, review_index in candidates:
        if prediction_index in used_predictions or review_index in used_reviews:
            continue
        used_predictions.add(prediction_index)
        used_reviews.add(review_index)
        matches.append((prediction_index, review_index, iou))
    unmatched_predictions = set(range(len(predictions))) - used_predictions
    unmatched_reviews = set(range(len(reviews))) - used_reviews
    return matches, unmatched_predictions, unmatched_reviews


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(end)) for start, end in intervals)
    if not ordered:
        return []
    merged: list[tuple[int, int]] = []
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end + 1:
            end = max(end, next_end)
            continue
        merged.append((start, end))
        start, end = next_start, next_end
    merged.append((start, end))
    return merged


def interval_length(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in merge_intervals(intervals))


def interval_intersection_length(
    first: Iterable[tuple[int, int]],
    second: Iterable[tuple[int, int]],
) -> int:
    left = merge_intervals(first)
    right = merge_intervals(second)
    first_index = second_index = 0
    total = 0
    while first_index < len(left) and second_index < len(right):
        first_start, first_end = left[first_index]
        second_start, second_end = right[second_index]
        total += max(
            0,
            min(first_end, second_end) - max(first_start, second_start) + 1,
        )
        if first_end < second_end:
            first_index += 1
        else:
            second_index += 1
    return total


def score_group(
    *,
    scope: str,
    video_id: str,
    event_kind: str,
    label: str,
    class_id: str,
    track_id: str,
    predictions: Sequence[PredictionBout],
    reviews: Sequence[ReviewBout],
    scope_complete: bool,
    iou_threshold: float,
    video_frame_counts: dict[str, int],
) -> ScoreRow:
    matches, unmatched_predictions, unmatched_reviews = greedy_event_match(
        predictions,
        reviews,
        iou_threshold,
    )
    true_positive_events = len(matches)
    false_positive_events = len(unmatched_predictions)
    false_negative_events = len(unmatched_reviews)
    event_precision = safe_ratio(
        true_positive_events,
        true_positive_events + false_positive_events,
    )
    event_recall = safe_ratio(
        true_positive_events,
        true_positive_events + false_negative_events,
    )
    matched_ious = [match[2] for match in matches]
    start_errors = [
        abs(
            predictions[prediction_index].start_frame
            - reviews[review_index].start_frame
        )
        for prediction_index, review_index, _iou in matches
    ]
    end_errors = [
        abs(
            predictions[prediction_index].end_frame
            - reviews[review_index].end_frame
        )
        for prediction_index, review_index, _iou in matches
    ]
    duration_errors = [
        abs(
            predictions[prediction_index].frames
            - reviews[review_index].frames
        )
        for prediction_index, review_index, _iou in matches
    ]

    channel_keys = {
        (bout.video_id, bout.class_id, bout.label, bout.track_id)
        for bout in predictions
    } | {
        (bout.video_id, bout.class_id, bout.label, bout.track_id)
        for bout in reviews
    }
    predicted_frames = reviewed_frames = true_positive_frames = 0
    for channel_key in channel_keys:
        predicted_intervals = [
            (bout.start_frame, bout.end_frame)
            for bout in predictions
            if (
                bout.video_id,
                bout.class_id,
                bout.label,
                bout.track_id,
            )
            == channel_key
        ]
        reviewed_intervals = [
            (bout.start_frame, bout.end_frame)
            for bout in reviews
            if (
                bout.video_id,
                bout.class_id,
                bout.label,
                bout.track_id,
            )
            == channel_key
        ]
        predicted_frames += interval_length(predicted_intervals)
        reviewed_frames += interval_length(reviewed_intervals)
        true_positive_frames += interval_intersection_length(
            predicted_intervals,
            reviewed_intervals,
        )
    false_positive_frames = predicted_frames - true_positive_frames
    false_negative_frames = reviewed_frames - true_positive_frames
    frame_precision = safe_ratio(true_positive_frames, predicted_frames)
    frame_recall = safe_ratio(true_positive_frames, reviewed_frames)
    frame_union = (
        true_positive_frames + false_positive_frames + false_negative_frames
    )
    evaluated_channel_frames = sum(
        video_frame_counts.get(channel_key[0], 0)
        for channel_key in channel_keys
    )
    true_negative_frames = max(
        0,
        evaluated_channel_frames
        - true_positive_frames
        - false_positive_frames
        - false_negative_frames,
    )
    frame_specificity = safe_ratio(
        true_negative_frames,
        true_negative_frames + false_positive_frames,
    )
    frame_accuracy = safe_ratio(
        true_positive_frames + true_negative_frames,
        evaluated_channel_frames,
    )
    frame_balanced_accuracy = (
        statistics.fmean((frame_recall, frame_specificity))
        if frame_recall is not None and frame_specificity is not None
        else None
    )
    if evaluated_channel_frames:
        observed_agreement = (
            true_positive_frames + true_negative_frames
        ) / evaluated_channel_frames
        predicted_prevalence = predicted_frames / evaluated_channel_frames
        reviewed_prevalence = reviewed_frames / evaluated_channel_frames
        expected_agreement = (
            predicted_prevalence * reviewed_prevalence
            + (1.0 - predicted_prevalence) * (1.0 - reviewed_prevalence)
        )
        frame_cohen_kappa = (
            (observed_agreement - expected_agreement)
            / (1.0 - expected_agreement)
            if not math.isclose(expected_agreement, 1.0)
            else None
        )
    else:
        frame_cohen_kappa = None
    mcc_denominator = math.sqrt(
        (true_positive_frames + false_positive_frames)
        * (true_positive_frames + false_negative_frames)
        * (true_negative_frames + false_positive_frames)
        * (true_negative_frames + false_negative_frames)
    )
    frame_mcc = (
        (
            true_positive_frames * true_negative_frames
            - false_positive_frames * false_negative_frames
        )
        / mcc_denominator
        if mcc_denominator
        else None
    )

    return ScoreRow(
        scope=scope,
        video_id=video_id,
        event_kind=event_kind,
        label=label,
        class_id=class_id,
        track_id=track_id,
        scope_complete=scope_complete,
        temporal_iou_threshold=iou_threshold,
        predicted_events=len(predictions),
        reviewed_events=len(reviews),
        true_positive_events=true_positive_events,
        false_positive_events=false_positive_events,
        false_negative_events=false_negative_events,
        event_precision=event_precision,
        event_recall=event_recall,
        event_f1=f1_score(event_precision, event_recall),
        mean_matched_iou=(
            statistics.fmean(matched_ious) if matched_ious else None
        ),
        mean_abs_start_error_frames=(
            statistics.fmean(start_errors) if start_errors else None
        ),
        mean_abs_end_error_frames=(
            statistics.fmean(end_errors) if end_errors else None
        ),
        predicted_positive_frames=predicted_frames,
        reviewed_positive_frames=reviewed_frames,
        true_positive_frames=true_positive_frames,
        false_positive_frames=false_positive_frames,
        false_negative_frames=false_negative_frames,
        frame_precision=frame_precision,
        frame_recall=frame_recall,
        frame_f1=f1_score(frame_precision, frame_recall),
        frame_iou=safe_ratio(true_positive_frames, frame_union),
        true_negative_frames=true_negative_frames,
        evaluated_channel_frames=evaluated_channel_frames,
        frame_specificity=frame_specificity,
        frame_accuracy=frame_accuracy,
        frame_balanced_accuracy=frame_balanced_accuracy,
        frame_cohen_kappa=frame_cohen_kappa,
        frame_mcc=frame_mcc,
        median_abs_start_error_frames=(
            statistics.median(start_errors) if start_errors else None
        ),
        median_abs_end_error_frames=(
            statistics.median(end_errors) if end_errors else None
        ),
        mean_abs_duration_error_frames=(
            statistics.fmean(duration_errors) if duration_errors else None
        ),
        median_abs_duration_error_frames=(
            statistics.median(duration_errors) if duration_errors else None
        ),
    )


def score_store(
    store: ReviewStore,
    *,
    iou_threshold: float = 0.5,
) -> list[ScoreRow]:
    predictions = store.list_predictions()
    reviews = [
        bout
        for bout in store.list_review_bouts(include_inactive=False)
        if bout.decision in REFERENCE_DECISIONS
    ]
    video_ids = store.list_video_ids()
    video_frame_counts = store.video_frame_counts()
    rows: list[ScoreRow] = []

    detailed_keys = sorted(
        {
            (
                bout.video_id,
                bout.event_kind,
                -1 if bout.class_id is None else bout.class_id,
                bout.label,
                bout.track_id,
            )
            for bout in predictions
        }
        | {
            (
                bout.video_id,
                bout.event_kind,
                -1 if bout.class_id is None else bout.class_id,
                bout.label,
                bout.track_id,
            )
            for bout in reviews
        }
    )
    for video_id, event_kind, normalized_class_id, label, track_id in detailed_keys:
        class_id = None if normalized_class_id == -1 else normalized_class_id
        group_predictions = [
            bout
            for bout in predictions
            if (
                bout.video_id,
                bout.event_kind,
                bout.class_id,
                bout.label,
                bout.track_id,
            )
            == (video_id, event_kind, class_id, label, track_id)
        ]
        group_reviews = [
            bout
            for bout in reviews
            if (
                bout.video_id,
                bout.event_kind,
                bout.class_id,
                bout.label,
                bout.track_id,
            )
            == (video_id, event_kind, class_id, label, track_id)
        ]
        rows.append(
            score_group(
                scope="video_label_track",
                video_id=video_id,
                event_kind=event_kind,
                label=label,
                class_id=(
                    "" if class_id is None else str(class_id)
                ),
                track_id=str(track_id),
                predictions=group_predictions,
                reviews=group_reviews,
                scope_complete=store.scope_complete(
                    video_id,
                    event_kind,
                    track_id if event_kind == BEHAVIOR else None,
                ),
                iou_threshold=iou_threshold,
                video_frame_counts=video_frame_counts,
            )
        )

    for video_id in video_ids:
        for event_kind in EVENT_KINDS:
            group_predictions = [
                bout
                for bout in predictions
                if bout.video_id == video_id and bout.event_kind == event_kind
            ]
            group_reviews = [
                bout
                for bout in reviews
                if bout.video_id == video_id and bout.event_kind == event_kind
            ]
            rows.append(
                score_group(
                    scope="video_event_kind",
                    video_id=video_id,
                    event_kind=event_kind,
                    label="ALL",
                    class_id="ALL",
                    track_id="ALL",
                    predictions=group_predictions,
                    reviews=group_reviews,
                    scope_complete=store.scope_complete(video_id, event_kind),
                    iou_threshold=iou_threshold,
                    video_frame_counts=video_frame_counts,
                )
            )

    for event_kind in EVENT_KINDS:
        group_predictions = [
            bout for bout in predictions if bout.event_kind == event_kind
        ]
        group_reviews = [
            bout for bout in reviews if bout.event_kind == event_kind
        ]
        complete = all(
            store.scope_complete(video_id, event_kind) for video_id in video_ids
        )
        rows.append(
            score_group(
                scope="batch_event_kind",
                video_id="ALL",
                event_kind=event_kind,
                label="ALL",
                class_id="ALL",
                track_id="ALL",
                predictions=group_predictions,
                reviews=group_reviews,
                scope_complete=complete,
                iou_threshold=iou_threshold,
                video_frame_counts=video_frame_counts,
            )
        )
    return rows


def score_store_sweep(
    store: ReviewStore,
    *,
    advanced: bool = False,
    primary_threshold: float = 0.5,
) -> list[ScoreRow]:
    thresholds = (
        (0.25, 0.5, 0.75, 0.95)
        if advanced
        else (float(primary_threshold),)
    )
    return [
        row
        for threshold in thresholds
        for row in score_store(store, iou_threshold=threshold)
    ]


def score_rows_as_dicts(rows: Sequence[ScoreRow]) -> list[dict[str, object]]:
    return [asdict(row) for row in rows]

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .models import (
    ACCEPTED,
    ADDED,
    BEHAVIOR,
    FINAL_DECISIONS,
    MODIFIED,
    BehaviorCorrectionRow,
    PredictionBout,
    ReviewBout,
)
from .scoring import safe_ratio
from .store import ReviewStore


REFERENCE_DECISIONS = {ACCEPTED, MODIFIED, ADDED}


def _class_key(class_id: int | None, label: str) -> tuple[int | None, str]:
    return class_id, label


def _class_text(class_id: int | None) -> str:
    return "" if class_id is None else str(class_id)


def behavior_correction_rows(
    store: ReviewStore,
) -> list[BehaviorCorrectionRow]:
    """Summarize unique prediction corrections without counting button clicks."""

    predictions = [
        bout
        for bout in store.list_predictions(event_kind=BEHAVIOR)
    ]
    reviews = store.list_review_bouts(
        event_kind=BEHAVIOR,
        include_inactive=True,
    )
    prediction_by_id = {
        prediction.prediction_id: prediction for prediction in predictions
    }
    active_reference = [
        bout
        for bout in reviews
        if bout.active and bout.decision in REFERENCE_DECISIONS
    ]
    all_by_origin: dict[str, list[ReviewBout]] = defaultdict(list)
    reference_by_origin: dict[str, list[ReviewBout]] = defaultdict(list)
    for bout in reviews:
        for prediction_id in bout.origin_prediction_ids:
            if prediction_id in prediction_by_id:
                all_by_origin[prediction_id].append(bout)
    for bout in active_reference:
        for prediction_id in bout.origin_prediction_ids:
            if prediction_id in prediction_by_id:
                reference_by_origin[prediction_id].append(bout)

    outcomes: dict[str, dict[str, bool]] = {}
    reclassified_into: dict[tuple[str, int | None, str], set[str]] = defaultdict(set)
    for prediction in predictions:
        rows = all_by_origin.get(prediction.prediction_id, [])
        references = reference_by_origin.get(prediction.prediction_id, [])
        reviewed = any(row.decision in FINAL_DECISIONS for row in rows)
        accepted_unchanged = (
            len(references) == 1
            and references[0].decision == ACCEPTED
            and references[0].event_kind == prediction.event_kind
            and references[0].class_id == prediction.class_id
            and references[0].label == prediction.label
            and references[0].track_id == prediction.track_id
            and references[0].start_frame == prediction.start_frame
            and references[0].end_frame == prediction.end_frame
        )
        boundary_corrected = any(
            reference.start_frame != prediction.start_frame
            or reference.end_frame != prediction.end_frame
            for reference in references
        )
        reclassified = any(
            reference.class_id != prediction.class_id
            or reference.label != prediction.label
            for reference in references
        )
        track_corrected = any(
            reference.track_id != prediction.track_id
            for reference in references
        )
        removed = reviewed and not references
        split_source = len({row.review_id for row in references}) > 1
        merged_source = any(
            len(reference.origin_prediction_ids) > 1
            for reference in references
        )
        outcomes[prediction.prediction_id] = {
            "reviewed": reviewed,
            "accepted_unchanged": accepted_unchanged,
            "changed": reviewed and not accepted_unchanged,
            "boundary_corrected": boundary_corrected,
            "reclassified": reclassified,
            "track_corrected": track_corrected,
            "removed": removed,
            "split_source": split_source,
            "merged_source": merged_source,
        }
        for reference in references:
            if (
                reference.class_id != prediction.class_id
                or reference.label != prediction.label
            ):
                reclassified_into[
                    (
                        reference.video_id,
                        reference.class_id,
                        reference.label,
                    )
                ].add(prediction.prediction_id)

    class_names: dict[tuple[str, int | None], str] = {}
    for prediction in predictions:
        class_names[(prediction.video_id, prediction.class_id)] = prediction.label
    for reference in active_reference:
        class_names[(reference.video_id, reference.class_id)] = reference.label

    video_ids = sorted(
        {
            bout.video_id for bout in predictions
        }
        | {
            bout.video_id for bout in active_reference
        }
    )

    def make_row(
        *,
        scope: str,
        video_id: str,
        class_id: int | None | str,
        behavior: str,
        selected_predictions: Iterable[PredictionBout],
        selected_references: Iterable[ReviewBout],
    ) -> BehaviorCorrectionRow:
        prediction_rows = list(selected_predictions)
        reference_rows = list(selected_references)
        prediction_ids = {row.prediction_id for row in prediction_rows}
        reviewed_ids = {
            prediction_id
            for prediction_id in prediction_ids
            if outcomes[prediction_id]["reviewed"]
        }
        class_filter = None if class_id == "ALL" else class_id
        if class_id == "ALL":
            into_ids = {
                prediction_id
                for (_video, _class_id, _label), ids in reclassified_into.items()
                if video_id == "ALL" or _video == video_id
                for prediction_id in ids
            }
        else:
            into_ids = {
                prediction_id
                for (_video, destination_id, destination_label), ids
                in reclassified_into.items()
                if (video_id == "ALL" or _video == video_id)
                and destination_id == class_filter
                and destination_label == behavior
                for prediction_id in ids
            }
        accepted = sum(
            outcomes[prediction_id]["accepted_unchanged"]
            for prediction_id in prediction_ids
        )
        changed = sum(
            outcomes[prediction_id]["changed"]
            for prediction_id in prediction_ids
        )
        reviewed_count = len(reviewed_ids)
        manually_added = sum(
            not reference.origin_prediction_ids
            for reference in reference_rows
        )
        return BehaviorCorrectionRow(
            scope=scope,
            video_id=video_id,
            class_id=(
                "ALL" if class_id == "ALL" else _class_text(class_id)
            ),
            behavior=behavior,
            predicted_bouts=len(prediction_rows),
            reviewed_predicted_bouts=reviewed_count,
            unreviewed_predicted_bouts=len(prediction_rows) - reviewed_count,
            accepted_unchanged=accepted,
            changed_unique_predictions=changed,
            boundary_corrected=sum(
                outcomes[prediction_id]["boundary_corrected"]
                for prediction_id in prediction_ids
            ),
            reclassified_from=sum(
                outcomes[prediction_id]["reclassified"]
                for prediction_id in prediction_ids
            ),
            reclassified_into=len(into_ids),
            track_corrected=sum(
                outcomes[prediction_id]["track_corrected"]
                for prediction_id in prediction_ids
            ),
            removed_from_reference=sum(
                outcomes[prediction_id]["removed"]
                for prediction_id in prediction_ids
            ),
            split_source_bouts=sum(
                outcomes[prediction_id]["split_source"]
                for prediction_id in prediction_ids
            ),
            merged_source_bouts=sum(
                outcomes[prediction_id]["merged_source"]
                for prediction_id in prediction_ids
            ),
            manually_added_bouts=manually_added,
            final_reference_bouts=len(reference_rows),
            correct_review_ratio=safe_ratio(accepted, reviewed_count),
            incorrect_review_ratio=safe_ratio(changed, reviewed_count),
        )

    result: list[BehaviorCorrectionRow] = []
    for video_id in video_ids:
        video_predictions = [
            row for row in predictions if row.video_id == video_id
        ]
        video_references = [
            row for row in active_reference if row.video_id == video_id
        ]
        result.append(
            make_row(
                scope="video_behavior",
                video_id=video_id,
                class_id="ALL",
                behavior="ALL",
                selected_predictions=video_predictions,
                selected_references=video_references,
            )
        )
        class_keys = sorted(
            {
                _class_key(row.class_id, row.label)
                for row in video_predictions
            }
            | {
                _class_key(row.class_id, row.label)
                for row in video_references
            },
            key=lambda item: (
                -1 if item[0] is None else item[0],
                item[1],
            ),
        )
        for class_id, label in class_keys:
            result.append(
                make_row(
                    scope="video_behavior_class",
                    video_id=video_id,
                    class_id=class_id,
                    behavior=label,
                    selected_predictions=[
                        row
                        for row in video_predictions
                        if _class_key(row.class_id, row.label)
                        == (class_id, label)
                    ],
                    selected_references=[
                        row
                        for row in video_references
                        if _class_key(row.class_id, row.label)
                        == (class_id, label)
                    ],
                )
            )

    result.append(
        make_row(
            scope="batch_behavior",
            video_id="ALL",
            class_id="ALL",
            behavior="ALL",
            selected_predictions=predictions,
            selected_references=active_reference,
        )
    )
    batch_class_keys = sorted(
        {
            _class_key(row.class_id, row.label) for row in predictions
        }
        | {
            _class_key(row.class_id, row.label) for row in active_reference
        },
        key=lambda item: (
            -1 if item[0] is None else item[0],
            item[1],
        ),
    )
    for class_id, label in batch_class_keys:
        result.append(
            make_row(
                scope="batch_behavior_class",
                video_id="ALL",
                class_id=class_id,
                behavior=label,
                selected_predictions=[
                    row
                    for row in predictions
                    if _class_key(row.class_id, row.label)
                    == (class_id, label)
                ],
                selected_references=[
                    row
                    for row in active_reference
                    if _class_key(row.class_id, row.label)
                    == (class_id, label)
                ],
            )
        )
    return result


def behavior_transition_rows(store: ReviewStore) -> list[dict[str, Any]]:
    predictions = {
        bout.prediction_id: bout
        for bout in store.list_predictions(event_kind=BEHAVIOR)
    }
    references = [
        bout
        for bout in store.list_review_bouts(
            event_kind=BEHAVIOR,
            include_inactive=False,
        )
        if bout.active and bout.decision in REFERENCE_DECISIONS
    ]
    all_reviews = store.list_review_bouts(
        event_kind=BEHAVIOR,
        include_inactive=True,
    )
    adjudicated_origins = {
        prediction_id
        for bout in all_reviews
        if bout.decision in FINAL_DECISIONS
        for prediction_id in bout.origin_prediction_ids
    }
    destinations: dict[str, list[ReviewBout]] = defaultdict(list)
    for reference in references:
        for prediction_id in reference.origin_prediction_ids:
            if prediction_id in predictions:
                destinations[prediction_id].append(reference)
    result: list[dict[str, Any]] = []
    for prediction_id, prediction in sorted(predictions.items()):
        target_rows = destinations.get(prediction_id, [])
        if not target_rows:
            outcome = (
                "removed"
                if prediction_id in adjudicated_origins
                else "unreviewed"
            )
            result.append(
                {
                    "video_id": prediction.video_id,
                    "prediction_id": prediction_id,
                    "original_class_id": prediction.class_id,
                    "original_behavior": prediction.label,
                    "original_track_id": prediction.track_id,
                    "review_id": "",
                    "reviewed_class_id": "",
                    "reviewed_behavior": outcome.upper(),
                    "reviewed_track_id": "",
                    "transition": outcome,
                }
            )
            continue
        for reference in sorted(
            target_rows,
            key=lambda row: (
                row.start_frame,
                row.end_frame,
                row.review_id,
            ),
        ):
            class_changed = (
                reference.class_id != prediction.class_id
                or reference.label != prediction.label
            )
            track_changed = reference.track_id != prediction.track_id
            transition = (
                "class_and_track_changed"
                if class_changed and track_changed
                else "class_changed"
                if class_changed
                else "track_changed"
                if track_changed
                else "class_and_track_unchanged"
            )
            result.append(
                {
                    "video_id": prediction.video_id,
                    "prediction_id": prediction_id,
                    "original_class_id": prediction.class_id,
                    "original_behavior": prediction.label,
                    "original_track_id": prediction.track_id,
                    "review_id": reference.review_id,
                    "reviewed_class_id": reference.class_id,
                    "reviewed_behavior": reference.label,
                    "reviewed_track_id": reference.track_id,
                    "transition": transition,
                }
            )
    return result


def behavior_transition_matrix_rows(
    store: ReviewStore,
) -> list[dict[str, Any]]:
    counts: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in behavior_transition_rows(store):
        key = (
            row["original_class_id"],
            row["original_behavior"],
            row["reviewed_class_id"],
            row["reviewed_behavior"],
            row["transition"],
        )
        counts[key].add(str(row["prediction_id"]))
    return [
        {
            "original_class_id": key[0],
            "original_behavior": key[1],
            "reviewed_class_id": key[2],
            "reviewed_behavior": key[3],
            "transition": key[4],
            "unique_prediction_bouts": len(prediction_ids),
        }
        for key, prediction_ids in sorted(
            counts.items(),
            key=lambda item: tuple(str(value) for value in item[0]),
        )
    ]

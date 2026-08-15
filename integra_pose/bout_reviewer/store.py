from __future__ import annotations

import json
import hashlib
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import (
    ACCEPTED,
    ACTIVE_DECISIONS,
    ADDED,
    BEHAVIOR,
    DATABASE_SCHEMA_VERSION,
    FINGERPRINT_SCHEME,
    FINAL_DECISIONS,
    MODIFIED,
    REJECTED,
    SUPERSEDED_MERGE,
    SUPERSEDED_SPLIT,
    UNREVIEWED,
    PredictionBout,
    ProjectData,
    ReviewBout,
    ReviewError,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewStore:
    """SQLite sidecar store. IntegraPose prediction files remain immutable."""

    def __init__(self, database: str | Path) -> None:
        self.database = database
        if isinstance(database, Path):
            database.parent.mkdir(parents=True, exist_ok=True)
            database_value = str(database)
        else:
            database_value = database
        try:
            self.connection = sqlite3.connect(database_value)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            if database_value != ":memory:":
                self.connection.execute("PRAGMA journal_mode = WAL")
                self.connection.execute("PRAGMA synchronous = FULL")
            self._create_schema()
        except sqlite3.Error as exc:
            raise ReviewError(
                f"Could not initialize review database: {database_value}"
            ) from exc

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                video_name TEXT NOT NULL,
                video_stem TEXT NOT NULL,
                display_video_relative TEXT NOT NULL,
                fps REAL NOT NULL,
                frame_count INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                source_fingerprint TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predicted_bouts (
                prediction_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES videos(video_id),
                event_kind TEXT NOT NULL,
                label TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                source_file TEXT NOT NULL,
                source_row INTEGER NOT NULL,
                class_id INTEGER
            );

            CREATE TABLE IF NOT EXISTS review_bouts (
                review_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES videos(video_id),
                event_kind TEXT NOT NULL,
                label TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                decision TEXT NOT NULL,
                active INTEGER NOT NULL,
                origin_prediction_ids TEXT NOT NULL,
                parent_review_ids TEXT NOT NULL,
                note TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                class_id INTEGER
            );

            CREATE TABLE IF NOT EXISTS review_scopes (
                video_id TEXT NOT NULL REFERENCES videos(video_id),
                event_kind TEXT NOT NULL,
                complete INTEGER NOT NULL,
                reviewer TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                PRIMARY KEY (video_id, event_kind)
            );

            CREATE TABLE IF NOT EXISTS actions (
                action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_at TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                video_id TEXT NOT NULL,
                action TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS behavior_review_scopes (
                video_id TEXT NOT NULL REFERENCES videos(video_id),
                event_kind TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                complete INTEGER NOT NULL,
                reviewer TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                PRIMARY KEY (video_id, event_kind, track_id)
            );

            CREATE TABLE IF NOT EXISTS behavior_overlap_acknowledgements (
                signature TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                left_review_id TEXT NOT NULL,
                right_review_id TEXT NOT NULL,
                acknowledged_by TEXT NOT NULL,
                acknowledged_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_video_kind
                ON predicted_bouts(video_id, event_kind, label, track_id);
            CREATE INDEX IF NOT EXISTS idx_reviews_video_kind
                ON review_bouts(video_id, event_kind, label, track_id, active);
            """
        )
        self._ensure_column("predicted_bouts", "class_id", "INTEGER")
        self._ensure_column("review_bouts", "class_id", "INTEGER")
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, sql_type: str) -> None:
        columns = {
            str(row["name"])
            for row in self.connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            self.connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"
            )

    def _metadata(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row["value"])

    def _set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def _insert_prediction_snapshot(
        self,
        prediction: PredictionBout,
        *,
        created_at: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO predicted_bouts(
                prediction_id, video_id, event_kind, label, track_id,
                start_frame, end_frame, source_file, source_row, class_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction.prediction_id,
                prediction.video_id,
                prediction.event_kind,
                prediction.label,
                prediction.track_id,
                prediction.start_frame,
                prediction.end_frame,
                prediction.source_file,
                prediction.source_row,
                prediction.class_id,
            ),
        )
        self.connection.execute(
            """
            INSERT INTO review_bouts(
                review_id, video_id, event_kind, label, track_id,
                start_frame, end_frame, decision, active,
                origin_prediction_ids, parent_review_ids, note,
                reviewer, created_at, updated_at, class_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"review_{prediction.prediction_id}",
                prediction.video_id,
                prediction.event_kind,
                prediction.label,
                prediction.track_id,
                prediction.start_frame,
                prediction.end_frame,
                UNREVIEWED,
                1,
                json.dumps([prediction.prediction_id]),
                "[]",
                "",
                "",
                created_at,
                created_at,
                prediction.class_id,
            ),
        )

    def sync_project(self, project: ProjectData) -> None:
        existing_schema = self._metadata("schema_version")
        if existing_schema is not None and int(existing_schema) not in {
            1,
            DATABASE_SCHEMA_VERSION,
        }:
            raise ReviewError(
                f"Review database schema {existing_schema} is incompatible with "
                f"application schema {DATABASE_SCHEMA_VERSION}."
            )
        existing_session = self._metadata("session_id")
        if existing_session is not None and existing_session != project.session_id:
            raise ReviewError(
                "This review database belongs to a different IntegraPose session "
                f"({existing_session}), not {project.session_id}. Choose a new database."
            )
        stored_fingerprint_scheme = self._metadata("fingerprint_scheme")
        if stored_fingerprint_scheme not in {
            None,
            "portable-content-v2",
            FINGERPRINT_SCHEME,
        }:
            raise ReviewError(
                "This review database uses an unsupported prediction-fingerprint "
                f"scheme: {stored_fingerprint_scheme}."
            )
        migrating_database = (
            existing_session is not None
            and (
                existing_schema != str(DATABASE_SCHEMA_VERSION)
                or stored_fingerprint_scheme != FINGERPRINT_SCHEME
            )
        )

        current_video_ids = {video.video_id for video in project.videos}
        stored_video_ids = {
            str(row["video_id"])
            for row in self.connection.execute("SELECT video_id FROM videos")
        }
        removed_video_ids = stored_video_ids - current_video_ids
        if removed_video_ids:
            raise ReviewError(
                "The current batch is missing videos already snapshotted in the "
                f"review database: {sorted(removed_video_ids)}. The database was "
                "not altered."
            )

        existing_video_ids: set[str] = set()
        additions_by_video: dict[str, list[PredictionBout]] = {}
        for video in project.videos:
            existing = self.connection.execute(
                """
                SELECT source_fingerprint, frame_count
                FROM videos WHERE video_id = ?
                """,
                (video.video_id,),
            ).fetchone()
            if existing is not None:
                existing_video_ids.add(video.video_id)
                if int(existing["frame_count"]) != video.frame_count:
                    raise ReviewError(
                        f"Review-video frame count changed for {video.video_id}: "
                        f"stored={int(existing['frame_count'])}, "
                        f"current={video.frame_count}. The database was not altered."
                    )
                stored_ids = {
                    str(row["prediction_id"])
                    for row in self.connection.execute(
                        "SELECT prediction_id FROM predicted_bouts WHERE video_id = ?",
                        (video.video_id,),
                    )
                }
                current_by_id = {
                    bout.prediction_id: bout for bout in video.predictions
                }
                current_ids = set(current_by_id)
                removed_ids = stored_ids - current_ids
                if removed_ids:
                    raise ReviewError(
                        f"Prediction snapshot mismatch for {video.video_id}; the "
                        "existing review database was not altered."
                    )
                added_ids = current_ids - stored_ids
                additions = [current_by_id[prediction_id] for prediction_id in added_ids]
                if added_ids and (
                    stored_fingerprint_scheme == FINGERPRINT_SCHEME
                    or any(bout.event_kind != BEHAVIOR for bout in additions)
                ):
                    raise ReviewError(
                        f"Prediction snapshot mismatch for {video.video_id}; the "
                        "existing review database was not altered."
                    )
                additions_by_video[video.video_id] = sorted(
                    additions,
                    key=lambda bout: (
                        bout.track_id,
                        -1 if bout.class_id is None else bout.class_id,
                        bout.start_frame,
                        bout.end_frame,
                    ),
                )
                if (
                    str(existing["source_fingerprint"])
                    != video.source_fingerprint
                    and stored_fingerprint_scheme == FINGERPRINT_SCHEME
                ):
                    raise ReviewError(
                        f"Prediction inputs changed for {video.video_id}. The existing "
                        "review database was not altered. Start a new review database "
                        "or restore the original IntegraPose bout CSVs."
                    )

        self._set_metadata("schema_version", str(DATABASE_SCHEMA_VERSION))
        self._set_metadata("session_id", project.session_id)
        self._set_metadata("project_label", project.project_label)
        self._set_metadata("fingerprint_scheme", FINGERPRINT_SCHEME)

        for video in project.videos:
            if video.video_id in existing_video_ids:
                self.connection.execute(
                    """
                    UPDATE videos
                    SET video_name = ?, video_stem = ?,
                        display_video_relative = ?, fps = ?, frame_count = ?,
                        width = ?, height = ?, source_fingerprint = ?
                    WHERE video_id = ?
                    """,
                    (
                        video.video_name,
                        video.video_stem,
                        video.display_video_relative,
                        video.fps,
                        video.frame_count,
                        video.width,
                        video.height,
                        video.source_fingerprint,
                        video.video_id,
                    ),
                )
                now = utc_now()
                for prediction in additions_by_video.get(video.video_id, []):
                    self._insert_prediction_snapshot(
                        prediction,
                        created_at=now,
                    )
                continue

            self.connection.execute(
                """
                INSERT INTO videos(
                    video_id, video_name, video_stem, display_video_relative,
                    fps, frame_count, width, height, source_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video.video_id,
                    video.video_name,
                    video.video_stem,
                    video.display_video_relative,
                    video.fps,
                    video.frame_count,
                    video.width,
                    video.height,
                    video.source_fingerprint,
                ),
            )
            now = utc_now()
            for prediction in video.predictions:
                self._insert_prediction_snapshot(
                    prediction,
                    created_at=now,
                )
        if migrating_database:
            self._log(
                "system",
                "ALL",
                "upgrade_prediction_fingerprint",
                {
                    "from": (
                        stored_fingerprint_scheme
                        or "legacy-path-and-mtime-v1"
                    ),
                    "to": FINGERPRINT_SCHEME,
                    "database_schema_from": existing_schema,
                    "database_schema_to": DATABASE_SCHEMA_VERSION,
                    "behavior_predictions_appended": sum(
                        len(rows) for rows in additions_by_video.values()
                    ),
                    "safety_check": (
                        "Every stored prediction ID remained present; only new "
                        "behavior-class predictions were eligible for append."
                    ),
                },
            )
        self.connection.commit()

    @staticmethod
    def _prediction_from_row(row: sqlite3.Row) -> PredictionBout:
        return PredictionBout(
            prediction_id=str(row["prediction_id"]),
            video_id=str(row["video_id"]),
            event_kind=str(row["event_kind"]),
            label=str(row["label"]),
            track_id=int(row["track_id"]),
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            source_file=str(row["source_file"]),
            source_row=int(row["source_row"]),
            class_id=(
                None if row["class_id"] is None else int(row["class_id"])
            ),
        )

    @staticmethod
    def _review_from_row(row: sqlite3.Row) -> ReviewBout:
        return ReviewBout(
            review_id=str(row["review_id"]),
            video_id=str(row["video_id"]),
            event_kind=str(row["event_kind"]),
            label=str(row["label"]),
            track_id=int(row["track_id"]),
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            decision=str(row["decision"]),
            active=bool(row["active"]),
            origin_prediction_ids=list(json.loads(row["origin_prediction_ids"])),
            parent_review_ids=list(json.loads(row["parent_review_ids"])),
            note=str(row["note"]),
            reviewer=str(row["reviewer"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            class_id=(
                None if row["class_id"] is None else int(row["class_id"])
            ),
        )

    def list_predictions(
        self,
        video_id: str | None = None,
        event_kind: str | None = None,
    ) -> list[PredictionBout]:
        clauses: list[str] = []
        values: list[Any] = []
        if video_id is not None:
            clauses.append("video_id = ?")
            values.append(video_id)
        if event_kind is not None:
            clauses.append("event_kind = ?")
            values.append(event_kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"""
            SELECT * FROM predicted_bouts
            {where}
            ORDER BY video_id, event_kind, label, track_id, start_frame, end_frame
            """,
            values,
        ).fetchall()
        return [self._prediction_from_row(row) for row in rows]

    def list_review_bouts(
        self,
        video_id: str | None = None,
        event_kind: str | None = None,
        *,
        include_inactive: bool = True,
    ) -> list[ReviewBout]:
        clauses: list[str] = []
        values: list[Any] = []
        if video_id is not None:
            clauses.append("video_id = ?")
            values.append(video_id)
        if event_kind is not None:
            clauses.append("event_kind = ?")
            values.append(event_kind)
        if not include_inactive:
            clauses.append("active = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"""
            SELECT * FROM review_bouts
            {where}
            ORDER BY video_id, event_kind, label, track_id, start_frame, end_frame
            """,
            values,
        ).fetchall()
        return [self._review_from_row(row) for row in rows]

    def get_review(self, review_id: str) -> ReviewBout:
        row = self.connection.execute(
            "SELECT * FROM review_bouts WHERE review_id = ?", (review_id,)
        ).fetchone()
        if row is None:
            raise ReviewError(f"Unknown review bout: {review_id}")
        return self._review_from_row(row)

    def get_prediction(self, prediction_id: str) -> PredictionBout:
        row = self.connection.execute(
            "SELECT * FROM predicted_bouts WHERE prediction_id = ?",
            (prediction_id,),
        ).fetchone()
        if row is None:
            raise ReviewError(f"Unknown prediction bout: {prediction_id}")
        return self._prediction_from_row(row)

    def _video_frame_count(self, video_id: str) -> int:
        row = self.connection.execute(
            "SELECT frame_count FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        if row is None:
            raise ReviewError(f"Unknown video: {video_id}")
        return int(row["frame_count"])

    def _validate_interval(self, video_id: str, start: int, end: int) -> None:
        frame_count = self._video_frame_count(video_id)
        if start < 0 or end < start or end >= frame_count:
            raise ReviewError(
                f"Invalid inclusive interval [{start}, {end}] for "
                f"{video_id} with {frame_count} frames."
            )

    def _log(
        self,
        reviewer: str,
        video_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO actions(action_at, reviewer, video_id, action, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                reviewer,
                video_id,
                action,
                json.dumps(payload, sort_keys=True),
            ),
        )

    def _decision_for_active(self, bout: ReviewBout) -> str:
        if not bout.origin_prediction_ids:
            return ADDED
        if len(bout.origin_prediction_ids) != 1:
            return MODIFIED
        prediction = self.get_prediction(bout.origin_prediction_ids[0])
        same = (
            bout.event_kind == prediction.event_kind
            and bout.label == prediction.label
            and bout.class_id == prediction.class_id
            and bout.track_id == prediction.track_id
            and bout.start_frame == prediction.start_frame
            and bout.end_frame == prediction.end_frame
        )
        return ACCEPTED if same else MODIFIED

    def update_bout(
        self,
        review_id: str,
        *,
        event_kind: str,
        label: str,
        track_id: int,
        start_frame: int,
        end_frame: int,
        note: str,
        reviewer: str,
        class_id: int | None = None,
    ) -> ReviewBout:
        before = self.get_review(review_id)
        if not before.active:
            raise ReviewError("Inactive/rejected bouts must be restored before editing.")
        label = label.strip()
        if not label:
            raise ReviewError("Bout label cannot be empty.")
        if event_kind == BEHAVIOR and class_id is None:
            raise ReviewError("Behavior bouts require a class ID.")
        if event_kind != BEHAVIOR:
            class_id = None
        if int(track_id) < 0:
            raise ReviewError("Track ID cannot be negative.")
        if class_id is not None and int(class_id) < 0:
            raise ReviewError("Class ID cannot be negative.")
        self._validate_interval(before.video_id, start_frame, end_frame)
        provisional = ReviewBout(
            review_id=before.review_id,
            video_id=before.video_id,
            event_kind=event_kind,
            label=label,
            track_id=int(track_id),
            start_frame=int(start_frame),
            end_frame=int(end_frame),
            decision=before.decision,
            active=True,
            origin_prediction_ids=before.origin_prediction_ids,
            parent_review_ids=before.parent_review_ids,
            note=note.strip(),
            reviewer=reviewer,
            created_at=before.created_at,
            updated_at=utc_now(),
            class_id=class_id,
        )
        decision = self._decision_for_active(provisional)
        self.connection.execute(
            """
            UPDATE review_bouts
            SET event_kind = ?, label = ?, track_id = ?, start_frame = ?,
                end_frame = ?, decision = ?, active = 1, note = ?,
                reviewer = ?, updated_at = ?, class_id = ?
            WHERE review_id = ?
            """,
            (
                event_kind,
                label,
                int(track_id),
                int(start_frame),
                int(end_frame),
                decision,
                note.strip(),
                reviewer,
                provisional.updated_at,
                class_id,
                review_id,
            ),
        )
        after = self.get_review(review_id)
        change_types: list[str] = []
        if before.event_kind != after.event_kind:
            change_types.append("event_kind")
        if before.class_id != after.class_id or before.label != after.label:
            change_types.append("class")
        if before.track_id != after.track_id:
            change_types.append("track")
        if (
            before.start_frame != after.start_frame
            or before.end_frame != after.end_frame
        ):
            change_types.append("boundary")
        if before.note != after.note:
            change_types.append("note")
        self._log(
            reviewer,
            before.video_id,
            "update_bout",
            {
                "before": before.to_dict(),
                "after": after.to_dict(),
                "change_types": change_types,
            },
        )
        self.connection.commit()
        return after

    def accept(self, review_ids: Iterable[str], reviewer: str) -> None:
        changed: list[dict[str, Any]] = []
        for review_id in review_ids:
            bout = self.get_review(review_id)
            if not bout.active:
                raise ReviewError("Cannot accept an inactive/rejected bout.")
            decision = self._decision_for_active(bout)
            now = utc_now()
            self.connection.execute(
                """
                UPDATE review_bouts
                SET decision = ?, reviewer = ?, updated_at = ?
                WHERE review_id = ?
                """,
                (decision, reviewer, now, review_id),
            )
            changed.append(
                {
                    "review_id": review_id,
                    "before_decision": bout.decision,
                    "after_decision": decision,
                }
            )
        if changed:
            video_id = self.get_review(changed[0]["review_id"]).video_id
            self._log(reviewer, video_id, "accept_bouts", {"changes": changed})
            self.connection.commit()

    def reject(self, review_ids: Iterable[str], reviewer: str) -> None:
        changed: list[dict[str, Any]] = []
        video_id = ""
        for review_id in review_ids:
            bout = self.get_review(review_id)
            video_id = video_id or bout.video_id
            if bout.video_id != video_id:
                raise ReviewError("Reject one video at a time.")
            self.connection.execute(
                """
                UPDATE review_bouts
                SET decision = ?, active = 0, reviewer = ?, updated_at = ?
                WHERE review_id = ?
                """,
                (REJECTED, reviewer, utc_now(), review_id),
            )
            changed.append(bout.to_dict())
        if changed:
            self._log(reviewer, video_id, "reject_bouts", {"before": changed})
            self.connection.commit()

    def restore(self, review_ids: Iterable[str], reviewer: str) -> None:
        changed: list[dict[str, Any]] = []
        video_id = ""
        for review_id in review_ids:
            before = self.get_review(review_id)
            video_id = video_id or before.video_id
            if before.video_id != video_id:
                raise ReviewError("Restore one video at a time.")
            if before.decision != REJECTED:
                continue
            restored = ReviewBout(**before.to_dict())
            restored.active = True
            restored.decision = self._decision_for_active(restored)
            self.connection.execute(
                """
                UPDATE review_bouts
                SET active = 1, decision = ?, reviewer = ?, updated_at = ?
                WHERE review_id = ?
                """,
                (restored.decision, reviewer, utc_now(), review_id),
            )
            changed.append(
                {
                    "review_id": review_id,
                    "before_decision": before.decision,
                    "after_decision": restored.decision,
                }
            )
        if changed:
            self._log(reviewer, video_id, "restore_bouts", {"changes": changed})
            self.connection.commit()

    def add_bout(
        self,
        *,
        video_id: str,
        event_kind: str,
        label: str,
        track_id: int,
        start_frame: int,
        end_frame: int,
        note: str,
        reviewer: str,
        class_id: int | None = None,
    ) -> ReviewBout:
        label = label.strip()
        if not label:
            raise ReviewError("Bout label cannot be empty.")
        if event_kind == BEHAVIOR and class_id is None:
            raise ReviewError("Behavior bouts require a class ID.")
        if event_kind != BEHAVIOR:
            class_id = None
        if int(track_id) < 0:
            raise ReviewError("Track ID cannot be negative.")
        if class_id is not None and int(class_id) < 0:
            raise ReviewError("Class ID cannot be negative.")
        self._validate_interval(video_id, start_frame, end_frame)
        review_id = f"manual_{uuid.uuid4().hex}"
        now = utc_now()
        self.connection.execute(
            """
            INSERT INTO review_bouts(
                review_id, video_id, event_kind, label, track_id,
                start_frame, end_frame, decision, active,
                origin_prediction_ids, parent_review_ids, note,
                reviewer, created_at, updated_at, class_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, '[]', '[]', ?, ?, ?, ?, ?)
            """,
            (
                review_id,
                video_id,
                event_kind,
                label,
                int(track_id),
                int(start_frame),
                int(end_frame),
                ADDED,
                note.strip(),
                reviewer,
                now,
                now,
                class_id,
            ),
        )
        bout = self.get_review(review_id)
        self._log(reviewer, video_id, "add_bout", {"after": bout.to_dict()})
        self.connection.commit()
        return bout

    def split_bout(
        self,
        review_id: str,
        split_after_frame: int,
        reviewer: str,
    ) -> tuple[ReviewBout, ReviewBout]:
        before = self.get_review(review_id)
        if not before.active:
            raise ReviewError("Only an active bout can be split.")
        if not (before.start_frame <= split_after_frame < before.end_frame):
            raise ReviewError(
                "Split frame must be inside the bout and before its end frame."
            )
        now = utc_now()
        self.connection.execute(
            """
            UPDATE review_bouts
            SET active = 0, decision = ?, reviewer = ?, updated_at = ?
            WHERE review_id = ?
            """,
            (SUPERSEDED_SPLIT, reviewer, now, review_id),
        )
        children: list[ReviewBout] = []
        for start, end in (
            (before.start_frame, split_after_frame),
            (split_after_frame + 1, before.end_frame),
        ):
            child_id = f"split_{uuid.uuid4().hex}"
            decision = MODIFIED if before.origin_prediction_ids else ADDED
            self.connection.execute(
                """
                INSERT INTO review_bouts(
                    review_id, video_id, event_kind, label, track_id,
                    start_frame, end_frame, decision, active,
                    origin_prediction_ids, parent_review_ids, note,
                    reviewer, created_at, updated_at, class_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    child_id,
                    before.video_id,
                    before.event_kind,
                    before.label,
                    before.track_id,
                    start,
                    end,
                    decision,
                    json.dumps(before.origin_prediction_ids),
                    json.dumps([before.review_id]),
                    before.note,
                    reviewer,
                    now,
                    now,
                    before.class_id,
                ),
            )
            children.append(self.get_review(child_id))
        self._log(
            reviewer,
            before.video_id,
            "split_bout",
            {
                "before": before.to_dict(),
                "split_after_frame": split_after_frame,
                "after": [child.to_dict() for child in children],
            },
        )
        self.connection.commit()
        return children[0], children[1]

    def merge_bouts(
        self,
        review_ids: Iterable[str],
        reviewer: str,
    ) -> ReviewBout:
        unique_ids = list(dict.fromkeys(review_ids))
        if len(unique_ids) < 2:
            raise ReviewError("Select at least two bouts to merge.")
        bouts = [self.get_review(review_id) for review_id in unique_ids]
        if any(not bout.active for bout in bouts):
            raise ReviewError("Only active bouts can be merged.")
        identity = {
            (
                bout.video_id,
                bout.event_kind,
                bout.class_id,
                bout.label,
                bout.track_id,
            )
            for bout in bouts
        }
        if len(identity) != 1:
            raise ReviewError(
                "Merged bouts must share video, event type, class, label, and track."
            )
        first = bouts[0]
        origins = sorted(
            {
                prediction_id
                for bout in bouts
                for prediction_id in bout.origin_prediction_ids
            }
        )
        start = min(bout.start_frame for bout in bouts)
        end = max(bout.end_frame for bout in bouts)
        now = utc_now()
        for bout in bouts:
            self.connection.execute(
                """
                UPDATE review_bouts
                SET active = 0, decision = ?, reviewer = ?, updated_at = ?
                WHERE review_id = ?
                """,
                (SUPERSEDED_MERGE, reviewer, now, bout.review_id),
            )
        review_id = f"merge_{uuid.uuid4().hex}"
        decision = MODIFIED if origins else ADDED
        notes = " | ".join(dict.fromkeys(bout.note for bout in bouts if bout.note))
        self.connection.execute(
            """
            INSERT INTO review_bouts(
                review_id, video_id, event_kind, label, track_id,
                start_frame, end_frame, decision, active,
                origin_prediction_ids, parent_review_ids, note,
                reviewer, created_at, updated_at, class_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review_id,
                first.video_id,
                first.event_kind,
                first.label,
                first.track_id,
                start,
                end,
                decision,
                json.dumps(origins),
                json.dumps(unique_ids),
                notes,
                reviewer,
                now,
                now,
                first.class_id,
            ),
        )
        merged = self.get_review(review_id)
        self._log(
            reviewer,
            first.video_id,
            "merge_bouts",
            {
                "before": [bout.to_dict() for bout in bouts],
                "after": merged.to_dict(),
            },
        )
        self.connection.commit()
        return merged

    def mark_scope(
        self,
        video_id: str,
        event_kind: str,
        complete: bool,
        reviewer: str,
        *,
        track_id: int | None = None,
    ) -> None:
        if event_kind == BEHAVIOR and track_id is None:
            raise ReviewError(
                "Behavior review completion is tracked per animal. "
                "Choose one track before marking the scope complete."
            )
        if event_kind != BEHAVIOR and track_id is not None:
            raise ReviewError(
                "Track-specific completion applies only to behavior review."
            )
        if complete:
            reviewed, total = self.review_progress(
                video_id,
                event_kind,
                track_id=track_id,
            )
            if reviewed != total:
                raise ReviewError(
                    f"Cannot mark this scope complete: {reviewed} of {total} "
                    "predicted bouts have final decisions."
                )
        completed_at = utc_now() if complete else ""
        if event_kind == BEHAVIOR:
            self.connection.execute(
                """
                INSERT INTO behavior_review_scopes(
                    video_id, event_kind, track_id, complete,
                    reviewer, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id, event_kind, track_id) DO UPDATE SET
                    complete = excluded.complete,
                    reviewer = excluded.reviewer,
                    completed_at = excluded.completed_at
                """,
                (
                    video_id,
                    event_kind,
                    int(track_id),
                    int(complete),
                    reviewer,
                    completed_at,
                ),
            )
        else:
            self.connection.execute(
                """
                INSERT INTO review_scopes(
                    video_id, event_kind, complete, reviewer, completed_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(video_id, event_kind) DO UPDATE SET
                    complete = excluded.complete,
                    reviewer = excluded.reviewer,
                    completed_at = excluded.completed_at
                """,
                (video_id, event_kind, int(complete), reviewer, completed_at),
            )
        self._log(
            reviewer,
            video_id,
            "mark_scope_complete" if complete else "reopen_scope",
            {
                "event_kind": event_kind,
                "track_id": track_id,
                "complete": complete,
            },
        )
        self.connection.commit()

    def behavior_track_ids(self, video_id: str) -> list[int]:
        rows = self.connection.execute(
            """
            SELECT track_id FROM predicted_bouts
            WHERE video_id = ? AND event_kind = ?
            UNION
            SELECT track_id FROM review_bouts
            WHERE video_id = ? AND event_kind = ?
            ORDER BY track_id
            """,
            (video_id, BEHAVIOR, video_id, BEHAVIOR),
        ).fetchall()
        return [int(row["track_id"]) for row in rows]

    def scope_complete(
        self,
        video_id: str,
        event_kind: str,
        track_id: int | None = None,
    ) -> bool:
        if event_kind == BEHAVIOR:
            if track_id is None:
                track_ids = self.behavior_track_ids(video_id)
                return bool(track_ids) and all(
                    self.scope_complete(video_id, event_kind, item)
                    for item in track_ids
                )
            row = self.connection.execute(
                """
                SELECT complete FROM behavior_review_scopes
                WHERE video_id = ? AND event_kind = ? AND track_id = ?
                """,
                (video_id, event_kind, int(track_id)),
            ).fetchone()
            return bool(row["complete"]) if row is not None else False
        row = self.connection.execute(
            """
            SELECT complete FROM review_scopes
            WHERE video_id = ? AND event_kind = ?
            """,
            (video_id, event_kind),
        ).fetchone()
        return bool(row["complete"]) if row is not None else False

    def scope_rows(self) -> list[dict[str, Any]]:
        spatial = [
            {
                **dict(row),
                "track_id": None,
            }
            for row in self.connection.execute(
                """
                SELECT video_id, event_kind, complete, reviewer, completed_at
                FROM review_scopes ORDER BY video_id, event_kind
                """
            )
        ]
        behavioral = [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT video_id, event_kind, track_id, complete,
                       reviewer, completed_at
                FROM behavior_review_scopes
                ORDER BY video_id, track_id
                """
            )
        ]
        return sorted(
            spatial + behavioral,
            key=lambda row: (
                str(row["video_id"]),
                str(row["event_kind"]),
                -1 if row["track_id"] is None else int(row["track_id"]),
            ),
        )

    def review_progress(
        self,
        video_id: str,
        event_kind: str,
        *,
        track_id: int | None = None,
    ) -> tuple[int, int]:
        prediction_ids = {
            prediction.prediction_id
            for prediction in self.list_predictions(video_id, event_kind)
            if track_id is None or prediction.track_id == track_id
        }
        adjudicated: set[str] = set()
        for bout in self.list_review_bouts(video_id, event_kind):
            if bout.decision in FINAL_DECISIONS:
                adjudicated.update(bout.origin_prediction_ids)
        return len(adjudicated & prediction_ids), len(prediction_ids)

    @staticmethod
    def _overlap_signature(left: ReviewBout, right: ReviewBout) -> str:
        ordered = sorted((left, right), key=lambda bout: bout.review_id)
        payload = [
            {
                "review_id": bout.review_id,
                "track_id": bout.track_id,
                "class_id": bout.class_id,
                "label": bout.label,
                "start_frame": bout.start_frame,
                "end_frame": bout.end_frame,
            }
            for bout in ordered
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def behavior_overlap_rows(
        self,
        video_id: str | None = None,
    ) -> list[dict[str, Any]]:
        bouts = [
            bout
            for bout in self.list_review_bouts(
                video_id=video_id,
                event_kind=BEHAVIOR,
                include_inactive=False,
            )
            if bout.active
        ]
        acknowledgement_rows = {
            str(row["signature"]): dict(row)
            for row in self.connection.execute(
                """
                SELECT signature, acknowledged_by, acknowledged_at
                FROM behavior_overlap_acknowledgements
                """
            )
        }
        grouped: dict[tuple[str, int], list[ReviewBout]] = {}
        for bout in bouts:
            grouped.setdefault((bout.video_id, bout.track_id), []).append(bout)
        overlaps: list[dict[str, Any]] = []
        for (group_video_id, track_id), group in sorted(grouped.items()):
            ordered = sorted(
                group,
                key=lambda bout: (
                    bout.start_frame,
                    bout.end_frame,
                    -1 if bout.class_id is None else bout.class_id,
                    bout.review_id,
                ),
            )
            for left_index, left in enumerate(ordered):
                for right in ordered[left_index + 1 :]:
                    if right.start_frame > left.end_frame:
                        break
                    overlap_start = max(left.start_frame, right.start_frame)
                    overlap_end = min(left.end_frame, right.end_frame)
                    if overlap_start > overlap_end:
                        continue
                    signature = self._overlap_signature(left, right)
                    acknowledgement = acknowledgement_rows.get(signature)
                    same_class = (
                        left.class_id == right.class_id
                        if left.class_id is not None and right.class_id is not None
                        else left.label == right.label
                    )
                    overlaps.append(
                        {
                            "signature": signature,
                            "video_id": group_video_id,
                            "track_id": track_id,
                            "left_review_id": left.review_id,
                            "left_class_id": left.class_id,
                            "left_behavior": left.label,
                            "left_start_frame": left.start_frame,
                            "left_end_frame": left.end_frame,
                            "right_review_id": right.review_id,
                            "right_class_id": right.class_id,
                            "right_behavior": right.label,
                            "right_start_frame": right.start_frame,
                            "right_end_frame": right.end_frame,
                            "overlap_start_frame": overlap_start,
                            "overlap_end_frame": overlap_end,
                            "overlap_frames": overlap_end - overlap_start + 1,
                            "same_class": int(same_class),
                            "severity": (
                                "same_class_duplicate_or_merge_candidate"
                                if same_class
                                else "possible_behavior_cooccurrence"
                            ),
                            "acknowledged": int(acknowledgement is not None),
                            "acknowledged_by": (
                                ""
                                if acknowledgement is None
                                else str(acknowledgement["acknowledged_by"])
                            ),
                            "acknowledged_at": (
                                ""
                                if acknowledgement is None
                                else str(acknowledgement["acknowledged_at"])
                            ),
                        }
                    )
        return overlaps

    def acknowledge_overlaps(
        self,
        review_ids: Iterable[str],
        reviewer: str,
    ) -> int:
        selected = set(review_ids)
        if len(selected) < 2:
            raise ReviewError(
                "Select at least two overlapping behavior bouts to acknowledge."
            )
        overlaps = [
            row
            for row in self.behavior_overlap_rows()
            if row["left_review_id"] in selected
            and row["right_review_id"] in selected
        ]
        if not overlaps:
            raise ReviewError(
                "The selected behavior bouts do not overlap on the same track."
            )
        now = utc_now()
        for row in overlaps:
            self.connection.execute(
                """
                INSERT INTO behavior_overlap_acknowledgements(
                    signature, video_id, left_review_id, right_review_id,
                    acknowledged_by, acknowledged_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(signature) DO UPDATE SET
                    acknowledged_by = excluded.acknowledged_by,
                    acknowledged_at = excluded.acknowledged_at
                """,
                (
                    row["signature"],
                    row["video_id"],
                    row["left_review_id"],
                    row["right_review_id"],
                    reviewer,
                    now,
                ),
            )
        video_ids = {str(row["video_id"]) for row in overlaps}
        self._log(
            reviewer,
            next(iter(video_ids)) if len(video_ids) == 1 else "ALL",
            "acknowledge_behavior_overlap",
            {
                "signatures": [row["signature"] for row in overlaps],
                "pairs": [
                    [row["left_review_id"], row["right_review_id"]]
                    for row in overlaps
                ],
            },
        )
        self.connection.commit()
        return len(overlaps)

    def snapshot_video(self, video_id: str) -> dict[str, Any]:
        bouts = [
            bout.to_dict()
            for bout in self.list_review_bouts(video_id, include_inactive=True)
        ]
        scopes = [
            row for row in self.scope_rows() if row["video_id"] == video_id
        ]
        overlap_acknowledgements = [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT signature, video_id, left_review_id, right_review_id,
                       acknowledged_by, acknowledged_at
                FROM behavior_overlap_acknowledgements
                WHERE video_id = ? ORDER BY signature
                """,
                (video_id,),
            )
        ]
        return {
            "video_id": video_id,
            "bouts": bouts,
            "scopes": scopes,
            "overlap_acknowledgements": overlap_acknowledgements,
        }

    def restore_snapshot(
        self,
        snapshot: dict[str, Any],
        reviewer: str,
        action_name: str,
    ) -> None:
        video_id = str(snapshot["video_id"])
        self.connection.execute(
            "DELETE FROM review_bouts WHERE video_id = ?", (video_id,)
        )
        for raw in snapshot["bouts"]:
            bout = ReviewBout(**raw)
            self.connection.execute(
                """
                INSERT INTO review_bouts(
                    review_id, video_id, event_kind, label, track_id,
                    start_frame, end_frame, decision, active,
                    origin_prediction_ids, parent_review_ids, note,
                    reviewer, created_at, updated_at, class_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bout.review_id,
                    bout.video_id,
                    bout.event_kind,
                    bout.label,
                    bout.track_id,
                    bout.start_frame,
                    bout.end_frame,
                    bout.decision,
                    int(bout.active),
                    json.dumps(bout.origin_prediction_ids),
                    json.dumps(bout.parent_review_ids),
                    bout.note,
                    bout.reviewer,
                    bout.created_at,
                    bout.updated_at,
                    bout.class_id,
                ),
            )
        self.connection.execute(
            "DELETE FROM review_scopes WHERE video_id = ?", (video_id,)
        )
        self.connection.execute(
            "DELETE FROM behavior_review_scopes WHERE video_id = ?",
            (video_id,),
        )
        for scope in snapshot["scopes"]:
            if scope["event_kind"] == BEHAVIOR:
                self.connection.execute(
                    """
                    INSERT INTO behavior_review_scopes(
                        video_id, event_kind, track_id, complete,
                        reviewer, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scope["video_id"],
                        scope["event_kind"],
                        int(scope["track_id"]),
                        int(scope["complete"]),
                        scope["reviewer"],
                        scope["completed_at"],
                    ),
                )
            else:
                self.connection.execute(
                    """
                    INSERT INTO review_scopes(
                        video_id, event_kind, complete, reviewer, completed_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        scope["video_id"],
                        scope["event_kind"],
                        int(scope["complete"]),
                        scope["reviewer"],
                        scope["completed_at"],
                    ),
                )
        self.connection.execute(
            "DELETE FROM behavior_overlap_acknowledgements WHERE video_id = ?",
            (video_id,),
        )
        for row in snapshot.get("overlap_acknowledgements", []):
            self.connection.execute(
                """
                INSERT INTO behavior_overlap_acknowledgements(
                    signature, video_id, left_review_id, right_review_id,
                    acknowledged_by, acknowledged_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["signature"],
                    row["video_id"],
                    row["left_review_id"],
                    row["right_review_id"],
                    row["acknowledged_by"],
                    row["acknowledged_at"],
                ),
            )
        self._log(
            reviewer,
            video_id,
            action_name,
            {"restored_review_rows": len(snapshot["bouts"])},
        )
        self.connection.commit()

    def action_rows(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT action_id, action_at, reviewer, video_id, action, payload_json
                FROM actions ORDER BY action_id
                """
            )
        ]

    def metadata_rows(self) -> dict[str, str]:
        return {
            str(row["key"]): str(row["value"])
            for row in self.connection.execute(
                "SELECT key, value FROM metadata ORDER BY key"
            )
        }

    def list_video_ids(self) -> list[str]:
        return [
            str(row["video_id"])
            for row in self.connection.execute(
                "SELECT video_id FROM videos ORDER BY video_id"
            )
        ]

    def video_frame_counts(self) -> dict[str, int]:
        return {
            str(row["video_id"]): int(row["frame_count"])
            for row in self.connection.execute(
                "SELECT video_id, frame_count FROM videos ORDER BY video_id"
            )
        }

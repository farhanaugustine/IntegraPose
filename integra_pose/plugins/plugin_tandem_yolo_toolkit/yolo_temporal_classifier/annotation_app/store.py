from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import colorsys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import yaml

from .models import (
    AnnotationRecord,
    AnnotationStatus,
    BehaviorRecord,
    HotkeyBinding,
    ProjectRecord,
    VideoRecord,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "behavior"


def default_color(index: int) -> str:
    hue = (index * 137.508) % 360.0
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.48, 0.62)
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))


DEFAULT_HOTKEYS = {
    "save_project": "Ctrl+S",
    "play_pause": "Space",
    "next_video": "N",
    "previous_video": "Shift+N",
    "next_frame": "]",
    "previous_frame": "[",
    "jump_forward": "Ctrl+Right",
    "jump_backward": "Ctrl+Left",
    "quick_bookmark": "Q",
    "approve_selected": "Ctrl+Enter",
    "reject_selected": "X",
    "delete_selected": "Delete",
    "toggle_ambiguous": "Ctrl+M",
    "toggle_lock_selected": "Ctrl+L",
    "set_pending_start": "Shift+[",
    "set_pending_end": "Shift+]",
    "clear_pending_boundaries": "Ctrl+[",
    "rate_0_1x": "",
    "rate_0_25x": "",
    "rate_0_5x": "",
    "rate_1x": "Ctrl+1",
    "rate_2x": "Ctrl+2",
    "rate_4x": "Ctrl+4",
    "rate_6x": "Ctrl+6",
}

PROJECT_METADATA_FIELDS = [
    ("project_title", "Project title"),
    ("study_id", "Study ID"),
    ("lab_name", "Lab name"),
    ("institution", "Institution"),
    ("principal_investigator", "Principal investigator"),
    ("lead_annotator", "Lead annotator"),
    ("reviewer", "Reviewer"),
    ("species", "Species"),
    ("strain", "Strain"),
    ("cohort", "Cohort"),
    ("experimental_condition", "Experimental condition"),
    ("ethics_protocol", "Ethics protocol"),
    ("contact_email", "Contact email"),
    ("project_summary", "Project summary"),
    ("peer_review_notes", "Peer review notes"),
]

LEGACY_HOTKEY_DEFAULTS = {
    "next_frame": "Right",
    "previous_frame": "Left",
    "set_selected_start": "[",
    "set_selected_end": "]",
}


class AnnotationStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def backup_to(self, destination: Path) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        backup_conn = sqlite3.connect(str(target))
        try:
            self.conn.backup(backup_conn)
            backup_conn.commit()
        finally:
            backup_conn.close()
        return target

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_opened_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS project_settings (
                project_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (project_id, key),
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                path TEXT NOT NULL,
                filename TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                fps REAL NOT NULL,
                total_frames INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                codec TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                last_position_ms INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'unseen',
                notes TEXT NOT NULL DEFAULT '',
                UNIQUE(project_id, sha256, file_size),
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS behaviors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                slug TEXT NOT NULL,
                color TEXT NOT NULL,
                definition TEXT NOT NULL DEFAULT '',
                hotkey TEXT,
                sort_order INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(project_id, slug),
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                video_id INTEGER NOT NULL,
                behavior_id INTEGER NOT NULL,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                confidence REAL NOT NULL DEFAULT 1.0,
                notes TEXT NOT NULL DEFAULT '',
                is_ambiguous INTEGER NOT NULL DEFAULT 0,
                is_locked INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                extracted_clip_path TEXT,
                source_version TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE,
                FOREIGN KEY(behavior_id) REFERENCES behaviors(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS hotkeys (
                project_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                key_sequence TEXT NOT NULL,
                PRIMARY KEY (project_id, action),
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            """
        )
        self._apply_schema_migrations()
        self.conn.commit()

    def _apply_schema_migrations(self) -> None:
        behavior_columns = {
            str(row["name"])
            for row in self.conn.execute("PRAGMA table_info(behaviors)").fetchall()
        }
        if "definition" not in behavior_columns:
            self.conn.execute(
                "ALTER TABLE behaviors ADD COLUMN definition TEXT NOT NULL DEFAULT ''"
            )
        annotation_columns = {
            str(row["name"])
            for row in self.conn.execute("PRAGMA table_info(annotations)").fetchall()
        }
        if "is_locked" not in annotation_columns:
            self.conn.execute(
                "ALTER TABLE annotations ADD COLUMN is_locked INTEGER NOT NULL DEFAULT 0"
            )

    def get_or_create_default_project(self, name: str = "Default Project") -> ProjectRecord:
        row = self.conn.execute(
            "SELECT id, name FROM projects ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            now = utc_now()
            cursor = self.conn.execute(
                """
                INSERT INTO projects (name, created_at, updated_at, last_opened_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, now, now, now),
            )
            project_id = int(cursor.lastrowid)
            self.conn.commit()
            self._seed_default_settings(project_id)
            self._seed_default_hotkeys(project_id)
            self._seed_default_behaviors(project_id)
            return ProjectRecord(id=project_id, name=name, db_path=str(self.db_path))

        project_id = int(row["id"])
        self.conn.execute(
            "UPDATE projects SET last_opened_at = ?, updated_at = ? WHERE id = ?",
            (utc_now(), utc_now(), project_id),
        )
        self.conn.commit()
        self._seed_default_settings(project_id)
        self._seed_default_hotkeys(project_id)
        return ProjectRecord(
            id=project_id,
            name=str(row["name"]),
            db_path=str(self.db_path),
        )

    def _seed_default_settings(self, project_id: int) -> None:
        defaults = {
            "dataset_root": str((self.db_path.parent / "full_video_dataset").resolve()),
            "normalized_fps": "30",
            "min_clip_frames": "30",
            "resume_video_id": "",
            "resume_position_ms": "0",
            "resume_behavior_id": "",
            "bookmark_span_ms": "2000",
            "full_video_annotation_guidance_seen": "0",
            "full_video_annotation_root": str((self.db_path.parent / "full_video_annotations").resolve()),
            "full_video_val_ratio": "0.20",
        }
        for key, _label in PROJECT_METADATA_FIELDS:
            defaults.setdefault(key, "")
        for key, value in defaults.items():
            self.conn.execute(
                """
                INSERT INTO project_settings (project_id, key, value)
                VALUES (?, ?, ?)
                ON CONFLICT(project_id, key) DO NOTHING
                """,
                (project_id, key, value),
            )
        self.conn.commit()

    def _seed_default_hotkeys(self, project_id: int) -> None:
        for action, key_sequence in DEFAULT_HOTKEYS.items():
            self.conn.execute(
                """
                INSERT INTO hotkeys (project_id, action, key_sequence)
                VALUES (?, ?, ?)
                ON CONFLICT(project_id, action) DO NOTHING
                """,
                (project_id, action, key_sequence),
            )
        self._migrate_hotkeys(project_id)
        self.conn.commit()

    def _migrate_hotkeys(self, project_id: int) -> None:
        row = self.conn.execute(
            "SELECT key_sequence FROM hotkeys WHERE project_id = ? AND action = ?",
            (project_id, "previous_frame"),
        ).fetchone()
        if row is not None and str(row["key_sequence"]) == LEGACY_HOTKEY_DEFAULTS["previous_frame"]:
            self.conn.execute(
                "UPDATE hotkeys SET key_sequence = ? WHERE project_id = ? AND action = ?",
                (DEFAULT_HOTKEYS["previous_frame"], project_id, "previous_frame"),
            )

        row = self.conn.execute(
            "SELECT key_sequence FROM hotkeys WHERE project_id = ? AND action = ?",
            (project_id, "next_frame"),
        ).fetchone()
        if row is not None and str(row["key_sequence"]) == LEGACY_HOTKEY_DEFAULTS["next_frame"]:
            self.conn.execute(
                "UPDATE hotkeys SET key_sequence = ? WHERE project_id = ? AND action = ?",
                (DEFAULT_HOTKEYS["next_frame"], project_id, "next_frame"),
            )

        for legacy_action in ("set_selected_start", "set_selected_end"):
            self.conn.execute(
                "DELETE FROM hotkeys WHERE project_id = ? AND action = ?",
                (project_id, legacy_action),
            )

        row = self.conn.execute(
            "SELECT key_sequence FROM hotkeys WHERE project_id = ? AND action = ?",
            (project_id, "approve_selected"),
        ).fetchone()
        if row is not None and str(row["key_sequence"]).strip().upper() == "A":
            self.conn.execute(
                "UPDATE hotkeys SET key_sequence = ? WHERE project_id = ? AND action = ?",
                (DEFAULT_HOTKEYS["approve_selected"], project_id, "approve_selected"),
            )

    def _seed_default_behaviors(self, project_id: int) -> None:
        rows = self.conn.execute(
            "SELECT COUNT(*) AS count FROM behaviors WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if rows is not None and int(rows["count"]) > 0:
            return
        defaults = [
            ("Investigation", "I", ""),
            ("Attack", "A", ""),
            ("Mount", "M", ""),
            ("Grooming", "G", ""),
            ("Locomotion", "L", ""),
        ]
        now = utc_now()
        for index, (name, hotkey, definition) in enumerate(defaults):
            self.conn.execute(
                """
                INSERT INTO behaviors
                (project_id, name, slug, color, definition, hotkey, sort_order, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    project_id,
                    name,
                    slugify(name),
                    default_color(index),
                    definition,
                    hotkey,
                    index,
                    now,
                    now,
                ),
            )
        self.conn.commit()

    def get_setting(self, project_id: int, key: str, default: str = "") -> str:
        row = self.conn.execute(
            "SELECT value FROM project_settings WHERE project_id = ? AND key = ?",
            (project_id, key),
        ).fetchone()
        if row is None:
            return default
        return str(row["value"])

    def set_setting(self, project_id: int, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO project_settings (project_id, key, value)
            VALUES (?, ?, ?)
            ON CONFLICT(project_id, key) DO UPDATE SET value = excluded.value
            """,
            (project_id, key, value),
        )
        self.conn.commit()

    def list_hotkeys(self, project_id: int) -> list[HotkeyBinding]:
        placeholders = ",".join("?" for _ in DEFAULT_HOTKEYS)
        rows = self.conn.execute(
            f"SELECT action, key_sequence FROM hotkeys WHERE project_id = ? AND action IN ({placeholders}) ORDER BY action",
            (project_id, *DEFAULT_HOTKEYS.keys()),
        ).fetchall()
        return [
            HotkeyBinding(action=str(row["action"]), key_sequence=str(row["key_sequence"]))
            for row in rows
        ]

    def save_hotkeys(self, project_id: int, bindings: Iterable[HotkeyBinding]) -> None:
        for binding in bindings:
            self.conn.execute(
                """
                INSERT INTO hotkeys (project_id, action, key_sequence)
                VALUES (?, ?, ?)
                ON CONFLICT(project_id, action) DO UPDATE SET key_sequence = excluded.key_sequence
                """,
                (project_id, binding.action, binding.key_sequence),
            )
        self.conn.commit()

    def get_project_metadata(self, project_id: int) -> dict[str, str]:
        defaults = {key: "" for key, _ in PROJECT_METADATA_FIELDS}
        rows = self.conn.execute(
            """
            SELECT key, value
            FROM project_settings
            WHERE project_id = ? AND key IN ({})
            """.format(",".join("?" for _ in PROJECT_METADATA_FIELDS)),
            (project_id, *(key for key, _ in PROJECT_METADATA_FIELDS)),
        ).fetchall()
        for row in rows:
            defaults[str(row["key"])] = str(row["value"])
        project_row = self.conn.execute(
            "SELECT name, created_at, updated_at, last_opened_at FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        if project_row is not None:
            defaults["project_title"] = str(project_row["name"])
            defaults["created_at"] = str(project_row["created_at"])
            defaults["updated_at"] = str(project_row["updated_at"])
            defaults["last_opened_at"] = str(project_row["last_opened_at"])
        return defaults

    def save_project_metadata(self, project_id: int, metadata: dict[str, str]) -> None:
        now = utc_now()
        title = str(metadata.get("project_title", "")).strip() or "Default Project"
        self.conn.execute(
            "UPDATE projects SET name = ?, updated_at = ? WHERE id = ?",
            (title, now, project_id),
        )
        for key, _label in PROJECT_METADATA_FIELDS:
            if key == "project_title":
                continue
            self.conn.execute(
                """
                INSERT INTO project_settings (project_id, key, value)
                VALUES (?, ?, ?)
                ON CONFLICT(project_id, key) DO UPDATE SET value = excluded.value
                """,
                (project_id, key, str(metadata.get(key, "")).strip()),
            )
        self.conn.commit()

    def export_project_bundle(self, project_id: int, destination_db: Path) -> tuple[Path, Path]:
        exported_db = self.backup_to(destination_db)
        project_metadata = self.get_project_metadata(project_id)
        manifest = {
            "project_id": project_id,
            "project_name": project_metadata.get("project_title", ""),
            "database_path": str(exported_db),
            "exported_at": utc_now(),
            "metadata": {key: project_metadata.get(key, "") for key, _label in PROJECT_METADATA_FIELDS},
            "video_count": len(self.list_videos(project_id)),
            "behavior_count": len(self.list_behaviors(project_id)),
            "approved_annotation_count": len(self.approved_annotations_for_export(project_id)),
        }
        manifest_path = exported_db.with_suffix(exported_db.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return exported_db, manifest_path

    def list_behaviors(self, project_id: int) -> list[BehaviorRecord]:
        rows = self.conn.execute(
            """
            SELECT id, project_id, name, slug, color, definition, hotkey, sort_order, is_active
            FROM behaviors
            WHERE project_id = ? AND is_active = 1
            ORDER BY sort_order, name
            """,
            (project_id,),
        ).fetchall()
        return [
            BehaviorRecord(
                id=int(row["id"]),
                project_id=int(row["project_id"]),
                name=str(row["name"]),
                slug=str(row["slug"]),
                color=str(row["color"]),
                definition=str(row["definition"] or ""),
                hotkey=str(row["hotkey"]) if row["hotkey"] is not None else None,
                sort_order=int(row["sort_order"]),
                is_active=bool(row["is_active"]),
            )
            for row in rows
        ]

    def upsert_behavior(
        self,
        project_id: int,
        *,
        name: str,
        color: str | None = None,
        definition: str | None = None,
        hotkey: str | None = None,
        behavior_id: int | None = None,
    ) -> int:
        now = utc_now()
        slug = slugify(name)
        if behavior_id is None:
            count_row = self.conn.execute(
                "SELECT COUNT(*) AS count FROM behaviors WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            index = int(count_row["count"]) if count_row is not None else 0
            cursor = self.conn.execute(
                """
                INSERT INTO behaviors
                (project_id, name, slug, color, definition, hotkey, sort_order, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(project_id, slug) DO UPDATE SET
                    name = excluded.name,
                    color = excluded.color,
                    definition = excluded.definition,
                    hotkey = excluded.hotkey,
                    updated_at = excluded.updated_at
                """,
                (
                    project_id,
                    name.strip(),
                    slug,
                    color or default_color(index),
                    str(definition or "").strip(),
                    hotkey,
                    index,
                    now,
                    now,
                ),
            )
            self.conn.commit()
            if cursor.lastrowid:
                return int(cursor.lastrowid)
            row = self.conn.execute(
                "SELECT id FROM behaviors WHERE project_id = ? AND slug = ?",
                (project_id, slug),
            ).fetchone()
            if row is None:
                raise RuntimeError("Behavior upsert failed.")
            return int(row["id"])

        self.conn.execute(
            """
            UPDATE behaviors
            SET name = ?, slug = ?, color = ?, definition = ?, hotkey = ?, updated_at = ?
            WHERE id = ? AND project_id = ?
            """,
            (
                name.strip(),
                slug,
                color or default_color(0),
                str(definition or "").strip(),
                hotkey,
                now,
                behavior_id,
                project_id,
            ),
        )
        self.conn.commit()
        return int(behavior_id)

    def behavior_annotation_count(self, behavior_id: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS count FROM annotations WHERE behavior_id = ?",
            (behavior_id,),
        ).fetchone()
        return 0 if row is None else int(row["count"] or 0)

    def deactivate_behavior(self, project_id: int, behavior_id: int) -> None:
        self.conn.execute(
            """
            UPDATE behaviors
            SET is_active = 0, updated_at = ?
            WHERE id = ? AND project_id = ?
            """,
            (utc_now(), behavior_id, project_id),
        )
        self.conn.commit()

    def sync_behaviors(self, project_id: int, rows: list[dict]) -> list[str]:
        warnings: list[str] = []
        existing = {behavior.id: behavior for behavior in self.list_behaviors(project_id)}
        kept_ids: set[int] = set()

        for index, row in enumerate(rows):
            behavior_id = row.get("id")
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            color = str(row.get("color") or default_color(index))
            definition = str(row.get("definition") or "").strip()
            hotkey = row.get("hotkey")
            if hotkey is not None:
                hotkey = str(hotkey).strip() or None
            saved_id = self.upsert_behavior(
                project_id,
                name=name,
                color=color,
                definition=definition,
                hotkey=hotkey,
                behavior_id=int(behavior_id) if behavior_id not in (None, "") else None,
            )
            self.conn.execute(
                "UPDATE behaviors SET sort_order = ? WHERE id = ? AND project_id = ?",
                (index, saved_id, project_id),
            )
            kept_ids.add(int(saved_id))

        self.conn.commit()

        for behavior_id, behavior in existing.items():
            if behavior_id in kept_ids:
                continue
            if self.behavior_annotation_count(behavior_id) > 0:
                warnings.append(
                    f"Kept behavior '{behavior.name}' active because existing annotations still use it."
                )
                continue
            self.deactivate_behavior(project_id, behavior_id)

        return warnings

    def import_behaviors_from_yaml(self, project_id: int, yaml_path: Path) -> None:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "behaviors" not in payload:
            raise ValueError("YAML must contain a top-level 'behaviors' list.")
        behaviors = payload["behaviors"]
        if not isinstance(behaviors, list):
            raise ValueError("'behaviors' must be a list.")
        for index, item in enumerate(behaviors):
            if isinstance(item, str):
                name = item
                color = default_color(index)
                definition = ""
                hotkey = None
            elif isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                color = str(item.get("color") or default_color(index))
                definition = str(item.get("definition") or item.get("description") or "").strip()
                hotkey = (
                    str(item.get("hotkey")).strip()
                    if item.get("hotkey") not in (None, "")
                    else None
                )
            else:
                raise ValueError(f"Unsupported behavior entry: {item!r}")
            if not name:
                continue
            self.upsert_behavior(
                project_id,
                name=name,
                color=color,
                definition=definition,
                hotkey=hotkey,
            )

    def import_videos(self, project_id: int, paths: Iterable[Path]) -> int:
        imported = 0
        for path in paths:
            video_path = Path(path).resolve()
            if not video_path.exists() or not video_path.is_file():
                continue
            sha = partial_sha256(video_path)
            file_size = int(video_path.stat().st_size)
            exists = self.conn.execute(
                """
                SELECT id FROM videos
                WHERE project_id = ? AND sha256 = ? AND file_size = ?
                """,
                (project_id, sha, file_size),
            ).fetchone()
            if exists is not None:
                continue
            metadata = probe_video(video_path)
            now = utc_now()
            self.conn.execute(
                """
                INSERT INTO videos
                (project_id, path, filename, sha256, file_size, fps, total_frames,
                 duration_ms, width, height, codec, imported_at, last_position_ms, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'unseen', '')
                """,
                (
                    project_id,
                    str(video_path),
                    video_path.name,
                    sha,
                    file_size,
                    metadata["fps"],
                    metadata["total_frames"],
                    metadata["duration_ms"],
                    metadata["width"],
                    metadata["height"],
                    metadata["codec"],
                    now,
                ),
            )
            imported += 1
        self.conn.commit()
        return imported

    def list_videos(self, project_id: int) -> list[VideoRecord]:
        rows = self.conn.execute(
            """
            SELECT
                v.id,
                v.project_id,
                v.path,
                v.filename,
                v.sha256,
                v.file_size,
                v.fps,
                v.total_frames,
                v.duration_ms,
                v.width,
                v.height,
                v.codec,
                v.imported_at,
                v.last_position_ms,
                v.status,
                COUNT(a.id) AS annotation_count,
                SUM(CASE WHEN a.is_ambiguous = 1 THEN 1 ELSE 0 END) AS ambiguous_count,
                SUM(CASE WHEN a.status = 'approved' THEN 1 ELSE 0 END) AS approved_count
            FROM videos v
            LEFT JOIN annotations a ON a.video_id = v.id
            WHERE v.project_id = ?
            GROUP BY v.id
            ORDER BY v.filename
            """
            ,
            (project_id,),
        ).fetchall()
        return [
            VideoRecord(
                id=int(row["id"]),
                project_id=int(row["project_id"]),
                path=str(row["path"]),
                filename=str(row["filename"]),
                sha256=str(row["sha256"]),
                file_size=int(row["file_size"]),
                fps=float(row["fps"]),
                total_frames=int(row["total_frames"]),
                duration_ms=int(row["duration_ms"]),
                width=int(row["width"]),
                height=int(row["height"]),
                codec=str(row["codec"]),
                imported_at=str(row["imported_at"]),
                last_position_ms=int(row["last_position_ms"]),
                status=str(row["status"]),
                annotation_count=int(row["annotation_count"] or 0),
                ambiguous_count=int(row["ambiguous_count"] or 0),
                approved_count=int(row["approved_count"] or 0),
            )
            for row in rows
        ]

    def get_video(self, video_id: int) -> VideoRecord | None:
        rows = self.conn.execute(
            """
            SELECT project_id FROM videos WHERE id = ?
            """,
            (video_id,),
        ).fetchone()
        if rows is None:
            return None
        project_id = int(rows["project_id"])
        for record in self.list_videos(project_id):
            if record.id == video_id:
                return record
        return None

    def update_video_progress(self, video_id: int, *, position_ms: int, status: str | None = None) -> None:
        if status is None:
            self.conn.execute(
                "UPDATE videos SET last_position_ms = ? WHERE id = ?",
                (int(position_ms), video_id),
            )
        else:
            self.conn.execute(
                "UPDATE videos SET last_position_ms = ?, status = ? WHERE id = ?",
                (int(position_ms), status, video_id),
            )
        self.conn.commit()

    def save_resume_state(
        self,
        project_id: int,
        *,
        video_id: int | None,
        position_ms: int,
        behavior_id: int | None,
    ) -> None:
        self.set_setting(project_id, "resume_video_id", "" if video_id is None else str(video_id))
        self.set_setting(project_id, "resume_position_ms", str(int(position_ms)))
        self.set_setting(project_id, "resume_behavior_id", "" if behavior_id is None else str(behavior_id))

    def list_annotations(self, video_id: int) -> list[AnnotationRecord]:
        rows = self.conn.execute(
            """
            SELECT
                a.id,
                a.project_id,
                a.video_id,
                a.behavior_id,
                b.name AS behavior_name,
                b.color AS color,
                a.start_frame,
                a.end_frame,
                a.start_ms,
                a.end_ms,
                a.status,
                a.confidence,
                a.notes,
                a.is_ambiguous,
                a.is_locked,
                a.created_at,
                a.updated_at,
                a.extracted_clip_path
            FROM annotations a
            JOIN behaviors b ON b.id = a.behavior_id
            WHERE a.video_id = ?
            ORDER BY a.start_ms, a.end_ms
            """,
            (video_id,),
        ).fetchall()
        return [
            AnnotationRecord(
                id=int(row["id"]),
                project_id=int(row["project_id"]),
                video_id=int(row["video_id"]),
                behavior_id=int(row["behavior_id"]),
                behavior_name=str(row["behavior_name"]),
                color=str(row["color"]),
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
                status=str(row["status"]),
                confidence=float(row["confidence"]),
                notes=str(row["notes"]),
                is_ambiguous=bool(row["is_ambiguous"]),
                is_locked=bool(row["is_locked"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                extracted_clip_path=(
                    str(row["extracted_clip_path"])
                    if row["extracted_clip_path"] is not None
                    else None
                ),
            )
            for row in rows
        ]

    def create_annotation(
        self,
        project_id: int,
        *,
        video_id: int,
        behavior_id: int,
        start_frame: int,
        end_frame: int,
        start_ms: int,
        end_ms: int,
        status: AnnotationStatus = AnnotationStatus.DRAFT,
    ) -> int:
        now = utc_now()
        cursor = self.conn.execute(
            """
            INSERT INTO annotations
            (project_id, video_id, behavior_id, start_frame, end_frame, start_ms, end_ms,
             status, confidence, notes, is_ambiguous, created_at, updated_at, extracted_clip_path, source_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1.0, '', 0, ?, ?, NULL, ?)
            """,
            (
                project_id,
                video_id,
                behavior_id,
                int(start_frame),
                int(end_frame),
                int(start_ms),
                int(end_ms),
                status.value,
                now,
                now,
                f"{start_frame}:{end_frame}:{behavior_id}",
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def update_annotation(
        self,
        annotation_id: int,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        behavior_id: int | None = None,
        status: AnnotationStatus | None = None,
        confidence: float | None = None,
        notes: str | None = None,
        is_ambiguous: bool | None = None,
        is_locked: bool | None = None,
    ) -> None:
        row = self.conn.execute(
            """
            SELECT start_frame, end_frame, start_ms, end_ms, behavior_id, status, confidence, notes, is_ambiguous, is_locked
            FROM annotations WHERE id = ?
            """,
            (annotation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Annotation not found: {annotation_id}")

        payload = {
            "start_frame": int(row["start_frame"]) if start_frame is None else int(start_frame),
            "end_frame": int(row["end_frame"]) if end_frame is None else int(end_frame),
            "start_ms": int(row["start_ms"]) if start_ms is None else int(start_ms),
            "end_ms": int(row["end_ms"]) if end_ms is None else int(end_ms),
            "behavior_id": int(row["behavior_id"]) if behavior_id is None else int(behavior_id),
            "status": str(row["status"]) if status is None else status.value,
            "confidence": float(row["confidence"]) if confidence is None else float(confidence),
            "notes": str(row["notes"]) if notes is None else str(notes),
            "is_ambiguous": int(row["is_ambiguous"]) if is_ambiguous is None else int(bool(is_ambiguous)),
            "is_locked": int(row["is_locked"]) if is_locked is None else int(bool(is_locked)),
        }

        self.conn.execute(
            """
            UPDATE annotations
            SET start_frame = ?, end_frame = ?, start_ms = ?, end_ms = ?, behavior_id = ?,
                status = ?, confidence = ?, notes = ?, is_ambiguous = ?, is_locked = ?, updated_at = ?, source_version = ?
            WHERE id = ?
            """,
            (
                payload["start_frame"],
                payload["end_frame"],
                payload["start_ms"],
                payload["end_ms"],
                payload["behavior_id"],
                payload["status"],
                payload["confidence"],
                payload["notes"],
                payload["is_ambiguous"],
                payload["is_locked"],
                utc_now(),
                f"{payload['start_frame']}:{payload['end_frame']}:{payload['behavior_id']}",
                annotation_id,
            ),
        )
        self.conn.commit()

    def set_annotation_status_for_video(
        self,
        video_id: int,
        status: AnnotationStatus,
    ) -> int:
        cursor = self.conn.execute(
            """
            UPDATE annotations
            SET status = ?, updated_at = ?
            WHERE video_id = ? AND status <> ?
            """,
            (
                status.value,
                utc_now(),
                int(video_id),
                status.value,
            ),
        )
        self.conn.commit()
        return max(0, int(cursor.rowcount or 0))

    def set_annotation_extracted(self, annotation_id: int, clip_path: Path) -> None:
        self.conn.execute(
            "UPDATE annotations SET extracted_clip_path = ?, updated_at = ? WHERE id = ?",
            (str(clip_path), utc_now(), annotation_id),
        )
        self.conn.commit()

    def delete_annotation(self, annotation_id: int) -> None:
        self.conn.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
        self.conn.commit()

    def approved_annotations_for_export(self, project_id: int) -> list[AnnotationRecord]:
        rows = self.conn.execute(
            """
            SELECT a.video_id
            FROM annotations a
            WHERE a.project_id = ? AND a.status = 'approved'
            GROUP BY a.video_id
            """,
            (project_id,),
        ).fetchall()
        out: list[AnnotationRecord] = []
        for row in rows:
            out.extend(
                [
                    ann
                    for ann in self.list_annotations(int(row["video_id"]))
                    if ann.status == AnnotationStatus.APPROVED.value
                ]
            )
        return out


def partial_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Return a full-file SHA-256 digest.

    The name is kept for database/API compatibility with earlier builds that
    used a partial hash. Full hashing prevents accidental dedup collisions
    between same-length clips that share an opening segment.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    digest.update(str(path.stat().st_size).encode("utf-8"))
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, int | float | str]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join(chr((fourcc_int >> shift) & 0xFF) for shift in (0, 8, 16, 24)).strip("\x00")
    cap.release()
    if fps <= 1e-6:
        fps = 30.0
    if total_frames <= 0:
        cap = cv2.VideoCapture(str(path))
        counted = 0
        if cap.isOpened():
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                counted += 1
        cap.release()
        total_frames = int(counted)
    duration_ms = int(round((float(total_frames) / max(fps, 1e-6)) * 1000.0))
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_ms": duration_ms,
        "width": width,
        "height": height,
        "codec": codec or "unknown",
    }


def write_clip_manifest(dataset_root: Path, payload: dict) -> Path:
    path = dataset_root / "clips.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

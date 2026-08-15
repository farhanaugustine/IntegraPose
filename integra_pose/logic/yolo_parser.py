from __future__ import annotations

from pathlib import Path

import numpy as np

from integra_pose.utils.frame_identity import (
    FrameIdentityError,
    resolve_frame_label_indices,
)
from integra_pose.utils.yolo_pose_labels import (
    load_pose_label_schema,
    parse_yolo_pose_label,
)


class YoloParser:
    def __init__(self, app):
        self.app = app

    @staticmethod
    def resolve_frame_files(yolo_folder, *, source=None):
        """Return a unique zero-based ``frame -> filename`` lookup."""
        directory = Path(yolo_folder)
        if not directory.is_dir():
            raise FileNotFoundError(f"YOLO output directory does not exist: {directory}")

        filenames = sorted(
            entry.name
            for entry in directory.iterdir()
            if entry.is_file()
            and not entry.name.startswith(".")
            and entry.suffix.lower() == ".txt"
        )
        frame_by_filename = resolve_frame_label_indices(filenames, source=source)
        if not frame_by_filename:
            raise FrameIdentityError(
                "No frame-indexed detection TXT files were found. "
                "Files such as classes.txt and notes.txt are not inference frames."
            )
        return {
            frame_index: filename
            for filename, frame_index in sorted(
                frame_by_filename.items(),
                key=lambda item: (item[1], item[0].casefold()),
            )
        }

    @staticmethod
    def _read_nonempty_rows(path: Path) -> list[list[str]]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Could not read {path.name}: {exc}") from exc
        return [line.split() for line in lines if line.strip()]

    @staticmethod
    def _infer_legacy_keypoint_count(values: list[str]) -> int:
        """Infer a legacy pose count, preferring 3D rows with short suffixes."""
        value_count = len(values)
        candidates: list[tuple[int, int, int]] = []
        for suffix_count in (0, 1, 2):
            pose_value_count = value_count - 5 - suffix_count
            if pose_value_count <= 0:
                continue
            for dimensions in (3, 2):
                if pose_value_count % dimensions == 0:
                    keypoint_count = pose_value_count // dimensions
                    candidates.append((suffix_count, -dimensions, keypoint_count))
        if not candidates:
            raise ValueError(
                f"Could not infer a pose layout from a {value_count}-column row."
            )
        return min(candidates)[2]

    def validate_yolo_files(
        self,
        yolo_folder,
        sample_size=10,
        *,
        source=None,
        expected_keypoints=None,
    ):
        """Validate frame labels and return their consistent keypoint count."""
        try:
            frame_lookup = self.resolve_frame_files(yolo_folder, source=source)
            try:
                schema = load_pose_label_schema(
                    yolo_folder,
                    expected_keypoint_count=expected_keypoints,
                )
            except ValueError as exc:
                return False, 0, str(exc)

            detected_keypoint_count = schema.keypoint_count if schema is not None else None
            if expected_keypoints is not None:
                detected_keypoint_count = int(expected_keypoints)

            ordered_files = list(frame_lookup.values())
            if sample_size > 0:
                ordered_files = ordered_files[: min(sample_size, len(ordered_files))]

            parsed_rows = 0
            for filename in ordered_files:
                rows = self._read_nonempty_rows(Path(yolo_folder) / filename)
                for line_number, values in enumerate(rows, start=1):
                    row_keypoint_count = detected_keypoint_count
                    if row_keypoint_count is None:
                        row_keypoint_count = self._infer_legacy_keypoint_count(values)
                    try:
                        parsed = parse_yolo_pose_label(
                            values,
                            keypoint_count=row_keypoint_count,
                            schema=schema,
                        )
                    except ValueError as exc:
                        return (
                            False,
                            0,
                            f"Invalid pose row in {filename}:{line_number}: {exc}",
                        )
                    if detected_keypoint_count is None:
                        detected_keypoint_count = len(parsed.keypoints)
                    elif len(parsed.keypoints) != detected_keypoint_count:
                        return (
                            False,
                            0,
                            "Inconsistent keypoint counts across inference labels: "
                            f"{filename}:{line_number} has {len(parsed.keypoints)}, "
                            f"expected {detected_keypoint_count}.",
                        )
                    parsed_rows += 1

            if detected_keypoint_count is None or parsed_rows == 0:
                return False, 0, "No pose detections were found in the frame label files."
            return True, detected_keypoint_count, ""
        except (FileNotFoundError, FrameIdentityError, ValueError) as exc:
            return False, 0, f"Validation error: {exc}"

    def _configured_source(self):
        try:
            return self.app.config.get_setting("analytics.source_video_path_var") or None
        except Exception:
            return None

    def get_pose_from_frame(
        self,
        frame_num,
        filename=None,
        track_id=None,
        expected_keypoints=None,
    ):
        """Extract normalized XY pose data for one frame and optional track."""
        try:
            yolo_folder = self.app.config.get_setting("analytics.yolo_output_path_var")
            if not yolo_folder:
                self.app.log_message("YOLO output folder not set.", "ERROR")
                return None, None

            if filename is None:
                try:
                    frame_lookup = self.resolve_frame_files(
                        yolo_folder,
                        source=self._configured_source(),
                    )
                except (FileNotFoundError, FrameIdentityError) as exc:
                    self.app.log_message(f"Could not resolve YOLO frame files: {exc}", "ERROR")
                    return None, None
                filename = frame_lookup.get(int(frame_num))
                if filename is None:
                    self.app.log_message(
                        f"No YOLO output file maps to frame {frame_num}.",
                        "WARNING",
                    )
                    return None, None

            filepath = Path(yolo_folder) / filename
            if not filepath.is_file():
                self.app.log_message(f"YOLO output file not found: {filepath}", "WARNING")
                return None, None

            try:
                schema = load_pose_label_schema(
                    yolo_folder,
                    expected_keypoint_count=expected_keypoints,
                )
                rows = self._read_nonempty_rows(filepath)
            except ValueError as exc:
                self.app.log_message(str(exc), "ERROR")
                return None, None
            if not rows:
                return None, schema.keypoint_count if schema is not None else expected_keypoints

            keypoint_count = (
                schema.keypoint_count
                if schema is not None
                else int(expected_keypoints)
                if expected_keypoints is not None
                else self._infer_legacy_keypoint_count(rows[0])
            )

            saw_untracked_detection = False
            for line_number, values in enumerate(rows, start=1):
                try:
                    parsed = parse_yolo_pose_label(
                        values,
                        keypoint_count=keypoint_count,
                        schema=schema,
                    )
                except ValueError as exc:
                    self.app.log_message(
                        f"Invalid pose row in {filename}:{line_number}: {exc}",
                        "ERROR",
                    )
                    return None, None

                if track_id is not None:
                    if parsed.track_id is None:
                        saw_untracked_detection = True
                        continue
                    if parsed.track_id != int(track_id):
                        continue

                keypoints_xy = np.asarray(
                    [(point[0], point[1]) for point in parsed.keypoints],
                    dtype=float,
                ).reshape(-1)
                if not np.all(np.isfinite(keypoints_xy)):
                    self.app.log_message(
                        f"Non-finite keypoint values in {filename}:{line_number}",
                        "ERROR",
                    )
                    return None, None
                return keypoints_xy, len(parsed.keypoints)

            if track_id is not None:
                if saw_untracked_detection:
                    detail = f"{filename} lacks tracking data"
                else:
                    detail = f"no detection matched track {track_id} in {filename}"
                self.app.log_message(f"Could not extract track {track_id}: {detail}.", "WARNING")
            return None, keypoint_count
        except Exception as exc:
            self.app.log_message(
                f"Error extracting pose for track {track_id} in {filename}: {exc}",
                "ERROR",
            )
            return None, None

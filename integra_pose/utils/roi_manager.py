import copy
import cv2
import numpy as np
import pandas as pd
from collections import OrderedDict
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

def get_frame_num_from_filename(filename):
    """Extract the frame number from a filename using a robust regex."""
    match = re.search(r'(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))

    match = re.search(r'^(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))

    return None

class ROIManager:
    """Manages user-defined regions of interest for analytics workflows.

    Storage model
    -------------
    Each ROI entry stores:

      * ``polygons`` — list of ``np.int32`` (N, 2) arrays. **Canonical** for
        analytics. Hit-tests, occupancy, all per-frame ROI logic only ever
        looks at this list. A "rectangle" or "circle" is sampled into vertex
        polygons before storage so analytics never needs to branch on shape.
      * ``shape_metadata`` — *optional* dict the editor uses to round-trip
        the typed shape (rectangle / square / circle / polygon) plus rotation
        and reference frame index. Schema:

            {
                "shape": "rectangle" | "square" | "circle" | "polygon",
                "rotation_deg": 0.0,
                "reference_frame_index": 0,
                # rectangle | square: x, y, w, h
                # circle:              cx, cy, r
                # polygon:             vertices (list of [x, y])
            }

        Old projects that pre-date PR 2B have ``shape_metadata = None`` and
        the editor opens them as freeform polygons.
      * ``color`` / ``enabled`` / ``notes`` — unchanged from previous schema.
    """

    DEFAULT_COLORS: Tuple[Tuple[int, int, int], ...] = (
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (0, 200, 0),
        (200, 0, 0),
        (0, 128, 255),
    )

    def __init__(self):
        self.rois: OrderedDict[str, Dict[str, object]] = OrderedDict()
        self.animal_states: Dict[int, Dict[str, object]] = {}
        self.behavior_names: Dict[int, str] = {}
        self.detailed_bouts: List[Dict[str, object]] = []

    def add_roi(
        self,
        name: str,
        polygon: Iterable[Tuple[int, int]],
        *,
        color: Optional[Tuple[int, int, int]] = None,
        enabled: Optional[bool] = None,
        notes: Optional[str] = None,
        shape_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Adds a polygon to an ROI entry, creating it if necessary.

        ``shape_metadata`` is the typed-shape sidecar (rectangle/circle/etc.).
        Pass it when the polygon was generated from a typed shape so the
        editor can later reload the original geometry. Pass ``None`` (or omit)
        when the polygon was drawn freehand. When updating an existing entry,
        ``None`` does *not* clobber a previously-set sidecar — only an
        explicit non-None value replaces the stored metadata.
        """
        if not name:
            raise ValueError("ROI name cannot be empty")

        polygon_np = np.array(list(polygon), dtype=np.int32)
        if polygon_np.size == 0 or polygon_np.shape[1] != 2:
            raise ValueError("ROI polygon must contain (x, y) points")

        entry = self.rois.get(name)
        if entry is None:
            entry = {
                'polygons': [],
                'color': tuple(color) if color else self.DEFAULT_COLORS[len(self.rois) % len(self.DEFAULT_COLORS)],
                'enabled': True if enabled is None else bool(enabled),
                'notes': notes or "",
                'shape_metadata': copy.deepcopy(shape_metadata) if shape_metadata is not None else None,
            }
            self.rois[name] = entry
        else:
            if color is not None:
                entry['color'] = tuple(color)
            if enabled is not None:
                entry['enabled'] = bool(enabled)
            if notes is not None:
                entry['notes'] = notes
            if shape_metadata is not None:
                entry['shape_metadata'] = copy.deepcopy(shape_metadata)

        entry['polygons'].append(polygon_np)
        return len(entry['polygons'])

    def set_shape_metadata(
        self,
        name: str,
        shape_metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Replace the shape-metadata sidecar for an existing ROI without
        touching polygons.

        Useful when the editor saves an edit (geometry already replaced via
        a remove + add round-trip) and just needs to refresh the typed-shape
        record. Passing ``None`` explicitly clears the sidecar — the editor
        will then treat the ROI as freeform on next open.
        """
        entry = self.rois.get(name)
        if entry is None:
            raise KeyError(f"ROI {name!r} does not exist")
        entry['shape_metadata'] = (
            copy.deepcopy(shape_metadata) if shape_metadata is not None else None
        )

    def get_shape_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Return a deep copy of the shape-metadata sidecar (or None)."""
        entry = self.rois.get(name)
        if entry is None:
            return None
        meta = entry.get('shape_metadata')
        return copy.deepcopy(meta) if meta is not None else None

    def remove_roi(self, name: str) -> None:
        self.rois.pop(name, None)

    def clear(self) -> None:
        self.rois.clear()
        self.animal_states.clear()
        self.detailed_bouts.clear()

    def iter_enabled_polygons(self) -> Iterable[Tuple[str, np.ndarray]]:
        for name, data in self.rois.items():
            if not data.get('enabled', True):
                continue
            for polygon in data.get('polygons', []):
                yield name, np.array(polygon, dtype=np.int32)

    def to_serializable(self) -> Dict[str, Dict[str, object]]:
        """Serialise to JSON-friendly nested dicts.

        ``shape_metadata`` is included only when set (non-None) so projects
        without typed shapes don't grow a noisy ``"shape_metadata": null``
        key on every entry.
        """
        serializable: Dict[str, Dict[str, object]] = OrderedDict()
        for name, data in self.rois.items():
            polygons = [
                [(int(x), int(y)) for x, y in np.array(poly, dtype=np.int32).tolist()]
                for poly in data.get('polygons', [])
            ]
            entry: Dict[str, object] = {
                'polygons': polygons,
                'color': list(data.get('color', self.DEFAULT_COLORS[0])),
                'enabled': bool(data.get('enabled', True)),
                'notes': data.get('notes', ""),
            }
            shape_metadata = data.get('shape_metadata')
            if shape_metadata is not None:
                entry['shape_metadata'] = copy.deepcopy(shape_metadata)
            serializable[name] = entry
        return serializable

    def load_from_serializable(self, payload: Dict[str, Dict[str, object]]) -> None:
        """Populate from JSON-friendly dicts. Missing ``shape_metadata`` keys
        produce entries with ``shape_metadata = None`` (legacy projects)."""
        self.rois = OrderedDict()
        if not payload:
            return
        for name, data in payload.items():
            polygons = data.get('polygons', []) or []
            color = tuple(data.get('color', self.DEFAULT_COLORS[0]))
            enabled = bool(data.get('enabled', True))
            notes = data.get('notes', "")
            shape_metadata = data.get('shape_metadata')  # may be missing → None
            # First polygon carries the shape_metadata; subsequent polygons
            # under the same name (legacy multi-polygon entries) do not
            # clobber it — add_roi treats subsequent shape_metadata=None as
            # "leave existing alone."
            for idx, polygon in enumerate(polygons):
                self.add_roi(
                    name,
                    polygon,
                    color=color,
                    enabled=enabled,
                    notes=notes,
                    shape_metadata=shape_metadata if idx == 0 else None,
                )

    def _reset_analytics(self):
        self.animal_states = {}
        self.behavior_names = {}
        self.detailed_bouts = []

    def _get_roi_for_point(self, point: Tuple[int, int]) -> Optional[str]:
        for name, polygon in self.iter_enabled_polygons():
            if cv2.pointPolygonTest(polygon, point, False) >= 0:
                return name
        return None

    @staticmethod
    def _looks_like_track_column(series: pd.Series) -> bool:
        cleaned = series.dropna()
        if cleaned.empty:
            return False
        try:
            values = cleaned.to_numpy(dtype=float)
        except Exception:
            return False
        if not np.isfinite(values).all():
            return False
        rounded = np.round(values)
        return np.all(np.isclose(values, rounded)) and np.all(rounded >= 0)

    def locate_absolute(self, x: float, y: float) -> Optional[str]:
        """Public helper to find the ROI associated with an absolute pixel coordinate."""
        return self._get_roi_for_point((int(x), int(y)))

    def _finalize_bout(self, animal_id, end_frame, min_bout_duration_frames):
        state = self.animal_states.get(animal_id)
        if not state or not state.get("current_roi") or not state.get("current_bout"):
            return

        bout = state["current_bout"]
        duration = end_frame - bout["start_frame"]

        if duration >= min_bout_duration_frames:
            roi_name = state["current_roi"]
            behavior_name = self.behavior_names.get(bout["class_id"], f"Unknown_{bout['class_id']}")

            self.detailed_bouts.append({
                "ROI Name": roi_name,
                "Animal ID": animal_id,
                "Behavior": behavior_name,
                "Bout Start Frame": bout["start_frame"],
                "Bout End Frame": end_frame,
            })

        state["current_bout"] = None

    def _update_animal_state(self, animal_id, detection, frame_num, video_dims, min_bout_duration_frames):
        state = self.animal_states.get(animal_id, {})
        video_width, video_height = video_dims

        current_roi = None
        if detection:
            x_abs = detection['cx'] * video_width
            y_abs = detection['cy'] * video_height
            current_roi = self._get_roi_for_point((int(x_abs), int(y_abs)))

        if not state:
            state = {"last_seen_frame": frame_num, "current_roi": None, "current_bout": None}
            self.animal_states[animal_id] = state

        prev_roi = state.get("current_roi")

        if current_roi != prev_roi and prev_roi is not None:
            self._finalize_bout(animal_id, frame_num, min_bout_duration_frames)

        if current_roi:
            current_bout = state.get("current_bout")
            detected_class_id = detection['class_id']

            if not current_bout or current_bout["class_id"] != detected_class_id:
                self._finalize_bout(animal_id, frame_num, min_bout_duration_frames)
                state["current_bout"] = {"class_id": detected_class_id, "start_frame": frame_num}

        state["current_roi"] = current_roi
        if detection:
            state["last_seen_frame"] = frame_num

    def process_yolo_path(self, yolo_path, behavior_names, video_width, video_height, max_frame_gap, min_bout_duration_frames):
        self._reset_analytics()
        self.behavior_names = behavior_names
        video_dims = (video_width, video_height)

        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLO output path does not exist: {yolo_path}")

        txt_files_with_frames: List[Tuple[int, str]] = []
        for f in os.listdir(yolo_path):
            if f.endswith('.txt'):
                frame_num = get_frame_num_from_filename(f)
                if frame_num is not None:
                    txt_files_with_frames.append((frame_num, os.path.join(yolo_path, f)))
        txt_files_with_frames.sort()

        if not txt_files_with_frames:
            raise FileNotFoundError("No .txt files with valid frame numbers found.")

        last_processed_frame = -1
        for frame_num, file_path in txt_files_with_frames:
            if frame_num > last_processed_frame + 1:
                for animal_id in list(self.animal_states.keys()):
                    last_seen = self.animal_states[animal_id].get("last_seen_frame", frame_num)
                    if frame_num - last_seen >= max_frame_gap:
                        self._finalize_bout(animal_id, last_seen, min_bout_duration_frames)
                        del self.animal_states[animal_id]

            detections_this_frame: Dict[int, Dict[str, float]] = {}
            try:
                df_raw = pd.read_csv(file_path, sep=' ', header=None)
            except pd.errors.EmptyDataError:
                df_raw = pd.DataFrame()
            except Exception:
                df_raw = pd.DataFrame()
            if not df_raw.empty:
                df = df_raw.apply(pd.to_numeric, errors='coerce')
                total_cols = df.shape[1]
                track_col_idx = None
                if total_cols >= 1 and self._looks_like_track_column(df.iloc[:, -1]):
                    track_col_idx = total_cols - 1
                for row_idx, row in df.iterrows():
                    if total_cols < 3:
                        continue
                    try:
                        track_identifier = int(round(row.iloc[track_col_idx])) if track_col_idx is not None else 0
                    except (ValueError, TypeError):
                        track_identifier = 0
                    try:
                        class_id = int(round(row.iloc[0]))
                    except (ValueError, TypeError):
                        class_id = 0
                    try:
                        cx = float(row.iloc[1])
                        cy = float(row.iloc[2])
                    except (ValueError, TypeError):
                        continue
                    detections_this_frame[track_identifier] = {
                        'class_id': class_id,
                        'cx': cx,
                        'cy': cy,
                    }

            all_known_ids = set(self.animal_states.keys()) | set(detections_this_frame.keys())
            for animal_id in all_known_ids:
                self._update_animal_state(animal_id, detections_this_frame.get(animal_id), frame_num, video_dims, min_bout_duration_frames)

            last_processed_frame = frame_num

        end_frame = last_processed_frame + 1
        for animal_id in list(self.animal_states.keys()):
            self._finalize_bout(animal_id, end_frame, min_bout_duration_frames)

    def get_analytics(self):
        self.detailed_bouts.sort(key=lambda x: (x['ROI Name'], x['Animal ID'], x['Bout Start Frame']))
        return self.detailed_bouts

    def export_csv(self, file_path):
        bouts = self.get_analytics()
        if not bouts:
            return
        df = pd.DataFrame(bouts)
        df.to_csv(file_path, index=False)

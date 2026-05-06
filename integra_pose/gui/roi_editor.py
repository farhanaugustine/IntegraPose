"""Unified interactive ROI editor (PR 2B Commit 4).

Replaces the legacy ``_ROIDialog`` and ``_BoxROIDialog`` Toplevels with a
single editor that handles all four shape types (rectangle, square,
circle, polygon), supports translate / resize / rotate / vertex
add-remove, and round-trips through the ``shape_metadata`` schema added
in Commit 3.

Design decisions
----------------

* **Internal data model = the storage schema.** Each ROI is a plain dict
  matching ``ROIManager``'s ``shape_metadata`` sidecar:

      {
          "name": str,
          "shape": "rectangle" | "square" | "circle" | "polygon",
          "rotation_deg": float,
          "reference_frame_index": int,
          # rectangle | square: x, y, w, h
          # circle:              cx, cy, r
          # polygon:             vertices: list[[x, y]]
      }

  No conversion layer; ``on_save`` receives the same shape that ``rois``
  was provided in.

* **Source-image coordinates are canonical.** The display canvas is
  scaled (typically 1280x720 max). All hit-tests, mutations, and saved
  geometry are in source-image pixel space; only render-time draws use
  display coordinates.

* **One editor instance edits a *batch* of ROIs.** Researchers click
  "Build ROIs", configure N rows in the builder dialog, and the editor
  opens with all N pre-placed shapes. They drag, resize, rotate, and
  click Save once. (Commit 5 is the file that wires this end-to-end;
  the editor itself is unaware of where the rois come from.)

* **Object ROIs do NOT use this editor.** The ``_ObjectROIDialog``
  click-center-with-fixed-size workflow stays. This editor is
  exclusively for arena ROIs that the new builder dialog produces.
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Any, Callable, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    import numpy as np

# numpy / Pillow / OpenCV are imported lazily on first ``RoiEditor``
# instantiation so the pure-geometry helpers in this module remain
# importable in test / lint environments that don't have the full
# GUI dependency stack.
Image = None  # type: ignore[assignment]
ImageTk = None  # type: ignore[assignment]
cv2 = None  # type: ignore[assignment]
_np = None  # type: ignore[assignment]


def _ensure_imaging_imports() -> None:
    """Import numpy + Pillow + OpenCV lazily on first ``RoiEditor`` use."""
    global Image, ImageTk, cv2, _np  # noqa: PLW0603
    if Image is not None and ImageTk is not None and _np is not None:
        return
    try:
        import numpy as _numpy  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "numpy is required to construct RoiEditor."
        ) from exc
    _np = _numpy
    try:
        from PIL import Image as _Image, ImageTk as _ImageTk  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Pillow is required to construct RoiEditor; install Pillow "
            "(or run the editor from the full IntegraPose environment)."
        ) from exc
    Image = _Image
    ImageTk = _ImageTk
    try:
        import cv2 as _cv2  # noqa: PLC0415
    except Exception:  # pragma: no cover - cv2 is optional
        _cv2 = None
    cv2 = _cv2

from integra_pose.utils.roi_layout import (
    ALLOWED_SHAPES,
    POLYGON_DEFAULT_VERTICES,
    POLYGON_MAX_VERTICES,
    POLYGON_MIN_VERTICES,
    SHAPE_CIRCLE,
    SHAPE_POLYGON,
    SHAPE_RECTANGLE,
    SHAPE_SQUARE,
)


# ---------------------------------------------------------------------------
# Geometry helpers (pure functions, testable without Tk)
# ---------------------------------------------------------------------------


def rotate_point(
    px: float, py: float, cx: float, cy: float, degrees: float
) -> Tuple[float, float]:
    """Rotate ``(px, py)`` around ``(cx, cy)`` by ``degrees`` (counter-clockwise)."""
    if degrees == 0.0:
        return px, py
    rad = math.radians(degrees)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    dx = px - cx
    dy = py - cy
    return (cx + dx * cos_r - dy * sin_r, cy + dx * sin_r + dy * cos_r)


def rectangle_corners(roi: dict) -> List[Tuple[float, float]]:
    """Four rotated corners of a rectangle/square ROI (tl, tr, br, bl)."""
    x = float(roi["x"])
    y = float(roi["y"])
    w = float(roi["w"])
    h = float(roi["h"])
    rot = float(roi.get("rotation_deg", 0.0))
    cx = x + w / 2.0
    cy = y + h / 2.0
    raw = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return [rotate_point(px, py, cx, cy, rot) for px, py in raw]


def polygon_vertices_rotated(roi: dict) -> List[Tuple[float, float]]:
    """Polygon vertices with rotation applied around the centroid."""
    verts = [(float(v[0]), float(v[1])) for v in roi.get("vertices", [])]
    rot = float(roi.get("rotation_deg", 0.0))
    if rot == 0.0 or not verts:
        return verts
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    return [rotate_point(px, py, cx, cy, rot) for px, py in verts]


def shape_centroid(roi: dict) -> Tuple[float, float]:
    """Centroid of an ROI in source-image pixel space."""
    shape = roi["shape"]
    if shape in (SHAPE_RECTANGLE, SHAPE_SQUARE):
        return (
            float(roi["x"]) + float(roi["w"]) / 2.0,
            float(roi["y"]) + float(roi["h"]) / 2.0,
        )
    if shape == SHAPE_CIRCLE:
        return float(roi["cx"]), float(roi["cy"])
    if shape == SHAPE_POLYGON:
        verts = roi.get("vertices", [])
        if not verts:
            return 0.0, 0.0
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        return float(cx), float(cy)
    raise ValueError(f"Unsupported shape {shape!r}")


def point_in_polygon(px: float, py: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    """Standard ray-cast point-in-polygon test."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def point_in_circle(px: float, py: float, cx: float, cy: float, r: float) -> bool:
    return math.hypot(px - cx, py - cy) <= r


def point_to_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> Tuple[float, float]:
    """Distance from point P to segment AB plus the parameter t in [0, 1]
    locating the closest point on AB."""
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y), t


def point_inside_roi(px: float, py: float, roi: dict) -> bool:
    """Hit-test a point against an ROI's body (rotation honoured)."""
    shape = roi["shape"]
    if shape == SHAPE_CIRCLE:
        return point_in_circle(px, py, float(roi["cx"]), float(roi["cy"]), float(roi["r"]))
    if shape in (SHAPE_RECTANGLE, SHAPE_SQUARE):
        return point_in_polygon(px, py, rectangle_corners(roi))
    if shape == SHAPE_POLYGON:
        return point_in_polygon(px, py, polygon_vertices_rotated(roi))
    return False


def insert_polygon_vertex(roi: dict, click_x: float, click_y: float) -> Optional[int]:
    """Insert a vertex on the edge nearest ``(click_x, click_y)``.

    Returns the index of the newly-inserted vertex, or ``None`` if the
    polygon is too short to have any edges (shouldn't happen — caller
    should ensure ≥3 vertices first).
    """
    if roi["shape"] != SHAPE_POLYGON:
        return None
    if len(roi.get("vertices", [])) >= POLYGON_MAX_VERTICES:
        return None
    verts = polygon_vertices_rotated(roi)
    n = len(verts)
    if n < 2:
        return None
    best_dist = float("inf")
    best_edge = 0
    best_t = 0.0
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        dist, t = point_to_segment_distance(click_x, click_y, ax, ay, bx, by)
        if dist < best_dist:
            best_dist = dist
            best_edge = i
            best_t = t
    # Insert in *unrotated* space so the stored polygon reflects the user's
    # intent regardless of current rotation. We unrotate the click point
    # against the current centroid.
    rot = float(roi.get("rotation_deg", 0.0))
    raw_verts = [(float(v[0]), float(v[1])) for v in roi["vertices"]]
    cx = sum(v[0] for v in raw_verts) / len(raw_verts)
    cy = sum(v[1] for v in raw_verts) / len(raw_verts)
    a_raw = raw_verts[best_edge]
    b_raw = raw_verts[(best_edge + 1) % len(raw_verts)]
    new_x = a_raw[0] + best_t * (b_raw[0] - a_raw[0])
    new_y = a_raw[1] + best_t * (b_raw[1] - a_raw[1])
    raw_verts.insert(best_edge + 1, (new_x, new_y))
    roi["vertices"] = [[int(round(vx)), int(round(vy))] for vx, vy in raw_verts]
    _ = rot, cx, cy  # used implicitly above
    return best_edge + 1


def remove_polygon_vertex(roi: dict, index: int) -> bool:
    """Remove vertex ``index`` from the polygon if doing so leaves ≥3 vertices."""
    if roi["shape"] != SHAPE_POLYGON:
        return False
    verts = list(roi.get("vertices", []))
    if len(verts) <= POLYGON_MIN_VERTICES:
        return False
    if not 0 <= index < len(verts):
        return False
    verts.pop(index)
    roi["vertices"] = verts
    return True


def normalised_default_name(existing_names: Sequence[str], stem: str = "ROI") -> str:
    """Pick the next free ``"<stem> N"`` name not in ``existing_names``."""
    idx = 1
    used = set(existing_names)
    while f"{stem} {idx}" in used:
        idx += 1
    return f"{stem} {idx}"


# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_DISPLAY = (1280, 720)
_HANDLE_RADIUS_DISPLAY = 5  # pixels on the canvas
_ROTATION_HANDLE_OFFSET_DISPLAY = 30  # pixels above shape's bbox top in display space
_HIT_TOLERANCE_DISPLAY = 8  # pixels — how close a click must be to a handle/edge


# Selection state — what the user has clicked.
_SELECTION_NONE = 0
_SELECTION_BODY = 1                  # dragging the shape's body
_SELECTION_RESIZE_CORNER = 2         # rectangle corner handle
_SELECTION_RESIZE_RADIUS = 3         # circle radius handle
_SELECTION_VERTEX = 4                # polygon vertex handle
_SELECTION_ROTATION = 5              # rotation handle


# ---------------------------------------------------------------------------
# RoiEditor Toplevel
# ---------------------------------------------------------------------------


class RoiEditor(tk.Toplevel):
    """Unified ROI editor: drag, resize, rotate, vertex-edit one or more ROIs.

    Args:
        parent: Tk widget that owns the editor.
        frame_image_bgr: An OpenCV-style BGR (or BGRA / grayscale) numpy array
            used as the editor's background. Almost certainly the result of a
            ``cv2.VideoCapture(...).read()``. The image is auto-scaled to fit
            ``max_display_size`` while preserving aspect ratio.
        rois: List of ROI dicts in the shape_metadata schema (see module
            docstring). Position values are in source-image pixel space.
        reference_frame_index: Frame index from which ``frame_image_bgr`` was
            grabbed. Round-tripped onto every saved ROI's
            ``reference_frame_index`` field.
        existing_roi_names: Names that already exist in the parent's ROI
            manager outside this batch. Used to warn on rename collisions.
        on_save: Callback invoked with the edited list of ROI dicts when the
            user clicks Save. Editor destroys itself before firing.
        on_cancel: Optional callback when the user cancels (Esc / WM close).
        max_display_size: Maximum (width, height) of the display image in
            pixels.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        frame_image_bgr: "np.ndarray",
        rois: Sequence[dict],
        reference_frame_index: int = 0,
        existing_roi_names: Sequence[str] = (),
        on_save: Callable[[List[dict]], None],
        on_cancel: Optional[Callable[[], None]] = None,
        on_skip: Optional[Callable[[], None]] = None,
        loop_position: Optional[Tuple[int, int]] = None,
        max_display_size: Tuple[int, int] = _DEFAULT_MAX_DISPLAY,
        title: str = "ROI Editor",
    ) -> None:
        """Loop-mode kwargs (PR 2B coherence-pass 2f.1):

        ``on_skip`` and ``loop_position`` activate **loop mode**, used by the
        wizard's "Build ROIs for Queue" flow to step through every video in
        the batch. When set:
          * A "Skip" button appears between Save and Cancel.
          * The window title is annotated ``(N of M)`` so the researcher
            knows where they are in the queue.
        Tab 6's single-video editor leaves both kwargs ``None`` and the UI
        stays unchanged from PR 2B.
        """
        _ensure_imaging_imports()
        super().__init__(parent)
        self._loop_position = loop_position
        if loop_position is not None and len(loop_position) == 2:
            current, total = loop_position
            self.title(f"{title} ({current} of {total})")
        else:
            self.title(title)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel_clicked)

        self._on_save = on_save
        self._on_cancel = on_cancel
        self._on_skip = on_skip
        self._reference_frame_index = int(reference_frame_index)
        self._existing_outside_names = {str(n).strip() for n in existing_roi_names}

        # Deep-copy so editor mutations don't leak back to the caller until Save.
        self._rois: List[dict] = [self._normalise_roi(r) for r in rois]
        self._selected_index: Optional[int] = (0 if self._rois else None)

        # Drag state (set on press, cleared on release).
        self._drag_kind: int = _SELECTION_NONE
        self._drag_handle_index: int = -1   # corner index for rect, vertex index for polygon
        self._drag_anchor_source: Tuple[float, float] = (0.0, 0.0)  # source-image coords
        self._drag_initial_roi: Optional[dict] = None  # snapshot at press-time

        # Render the source image (BGR → RGB → PIL → ImageTk).
        self._original_size, self._scale, self._photo_image = self._prepare_display_image(
            frame_image_bgr, max_display_size
        )

        self._build_ui()
        self._bind_events()
        self._redraw_all()
        self._refresh_listbox_selection()
        self._refresh_status()
        self._centre_on_parent(parent)

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_roi(roi: dict) -> dict:
        """Defensive deepcopy + sanity defaults for each ROI dict."""
        out = {
            "name": str(roi.get("name", "ROI")).strip() or "ROI",
            "shape": str(roi.get("shape", SHAPE_RECTANGLE)).lower(),
            "rotation_deg": float(roi.get("rotation_deg", 0.0)),
            "reference_frame_index": int(roi.get("reference_frame_index", 0)),
        }
        if out["shape"] not in ALLOWED_SHAPES:
            out["shape"] = SHAPE_RECTANGLE
        if out["shape"] in (SHAPE_RECTANGLE, SHAPE_SQUARE):
            out["x"] = int(roi.get("x", 0))
            out["y"] = int(roi.get("y", 0))
            out["w"] = max(2, int(roi.get("w", 100)))
            out["h"] = max(2, int(roi.get("h", 60)))
        elif out["shape"] == SHAPE_CIRCLE:
            out["cx"] = int(roi.get("cx", 0))
            out["cy"] = int(roi.get("cy", 0))
            out["r"] = max(2, int(roi.get("r", 30)))
        elif out["shape"] == SHAPE_POLYGON:
            verts_in = roi.get("vertices", [])
            verts: List[List[int]] = []
            for v in verts_in:
                try:
                    verts.append([int(v[0]), int(v[1])])
                except (TypeError, ValueError, IndexError):
                    continue
            while len(verts) < POLYGON_MIN_VERTICES:
                # Should not happen in normal flow; fail-safe pads with corners.
                verts.append([0, 0])
            out["vertices"] = verts[:POLYGON_MAX_VERTICES]
        return out

    @staticmethod
    def _prepare_display_image(
        frame,  # numpy.ndarray; runtime-typed
        max_display_size: Tuple[int, int],
    ):
        if frame is None:
            raise ValueError("frame_image_bgr is required")
        if frame.ndim == 2:
            rgb = _np.stack([frame] * 3, axis=-1)
        else:
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if cv2 is not None else frame[..., ::-1]
        pil = Image.fromarray(rgb)
        original_size = pil.size  # (W, H)
        max_w, max_h = max_display_size
        scale = min(1.0, max_w / pil.size[0], max_h / pil.size[1])
        if scale < 1.0:
            new_size = (int(round(pil.size[0] * scale)), int(round(pil.size[1] * scale)))
            pil = pil.resize(new_size, Image.BILINEAR)
        else:
            scale = 1.0
        return original_size, scale, ImageTk.PhotoImage(pil)

    def _build_ui(self) -> None:
        self.minsize(800, 500)
        outer = ttk.Frame(self, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0)  # listbox + shortcut panel
        outer.rowconfigure(1, weight=1)

        # Top toolbar.
        toolbar = ttk.Frame(outer)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        toolbar.columnconfigure(0, weight=1)

        self._frame_label_var = tk.StringVar(value=f"Frame: {self._reference_frame_index}")
        ttk.Label(toolbar, textvariable=self._frame_label_var).grid(row=0, column=0, sticky="w")

        self._status_var = tk.StringVar(value="")
        ttk.Label(toolbar, textvariable=self._status_var, style="Subtle.TLabel").grid(
            row=0, column=1, sticky="e", padx=(8, 0)
        )

        # Canvas (centre).
        canvas_frame = ttk.Frame(outer)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        self._canvas = tk.Canvas(
            canvas_frame,
            width=self._photo_image.width(),
            height=self._photo_image.height(),
            highlightthickness=1,
            highlightbackground="#888",
            cursor="arrow",
            background="#222",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo_image, tags="bg")

        # Right-side panel: ROI list + shortcut help.
        right = ttk.Frame(outer)
        right.grid(row=1, column=1, sticky="ns", padx=(8, 0))
        right.rowconfigure(1, weight=1)

        ttk.Label(right, text="ROIs in this batch", anchor="w").grid(row=0, column=0, sticky="ew")
        list_frame = ttk.Frame(right)
        list_frame.grid(row=1, column=0, sticky="nsew")
        self._listbox = tk.Listbox(list_frame, exportselection=False, height=8, width=22)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._listbox.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.configure(yscrollcommand=list_scroll.set)
        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self._refresh_listbox_contents()

        list_buttons = ttk.Frame(right)
        list_buttons.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        list_buttons.columnconfigure(0, weight=1)
        list_buttons.columnconfigure(1, weight=1)
        list_buttons.columnconfigure(2, weight=1)
        list_buttons.columnconfigure(3, weight=1)
        ttk.Button(list_buttons, text="+", width=3, command=self._on_add_clicked).grid(
            row=0, column=0, sticky="ew", padx=1
        )
        ttk.Button(list_buttons, text="Dup", width=4, command=self._on_duplicate_clicked).grid(
            row=0, column=1, sticky="ew", padx=1
        )
        ttk.Button(list_buttons, text="Ren", width=4, command=self._on_rename_clicked).grid(
            row=0, column=2, sticky="ew", padx=1
        )
        ttk.Button(list_buttons, text="Del", width=4, command=self._on_delete_clicked).grid(
            row=0, column=3, sticky="ew", padx=1
        )

        # Shortcut panel (always opens expanded, no preferences file).
        shortcut_frame = ttk.LabelFrame(right, text="Shortcuts", padding=6)
        shortcut_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        for line in (
            "Click body: select / drag",
            "Drag handle: resize",
            "Q / E: rotate −15° / +15°",
            "Shift+Q / Shift+E: −1° / +1°",
            "Right-click edge: insert polygon vertex",
            "Right-click vertex: remove polygon vertex",
            "Ctrl+D: duplicate selected",
            "Delete / Backspace: remove",
            "F2: rename",
            "Ctrl+S: save · Esc: cancel",
        ):
            ttk.Label(shortcut_frame, text=line, anchor="w", justify=tk.LEFT).pack(
                fill=tk.X, pady=1
            )

        # Bottom button row: save / cancel (+ skip in loop mode).
        button_row = ttk.Frame(outer)
        button_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        button_row.columnconfigure(0, weight=1)
        ttk.Button(button_row, text="Cancel", command=self._on_cancel_clicked).grid(
            row=0, column=1, sticky="e", padx=(0, 4)
        )
        # PR 2B coherence-pass 2f.1: Skip button surfaces only when the
        # caller passed `on_skip` — used by the wizard's queue loop to
        # advance to the next video without saving the current one. Tab 6
        # leaves on_skip=None and this button never appears.
        if self._on_skip is not None:
            ttk.Button(button_row, text="Skip", command=self._on_skip_clicked).grid(
                row=0, column=2, sticky="e", padx=(0, 4)
            )
            save_col = 3
        else:
            save_col = 2
        ttk.Button(
            button_row, text="Save", command=self._on_save_clicked, style="Accent.TButton"
        ).grid(row=0, column=save_col, sticky="e")

    def _bind_events(self) -> None:
        self._canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self._canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self._canvas.bind("<Button-3>", self._on_canvas_right_click)

        self.bind("<Escape>", lambda _e: self._on_cancel_clicked())
        self.bind("<Control-s>", lambda _e: self._on_save_clicked())
        self.bind("<Control-S>", lambda _e: self._on_save_clicked())
        self.bind("<Control-d>", lambda _e: self._on_duplicate_clicked())
        self.bind("<Control-D>", lambda _e: self._on_duplicate_clicked())
        self.bind("<Delete>", lambda _e: self._on_delete_clicked())
        self.bind("<BackSpace>", lambda _e: self._on_delete_clicked())
        self.bind("<F2>", lambda _e: self._on_rename_clicked())

        # Rotation: bind both shifted and unshifted keys at the toplevel.
        for key, delta in (
            ("q", -15.0), ("Q", -15.0),
            ("e", 15.0), ("E", 15.0),
        ):
            self.bind(f"<KeyPress-{key}>", lambda e, d=delta: self._handle_rotate_key(e, d))

    def _centre_on_parent(self, parent: tk.Misc) -> None:
        try:
            self.update_idletasks()
            top = parent.winfo_toplevel() if parent is not None else None
            if top is not None:
                px = top.winfo_rootx()
                py = top.winfo_rooty()
                pw = top.winfo_width()
                ph = top.winfo_height()
            else:
                px = py = 0
                pw = self.winfo_screenwidth()
                ph = self.winfo_screenheight()
            x = px + max(0, (pw - self.winfo_width()) // 2)
            y = py + max(0, (ph - self.winfo_height()) // 2)
            self.geometry(f"+{int(x)}+{int(y)}")
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # Display ↔ source coordinate conversion
    # ------------------------------------------------------------------

    def _src_to_display(self, x: float, y: float) -> Tuple[float, float]:
        return x * self._scale, y * self._scale

    def _display_to_src(self, x: float, y: float) -> Tuple[float, float]:
        return x / self._scale, y / self._scale

    def _clamp_to_frame(self, x: float, y: float) -> Tuple[float, float]:
        ow, oh = self._original_size
        return (max(0.0, min(float(ow - 1), x)), max(0.0, min(float(oh - 1), y)))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _redraw_all(self) -> None:
        self._canvas.delete("overlay")
        for idx, roi in enumerate(self._rois):
            selected = (idx == self._selected_index)
            self._render_roi(roi, selected=selected)

    def _render_roi(self, roi: dict, *, selected: bool) -> None:
        outline = "#FFD53F" if selected else "#1e90ff"
        line_width = 3 if selected else 2
        shape = roi["shape"]

        if shape == SHAPE_CIRCLE:
            cx, cy = self._src_to_display(float(roi["cx"]), float(roi["cy"]))
            r_disp = float(roi["r"]) * self._scale
            self._canvas.create_oval(
                cx - r_disp, cy - r_disp, cx + r_disp, cy + r_disp,
                outline=outline, width=line_width, tags="overlay",
            )
        elif shape in (SHAPE_RECTANGLE, SHAPE_SQUARE):
            corners_src = rectangle_corners(roi)
            display_pts = [self._src_to_display(px, py) for px, py in corners_src]
            flat = [coord for pt in display_pts for coord in pt]
            # Draw closed quadrilateral as four lines so we can show rotation.
            for i in range(4):
                a = display_pts[i]
                b = display_pts[(i + 1) % 4]
                self._canvas.create_line(
                    a[0], a[1], b[0], b[1],
                    fill=outline, width=line_width, tags="overlay",
                )
            _ = flat  # silence linter
        elif shape == SHAPE_POLYGON:
            pts_src = polygon_vertices_rotated(roi)
            display_pts = [self._src_to_display(px, py) for px, py in pts_src]
            n = len(display_pts)
            for i in range(n):
                a = display_pts[i]
                b = display_pts[(i + 1) % n]
                self._canvas.create_line(
                    a[0], a[1], b[0], b[1],
                    fill=outline, width=line_width, tags="overlay",
                )

        # Name label near centroid.
        ccx, ccy = shape_centroid(roi)
        cdx, cdy = self._src_to_display(ccx, ccy)
        self._canvas.create_text(
            cdx, cdy,
            text=roi["name"],
            fill=outline,
            font=("Segoe UI", 10, "bold" if selected else "normal"),
            tags="overlay",
        )

        if selected:
            self._render_handles(roi)

    def _render_handles(self, roi: dict) -> None:
        shape = roi["shape"]
        # Resize handles.
        if shape in (SHAPE_RECTANGLE, SHAPE_SQUARE):
            for corner in rectangle_corners(roi):
                cx, cy = self._src_to_display(*corner)
                self._draw_handle_dot(cx, cy)
        elif shape == SHAPE_CIRCLE:
            # Single radius handle to the right of the centre.
            cx_d, cy_d = self._src_to_display(float(roi["cx"]), float(roi["cy"]))
            r_d = float(roi["r"]) * self._scale
            self._draw_handle_dot(cx_d + r_d, cy_d)
        elif shape == SHAPE_POLYGON:
            for vx, vy in polygon_vertices_rotated(roi):
                dx, dy = self._src_to_display(vx, vy)
                self._draw_handle_dot(dx, dy)

        # Rotation handle (everyone except circles).
        if shape != SHAPE_CIRCLE:
            rh_d = self._rotation_handle_display_pos(roi)
            ccx_s, ccy_s = shape_centroid(roi)
            ccx_d, ccy_d = self._src_to_display(ccx_s, ccy_s)
            self._canvas.create_line(
                ccx_d, ccy_d, rh_d[0], rh_d[1],
                fill="#FFD53F", width=1, dash=(3, 2), tags="overlay",
            )
            self._draw_handle_dot(rh_d[0], rh_d[1], fill="#FFD53F")

    def _draw_handle_dot(self, dx: float, dy: float, *, fill: str = "#FFFFFF") -> None:
        r = _HANDLE_RADIUS_DISPLAY
        self._canvas.create_oval(
            dx - r, dy - r, dx + r, dy + r,
            fill=fill, outline="#000000", width=1, tags="overlay",
        )

    def _rotation_handle_display_pos(self, roi: dict) -> Tuple[float, float]:
        """Display-space position of the rotation handle.

        Sits a fixed display-pixel distance above the rotated bbox's top edge
        midpoint, so the handle stays a comfortable distance from the shape
        regardless of zoom.
        """
        shape = roi["shape"]
        rot = float(roi.get("rotation_deg", 0.0))
        if shape in (SHAPE_RECTANGLE, SHAPE_SQUARE):
            # Top-edge midpoint in unrotated source space, then rotate.
            x = float(roi["x"])
            y = float(roi["y"])
            w = float(roi["w"])
            top_mid = (x + w / 2.0, y)
            cx_s, cy_s = shape_centroid(roi)
            top_rot_x, top_rot_y = rotate_point(top_mid[0], top_mid[1], cx_s, cy_s, rot)
            top_d = self._src_to_display(top_rot_x, top_rot_y)
            cd = self._src_to_display(cx_s, cy_s)
        elif shape == SHAPE_POLYGON:
            verts_rot = polygon_vertices_rotated(roi)
            cx_s, cy_s = shape_centroid(roi)
            top_rot_x, top_rot_y = min(verts_rot, key=lambda p: p[1])
            top_d = self._src_to_display(top_rot_x, top_rot_y)
            cd = self._src_to_display(cx_s, cy_s)
        else:
            cx_s, cy_s = shape_centroid(roi)
            cd = self._src_to_display(cx_s, cy_s)
            top_d = (cd[0], cd[1] - 1.0)

        # Vector from centroid to top, normalised to fixed display offset.
        dx = top_d[0] - cd[0]
        dy = top_d[1] - cd[1]
        norm = math.hypot(dx, dy) or 1.0
        offset = _HANDLE_RADIUS_DISPLAY + _ROTATION_HANDLE_OFFSET_DISPLAY
        return top_d[0] + dx / norm * offset, top_d[1] + dy / norm * offset

    # ------------------------------------------------------------------
    # ROI list / status
    # ------------------------------------------------------------------

    def _refresh_listbox_contents(self) -> None:
        self._listbox.delete(0, tk.END)
        for roi in self._rois:
            label = f"{roi['name']}  [{roi['shape']}]"
            self._listbox.insert(tk.END, label)

    def _refresh_listbox_selection(self) -> None:
        self._listbox.selection_clear(0, tk.END)
        if self._selected_index is not None and 0 <= self._selected_index < len(self._rois):
            self._listbox.selection_set(self._selected_index)
            self._listbox.see(self._selected_index)

    def _refresh_status(self) -> None:
        if self._selected_index is None or not self._rois:
            self._status_var.set("Click an ROI to select; drag to move.")
            return
        roi = self._rois[self._selected_index]
        rot = roi.get("rotation_deg", 0.0)
        self._status_var.set(
            f"Selected: {roi['name']} ({roi['shape']}, {rot:.1f}°)"
        )

    def _on_listbox_select(self, _event: tk.Event) -> None:
        sel = self._listbox.curselection()
        if not sel:
            return
        idx = int(sel[0])
        self._selected_index = idx
        self._redraw_all()
        self._refresh_status()

    # ------------------------------------------------------------------
    # Hit-testing
    # ------------------------------------------------------------------

    def _hit_test_at(self, src_x: float, src_y: float) -> Tuple[Optional[int], int, int]:
        """Pick the topmost ROI under the cursor, plus the kind/handle index.

        Returns ``(roi_index, kind, handle_index)``. ``roi_index`` is None if
        nothing was hit.
        """
        # Iterate selected ROI first (handles are only drawn for it). Other
        # ROIs only respond to body clicks.
        if self._selected_index is not None:
            roi = self._rois[self._selected_index]
            kind, handle_idx = self._hit_test_handles(roi, src_x, src_y)
            if kind != _SELECTION_NONE:
                return self._selected_index, kind, handle_idx

        # Body hit-test, top-down (later items render on top, so we iterate
        # in reverse to mimic z-order).
        for idx in reversed(range(len(self._rois))):
            roi = self._rois[idx]
            if point_inside_roi(src_x, src_y, roi):
                return idx, _SELECTION_BODY, -1
        return None, _SELECTION_NONE, -1

    def _hit_test_handles(self, roi: dict, src_x: float, src_y: float) -> Tuple[int, int]:
        tol_src = _HIT_TOLERANCE_DISPLAY / max(self._scale, 1e-6)
        shape = roi["shape"]

        # Rotation handle (display-space hit-test for stable feel).
        if shape != SHAPE_CIRCLE:
            rx_d, ry_d = self._rotation_handle_display_pos(roi)
            click_dx, click_dy = self._src_to_display(src_x, src_y)
            if math.hypot(click_dx - rx_d, click_dy - ry_d) <= _HIT_TOLERANCE_DISPLAY:
                return _SELECTION_ROTATION, -1

        if shape in (SHAPE_RECTANGLE, SHAPE_SQUARE):
            for i, corner in enumerate(rectangle_corners(roi)):
                if math.hypot(corner[0] - src_x, corner[1] - src_y) <= tol_src:
                    return _SELECTION_RESIZE_CORNER, i
        elif shape == SHAPE_CIRCLE:
            # Radius handle is at (cx + r, cy) in source space.
            cx = float(roi["cx"]); cy = float(roi["cy"]); r = float(roi["r"])
            if math.hypot((cx + r) - src_x, cy - src_y) <= tol_src:
                return _SELECTION_RESIZE_RADIUS, 0
        elif shape == SHAPE_POLYGON:
            for i, (vx, vy) in enumerate(polygon_vertices_rotated(roi)):
                if math.hypot(vx - src_x, vy - src_y) <= tol_src:
                    return _SELECTION_VERTEX, i
        return _SELECTION_NONE, -1

    # ------------------------------------------------------------------
    # Mouse: press / drag / release
    # ------------------------------------------------------------------

    def _on_canvas_press(self, event: tk.Event) -> None:
        # Focus self so keyboard shortcuts route here even if listbox had focus.
        self.focus_set()
        src_x, src_y = self._display_to_src(event.x, event.y)
        roi_idx, kind, handle_idx = self._hit_test_at(src_x, src_y)
        if roi_idx is None:
            self._selected_index = None
            self._redraw_all()
            self._refresh_listbox_selection()
            self._refresh_status()
            return
        self._selected_index = roi_idx
        self._drag_kind = kind
        self._drag_handle_index = handle_idx
        self._drag_anchor_source = (src_x, src_y)
        # Snapshot current geometry so resize/rotation maths are stable.
        self._drag_initial_roi = self._copy_roi(self._rois[roi_idx])
        self._refresh_listbox_selection()
        self._redraw_all()
        self._refresh_status()

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self._selected_index is None or self._drag_kind == _SELECTION_NONE:
            return
        src_x, src_y = self._display_to_src(event.x, event.y)
        src_x, src_y = self._clamp_to_frame(src_x, src_y)
        if self._drag_kind == _SELECTION_BODY:
            self._drag_translate(src_x, src_y)
        elif self._drag_kind == _SELECTION_RESIZE_CORNER:
            self._drag_resize_corner(src_x, src_y)
        elif self._drag_kind == _SELECTION_RESIZE_RADIUS:
            self._drag_resize_radius(src_x, src_y)
        elif self._drag_kind == _SELECTION_VERTEX:
            self._drag_polygon_vertex(src_x, src_y)
        elif self._drag_kind == _SELECTION_ROTATION:
            self._drag_rotation(src_x, src_y)
        self._redraw_all()
        self._refresh_status()

    def _on_canvas_release(self, _event: tk.Event) -> None:
        self._drag_kind = _SELECTION_NONE
        self._drag_handle_index = -1
        self._drag_initial_roi = None

    def _on_canvas_right_click(self, event: tk.Event) -> None:
        if self._selected_index is None:
            return
        roi = self._rois[self._selected_index]
        if roi["shape"] != SHAPE_POLYGON:
            return
        src_x, src_y = self._display_to_src(event.x, event.y)
        # Vertex hit first → remove. Otherwise insert on the nearest edge.
        kind, idx = self._hit_test_handles(roi, src_x, src_y)
        if kind == _SELECTION_VERTEX:
            if remove_polygon_vertex(roi, idx):
                self._redraw_all()
            else:
                messagebox.showinfo(
                    "Polygon vertex",
                    "Polygons must have at least 3 vertices.",
                    parent=self,
                )
        else:
            if insert_polygon_vertex(roi, src_x, src_y) is not None:
                self._redraw_all()

    # ------------------------------------------------------------------
    # Drag operations (operate on self._rois[self._selected_index])
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_roi(roi: dict) -> dict:
        out = dict(roi)
        if "vertices" in roi:
            out["vertices"] = [list(v) for v in roi["vertices"]]
        return out

    def _drag_translate(self, src_x: float, src_y: float) -> None:
        if self._drag_initial_roi is None:
            return
        anchor_x, anchor_y = self._drag_anchor_source
        dx = src_x - anchor_x
        dy = src_y - anchor_y
        roi = self._rois[self._selected_index]  # type: ignore[index]
        initial = self._drag_initial_roi
        if roi["shape"] in (SHAPE_RECTANGLE, SHAPE_SQUARE):
            roi["x"] = int(round(float(initial["x"]) + dx))
            roi["y"] = int(round(float(initial["y"]) + dy))
        elif roi["shape"] == SHAPE_CIRCLE:
            roi["cx"] = int(round(float(initial["cx"]) + dx))
            roi["cy"] = int(round(float(initial["cy"]) + dy))
        elif roi["shape"] == SHAPE_POLYGON:
            roi["vertices"] = [
                [int(round(v[0] + dx)), int(round(v[1] + dy))]
                for v in initial["vertices"]
            ]

    def _drag_resize_corner(self, src_x: float, src_y: float) -> None:
        if self._drag_initial_roi is None:
            return
        roi = self._rois[self._selected_index]  # type: ignore[index]
        if roi["shape"] not in (SHAPE_RECTANGLE, SHAPE_SQUARE):
            return
        # Resize semantics: drag the picked corner; the diagonally-opposite
        # corner stays anchored. To keep maths simple we ignore rotation
        # during resize (operate in unrotated bbox space). The shape's
        # rotation is preserved.
        initial = self._drag_initial_roi
        x0 = float(initial["x"])
        y0 = float(initial["y"])
        x1 = x0 + float(initial["w"])
        y1 = y0 + float(initial["h"])
        # Map src (rotated-aware) click back into unrotated frame.
        cx = x0 + (x1 - x0) / 2.0
        cy = y0 + (y1 - y0) / 2.0
        rot = float(initial.get("rotation_deg", 0.0))
        ux, uy = rotate_point(src_x, src_y, cx, cy, -rot)
        # Which corner is being dragged? handle_index uses
        # rectangle_corners ordering: tl, tr, br, bl.
        handle = self._drag_handle_index
        if handle == 0:    # tl
            new_x0, new_y0, new_x1, new_y1 = ux, uy, x1, y1
        elif handle == 1:  # tr
            new_x0, new_y0, new_x1, new_y1 = x0, uy, ux, y1
        elif handle == 2:  # br
            new_x0, new_y0, new_x1, new_y1 = x0, y0, ux, uy
        elif handle == 3:  # bl
            new_x0, new_y0, new_x1, new_y1 = ux, y0, x1, uy
        else:
            return
        # Keep min size of 4 px.
        if abs(new_x1 - new_x0) < 4 or abs(new_y1 - new_y0) < 4:
            return
        x_lo, x_hi = min(new_x0, new_x1), max(new_x0, new_x1)
        y_lo, y_hi = min(new_y0, new_y1), max(new_y0, new_y1)
        if roi["shape"] == SHAPE_SQUARE:
            side = min(x_hi - x_lo, y_hi - y_lo)
            x_hi = x_lo + side
            y_hi = y_lo + side
        roi["x"] = int(round(x_lo))
        roi["y"] = int(round(y_lo))
        roi["w"] = int(round(x_hi - x_lo))
        roi["h"] = int(round(y_hi - y_lo))

    def _drag_resize_radius(self, src_x: float, src_y: float) -> None:
        if self._drag_initial_roi is None:
            return
        roi = self._rois[self._selected_index]  # type: ignore[index]
        if roi["shape"] != SHAPE_CIRCLE:
            return
        cx = float(self._drag_initial_roi["cx"])
        cy = float(self._drag_initial_roi["cy"])
        new_r = max(2.0, math.hypot(src_x - cx, src_y - cy))
        roi["r"] = int(round(new_r))

    def _drag_polygon_vertex(self, src_x: float, src_y: float) -> None:
        if self._drag_initial_roi is None:
            return
        roi = self._rois[self._selected_index]  # type: ignore[index]
        if roi["shape"] != SHAPE_POLYGON:
            return
        idx = self._drag_handle_index
        verts = list(roi["vertices"])
        if not 0 <= idx < len(verts):
            return
        # Inverse-rotate the click into unrotated polygon space for storage.
        rot = float(roi.get("rotation_deg", 0.0))
        cx, cy = shape_centroid(roi)
        ux, uy = rotate_point(src_x, src_y, cx, cy, -rot)
        verts[idx] = [int(round(ux)), int(round(uy))]
        roi["vertices"] = verts

    def _drag_rotation(self, src_x: float, src_y: float) -> None:
        if self._drag_initial_roi is None:
            return
        roi = self._rois[self._selected_index]  # type: ignore[index]
        if roi["shape"] == SHAPE_CIRCLE:
            return
        cx, cy = shape_centroid(self._drag_initial_roi)
        # Initial drag-anchor angle (where the user grabbed the rotation
        # handle) → press-time rotation. New angle from cursor → updated.
        anchor_x, anchor_y = self._drag_anchor_source
        a0 = math.degrees(math.atan2(anchor_y - cy, anchor_x - cx))
        a1 = math.degrees(math.atan2(src_y - cy, src_x - cx))
        delta = a1 - a0
        roi["rotation_deg"] = (float(self._drag_initial_roi.get("rotation_deg", 0.0)) + delta) % 360.0

    # ------------------------------------------------------------------
    # Keyboard: rotation (Q/E)
    # ------------------------------------------------------------------

    def _handle_rotate_key(self, event: tk.Event, base_delta: float) -> None:
        if self._selected_index is None:
            return
        roi = self._rois[self._selected_index]
        if roi["shape"] == SHAPE_CIRCLE:
            return
        delta = base_delta
        # Tk reports Shift via the state bit (0x0001).
        if (event.state & 0x0001) and not (event.state & 0x0004):
            delta = base_delta / 15.0  # ±1° fine adjust
        roi["rotation_deg"] = (float(roi.get("rotation_deg", 0.0)) + delta) % 360.0
        self._redraw_all()
        self._refresh_status()

    # ------------------------------------------------------------------
    # ROI list mutations: add / duplicate / delete / rename
    # ------------------------------------------------------------------

    def _all_taken_names(self) -> List[str]:
        return [r["name"] for r in self._rois] + list(self._existing_outside_names)

    def _on_add_clicked(self) -> None:
        # Add a default rectangle at the frame centre.
        ow, oh = self._original_size
        size = min(ow, oh) // 6 or 50
        new_roi = {
            "name": normalised_default_name(self._all_taken_names()),
            "shape": SHAPE_RECTANGLE,
            "rotation_deg": 0.0,
            "reference_frame_index": self._reference_frame_index,
            "x": int(ow // 2 - size),
            "y": int(oh // 2 - size // 2),
            "w": int(2 * size),
            "h": int(size),
        }
        self._rois.append(new_roi)
        self._selected_index = len(self._rois) - 1
        self._refresh_listbox_contents()
        self._refresh_listbox_selection()
        self._redraw_all()
        self._refresh_status()

    def _on_duplicate_clicked(self) -> None:
        if self._selected_index is None:
            return
        original = self._copy_roi(self._rois[self._selected_index])
        original["name"] = normalised_default_name(
            self._all_taken_names(), stem=f"{original['name']} copy"
        )
        # Offset so the duplicate isn't perfectly hidden under the original.
        offset = max(8, int(min(self._original_size) * 0.02))
        if original["shape"] in (SHAPE_RECTANGLE, SHAPE_SQUARE):
            original["x"] += offset
            original["y"] += offset
        elif original["shape"] == SHAPE_CIRCLE:
            original["cx"] += offset
            original["cy"] += offset
        elif original["shape"] == SHAPE_POLYGON:
            original["vertices"] = [[v[0] + offset, v[1] + offset] for v in original["vertices"]]
        self._rois.append(original)
        self._selected_index = len(self._rois) - 1
        self._refresh_listbox_contents()
        self._refresh_listbox_selection()
        self._redraw_all()
        self._refresh_status()

    def _on_delete_clicked(self) -> None:
        if self._selected_index is None:
            return
        if len(self._rois) <= 1:
            # Editor must have at least one ROI to be a meaningful editor.
            # If the user wants none, they should Cancel.
            messagebox.showinfo(
                "ROI Editor",
                "At least one ROI must remain. Click Cancel to discard the batch.",
                parent=self,
            )
            return
        del self._rois[self._selected_index]
        self._selected_index = max(0, self._selected_index - 1)
        self._refresh_listbox_contents()
        self._refresh_listbox_selection()
        self._redraw_all()
        self._refresh_status()

    def _on_rename_clicked(self) -> None:
        if self._selected_index is None:
            return
        roi = self._rois[self._selected_index]
        current = roi["name"]
        new_name = simpledialog.askstring(
            "Rename ROI",
            "New name:",
            initialvalue=current,
            parent=self,
        )
        if new_name is None:
            return
        new_name = new_name.strip()
        if not new_name:
            messagebox.showwarning("Rename ROI", "Name cannot be blank.", parent=self)
            return
        # Refuse to collide with another ROI in this batch or with names
        # outside this editor's scope.
        existing = self._all_taken_names()
        existing.remove(current)  # current name is OK
        if new_name in existing:
            messagebox.showwarning(
                "Rename ROI",
                f"An ROI named {new_name!r} already exists.",
                parent=self,
            )
            return
        roi["name"] = new_name
        self._refresh_listbox_contents()
        self._refresh_listbox_selection()
        self._redraw_all()
        self._refresh_status()

    # ------------------------------------------------------------------
    # Save / skip / cancel
    # ------------------------------------------------------------------

    def _on_skip_clicked(self) -> None:
        """Loop mode only — skip to the next video without saving this one.

        PR 2B coherence-pass 2f.1. Builds no payload, just signals the
        controller to advance.
        """
        cb = self._on_skip
        try:
            self.destroy()
        finally:
            if cb is not None:
                try:
                    cb()
                except Exception:  # pragma: no cover
                    import traceback
                    traceback.print_exc()

    def _on_save_clicked(self) -> None:
        # Stamp reference frame index onto each ROI so re-opening uses the
        # same background frame the user just edited against.
        for roi in self._rois:
            roi["reference_frame_index"] = int(self._reference_frame_index)
        out = [self._copy_roi(r) for r in self._rois]
        cb = self._on_save
        try:
            self.destroy()
        finally:
            try:
                cb(out)
            except Exception:  # pragma: no cover - propagate via stderr
                import traceback
                traceback.print_exc()

    def _on_cancel_clicked(self) -> None:
        cb = self._on_cancel
        try:
            self.destroy()
        finally:
            if cb is not None:
                try:
                    cb()
                except Exception:  # pragma: no cover
                    import traceback
                    traceback.print_exc()


__all__ = [
    "RoiEditor",
    # Helpers exposed so they can be unit-tested.
    "rotate_point",
    "rectangle_corners",
    "polygon_vertices_rotated",
    "shape_centroid",
    "point_in_polygon",
    "point_in_circle",
    "point_inside_roi",
    "point_to_segment_distance",
    "insert_polygon_vertex",
    "remove_polygon_vertex",
    "normalised_default_name",
]

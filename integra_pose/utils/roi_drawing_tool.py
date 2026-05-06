from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Optional, Tuple
import math

import numpy as np

try:  # Prefer OpenCV for accurate colour conversion if available
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover - plugin optional dependency
    raise ImportError("Pillow is required for ROI drawing") from exc

Point = Tuple[int, int]


class _ROIDialog(tk.Toplevel):
    """Modal dialog that lets the user draw a polygon on top of an image."""

    def __init__(
        self,
        master: tk.Misc,
        photo_image: ImageTk.PhotoImage,
        original_size: Tuple[int, int],
        scale: float,
        title: str = "Draw ROI",
    ) -> None:
        super().__init__(master)
        self.transient(master)
        self.title(title)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self._photo = photo_image
        self._original_width, self._original_height = original_size
        self._scale = scale
        self._points: list[Point] = []
        self.result: list[Point] = []

        self._build_ui()
        self._bind_events()
        self._centre_on_parent()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        instructions = ttk.Label(
            self,
            text="Left-click to add vertices. Right-click or Backspace to undo. "
            "Press Enter or click Finish when done, Esc to cancel.",
            wraplength=420,
            justify=tk.LEFT,
        )
        instructions.pack(fill=tk.X, padx=12, pady=(12, 8))

        canvas_width = self._photo.width()
        canvas_height = self._photo.height()

        self._canvas = tk.Canvas(
            self,
            width=canvas_width,
            height=canvas_height,
            highlightthickness=1,
            highlightbackground="#888",
            cursor="tcross",
        )
        self._canvas.pack(padx=12, pady=(0, 12))
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo, tags="frame")

        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=12, pady=(0, 12))

        self._undo_button = ttk.Button(button_frame, text="Undo", command=self._undo_last_point)
        self._undo_button.pack(side=tk.LEFT)

        ttk.Button(button_frame, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(button_frame, text="Finish", command=self._finish).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Cancel", command=self._cancel).pack(side=tk.RIGHT, padx=(0, 6))

    def _bind_events(self) -> None:
        self._canvas.bind("<Button-1>", self._on_left_click)
        self._canvas.bind("<Button-3>", self._on_right_click)
        self.bind("<Return>", lambda event: self._finish())
        self.bind("<Escape>", lambda event: self._cancel())
        self.bind("<BackSpace>", lambda event: self._undo_last_point())

    def _centre_on_parent(self) -> None:
        try:
            self.update_idletasks()
            parent = self.master.winfo_toplevel() if self.master else None
            if parent:
                px = parent.winfo_rootx()
                py = parent.winfo_rooty()
                pw = parent.winfo_width()
                ph = parent.winfo_height()
            else:
                screen_w = self.winfo_screenwidth()
                screen_h = self.winfo_screenheight()
                px = (screen_w - self.winfo_width()) // 2
                py = (screen_h - self.winfo_height()) // 2
                pw = self.winfo_width()
                ph = self.winfo_height()

            x = px + (pw - self.winfo_width()) // 2
            y = py + (ph - self.winfo_height()) // 2
            self.geometry(f"+{max(int(x), 0)}+{max(int(y), 0)}")
        except Exception:  # pragma: no cover - centering best-effort
            pass

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        self._canvas.delete("overlay")
        if not self._points:
            self._undo_button.state(["disabled"])
            return

        self._undo_button.state(["!disabled"])

        display_points = [
            (point[0] * self._scale, point[1] * self._scale) for point in self._points
        ]

        # Draw polygon edges
        if len(display_points) > 1:
            flat_coords = [coord for point in display_points for coord in point]
            self._canvas.create_line(
                *flat_coords,
                fill="#1e90ff",
                width=2,
                tags="overlay",
            )

        # Draw closing edge hint
        if len(display_points) > 2:
            first = display_points[0]
            last = display_points[-1]
            self._canvas.create_line(
                last[0],
                last[1],
                first[0],
                first[1],
                fill="#1e90ff",
                dash=(4, 2),
                width=1,
                tags="overlay",
            )

        # Draw vertices
        for x, y in display_points:
            self._canvas.create_oval(
                x - 3,
                y - 3,
                x + 3,
                y + 3,
                fill="#1e90ff",
                outline="white",
                width=1,
                tags="overlay",
            )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_left_click(self, event: tk.Event) -> None:
        x = max(0, min(event.x, self._photo.width() - 1))
        y = max(0, min(event.y, self._photo.height() - 1))
        original_x = int(round(x / self._scale))
        original_y = int(round(y / self._scale))
        original_x = max(0, min(original_x, self._original_width - 1))
        original_y = max(0, min(original_y, self._original_height - 1))
        self._points.append((original_x, original_y))
        self._redraw()

    def _on_right_click(self, event: tk.Event) -> None:
        self._undo_last_point()

    def _undo_last_point(self) -> None:
        if self._points:
            self._points.pop()
            self._redraw()

    def _reset(self) -> None:
        self._points.clear()
        self._redraw()

    # ------------------------------------------------------------------
    # Dialog lifecycle
    # ------------------------------------------------------------------

    def _finish(self) -> None:
        if len(self._points) < 3:
            messagebox.showwarning(
                "Draw ROI",
                "Please define at least three vertices before finishing the polygon.",
                parent=self,
            )
            return
        self.result = list(self._points)
        self.destroy()

    def _cancel(self) -> None:
        self.result = []
        self.destroy()


class _ObjectROIDialog(tk.Toplevel):
    """Modal dialog that places fixed-size object ROIs by center clicks."""

    def __init__(
        self,
        master: tk.Misc,
        photo_image: ImageTk.PhotoImage,
        original_size: Tuple[int, int],
        scale: float,
        *,
        object_names: List[str],
        roi_size_px: int,
        shape: str,
        title: str = "Place Object ROIs",
    ) -> None:
        super().__init__(master)
        self.transient(master)
        self.title(title)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self._photo = photo_image
        self._original_width, self._original_height = original_size
        self._scale = scale
        self._object_names = list(object_names)
        self._roi_size_px = max(2, int(roi_size_px))
        self._shape = str(shape or "circle").strip().lower()
        if self._shape not in {"circle", "square"}:
            self._shape = "circle"
        self._centers: list[Point] = []
        self.result: list[list[Point]] = []
        self._status_var = tk.StringVar(value="")

        self._build_ui()
        self._bind_events()
        self._update_status()
        self._centre_on_parent()

    def _build_ui(self) -> None:
        instructions = ttk.Label(
            self,
            text=(
                "Click the center of each object in order. "
                "Right-click or Backspace to undo. Enter to finish when all objects are placed."
            ),
            wraplength=460,
            justify=tk.LEFT,
        )
        instructions.pack(fill=tk.X, padx=12, pady=(12, 4))

        ttk.Label(self, textvariable=self._status_var, justify=tk.LEFT).pack(fill=tk.X, padx=12, pady=(0, 8))

        canvas_width = self._photo.width()
        canvas_height = self._photo.height()
        self._canvas = tk.Canvas(
            self,
            width=canvas_width,
            height=canvas_height,
            highlightthickness=1,
            highlightbackground="#888",
            cursor="crosshair",
        )
        self._canvas.pack(padx=12, pady=(0, 12))
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo, tags="frame")

        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=12, pady=(0, 12))
        self._undo_button = ttk.Button(button_frame, text="Undo", command=self._undo_last_point)
        self._undo_button.pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(button_frame, text="Finish", command=self._finish).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Cancel", command=self._cancel).pack(side=tk.RIGHT, padx=(0, 6))

    def _bind_events(self) -> None:
        self._canvas.bind("<Button-1>", self._on_left_click)
        self._canvas.bind("<Button-3>", self._on_right_click)
        self.bind("<Return>", lambda event: self._finish())
        self.bind("<Escape>", lambda event: self._cancel())
        self.bind("<BackSpace>", lambda event: self._undo_last_point())

    def _centre_on_parent(self) -> None:
        try:
            self.update_idletasks()
            parent = self.master.winfo_toplevel() if self.master else None
            if parent:
                px = parent.winfo_rootx()
                py = parent.winfo_rooty()
                pw = parent.winfo_width()
                ph = parent.winfo_height()
            else:
                screen_w = self.winfo_screenwidth()
                screen_h = self.winfo_screenheight()
                px = (screen_w - self.winfo_width()) // 2
                py = (screen_h - self.winfo_height()) // 2
                pw = self.winfo_width()
                ph = self.winfo_height()
            x = px + (pw - self.winfo_width()) // 2
            y = py + (ph - self.winfo_height()) // 2
            self.geometry(f"+{max(int(x), 0)}+{max(int(y), 0)}")
        except Exception:  # pragma: no cover
            pass

    def _shape_outline(self, center: Point) -> list[Point]:
        cx, cy = center
        radius = max(1.0, float(self._roi_size_px) / 2.0)
        if self._shape == "square":
            x1 = int(round(cx - radius))
            y1 = int(round(cy - radius))
            x2 = int(round(cx + radius))
            y2 = int(round(cy + radius))
            x1 = max(0, min(x1, self._original_width - 1))
            x2 = max(0, min(x2, self._original_width - 1))
            y1 = max(0, min(y1, self._original_height - 1))
            y2 = max(0, min(y2, self._original_height - 1))
            return [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]

        sample_count = 24
        points: list[Point] = []
        for idx in range(sample_count):
            angle = (2.0 * math.pi * idx) / float(sample_count)
            x = int(round(cx + radius * math.cos(angle)))
            y = int(round(cy + radius * math.sin(angle)))
            x = max(0, min(x, self._original_width - 1))
            y = max(0, min(y, self._original_height - 1))
            points.append((x, y))
        return points

    def _update_status(self) -> None:
        total = len(self._object_names)
        placed = len(self._centers)
        next_name = self._object_names[placed] if placed < total else "All objects placed"
        self._status_var.set(f"Placed {placed}/{total}. Next: {next_name}")

    def _redraw(self) -> None:
        self._canvas.delete("overlay")
        self._undo_button.state(["!disabled"] if self._centers else ["disabled"])
        radius_display = (float(self._roi_size_px) / 2.0) * self._scale
        for idx, center in enumerate(self._centers):
            cx, cy = center
            dx = cx * self._scale
            dy = cy * self._scale
            label = self._object_names[idx] if idx < len(self._object_names) else f"Object {idx + 1}"
            if self._shape == "square":
                self._canvas.create_rectangle(
                    dx - radius_display,
                    dy - radius_display,
                    dx + radius_display,
                    dy + radius_display,
                    outline="#1e90ff",
                    width=2,
                    tags="overlay",
                )
            else:
                self._canvas.create_oval(
                    dx - radius_display,
                    dy - radius_display,
                    dx + radius_display,
                    dy + radius_display,
                    outline="#1e90ff",
                    width=2,
                    tags="overlay",
                )
            self._canvas.create_oval(dx - 2, dy - 2, dx + 2, dy + 2, fill="#1e90ff", outline="white", tags="overlay")
            self._canvas.create_text(
                dx + 6,
                dy - 6,
                text=label,
                anchor=tk.SW,
                fill="#1e90ff",
                font=("Segoe UI", 9, "bold"),
                tags="overlay",
            )

    def _on_left_click(self, event: tk.Event) -> None:
        if len(self._centers) >= len(self._object_names):
            self.bell()
            return
        x = max(0, min(event.x, self._photo.width() - 1))
        y = max(0, min(event.y, self._photo.height() - 1))
        original_x = int(round(x / self._scale))
        original_y = int(round(y / self._scale))
        original_x = max(0, min(original_x, self._original_width - 1))
        original_y = max(0, min(original_y, self._original_height - 1))
        self._centers.append((original_x, original_y))
        self._update_status()
        self._redraw()

    def _on_right_click(self, _event: tk.Event) -> None:
        self._undo_last_point()

    def _undo_last_point(self) -> None:
        if self._centers:
            self._centers.pop()
            self._update_status()
            self._redraw()

    def _reset(self) -> None:
        self._centers.clear()
        self._update_status()
        self._redraw()

    def _finish(self) -> None:
        expected = len(self._object_names)
        if len(self._centers) != expected:
            messagebox.showwarning(
                "Place Object ROIs",
                f"Place all objects first ({len(self._centers)}/{expected}).",
                parent=self,
            )
            return
        self.result = [self._shape_outline(center) for center in self._centers]
        self.destroy()

    def _cancel(self) -> None:
        self.result = []
        self.destroy()


class _BoxROIDialog(tk.Toplevel):
    """Modal dialog that lets the user draw a rectangle or circle ROI on top of an image."""

    def __init__(
        self,
        master: tk.Misc,
        photo_image: ImageTk.PhotoImage,
        original_size: Tuple[int, int],
        scale: float,
        *,
        shape: str,
        title: str = "Draw ROI",
    ) -> None:
        super().__init__(master)
        self.transient(master)
        self.title(title)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self._photo = photo_image
        self._original_width, self._original_height = original_size
        self._scale = scale
        self._shape = str(shape or "rectangle").strip().lower()
        if self._shape not in {"rectangle", "circle"}:
            self._shape = "rectangle"

        self._start_display: Optional[Tuple[float, float]] = None
        self._current_box: Optional[Tuple[int, int, int, int]] = None
        self.result: Optional[Tuple[int, int, int, int]] = None

        self._build_ui()
        self._bind_events()
        self._centre_on_parent()
        self.focus_force()

    def _build_ui(self) -> None:
        instructions = ttk.Label(
            self,
            text=(
                "Click and drag to define the ROI. "
                "Release to preview. Press Enter or click Finish to accept, Esc to cancel."
            ),
            wraplength=460,
            justify=tk.LEFT,
        )
        instructions.pack(fill=tk.X, padx=12, pady=(12, 8))

        canvas_width = self._photo.width()
        canvas_height = self._photo.height()
        self._canvas = tk.Canvas(
            self,
            width=canvas_width,
            height=canvas_height,
            highlightthickness=1,
            highlightbackground="#888",
            cursor="crosshair",
        )
        self._canvas.pack(padx=12, pady=(0, 12))
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo, tags="frame")

        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=12, pady=(0, 12))
        ttk.Button(button_frame, text="Reset", command=self._reset).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Finish", command=self._finish).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Cancel", command=self._cancel).pack(side=tk.RIGHT, padx=(0, 6))

    def _bind_events(self) -> None:
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Return>", lambda event: self._finish())
        self.bind("<Escape>", lambda event: self._cancel())

    def _centre_on_parent(self) -> None:
        try:
            self.update_idletasks()
            parent = self.master.winfo_toplevel() if self.master else None
            if parent:
                px = parent.winfo_rootx()
                py = parent.winfo_rooty()
                pw = parent.winfo_width()
                ph = parent.winfo_height()
            else:
                screen_w = self.winfo_screenwidth()
                screen_h = self.winfo_screenheight()
                px = (screen_w - self.winfo_width()) // 2
                py = (screen_h - self.winfo_height()) // 2
                pw = self.winfo_width()
                ph = self.winfo_height()
            x = px + (pw - self.winfo_width()) // 2
            y = py + (ph - self.winfo_height()) // 2
            self.geometry(f"+{max(int(x), 0)}+{max(int(y), 0)}")
        except Exception:  # pragma: no cover
            pass

    def _on_press(self, event: tk.Event) -> None:
        self._start_display = (
            float(max(0, min(event.x, self._photo.width() - 1))),
            float(max(0, min(event.y, self._photo.height() - 1))),
        )

    def _on_drag(self, event: tk.Event) -> None:
        if self._start_display is None:
            return
        self._update_box(event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        if self._start_display is None:
            return
        self._update_box(event.x, event.y)
        self._start_display = None

    def _display_to_original(self, x: float, y: float) -> Tuple[int, int]:
        original_x = int(round(float(x) / self._scale))
        original_y = int(round(float(y) / self._scale))
        original_x = max(0, min(original_x, self._original_width - 1))
        original_y = max(0, min(original_y, self._original_height - 1))
        return original_x, original_y

    def _update_box(self, x: float, y: float) -> None:
        if self._start_display is None:
            return
        sx, sy = self._start_display
        ex = float(max(0, min(x, self._photo.width() - 1)))
        ey = float(max(0, min(y, self._photo.height() - 1)))
        x0, y0 = self._display_to_original(min(sx, ex), min(sy, ey))
        x1, y1 = self._display_to_original(max(sx, ex), max(sy, ey))
        width = max(int(x1 - x0), 4)
        height = max(int(y1 - y0), 4)
        if self._shape == "circle":
            size = max(4, min(width, height))
            self._current_box = (x0, y0, size, size)
        else:
            self._current_box = (x0, y0, width, height)
        self._redraw()

    def _redraw(self) -> None:
        self._canvas.delete("overlay")
        if self._current_box is None:
            return
        x, y, w, h = self._current_box
        dx = x * self._scale
        dy = y * self._scale
        dw = w * self._scale
        dh = h * self._scale
        if self._shape == "circle":
            self._canvas.create_oval(
                dx,
                dy,
                dx + dw,
                dy + dh,
                outline="#1e90ff",
                width=2,
                tags="overlay",
            )
        else:
            self._canvas.create_rectangle(
                dx,
                dy,
                dx + dw,
                dy + dh,
                outline="#1e90ff",
                width=2,
                tags="overlay",
            )

    def _reset(self) -> None:
        self._start_display = None
        self._current_box = None
        self._redraw()

    def _finish(self) -> None:
        if self._current_box is None:
            messagebox.showwarning(
                "Draw ROI",
                "Please draw an ROI before finishing.",
                parent=self,
            )
            return
        self.result = self._current_box
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


def _frame_to_rgb_image(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        raise ValueError("Frame is required to draw ROI.")
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError("Frame must be an HxWx3 image array.")

    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    if cv2 is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame[..., ::-1]


def _resize_image(image: Image.Image, max_width: int, max_height: int) -> Tuple[Image.Image, float]:
    width, height = image.size
    if width == 0 or height == 0:
        raise ValueError("Frame has invalid dimensions.")
    scale = min(1.0, max_width / width, max_height / height)
    if scale >= 0.999:
        return image, 1.0
    resized = image.resize((int(round(width * scale)), int(round(height * scale))), Image.BILINEAR)
    return resized, scale


def draw_roi(
    frame: np.ndarray,
    *,
    master: Optional[tk.Misc] = None,
    title: str = "Draw ROI",
    max_display_size: Tuple[int, int] = (1280, 720),
) -> List[Point]:
    """Display a modal ROI editor and return a polygon of integer pixel coordinates."""

    rgb_frame = _frame_to_rgb_image(frame)
    pil_image = Image.fromarray(rgb_frame)
    display_image, scale = _resize_image(pil_image, *max_display_size)

    root = master or tk._get_default_root()
    owns_root = False
    if root is None:
        root = tk.Tk()
        root.withdraw()
        owns_root = True

    dialog = _ROIDialog(
        master=root,
        photo_image=ImageTk.PhotoImage(display_image),
        original_size=pil_image.size,
        scale=scale,
        title=title,
    )
    dialog.grab_set()
    dialog.wait_window()

    if owns_root:
        root.destroy()

    return dialog.result


def draw_box_roi(
    frame: np.ndarray,
    *,
    shape: str = "rectangle",
    master: Optional[tk.Misc] = None,
    title: str = "Draw ROI",
    max_display_size: Tuple[int, int] = (1280, 720),
) -> Optional[Tuple[int, int, int, int]]:
    """Display a modal ROI editor and return a rectangle/circle bounding box."""

    rgb_frame = _frame_to_rgb_image(frame)
    pil_image = Image.fromarray(rgb_frame)
    display_image, scale = _resize_image(pil_image, *max_display_size)

    root = master or tk._get_default_root()
    owns_root = False
    if root is None:
        root = tk.Tk()
        root.withdraw()
        owns_root = True

    dialog = _BoxROIDialog(
        master=root,
        photo_image=ImageTk.PhotoImage(display_image),
        original_size=pil_image.size,
        scale=scale,
        shape=shape,
        title=title,
    )
    dialog.grab_set()
    dialog.wait_window()

    if owns_root:
        root.destroy()

    return dialog.result


def draw_object_rois(
    frame: np.ndarray,
    *,
    object_count: int,
    roi_size_px: int = 20,
    shape: str = "circle",
    object_names: Optional[List[str]] = None,
    master: Optional[tk.Misc] = None,
    title: str = "Place Object ROIs",
    max_display_size: Tuple[int, int] = (1280, 720),
) -> List[List[Point]]:
    """Place fixed-size object ROIs by clicking object centers."""

    total = max(0, int(object_count or 0))
    if total <= 0:
        return []
    names = list(object_names or [])
    if len(names) != total:
        names = [f"Object_{idx + 1}" for idx in range(total)]

    rgb_frame = _frame_to_rgb_image(frame)
    pil_image = Image.fromarray(rgb_frame)
    display_image, scale = _resize_image(pil_image, *max_display_size)

    root = master or tk._get_default_root()
    owns_root = False
    if root is None:
        root = tk.Tk()
        root.withdraw()
        owns_root = True

    dialog = _ObjectROIDialog(
        master=root,
        photo_image=ImageTk.PhotoImage(display_image),
        original_size=pil_image.size,
        scale=scale,
        object_names=names,
        roi_size_px=max(2, int(roi_size_px or 20)),
        shape=shape,
        title=title,
    )
    dialog.grab_set()
    dialog.wait_window()

    if owns_root:
        root.destroy()

    return dialog.result


__all__ = ["draw_roi", "draw_box_roi", "draw_object_rois"]

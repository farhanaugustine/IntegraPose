"""Modal dialog: configure N ROIs upfront before drawing.

The researcher fills in name + shape (+ polygon vertex count) for each ROI
they want, clicks *Place on frame*, and the parent surface takes over with
the auto-placed shapes ready for interactive editing.

This module is intentionally callback-based so the parent can decide what
"place on frame" means (open an editor, write into the ROI manager, etc.)
without this module knowing about either.

Public API:

    from integra_pose.gui.roi_builder_dialog import (
        RoiBuilderDialog,
        RoiBuilderRow,
    )

    def _on_confirm(rows: list[RoiBuilderRow]) -> None:
        ...

    RoiBuilderDialog(parent_widget, on_confirm=_on_confirm)
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, field
from tkinter import messagebox, ttk
from typing import Callable, List, Optional, Sequence

from integra_pose.utils.roi_layout import (
    ALLOWED_SHAPES,
    POLYGON_DEFAULT_VERTICES,
    POLYGON_MAX_VERTICES,
    POLYGON_MIN_VERTICES,
    SHAPE_POLYGON,
    SHAPE_RECTANGLE,
)

# Display labels (sentence case) ↔ canonical shape strings (lowercase).
_SHAPE_DISPLAY_TO_KEY = {
    "Rectangle": "rectangle",
    "Square": "square",
    "Circle": "circle",
    "Polygon": "polygon",
}
_SHAPE_KEY_TO_DISPLAY = {v: k for k, v in _SHAPE_DISPLAY_TO_KEY.items()}


@dataclass
class RoiBuilderRow:
    """One row of the builder dialog: a single ROI to be created."""

    name: str = ""
    shape: str = SHAPE_RECTANGLE  # one of ALLOWED_SHAPES
    polygon_vertices: int = POLYGON_DEFAULT_VERTICES

    def __post_init__(self) -> None:
        if self.shape not in ALLOWED_SHAPES:
            self.shape = SHAPE_RECTANGLE
        # Clamp vertex count into the allowed band even if a caller supplied
        # something silly. The dialog widget also clamps at edit time.
        try:
            n = int(self.polygon_vertices)
        except (TypeError, ValueError):
            n = POLYGON_DEFAULT_VERTICES
        self.polygon_vertices = max(
            POLYGON_MIN_VERTICES, min(POLYGON_MAX_VERTICES, n)
        )


@dataclass
class _RowWidgets:
    """Internal: the live widgets backing one builder row."""

    frame: ttk.Frame
    name_var: tk.StringVar
    shape_var: tk.StringVar
    vertex_var: tk.StringVar
    vertex_entry: ttk.Entry
    name_entry: ttk.Entry
    shape_combo: ttk.Combobox
    remove_button: ttk.Button


class RoiBuilderDialog(tk.Toplevel):
    """Upfront configuration modal: number of ROIs, shape per ROI, name per ROI.

    The dialog does *not* draw anything onto a video frame itself — that's the
    editor's job in Commit 4. This dialog just collects the configuration and
    calls ``on_confirm`` with a list of :class:`RoiBuilderRow`.

    Args:
        parent: Tk widget that owns this Toplevel.
        on_confirm: Callable invoked when the researcher clicks Place on frame.
            Receives the validated list of rows; the dialog has already
            destroyed itself by the time this fires, so the callback is free
            to open an editor / show another dialog.
        existing_rows: Optional pre-fill (e.g., re-opening to edit a config
            that wasn't yet committed). The defaults populate one rectangle
            row called "ROI 1".
        existing_roi_names: Names already in use by the project's ROIManager.
            The dialog refuses to confirm rows that collide with these
            (or with each other) so the caller doesn't have to dedupe later.
        title: Window title; defaults to "Build ROIs".
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_confirm: Callable[[List[RoiBuilderRow]], None],
        existing_rows: Sequence[RoiBuilderRow] = (),
        existing_roi_names: Sequence[str] = (),
        title: str = "Build ROIs",
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.resizable(True, True)
        self.minsize(540, 320)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._on_confirm = on_confirm
        self._existing_roi_names = {str(n).strip() for n in existing_roi_names if str(n).strip()}
        self._rows: List[_RowWidgets] = []
        # Track row counter so default names don't collide on Add.
        self._next_default_index = 1

        self._build_ui()

        # Seed the dialog. If existing_rows are provided, honour them verbatim
        # (the caller already picked names). Otherwise fall through to the
        # default-naming path in _add_row() so existing_roi_names is honoured.
        if existing_rows:
            for row in existing_rows:
                self._add_row(initial=row)
        else:
            self._add_row()  # pulls the next free "ROI N" name automatically

        self._centre_on_parent(parent)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        intro = ttk.Label(
            outer,
            text=(
                "Define the ROIs you want to draw. Each row becomes one shape "
                "placed on the frame at a default position; you'll move and "
                "resize them in the editor that opens next."
            ),
            wraplength=520,
            justify=tk.LEFT,
        )
        intro.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        # Header strip mirrors the row layout below for visual alignment.
        header = ttk.Frame(outer)
        header.grid(row=1, column=0, sticky="ew", padx=2)
        for col_index, (text, weight, width) in enumerate(
            (
                ("Name", 1, 18),
                ("Shape", 0, 12),
                ("Polygon vertices", 0, 14),
                ("", 0, 4),
            )
        ):
            label = ttk.Label(header, text=text, anchor="w")
            label.grid(row=0, column=col_index, sticky="ew", padx=(0, 6), pady=(0, 2))
            header.columnconfigure(col_index, weight=weight)

        # Scrollable rows region. Long lists shouldn't blow out the dialog.
        self._rows_canvas = tk.Canvas(outer, highlightthickness=0, height=200)
        self._rows_canvas.grid(row=2, column=0, sticky="nsew")
        scroll_y = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=self._rows_canvas.yview)
        scroll_y.grid(row=2, column=1, sticky="ns")
        self._rows_canvas.configure(yscrollcommand=scroll_y.set)

        self._rows_inner = ttk.Frame(self._rows_canvas)
        self._rows_inner_window = self._rows_canvas.create_window(
            (0, 0), window=self._rows_inner, anchor="nw"
        )

        def _on_inner_configure(_event: tk.Event) -> None:
            self._rows_canvas.configure(scrollregion=self._rows_canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:
            # Match inner-frame width to canvas so columns can expand.
            self._rows_canvas.itemconfigure(self._rows_inner_window, width=event.width)

        self._rows_inner.bind("<Configure>", _on_inner_configure)
        self._rows_canvas.bind("<Configure>", _on_canvas_configure)

        # Add another row + footer buttons.
        controls = ttk.Frame(outer)
        controls.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        controls.columnconfigure(1, weight=1)

        add_btn = ttk.Button(controls, text="+ Add another ROI", command=self._on_add_row)
        add_btn.grid(row=0, column=0, sticky="w")

        cancel_btn = ttk.Button(controls, text="Cancel", command=self._on_cancel)
        cancel_btn.grid(row=0, column=2, sticky="e", padx=(0, 4))

        confirm_btn = ttk.Button(
            controls,
            text="Place on frame",
            command=self._on_confirm_clicked,
            style="Accent.TButton",
        )
        confirm_btn.grid(row=0, column=3, sticky="e")

        self.bind("<Escape>", lambda _e: self._on_cancel())
        self.bind("<Return>", lambda _e: self._on_confirm_clicked())

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
        except Exception:  # pragma: no cover - centering best-effort
            pass

    # ------------------------------------------------------------------
    # Row management
    # ------------------------------------------------------------------

    def _add_row(self, *, initial: Optional[RoiBuilderRow] = None) -> None:
        if initial is None:
            seed_name = f"ROI {self._next_default_index}"
            # Skip default names that already exist in the project so Add
            # doesn't dump the user into an instant collision.
            while seed_name in self._existing_roi_names:
                self._next_default_index += 1
                seed_name = f"ROI {self._next_default_index}"
            self._next_default_index += 1
            initial = RoiBuilderRow(name=seed_name)
        else:
            # Track default indexing past any seeded "ROI N" names.
            try:
                if initial.name.startswith("ROI ") and initial.name[4:].isdigit():
                    self._next_default_index = max(
                        self._next_default_index, int(initial.name[4:]) + 1
                    )
            except Exception:
                pass

        row_frame = ttk.Frame(self._rows_inner)
        row_index = len(self._rows)
        row_frame.grid(row=row_index, column=0, sticky="ew", padx=2, pady=2)
        row_frame.columnconfigure(0, weight=1)
        row_frame.columnconfigure(1, weight=0)
        row_frame.columnconfigure(2, weight=0)
        row_frame.columnconfigure(3, weight=0)

        name_var = tk.StringVar(value=initial.name)
        name_entry = ttk.Entry(row_frame, textvariable=name_var)
        name_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        shape_display = _SHAPE_KEY_TO_DISPLAY.get(initial.shape, "Rectangle")
        shape_var = tk.StringVar(value=shape_display)
        shape_combo = ttk.Combobox(
            row_frame,
            textvariable=shape_var,
            values=list(_SHAPE_DISPLAY_TO_KEY.keys()),
            state="readonly",
            width=12,
        )
        shape_combo.grid(row=0, column=1, sticky="w", padx=(0, 6))

        vertex_var = tk.StringVar(value=str(initial.polygon_vertices))
        vertex_entry = ttk.Entry(row_frame, textvariable=vertex_var, width=6)
        vertex_entry.grid(row=0, column=2, sticky="w", padx=(0, 6))

        remove_button = ttk.Button(
            row_frame,
            text="×",  # × multiplication sign
            width=3,
        )
        remove_button.grid(row=0, column=3, sticky="e")

        widgets = _RowWidgets(
            frame=row_frame,
            name_var=name_var,
            shape_var=shape_var,
            vertex_var=vertex_var,
            vertex_entry=vertex_entry,
            name_entry=name_entry,
            shape_combo=shape_combo,
            remove_button=remove_button,
        )

        # The remove handler captures the widget reference so it can find its
        # own slot in self._rows even after other rows have shifted indexes.
        remove_button.configure(command=lambda w=widgets: self._on_remove_row(w))

        # Polygon-vertex entry is enabled only when shape == polygon.
        def _refresh_vertex_state(*_args: object, w: _RowWidgets = widgets) -> None:
            shape_key = _SHAPE_DISPLAY_TO_KEY.get(w.shape_var.get(), SHAPE_RECTANGLE)
            if shape_key == SHAPE_POLYGON:
                w.vertex_entry.state(["!disabled"])
            else:
                w.vertex_entry.state(["disabled"])

        shape_combo.bind("<<ComboboxSelected>>", _refresh_vertex_state)
        _refresh_vertex_state()

        self._rows.append(widgets)
        self._refresh_remove_buttons()

    def _on_add_row(self) -> None:
        self._add_row()

    def _on_remove_row(self, widget: _RowWidgets) -> None:
        if len(self._rows) <= 1:
            # Always keep at least one row; users who don't want any ROIs
            # should click Cancel instead.
            return
        try:
            self._rows.remove(widget)
        except ValueError:
            return
        widget.frame.destroy()
        # Re-grid surviving rows so layout collapses nicely.
        for new_idx, w in enumerate(self._rows):
            w.frame.grid_configure(row=new_idx)
        self._refresh_remove_buttons()

    def _refresh_remove_buttons(self) -> None:
        # When only one row remains, disable its × button (we keep at least one).
        only_one = len(self._rows) <= 1
        for w in self._rows:
            try:
                if only_one:
                    w.remove_button.state(["disabled"])
                else:
                    w.remove_button.state(["!disabled"])
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Confirm / cancel
    # ------------------------------------------------------------------

    def _collect_rows(self) -> List[RoiBuilderRow]:
        out: List[RoiBuilderRow] = []
        for w in self._rows:
            name = str(w.name_var.get() or "").strip()
            shape_display = str(w.shape_var.get() or "Rectangle").strip()
            shape = _SHAPE_DISPLAY_TO_KEY.get(shape_display, SHAPE_RECTANGLE)
            try:
                vertices = int(str(w.vertex_var.get() or POLYGON_DEFAULT_VERTICES).strip())
            except ValueError:
                vertices = POLYGON_DEFAULT_VERTICES
            vertices = max(POLYGON_MIN_VERTICES, min(POLYGON_MAX_VERTICES, vertices))
            out.append(RoiBuilderRow(name=name, shape=shape, polygon_vertices=vertices))
        return out

    def _validate(self, rows: List[RoiBuilderRow]) -> Optional[str]:
        seen_names: set[str] = set()
        for idx, row in enumerate(rows, start=1):
            if not row.name:
                return f"Row {idx}: name cannot be blank."
            if row.name in seen_names:
                return f"Row {idx}: duplicate name {row.name!r} in this dialog."
            if row.name in self._existing_roi_names:
                return (
                    f"Row {idx}: an ROI named {row.name!r} already exists in "
                    "the project. Pick a different name or rename the existing one."
                )
            seen_names.add(row.name)
            if row.shape not in ALLOWED_SHAPES:
                return f"Row {idx}: unsupported shape {row.shape!r}."
            if row.shape == SHAPE_POLYGON and not (
                POLYGON_MIN_VERTICES <= row.polygon_vertices <= POLYGON_MAX_VERTICES
            ):
                return (
                    f"Row {idx}: polygon vertex count must be between "
                    f"{POLYGON_MIN_VERTICES} and {POLYGON_MAX_VERTICES}."
                )
        return None

    def _on_confirm_clicked(self) -> None:
        rows = self._collect_rows()
        error = self._validate(rows)
        if error is not None:
            messagebox.showwarning("ROI Builder", error, parent=self)
            return

        # Hand off to the caller before destroying so the callback can react
        # to a fresh state without the dialog still in the focus stack.
        self.destroy()
        try:
            self._on_confirm(rows)
        except Exception:
            # Errors in the parent's callback are not the dialog's problem,
            # but we shouldn't let them leak out as unhandled exceptions
            # either; log via Tk's stderr surface.
            import traceback

            traceback.print_exc()

    def _on_cancel(self) -> None:
        self.destroy()


__all__ = ["RoiBuilderDialog", "RoiBuilderRow"]

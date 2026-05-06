from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Optional, Tuple


def _is_descendant(widget: Any, ancestor: Any) -> bool:
    current = widget
    while current is not None:
        if current is ancestor:
            return True
        current = getattr(current, "master", None)
    return False


def _event_belongs_to_window(event_widget: Any, owning_window: Any) -> bool:
    """True iff ``event_widget`` lives inside ``owning_window`` (same Toplevel).

    ``bind_all`` mousewheel events fire for *every* widget in the
    application — including widgets inside Toplevel dialogs (sanity
    check, ROI editor, builder, etc.). Without this guard, scrolling
    inside a Toplevel would also scroll the main notebook tab beneath.
    Each dispatcher checks this before doing any work.
    """
    if event_widget is None or owning_window is None:
        return False
    try:
        event_top = event_widget.winfo_toplevel()
        owning_top = owning_window.winfo_toplevel()
    except Exception:
        return False
    return event_top is owning_top


def _prune_scroll_targets(app: Any) -> None:
    targets = getattr(app, "_scrollable_targets", [])
    alive = []
    for target in targets:
        canvas = target.get("canvas")
        try:
            if canvas is not None and canvas.winfo_exists():
                alive.append(target)
        except Exception:
            continue
    app._scrollable_targets = alive


def _resolve_scroll_target(app: Any, event_widget: Any) -> Optional[dict]:
    _prune_scroll_targets(app)
    targets = getattr(app, "_scrollable_targets", [])
    if not targets:
        return None

    # Prefer the canvas that contains the event widget.
    for target in targets:
        canvas = target.get("canvas")
        if canvas is not None and _is_descendant(event_widget, canvas):
            return target

    # Fall back to the currently selected notebook tab.
    notebook = getattr(app, "notebook", None)
    if notebook is not None:
        try:
            selected = notebook.select()
            for target in targets:
                tab = target.get("tab")
                if tab is not None and str(tab) == selected:
                    return target
        except Exception:
            pass
    return None


def _dispatch_global_mousewheel(app: Any, event: tk.Event) -> None:
    # Skip wheel events that originated inside a child Toplevel — those
    # belong to that Toplevel's own scroll surface (or to no scroll
    # surface at all). Without this guard, scrolling inside a modal
    # dialog also scrolls the main notebook tab beneath it.
    event_widget = getattr(event, "widget", None)
    root = getattr(app, "root", None)
    if root is not None and not _event_belongs_to_window(event_widget, root):
        return
    target = _resolve_scroll_target(app, event_widget)
    if target is None:
        return

    canvas = target["canvas"]
    scroll_state = target["scroll_state"]
    delta = getattr(event, "delta", 0)
    num = getattr(event, "num", 0)
    horizontal = bool(getattr(event, "state", 0) & 0x1)  # Shift key

    if horizontal and not scroll_state.get("horizontal", False):
        return
    if not horizontal and not scroll_state.get("vertical", False):
        return

    if delta:
        # Normalize wheel movement across mouse/touchpad devices.
        units = -float(delta) / 120.0
    elif num in (4, 5):
        units = -1.0 if num == 4 else 1.0
    else:
        return

    target["wheel_remainder"] += units
    whole_units = int(target["wheel_remainder"])
    if whole_units == 0:
        return
    target["wheel_remainder"] -= whole_units

    if horizontal:
        canvas.xview_scroll(whole_units, "units")
    else:
        canvas.yview_scroll(whole_units, "units")


def _ensure_global_mousewheel_binding(app: Any) -> None:
    if getattr(app, "_scrollable_global_wheel_bound", False):
        return

    root = getattr(app, "root", None)
    if root is None:
        return

    root.bind_all("<MouseWheel>", lambda event: _dispatch_global_mousewheel(app, event), add="+")
    root.bind_all("<Shift-MouseWheel>", lambda event: _dispatch_global_mousewheel(app, event), add="+")
    root.bind_all("<Button-4>", lambda event: _dispatch_global_mousewheel(app, event), add="+")
    root.bind_all("<Button-5>", lambda event: _dispatch_global_mousewheel(app, event), add="+")
    app._scrollable_global_wheel_bound = True


def _prune_section_scroll_targets(root: tk.Misc) -> None:
    targets = getattr(root, "_scrollable_section_targets", [])
    alive = []
    for target in targets:
        canvas = target.get("canvas")
        try:
            if canvas is not None and canvas.winfo_exists():
                alive.append(target)
        except Exception:
            continue
    root._scrollable_section_targets = alive  # type: ignore[attr-defined]


def _resolve_section_scroll_target(root: tk.Misc, event_widget: Any) -> Optional[dict]:
    _prune_section_scroll_targets(root)
    for target in getattr(root, "_scrollable_section_targets", []):
        canvas = target.get("canvas")
        if canvas is not None and _is_descendant(event_widget, canvas):
            return target
    return None


def _dispatch_section_mousewheel(root: tk.Misc, event: tk.Event) -> None:
    # Skip wheel events from widgets in a different Toplevel (e.g.,
    # modal ROI editor, sanity check dialog). They are not ours to handle.
    event_widget = getattr(event, "widget", None)
    if not _event_belongs_to_window(event_widget, root):
        return
    target = _resolve_section_scroll_target(root, event_widget)
    if target is None:
        return

    canvas = target["canvas"]
    scroll_state = target["scroll_state"]
    delta = getattr(event, "delta", 0)
    num = getattr(event, "num", 0)
    horizontal = bool(getattr(event, "state", 0) & 0x1)

    if horizontal and not scroll_state.get("horizontal", False):
        return
    if not horizontal and not scroll_state.get("vertical", False):
        return

    if delta:
        units = -float(delta) / 120.0
    elif num in (4, 5):
        units = -1.0 if num == 4 else 1.0
    else:
        return

    target["wheel_remainder"] += units
    whole_units = int(target["wheel_remainder"])
    if whole_units == 0:
        return
    target["wheel_remainder"] -= whole_units

    if horizontal:
        canvas.xview_scroll(whole_units, "units")
    else:
        canvas.yview_scroll(whole_units, "units")


def _ensure_section_mousewheel_binding(root: tk.Misc) -> None:
    if getattr(root, "_scrollable_section_wheel_bound", False):
        return

    root.bind_all("<MouseWheel>", lambda event: _dispatch_section_mousewheel(root, event), add="+")
    root.bind_all("<Shift-MouseWheel>", lambda event: _dispatch_section_mousewheel(root, event), add="+")
    root.bind_all("<Button-4>", lambda event: _dispatch_section_mousewheel(root, event), add="+")
    root.bind_all("<Button-5>", lambda event: _dispatch_section_mousewheel(root, event), add="+")
    root._scrollable_section_wheel_bound = True  # type: ignore[attr-defined]


def create_scrollable_tab(app, tab: ttk.Frame, *, xscroll: bool = False) -> ttk.Frame:
    """
    Wrap a Notebook tab inside a scrollable canvas and return the content frame.

    Args:
        app: Application instance that exposes ``root`` for global mousewheel binding.
        tab: The ttk.Frame associated with the Notebook tab.
        xscroll: When True, enable a horizontal scrollbar in addition to the vertical one.

    Returns:
        ttk.Frame: Frame that callers should populate with widgets.
    """
    container = ttk.Frame(tab)
    container.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(container, highlightthickness=0, borderwidth=0)
    vbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vbar.set)

    container.grid_rowconfigure(0, weight=1)
    container.grid_columnconfigure(0, weight=1)

    canvas.grid(row=0, column=0, sticky="nsew")
    vbar.grid(row=0, column=1, sticky="ns")
    vbar.grid_remove()

    hbar = None
    if xscroll:
        hbar = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=hbar.set)
        hbar.grid(row=1, column=0, sticky="ew")
        hbar.grid_remove()

    content_frame = ttk.Frame(canvas)
    window_id = canvas.create_window((0, 0), window=content_frame, anchor="nw")

    scroll_state = {"vertical": False, "horizontal": False}
    adjust_job = {"id": None}

    def _adjust_scrollbars() -> None:
        adjust_job["id"] = None
        canvas.update_idletasks()
        bbox = canvas.bbox("all")
        if not bbox:
            return
        content_width = bbox[2] - bbox[0]
        content_height = bbox[3] - bbox[1]
        visible_width = max(canvas.winfo_width(), 1)
        visible_height = max(canvas.winfo_height(), 1)

        need_vertical = content_height > visible_height + 2
        if need_vertical and not scroll_state["vertical"]:
            canvas.configure(yscrollcommand=vbar.set)
            vbar.grid()
            scroll_state["vertical"] = True
        elif not need_vertical and scroll_state["vertical"]:
            canvas.configure(yscrollcommand=None)
            vbar.grid_remove()
            canvas.yview_moveto(0)
            scroll_state["vertical"] = False

        if hbar is not None:
            need_horizontal = content_width > visible_width + 2
            if need_horizontal and not scroll_state["horizontal"]:
                canvas.configure(xscrollcommand=hbar.set)
                hbar.grid()
                scroll_state["horizontal"] = True
            elif not need_horizontal and scroll_state["horizontal"]:
                canvas.configure(xscrollcommand=None)
                hbar.grid_remove()
                canvas.xview_moveto(0)
                scroll_state["horizontal"] = False

    def _schedule_adjust() -> None:
        existing = adjust_job.get("id")
        if existing is not None:
            try:
                canvas.after_cancel(existing)
            except Exception:
                pass
        adjust_job["id"] = canvas.after(16, _adjust_scrollbars)

    def _update_scroll_region(_event=None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))
        _schedule_adjust()

    content_frame.bind("<Configure>", _update_scroll_region)

    def _resize_content(event) -> None:
        canvas.itemconfigure(window_id, width=event.width)
        _schedule_adjust()

    canvas.bind("<Configure>", _resize_content)

    if not hasattr(app, "_scrollable_targets"):
        app._scrollable_targets = []
    app._scrollable_targets.append(
        {
            "tab": tab,
            "canvas": canvas,
            "scroll_state": scroll_state,
            "wheel_remainder": 0.0,
        }
    )
    _ensure_global_mousewheel_binding(app)

    # Keep references to avoid garbage collection of local widgets/callbacks.
    tab._scrollable_refs = (canvas, vbar, hbar, scroll_state, _adjust_scrollbars, _schedule_adjust)  # type: ignore[attr-defined]

    canvas.after(0, _schedule_adjust)
    return content_frame


def create_scrollable_section(root: tk.Misc, parent: ttk.Frame, *, xscroll: bool = False) -> Tuple[tk.Canvas, ttk.Frame]:
    """
    Create a scrollable section that can be embedded within dialogs or standalone windows.

    Args:
        root: The Tk root or any widget supporting ``bind_all``.
        parent: Parent frame where the scrollable canvas should live.
        xscroll: When True, include a horizontal scrollbar.

    Returns:
        Tuple[tk.Canvas, ttk.Frame]: The canvas and the inner content frame.
    """
    binding_root = root._root()
    canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0)
    vbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vbar.set)

    parent.grid_rowconfigure(0, weight=1)
    parent.grid_columnconfigure(0, weight=1)

    canvas.grid(row=0, column=0, sticky="nsew")
    vbar.grid(row=0, column=1, sticky="ns")
    vbar.grid_remove()

    hbar = None
    if xscroll:
        hbar = ttk.Scrollbar(parent, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=hbar.set)
        hbar.grid(row=1, column=0, sticky="ew")
        hbar.grid_remove()

    content_frame = ttk.Frame(canvas)
    window_id = canvas.create_window((0, 0), window=content_frame, anchor="nw")

    scroll_state = {"vertical": False, "horizontal": False}
    adjust_job = {"id": None}

    def _adjust_scrollbars() -> None:
        adjust_job["id"] = None
        canvas.update_idletasks()
        bbox = canvas.bbox("all")
        if not bbox:
            return
        content_width = bbox[2] - bbox[0]
        content_height = bbox[3] - bbox[1]
        visible_width = max(canvas.winfo_width(), 1)
        visible_height = max(canvas.winfo_height(), 1)

        need_vertical = content_height > visible_height + 2
        if need_vertical and not scroll_state["vertical"]:
            canvas.configure(yscrollcommand=vbar.set)
            vbar.grid()
            scroll_state["vertical"] = True
        elif not need_vertical and scroll_state["vertical"]:
            canvas.configure(yscrollcommand=None)
            vbar.grid_remove()
            canvas.yview_moveto(0)
            scroll_state["vertical"] = False

        if hbar is not None:
            need_horizontal = content_width > visible_width + 2
            if need_horizontal and not scroll_state["horizontal"]:
                canvas.configure(xscrollcommand=hbar.set)
                hbar.grid()
                scroll_state["horizontal"] = True
            elif not need_horizontal and scroll_state["horizontal"]:
                canvas.configure(xscrollcommand=None)
                hbar.grid_remove()
                canvas.xview_moveto(0)
                scroll_state["horizontal"] = False

    def _schedule_adjust() -> None:
        existing = adjust_job.get("id")
        if existing is not None:
            try:
                canvas.after_cancel(existing)
            except Exception:
                pass
        adjust_job["id"] = canvas.after(16, _adjust_scrollbars)

    def _update_scroll_region(_event=None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))
        _schedule_adjust()

    content_frame.bind("<Configure>", _update_scroll_region)

    def _resize_content(event) -> None:
        canvas.itemconfigure(window_id, width=event.width)
        _schedule_adjust()

    canvas.bind("<Configure>", _resize_content)

    if not hasattr(binding_root, "_scrollable_section_targets"):
        binding_root._scrollable_section_targets = []  # type: ignore[attr-defined]
    binding_root._scrollable_section_targets.append(  # type: ignore[attr-defined]
        {
            "canvas": canvas,
            "scroll_state": scroll_state,
            "wheel_remainder": 0.0,
        }
    )
    _ensure_section_mousewheel_binding(binding_root)

    parent._scrollable_refs = (canvas, vbar, hbar, scroll_state, _adjust_scrollbars, _schedule_adjust)  # type: ignore[attr-defined]

    canvas.after(0, _schedule_adjust)
    return canvas, content_frame

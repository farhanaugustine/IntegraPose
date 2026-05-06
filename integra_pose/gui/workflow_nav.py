from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def add_workflow_footer(app, parent: ttk.Frame, *, padx: int = 5, pady: tuple[int, int] = (12, 6)) -> ttk.Frame:
    """Append consistent Back/Next workflow controls to a packed tab container."""
    frame = ttk.LabelFrame(parent, text="Workflow Navigation", padding=8)
    frame.pack(fill=tk.X, padx=padx, pady=pady)
    frame.columnconfigure(1, weight=1)

    back_btn = ttk.Button(
        frame,
        text="Back",
        command=lambda: app._navigate_workflow_tab(-1),
        style="NavBack.TButton",
    )
    back_btn.grid(row=0, column=0, sticky="w")

    ttk.Label(
        frame,
        textvariable=app.workflow_step_summary_var,
        style="Status.TLabel",
        justify="left",
    ).grid(row=0, column=1, sticky="w", padx=12)

    next_btn = ttk.Button(
        frame,
        text="Next",
        command=lambda: app._navigate_workflow_tab(1),
        style="NavNext.TButton",
    )
    next_btn.grid(row=0, column=2, sticky="e")

    app._register_workflow_nav_button(back_btn, direction=-1)
    app._register_workflow_nav_button(next_btn, direction=1)
    return frame


def add_workflow_footer_grid(
    app,
    parent: ttk.Frame,
    *,
    row: int,
    column: int = 0,
    columnspan: int = 1,
    padx: int = 5,
    pady: tuple[int, int] = (12, 6),
) -> ttk.Frame:
    """Append consistent Back/Next workflow controls to a gridded tab container."""
    frame = ttk.LabelFrame(parent, text="Workflow Navigation", padding=8)
    frame.grid(row=row, column=column, columnspan=columnspan, sticky="ew", padx=padx, pady=pady)
    frame.columnconfigure(1, weight=1)

    back_btn = ttk.Button(
        frame,
        text="Back",
        command=lambda: app._navigate_workflow_tab(-1),
        style="NavBack.TButton",
    )
    back_btn.grid(row=0, column=0, sticky="w")

    ttk.Label(
        frame,
        textvariable=app.workflow_step_summary_var,
        style="Status.TLabel",
        justify="left",
    ).grid(row=0, column=1, sticky="w", padx=12)

    next_btn = ttk.Button(
        frame,
        text="Next",
        command=lambda: app._navigate_workflow_tab(1),
        style="NavNext.TButton",
    )
    next_btn.grid(row=0, column=2, sticky="e")

    app._register_workflow_nav_button(back_btn, direction=-1)
    app._register_workflow_nav_button(next_btn, direction=1)
    return frame


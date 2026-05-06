from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .theme import CANONICAL_ACCENT, CANONICAL_THEME, apply_global_ttk_theme


def apply_plugin_chrome(window: tk.Misc, host_app=None) -> ttk.Style:
    theme = getattr(host_app, "_CANONICAL_THEME", CANONICAL_THEME)
    accent = getattr(host_app, "_CANONICAL_ACCENT", CANONICAL_ACCENT)
    return apply_global_ttk_theme(window, theme=theme, accent=accent)


def build_plugin_header(parent: tk.Misc, *, title: str, summary: str) -> ttk.Frame:
    frame = ttk.Frame(parent, style="Hero.TFrame", padding=(14, 12))
    frame.columnconfigure(0, weight=1)
    ttk.Label(frame, text=title, style="HeroTitle.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(
        frame,
        text=summary,
        style="HeroBody.TLabel",
        justify=tk.LEFT,
        wraplength=1080,
    ).grid(row=1, column=0, sticky="w", pady=(4, 0))
    return frame

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

try:
    import sv_ttk  # type: ignore
except ImportError:  # pragma: no cover - optional dependency handled gracefully
    sv_ttk = None  # type: ignore[assignment]

DEFAULT_FONT = ("Segoe UI", 12)
SMALL_FONT = ("Segoe UI", 11)
BOLD_FONT = ("Segoe UI", 12, "bold")
SECTION_TITLE_FONT = ("Segoe UI", 14, "bold")
HEADER_FONT = ("Segoe UI", 18, "bold")
# Caption / hint text used under fields and inside frames as guidance.
CAPTION_FONT = ("Segoe UI", 9, "italic")
# Compact bold for inline labels (e.g., review-panel summary line).
INLINE_BOLD_FONT = ("Segoe UI", 10, "bold")

# Spacing constants — replace ad-hoc ints scattered through GUI code so
# the same role gets the same value across tabs and plugins.
PADDING_STANDARD = 10  # LabelFrame default
PADDING_COMPACT = 6     # Nested frames, dense forms
PADDING_LOOSE = 12      # Hero-section / important containers
GRID_PADX = 5           # Default cell padding inside grid layouts
GRID_PADY = 5

# Foreground colors used in multiple places for the same role.
MUTED_FG = "#6b7280"     # Hints, captions, low-priority text
WARNING_FG = "#c25700"   # Inline warning copy
ERROR_FG = "#a00000"     # Error / failure copy

CANONICAL_THEME = "classic"
CANONICAL_ACCENT = "#6ee7b7"


def _resolve_color(style: ttk.Style, element: str, option: str, fallback: str) -> str:
    try:
        value = style.lookup(element, option)
    except tk.TclError:
        value = ""
    return value or fallback


def _configure_legacy_widget_defaults(root: tk.Misc, style: ttk.Style, accent: str, theme_name: str) -> None:
    """Ensure classic tk widgets inherit the modern palette."""
    if theme_name == "light":
        default_bg = "#f8fafc"
        default_fg = "#0f172a"
        entry_bg = "#eef2ff"
        listbox_bg = "#f1f5f9"
        button_bg = "#e2e8f0"
        button_active_bg = "#cbd5e1"
        button_fg = "#0f172a"
    elif theme_name == "dark":
        default_bg = "#1f1f1f"
        default_fg = "#f5f5f5"
        entry_bg = "#111111"
        listbox_bg = "#111111"
        button_bg = "#374151"
        button_active_bg = "#4b5563"
        button_fg = "#f3f4f6"
    else:  # classic
        default_bg = "#f8fafc"
        default_fg = "#202020"
        entry_bg = "#f8fafc"
        listbox_bg = "#f8fafc"
        button_bg = "#ecfdf5"
        button_active_bg = "#d1fae5"
        button_fg = "#202020"

    bg = _resolve_color(style, "TFrame", "background", default_bg)
    fg = _resolve_color(style, "TLabel", "foreground", default_fg)
    select_bg = accent or "#34d399"

    root.configure(background=bg)

    root.option_add("*Font", DEFAULT_FONT)
    root.option_add("*Background", bg)
    root.option_add("*foreground", fg)
    root.option_add("*Label.font", DEFAULT_FONT)
    root.option_add("*Label.background", bg)
    root.option_add("*Button.font", DEFAULT_FONT)
    root.option_add("*Button.background", button_bg)
    root.option_add("*Button.foreground", button_fg)
    root.option_add("*Button.activeBackground", button_active_bg)
    root.option_add("*Button.activeForeground", button_fg)
    root.option_add("*Entry.font", DEFAULT_FONT)
    root.option_add("*Entry.background", entry_bg)
    root.option_add("*Entry.foreground", fg)
    root.option_add("*Listbox.font", DEFAULT_FONT)
    root.option_add("*Listbox.background", listbox_bg)
    root.option_add("*Listbox.foreground", fg)
    root.option_add("*Listbox.selectBackground", select_bg)
    root.option_add("*Listbox.selectForeground", fg)
    root.option_add("*Treeview.fieldbackground", bg)


def apply_global_ttk_theme(
    root: tk.Misc | None = None,
    *,
    theme: str = CANONICAL_THEME,
    accent: str = CANONICAL_ACCENT,
) -> ttk.Style:
    """Configure a shared ttk theme for IntegraPose GUIs.

    Args:
        root: Tk root or Toplevel to apply the theme to.
        theme: Either ``"dark"`` or ``"light"`` when Sun Valley is available; ignored for fallback themes.
        accent: Hex colour used for primary highlights and focus rings.

    Returns:
        The ttk.Style instance bound to *root* after configuration.
    """
    if root is None:
        root = tk._get_default_root()
        if root is None:
            raise RuntimeError("apply_global_ttk_theme requires an active Tk root or Toplevel.")

    style = ttk.Style(root)

    theme_lower = (theme or "light").lower()
    use_classic = theme_lower == "classic"

    if sv_ttk is not None and not use_classic:
        try:
            sv_ttk.set_theme(theme_lower, root)
        except TypeError:
            sv_ttk.set_theme(theme_lower)
        for setter_name in ("set_default_color", "set_primary_color", "set_color"):
            setter = getattr(sv_ttk, setter_name, None)
            if setter is None:
                continue
            try:
                setter(accent)
                break
            except TypeError:
                continue
    else:  # pragma: no cover - fallback engaged when sv-ttk is unavailable or classic is requested
        for candidate in ("clam", "alt", "default"):
            try:
                style.theme_use(candidate)
                break
            except tk.TclError:
                continue

    palette_key = theme_lower if theme_lower in ("light", "dark") else "classic"

    style.configure("TLabel", font=DEFAULT_FONT, padding=(4, 2))
    style.configure(
        "TButton",
        font=DEFAULT_FONT,
        padding=(11, 6),
        relief="raised",
        borderwidth=1,
        background="#ffffff",
        foreground="#0f172a",
        bordercolor="#94a3b8",
        lightcolor="#ffffff",
        darkcolor="#94a3b8",
    )
    style.map(
        "TButton",
        background=[("pressed", "#bbf7d0"), ("active", "#dcfce7"), ("!disabled", "#ffffff")],
        foreground=[("disabled", "gray55"), ("!disabled", "#0f172a")],
        bordercolor=[("focus", accent), ("active", "#10b981"), ("!disabled", "#94a3b8")],
        lightcolor=[("focus", accent), ("active", "#10b981"), ("!disabled", "#ffffff")],
        darkcolor=[("focus", "#059669"), ("active", "#059669"), ("!disabled", "#94a3b8")],
        relief=[("pressed", "sunken"), ("!pressed", "raised")],
    )
    entry_bg = (
        "#eef2ff" if palette_key == "light" else "#111827" if palette_key == "dark" else "#ffffff"
    )
    entry_border = (
        "#94a3b8" if palette_key == "light" else "#374151" if palette_key == "dark" else "#cbd5e1"
    )
    text_fg = "#0f172a" if palette_key == "light" else "#f5f5f5" if palette_key == "dark" else "#202020"
    muted_fg = "#475569" if palette_key != "dark" else "#cbd5e1"
    base_panel_bg = _resolve_color(
        style,
        "TFrame",
        "background",
        "#f8fafc" if palette_key != "dark" else "#1f1f1f",
    )
    hero_bg = base_panel_bg
    hero_border = "#cbd5e1" if palette_key != "dark" else "#334155"
    card_bg = base_panel_bg
    card_border = "#cbd5e1" if palette_key != "dark" else "#334155"
    tree_selected_bg = "#bbf7d0" if palette_key != "dark" else "#14532d"
    tree_selected_fg = "#14532d" if palette_key != "dark" else "#ecfdf5"

    style.configure(
        "TEntry",
        font=DEFAULT_FONT,
        fieldbackground=entry_bg,
        foreground=text_fg,
        bordercolor=entry_border,
        lightcolor=entry_border,
        darkcolor=entry_border,
        insertcolor=text_fg,
    )
    style.map(
        "TEntry",
        bordercolor=[("focus", accent), ("!focus", entry_border)],
        lightcolor=[("focus", accent), ("!focus", entry_border)],
        darkcolor=[("focus", accent), ("!focus", entry_border)],
    )

    style.configure("TCombobox", font=DEFAULT_FONT, foreground=text_fg)
    style.map(
        "TCombobox",
        bordercolor=[("focus", accent), ("!focus", entry_border)],
        lightcolor=[("focus", accent), ("!focus", entry_border)],
        darkcolor=[("focus", accent), ("!focus", entry_border)],
    )
    style.configure("TCheckbutton", font=DEFAULT_FONT)
    style.configure("TRadiobutton", font=DEFAULT_FONT)
    style.configure("TMenubutton", font=DEFAULT_FONT)
    style.configure("TLabelframe", padding=10)
    style.configure("TLabelframe.Label", font=SECTION_TITLE_FONT, foreground=text_fg)
    style.configure(
        "Card.TLabelframe",
        padding=12,
        background=card_bg,
        bordercolor=card_border,
        lightcolor=card_border,
        darkcolor=card_border,
        relief="groove",
        borderwidth=1,
    )
    style.configure("Card.TLabelframe.Label", font=SECTION_TITLE_FONT, foreground=text_fg)
    style.configure("TNotebook.Tab", font=BOLD_FONT, padding=(14, 10))
    style.map(
        "TNotebook.Tab",
        background=[("selected", "#bbf7d0"), ("!selected", "#f8fafc")],
        foreground=[("selected", "#14532d"), ("!selected", text_fg)],
    )
    style.configure("Treeview", font=DEFAULT_FONT)
    style.configure("Treeview.Heading", font=BOLD_FONT)
    style.map(
        "Treeview",
        background=[("selected", tree_selected_bg)],
        foreground=[("selected", tree_selected_fg)],
    )

    style.configure(
        "Accent.TButton",
        font=BOLD_FONT,
        foreground="#111111",
        background="#86efac",
        padding=(12, 6),
        relief="raised",
        borderwidth=1,
        bordercolor="#059669",
        lightcolor="#a7f3d0",
        darkcolor="#047857",
    )
    style.map(
        "Accent.TButton",
        foreground=[("disabled", "gray55"), ("!disabled", "#111111")],
        background=[("pressed", "#4ade80"), ("active", "#6ee7b7"), ("disabled", "#e2e8f0")],
        bordercolor=[("focus", "#059669"), ("active", "#047857"), ("!disabled", "#059669")],
        relief=[("pressed", "sunken"), ("!pressed", "raised")],
    )

    if palette_key == "dark":
        nav_next_bg = "#134e4a"
        nav_next_active = "#115e59"
        nav_back_bg = "#334155"
        nav_back_active = "#1e293b"
        workflow_done_bg = "#14532d"
        workflow_current_bg = "#166534"
        workflow_pending_bg = "#475569"
        nav_text_fg = "#f8fafc"
    else:
        nav_next_bg = "#a7f3d0"
        nav_next_active = "#6ee7b7"
        nav_back_bg = "#ffffff"
        nav_back_active = "#f0fdf4"
        workflow_done_bg = "#f1f5f9"
        workflow_current_bg = "#bbf7d0"
        workflow_pending_bg = "#ffffff"
        nav_text_fg = "#0f172a"

    style.configure(
        "NavNext.TButton",
        font=BOLD_FONT,
        foreground=nav_text_fg,
        background=nav_next_bg,
        padding=(12, 6),
        relief="raised",
        borderwidth=1,
        bordercolor="#059669",
        lightcolor="#d1fae5",
        darkcolor="#047857",
    )
    style.map(
        "NavNext.TButton",
        foreground=[("disabled", "gray45")],
        background=[("pressed", "#4ade80"), ("active", nav_next_active), ("disabled", "#9ca3af")],
        bordercolor=[("focus", "#059669"), ("active", "#047857"), ("!disabled", "#059669")],
        relief=[("pressed", "sunken"), ("!pressed", "raised")],
    )

    style.configure(
        "NavBack.TButton",
        font=BOLD_FONT,
        foreground=nav_text_fg,
        background=nav_back_bg,
        padding=(12, 6),
        relief="raised",
        borderwidth=1,
        bordercolor="#94a3b8",
        lightcolor="#ffffff",
        darkcolor="#94a3b8",
    )
    style.map(
        "NavBack.TButton",
        foreground=[("disabled", "gray45")],
        background=[("pressed", "#dcfce7"), ("active", nav_back_active), ("disabled", "#9ca3af")],
        bordercolor=[("focus", accent), ("active", "#10b981"), ("!disabled", "#94a3b8")],
        relief=[("pressed", "sunken"), ("!pressed", "raised")],
    )

    style.configure(
        "WorkflowCurrent.TButton",
        font=BOLD_FONT,
        foreground=nav_text_fg,
        background=workflow_current_bg,
        padding=(8, 4),
        relief="raised",
        borderwidth=1,
        bordercolor="#059669",
        lightcolor="#d1fae5",
        darkcolor="#047857",
    )
    style.map(
        "WorkflowCurrent.TButton",
        foreground=[("disabled", "gray45")],
        background=[("pressed", "#86efac"), ("active", workflow_current_bg), ("disabled", "#9ca3af")],
        bordercolor=[("focus", "#059669"), ("active", "#047857"), ("!disabled", "#059669")],
        relief=[("pressed", "sunken"), ("!pressed", "raised")],
    )
    style.configure(
        "WorkflowDone.TButton",
        font=BOLD_FONT,
        foreground=nav_text_fg,
        background=workflow_done_bg,
        padding=(8, 4),
        relief="raised",
        borderwidth=1,
        bordercolor="#cbd5e1",
        lightcolor="#ffffff",
        darkcolor="#94a3b8",
    )
    style.map(
        "WorkflowDone.TButton",
        foreground=[("disabled", "gray45")],
        background=[("pressed", "#e2e8f0"), ("active", "#e2e8f0"), ("disabled", "#9ca3af")],
        bordercolor=[("focus", "#cbd5e1"), ("active", "#94a3b8"), ("!disabled", "#cbd5e1")],
        relief=[("pressed", "sunken"), ("!pressed", "raised")],
    )
    style.configure(
        "WorkflowPending.TButton",
        font=BOLD_FONT,
        foreground=nav_text_fg,
        background=workflow_pending_bg,
        padding=(8, 4),
        relief="raised",
        borderwidth=1,
        bordercolor="#cbd5e1",
        lightcolor="#ffffff",
        darkcolor="#94a3b8",
    )
    style.map(
        "WorkflowPending.TButton",
        foreground=[("disabled", "gray45")],
        background=[("pressed", "#f1f5f9"), ("active", "#f1f5f9"), ("disabled", "#9ca3af")],
        bordercolor=[("focus", "#cbd5e1"), ("active", "#94a3b8"), ("!disabled", "#cbd5e1")],
        relief=[("pressed", "sunken"), ("!pressed", "raised")],
    )

    style.configure("Confirm.TButton", font=BOLD_FONT, foreground="#0b6623")
    style.configure("Danger.TButton", font=BOLD_FONT, foreground="#a40000")

    style.configure("Status.TLabel", font=SMALL_FONT)
    style.configure("Header.TLabel", font=HEADER_FONT, foreground=text_fg)
    style.configure("Subtle.TLabel", font=SMALL_FONT, foreground=muted_fg)
    style.configure("FieldHint.TLabel", font=SMALL_FONT, foreground=muted_fg)
    style.configure(
        "Hero.TFrame",
        background=hero_bg,
        bordercolor=hero_border,
        lightcolor=hero_border,
        darkcolor=hero_border,
        relief="solid",
        borderwidth=1,
    )
    style.configure("HeroTitle.TLabel", font=HEADER_FONT, foreground=text_fg, background=hero_bg)
    style.configure("HeroBody.TLabel", font=SMALL_FONT, foreground=muted_fg, background=hero_bg)
    style.configure(
        "SectionToggle.TButton",
        font=DEFAULT_FONT,
        padding=(11, 6),
        relief="raised",
        borderwidth=1,
        background="#ffffff",
        foreground="#0f172a",
        bordercolor="#94a3b8",
        lightcolor="#ffffff",
        darkcolor="#94a3b8",
    )
    style.map(
        "SectionToggle.TButton",
        background=[("pressed", "#bbf7d0"), ("active", "#dcfce7"), ("!disabled", "#ffffff")],
        foreground=[("disabled", "gray55"), ("!disabled", "#0f172a")],
        bordercolor=[("focus", accent), ("active", "#10b981"), ("!disabled", "#94a3b8")],
        lightcolor=[("focus", accent), ("active", "#10b981"), ("!disabled", "#ffffff")],
        darkcolor=[("focus", "#059669"), ("active", "#059669"), ("!disabled", "#94a3b8")],
        relief=[("pressed", "sunken"), ("!pressed", "raised")],
    )
    stage_text_fg = "#0f172a" if palette_key != "dark" else "#f8fafc"
    style.configure(
        "StageNotStarted.TLabel",
        font=SMALL_FONT,
        foreground=stage_text_fg,
        background="#e5e7eb" if palette_key != "dark" else "#374151",
        padding=(8, 6),
        borderwidth=1,
        relief="solid",
        anchor="center",
    )
    style.configure(
        "StageReady.TLabel",
        font=SMALL_FONT,
        foreground="#14532d" if palette_key != "dark" else "#ecfdf5",
        background="#bbf7d0" if palette_key != "dark" else "#166534",
        padding=(8, 6),
        borderwidth=1,
        relief="solid",
        anchor="center",
    )
    style.configure(
        "StageWarning.TLabel",
        font=SMALL_FONT,
        foreground="#713f12" if palette_key != "dark" else "#fef3c7",
        background="#fde68a" if palette_key != "dark" else "#92400e",
        padding=(8, 6),
        borderwidth=1,
        relief="solid",
        anchor="center",
    )
    style.configure(
        "StageError.TLabel",
        font=SMALL_FONT,
        foreground="#7f1d1d" if palette_key != "dark" else "#fee2e2",
        background="#fecaca" if palette_key != "dark" else "#991b1b",
        padding=(8, 6),
        borderwidth=1,
        relief="solid",
        anchor="center",
    )

    _configure_legacy_widget_defaults(root, style, accent, palette_key)

    return style

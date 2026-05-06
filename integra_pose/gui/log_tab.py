import tkinter as tk
from tkinter import ttk, scrolledtext

from .theme import SMALL_FONT


def _style_lookup(style: ttk.Style | None, element: str, option: str, fallback: str) -> str:
    if style is None:
        return fallback
    try:
        value = style.lookup(element, option)
    except tk.TclError:
        value = ""
    return value or fallback


class ConsoleRedirector:
    """A class to redirect stdout and stderr to a Tkinter text widget."""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.text_widget.config(state=tk.NORMAL) # Ensure it's writable

    def write(self, message):
        if self.text_widget.master.winfo_exists():
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END) # Auto-scroll
            self.text_widget.update_idletasks()

    def flush(self):
        pass

def create_log_tab(app):
    """
    Creates the 'Log' tab in the main application window.
    
    Args:
        app: The main YoloApp instance.
    """
    app.log_tab = ttk.Frame(app.notebook, padding="10")
    if hasattr(app, "tab_widgets"):
        app.tab_widgets["log"] = app.log_tab
    base_label = "Log"
    if hasattr(app, "tab_base_labels"):
        base_label = app.tab_base_labels.get("log", base_label)
    app.notebook.add(app.log_tab, text=base_label)

    style = getattr(app, "style", None)
    base_bg = _style_lookup(style, "TFrame", "background", "#f8fafc")
    text_bg = _style_lookup(style, "TEntry", "fieldbackground", base_bg)
    text_fg = _style_lookup(style, "TLabel", "foreground", "#0f172a")
    accent = _style_lookup(style, "Accent.TButton", "background", "#86efac")
    select_bg = accent or "#86efac"
    select_fg = "#111111" if select_bg.lower() != "#111111" else "#f8fafc"

    log_container = ttk.Frame(app.log_tab)
    log_container.pack(expand=True, fill=tk.BOTH)
    log_container.columnconfigure(0, weight=1)
    log_container.rowconfigure(0, weight=1)

    app.log_text_widget = scrolledtext.ScrolledText(
        log_container, 
        wrap=tk.NONE,
        state=tk.DISABLED, # Start as read-only
        bg=text_bg,
        fg=text_fg,
        insertbackground=text_fg,
        selectbackground=select_bg,
        selectforeground=select_fg,
        font=SMALL_FONT,
    )
    app.log_text_widget.grid(row=0, column=0, sticky="nsew")

    h_scroll = ttk.Scrollbar(log_container, orient=tk.HORIZONTAL, command=app.log_text_widget.xview)
    h_scroll.grid(row=1, column=0, sticky="ew")
    app.log_text_widget.configure(xscrollcommand=h_scroll.set)

    button_frame = ttk.Frame(log_container)
    button_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))

    def clear_log_text():
        """Clears the content of the log text widget."""
        app.log_text_widget.config(state=tk.NORMAL)
        app.log_text_widget.delete('1.0', tk.END)
        app.log_text_widget.config(state=tk.DISABLED)
        reset_fn = getattr(app, "_reset_log_status_counters", None)
        if callable(reset_fn):
            reset_fn()

    app.clear_log = clear_log_text

    clear_button = ttk.Button(button_frame, text="Clear Log", command=app.clear_log)
    clear_button.pack(side=tk.RIGHT)

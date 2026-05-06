from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class CollapsibleSection(ttk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str,
        description: str = "",
        expanded: bool = False,
        body_padding: int | tuple[int, ...] = 10,
    ) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self._expanded = expanded

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title_col = ttk.Frame(header)
        title_col.grid(row=0, column=0, sticky="ew")
        title_col.columnconfigure(0, weight=1)
        ttk.Label(title_col, text=title, style="TLabelframe.Label").grid(row=0, column=0, sticky="w")
        if description:
            ttk.Label(
                title_col,
                text=description,
                style="FieldHint.TLabel",
                justify=tk.LEFT,
                wraplength=760,
            ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        self._toggle_btn = ttk.Button(header, style="SectionToggle.TButton", command=self.toggle)
        self._toggle_btn.grid(row=0, column=1, sticky="e", padx=(10, 0))

        self.body = ttk.Frame(self, padding=body_padding)
        self._sync()

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._sync()

    def expand(self) -> None:
        self._expanded = True
        self._sync()

    def collapse(self) -> None:
        self._expanded = False
        self._sync()

    def _sync(self) -> None:
        self._toggle_btn.configure(text="Hide details" if self._expanded else "Show details")
        if self._expanded:
            self.body.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        else:
            self.body.grid_forget()

"""Sash-position helpers for ttk.Panedwindow.

``ttk.Panedwindow`` does not expose per-pane ``minsize`` natively. Without
clamping, a user can drag a sash to either extreme and hide the opposite
pane completely — a common usability complaint in IntegraPose's Tab 6.

These helpers intercept sash motion and clamp the position so neither pane
shrinks below a configured minimum, plus offer a one-call ``reset_panes``
to restore default proportional positions.
"""

from __future__ import annotations

from tkinter import ttk
from typing import Sequence


def clamp_paned_sashes(
    pane: ttk.Panedwindow,
    *,
    min_pane_px: int = 200,
) -> None:
    """Bind motion events on ``pane`` so a sash cannot hide either side.

    The minimum applies to *every* pane the widget contains: a sash can be
    dragged anywhere as long as both the pane before it and the pane after
    it remain at least ``min_pane_px`` along the orientation axis.
    """

    def _orientation_size() -> int:
        try:
            orient = str(pane.cget("orient")) or "horizontal"
        except Exception:
            orient = "horizontal"
        if orient.startswith("h"):
            return int(pane.winfo_width() or 0)
        return int(pane.winfo_height() or 0)

    def _enforce(_event=None) -> None:
        try:
            num_panes = len(pane.panes())
        except Exception:
            return
        if num_panes < 2:
            return
        total = _orientation_size()
        if total <= 0:
            return
        # ttk.Panedwindow has (num_panes - 1) sashes; each is identified by
        # its 0-based index. Walk through all of them and clamp.
        for i in range(num_panes - 1):
            try:
                pos = int(pane.sashpos(i))
            except Exception:
                continue
            min_pos = min_pane_px * (i + 1)
            max_pos = total - min_pane_px * (num_panes - 1 - i)
            if max_pos < min_pos:
                # Window is too small to honour both minimums; leave alone.
                continue
            clamped = max(min_pos, min(pos, max_pos))
            if clamped != pos:
                try:
                    pane.sashpos(i, clamped)
                except Exception:
                    pass

    pane.bind("<B1-Motion>", _enforce, add="+")
    pane.bind("<ButtonRelease-1>", _enforce, add="+")
    pane.bind("<Configure>", _enforce, add="+")


def reset_panes_proportionally(
    pane: ttk.Panedwindow,
    *,
    weights: Sequence[float],
) -> None:
    """Restore sash positions so each pane gets its share of ``weights``.

    Useful for a "Reset Layout" button — call it with the same weights the
    panes were originally added with, and the user gets back to the default
    configuration.
    """
    try:
        num_panes = len(pane.panes())
    except Exception:
        return
    if num_panes < 2 or len(weights) < num_panes:
        return
    try:
        orient = str(pane.cget("orient")) or "horizontal"
    except Exception:
        orient = "horizontal"
    total = int(pane.winfo_width() or 0) if orient.startswith("h") else int(pane.winfo_height() or 0)
    if total <= 0:
        return
    weight_sum = float(sum(weights[:num_panes])) or 1.0
    cumulative = 0.0
    for i in range(num_panes - 1):
        cumulative += float(weights[i]) / weight_sum
        target = int(cumulative * total)
        try:
            pane.sashpos(i, target)
        except Exception:
            pass

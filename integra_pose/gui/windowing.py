from __future__ import annotations

import tkinter as tk


def apply_adaptive_window_geometry(
    window: tk.Misc,
    *,
    preferred_size: tuple[int, int],
    min_size: tuple[int, int] | None = None,
    width_ratio: float = 0.92,
    height_ratio: float = 0.9,
) -> tuple[int, int]:
    """Clamp initial and minimum window sizes to the active display."""
    window.update_idletasks()

    screen_width = max(int(window.winfo_screenwidth() or 0), 1)
    screen_height = max(int(window.winfo_screenheight() or 0), 1)
    max_width = max(480, int(screen_width * float(width_ratio)))
    max_height = max(320, int(screen_height * float(height_ratio)))

    preferred_width = min(max(int(preferred_size[0]), 320), max_width)
    preferred_height = min(max(int(preferred_size[1]), 240), max_height)

    if min_size is not None:
        min_width = min(max(int(min_size[0]), 320), preferred_width)
        min_height = min(max(int(min_size[1]), 240), preferred_height)
        try:
            window.minsize(min_width, min_height)
        except Exception:
            pass

    try:
        window.geometry(f"{preferred_width}x{preferred_height}")
    except Exception:
        pass

    return preferred_width, preferred_height

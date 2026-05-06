"""Lightweight splash screen shown during IntegraPose startup."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class SplashScreen:
    """A minimal, borderless splash screen with a status line."""

    def __init__(
        self,
        root: tk.Misc,
        *,
        title: str = "IntegraPose",
        subtitle: str | None = None,
        gif_path: str | None = None,
        initial_status: str = "Starting...",
        width: int = 420,
        height: int = 180,
        auto_size: bool = True,
        max_screen_fraction: float = 0.8,
    ) -> None:
        self._root = root
        self._window = tk.Toplevel(root)
        self._window.overrideredirect(True)
        self._window.attributes("-topmost", True)
        self._animation_job: str | None = None
        self._gif_frames: list[tk.PhotoImage] = []
        self._gif_delays_ms: list[int] = []
        self._gif_frame_index = 0
        self._gif_display_size: tuple[int, int] | None = None

        screen_w, screen_h = self._screen_size()
        max_w = int(screen_w * max_screen_fraction) if screen_w else width
        max_h = int(screen_h * max_screen_fraction) if screen_h else height
        max_w = max(width, min(max_w, screen_w - 80)) if screen_w else max(width, max_w)
        max_h = max(height, min(max_h, screen_h - 80)) if screen_h else max(height, max_h)
        max_w = max(240, max_w)
        max_h = max(160, max_h)

        reserved_height = 170
        extra_width = 60
        max_image_w = max(120, max_w - extra_width)
        max_image_h = max(80, max_h - reserved_height)

        border = tk.Frame(self._window, background="#0f172a")
        border.pack(fill="both", expand=True)

        container = tk.Frame(border, background="#ffffff")
        container.pack(fill="both", expand=True, padx=2, pady=2)

        content = ttk.Frame(container, padding=(18, 16))
        content.pack(fill="both", expand=True)

        ttk.Label(content, text=title, font=("Segoe UI", 14, "bold")).pack(anchor="w")
        if subtitle:
            ttk.Label(content, text=subtitle).pack(anchor="w", pady=(2, 10))
        else:
            ttk.Label(content, text="").pack(anchor="w", pady=(2, 10))

        self._image_label = ttk.Label(content)
        if gif_path:
            self._load_gif(gif_path, max_size=(max_image_w, max_image_h))
        if self._gif_frames:
            self._image_label.pack(anchor="center", pady=(0, 10))
            self._start_animation()

        self._status_var = tk.StringVar(value=initial_status)
        ttk.Label(content, textvariable=self._status_var).pack(anchor="w", pady=(0, 10))

        self._progress = ttk.Progressbar(content, mode="indeterminate")
        self._progress.pack(fill="x")
        self._progress.start(10)

        window_w = width
        window_h = height
        if auto_size and self._gif_display_size:
            gif_w, gif_h = self._gif_display_size
            window_w = max(width, min(max_w, gif_w + extra_width))
            window_h = max(height, min(max_h, gif_h + reserved_height))

        x, y = self._center_geometry(window_w, window_h)
        self._window.geometry(f"{window_w}x{window_h}+{x}+{y}")

        try:
            self._window.update_idletasks()
        except Exception:
            pass

    def set_status(self, message: str) -> None:
        try:
            self._status_var.set(message)
            self._window.update_idletasks()
        except Exception:
            pass

    def close(self) -> None:
        if self._animation_job is not None:
            try:
                self._window.after_cancel(self._animation_job)
            except Exception:
                pass
            self._animation_job = None
        try:
            self._progress.stop()
        except Exception:
            pass
        try:
            if self._window.winfo_exists():
                self._window.destroy()
        except Exception:
            pass

    def _center_geometry(self, width: int, height: int) -> tuple[int, int]:
        screen_w, screen_h = self._screen_size()
        if not screen_w or not screen_h:
            return 100, 100
        x = max(0, int((screen_w - width) / 2))
        y = max(0, int((screen_h - height) / 2))
        return x, y

    def _screen_size(self) -> tuple[int | None, int | None]:
        try:
            return self._window.winfo_screenwidth(), self._window.winfo_screenheight()
        except Exception:
            return None, None

    def _load_gif(self, path: str, *, max_size: tuple[int, int] | None = None) -> None:
        try:
            from PIL import Image, ImageSequence, ImageTk  # type: ignore
        except Exception:
            return

        try:
            img = Image.open(path)
        except Exception:
            return

        target_size = img.size
        if max_size:
            max_w, max_h = max_size
            try:
                img_w, img_h = img.size
                scale = min(max_w / img_w, max_h / img_h, 1.0)
                target_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
            except Exception:
                target_size = img.size

        resample = None
        try:
            resample = Image.Resampling.LANCZOS  # Pillow >= 9
        except Exception:
            try:
                resample = Image.LANCZOS  # Pillow < 9
            except Exception:
                resample = None

        frames: list[tk.PhotoImage] = []
        delays: list[int] = []
        try:
            for frame in ImageSequence.Iterator(img):
                duration = int(frame.info.get("duration") or 80)
                duration = max(20, min(2000, duration))
                delays.append(duration)
                rgba = frame.convert("RGBA")
                if target_size and rgba.size != target_size:
                    if resample is not None:
                        rgba = rgba.resize(target_size, resample=resample)
                    else:
                        rgba = rgba.resize(target_size)
                frames.append(ImageTk.PhotoImage(rgba))
        except Exception:
            frames = []
            delays = []

        if frames:
            self._gif_frames = frames
            self._gif_delays_ms = delays
            self._gif_frame_index = 0
            try:
                self._gif_display_size = (frames[0].width(), frames[0].height())
            except Exception:
                self._gif_display_size = target_size
            try:
                self._image_label.configure(image=self._gif_frames[0])
            except Exception:
                pass

    def _start_animation(self) -> None:
        if not self._gif_frames:
            return

        def _advance() -> None:
            if not self._gif_frames:
                return
            try:
                self._gif_frame_index = (self._gif_frame_index + 1) % len(self._gif_frames)
                self._image_label.configure(image=self._gif_frames[self._gif_frame_index])
            except Exception:
                return
            delay = self._gif_delays_ms[self._gif_frame_index] if self._gif_delays_ms else 80
            self._animation_job = self._window.after(delay, _advance)

        delay0 = self._gif_delays_ms[0] if self._gif_delays_ms else 80
        self._animation_job = self._window.after(delay0, _advance)

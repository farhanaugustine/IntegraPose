"""Startup bootstrap helpers (splash screen + lazy imports)."""

from __future__ import annotations

import importlib
import queue
import threading
import traceback
from pathlib import Path
from typing import Sequence


_ProgressEvent = tuple[str, str]


def _preload_modules(progress: "queue.Queue[_ProgressEvent]", modules: Sequence[tuple[str, str]]) -> None:
    """Import heavyweight modules in a background thread and stream status updates."""
    try:
        for module_name, message in modules:
            progress.put(("status", message))
            importlib.import_module(module_name)
    except Exception:
        progress.put(("error", traceback.format_exc()))
    else:
        progress.put(("done", ""))


def _run_splash_process(
    conn: object,
    *,
    title: str,
    subtitle: str | None,
    gif_path: str | None,
    initial_status: str,
) -> None:
    """Run a Tk splash screen in a dedicated process."""
    import tkinter as tk

    from integra_pose.gui.splash import SplashScreen

    try:
        connection = conn
        poll = getattr(connection, "poll", None)
        recv = getattr(connection, "recv", None)
    except Exception:
        return

    if not callable(poll) or not callable(recv):
        return

    root = tk.Tk()
    root.withdraw()
    splash = SplashScreen(
        root,
        title=title,
        subtitle=subtitle,
        gif_path=gif_path,
        initial_status=initial_status,
    )

    def _pump_messages() -> None:
        try:
            while poll(0):
                event, payload = recv()
                if event == "status":
                    splash.set_status(str(payload))
                elif event == "close":
                    splash.close()
                    root.destroy()
                    return
        except EOFError:
            try:
                splash.close()
            except Exception:
                pass
            try:
                root.destroy()
            except Exception:
                pass
            return
        except Exception:
            pass

        try:
            if root.winfo_exists():
                root.after(50, _pump_messages)
        except Exception:
            pass

    root.after(0, _pump_messages)
    try:
        root.mainloop()
    finally:
        try:
            close = getattr(conn, "close", None)
            if callable(close):
                close()
        except Exception:
            pass


def _launch_gui_threaded(*, debug: bool = False) -> None:
    """Launch the GUI with a single-process splash (may freeze during imports)."""
    import tkinter as tk
    from tkinter import messagebox

    from integra_pose import __version__
    from integra_pose.gui.splash import SplashScreen

    root = tk.Tk()
    root.withdraw()

    gif_candidate = Path(__file__).resolve().parent / "assets" / "splash" / "splash.gif"
    gif_path = str(gif_candidate) if gif_candidate.is_file() else None

    splash = SplashScreen(
        root,
        title="IntegraPose",
        subtitle=f"v{__version__}",
        gif_path=gif_path,
        initial_status="Starting up...",
    )

    progress_queue: "queue.Queue[_ProgressEvent]" = queue.Queue()
    preload_steps: Sequence[tuple[str, str]] = (
        ("numpy", "Loading NumPy..."),
        ("pandas", "Loading pandas..."),
        ("PIL.Image", "Loading Pillow..."),
        ("cv2", "Loading OpenCV..."),
        ("yaml", "Loading YAML..."),
        ("supervision", "Loading Supervision..."),
    )

    worker = threading.Thread(
        target=_preload_modules,
        args=(progress_queue, preload_steps),
        daemon=True,
    )
    worker.start()

    state = {"started": False}

    def _fail(exc: Exception, details: str | None = None) -> None:
        try:
            splash.set_status("Startup failed.")
        except Exception:
            pass
        message = f"Failed to start IntegraPose:\n\n{exc}"
        if details:
            message = f"{message}\n\nDetails:\n{details}"
        try:
            messagebox.showerror("IntegraPose Startup Error", message, parent=root)
        except Exception:
            pass
        try:
            splash.close()
        finally:
            root.destroy()

    def _start_app() -> None:
        splash.set_status("Building UI...")
        try:
            from integra_pose import main_gui_app

            main_gui_app.configure_terminal_logging(debug=debug)
            app = main_gui_app.YoloApp(root)
        except Exception as exc:
            _fail(exc, traceback.format_exc())
            return

        try:
            splash.close()
        except Exception:
            pass
        try:
            root.deiconify()
            root.lift()
        except Exception:
            pass

    def _poll_queue() -> None:
        try:
            while True:
                event, payload = progress_queue.get_nowait()
                if event == "status":
                    splash.set_status(payload)
                elif event == "error":
                    state["started"] = True
                    _fail(RuntimeError("One or more modules failed to import."), payload)
                    return
                elif event == "done":
                    if not state["started"]:
                        state["started"] = True
                        root.after(0, _start_app)
                    return
        except queue.Empty:
            pass

        if not state["started"]:
            root.after(60, _poll_queue)

    root.after(0, _poll_queue)
    root.mainloop()


def launch_gui(*, debug: bool = False) -> None:
    """Launch the IntegraPose GUI with a responsive startup splash screen."""
    from integra_pose import __version__

    gif_candidate = Path(__file__).resolve().parent / "assets" / "splash" / "splash.gif"
    gif_path = str(gif_candidate) if gif_candidate.is_file() else None

    preload_steps: Sequence[tuple[str, str]] = (
        ("numpy", "Loading NumPy..."),
        ("pandas", "Loading pandas..."),
        ("PIL.Image", "Loading Pillow..."),
        ("cv2", "Loading OpenCV..."),
        ("yaml", "Loading YAML..."),
        ("supervision", "Loading Supervision..."),
    )

    splash_conn = None
    splash_process = None
    try:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        splash_conn, child_conn = ctx.Pipe()
        splash_process = ctx.Process(
            target=_run_splash_process,
            args=(child_conn,),
            kwargs={
                "title": "IntegraPose",
                "subtitle": f"v{__version__}",
                "gif_path": gif_path,
                "initial_status": "Starting up...",
            },
            daemon=True,
        )
        splash_process.start()
        try:
            child_conn.close()
        except Exception:
            pass
    except Exception:
        return _launch_gui_threaded(debug=debug)

    def _splash(event: str, payload: str = "") -> None:
        if splash_conn is None:
            return
        try:
            splash_conn.send((event, payload))
        except Exception:
            pass

    def _close_splash() -> None:
        nonlocal splash_conn, splash_process
        if splash_conn is not None:
            try:
                splash_conn.send(("close", ""))
            except Exception:
                pass
            try:
                splash_conn.close()
            except Exception:
                pass
            splash_conn = None
        if splash_process is not None:
            try:
                splash_process.join(timeout=1.0)
            except Exception:
                pass
            try:
                if splash_process.is_alive():
                    splash_process.terminate()
                    splash_process.join(timeout=0.5)
            except Exception:
                pass
            splash_process = None

    def _show_error(exc: Exception, details: str | None = None) -> None:
        message = f"Failed to start IntegraPose:\n\n{exc}"
        if details:
            message = f"{message}\n\nDetails:\n{details}"
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            try:
                messagebox.showerror("IntegraPose Startup Error", message, parent=root)
            finally:
                root.destroy()
        except Exception:
            pass

    try:
        for module_name, message in preload_steps:
            _splash("status", message)
            importlib.import_module(module_name)

        _splash("status", "Building UI...")
        from integra_pose import main_gui_app

        main_gui_app.configure_terminal_logging(debug=debug)

        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        main_gui_app.YoloApp(root)

        _splash("close", "")
        try:
            if splash_conn is not None:
                splash_conn.close()
        except Exception:
            pass

        try:
            root.deiconify()
            root.lift()
        except Exception:
            pass
        try:
            root.mainloop()
        finally:
            _close_splash()
    except Exception as exc:
        details = traceback.format_exc()
        try:
            _splash("status", "Startup failed.")
        except Exception:
            pass
        _close_splash()
        _show_error(exc, details)

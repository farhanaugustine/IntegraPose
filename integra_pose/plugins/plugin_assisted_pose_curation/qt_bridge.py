"""Run the Qt editor separately while retaining the proven curation controller.

Only JSON travels over private child-process pipes. Tk and Qt never share an
event loop; all controller calls, including dialogs, run on the Tk main thread.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Any

from . import ui
from .core import serialize_pose_points, sanitize_skeleton_edges, VISIBILITY_VISIBLE
from .qt_protocol import COMMANDS, SETTINGS

AssistedPoseCurationUIError = ui.AssistedPoseCurationUIError


class _Controller(ui.AssistedPoseCurationWindow):
    """Keep the existing workflow widgets as controller state, without mapping them."""

    def _configure_adaptive_window(self):
        super()._configure_adaptive_window()
        self.withdraw()

    def _render_scene(self):
        # Qt renders original pixels and vector overlays, never a Tk screenshot.
        self._update_status_text()


class AssistedPoseCurationWindow:
    def __init__(self, main_app, parent=None):
        self._controller = _Controller(main_app, parent=parent)
        self._parent = parent or self._controller
        self._incoming: queue.Queue = queue.Queue()
        self._outgoing: queue.Queue = queue.Queue(maxsize=4)
        self._last_state = None
        self._closed = False
        self._closing = False
        self._ready = False
        self._fallback = False
        self._stderr: list[str] = []
        root = Path(__file__).resolve().parents[3]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(root), environment.get("PYTHONPATH", "")) if part
        )
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-m", f"{__package__}.qt_editor"],
                cwd=root, env=environment, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self._controller.destroy()
            raise
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._write, daemon=True).start()
        threading.Thread(target=self._read_errors, daemon=True).start()
        self._parent.after(60, self._poll)
        self._parent.bind("<Destroy>", self._parent_destroyed, add="+")

    def _parent_destroyed(self, event):
        if event.widget is self._parent:
            self._closed = True
            if self._process.poll() is None:
                self._process.terminate()
            self._outgoing.put_nowait(None) if not self._outgoing.full() else None

    def _read(self):
        try:
            for line in self._process.stdout:
                try:
                    message = json.loads(line)
                    if isinstance(message, dict):
                        self._incoming.put(message)
                except ValueError:
                    self._stderr.append(line.strip())
        except (OSError, ValueError):
            pass

    def _read_errors(self):
        for line in self._process.stderr:
            self._stderr.append(line.rstrip())
            self._stderr[:] = self._stderr[-30:]

    def _write(self):
        try:
            while True:
                message = self._outgoing.get()
                if message is None:
                    return
                self._process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
                self._process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            return

    def _send(self, message):
        if not self._outgoing.full():
            self._outgoing.put_nowait(message)
            return True
        return False

    def _snapshot(self):
        c = self._controller
        def instances(items):
            return [dict(class_id=int(item.get("class_id", 0)),
                         pose=serialize_pose_points(item.get("pose", []))) for item in items]
        return {
            "path": str(c._current_path or ""), "index": c._current_index,
            "frames": [str(p) for p in c._image_paths],
            "instances": instances(c._frame_instances),
            "assist": instances(c._assist_instances),
            "active_instance": c._active_instance_index, "active_point": c._active_kp_index,
            "classes": c._class_names, "keypoints": c._keypoint_names,
            "skeleton": c._skeleton_edges, "dirty": c._dirty,
            "integrity_error": c._label_integrity_error,
            "settings": {name: getattr(c, f"_{name}_var").get() for name in SETTINGS},
            "status": c._status_var.get(), "review": c._review_state_var.get(),
            "counts": c._review_counts_var.get(), "task": c._task_status_var.get(),
            "progress": c._task_progress_value.get(),
            "progress_max": float(c._task_progress.cget("maximum")),
            "progress_mode": str(c._task_progress.cget("mode")),
            "log": c._log_text.get("1.0", "end")[-16000:],
            "export": c._export_summary.get("1.0", "end"),
        }

    def _poll(self):
        if self._closed or not self._controller.winfo_exists():
            return
        if self._process.poll() is not None:
            if self._closing:
                self._closed = True
                self._controller.destroy()
            else:
                # Never discard unsaved edits if Qt or its DLLs fail.
                self._fallback = True
                self._controller._render_scene = ui.AssistedPoseCurationWindow._render_scene.__get__(self._controller)
                self._controller.deiconify()
                self._controller._render_scene()
                detail = "\n".join(self._stderr[-12:])
                self._controller._append_error(f"Qt editor exited; session recovered in Tk. {detail}")
                ui.messagebox.showerror("Qt editor closed", "The Qt editor exited unexpectedly. Your session is available in the original window.\n\n" + detail, parent=self._controller)
            self._outgoing.put_nowait(None) if not self._outgoing.full() else None
            return
        # Reserve space for acknowledgements; periodic state is expendable.
        if not self._incoming.empty() and not self._outgoing.full():
            request = self._incoming.get_nowait()
            try:
                self._dispatch(request)
                error = None
            except Exception as exc:
                error = str(exc)
                self._controller._append_error(error)
            self._send({"type": "ack", "id": request.get("id"), "error": error,
                        "state": self._snapshot(), "close": self._closing})
            self._last_state = None
        elif self._incoming.empty():
            state = self._snapshot()
            if state != self._last_state and self._send({"type": "state", "state": state}):
                self._last_state = state
        self._parent.after(60, self._poll)

    def _dispatch(self, request: dict[str, Any]):
        c = self._controller
        command = request.get("command")
        data = request.get("data") or {}
        if self._closing:
            return
        if command == "ready":
            self._ready = True
        elif command in COMMANDS:
            getattr(c, COMMANDS[command])()
        elif command == "settings":
            values = data.get("values", {})
            unknown = set(values) - set(SETTINGS)
            if unknown:
                raise ValueError(f"Unknown settings: {sorted(unknown)}")
            if c._dirty and not c._commit_before_navigation():
                return
            for name, value in values.items():
                getattr(c, f"_{name}_var").set(value)
            c._invalidate_model_cache(include_embed=True)
            c._set_standard_layout()
            followup = data.get("then")
            if followup in COMMANDS:
                getattr(c, COMMANDS[followup])()
        elif command == "schema":
            if not c._commit_before_navigation():
                return
            names = [str(s).strip() for s in data["keypoints"] if str(s).strip()]
            classes = [str(s).strip() for s in data["classes"] if str(s).strip()]
            if not names or not classes or len(set(names)) != len(names):
                raise ValueError("Provide nonempty classes and unique keypoint names.")
            edges = sanitize_skeleton_edges(data["skeleton"], names)
            c._class_text.delete("1.0", "end")
            c._class_text.insert("1.0", "\n".join(classes))
            c._apply_class_names()
            c._keypoint_text.delete("1.0", "end")
            c._keypoint_text.insert("1.0", "\n".join(names))
            c._apply_keypoint_names()
            c._skeleton_edges = edges
            c._refresh_skeleton_controls()
        elif command == "jump":
            index = int(data["index"])
            if 0 <= index < len(c._image_paths) and index != c._current_index and c._commit_before_navigation():
                c._current_index = index
                c._open_current_image()
        elif command in {"select", "edit", "class"}:
            if data.get("path") != str(c._current_path or ""):
                raise ValueError("The frame changed before this edit arrived. Please try again.")
            instance = int(data["instance"])
            if not 0 <= instance < len(c._frame_instances):
                raise ValueError("The selected instance is no longer available.")
            c._select_instance(instance)
            if command == "class":
                class_id = int(data["class_id"])
                if not 0 <= class_id < len(c._class_names):
                    raise ValueError("Unknown class.")
                c._assign_selected_instance_class(class_id)
                return
            index = int(data["point"])
            if not 0 <= index < len(c._current_pose):
                raise ValueError("Unknown keypoint.")
            c._active_kp_index = index
            if command == "edit":
                if c._label_integrity_error:
                    raise ValueError(c._label_integrity_error)
                x, y = float(data["x"]), float(data["y"])
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("Keypoint coordinates must be finite.")
                h, w = c._current_image.shape[:2]
                point = c._current_pose[index]
                point.x, point.y = max(0, min(w - 1, x)), max(0, min(h - 1, y))
                point.v, point.conf = int(data.get("visibility", VISIBILITY_VISIBLE)), 0.0
                if point.v not in (0, 1, 2):
                    point.v = VISIBILITY_VISIBLE
                c._mark_pose_edited()
            c._refresh_keypoint_tree()
            c._render_scene()
        elif command == "close":
            choice = data.get("choice", "cancel")
            if choice == "save" and c._dirty and not c._save_current_pose():
                return
            if choice in {"save", "discard"}:
                self._closing = True
        else:
            raise ValueError(f"Unknown curation command: {command}")

    def apply_prefill(self, **kwargs):
        if self._controller._dirty and any(value for value in kwargs.values()):
            if not self._controller._commit_before_navigation():
                return
        self._controller.apply_prefill(**kwargs)
        self._last_state = None

    def winfo_exists(self):
        return not self._closed and self._controller.winfo_exists()

    def lift(self):
        if self._fallback:
            self._controller.lift()
        else:
            self._send({"type": "raise"})

    def focus_force(self):
        self.lift()

    def bind(self, sequence, callback, add=None):
        if sequence == "<Destroy>":
            self._controller.bind(sequence, lambda e: callback(e) if e.widget is self._controller else None, add=add)

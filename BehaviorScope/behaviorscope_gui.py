from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:  # pragma: no cover - optional dependency (needed for real-time preview)
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover
    Image = None
    ImageTk = None


APP_DIR = Path(__file__).resolve().parent


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 500):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._tip_window: tk.Toplevel | None = None

        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, _event=None):
        self._schedule()

    def _on_leave(self, _event=None):
        self._cancel()
        self._hide()

    def _schedule(self):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        if self._after_id is None:
            return
        try:
            self.widget.after_cancel(self._after_id)
        except Exception:
            pass
        self._after_id = None

    def _show(self):
        if self._tip_window is not None or not self.text:
            return
        root = self.widget.winfo_toplevel()
        self._tip_window = tk.Toplevel(root)
        self._tip_window.wm_overrideredirect(True)
        self._tip_window.attributes("-topmost", True)

        label = ttk.Label(
            self._tip_window,
            text=self.text,
            justify="left",
            padding=(10, 6),
            style="Tooltip.TLabel",
            wraplength=460,
        )
        label.pack()

        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self._tip_window.wm_geometry(f"+{x}+{y}")

    def _hide(self):
        if self._tip_window is None:
            return
        try:
            self._tip_window.destroy()
        except Exception:
            pass
        self._tip_window = None


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        self.inner = ttk.Frame(self.canvas)
        self._inner_window = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.inner.bind("<Configure>", self._on_inner_configure, add="+")
        self.canvas.bind("<Configure>", self._on_canvas_configure, add="+")

        self.inner.bind("<Enter>", self._bind_mousewheel, add="+")
        self.inner.bind("<Leave>", self._unbind_mousewheel, add="+")

    def _on_inner_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, _event=None):
        self.canvas.itemconfig(self._inner_window, width=self.canvas.winfo_width())

    def _bind_mousewheel(self, _event=None):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux, add="+")
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux, add="+")

    def _unbind_mousewheel(self, _event=None):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if event.delta == 0:
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


@dataclass
class RunningProcess:
    process: subprocess.Popen
    output_queue: "queue.Queue[str]"
    reader_thread: threading.Thread


@dataclass
class RunningRealtime:
    stop_event: threading.Event
    output_queue: "queue.Queue[dict]"
    worker_thread: threading.Thread


@dataclass(frozen=True)
class RealtimeConfig:
    model_path: Path
    yolo_weights: Path
    yolo_mode: str
    yolo_task: str | None
    source_kind: str
    camera_index: int | None
    video_path: Path | None
    device: str
    prob_threshold: float
    window_size: int | None
    window_stride: int | None
    crop_size: int | None
    pad_ratio: float
    yolo_conf: float
    yolo_iou: float
    yolo_imgsz: int
    keep_last_box: bool
    draw_bbox: bool
    overlay_font_scale: float
    overlay_thickness: int
    smoothing: str
    ema_alpha: float
    interp_max_gap: int
    show_speed: bool


class BehaviorScopeGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("BehaviorScope GUI")
        self.root.minsize(980, 720)

        self._running: RunningProcess | None = None
        self._realtime: RunningRealtime | None = None
        self._realtime_photo = None
        self._configure_style()
        self._build_menu()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.preprocess_tab = self._build_preprocess_tab()
        self.train_tab = self._build_train_tab()
        self.infer_tab = self._build_infer_tab()
        self.batch_tab = self._build_batch_tab()
        self.realtime_tab = self._build_realtime_tab()
        self.benchmark_tab = self._build_benchmark_tab()
        self.review_tab = self._build_review_tab()

        self.notebook.add(self.preprocess_tab["frame"], text="Preprocess")      
        self.notebook.add(self.train_tab["frame"], text="Train")
        self.notebook.add(self.infer_tab["frame"], text="Inference")
        self.notebook.add(self.batch_tab["frame"], text="Batch")
        self.notebook.add(self.realtime_tab["frame"], text="Real-time")
        self.notebook.add(self.benchmark_tab["frame"], text="Benchmark")
        self.notebook.add(self.review_tab["frame"], text="Review")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Tooltip.TLabel", background="#ffffe0", relief="solid", borderwidth=1)
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Hint.TLabel", foreground="#555555")

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save GUI settings...", command=self._save_settings)
        file_menu.add_command(label="Load GUI settings...", command=self._load_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Generate default config...", command=self._generate_default_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

    def _generate_default_config(self):
        initial = str(APP_DIR)
        path = filedialog.asksaveasfilename(
            title="Save default configuration",
            initialdir=initial,
            initialfile="config.yaml",
            defaultextension=".yaml",
            filetypes=[("YAML Config", "*.yaml"), ("All files", "*.*")],
        )
        if not path:
            return
        
        src = APP_DIR / "default_config.yaml"
        if not src.exists():
            messagebox.showerror("Error", "default_config.yaml not found in application directory.")
            return
            
        try:
            dest = Path(path)
            dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            messagebox.showinfo("Success", f"Saved configuration to:\n{dest}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save config: {exc}")

    def _load_config_to_ui(self, config_path: Path):
        try:
            import yaml
        except ImportError:
            messagebox.showerror("Error", "PyYAML is required to load config files. pip install pyyaml")
            return

        if not config_path.exists():
            messagebox.showerror("Error", f"Config file not found: {config_path}")
            return

        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            
            # Helper to set var if exists
            def _set(tab_prefix, key, value):
                var_key = f"{tab_prefix}.{key}"
                if value is None:
                    value = ""
                # Check train tab vars
                if hasattr(self, "train_tab") and var_key in self.train_tab["vars"]:
                    self.train_tab["vars"][var_key].set(str(value).lower() if isinstance(value, bool) else str(value))
                # Check preprocess tab vars
                elif hasattr(self, "preprocess_tab") and var_key in self.preprocess_tab["vars"]:
                    self.preprocess_tab["vars"][var_key].set(str(value).lower() if isinstance(value, bool) else str(value))

            def _to_bool(value):
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return bool(value)
                if isinstance(value, str):
                    token = value.strip().lower()
                    if token in {"1", "true", "t", "yes", "y", "on"}:
                        return True
                    if token in {"0", "false", "f", "no", "n", "off"}:
                        return False
                return bool(value)

            # Mapping logic
            # Model section -> Train tab
            if "model" in cfg:
                for k, v in cfg["model"].items():
                    if k == "no_pretrained_backbone":
                        _set("train", "use_pretrained", not _to_bool(v))
                        continue
                    if k == "pretrained_backbone":
                        _set("train", "use_pretrained", _to_bool(v))
                        continue
                    _set("train", k, v)
             
            # Training section -> Train tab
            if "training" in cfg:
                for k, v in cfg["training"].items():
                    if k == "no_pretrained_backbone":
                        _set("train", "use_pretrained", not _to_bool(v))
                        continue
                    if k == "pretrained_backbone":
                        _set("train", "use_pretrained", _to_bool(v))
                        continue
                    _set("train", k, v)
            
            # Preprocessing section -> Preprocess tab
            if "preprocessing" in cfg:
                for k, v in cfg["preprocessing"].items():
                    _set("preprocess", k, v)
            
            # Inference section -> Inference tab? (GUI doesn't expose all inference params in vars dict easily accessible here, but we can try)
            # Actually inference vars are local to _build_infer_tab unless I stored them.
            # I didn't store inference vars in self.infer_tab["vars"] in my last check (I only checked train/prep).
            # So I'll skip inference for now to avoid errors.

            messagebox.showinfo("Loaded", f"Configuration loaded from {config_path.name}")

        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load config: {exc}")

    def _show_about(self):
        messagebox.showinfo(
            "About BehaviorScope GUI",
            "BehaviorScope GUI wraps preprocessing, training, inference, and review into a tabbed UI.\n\n"
            "Hover over fields to read detailed tooltips.",
        )

    def _on_close(self):
        if self._running is not None or self._realtime is not None:
            if not messagebox.askyesno(
                "Process running",
                "A process (or real-time inference) is still running. Stop it and exit?",
            ):
                return
            self._stop_process()
            self._stop_realtime()
        self.root.destroy()

    def _append_log(self, log: ScrolledText, text: str):
        log.configure(state="normal")
        log.insert("end", text)
        log.see("end")
        log.configure(state="disabled")

    def _clear_log(self, log: ScrolledText):
        log.configure(state="normal")
        log.delete("1.0", "end")
        log.configure(state="disabled")

    def _set_enabled(self, widget: ttk.Widget, enabled: bool):
        try:
            if isinstance(widget, ttk.Combobox):
                widget.configure(state=("readonly" if enabled else "disabled"))
                return
            widget.configure(state=("normal" if enabled else "disabled"))
        except Exception:
            pass

    def _run_process(
        self,
        cmd: list[str],
        log: ScrolledText,
        run_btn: ttk.Button,
        stop_btn: ttk.Button,
    ):
        if self._running is not None or self._realtime is not None:
            messagebox.showwarning(
                "Already running",
                "A pipeline step (or real-time inference) is already running. Stop it before starting another.",
            )
            return

        self._append_log(log, f"\n$ {' '.join(cmd)}\n\n")
        run_btn.configure(state="disabled")
        stop_btn.configure(state="normal")

        q: "queue.Queue[str]" = queue.Queue()
        proc = subprocess.Popen(
            cmd,
            cwd=str(APP_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _reader():
            assert proc.stdout is not None
            for line in proc.stdout:
                q.put(line)
            proc.stdout.close()

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        self._running = RunningProcess(process=proc, output_queue=q, reader_thread=t)

        def _poll():
            running = self._running
            if running is None:
                return
            drained = 0
            while drained < 200:
                try:
                    line = running.output_queue.get_nowait()
                except queue.Empty:
                    break
                self._append_log(log, line)
                drained += 1

            if proc.poll() is None:
                self.root.after(80, _poll)
                return

            self._append_log(log, f"\n[Process exited with code {proc.returncode}]\n")
            self._running = None
            run_btn.configure(state="normal")
            stop_btn.configure(state="disabled")

        self.root.after(80, _poll)

    def _stop_process(self):
        running = self._running
        if running is None:
            return
        stop_file = None
        try:
            args = getattr(running.process, "args", None)
            if isinstance(args, list) and "--stop_file" in args:
                idx = args.index("--stop_file") + 1
                if idx < len(args):
                    stop_file = Path(str(args[idx]))
        except Exception:
            stop_file = None
        if stop_file is not None:
            try:
                stop_file.parent.mkdir(parents=True, exist_ok=True)
                stop_file.write_text("stop\n", encoding="utf-8")
                return
            except Exception:
                pass
        try:
            running.process.terminate()
        except Exception:
            pass

    def _stop_realtime(self):
        running = self._realtime
        if running is None:
            return
        running.stop_event.set()

    def _realtime_worker(
        self,
        cfg: RealtimeConfig,
        out_q: "queue.Queue[dict]",
        stop_event: threading.Event,
    ):
        def _put(msg: dict) -> None:
            try:
                out_q.put_nowait(msg)
            except queue.Full:
                try:
                    out_q.get_nowait()
                except queue.Empty:
                    return
                try:
                    out_q.put_nowait(msg)
                except Exception:
                    return

        try:
            import time
            from bisect import bisect_left
            from collections import deque

            import cv2
            import numpy as np
            import torch
            from torch.nn import functional as F
        except Exception as exc:
            _put({"type": "error", "message": f"Failed to import runtime deps: {exc}"})
            return

        try:
            import cropping
            import infer_behavior_lstm
            import quantification
            import review_annotations
            from utils.pose_features import extract_rich_pose_features
        except Exception as exc:
            _put({"type": "error", "message": f"Failed to import project modules: {exc}"})
            return

        if cfg.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(cfg.device)

        _put({"type": "status", "text": f"Loading behavior model on {device}..."})
        try:
            model, class_names, hparams = infer_behavior_lstm.load_checkpoint(
                cfg.model_path, device
            )
        except Exception as exc:
            _put({"type": "error", "message": f"Failed to load model checkpoint: {exc}"})
            return

        default_num_frames = int(hparams.get("num_frames", 16))
        default_frame_size = int(hparams.get("frame_size", 160))

        window_size = cfg.window_size or default_num_frames
        stride = cfg.window_stride or max(1, window_size // 2)
        crop_size = cfg.crop_size or default_frame_size

        backbone_name = str(hparams.get("backbone", "")).lower()
        if backbone_name.startswith("vit") and int(crop_size) != 224:
            _put(
                {
                    "type": "error",
                    "message": (
                        "ViT backbones require 224x224 crops in this implementation. "
                        "Leave crop_size blank or set it to 224."
                    ),
                }
            )
            return

        if device.type == "cuda":
            yolo_device = f"cuda:{device.index}" if device.index is not None else "cuda"
        else:
            yolo_device = "cpu"

        _put({"type": "status", "text": f"Loading YOLO model on {yolo_device}..."})
        try:
            # Use the explicitly provided task, or fallback to auto (None)
            yolo_task = cfg.yolo_task if cfg.yolo_task and cfg.yolo_task != "auto" else None
            yolo_model = cropping.load_yolo_model(cfg.yolo_weights, device=yolo_device, task=yolo_task)
        except Exception as exc:
            _put({"type": "error", "message": f"Failed to load YOLO weights: {exc}"})
            return

        transform = infer_behavior_lstm.build_frame_transform(crop_size, augment=False)
        yolo_mode = str(cfg.yolo_mode or "auto").strip().lower()
        if yolo_mode not in {"auto", "bbox", "pose"}:
            _put(
                {
                    "type": "error",
                    "message": f"Unknown YOLO mode: {cfg.yolo_mode!r} (expected auto/bbox/pose).",
                }
            )
            return

        if cfg.source_kind == "webcam":
            source = int(cfg.camera_index or 0)
        else:
            assert cfg.video_path is not None
            source = str(cfg.video_path)

        _put({"type": "status", "text": f"Opening video source: {source}"})
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            _put({"type": "error", "message": f"Unable to open video source: {source}"})
            try:
                cap.release()
            except Exception:
                pass
            return

        started = time.perf_counter()
        last_frame_time = started
        last_ui_update = 0.0
        min_ui_interval = 1.0 / 15.0

        frames_seen = 0
        windows_done = 0
        ema_fps: float | None = None

        last_bbox = None
        last_conf = 0.0
        last_keypoints = None
        buffer: "deque[cropping.CropDetection]" = deque()

        last_pred_label = "Warming up..."
        last_pred_conf = 0.0

        delay_frames = int(max(0, window_size))
        frame_queue: "deque[dict]" = deque()
        past_context: "deque[dict]" = deque(maxlen=max(10, delay_frames))
        latest_ready: dict | None = None

        window_centers: list[int] = []
        window_preds: list[tuple[str, float]] = []

        last_motion_xy: tuple[float, float] | None = None
        last_motion_time: float | None = None

        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                frames_seen += 1
                capture_ts = time.perf_counter()

                best_bbox = None
                best_conf = None
                best_keypoints = None
                yolo_status = "missed"
                try:
                    results = yolo_model.predict(
                        source=frame,
                        device=yolo_device,
                        conf=cfg.yolo_conf,
                        iou=cfg.yolo_iou,
                        imgsz=cfg.yolo_imgsz,
                        max_det=1,
                        nms=True,
                        verbose=False,
                    )
                except Exception as exc:
                    _put({"type": "error", "message": f"YOLO inference failed: {exc}"})
                    return

                if results:
                    result = results[0]
                    boxes = getattr(result, "boxes", None)
                    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
                        xyxy = boxes.xyxy
                        if xyxy is not None and len(xyxy) > 0:
                            confs = boxes.conf.detach().cpu().numpy()
                            best_idx = int(confs.argmax())
                            best_bbox = xyxy[best_idx].detach().cpu().numpy()
                            best_conf = float(confs[best_idx])
                            last_bbox = best_bbox
                            last_conf = best_conf
                            if yolo_mode != "bbox":
                                try:
                                    keypoints = getattr(result, "keypoints", None)
                                    if keypoints is not None:
                                        xy = getattr(keypoints, "xy", None)
                                        if xy is not None and len(xy) > best_idx:
                                            xy_i = xy[best_idx]
                                            if hasattr(xy_i, "detach"):
                                                xy_i = xy_i.detach()
                                            if hasattr(xy_i, "cpu"):
                                                xy_i = xy_i.cpu()
                                            xy_np = (
                                                xy_i.numpy()
                                                if hasattr(xy_i, "numpy")
                                                else np.asarray(xy_i)
                                            )
                                            conf_list = None
                                            conf = getattr(keypoints, "conf", None)
                                            if conf is not None and len(conf) > best_idx:
                                                conf_i = conf[best_idx]
                                                if hasattr(conf_i, "detach"):
                                                    conf_i = conf_i.detach()
                                                if hasattr(conf_i, "cpu"):
                                                    conf_i = conf_i.cpu()
                                                conf_np = (
                                                    conf_i.numpy()
                                                    if hasattr(conf_i, "numpy")
                                                    else np.asarray(conf_i)
                                                )
                                                conf_list = conf_np.tolist()
                                            pts = []
                                            for j, (x, y) in enumerate(xy_np.tolist()):
                                                c_val = None
                                                if conf_list is not None and j < len(conf_list):
                                                    try:
                                                        c_val = float(conf_list[j])
                                                        if not np.isfinite(c_val):
                                                            c_val = None
                                                    except Exception:
                                                        c_val = None
                                                pts.append((float(x), float(y), c_val))
                                            if pts:
                                                best_keypoints = pts
                                except Exception:
                                    best_keypoints = None
                                last_keypoints = best_keypoints
                            else:
                                best_keypoints = None
                                last_keypoints = None
                            yolo_status = "detected"

                if best_bbox is None and cfg.keep_last_box and last_bbox is not None:
                    best_bbox = last_bbox
                    best_conf = last_conf
                    best_keypoints = last_keypoints
                    yolo_status = "reused"

                bbox_xyxy = None
                if best_bbox is not None:
                    bbox_xyxy = tuple(int(v) for v in best_bbox)
                    crop_xyxy = cropping.compute_crop_xyxy(frame, best_bbox, cfg.pad_ratio)
                    crop_rgb = cropping.crop_and_resize(
                        frame, best_bbox, cfg.pad_ratio, crop_size
                    )
                    keypoints_xy = None
                    keypoints_conf = None
                    if best_keypoints:
                        keypoints_xy = tuple((x, y) for x, y, _ in best_keypoints)
                        keypoints_conf = tuple(
                            float(c) if c is not None else 0.0
                            for _, _, c in best_keypoints
                        )
                    buffer.append(
                        cropping.CropDetection(
                            frame_idx=frames_seen - 1,
                            crop_rgb=crop_rgb,
                            confidence=float(best_conf or 0.0),
                            bbox_xyxy=bbox_xyxy,
                            crop_xyxy=tuple(int(v) for v in crop_xyxy),
                            yolo_status=str(yolo_status),
                            keypoints_xy=keypoints_xy,
                            keypoints_conf=keypoints_conf,
                        )
                    )

                frame_queue.append(
                    {
                        "frame_idx": int(frames_seen - 1),
                        "frame_bgr": frame,
                        "ts": float(capture_ts),
                        "bbox_xyxy": bbox_xyxy,
                        "yolo_status": str(yolo_status),
                        "yolo_conf": float(best_conf or 0.0),
                        "keypoints": best_keypoints,
                    }
                )
                while len(frame_queue) > delay_frames:
                    latest_ready = frame_queue.popleft()
                    past_context.append(latest_ready)

                while len(buffer) >= window_size:
                    entries = list(buffer)[:window_size]
                    processed = [transform(det.crop_rgb) for det in entries]
                    clip_tensor = (
                        torch.stack(processed, dim=0).unsqueeze(0).to(device)
                    )
                    
                    pose_tensor = None
                    # Only extract pose if we have keypoints and mode is not bbox
                    if yolo_mode != "bbox" and entries:
                        try:
                            # Only compute pose features when ALL frames have keypoints.
                            first_xy = entries[0].keypoints_xy
                            if first_xy is not None:
                                num_kpts = len(first_xy)
                                pose_all = True
                                for entry in entries:
                                    if entry.keypoints_xy is None or len(entry.keypoints_xy) != num_kpts:
                                        pose_all = False
                                        break
                                    if entry.crop_xyxy is None:
                                        pose_all = False
                                        break
                                if pose_all:
                                    kpts_arr = np.zeros((len(entries), num_kpts, 3), dtype=np.float32)
                                    for i, entry in enumerate(entries):
                                        xy = entry.keypoints_xy
                                        conf = entry.keypoints_conf
                                        crop_xyxy = entry.crop_xyxy
                                        assert xy is not None
                                        assert crop_xyxy is not None
                                        crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
                                        crop_w = max(int(crop_x2) - int(crop_x1), 1)
                                        crop_h = max(int(crop_y2) - int(crop_y1), 1)
                                        for k in range(num_kpts):
                                            x_raw, y_raw = xy[k]
                                            kpts_arr[i, k, 0] = (float(x_raw) - float(crop_x1)) / float(crop_w)
                                            kpts_arr[i, k, 1] = (float(y_raw) - float(crop_y1)) / float(crop_h)
                                            c = 1.0
                                            if conf is not None and k < len(conf):
                                                try:
                                                    c = float(conf[k])
                                                except Exception:
                                                    c = 1.0
                                            kpts_arr[i, k, 2] = c
                                    feats = extract_rich_pose_features(kpts_arr)
                                    pose_tensor = torch.from_numpy(feats).float().unsqueeze(0).to(device)
                        except Exception:
                            # If pose extraction fails, fallback to None (RGB only)
                            pose_tensor = None

                    with torch.no_grad():
                        inputs = (clip_tensor, pose_tensor) if pose_tensor is not None else clip_tensor
                        logits = model(inputs)
                        probs = F.softmax(logits, dim=1)
                    top_p, top_idx = probs.max(dim=1)
                    prob = float(top_p.item())
                    idx = int(top_idx.item())
                    if prob < cfg.prob_threshold:
                        last_pred_label = "Unknown"
                    else:
                        last_pred_label = class_names[idx]
                    last_pred_conf = prob
                    start_frame = int(entries[0].frame_idx)
                    end_frame = int(entries[-1].frame_idx)
                    center_frame = int((start_frame + end_frame) // 2)
                    window_centers.append(center_frame)
                    window_preds.append((str(last_pred_label), float(last_pred_conf)))
                    windows_done += 1

                    for _ in range(stride):
                        if buffer:
                            buffer.popleft()

                now = time.perf_counter()
                dt = now - last_frame_time
                last_frame_time = now
                if dt > 1e-6:
                    inst = 1.0 / dt
                    ema_fps = inst if ema_fps is None else (0.9 * ema_fps + 0.1 * inst)

                elapsed = now - started
                avg_fps = frames_seen / elapsed if elapsed > 1e-6 else 0.0
                win_per_sec = windows_done / elapsed if elapsed > 1e-6 else 0.0

                if now - last_ui_update >= min_ui_interval:
                    last_ui_update = now
                    if latest_ready is None:
                        continue

                    if len(window_centers) > 5000:
                        window_centers[:] = window_centers[-3000:]
                        window_preds[:] = window_preds[-3000:]

                    display_frame = latest_ready["frame_bgr"]
                    display_idx = int(latest_ready["frame_idx"])
                    display_bbox = latest_ready.get("bbox_xyxy")
                    display_status = str(latest_ready.get("yolo_status") or "")
                    display_ts = float(latest_ready.get("ts") or 0.0)
                    display_keypoints = latest_ready.get("keypoints")

                    pred_label = "Warming up..."
                    pred_conf = float("nan")
                    if window_centers:
                        pos = int(bisect_left(window_centers, display_idx))
                        if pos <= 0:
                            pred_label, pred_conf = window_preds[0]
                        elif pos >= len(window_centers):
                            pred_label, pred_conf = window_preds[-1]
                        else:
                            prev_c = int(window_centers[pos - 1])
                            next_c = int(window_centers[pos])
                            if (display_idx - prev_c) <= (next_c - display_idx):
                                pred_label, pred_conf = window_preds[pos - 1]
                            else:
                                pred_label, pred_conf = window_preds[pos]

                        keep_from = int(
                            bisect_left(window_centers, display_idx - window_size * 2)
                        )
                        if keep_from > 0:
                            window_centers[:] = window_centers[keep_from:]
                            window_preds[:] = window_preds[keep_from:]

                    if display_status == "missed":
                        pred_label = "Unknown"
                        pred_conf = float("nan")

                    speed_px_s: float | None = None
                    if cfg.show_speed and display_bbox is not None and display_status != "missed":
                        context_packets = list(past_context) + list(frame_queue)
                        bboxes = [p.get("bbox_xyxy") for p in context_packets]
                        centers_raw = quantification.compute_centers_xy(bboxes)
                        centers_smooth = quantification.smooth_centers_xy(
                            centers_raw,
                            method=cfg.smoothing,
                            ema_alpha=cfg.ema_alpha,
                            interp_max_gap=cfg.interp_max_gap,
                        )
                        idx_ready = max(0, len(past_context) - 1)
                        cx = float(centers_smooth[idx_ready, 0])
                        cy = float(centers_smooth[idx_ready, 1])
                        if np.isfinite(cx) and np.isfinite(cy):
                            if (
                                last_motion_xy is not None
                                and last_motion_time is not None
                                and display_ts > last_motion_time + 1e-6
                            ):
                                dx = cx - float(last_motion_xy[0])
                                dy = cy - float(last_motion_xy[1])
                                dist = float(np.hypot(dx, dy))
                                dt_s = float(display_ts - last_motion_time)
                                speed_px_s = dist / dt_s
                            last_motion_xy = (cx, cy)
                            last_motion_time = float(display_ts)

                    label_text = str(pred_label)
                    if speed_px_s is not None and np.isfinite(speed_px_s):
                        label_text = f"{label_text} | {speed_px_s:.1f}px/s"

                    if cfg.draw_bbox:
                        overlay = review_annotations.annotate_frame_bbox(
                            display_frame,
                            display_bbox,
                            label_text,
                            float(pred_conf) if np.isfinite(pred_conf) else None,
                            display_status,
                            font_scale=float(cfg.overlay_font_scale),
                            thickness=int(cfg.overlay_thickness),
                            keypoints=display_keypoints,
                        )
                    else:
                        overlay = display_frame.copy()
                        cv2.putText(
                            overlay,
                            f"{label_text} ({float(pred_conf):.2f})"
                            if np.isfinite(pred_conf)
                            else label_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            float(cfg.overlay_font_scale),
                            (255, 255, 255),
                            int(cfg.overlay_thickness),
                            lineType=cv2.LINE_AA,
                        )

                    rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    _put(
                        {
                            "type": "frame",
                            "rgb": rgb,
                            "pred_label": str(pred_label),
                            "pred_conf": float(pred_conf) if np.isfinite(pred_conf) else 0.0,
                            "fps": float(ema_fps or 0.0),
                            "fps_avg": float(avg_fps),
                            "win_per_sec": float(win_per_sec),
                            "frames_seen": frames_seen,
                            "windows_done": windows_done,
                        }
                    )
                    continue
                    overlay = frame.copy()
                    if cfg.draw_bbox and bbox_xyxy is not None:
                        x1, y1, x2, y2 = bbox_xyxy
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    warmup = ""
                    if len(buffer) < window_size:
                        warmup = f"  warmup {len(buffer)}/{window_size}"

                    cv2.putText(
                        overlay,
                        f"Pred: {last_pred_label} ({last_pred_conf:.2f}){warmup}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        overlay,
                        f"FPS: {float(ema_fps or 0.0):.2f} (avg {avg_fps:.2f})  win/s {win_per_sec:.2f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )

                    rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    _put(
                        {
                            "type": "frame",
                            "rgb": rgb,
                            "pred_label": last_pred_label,
                            "pred_conf": last_pred_conf,
                            "fps": float(ema_fps or 0.0),
                            "fps_avg": float(avg_fps),
                            "win_per_sec": float(win_per_sec),
                            "frames_seen": frames_seen,
                            "windows_done": windows_done,
                        }
                    )
        finally:
            try:
                cap.release()
            except Exception:
                pass

        elapsed_total = time.perf_counter() - started
        _put(
            {
                "type": "stopped",
                "text": (
                    f"Stopped. frames={frames_seen} windows={windows_done} "
                    f"runtime={elapsed_total:.2f}s avg_fps={(frames_seen / max(elapsed_total, 1e-6)):.2f}"
                ),
            }
        )

    def _browse_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory(initialdir=var.get() or str(APP_DIR))    
        if path:
            var.set(path)

    def _browse_open_file(
        self,
        var: tk.StringVar,
        title: str,
        filetypes: list[tuple[str, str]],
    ):
        initial = str(Path(var.get()).parent) if var.get() else str(APP_DIR)
        path = filedialog.askopenfilename(title=title, initialdir=initial, filetypes=filetypes)
        if path:
            var.set(path)

    def _browse_save_file(
        self,
        var: tk.StringVar,
        title: str,
        defaultextension: str,
        filetypes: list[tuple[str, str]],
    ):
        initial = str(Path(var.get()).parent) if var.get() else str(APP_DIR)
        path = filedialog.asksaveasfilename(
            title=title,
            initialdir=initial,
            defaultextension=defaultextension,
            filetypes=filetypes,
        )
        if path:
            var.set(path)

    def _add_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        widget: ttk.Widget,
        tooltip: str | None = None,
    ):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
        widget.grid(row=row, column=1, sticky="ew", pady=3)
        parent.grid_columnconfigure(1, weight=1)
        if tooltip:
            Tooltip(lbl, tooltip)
            Tooltip(widget, tooltip)

    def _add_path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        var: tk.StringVar,
        browse_kind: str,
        tooltip: str,
        filetypes: list[tuple[str, str]] | None = None,
        save_ext: str | None = None,
    ):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)

        container = ttk.Frame(parent)
        container.grid(row=row, column=1, sticky="ew", pady=3)
        container.grid_columnconfigure(0, weight=1)

        entry = ttk.Entry(container, textvariable=var)
        entry.grid(row=0, column=0, sticky="ew")

        def _browse():
            if browse_kind == "dir":
                self._browse_dir(var)
            elif browse_kind == "open_file":
                assert filetypes is not None
                self._browse_open_file(var, title=label, filetypes=filetypes)
            elif browse_kind == "save_file":
                assert filetypes is not None and save_ext is not None
                self._browse_save_file(var, title=label, defaultextension=save_ext, filetypes=filetypes)

        btn = ttk.Button(container, text="Browse…", command=_browse)
        btn.grid(row=0, column=1, padx=(8, 0))

        parent.grid_columnconfigure(1, weight=1)
        Tooltip(lbl, tooltip)
        Tooltip(entry, tooltip)
        Tooltip(btn, tooltip)

        return entry

    def _parse_int(self, name: str, raw: str, allow_blank: bool = False) -> int | None:
        text = raw.strip()
        if allow_blank and not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer.") from exc

    def _parse_float(self, name: str, raw: str, allow_blank: bool = False) -> float | None:
        text = raw.strip()
        if allow_blank and not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number.") from exc

    def _build_preprocess_tab(self):
        tab = ttk.Frame(self.notebook)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        form = ScrollableFrame(tab)
        form.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        buttons = ttk.Frame(tab)
        buttons.grid(row=1, column=0, sticky="ew", padx=10)

        log_frame = ttk.Labelframe(tab, text="Log")
        log_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(8, 10))
        tab.grid_rowconfigure(2, weight=1)

        log = ScrolledText(log_frame, height=10, state="disabled")
        log.pack(fill="both", expand=True)

        header = ttk.Label(
            form.inner,
            text=(
                "Preprocessing: run YOLO on class-sorted clips, crop the animal, "
                "and build overlapping windows + a manifest JSON."
            ),
            style="Header.TLabel",
            wraplength=900,
        )
        header.grid(row=0, column=0, sticky="w", pady=(0, 10))

        paths = ttk.Labelframe(form.inner, text="Paths")
        paths.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        dataset_root = tk.StringVar(value="")
        yolo_weights = tk.StringVar(value="")
        output_root = tk.StringVar(value="")
        manifest_path = tk.StringVar(value="")

        self._add_path_row(
            paths,
            0,
            "Dataset root (class folders)",
            dataset_root,
            "dir",
            tooltip=(
                "Folder containing one subfolder per behavior/class. Each class folder contains videos.\n\n"
                "Example:\n  dataset_root/Ambulatory/*.mp4\n  dataset_root/Grooming/*.mp4"
            ),
        )
        self._add_path_row(
            paths,
            1,
            "YOLO weights (.pt)",
            yolo_weights,
            "open_file",
            tooltip=(
                "Path to your Ultralytics YOLO weights (e.g., best.pt) trained to detect the rodent."
            ),
            filetypes=[("YOLO Weights", "*.pt;*.onnx;*.engine;*.xml;*.h5"), ("All files", "*.*")],
        )
        self._add_path_row(
            paths,
            2,
            "Output root (optional)",
            output_root,
            "dir",
            tooltip=(
                "Where to write qa_crops/, sequence_npz/, and the manifest.\n\n"
                "If blank, defaults to <dataset_root>/processed_sequences."
            ),
        )
        self._add_path_row(
            paths,
            3,
            "Manifest path (optional)",
            manifest_path,
            "save_file",
            tooltip=(
                "Optional override for the manifest JSON location.\n\n"
                "If blank, defaults to <output_root>/sequence_manifest.json."
            ),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            save_ext=".json",
        )

        windowing = ttk.Labelframe(form.inner, text="Windowing & crops")
        windowing.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        window_size = tk.StringVar(value="6")
        window_stride = tk.StringVar(value="2")
        crop_size = tk.StringVar(value="224")
        pad_ratio = tk.StringVar(value="0.25")

        self._add_row(
            windowing,
            0,
            "Window size (frames)",
            ttk.Entry(windowing, textvariable=window_size),
            tooltip=(
                "How many consecutive frames are stacked into ONE clip/window.\n\n"
                "Bigger window_size = more temporal context, but more compute and memory.\n"
                "This should match training/inference (or let them auto-read it from the manifest/checkpoint)."
            ),
        )
        self._add_row(
            windowing,
            1,
            "Window stride (frames)",
            ttk.Entry(windowing, textvariable=window_stride),
            tooltip=(
                "How far the sliding window advances each step.\n\n"
                "Smaller stride = more overlap and more training examples (slower + larger dataset).\n"
                "Common choice: window_stride = window_size/2."
            ),
        )
        self._add_row(
            windowing,
            2,
            "Crop size (pixels)",
            ttk.Entry(windowing, textvariable=crop_size),
            tooltip=(
                "Final square crop size after resizing YOLO crops (e.g., 224x224).\n\n"
                "Must match training/inference frame_size to avoid shape mismatches."
            ),
        )
        self._add_row(
            windowing,
            3,
            "Pad ratio",
            ttk.Entry(windowing, textvariable=pad_ratio),
            tooltip=(
                "Expands the YOLO bounding box by this fraction before cropping.\n\n"
                "Higher pad_ratio reduces the chance of cutting off limbs, but includes more background."
            ),
        )

        yolo = ttk.Labelframe(form.inner, text="YOLO runtime")
        yolo.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        yolo_conf = tk.StringVar(value="0.10")
        yolo_iou = tk.StringVar(value="0.50")
        yolo_imgsz = tk.StringVar(value="640")
        device = tk.StringVar(value="cpu")
        keep_last_box = tk.BooleanVar(value=True)
        yolo_task = tk.StringVar(value="auto")

        self._add_row(
            yolo,
            0,
            "YOLO confidence",
            ttk.Entry(yolo, textvariable=yolo_conf),
            tooltip="Minimum detection confidence. Lower = more detections (and more false positives).",
        )
        self._add_row(
            yolo,
            1,
            "YOLO IoU (NMS)",
            ttk.Entry(yolo, textvariable=yolo_iou),
            tooltip="Non-maximum suppression IoU threshold used by YOLO to filter overlapping boxes.",
        )
        self._add_row(
            yolo,
            2,
            "YOLO imgsz",
            ttk.Entry(yolo, textvariable=yolo_imgsz),
            tooltip="YOLO inference resolution. Higher can improve detections but slows inference (e.g., 640).",
        )
        self._add_row(
            yolo,
            3,
            "Device",
            ttk.Entry(yolo, textvariable=device),
            tooltip="Device for YOLO inference (e.g., cpu, cuda, cuda:0).",
        )
        self._add_row(
            yolo,
            4,
            "Keep last box",
            ttk.Checkbutton(yolo, variable=keep_last_box),
            tooltip="If YOLO misses a frame, reuse the last valid bounding box to keep crops stable.",
        )
        
        yolo_task_combo = ttk.Combobox(
            yolo, textvariable=yolo_task, values=("auto", "detect", "pose"), state="readonly"
        )
        self._add_row(
            yolo,
            5,
            "YOLO Task",
            yolo_task_combo,
            tooltip="Explicitly set YOLO task (detect vs pose). 'Auto' infers it from the weights.",
        )

        outputs = ttk.Labelframe(form.inner, text="Outputs")
        outputs.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        save_npz = tk.BooleanVar(value=True)
        save_frame_crops = tk.BooleanVar(value=False)
        skip_existing = tk.BooleanVar(value=True)

        self._add_row(
            outputs,
            0,
            "Save sequence NPZ",
            ttk.Checkbutton(outputs, variable=save_npz),
            tooltip=(
                "Writes one .npz per window (frames tensor). Recommended for training speed.\n"
                "If disabled, training can load crops from JPEGs instead."
            ),
        )
        self._add_row(
            outputs,
            1,
            "Save frame crops (QA JPEGs)",
            ttk.Checkbutton(outputs, variable=save_frame_crops),
            tooltip="Writes per-frame crops as JPEGs for visual QA. Uses more disk and time.",
        )
        self._add_row(
            outputs,
            2,
            "Skip existing outputs",
            ttk.Checkbutton(outputs, variable=skip_existing),
            tooltip="Skips videos that already have outputs in the destination folders.",
        )

        splits = ttk.Labelframe(form.inner, text="Train/Val/Test split")
        splits.grid(row=5, column=0, sticky="ew", pady=(0, 10))

        train_ratio = tk.StringVar(value="0.80")
        val_ratio = tk.StringVar(value="0.10")
        test_ratio = tk.StringVar(value="0.10")
        seed = tk.StringVar(value="42")
        min_sequence_count = tk.StringVar(value="1")
        default_fps = tk.StringVar(value="30.0")
        split_strategy = tk.StringVar(value="window")

        self._add_row(
            splits,
            0,
            "Train ratio",
            ttk.Entry(splits, textvariable=train_ratio),
            tooltip="Fraction of windows per class assigned to the training split.",
        )
        self._add_row(
            splits,
            1,
            "Val ratio",
            ttk.Entry(splits, textvariable=val_ratio),
            tooltip="Fraction of windows per class assigned to the validation split.",
        )
        self._add_row(
            splits,
            2,
            "Test ratio",
            ttk.Entry(splits, textvariable=test_ratio),
            tooltip="Fraction of windows per class assigned to the test split (can be 0).",
        )
        self._add_row(
            splits,
            3,
            "Seed",
            ttk.Entry(splits, textvariable=seed),
            tooltip="Random seed for reproducible stratified splitting.",
        )
        self._add_row(
            splits,
            4,
            "Min sequence count",
            ttk.Entry(splits, textvariable=min_sequence_count),
            tooltip="Drops source videos that produce fewer than this many windows.",
        )
        self._add_row(
            splits,
            5,
            "Default FPS",
            ttk.Entry(splits, textvariable=default_fps),
            tooltip="Fallback FPS used when a video's metadata does not expose FPS.",
        )
        split_strategy_combo = ttk.Combobox(
            splits,
            textvariable=split_strategy,
            values=("window", "video"),
            state="readonly",
        )
        self._add_row(
            splits,
            6,
            "Split strategy",
            split_strategy_combo,
            tooltip=(
                "window: split individual windows (may leak overlapping windows across splits) | "
                "video: keep all windows from the same source video together (recommended)."
            ),
        )

        def _suggest_outputs():
            try:
                root_path = Path(dataset_root.get().strip())
                if not root_path.exists():
                    return
                suggested_output = root_path / "processed_sequences"
                if not output_root.get().strip():
                    output_root.set(str(suggested_output))
                if not manifest_path.get().strip():
                    manifest_path.set(str(Path(output_root.get().strip() or suggested_output) / "sequence_manifest.json"))
            except Exception:
                return

        ttk.Button(form.inner, text="Suggest output paths", command=_suggest_outputs).grid(
            row=6, column=0, sticky="w", pady=(0, 10)
        )

        run_btn = ttk.Button(buttons, text="Run preprocessing")
        stop_btn = ttk.Button(buttons, text="Stop", state="disabled", command=self._stop_process)
        clear_btn = ttk.Button(buttons, text="Clear log", command=lambda: self._clear_log(log))
        run_btn.grid(row=0, column=0, sticky="w")
        stop_btn.grid(row=0, column=1, sticky="w", padx=(10, 0))
        clear_btn.grid(row=0, column=2, sticky="w", padx=(10, 0))

        def _run():
            try:
                ds = Path(dataset_root.get().strip())
                yw = Path(yolo_weights.get().strip())
                if not ds.exists():
                    raise ValueError("Dataset root does not exist.")
                if not yw.exists():
                    raise ValueError("YOLO weights file does not exist.")

                ws = self._parse_int("Window size", window_size.get())
                st = self._parse_int("Window stride", window_stride.get())
                cs = self._parse_int("Crop size", crop_size.get())
                pr = self._parse_float("Pad ratio", pad_ratio.get())
                yc = self._parse_float("YOLO confidence", yolo_conf.get())
                yi = self._parse_float("YOLO IoU", yolo_iou.get())
                ys = self._parse_int("YOLO imgsz", yolo_imgsz.get())
                tr = self._parse_float("Train ratio", train_ratio.get())
                vr = self._parse_float("Val ratio", val_ratio.get())
                ter = self._parse_float("Test ratio", test_ratio.get())
                sd = self._parse_int("Seed", seed.get())
                msc = self._parse_int("Min sequence count", min_sequence_count.get())
                dfps = self._parse_float("Default FPS", default_fps.get())

                if not save_npz.get() and not save_frame_crops.get():
                    raise ValueError(
                        "Enable at least one output: Save sequence NPZ and/or Save frame crops."
                    )

                cmd = [
                    sys.executable,
                    str(APP_DIR / "prepare_sequences_from_yolo.py"),
                    "--dataset_root",
                    str(ds),
                    "--yolo_weights",
                    str(yw),
                    "--window_size",
                    str(ws),
                    "--window_stride",
                    str(st),
                    "--crop_size",
                    str(cs),
                    "--pad_ratio",
                    str(pr),
                    "--yolo_conf",
                    str(yc),
                    "--yolo_iou",
                    str(yi),
                    "--yolo_imgsz",
                    str(ys),
                    "--device",
                    device.get().strip() or "cpu",
                    "--train_ratio",
                    str(tr),
                    "--val_ratio",
                    str(vr),
                    "--test_ratio",
                    str(ter),
                    "--split_strategy",
                    split_strategy.get().strip() or "window",
                    "--seed",
                    str(sd),
                    "--min_sequence_count",
                    str(msc),
                    "--default_fps",
                    str(dfps),
                ]

                if yolo_task.get() and yolo_task.get() != "auto":
                    cmd += ["--yolo_task", yolo_task.get()]

                if output_root.get().strip():
                    cmd += ["--output_root", output_root.get().strip()]
                if manifest_path.get().strip():
                    cmd += ["--manifest_path", manifest_path.get().strip()]
                if keep_last_box.get():
                    cmd.append("--keep_last_box")
                if save_npz.get():
                    cmd.append("--save_sequence_npz")
                if save_frame_crops.get():
                    cmd.append("--save_frame_crops")
                if skip_existing.get():
                    cmd.append("--skip_existing")

                self._run_process(cmd, log, run_btn, stop_btn)
            except Exception as exc:
                messagebox.showerror("Preprocess error", str(exc))

        run_btn.configure(command=_run)

        return {
            "frame": tab,
            "log": log,
            "vars": {
                "preprocess.dataset_root": dataset_root,
                "preprocess.yolo_weights": yolo_weights,
                "preprocess.output_root": output_root,
                "preprocess.manifest_path": manifest_path,
                "preprocess.window_size": window_size,
                "preprocess.window_stride": window_stride,
                "preprocess.crop_size": crop_size,
                "preprocess.pad_ratio": pad_ratio,
                "preprocess.yolo_conf": yolo_conf,
                "preprocess.yolo_iou": yolo_iou,
                "preprocess.yolo_imgsz": yolo_imgsz,
                "preprocess.device": device,
                "preprocess.keep_last_box": keep_last_box,
                "preprocess.save_sequence_npz": save_npz,
                "preprocess.save_frame_crops": save_frame_crops,
                "preprocess.skip_existing": skip_existing,
                "preprocess.train_ratio": train_ratio,
                "preprocess.val_ratio": val_ratio,
                "preprocess.test_ratio": test_ratio,
                "preprocess.split_strategy": split_strategy,
                "preprocess.seed": seed,
                "preprocess.min_sequence_count": min_sequence_count,
                "preprocess.default_fps": default_fps,
            },
        }

    def _build_train_tab(self):
        tab = ttk.Frame(self.notebook)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        form = ScrollableFrame(tab)
        form.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        buttons = ttk.Frame(tab)
        buttons.grid(row=1, column=0, sticky="ew", padx=10)

        log_frame = ttk.Labelframe(tab, text="Log")
        log_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(8, 10))
        tab.grid_rowconfigure(2, weight=1)

        log = ScrolledText(log_frame, height=10, state="disabled")
        log.pack(fill="both", expand=True)

        header = ttk.Label(
            form.inner,
            text="Training: train a temporal behavior classifier from a preprocessing manifest.",
            style="Header.TLabel",
            wraplength=900,
        )
        header.grid(row=0, column=0, sticky="w", pady=(0, 10))

        paths = ttk.Labelframe(form.inner, text="Paths")
        paths.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        config_path = tk.StringVar(value="")
        manifest_path = tk.StringVar(value="")
        output_dir = tk.StringVar(value=str(APP_DIR / "runs"))
        run_name = tk.StringVar(value="slowfast_win6")
        resume_training = tk.BooleanVar(value=False)

        config_entry = self._add_path_row(
            paths,
            0,
            "Config YAML (optional)",
            config_path,
            "open_file",
            tooltip="Path to a YAML configuration file. If provided, values in the file override the defaults (but GUI settings below might still override command line args if not handled carefully). In this implementation, CLI args take precedence over config defaults if passed.",
            filetypes=[("YAML Config", "*.yaml"), ("All files", "*.*")],
        )
        
        def _trigger_load_config():
            path = config_path.get().strip()
            if not path:
                return
            self._load_config_to_ui(Path(path))

        load_btn = ttk.Button(config_entry.master, text="Load", command=_trigger_load_config, width=6)
        load_btn.grid(row=0, column=2, padx=(5, 0))
        Tooltip(load_btn, "Parse this YAML file and populate the GUI fields below.")
        self._add_path_row(
            paths,
            1,
            "Manifest JSON",
            manifest_path,
            "open_file",
            tooltip="Manifest produced by preprocessing (sequence_manifest.json). Contains splits + window/crop metadata.",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        self._add_path_row(
            paths,
            2,
            "Output directory",
            output_dir,
            "dir",
            tooltip="Folder where training runs are saved (subfolder named by run_name).",
        )
        self._add_row(
            paths,
            3,
            "Run name",
            ttk.Entry(paths, textvariable=run_name),
            tooltip="Subfolder name for this training run (checkpoints, logs, class mapping).",
        )
        self._add_row(
            paths,
            4,
            "Resume run",
            ttk.Checkbutton(paths, variable=resume_training),
            tooltip="Resume from an interrupted run (loads last_checkpoint.pt if present in the run folder).",
        )

        basics = ttk.Labelframe(form.inner, text="Training basics")
        basics.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        epochs = tk.StringVar(value="500")
        batch_size = tk.StringVar(value="8")
        lr = tk.StringVar(value="5e-5")
        weight_decay = tk.StringVar(value="3e-4")
        dropout = tk.StringVar(value="0.5")
        device = tk.StringVar(value="cuda:0")
        seed = tk.StringVar(value="42")
        patience = tk.StringVar(value="50")
        num_workers = tk.StringVar(value="0")
        log_interval = tk.StringVar(value="0")
        backbone_lr = tk.StringVar(value="1e-4")
        scheduler = tk.StringVar(value="none")
        disable_augment = tk.BooleanVar(value=False)
        per_class_metrics = tk.BooleanVar(value=False)
        confusion_matrix = tk.BooleanVar(value=False)
        confusion_matrix_interval = tk.StringVar(value="1")
        class_weighting = tk.StringVar(value="none")
        class_weight_max = tk.StringVar(value="0.0")
        train_sampler = tk.StringVar(value="shuffle")
        
        scheduler_combo = ttk.Combobox(
            basics, 
            textvariable=scheduler, 
            state="readonly", 
            values=["none", "plateau", "cosine"]
        )
        class_weighting_combo = ttk.Combobox(
            basics,
            textvariable=class_weighting,
            state="readonly",
            values=["none", "inverse", "sqrt_inverse"],
        )
        class_weight_max_entry = ttk.Entry(basics, textvariable=class_weight_max)
        train_sampler_combo = ttk.Combobox(
            basics,
            textvariable=train_sampler,
            state="readonly",
            values=["shuffle", "weighted"],
        )

        self._add_row(basics, 0, "Epochs", ttk.Entry(basics, textvariable=epochs), tooltip="Number of passes over the training set.")
        self._add_row(basics, 1, "Batch size", ttk.Entry(basics, textvariable=batch_size), tooltip="Clips per batch. Increase until you hit GPU memory limits.")
        self._add_row(
            basics,
            2,
            "Head Learning rate",
            ttk.Entry(basics, textvariable=lr),
            tooltip="Optimizer step size for the head (and backbone if 'Backbone LR' is empty).",
        )
        self._add_row(
            basics,
            3,
            "Backbone LR (Differential)",
            ttk.Entry(basics, textvariable=backbone_lr),
            tooltip=(
                "Optional lower learning rate for the backbone (e.g., 1e-5) when fine-tuning.\n"
                "Leave empty to use the main learning rate for everything."
            ),
        )
        self._add_row(basics, 4, "Weight decay", ttk.Entry(basics, textvariable=weight_decay), tooltip="L2-style regularization (AdamW). Helps reduce overfitting.")
        self._add_row(
            basics,
            5,
            "Dropout",
            ttk.Entry(basics, textvariable=dropout),
            tooltip=(
                "Dropout probability used in the classifier head and temporal attention/pooling blocks.\n\n"
                "Very high values (e.g., 0.6+) can prevent attention components from learning."
            ),
        )
        self._add_row(basics, 6, "Device", ttk.Entry(basics, textvariable=device), tooltip="Device for training (e.g., cuda:0, cpu, auto).")
        self._add_row(basics, 7, "Seed", ttk.Entry(basics, textvariable=seed), tooltip="Random seed for reproducibility.")
        self._add_row(basics, 8, "Early-stop patience", ttk.Entry(basics, textvariable=patience), tooltip="Stop if validation accuracy doesn't improve for this many epochs.")
        self._add_row(basics, 9, "DataLoader workers", ttk.Entry(basics, textvariable=num_workers), tooltip="Background workers for data loading. On Windows, 0 is often simplest.")
        self._add_row(basics, 10, "Log interval (steps)", ttk.Entry(basics, textvariable=log_interval), tooltip="Print batch-level progress every N steps (0 disables).")
        self._add_row(
            basics,
            11,
            "LR Scheduler",
            scheduler_combo,
            tooltip=(
                "Strategy to adjust learning rate during training.\n"
                "- none: Constant LR.\n"
                "- plateau: Reduce LR when validation accuracy stalls (factor=0.5).\n"
                "- cosine: Cosine Annealing (smooth decay to 1% of initial LR)."
            ),
        )
        self._add_row(basics, 12, "Disable augmentations", ttk.Checkbutton(basics, variable=disable_augment), tooltip="Turn off training-time frame augmentations.")
        self._add_row(
            basics,
            13,
            "Per-class metrics",
            ttk.Checkbutton(basics, variable=per_class_metrics),
            tooltip="Print per-class validation/test accuracy each epoch (helps spot confused behaviors).",
        )
        self._add_row(
            basics,
            14,
            "Confusion matrices",
            ttk.Checkbutton(basics, variable=confusion_matrix),
            tooltip="Print raw + normalized confusion matrices and save PNG heatmaps to the run directory.",
        )
        self._add_row(
            basics,
            15,
            "Confusion interval (epochs)",
            ttk.Entry(basics, textvariable=confusion_matrix_interval),
            tooltip="How often to print validation confusion matrices (1 = every epoch).",
        )
        self._add_row(
            basics,
            16,
            "Class weighting",
            class_weighting_combo,
            tooltip=(
                "Reweight CrossEntropyLoss using training-set label frequency.\n\n"
                "- none: standard loss (majority classes dominate).\n"
                "- inverse: strong reweighting (rare classes matter more).\n"
                "- sqrt_inverse: milder reweighting.\n\n"
                "Tip: Use macro-F1 + confusion matrices to evaluate rare-class improvements."
            ),
        )
        self._add_row(
            basics,
            17,
            "Class weight clamp (max)",
            class_weight_max_entry,
            tooltip=(
                "Caps the largest class weight used by CrossEntropyLoss.\n\n"
                "Plain language: a safety limit on how much extra the model 'cares' about rare-class mistakes.\n\n"
                "0 disables clamping. Higher values allow stronger emphasis on rare classes; lower values prevent a "
                "very rare class from dominating training (often more stable). Example: max=10 turns a weight of 50 into 10."
            ),
        )
        self._add_row(
            basics,
            19,
            "Train sampler",
            train_sampler_combo,
            tooltip=(
                "How training samples are drawn each epoch.\n\n"
                "- shuffle: standard random shuffling.\n"
                "- weighted: oversample rare classes via WeightedRandomSampler (inverse frequency)."
            ),
        )

        def _update_class_weight_state(*_):
            enabled = class_weighting.get().strip().lower() != "none"
            self._set_enabled(class_weight_max_entry, enabled)

        class_weighting.trace_add("write", _update_class_weight_state)
        _update_class_weight_state()


        model_frame = ttk.Labelframe(form.inner, text="Model selection")
        model_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        backbone = tk.StringVar(value="slowfast_tiny")
        sequence_model = tk.StringVar(value="attention")
        use_pretrained = tk.BooleanVar(value=True)
        train_backbone = tk.BooleanVar(value=True)
        use_pose = tk.BooleanVar(value=False)

        backbone_combo = ttk.Combobox(
            model_frame,
            textvariable=backbone,
            state="readonly",
            values=["mobilenet_v3_small", "efficientnet_b0", "vit_b_16", "slowfast_tiny", "slowfast_r50"],
        )
        seq_combo = ttk.Combobox(
            model_frame,
            textvariable=sequence_model,
            state="readonly",
            values=["lstm", "attention"],
        )
        pretrained_check = ttk.Checkbutton(model_frame, variable=use_pretrained)
        train_backbone_check = ttk.Checkbutton(model_frame, variable=train_backbone)
        use_pose_check = ttk.Checkbutton(model_frame, variable=use_pose)
        
        pose_fusion_strategy = tk.StringVar(value="gated_attention")
        pose_strategy_combo = ttk.Combobox(
            model_frame,
            textvariable=pose_fusion_strategy,
            state="readonly",
            values=["gated_attention", "cross_modal_transformer"],
        )

        self._add_row(
            model_frame,
            0,
            "Backbone",
            backbone_combo,
            tooltip=(
                "Frame/video encoder.\n\n"
                "- mobilenet_v3_small / efficientnet_b0: encode each frame then aggregate over time.\n"
                "- vit_b_16: Vision Transformer frame encoder (heavier; best with 224x224 crops).\n"
                "- slowfast_tiny: Lightweight 3D CNN (trained from scratch).\n"
                "- slowfast_r50: Heavy 3D CNN pre-trained on Kinetics-400 (requires pytorchvideo)."
            ),
        )
        self._add_row(
            model_frame,
            1,
            "Temporal head",
            seq_combo,
            tooltip=(
                "How per-frame embeddings are aggregated over time.\n\n"
                "- lstm: recurrent model; can optionally use attention pooling.\n"
                "- attention: attention pooling across frames (no LSTM).\n\n"
                "Ignored/disabled for SlowFast because SlowFast already encodes temporal dynamics."
            ),
        )
        self._add_row(
            model_frame,
            2,
            "Use pretrained backbone",
            pretrained_check,
            tooltip="Use ImageNet pretrained weights for MobileNet/EfficientNet. Not used for SlowFast.",
        )
        self._add_row(
            model_frame,
            3,
            "Train backbone (unfreeze)",
            train_backbone_check,
            tooltip="If enabled, fine-tunes the backbone. If disabled, trains only head layers.",
        )
        self._add_row(
            model_frame,
            4,
            "Use Pose Fusion",
            use_pose_check,
            tooltip="Fuse YOLO-pose keypoints with frame features. Requires keypoints in the dataset.",
        )
        self._add_row(
            model_frame,
            5,
            "Pose Fusion Strategy",
            pose_strategy_combo,
            tooltip=(
                "Method for combining pose vectors with visual features.\n"
                "- gated_attention: (Default) Use pose to dynamically weight visual features.\n"
                "- cross_modal_transformer: (Deep) Use a Transformer to mix Visual and Pose tokens (Simultaneous Learning)."
            ),
        )

        def _update_pose_state(*_):
            if use_pose.get():
                pose_strategy_combo.state(["!disabled", "readonly"])
            else:
                pose_strategy_combo.state(["disabled", "readonly"])
        
        use_pose.trace_add("write", _update_pose_state)
        _update_pose_state()

        slowfast_note = ttk.Label(
            model_frame,
            text=(
                "Note: SlowFast is a video backbone (3D CNN). Temporal-head settings "
                "(LSTM/attention/attention pooling/SE) are ignored."
            ),
            style="Hint.TLabel",
            wraplength=860,
        )
        slowfast_note.grid(row=6, column=0, columnspan=2, sticky="w", pady=(0, 6))

        dims = ttk.Labelframe(form.inner, text="Clip shape overrides (optional)")
        dims.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        num_frames = tk.StringVar(value="")
        frame_size = tk.StringVar(value="")

        self._add_row(
            dims,
            0,
            "Num frames (override)",
            ttk.Entry(dims, textvariable=num_frames),
            tooltip=(
                "Override the number of frames per clip during training.\n\n"
                "Leave blank to use the manifest's window_size. Only change if preprocessing window_size matches."
            ),
        )
        self._add_row(
            dims,
            1,
            "Frame size (override)",
            ttk.Entry(dims, textvariable=frame_size),
            tooltip=(
                "Override the crop size during training.\n\n"
                "Leave blank to use the manifest's crop_size. Only change if preprocessing crop_size matches."
            ),
        )

        temporal = ttk.Labelframe(form.inner, text="Temporal options (LSTM / attention)")
        temporal.grid(row=5, column=0, sticky="ew", pady=(0, 10))

        hidden_dim = tk.StringVar(value="256")
        num_layers = tk.StringVar(value="1")
        bidirectional = tk.BooleanVar(value=False)
        temporal_attention_layers = tk.StringVar(value="0")
        positional_encoding = tk.StringVar(value="none")
        positional_encoding_max_len = tk.StringVar(value="512")
        attention_heads = tk.StringVar(value="4")
        attention_pool = tk.BooleanVar(value=True)
        feature_se = tk.BooleanVar(value=True)

        hidden_entry = ttk.Entry(temporal, textvariable=hidden_dim)
        layers_entry = ttk.Entry(temporal, textvariable=num_layers)
        bidi_check = ttk.Checkbutton(temporal, variable=bidirectional)
        ta_entry = ttk.Entry(temporal, textvariable=temporal_attention_layers)
        posenc_combo = ttk.Combobox(
            temporal,
            textvariable=positional_encoding,
            state="readonly",
            values=["none", "sinusoidal", "learned"],
        )
        posenc_len_entry = ttk.Entry(
            temporal, textvariable=positional_encoding_max_len
        )
        heads_combo = ttk.Combobox(
            temporal,
            textvariable=attention_heads,
            state="readonly",
            values=["1", "2", "4", "8", "12", "16"],
        )
        pool_check = ttk.Checkbutton(temporal, variable=attention_pool)
        se_check = ttk.Checkbutton(temporal, variable=feature_se)

        self._add_row(
            temporal,
            0,
            "LSTM hidden dim",
            hidden_entry,
            tooltip="Size of the LSTM hidden state (only used when Temporal head = lstm). Bigger = more capacity and compute.",
        )
        self._add_row(temporal, 1, "LSTM layers", layers_entry, tooltip="Number of stacked LSTM layers (only used when Temporal head = lstm).")
        self._add_row(temporal, 2, "Bidirectional LSTM", bidi_check, tooltip="If enabled, uses future context (not causal).")
        self._add_row(
            temporal,
            3,
            "Temporal self-attn layers",
            ta_entry,
            tooltip=(
                "Number of temporal self-attention blocks applied BEFORE the temporal head.\n\n"
                "If 0: no self-attention blocks are used.\n"
                "Note: 0 does NOT disable attention pooling. If Attention pooling is enabled (or Temporal head = attention), "
                "the model still uses multi-head attention to pool over timesteps."
            ),
        )
        self._add_row(
            temporal,
            4,
            "Positional encoding",
            posenc_combo,
            tooltip=(
                "Adds timestep positional information to temporal attention.\n\n"
                "Used by temporal self-attn layers and attention pooling to make attention more order-aware."
            ),
        )
        self._add_row(
            temporal,
            5,
            "Positional max len",
            posenc_len_entry,
            tooltip=(
                "Only used for learned positional embeddings.\n\n"
                "Must be >= the training clip length (num_frames/window_size)."
            ),
        )
        self._add_row(
            temporal,
            6,
            "Attention heads",
            heads_combo,
            tooltip=(
                "Number of heads inside the temporal Multi-Head Attention modules.\n\n"
                "Used by:\n"
                "- Temporal self-attn layers (if Temporal self-attn layers > 0)\n"
                "- Attention pooling (if enabled, or if Temporal head = attention)\n\n"
                "This is not 'multiple temporal classifiers' - it is one attention module split into multiple heads.\n"
                "Embedding dim must be divisible by attention_heads."
            ),
        )
        self._add_row(
            temporal,
            7,
            "Attention pooling",
            pool_check,
            tooltip=(
                "For LSTM: if enabled, pools all timesteps with attention instead of using the last timestep.\n\n"
                "For Temporal head = attention, attention pooling is always used."
            ),
        )
        self._add_row(
            temporal,
            8,
            "Feature Squeeze-Excitation",
            se_check,
            tooltip="Channel-wise gating on per-frame features (frame backbones only). Can improve robustness.",
        )

        slowfast = ttk.Labelframe(form.inner, text="SlowFast options (SlowFast backbones only)")
        slowfast.grid(row=6, column=0, sticky="ew", pady=(0, 10))

        slowfast_preset = tk.StringVar(value="tiny (48, 0.25)")
        slowfast_alpha = tk.StringVar(value="4")
        slowfast_fusion_ratio = tk.StringVar(value="0.25")
        slowfast_base_channels = tk.StringVar(value="48")

        preset_combo = ttk.Combobox(
            slowfast,
            textvariable=slowfast_preset,
            state="readonly",
            values=[
                "tiny (48, 0.25)",
                "small (64, 0.25)",
                "wide-slow (64, 0.50)",
                "base (96, 0.50)",
                "custom",
            ],
        )
        alpha_entry = ttk.Entry(slowfast, textvariable=slowfast_alpha)
        fusion_entry = ttk.Entry(slowfast, textvariable=slowfast_fusion_ratio)
        base_entry = ttk.Entry(slowfast, textvariable=slowfast_base_channels)

        self._add_row(
            slowfast,
            0,
            "Preset",
            preset_combo,
            tooltip=(
                "Convenience presets that set base_channels and fusion_ratio.\n\n"
                "Tip: For base_channels=48, fusion_ratio=0.25 hits the minimum slow-path width clamp (16 channels). "
                "Use a higher fusion_ratio (e.g., 0.4–0.5) to widen the slow pathway."
            ),
        )
        self._add_row(
            slowfast,
            1,
            "Alpha",
            alpha_entry,
            tooltip="Temporal stride between fast and slow pathways. Example alpha=4 means the slow path samples every 4th frame.",
        )
        self._add_row(
            slowfast,
            2,
            "Fusion ratio",
            fusion_entry,
            tooltip=(
                "Scales the slow-path channel width relative to base_channels.\n\n"
                "Implementation detail: slow_channels = max(int(base_channels * fusion_ratio), 16).\n"
                "So small values may be clamped to the same minimum width.\n\n"
                "Higher fusion_ratio generally increases slow-path capacity and compute."
            ),
        )
        self._add_row(
            slowfast,
            3,
            "Base channels",
            base_entry,
            tooltip="Base channel width for the SlowFast backbone. Larger = more capacity and GPU memory use.",
        )

        def _apply_slowfast_preset(*_):
            preset = slowfast_preset.get()
            if preset == "tiny (48, 0.25)":
                slowfast_base_channels.set("48")
                slowfast_fusion_ratio.set("0.25")
            elif preset == "small (64, 0.25)":
                slowfast_base_channels.set("64")
                slowfast_fusion_ratio.set("0.25")
            elif preset == "wide-slow (64, 0.50)":
                slowfast_base_channels.set("64")
                slowfast_fusion_ratio.set("0.50")
            elif preset == "base (96, 0.50)":
                slowfast_base_channels.set("96")
                slowfast_fusion_ratio.set("0.50")

        slowfast_preset.trace_add("write", _apply_slowfast_preset)

        BACKBONE_FRAME_DIMS = {
            "mobilenet_v3_small": 576,
            "efficientnet_b0": 1280,
            "vit_b_16": 768,
        }

        def _update_training_states(*_):
            bb = backbone.get()
            is_slowfast = bb.startswith("slowfast")
            is_r50 = bb == "slowfast_r50"

            if is_slowfast:
                slowfast_note.grid()
                # Update note text depending on model variant
                if is_r50:
                    slowfast_note.configure(text=(
                        "Note: SlowFast R50 is a fixed architecture. "
                        "Base channels and fusion ratio are ignored. Alpha is used."
                    ))
                else:
                    slowfast_note.configure(text=(
                        "Note: SlowFast is a video backbone (3D CNN). Temporal-head settings "
                        "(LSTM/attention/attention pooling/SE) are ignored."
                    ))
            else:
                slowfast_note.grid_remove()

            self._set_enabled(seq_combo, not is_slowfast)
            self._set_enabled(pretrained_check, not is_slowfast)

            # SlowFast-only options
            # Alpha is used by both tiny and r50
            self._set_enabled(alpha_entry, is_slowfast)
            
            # These determine architecture structure, so they are fixed for r50
            for w in [preset_combo, fusion_entry, base_entry]:
                self._set_enabled(w, is_slowfast and not is_r50)

            # Frame-backbone temporal options
            for w in [
                hidden_entry,
                layers_entry,
                bidi_check,
                ta_entry,
                posenc_combo,
                posenc_len_entry,
                heads_combo,
                pool_check,
                se_check,
            ]:
                self._set_enabled(w, not is_slowfast)

            if is_slowfast:
                return

            is_lstm = sequence_model.get() == "lstm"
            self._set_enabled(hidden_entry, is_lstm)
            self._set_enabled(layers_entry, is_lstm)
            self._set_enabled(bidi_check, is_lstm)
            self._set_enabled(pool_check, is_lstm)
            if not is_lstm:
                attention_pool.set(True)
                self._set_enabled(pool_check, False)

            try:
                tal = int(temporal_attention_layers.get().strip() or "0")
            except ValueError:
                tal = 0
            heads_needed = tal > 0 or attention_pool.get() or (sequence_model.get() == "attention")
            self._set_enabled(heads_combo, heads_needed)
            self._set_enabled(
                posenc_len_entry,
                positional_encoding.get().strip().lower() == "learned",
            )

        backbone.trace_add("write", _update_training_states)
        sequence_model.trace_add("write", _update_training_states)        
        temporal_attention_layers.trace_add("write", _update_training_states)
        attention_pool.trace_add("write", _update_training_states)        
        positional_encoding.trace_add("write", _update_training_states)
        _update_training_states()

        quant_frame = ttk.Labelframe(form.inner, text="Export / Quantization")
        quant_frame.grid(row=7, column=0, sticky="ew", pady=(0, 10))

        quant_model_path = tk.StringVar(value="")
        quant_out_path = tk.StringVar(value="")

        self._add_path_row(
            quant_frame,
            0,
            "Trained model (.pt)",
            quant_model_path,
            "open_file",
            tooltip="Select a trained BehaviorScope .pt model to quantize.",
            filetypes=[("PyTorch Checkpoint", "*.pt"), ("All files", "*.*")],
        )
        self._add_path_row(
            quant_frame,
            1,
            "Output path (optional)",
            quant_out_path,
            "save_file",
            tooltip="Path for the quantized model. Defaults to <input>_quantized.pt.",
            filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")],
            save_ext=".pt",
        )

        def _run_quantization():
            try:
                mp = Path(quant_model_path.get().strip())
                if not mp.exists():
                    raise ValueError("Select a valid model file to quantize.")
                
                cmd = [sys.executable, str(APP_DIR / "quantization_script.py"), "--model_path", str(mp)]
                if quant_out_path.get().strip():
                    cmd += ["--output_path", quant_out_path.get().strip()]
                
                self._run_process(cmd, log, run_btn, stop_btn)
            except Exception as exc:
                messagebox.showerror("Quantization Error", str(exc))

        ttk.Button(quant_frame, text="Quantize Model", command=_run_quantization).grid(
            row=2, column=1, sticky="w", pady=(5, 0)
        )

        run_btn = ttk.Button(buttons, text="Run training")
        stop_btn = ttk.Button(buttons, text="Stop", state="disabled", command=self._stop_process)
        clear_btn = ttk.Button(buttons, text="Clear log", command=lambda: self._clear_log(log))
        run_btn.grid(row=0, column=0, sticky="w")
        stop_btn.grid(row=0, column=1, sticky="w", padx=(10, 0))
        clear_btn.grid(row=0, column=2, sticky="w", padx=(10, 0))

        def _run():
            try:
                mp = Path(manifest_path.get().strip())
                if not mp.exists():
                    raise ValueError("Manifest JSON does not exist.")
                out_dir = Path(output_dir.get().strip())
                out_dir.mkdir(parents=True, exist_ok=True)
                if not run_name.get().strip():
                    raise ValueError("Run name cannot be empty.")

                ep = self._parse_int("Epochs", epochs.get())
                bs = self._parse_int("Batch size", batch_size.get())
                lr_val = self._parse_float("Learning rate", lr.get())
                blr_val = self._parse_float("Backbone LR", backbone_lr.get(), allow_blank=True)
                wd_val = self._parse_float("Weight decay", weight_decay.get())
                dr = self._parse_float("Dropout", dropout.get())
                sd = self._parse_int("Seed", seed.get())
                pat = self._parse_int("Patience", patience.get())
                nw = self._parse_int("Num workers", num_workers.get())
                li = self._parse_int("Log interval", log_interval.get())
                cw_scheme = class_weighting.get().strip() or "none"
                cw_max = self._parse_float("Class weight clamp (max)", class_weight_max.get())
                if cw_max is None or cw_max < 0:
                    raise ValueError("Class weight clamp (max) must be >= 0.")
                cmi = None
                if confusion_matrix.get():
                    cmi = self._parse_int(
                        "Confusion matrix interval", confusion_matrix_interval.get()
                    )
                    if cmi is None or cmi <= 0:
                        raise ValueError("Confusion matrix interval must be >= 1.")

                nf = self._parse_int("Num frames", num_frames.get(), allow_blank=True)
                fs = self._parse_int("Frame size", frame_size.get(), allow_blank=True)

                is_slowfast = backbone.get().startswith("slowfast")

                if is_slowfast:
                    alpha = self._parse_int("SlowFast alpha", slowfast_alpha.get())
                    fr = self._parse_float("SlowFast fusion ratio", slowfast_fusion_ratio.get())
                    bc = self._parse_int("SlowFast base channels", slowfast_base_channels.get())
                else:
                    tal = self._parse_int("Temporal self-attn layers", temporal_attention_layers.get())
                    heads = self._parse_int("Attention heads", attention_heads.get())
                    if heads is None or heads <= 0:
                        raise ValueError("Attention heads must be >= 1.")
                    # ... (rest of validation) ...
                    pass

                cmd = [
                    sys.executable,
                    str(APP_DIR / "train_behavior_lstm.py"),
                    "--manifest_path",
                    str(mp),
                    "--output_dir",
                    str(out_dir),
                    "--run_name",
                    run_name.get().strip(),
                    "--epochs",
                    str(ep),
                    "--batch_size",
                    str(bs),
                    "--device",
                    device.get().strip() or "auto",
                    "--backbone",
                    backbone.get(),
                    "--sequence_model",
                    sequence_model.get(),
                    "--lr",
                    str(lr_val),
                    "--weight_decay",
                    str(wd_val),
                    "--dropout",
                    str(dr),
                    "--num_workers",
                    str(nw),
                    "--patience",
                    str(pat),
                    "--seed",
                    str(sd),
                    "--log_interval",
                    str(li),
                    "--class_weighting",
                    cw_scheme,
                    "--class_weight_max",
                    str(cw_max),
                    "--train_sampler",
                    train_sampler.get().strip() or "shuffle",
                ]
                
                # Check for config file
                if config_path.get().strip():
                    cmd += ["--config", config_path.get().strip()]
                
                if blr_val is not None:
                    cmd += ["--backbone_lr", str(blr_val)]
                
                if scheduler.get() != "none":
                    cmd += ["--scheduler", scheduler.get()]

                cmd += [
                    "--stop_file",
                    str(out_dir / run_name.get().strip() / "stop_training.flag"),
                ]

                if resume_training.get():
                    cmd.append("--resume")

                if nf is not None:
                    cmd += ["--num_frames", str(nf)]
                if fs is not None:
                    cmd += ["--frame_size", str(fs)]

                if disable_augment.get():
                    cmd.append("--disable_augment")
                if per_class_metrics.get():
                    cmd.append("--per_class_metrics")
                if confusion_matrix.get():
                    cmd.append("--confusion_matrix")
                    if cmi is not None:
                        cmd += ["--confusion_matrix_interval", str(cmi)]
                if not use_pretrained.get():
                    cmd.append("--no_pretrained_backbone")
                if train_backbone.get():
                    cmd.append("--train_backbone")

                if is_slowfast:
                    cmd += [
                        "--slowfast_alpha",
                        str(alpha),
                        "--slowfast_fusion_ratio",
                        str(fr),
                        "--slowfast_base_channels",
                        str(bc),
                    ]
                else:
                    hd = self._parse_int("LSTM hidden dim", hidden_dim.get(), allow_blank=(sequence_model.get() != "lstm"))
                    nl = self._parse_int("LSTM layers", num_layers.get(), allow_blank=(sequence_model.get() != "lstm"))
                    tal = self._parse_int("Temporal self-attn layers", temporal_attention_layers.get())
                    heads = self._parse_int("Attention heads", attention_heads.get())
                    
                    # Fix: Define posenc here
                    posenc = positional_encoding.get().strip().lower()
                    pos_max_len = self._parse_int(
                        "Positional max len", 
                        positional_encoding_max_len.get(), 
                        allow_blank=(posenc != "learned")
                    )

                    if hd is not None:
                        cmd += ["--hidden_dim", str(hd)]
                    if nl is not None:
                        cmd += ["--num_layers", str(nl)]
                    cmd += ["--temporal_attention_layers", str(tal)]
                    cmd += ["--attention_heads", str(heads)]
                    if posenc != "none":
                        cmd += ["--positional_encoding", str(posenc)]
                        if pos_max_len is not None:
                            cmd += ["--positional_encoding_max_len", str(pos_max_len)]

                    if bidirectional.get():
                        cmd.append("--bidirectional")
                    if attention_pool.get() or sequence_model.get() == "attention":
                        cmd.append("--attention_pool")
                    if feature_se.get():
                        cmd.append("--feature_se")
                    if use_pose.get():
                        cmd.append("--use_pose")
                        cmd += ["--pose_fusion_strategy", pose_fusion_strategy.get()]

                self._run_process(cmd, log, run_btn, stop_btn)
            except Exception as exc:
                messagebox.showerror("Training error", str(exc))

        run_btn.configure(command=_run)

        return {
            "frame": tab,
            "log": log,
            "vars": {
                "train.config_path": config_path,
                "train.manifest_path": manifest_path,
                "train.output_dir": output_dir,
                "train.run_name": run_name,
                "train.resume": resume_training,
                "train.epochs": epochs,
                "train.batch_size": batch_size,
                "train.lr": lr,
                "train.backbone_lr": backbone_lr,
                "train.weight_decay": weight_decay,
                "train.dropout": dropout,
                "train.device": device,
                "train.seed": seed,
                "train.patience": patience,
                "train.num_workers": num_workers,
                "train.log_interval": log_interval,
                "train.disable_augment": disable_augment,
                "train.per_class_metrics": per_class_metrics,
                "train.confusion_matrix": confusion_matrix,
                "train.confusion_matrix_interval": confusion_matrix_interval,
                "train.class_weighting": class_weighting,
                "train.class_weight_max": class_weight_max,
                "train.train_sampler": train_sampler,
                "train.scheduler": scheduler,
                "train.backbone": backbone,
                "train.sequence_model": sequence_model,
                "train.use_pretrained": use_pretrained,
                "train.train_backbone": train_backbone,
                "train.use_pose": use_pose,
                "train.pose_fusion_strategy": pose_fusion_strategy,
                "train.num_frames": num_frames,
                "train.frame_size": frame_size,
                "train.hidden_dim": hidden_dim,
                "train.num_layers": num_layers,
                "train.bidirectional": bidirectional,
                "train.temporal_attention_layers": temporal_attention_layers,
                "train.positional_encoding": positional_encoding,
                "train.positional_encoding_max_len": positional_encoding_max_len,
                "train.attention_heads": attention_heads,
                "train.attention_pool": attention_pool,
                "train.feature_se": feature_se,
                "train.slowfast_preset": slowfast_preset,
                "train.slowfast_alpha": slowfast_alpha,
                "train.slowfast_fusion_ratio": slowfast_fusion_ratio,
                "train.slowfast_base_channels": slowfast_base_channels,
            },
        }

    def _build_infer_tab(self):
        tab = ttk.Frame(self.notebook)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        form = ScrollableFrame(tab)
        form.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        buttons = ttk.Frame(tab)
        buttons.grid(row=1, column=0, sticky="ew", padx=10)

        log_frame = ttk.Labelframe(tab, text="Log")
        log_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(8, 10))
        tab.grid_rowconfigure(2, weight=1)

        log = ScrolledText(log_frame, height=10, state="disabled")
        log.pack(fill="both", expand=True)

        header = ttk.Label(
            form.inner,
            text=(
                "Inference: run a trained model on a long video. Produces a merged CSV "
                "timeline (and optional per-window JSON)."
            ),
            style="Header.TLabel",
            wraplength=900,
        )
        header.grid(row=0, column=0, sticky="w", pady=(0, 10))

        paths = ttk.Labelframe(form.inner, text="Paths")
        paths.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        model_path = tk.StringVar(value="")
        video_path = tk.StringVar(value="")
        yolo_weights = tk.StringVar(value="")
        output_csv = tk.StringVar(value="")
        output_json = tk.StringVar(value="")
        output_xlsx = tk.StringVar(value="")

        self._add_path_row(
            paths,
            0,
            "Trained model checkpoint (.pt)",
            model_path,
            "open_file",
            tooltip="Path to best_model.pt produced by training.",
            filetypes=[("Model files", "*.pt;*.pth;*.bin"), ("All files", "*.*")],
        )
        self._add_path_row(
            paths,
            1,
            "Input video",
            video_path,
            "open_file",
            tooltip="Long video to annotate.",
            filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")],
        )
        self._add_path_row(
            paths,
            2,
            "YOLO weights (.pt)",
            yolo_weights,
            "open_file",
            tooltip="YOLO weights used to crop the animal during inference (often the same as preprocessing).",
            filetypes=[("YOLO Weights", "*.pt;*.onnx;*.engine;*.xml;*.h5"), ("All files", "*.*")],
        )

        self._add_path_row(
            paths,
            3,
            "Output CSV (optional)",
            output_csv,
            "save_file",
            tooltip="Merged timeline CSV path. If blank, defaults to <video>.behavior.csv.",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            save_ext=".csv",
        )
        self._add_path_row(
            paths,
            4,
            "Output JSON (optional)",
            output_json,
            "save_file",
            tooltip="Optional per-window prediction JSON path (useful for QA/debugging).",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            save_ext=".json",
        )
        self._add_path_row(
            paths,
            5,
            "Output XLSX (optional)",
            output_xlsx,
            "save_file",
            tooltip="Per-frame metrics workbook path. If blank, defaults to <video>.metrics.xlsx when Export XLSX is enabled.",
            filetypes=[("XLSX", "*.xlsx"), ("All files", "*.*")],
            save_ext=".xlsx",
        )

        options = ttk.Labelframe(form.inner, text="Inference options")
        options.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        window_size = tk.StringVar(value="")
        window_stride = tk.StringVar(value="")
        batch_size = tk.StringVar(value="8")
        prob_threshold = tk.StringVar(value="0.50")
        keep_unknown = tk.BooleanVar(value=False)
        skip_tail = tk.BooleanVar(value=False)
        device = tk.StringVar(value="auto")

        self._add_row(
            options,
            0,
            "Window size (override)",
            ttk.Entry(options, textvariable=window_size),
            tooltip=(
                "Frames per prediction window.\n\n"
                "Leave blank to use the training value stored in the checkpoint."
            ),
        )
        self._add_row(
            options,
            1,
            "Window stride (override)",
            ttk.Entry(options, textvariable=window_stride),
            tooltip="Stride between windows. Leave blank to default to window_size//2.",
        )
        self._add_row(options, 2, "Batch size", ttk.Entry(options, textvariable=batch_size), tooltip="Windows per inference batch.")
        self._add_row(
            options,
            3,
            "Prob threshold",
            ttk.Entry(options, textvariable=prob_threshold),
            tooltip="Minimum predicted probability required to accept a class; otherwise labeled 'Unknown'.",
        )
        self._add_row(options, 4, "Keep unknown", ttk.Checkbutton(options, variable=keep_unknown), tooltip="Include 'Unknown' segments in the merged CSV.")
        self._add_row(options, 5, "Skip tail window", ttk.Checkbutton(options, variable=skip_tail), tooltip="Skip the last partial window instead of forcing evaluation.")
        self._add_row(options, 6, "Device", ttk.Entry(options, textvariable=device), tooltip="Device for inference (auto/cuda:0/cpu).")

        yolo_opts = ttk.Labelframe(form.inner, text="YOLO crop options")
        yolo_opts.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        crop_size = tk.StringVar(value="")
        pad_ratio = tk.StringVar(value="0.20")
        yolo_conf = tk.StringVar(value="0.25")
        yolo_iou = tk.StringVar(value="0.45")
        yolo_imgsz = tk.StringVar(value="640")
        yolo_max_det = tk.StringVar(value="1")
        yolo_nms = tk.BooleanVar(value=True)
        keep_last_box = tk.BooleanVar(value=True)
        yolo_mode = tk.StringVar(value="auto")
        progress_interval = tk.StringVar(value="50")

        self._add_row(
            yolo_opts,
            0,
            "Crop size (override)",
            ttk.Entry(yolo_opts, textvariable=crop_size),
            tooltip="Resize crop size. Leave blank to use training frame_size from the checkpoint.",
        )
        self._add_row(yolo_opts, 1, "Pad ratio", ttk.Entry(yolo_opts, textvariable=pad_ratio), tooltip="Expands YOLO box by this ratio before cropping.")
        self._add_row(yolo_opts, 2, "YOLO confidence", ttk.Entry(yolo_opts, textvariable=yolo_conf), tooltip="Minimum YOLO confidence for a detection.")
        self._add_row(yolo_opts, 3, "YOLO IoU (NMS)", ttk.Entry(yolo_opts, textvariable=yolo_iou), tooltip="Non-maximum suppression IoU threshold for YOLO.")
        self._add_row(yolo_opts, 4, "YOLO imgsz", ttk.Entry(yolo_opts, textvariable=yolo_imgsz), tooltip="YOLO inference resolution.")
        max_det_entry = ttk.Entry(
            yolo_opts, textvariable=yolo_max_det, state="disabled"
        )
        self._add_row(
            yolo_opts,
            5,
            "YOLO max_det",
            max_det_entry,
            tooltip="Locked to 1 for single-animal experiments.",
        )
        nms_check = ttk.Checkbutton(yolo_opts, variable=yolo_nms, state="disabled")
        self._add_row(
            yolo_opts,
            6,
            "YOLO NMS",
            nms_check,
            tooltip="Locked ON for single-animal experiments.",
        )
        self._add_row(yolo_opts, 7, "Keep last box", ttk.Checkbutton(yolo_opts, variable=keep_last_box), tooltip="Reuse the last box when YOLO misses a frame.")
        self._add_row(
            yolo_opts,
            8,
            "Progress interval",
            ttk.Entry(yolo_opts, textvariable=progress_interval),
            tooltip="Print progress every N processed windows (0 disables).",
        )
        yolo_mode_combo = ttk.Combobox(
            yolo_opts,
            textvariable=yolo_mode,
            values=("auto", "bbox", "pose"),
            state="readonly",
        )
        self._add_row(
            yolo_opts,
            9,
            "YOLO mode",
            yolo_mode_combo,
            tooltip=(
                "auto: use keypoints if present | bbox: ignore keypoints | pose: require keypoints (YOLO-pose)."
            ),
        )

        metrics_opts = ttk.Labelframe(form.inner, text="Per-frame metrics (XLSX)")
        metrics_opts.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        export_xlsx = tk.BooleanVar(value=True)
        per_frame_mode = tk.StringVar(value="true")
        top_k = tk.StringVar(value="3")
        smoothing = tk.StringVar(value="none")
        ema_alpha = tk.StringVar(value="0.2")
        interp_max_gap = tk.StringVar(value="5")

        self._add_row(
            metrics_opts,
            0,
            "Export XLSX metrics",
            ttk.Checkbutton(metrics_opts, variable=export_xlsx),
            tooltip="If enabled, writes the per-frame metrics workbook to Output XLSX (or defaults to <video>.metrics.xlsx).",
        )
        per_frame_combo = ttk.Combobox(
            metrics_opts,
            textvariable=per_frame_mode,
            values=("none", "simple", "true"),
            state="readonly",
        )
        self._add_row(
            metrics_opts,
            1,
            "Per-frame mode",
            per_frame_combo,
            tooltip="none: no per-frame label mapping | simple: nearest window center | true: average overlapping window probs.",
        )
        self._add_row(
            metrics_opts,
            2,
            "Top-K (1-5)",
            ttk.Entry(metrics_opts, textvariable=top_k),
            tooltip="Number of behavior probabilities to export (capped at 5).",
        )
        smoothing_combo = ttk.Combobox(
            metrics_opts,
            textvariable=smoothing,
            values=("none", "ema", "kalman", "interpolation"),
            state="readonly",
        )
        self._add_row(
            metrics_opts,
            3,
            "Smoothing",
            smoothing_combo,
            tooltip="Smoothing applied to bbox centers before computing distance/speed/acceleration.",
        )
        self._add_row(
            metrics_opts,
            4,
            "EMA alpha",
            ttk.Entry(metrics_opts, textvariable=ema_alpha),
            tooltip="Used only when smoothing=ema (0-1).",
        )
        self._add_row(
            metrics_opts,
            5,
            "Interp max gap (frames)",
            ttk.Entry(metrics_opts, textvariable=interp_max_gap),
            tooltip="Used only when smoothing=interpolation.",
        )

        review_opts = ttk.Labelframe(form.inner, text="Review video (optional)")
        review_opts.grid(row=5, column=0, sticky="ew", pady=(0, 10))

        export_review_video = tk.BooleanVar(value=False)
        output_review_video = tk.StringVar(value="")
        draw_bbox = tk.BooleanVar(value=True)
        font_scale = tk.StringVar(value="0.7")
        thickness = tk.StringVar(value="2")
        label_bg_alpha = tk.StringVar(value="1.0")

        self._add_row(
            review_opts,
            0,
            "Export review video",
            ttk.Checkbutton(review_opts, variable=export_review_video),
            tooltip="If enabled, exports an annotated MP4 for visual QA.",
        )
        self._add_path_row(
            review_opts,
            1,
            "Output review video (optional)",
            output_review_video,
            "save_file",
            tooltip="Annotated MP4 path. If blank, defaults to <video>.review.mp4.",
            filetypes=[("MP4", "*.mp4"), ("All files", "*.*")],
            save_ext=".mp4",
        )
        self._add_row(
            review_opts,
            2,
            "Draw bbox overlay",
            ttk.Checkbutton(review_opts, variable=draw_bbox),
            tooltip="Draw YOLO bbox/keypoints on the review video. Disable for label-only overlay.",
        )
        self._add_row(
            review_opts,
            3,
            "Font scale",
            ttk.Entry(review_opts, textvariable=font_scale),
            tooltip="OpenCV font scale for review overlays (larger = bigger text).",
        )
        self._add_row(
            review_opts,
            4,
            "Thickness",
            ttk.Entry(review_opts, textvariable=thickness),
            tooltip="Text/border thickness for review overlays.",
        )
        self._add_row(
            review_opts,
            5,
            "Label bg alpha",
            ttk.Entry(review_opts, textvariable=label_bg_alpha),
            tooltip="Opacity (0-1) for label background rectangles (lower reduces occlusion).",
        )

        def _suggest_outputs():
            try:
                vp = Path(video_path.get().strip())
                if not vp.exists():
                    return
                if not output_csv.get().strip():
                    output_csv.set(str(vp.with_suffix(".behavior.csv")))
                if not output_json.get().strip():
                    output_json.set(str(vp.with_suffix(".behavior.windows.json")))
                if not output_xlsx.get().strip():
                    output_xlsx.set(str(vp.with_suffix(".metrics.xlsx")))
                if export_review_video.get() and not output_review_video.get().strip():
                    output_review_video.set(str(vp.with_suffix(".review.mp4")))
            except Exception:
                return

        ttk.Button(form.inner, text="Suggest output paths", command=_suggest_outputs).grid(
            row=6, column=0, sticky="w", pady=(0, 10)
        )

        run_btn = ttk.Button(buttons, text="Run inference")
        stop_btn = ttk.Button(buttons, text="Stop", state="disabled", command=self._stop_process)
        clear_btn = ttk.Button(buttons, text="Clear log", command=lambda: self._clear_log(log))
        run_btn.grid(row=0, column=0, sticky="w")
        stop_btn.grid(row=0, column=1, sticky="w", padx=(10, 0))
        clear_btn.grid(row=0, column=2, sticky="w", padx=(10, 0))

        def _run():
            try:
                mp = Path(model_path.get().strip())
                vp = Path(video_path.get().strip())
                yw = Path(yolo_weights.get().strip())
                if not mp.exists():
                    raise ValueError("Model checkpoint does not exist.")
                if not vp.exists():
                    raise ValueError("Input video does not exist.")
                if not yw.exists():
                    raise ValueError("YOLO weights file does not exist.")

                bs = self._parse_int("Batch size", batch_size.get())
                pt = self._parse_float("Prob threshold", prob_threshold.get())
                pr = self._parse_float("Pad ratio", pad_ratio.get())      
                yc = self._parse_float("YOLO confidence", yolo_conf.get())
                yi = self._parse_float("YOLO IoU", yolo_iou.get())        
                ys = self._parse_int("YOLO imgsz", yolo_imgsz.get())      
                pi = self._parse_int("Progress interval", progress_interval.get())
                tk_val = self._parse_int("Top-K", top_k.get())
                tk_val = max(1, min(5, int(tk_val)))
                ea = self._parse_float("EMA alpha", ema_alpha.get())
                ig = self._parse_int("Interp max gap (frames)", interp_max_gap.get())

                ws = self._parse_int("Window size", window_size.get(), allow_blank=True)
                st = self._parse_int("Window stride", window_stride.get(), allow_blank=True)
                cs = self._parse_int("Crop size", crop_size.get(), allow_blank=True)

                cmd = [
                    sys.executable,
                    str(APP_DIR / "infer_behavior_lstm.py"),
                    "--model_path",
                    str(mp),
                    "--video_path",
                    str(vp),
                    "--yolo_weights",
                    str(yw),
                    "--yolo_mode",
                    yolo_mode.get().strip() or "auto",
                    "--batch_size",
                    str(bs),
                    "--prob_threshold",
                    str(pt),
                    "--device",
                    device.get().strip() or "auto",
                    "--pad_ratio",
                    str(pr),
                    "--yolo_conf",
                    str(yc),
                    "--yolo_iou",
                    str(yi),
                    "--yolo_imgsz",
                    str(ys),
                    "--progress_interval",
                    str(pi),
                ]

                if output_csv.get().strip():
                    cmd += ["--output_csv", output_csv.get().strip()]
                if output_json.get().strip():
                    cmd += ["--output_json", output_json.get().strip()]
                if ws is not None:
                    cmd += ["--window_size", str(ws)]
                if st is not None:
                    cmd += ["--window_stride", str(st)]
                if cs is not None:
                    cmd += ["--crop_size", str(cs)]
                if keep_unknown.get():
                    cmd.append("--keep_unknown")
                if skip_tail.get():
                    cmd.append("--skip_tail")
                if keep_last_box.get():
                    cmd.append("--keep_last_box")

                if export_xlsx.get():
                    xlsx_path = output_xlsx.get().strip()
                    if not xlsx_path:
                        xlsx_path = str(vp.with_suffix(".metrics.xlsx"))
                        output_xlsx.set(xlsx_path)
                    cmd += [
                        "--output_xlsx",
                        xlsx_path,
                        "--per_frame_mode",
                        per_frame_mode.get().strip(),
                        "--top_k",
                        str(tk_val),
                        "--smoothing",
                        smoothing.get().strip(),
                        "--ema_alpha",
                        str(ea),
                        "--interp_max_gap",
                        str(ig),
                    ]

                if export_review_video.get():
                    fs = self._parse_float("Font scale", font_scale.get())
                    th = self._parse_int("Thickness", thickness.get())
                    bg = self._parse_float("Label bg alpha", label_bg_alpha.get())
                    if bg < 0.0 or bg > 1.0:
                        raise ValueError("Label bg alpha must be between 0 and 1.")
                    out_vid = output_review_video.get().strip()
                    if not out_vid:
                        out_vid = str(vp.with_suffix(".review.mp4"))
                        output_review_video.set(out_vid)
                    cmd += [
                        "--output_review_video",
                        out_vid,
                        "--font_scale",
                        str(fs),
                        "--thickness",
                        str(th),
                        "--label_bg_alpha",
                        str(bg),
                    ]
                    if not draw_bbox.get():
                        cmd.append("--no_draw_bbox")

                self._run_process(cmd, log, run_btn, stop_btn)
            except Exception as exc:
                messagebox.showerror("Inference error", str(exc))

        run_btn.configure(command=_run)

        return {
            "frame": tab,
            "log": log,
            "vars": {
                "infer.model_path": model_path,
                "infer.video_path": video_path,
                "infer.yolo_weights": yolo_weights,
                "infer.output_csv": output_csv,
                "infer.output_json": output_json,
                "infer.output_xlsx": output_xlsx,
                "infer.window_size": window_size,
                "infer.window_stride": window_stride,
                "infer.batch_size": batch_size,
                "infer.prob_threshold": prob_threshold,
                "infer.keep_unknown": keep_unknown,
                "infer.skip_tail": skip_tail,
                "infer.device": device,
                "infer.crop_size": crop_size,
                "infer.pad_ratio": pad_ratio,
                "infer.yolo_conf": yolo_conf,
                "infer.yolo_iou": yolo_iou,
                "infer.yolo_imgsz": yolo_imgsz,
                "infer.yolo_max_det": yolo_max_det,
                "infer.yolo_nms": yolo_nms,
                "infer.keep_last_box": keep_last_box,
                "infer.yolo_mode": yolo_mode,
                "infer.progress_interval": progress_interval,
                "infer.export_xlsx": export_xlsx,
                "infer.per_frame_mode": per_frame_mode,
                "infer.top_k": top_k,
                "infer.smoothing": smoothing,
                "infer.ema_alpha": ema_alpha,
                "infer.interp_max_gap": interp_max_gap,
                "infer.export_review_video": export_review_video,
                "infer.output_review_video": output_review_video,
                "infer.draw_bbox": draw_bbox,
                "infer.font_scale": font_scale,
                "infer.thickness": thickness,
                "infer.label_bg_alpha": label_bg_alpha,
            },
        }

    def _build_batch_tab(self):
        tab = ttk.Frame(self.notebook)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        form = ScrollableFrame(tab)
        form.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        buttons = ttk.Frame(tab)
        buttons.grid(row=1, column=0, sticky="ew", padx=10)

        log_frame = ttk.Labelframe(tab, text="Log")
        log_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(8, 10))
        tab.grid_rowconfigure(2, weight=1)

        log = ScrolledText(log_frame, height=10, state="disabled")
        log.pack(fill="both", expand=True)

        header = ttk.Label(
            form.inner,
            text=(
                "Batch inference: run a trained model over a folder of videos. "
                "Outputs are written under Output dir, mirroring the input folder structure."
            ),
            style="Header.TLabel",
            wraplength=900,
        )
        header.grid(row=0, column=0, sticky="w", pady=(0, 10))

        paths = ttk.Labelframe(form.inner, text="Paths")
        paths.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        input_dir = tk.StringVar(value="")
        output_dir = tk.StringVar(value="")
        model_path = tk.StringVar(value="")
        yolo_weights = tk.StringVar(value="")

        self._add_path_row(
            paths,
            0,
            "Input directory (videos)",
            input_dir,
            "dir",
            tooltip="Folder to scan recursively for videos (*.mp4/*.avi/*.mov/*.mkv).",
        )
        self._add_path_row(
            paths,
            1,
            "Output directory",
            output_dir,
            "dir",
            tooltip="Results are written here. Subfolders mirror the input directory structure.",
        )
        self._add_path_row(
            paths,
            2,
            "Trained model checkpoint (.pt)",
            model_path,
            "open_file",
            tooltip="Path to best_model.pt produced by training.",
            filetypes=[("Model files", "*.pt;*.pth;*.bin"), ("All files", "*.*")],
        )
        self._add_path_row(
            paths,
            3,
            "YOLO weights (.pt)",
            yolo_weights,
            "open_file",
            tooltip="YOLO weights used to crop the animal during inference.",
            filetypes=[("YOLO Weights", "*.pt;*.onnx;*.engine;*.xml;*.h5"), ("All files", "*.*")],
        )

        options = ttk.Labelframe(form.inner, text="Inference options")
        options.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        window_size = tk.StringVar(value="")
        window_stride = tk.StringVar(value="")
        batch_size = tk.StringVar(value="8")
        prob_threshold = tk.StringVar(value="0.50")
        skip_tail = tk.BooleanVar(value=False)
        device = tk.StringVar(value="auto")

        self._add_row(
            options,
            0,
            "Window size (override)",
            ttk.Entry(options, textvariable=window_size),
            tooltip="Frames per prediction window. Leave blank to use checkpoint num_frames.",
        )
        self._add_row(
            options,
            1,
            "Window stride (override)",
            ttk.Entry(options, textvariable=window_stride),
            tooltip="Stride in frames. Leave blank to default to window_size//2.",
        )
        self._add_row(
            options,
            2,
            "Batch size",
            ttk.Entry(options, textvariable=batch_size),
            tooltip="Windows per forward pass. Increase until you hit GPU memory limits.",
        )
        self._add_row(
            options,
            3,
            "Prob threshold",
            ttk.Entry(options, textvariable=prob_threshold),
            tooltip="Minimum probability to accept a label; otherwise it's 'Unknown'.",
        )
        self._add_row(
            options,
            4,
            "Skip tail window",
            ttk.Checkbutton(options, variable=skip_tail),
            tooltip="Skip the last partial window instead of forcing an evaluation.",
        )
        self._add_row(
            options,
            5,
            "Device",
            ttk.Entry(options, textvariable=device),
            tooltip="Device for inference (auto/cuda:0/cpu).",
        )

        yolo_opts = ttk.Labelframe(form.inner, text="YOLO crop options")
        yolo_opts.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        crop_size = tk.StringVar(value="")
        pad_ratio = tk.StringVar(value="0.2")
        yolo_conf = tk.StringVar(value="0.25")
        yolo_iou = tk.StringVar(value="0.45")
        yolo_imgsz = tk.StringVar(value="640")
        keep_last_box = tk.BooleanVar(value=True)
        yolo_mode = tk.StringVar(value="auto")

        self._add_row(
            yolo_opts,
            0,
            "Crop size (override)",
            ttk.Entry(yolo_opts, textvariable=crop_size),
            tooltip="Crop size used by the classifier. Leave blank to use checkpoint frame_size.",
        )
        self._add_row(
            yolo_opts,
            1,
            "Pad ratio",
            ttk.Entry(yolo_opts, textvariable=pad_ratio),
            tooltip="Expands YOLO box by this ratio before cropping.",
        )
        self._add_row(
            yolo_opts,
            2,
            "YOLO confidence",
            ttk.Entry(yolo_opts, textvariable=yolo_conf),
            tooltip="Minimum YOLO confidence for a detection.",
        )
        self._add_row(
            yolo_opts,
            3,
            "YOLO IoU (NMS)",
            ttk.Entry(yolo_opts, textvariable=yolo_iou),
            tooltip="Non-maximum suppression IoU threshold for YOLO.",
        )
        self._add_row(
            yolo_opts,
            4,
            "YOLO imgsz",
            ttk.Entry(yolo_opts, textvariable=yolo_imgsz),
            tooltip="YOLO inference resolution.",
        )
        self._add_row(
            yolo_opts,
            5,
            "Keep last box",
            ttk.Checkbutton(yolo_opts, variable=keep_last_box),
            tooltip="Reuse the last box when YOLO misses a frame (recommended).",
        )
        yolo_mode_combo = ttk.Combobox(
            yolo_opts,
            textvariable=yolo_mode,
            values=("auto", "bbox", "pose"),
            state="readonly",
        )
        self._add_row(
            yolo_opts,
            6,
            "YOLO mode",
            yolo_mode_combo,
            tooltip=(
                "auto: use keypoints if present | bbox: ignore keypoints | pose: require keypoints (YOLO-pose)."
            ),
        )

        outputs = ttk.Labelframe(form.inner, text="Outputs")
        outputs.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        drop_unknown = tk.BooleanVar(value=False)
        skip_json = tk.BooleanVar(value=False)
        skip_existing = tk.BooleanVar(value=False)
        skip_review_video = tk.BooleanVar(value=False)
        export_xlsx = tk.BooleanVar(value=True)
        per_frame_mode = tk.StringVar(value="true")
        top_k = tk.StringVar(value="3")
        smoothing = tk.StringVar(value="none")
        ema_alpha = tk.StringVar(value="0.2")
        interp_max_gap = tk.StringVar(value="5")
        draw_bbox = tk.BooleanVar(value=True)
        font_scale = tk.StringVar(value="0.7")
        thickness = tk.StringVar(value="2")
        label_bg_alpha = tk.StringVar(value="1.0")

        self._add_row(
            outputs,
            0,
            "Drop unknown segments",
            ttk.Checkbutton(outputs, variable=drop_unknown),
            tooltip="If enabled, omits 'Unknown' segments from the merged CSV.",
        )
        self._add_row(
            outputs,
            1,
            "Skip per-window JSON",
            ttk.Checkbutton(outputs, variable=skip_json),
            tooltip="If enabled, does not write per-window JSON files.",
        )
        self._add_row(
            outputs,
            2,
            "Skip existing outputs",
            ttk.Checkbutton(outputs, variable=skip_existing),
            tooltip="If enabled, skips videos whose outputs already exist.",
        )
        self._add_row(
            outputs,
            3,
            "Skip review video export",
            ttk.Checkbutton(outputs, variable=skip_review_video),
            tooltip="If enabled, does not export annotated MP4 review videos.",
        )
        self._add_row(
            outputs,
            4,
            "Export XLSX metrics",
            ttk.Checkbutton(outputs, variable=export_xlsx),
            tooltip="If enabled, exports a per-video metrics workbook (.metrics.xlsx).",
        )
        per_frame_combo = ttk.Combobox(
            outputs,
            textvariable=per_frame_mode,
            values=("none", "simple", "true"),
            state="readonly",
        )
        self._add_row(
            outputs,
            5,
            "Per-frame mode",
            per_frame_combo,
            tooltip="none: no per-frame label mapping | simple: nearest window center | true: average overlapping window probs.",
        )
        self._add_row(
            outputs,
            6,
            "Top-K (1-5)",
            ttk.Entry(outputs, textvariable=top_k),
            tooltip="Number of behavior probabilities to export (capped at 5).",
        )
        smoothing_combo = ttk.Combobox(
            outputs,
            textvariable=smoothing,
            values=("none", "ema", "kalman", "interpolation"),
            state="readonly",
        )
        self._add_row(
            outputs,
            7,
            "Smoothing",
            smoothing_combo,
            tooltip="Smoothing applied to bbox centers before computing distance/speed/acceleration (XLSX).",
        )
        self._add_row(
            outputs,
            8,
            "EMA alpha",
            ttk.Entry(outputs, textvariable=ema_alpha),
            tooltip="Used only when smoothing=ema (0-1).",
        )
        self._add_row(
            outputs,
            9,
            "Interp max gap (frames)",
            ttk.Entry(outputs, textvariable=interp_max_gap),
            tooltip="Used only when smoothing=interpolation.",
        )
        self._add_row(
            outputs,
            10,
            "Draw bbox overlay",
            ttk.Checkbutton(outputs, variable=draw_bbox),
            tooltip="Draw YOLO bbox/keypoints on exported review videos. Disable for label-only overlay.",
        )
        self._add_row(
            outputs,
            11,
            "Font scale",
            ttk.Entry(outputs, textvariable=font_scale),
            tooltip="OpenCV font scale for review overlays (larger = bigger text).",
        )
        self._add_row(
            outputs,
            12,
            "Thickness",
            ttk.Entry(outputs, textvariable=thickness),
            tooltip="Text/border thickness for review overlays.",
        )
        self._add_row(
            outputs,
            13,
            "Label bg alpha",
            ttk.Entry(outputs, textvariable=label_bg_alpha),
            tooltip="Opacity (0-1) for label background rectangles in review videos (lower reduces occlusion).",
        )

        run_btn = ttk.Button(buttons, text="Run batch inference")
        stop_btn = ttk.Button(
            buttons, text="Stop", state="disabled", command=self._stop_process
        )
        clear_btn = ttk.Button(buttons, text="Clear log", command=lambda: self._clear_log(log))
        run_btn.grid(row=0, column=0, sticky="w")
        stop_btn.grid(row=0, column=1, sticky="w", padx=(10, 0))
        clear_btn.grid(row=0, column=2, sticky="w", padx=(10, 0))

        def _run():
            try:
                inp = Path(input_dir.get().strip())
                out = Path(output_dir.get().strip())
                mp = Path(model_path.get().strip())
                yw = Path(yolo_weights.get().strip())
                if not inp.exists():
                    raise ValueError("Input directory does not exist.")
                if not mp.exists():
                    raise ValueError("Model checkpoint does not exist.")
                if not yw.exists():
                    raise ValueError("YOLO weights do not exist.")
                if not output_dir.get().strip():
                    raise ValueError("Output directory is required.")

                ws = self._parse_int("Window size", window_size.get(), allow_blank=True)
                st = self._parse_int("Window stride", window_stride.get(), allow_blank=True)
                bs = self._parse_int("Batch size", batch_size.get())
                pt = self._parse_float("Prob threshold", prob_threshold.get())
                cs = self._parse_int("Crop size", crop_size.get(), allow_blank=True)
                pr = self._parse_float("Pad ratio", pad_ratio.get())
                yc = self._parse_float("YOLO confidence", yolo_conf.get())
                yi = self._parse_float("YOLO IoU", yolo_iou.get())
                imgsz = self._parse_int("YOLO imgsz", yolo_imgsz.get())
                tk_val = self._parse_int("Top-K", top_k.get())
                tk_val = max(1, min(5, int(tk_val)))
                ea = self._parse_float("EMA alpha", ema_alpha.get())
                ig = self._parse_int("Interp max gap (frames)", interp_max_gap.get())
                bg = None
                fs = None
                th = None
                if not skip_review_video.get():
                    fs = self._parse_float("Font scale", font_scale.get())
                    th = self._parse_int("Thickness", thickness.get())
                    bg = self._parse_float("Label bg alpha", label_bg_alpha.get())
                    if bg < 0.0 or bg > 1.0:
                        raise ValueError("Label bg alpha must be between 0 and 1.")

                cmd = [
                    sys.executable,
                    str(APP_DIR / "batch_process_videos.py"),
                    "--input_dir",
                    str(inp),
                    "--output_dir",
                    str(out),
                    "--model_path",
                    str(mp),
                    "--yolo_weights",
                    str(yw),
                    "--yolo_mode",
                    yolo_mode.get().strip() or "auto",
                    "--device",
                    device.get().strip(),
                    "--batch_size",
                    str(bs),
                    "--prob_threshold",
                    str(pt),
                    "--pad_ratio",
                    str(pr),
                    "--yolo_conf",
                    str(yc),
                    "--yolo_iou",
                    str(yi),
                    "--yolo_imgsz",
                    str(imgsz),
                ]
                if not draw_bbox.get():
                    cmd.append("--no_draw_bbox")
                if not skip_review_video.get():
                    cmd += [
                        "--font_scale",
                        str(fs),
                        "--thickness",
                        str(th),
                        "--label_bg_alpha",
                        str(bg),
                    ]
                if ws is not None:
                    cmd += ["--window_size", str(ws)]
                if st is not None:
                    cmd += ["--window_stride", str(st)]
                if cs is not None:
                    cmd += ["--crop_size", str(cs)]
                if skip_tail.get():
                    cmd.append("--skip_tail")
                if drop_unknown.get():
                    cmd.append("--drop_unknown")
                if skip_json.get():
                    cmd.append("--skip_json")
                if skip_existing.get():
                    cmd.append("--skip_existing")
                if keep_last_box.get():
                    cmd.append("--keep_last_box")
                if skip_review_video.get():
                    cmd.append("--skip_review_video")
                if export_xlsx.get():
                    cmd.append("--export_xlsx")
                if export_xlsx.get() or (not skip_review_video.get()):
                    cmd += [
                        "--per_frame_mode",
                        per_frame_mode.get().strip(),
                        "--top_k",
                        str(tk_val),
                        "--smoothing",
                        smoothing.get().strip(),
                        "--ema_alpha",
                        str(ea),
                        "--interp_max_gap",
                        str(ig),
                    ]

                self._run_process(cmd, log, run_btn, stop_btn)
            except Exception as exc:
                messagebox.showerror("Batch inference error", str(exc))

        run_btn.configure(command=_run)

        return {
            "frame": tab,
            "log": log,
            "vars": {
                "batch.input_dir": input_dir,
                "batch.output_dir": output_dir,
                "batch.model_path": model_path,
                "batch.yolo_weights": yolo_weights,
                "batch.window_size": window_size,
                "batch.window_stride": window_stride,
                "batch.batch_size": batch_size,
                "batch.prob_threshold": prob_threshold,
                "batch.skip_tail": skip_tail,
                "batch.device": device,
                "batch.crop_size": crop_size,
                "batch.pad_ratio": pad_ratio,
                "batch.yolo_conf": yolo_conf,
                "batch.yolo_iou": yolo_iou,
                "batch.yolo_imgsz": yolo_imgsz,
                "batch.keep_last_box": keep_last_box,
                "batch.yolo_mode": yolo_mode,
                "batch.drop_unknown": drop_unknown,
                "batch.skip_json": skip_json,
                "batch.skip_existing": skip_existing,
                "batch.skip_review_video": skip_review_video,
                "batch.export_xlsx": export_xlsx,
                "batch.per_frame_mode": per_frame_mode,
                "batch.top_k": top_k,
                "batch.smoothing": smoothing,
                "batch.ema_alpha": ema_alpha,
                "batch.interp_max_gap": interp_max_gap,
                "batch.draw_bbox": draw_bbox,
                "batch.font_scale": font_scale,
                "batch.thickness": thickness,
                "batch.label_bg_alpha": label_bg_alpha,
            },
        }

    def _build_realtime_tab(self):
        tab = ttk.Frame(self.notebook)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        content = ttk.Frame(tab)
        content.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        tab.grid_rowconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)

        form = ScrollableFrame(content)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        preview = ttk.Labelframe(content, text="Live preview")
        preview.grid(row=0, column=1, sticky="nsew")
        preview.grid_rowconfigure(0, weight=1)
        preview.grid_columnconfigure(0, weight=1)

        image_label = ttk.Label(preview)
        image_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        metrics = ttk.Frame(preview)
        metrics.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        metrics.grid_columnconfigure(0, weight=1)

        pred_var = tk.StringVar(value="Pred: (idle)")
        fps_var = tk.StringVar(value="FPS: -")
        status_var = tk.StringVar(value="Idle.")
        ttk.Label(metrics, textvariable=pred_var).grid(row=0, column=0, sticky="w")
        ttk.Label(metrics, textvariable=fps_var).grid(row=1, column=0, sticky="w")
        ttk.Label(metrics, textvariable=status_var, style="Hint.TLabel").grid(
            row=2, column=0, sticky="w"
        )

        buttons = ttk.Frame(tab)
        buttons.grid(row=1, column=0, sticky="ew", padx=10)

        log_frame = ttk.Labelframe(tab, text="Log")
        log_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(8, 10))
        tab.grid_rowconfigure(2, weight=1)

        log = ScrolledText(log_frame, height=10, state="disabled")
        log.pack(fill="both", expand=True)

        header = ttk.Label(
            form.inner,
            text=(
                "Real-time inference: run the model on a webcam or video file and "
                "see live predictions + FPS."
            ),
            style="Header.TLabel",
            wraplength=900,
        )
        header.grid(row=0, column=0, sticky="w", pady=(0, 10))

        models = ttk.Labelframe(form.inner, text="Models")
        models.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        model_path = tk.StringVar(value="")
        yolo_weights = tk.StringVar(value="")

        self._add_path_row(
            models,
            0,
            "Trained model checkpoint (.pt)",
            model_path,
            "open_file",
            tooltip="Path to best_model.pt produced by training.",
            filetypes=[("Model files", "*.pt;*.pth;*.bin"), ("All files", "*.*")],
        )
        self._add_path_row(
            models,
            1,
            "YOLO weights (.pt)",
            yolo_weights,
            "open_file",
            tooltip="YOLO weights used to crop the animal each frame.",
            filetypes=[("YOLO Weights", "*.pt;*.onnx;*.engine;*.xml;*.h5"), ("All files", "*.*")],
        )

        source = ttk.Labelframe(form.inner, text="Source")
        source.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        source_kind = tk.StringVar(value="webcam")
        camera_index = tk.StringVar(value="0")
        video_path = tk.StringVar(value="")

        source_row = ttk.Frame(source)
        source_row.grid(row=0, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Radiobutton(
            source_row, text="Webcam", value="webcam", variable=source_kind
        ).pack(side="left")
        ttk.Radiobutton(
            source_row, text="Video file", value="video", variable=source_kind
        ).pack(side="left", padx=(10, 0))

        self._add_row(
            source,
            1,
            "Webcam index",
            ttk.Entry(source, textvariable=camera_index),
            tooltip="Camera index (0 is usually the default webcam).",
        )
        self._add_path_row(
            source,
            2,
            "Video path",
            video_path,
            "open_file",
            tooltip="Video file to stream (useful for repeatable benchmarking).",
            filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")],
        )

        opts = ttk.Labelframe(form.inner, text="Inference options")
        opts.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        device = tk.StringVar(value="auto")
        prob_threshold = tk.StringVar(value="0.50")
        window_size = tk.StringVar(value="")
        window_stride = tk.StringVar(value="")

        self._add_row(
            opts,
            0,
            "Device",
            ttk.Entry(opts, textvariable=device),
            tooltip="Device for inference (auto/cuda:0/cpu).",
        )
        self._add_row(
            opts,
            1,
            "Prob threshold",
            ttk.Entry(opts, textvariable=prob_threshold),
            tooltip="Minimum probability to accept a label; otherwise it's 'Unknown'.",
        )
        self._add_row(
            opts,
            2,
            "Window size (override)",
            ttk.Entry(opts, textvariable=window_size),
            tooltip="Frames per prediction window. Leave blank to use checkpoint num_frames.",
        )
        self._add_row(
            opts,
            3,
            "Window stride (override)",
            ttk.Entry(opts, textvariable=window_stride),
            tooltip="Stride in frames. Leave blank to default to window_size//2.",
        )

        yolo_opts = ttk.Labelframe(form.inner, text="YOLO crop options")
        yolo_opts.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        crop_size = tk.StringVar(value="")
        pad_ratio = tk.StringVar(value="0.2")
        yolo_conf = tk.StringVar(value="0.25")
        yolo_iou = tk.StringVar(value="0.45")
        yolo_imgsz = tk.StringVar(value="640")
        keep_last_box = tk.BooleanVar(value=True)
        draw_bbox = tk.BooleanVar(value=True)
        yolo_mode = tk.StringVar(value="auto")

        self._add_row(
            yolo_opts,
            0,
            "Crop size (override)",
            ttk.Entry(yolo_opts, textvariable=crop_size),
            tooltip="Crop size used by the classifier. Leave blank to use checkpoint frame_size.",
        )
        self._add_row(
            yolo_opts,
            1,
            "Pad ratio",
            ttk.Entry(yolo_opts, textvariable=pad_ratio),
            tooltip="Expands YOLO box by this ratio before cropping.",
        )
        self._add_row(
            yolo_opts,
            2,
            "YOLO confidence",
            ttk.Entry(yolo_opts, textvariable=yolo_conf),
            tooltip="Minimum YOLO confidence for a detection.",
        )
        self._add_row(
            yolo_opts,
            3,
            "YOLO IoU (NMS)",
            ttk.Entry(yolo_opts, textvariable=yolo_iou),
            tooltip="Non-maximum suppression IoU threshold for YOLO.",
        )
        self._add_row(
            yolo_opts,
            4,
            "YOLO imgsz",
            ttk.Entry(yolo_opts, textvariable=yolo_imgsz),
            tooltip="YOLO inference resolution.",
        )
        self._add_row(
            yolo_opts,
            5,
            "Keep last box",
            ttk.Checkbutton(yolo_opts, variable=keep_last_box),
            tooltip="Reuse the last box when YOLO misses a frame (recommended).",
        )
        self._add_row(
            yolo_opts,
            6,
            "Draw bbox overlay",
            ttk.Checkbutton(yolo_opts, variable=draw_bbox),
            tooltip="Draw the YOLO bounding box on the preview.",
        )
        yolo_mode_combo = ttk.Combobox(
            yolo_opts,
            textvariable=yolo_mode,
            values=("auto", "bbox", "pose"),
            state="readonly",
        )
        self._add_row(
            yolo_opts,
            7,
            "YOLO mode",
            yolo_mode_combo,
            tooltip=(
                "auto: use keypoints if present | bbox: ignore keypoints | pose: require keypoints (YOLO-pose)."
            ),
        )

        overlay_opts = ttk.Labelframe(form.inner, text="Overlay options")
        overlay_opts.grid(row=5, column=0, sticky="ew", pady=(0, 10))

        overlay_font_scale = tk.StringVar(value="0.7")
        overlay_thickness = tk.StringVar(value="2")
        show_speed = tk.BooleanVar(value=False)
        smoothing = tk.StringVar(value="none")
        ema_alpha = tk.StringVar(value="0.2")
        interp_max_gap = tk.StringVar(value="5")

        self._add_row(
            overlay_opts,
            0,
            "Font scale",
            ttk.Entry(overlay_opts, textvariable=overlay_font_scale),
            tooltip="OpenCV font scale for bbox labels (larger = bigger text).",
        )
        self._add_row(
            overlay_opts,
            1,
            "Line thickness",
            ttk.Entry(overlay_opts, textvariable=overlay_thickness),
            tooltip="Rectangle/text thickness for bbox labels.",
        )
        self._add_row(
            overlay_opts,
            2,
            "Show speed (px/s)",
            ttk.Checkbutton(overlay_opts, variable=show_speed),
            tooltip="Compute speed from smoothed bbox center and show it next to the behavior label.",
        )
        smoothing_combo = ttk.Combobox(
            overlay_opts,
            textvariable=smoothing,
            values=["none", "ema", "kalman", "interpolation"],
            state="readonly",
        )
        self._add_row(
            overlay_opts,
            3,
            "Center smoothing",
            smoothing_combo,
            tooltip="Smoothing method used for center-based speed estimates.",
        )
        self._add_row(
            overlay_opts,
            4,
            "EMA alpha",
            ttk.Entry(overlay_opts, textvariable=ema_alpha),
            tooltip="Used only when smoothing=ema (0-1).",
        )
        self._add_row(
            overlay_opts,
            5,
            "Interp max gap (frames)",
            ttk.Entry(overlay_opts, textvariable=interp_max_gap),
            tooltip="Used only when smoothing=interpolation.",
        )

        start_btn = ttk.Button(buttons, text="Start real-time inference")
        stop_btn = ttk.Button(
            buttons, text="Stop", state="disabled", command=self._stop_realtime
        )
        clear_btn = ttk.Button(buttons, text="Clear log", command=lambda: self._clear_log(log))
        start_btn.grid(row=0, column=0, sticky="w")
        stop_btn.grid(row=0, column=1, sticky="w", padx=(10, 0))
        clear_btn.grid(row=0, column=2, sticky="w", padx=(10, 0))

        error_shown = {"value": False}

        def _poll():
            running = self._realtime
            if running is None:
                return
            drained = 0
            while drained < 50:
                try:
                    msg = running.output_queue.get_nowait()
                except queue.Empty:
                    break
                drained += 1

                msg_type = msg.get("type")
                if msg_type == "status":
                    text = str(msg.get("text", ""))
                    status_var.set(text)
                    if text:
                        self._append_log(log, text + "\n")
                elif msg_type == "error":
                    text = str(msg.get("message", "Unknown error"))
                    status_var.set("Error.")
                    self._append_log(log, "[ERROR] " + text + "\n")
                    if not error_shown["value"]:
                        error_shown["value"] = True
                        messagebox.showerror("Real-time inference error", text)
                    self._stop_realtime()
                elif msg_type == "stopped":
                    text = str(msg.get("text", "Stopped."))
                    status_var.set(text)
                    if text:
                        self._append_log(log, text + "\n")
                elif msg_type == "frame":
                    pred_label = str(msg.get("pred_label", ""))
                    pred_conf = float(msg.get("pred_conf", 0.0))
                    fps = float(msg.get("fps", 0.0))
                    fps_avg = float(msg.get("fps_avg", 0.0))
                    win_per_sec = float(msg.get("win_per_sec", 0.0))

                    pred_var.set(f"Pred: {pred_label} ({pred_conf:.2f})")
                    fps_var.set(
                        f"FPS: {fps:.2f} (avg {fps_avg:.2f}) | win/s {win_per_sec:.2f}"
                    )

                    rgb = msg.get("rgb")
                    if rgb is not None and Image is not None and ImageTk is not None:
                        try:
                            img = Image.fromarray(rgb)
                            w = image_label.winfo_width()
                            h = image_label.winfo_height()
                            if w > 10 and h > 10:
                                img.thumbnail((w, h))
                            photo = ImageTk.PhotoImage(img)
                            self._realtime_photo = photo
                            image_label.configure(image=photo)
                        except Exception:
                            pass

            if running.worker_thread.is_alive():
                self.root.after(60, _poll)
                return

            self._realtime = None
            start_btn.configure(state="normal")
            stop_btn.configure(state="disabled")

        def _start():
            try:
                if Image is None or ImageTk is None:
                    raise RuntimeError(
                        "Pillow is required for the live preview. Install it with: pip install Pillow"
                    )
                if self._running is not None or self._realtime is not None:
                    raise RuntimeError(
                        "Another job is currently running. Stop it before starting real-time inference."
                    )

                mp = Path(model_path.get().strip())
                yw = Path(yolo_weights.get().strip())
                if not mp.exists():
                    raise ValueError("Model checkpoint does not exist.")
                if not yw.exists():
                    raise ValueError("YOLO weights do not exist.")

                sk = source_kind.get().strip()
                cam_idx = None
                vp = None
                if sk == "webcam":
                    cam_idx = self._parse_int("Webcam index", camera_index.get())
                else:
                    vp = Path(video_path.get().strip())
                    if not vp.exists():
                        raise ValueError("Video path does not exist.")

                pt = self._parse_float("Prob threshold", prob_threshold.get())
                ws = self._parse_int("Window size", window_size.get(), allow_blank=True)
                st = self._parse_int(
                    "Window stride", window_stride.get(), allow_blank=True
                )
                cs = self._parse_int("Crop size", crop_size.get(), allow_blank=True)
                pr = self._parse_float("Pad ratio", pad_ratio.get())
                yc = self._parse_float("YOLO confidence", yolo_conf.get())
                yi = self._parse_float("YOLO IoU", yolo_iou.get())
                imgsz = self._parse_int("YOLO imgsz", yolo_imgsz.get())

                ofs = self._parse_float("Overlay font scale", overlay_font_scale.get())
                oth = self._parse_int("Overlay thickness", overlay_thickness.get())
                sm = smoothing.get().strip() or "none"
                ea = self._parse_float("EMA alpha", ema_alpha.get())
                ig = self._parse_int("Interp max gap", interp_max_gap.get())

                cfg = RealtimeConfig(
                    model_path=mp,
                    yolo_weights=yw,
                    yolo_mode=yolo_mode.get().strip() or "auto",
                    yolo_task="auto",
                    source_kind=sk,
                    camera_index=cam_idx,
                    video_path=vp,
                    device=device.get().strip(),
                    prob_threshold=float(pt),
                    window_size=ws,
                    window_stride=st,
                    crop_size=cs,
                    pad_ratio=float(pr),
                    yolo_conf=float(yc),
                    yolo_iou=float(yi),
                    yolo_imgsz=int(imgsz),
                    keep_last_box=bool(keep_last_box.get()),
                    draw_bbox=bool(draw_bbox.get()),
                    overlay_font_scale=float(ofs),
                    overlay_thickness=int(oth),
                    smoothing=str(sm),
                    ema_alpha=float(ea),
                    interp_max_gap=int(ig),
                    show_speed=bool(show_speed.get()),
                )

                status_var.set("Starting...")
                pred_var.set("Pred: (starting)")
                fps_var.set("FPS: -")
                error_shown["value"] = False

                q: "queue.Queue[dict]" = queue.Queue(maxsize=3)
                stop_event = threading.Event()
                t = threading.Thread(
                    target=self._realtime_worker,
                    args=(cfg, q, stop_event),
                    daemon=True,
                )
                self._realtime = RunningRealtime(
                    stop_event=stop_event, output_queue=q, worker_thread=t
                )

                start_btn.configure(state="disabled")
                stop_btn.configure(state="normal")
                self._append_log(log, "\n[Real-time inference started]\n")
                t.start()
                self.root.after(60, _poll)
            except Exception as exc:
                messagebox.showerror("Real-time inference error", str(exc))

        start_btn.configure(command=_start)

        return {
            "frame": tab,
            "log": log,
            "vars": {
                "realtime.model_path": model_path,
                "realtime.yolo_weights": yolo_weights,
                "rt.yolo_mode": yolo_mode,
                "rt.source_kind": source_kind,
                "rt.camera_index": camera_index,
                "rt.video_path": video_path,
                "rt.device": device,
                "rt.prob_threshold": prob_threshold,
                "rt.window_size": window_size,
                "rt.window_stride": window_stride,
                "rt.crop_size": crop_size,
                "rt.pad_ratio": pad_ratio,
                "rt.yolo_conf": yolo_conf,
                "rt.yolo_iou": yolo_iou,
                "rt.yolo_imgsz": yolo_imgsz,
                "rt.keep_last_box": keep_last_box,
                "rt.draw_bbox": draw_bbox,
                "rt.overlay_font_scale": overlay_font_scale,
                "rt.overlay_thickness": overlay_thickness,
                "rt.show_speed": show_speed,
                "rt.smoothing": smoothing,
                "rt.ema_alpha": ema_alpha,
                "rt.interp_max_gap": interp_max_gap,
            },
        }

    def _build_benchmark_tab(self):
        tab = ttk.Frame(self.notebook)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        form = ScrollableFrame(tab)
        form.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        buttons = ttk.Frame(tab)
        buttons.grid(row=1, column=0, sticky="ew", padx=10)

        log_frame = ttk.Labelframe(tab, text="Log")
        log_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(8, 10))
        tab.grid_rowconfigure(2, weight=1)

        log = ScrolledText(log_frame, height=10, state="disabled")
        log.pack(fill="both", expand=True)

        header = ttk.Label(
            form.inner,
            text=(
                "Benchmark: evaluate per-class accuracy on a manifest split (default: val) "
                "and report full pipeline throughput (YOLO + crop + classifier)."
            ),
            style="Header.TLabel",
            wraplength=900,
        )
        header.grid(row=0, column=0, sticky="w", pady=(0, 10))

        paths = ttk.Labelframe(form.inner, text="Paths")
        paths.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        manifest_path = tk.StringVar(value="")
        model_path = tk.StringVar(value="")
        yolo_weights = tk.StringVar(value="")
        output_json = tk.StringVar(value="")

        self._add_path_row(
            paths,
            0,
            "Manifest JSON",
            manifest_path,
            "open_file",
            tooltip="Path to sequence_manifest.json produced by preprocessing.",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        self._add_path_row(
            paths,
            1,
            "Trained model checkpoint (.pt)",
            model_path,
            "open_file",
            tooltip="Path to best_model.pt produced by training.",
            filetypes=[("Model files", "*.pt;*.pth;*.bin"), ("All files", "*.*")],
        )
        self._add_path_row(
            paths,
            2,
            "YOLO weights (.pt)",
            yolo_weights,
            "open_file",
            tooltip="YOLO weights used to crop the animal (should match preprocessing).",
            filetypes=[("YOLO Weights", "*.pt;*.onnx;*.engine;*.xml;*.h5"), ("All files", "*.*")],
        )
        self._add_path_row(
            paths,
            3,
            "Output summary JSON (optional)",
            output_json,
            "save_file",
            tooltip="Optional path to write benchmark results as JSON.",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            save_ext=".json",
        )

        opts = ttk.Labelframe(form.inner, text="Benchmark options")
        opts.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        split = tk.StringVar(value="val")
        batch_size = tk.StringVar(value="8")
        prob_threshold = tk.StringVar(value="0.50")
        device = tk.StringVar(value="auto")
        limit_videos = tk.StringVar(value="")
        limit_windows = tk.StringVar(value="")

        split_combo = ttk.Combobox(opts, textvariable=split, values=("train", "val", "test"))
        self._add_row(
            opts,
            0,
            "Split",
            split_combo,
            tooltip="Manifest split to benchmark (train/val/test).",
        )
        self._add_row(
            opts,
            1,
            "Batch size",
            ttk.Entry(opts, textvariable=batch_size),
            tooltip="Windows per forward pass.",
        )
        self._add_row(
            opts,
            2,
            "Prob threshold",
            ttk.Entry(opts, textvariable=prob_threshold),
            tooltip="Minimum probability to accept a label; otherwise it's 'Unknown' (counts against accuracy).",
        )
        self._add_row(
            opts,
            3,
            "Device",
            ttk.Entry(opts, textvariable=device),
            tooltip="Device (auto/cuda:0/cpu).",
        )
        self._add_row(
            opts,
            4,
            "Limit videos (optional)",
            ttk.Entry(opts, textvariable=limit_videos),
            tooltip="Optional cap on number of unique videos processed.",
        )
        self._add_row(
            opts,
            5,
            "Limit windows (optional)",
            ttk.Entry(opts, textvariable=limit_windows),
            tooltip="Optional cap on number of windows evaluated (across all videos).",
        )

        windowing = ttk.Labelframe(form.inner, text="Windowing (optional overrides)")
        windowing.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        window_size = tk.StringVar(value="")
        window_stride = tk.StringVar(value="")
        crop_size = tk.StringVar(value="")
        include_tail = tk.BooleanVar(value=False)

        self._add_row(
            windowing,
            0,
            "Window size",
            ttk.Entry(windowing, textvariable=window_size),
            tooltip="Leave blank to use manifest meta (or checkpoint).",
        )
        self._add_row(
            windowing,
            1,
            "Window stride",
            ttk.Entry(windowing, textvariable=window_stride),
            tooltip="Leave blank to use manifest meta (or window_size//2).",
        )
        self._add_row(
            windowing,
            2,
            "Crop size",
            ttk.Entry(windowing, textvariable=crop_size),
            tooltip="Leave blank to use checkpoint frame_size.",
        )
        self._add_row(
            windowing,
            3,
            "Include tail window",
            ttk.Checkbutton(windowing, variable=include_tail),
            tooltip="Includes the last partial window (does not match preprocessing).",
        )

        yolo_opts = ttk.Labelframe(form.inner, text="YOLO options")
        yolo_opts.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        pad_ratio = tk.StringVar(value="")
        yolo_conf = tk.StringVar(value="")
        yolo_iou = tk.StringVar(value="")
        yolo_imgsz = tk.StringVar(value="")
        yolo_max_det = tk.StringVar(value="1")
        yolo_nms = tk.BooleanVar(value=True)
        keep_last_box = tk.BooleanVar(value=True)

        self._add_row(
            yolo_opts,
            0,
            "Pad ratio (override)",
            ttk.Entry(yolo_opts, textvariable=pad_ratio),
            tooltip="Leave blank to use manifest meta (or default).",
        )
        self._add_row(
            yolo_opts,
            1,
            "YOLO confidence (override)",
            ttk.Entry(yolo_opts, textvariable=yolo_conf),
            tooltip="Leave blank to use manifest meta (or default).",
        )
        self._add_row(
            yolo_opts,
            2,
            "YOLO IoU (override)",
            ttk.Entry(yolo_opts, textvariable=yolo_iou),
            tooltip="Leave blank to use manifest meta (or default).",
        )
        self._add_row(
            yolo_opts,
            3,
            "YOLO imgsz (override)",
            ttk.Entry(yolo_opts, textvariable=yolo_imgsz),
            tooltip="Leave blank to use manifest meta (or default).",
        )
        max_det_entry = ttk.Entry(
            yolo_opts, textvariable=yolo_max_det, state="disabled"
        )
        self._add_row(
            yolo_opts,
            4,
            "YOLO max_det",
            max_det_entry,
            tooltip="Locked to 1 for single-animal experiments.",
        )
        nms_check = ttk.Checkbutton(yolo_opts, variable=yolo_nms, state="disabled")
        self._add_row(
            yolo_opts,
            5,
            "YOLO NMS",
            nms_check,
            tooltip="Locked ON for single-animal experiments.",
        )
        self._add_row(
            yolo_opts,
            6,
            "Keep last box",
            ttk.Checkbutton(yolo_opts, variable=keep_last_box),
            tooltip="Reuse last box when YOLO misses a frame (recommended).",
        )

        run_btn = ttk.Button(buttons, text="Run benchmark")
        stop_btn = ttk.Button(buttons, text="Stop", state="disabled", command=self._stop_process)
        clear_btn = ttk.Button(buttons, text="Clear log", command=lambda: self._clear_log(log))
        run_btn.grid(row=0, column=0, sticky="w")
        stop_btn.grid(row=0, column=1, sticky="w", padx=(10, 0))
        clear_btn.grid(row=0, column=2, sticky="w", padx=(10, 0))

        def _run():
            try:
                mp = Path(manifest_path.get().strip())
                ckpt = Path(model_path.get().strip())
                yw = Path(yolo_weights.get().strip())
                if not mp.exists():
                    raise ValueError("Manifest file does not exist.")
                if not ckpt.exists():
                    raise ValueError("Model checkpoint does not exist.")
                if not yw.exists():
                    raise ValueError("YOLO weights file does not exist.")

                bs = self._parse_int("Batch size", batch_size.get())
                pt = self._parse_float("Prob threshold", prob_threshold.get())

                ws = self._parse_int("Window size", window_size.get(), allow_blank=True)
                st = self._parse_int("Window stride", window_stride.get(), allow_blank=True)
                cs = self._parse_int("Crop size", crop_size.get(), allow_blank=True)

                pr = self._parse_float("Pad ratio", pad_ratio.get(), allow_blank=True)
                yc = self._parse_float("YOLO confidence", yolo_conf.get(), allow_blank=True)
                yi = self._parse_float("YOLO IoU", yolo_iou.get(), allow_blank=True)
                ys = self._parse_int("YOLO imgsz", yolo_imgsz.get(), allow_blank=True)
                md = self._parse_int("YOLO max_det", yolo_max_det.get(), allow_blank=True) or 1

                lv = self._parse_int("Limit videos", limit_videos.get(), allow_blank=True)
                lw = self._parse_int("Limit windows", limit_windows.get(), allow_blank=True)

                cmd = [
                    sys.executable,
                    str(APP_DIR / "benchmark_pipeline.py"),
                    "--manifest_path",
                    str(mp),
                    "--model_path",
                    str(ckpt),
                    "--yolo_weights",
                    str(yw),
                    "--split",
                    split.get().strip() or "val",
                    "--device",
                    device.get().strip() or "auto",
                    "--batch_size",
                    str(bs),
                    "--prob_threshold",
                    str(pt),
                    "--yolo_max_det",
                    str(md),
                ]
                if ws is not None:
                    cmd += ["--window_size", str(ws)]
                if st is not None:
                    cmd += ["--window_stride", str(st)]
                if cs is not None:
                    cmd += ["--crop_size", str(cs)]
                if pr is not None:
                    cmd += ["--pad_ratio", str(pr)]
                if yc is not None:
                    cmd += ["--yolo_conf", str(yc)]
                if yi is not None:
                    cmd += ["--yolo_iou", str(yi)]
                if ys is not None:
                    cmd += ["--yolo_imgsz", str(ys)]
                if not yolo_nms.get():
                    cmd.append("--no_yolo_nms")
                if not keep_last_box.get():
                    cmd.append("--no_keep_last_box")
                if include_tail.get():
                    cmd.append("--include_tail")
                if lv is not None:
                    cmd += ["--limit_videos", str(lv)]
                if lw is not None:
                    cmd += ["--limit_windows", str(lw)]
                if output_json.get().strip():
                    cmd += ["--output_json", output_json.get().strip()]

                self._run_process(cmd, log, run_btn, stop_btn)
            except Exception as exc:
                messagebox.showerror("Benchmark error", str(exc))

        run_btn.configure(command=_run)

        return {
            "frame": tab,
            "log": log,
            "vars": {
                "bench.manifest_path": manifest_path,
                "bench.model_path": model_path,
                "bench.yolo_weights": yolo_weights,
                "bench.output_json": output_json,
                "bench.split": split,
                "bench.batch_size": batch_size,
                "bench.prob_threshold": prob_threshold,
                "bench.device": device,
                "bench.limit_videos": limit_videos,
                "bench.limit_windows": limit_windows,
                "bench.window_size": window_size,
                "bench.window_stride": window_stride,
                "bench.crop_size": crop_size,
                "bench.include_tail": include_tail,
                "bench.pad_ratio": pad_ratio,
                "bench.yolo_conf": yolo_conf,
                "bench.yolo_iou": yolo_iou,
                "bench.yolo_imgsz": yolo_imgsz,
                "bench.yolo_max_det": yolo_max_det,
                "bench.yolo_nms": yolo_nms,
                "bench.keep_last_box": keep_last_box,
            },
        }

    def _build_review_tab(self):
        tab = ttk.Frame(self.notebook)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        form = ScrollableFrame(tab)
        form.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        buttons = ttk.Frame(tab)
        buttons.grid(row=1, column=0, sticky="ew", padx=10)

        log_frame = ttk.Labelframe(tab, text="Log")
        log_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(8, 10))
        tab.grid_rowconfigure(2, weight=1)

        log = ScrolledText(log_frame, height=10, state="disabled")
        log.pack(fill="both", expand=True)

        header = ttk.Label(
            form.inner,
            text="Review: overlay the merged CSV annotations on a video for visual QA (optionally export an annotated MP4).",
            style="Header.TLabel",
            wraplength=900,
        )
        header.grid(row=0, column=0, sticky="w", pady=(0, 10))

        paths = ttk.Labelframe(form.inner, text="Paths")
        paths.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        video_path = tk.StringVar(value="")
        annotations_csv = tk.StringVar(value="")
        metrics_xlsx = tk.StringVar(value="")
        output_video = tk.StringVar(value="")

        self._add_path_row(
            paths,
            0,
            "Video path",
            video_path,
            "open_file",
            tooltip="Video to overlay annotations onto.",
            filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")],
        )
        self._add_path_row(
            paths,
            1,
            "Annotations CSV",
            annotations_csv,
            "open_file",
            tooltip="Merged CSV produced by inference (start/end time/frame, label, confidence).",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        self._add_path_row(
            paths,
            2,
            "Metrics XLSX (optional)",
            metrics_xlsx,
            "open_file",
            tooltip="Metrics workbook produced by inference (--output_xlsx). If provided, draws YOLO-style bbox labels.",
            filetypes=[("XLSX", "*.xlsx"), ("All files", "*.*")],
        )
        self._add_path_row(
            paths,
            3,
            "Output video (optional)",
            output_video,
            "save_file",
            tooltip="If provided, writes an annotated MP4 (mp4v).",
            filetypes=[("MP4", "*.mp4"), ("All files", "*.*")],
            save_ext=".mp4",
        )

        opts = ttk.Labelframe(form.inner, text="Overlay options")
        opts.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        no_display = tk.BooleanVar(value=False)
        font_scale = tk.StringVar(value="0.7")
        thickness = tk.StringVar(value="2")
        label_bg_alpha = tk.StringVar(value="1.0")
        wait_ms = tk.StringVar(value="1")

        self._add_row(
            opts,
            0,
            "No display (headless)",
            ttk.Checkbutton(opts, variable=no_display),
            tooltip="If enabled, does not open a preview window. Use this with Output video to export headlessly.",
        )
        self._add_row(opts, 1, "Font scale", ttk.Entry(opts, textvariable=font_scale), tooltip="OpenCV overlay text size.")
        self._add_row(opts, 2, "Thickness", ttk.Entry(opts, textvariable=thickness), tooltip="OpenCV text/border thickness.")
        self._add_row(opts, 3, "Label bg alpha", ttk.Entry(opts, textvariable=label_bg_alpha), tooltip="Opacity (0-1) for label background rectangles (lower reduces occlusion).")
        self._add_row(opts, 4, "Wait ms", ttk.Entry(opts, textvariable=wait_ms), tooltip="Delay for cv2.waitKey (increase to slow playback).")

        def _suggest_paths():
            try:
                vp = Path(video_path.get().strip())
                if not vp.exists():
                    return
                if not annotations_csv.get().strip():
                    annotations_csv.set(str(vp.with_suffix(".behavior.csv")))
                if not metrics_xlsx.get().strip():
                    metrics_xlsx.set(str(vp.with_suffix(".metrics.xlsx")))
            except Exception:
                return

        ttk.Button(form.inner, text="Suggest paths", command=_suggest_paths).grid(
            row=3, column=0, sticky="w", pady=(0, 10)
        )

        run_btn = ttk.Button(buttons, text="Run review")
        stop_btn = ttk.Button(buttons, text="Stop", state="disabled", command=self._stop_process)
        clear_btn = ttk.Button(buttons, text="Clear log", command=lambda: self._clear_log(log))
        run_btn.grid(row=0, column=0, sticky="w")
        stop_btn.grid(row=0, column=1, sticky="w", padx=(10, 0))
        clear_btn.grid(row=0, column=2, sticky="w", padx=(10, 0))

        def _run():
            try:
                vp = Path(video_path.get().strip())
                if not vp.exists():
                    raise ValueError("Video file does not exist.")
                metrics_path = (
                    Path(metrics_xlsx.get().strip())
                    if metrics_xlsx.get().strip()
                    else None
                )
                csv_path = (
                    Path(annotations_csv.get().strip())
                    if annotations_csv.get().strip()
                    else None
                )
                if metrics_path is None and csv_path is None:
                    raise ValueError("Provide either Metrics XLSX or Annotations CSV.")
                if metrics_path is not None and not metrics_path.exists():
                    raise ValueError("Metrics XLSX does not exist.")
                if metrics_path is None and csv_path is not None and not csv_path.exists():
                    raise ValueError("Annotations CSV does not exist.")

                fs = self._parse_float("Font scale", font_scale.get())
                th = self._parse_int("Thickness", thickness.get())
                bg = self._parse_float("Label bg alpha", label_bg_alpha.get())
                if bg < 0.0 or bg > 1.0:
                    raise ValueError("Label bg alpha must be between 0 and 1.")
                wm = self._parse_int("Wait ms", wait_ms.get())

                if no_display.get() and not output_video.get().strip():
                    raise ValueError("If No display is enabled, set Output video to export a file.")

                cmd = [
                    sys.executable,
                    str(APP_DIR / "review_annotations.py"),
                    "--video_path",
                    str(vp),
                    "--font_scale",
                    str(fs),
                    "--thickness",
                    str(th),
                    "--label_bg_alpha",
                    str(bg),
                    "--wait_ms",
                    str(wm),
                ]
                if metrics_path is not None:
                    cmd += ["--metrics_xlsx", str(metrics_path)]
                elif csv_path is not None:
                    cmd += ["--annotations_csv", str(csv_path)]
                if output_video.get().strip():
                    cmd += ["--output_video", output_video.get().strip()]
                if no_display.get():
                    cmd.append("--no_display")

                self._run_process(cmd, log, run_btn, stop_btn)
            except Exception as exc:
                messagebox.showerror("Review error", str(exc))

        run_btn.configure(command=_run)

        return {
            "frame": tab,
            "log": log,
            "vars": {
                "review.video_path": video_path,
                "review.annotations_csv": annotations_csv,
                "review.metrics_xlsx": metrics_xlsx,
                "review.output_video": output_video,
                "review.no_display": no_display,
                "review.font_scale": font_scale,
                "review.thickness": thickness,
                "review.label_bg_alpha": label_bg_alpha,
                "review.wait_ms": wait_ms,
            },
        }

    def _all_vars(self) -> dict[str, tk.Variable]:
        vars_map: dict[str, tk.Variable] = {}
        for tab in [
            self.preprocess_tab,
            self.train_tab,
            self.infer_tab,
            self.batch_tab,
            self.realtime_tab,
            self.benchmark_tab,
            self.review_tab,
        ]:
            vars_map.update(tab.get("vars", {}))
        return vars_map

    def _save_settings(self):
        path = filedialog.asksaveasfilename(
            title="Save GUI settings",
            initialdir=str(APP_DIR),
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        payload: dict[str, object] = {}
        for key, var in self._all_vars().items():
            payload[key] = var.get()
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        messagebox.showinfo("Saved", f"Saved settings to:\n{path}")

    def _load_settings(self):
        path = filedialog.askopenfilename(
            title="Load GUI settings",
            initialdir=str(APP_DIR),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vars_map = self._all_vars()
        applied = 0
        for key, val in payload.items():
            var = vars_map.get(key)
            if var is None:
                continue
            try:
                var.set(val)
                applied += 1
            except Exception:
                continue

        # Single-animal guardrails: these are intentionally locked.
        for key, val in (
            ("infer.yolo_max_det", "1"),
            ("infer.yolo_nms", True),
            ("bench.yolo_max_det", "1"),
            ("bench.yolo_nms", True),
        ):
            var = vars_map.get(key)
            if var is None:
                continue
            try:
                var.set(val)
            except Exception:
                continue
        messagebox.showinfo("Loaded", f"Applied {applied} settings from:\n{path}")


def main() -> None:
    root = tk.Tk()
    BehaviorScopeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

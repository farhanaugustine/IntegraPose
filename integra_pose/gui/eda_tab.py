import threading, os, tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image, ImageTk
from datetime import datetime
from pathlib import Path

from eda.service import EDAService
from eda.bridge import join_labels
from eda.generator import build_dataset

class EDATab(ttk.Frame):
    def __init__(self, parent, app=None):
        super().__init__(parent)
        self.app = app
        self.service = EDAService()
        self.df = None; self.X = None; self.models = None; self.labels = None
        self.cap = None; self.current_video_path = None; self.current_frame = 0; self.frame_count = 0; self._imgtk_ref = None
        self.sync_track_id = None; self.highlight = None; self.sync_enabled = True
        self._build()

    def _build(self):
        top = ttk.Frame(self); top.pack(fill="x", padx=8, pady=6)
        self.src_var = tk.StringVar(value="current")
        ttk.Radiobutton(top, text="Use current project outputs", variable=self.src_var, value="current").pack(side="left")
        ttk.Radiobutton(top, text="Load CSV…", variable=self.src_var, value="csv").pack(side="left", padx=(10,0))
        ttk.Button(top, text="Load", command=self.on_load).pack(side="left", padx=(10,0))
        ttk.Button(top, text="Build EDA Dataset", command=self.on_build_dataset).pack(side="left", padx=(10,0))
        ttk.Button(top, text="Help: CSV schema", command=self.on_help_schema).pack(side="left", padx=(10,0))

        filt = ttk.LabelFrame(self, text="Detections & Filtering"); filt.pack(fill="x", padx=8, pady=6)
        scope_frame = ttk.Frame(filt); scope_frame.pack(fill="x", padx=6, pady=4)
        ttk.Label(scope_frame, text="Detection scope:").pack(side="left")
        self.scope_var = tk.StringVar(value="multi")
        ttk.Radiobutton(scope_frame, text="All tracks (multi-animal)", variable=self.scope_var, value="multi").pack(side="left", padx=4)
        ttk.Radiobutton(scope_frame, text="First detection only", variable=self.scope_var, value="first").pack(side="left", padx=4)

        lists = ttk.Frame(filt); lists.pack(fill="x", padx=6, pady=4)
        l1 = ttk.Frame(lists); l1.pack(side="left", fill="both", expand=True, padx=4)
        ttk.Label(l1, text="Tracks").pack(anchor="w")
        self.track_list = tk.Listbox(l1, selectmode="extended", height=6, exportselection=False); self.track_list.pack(fill="both", expand=True)
        l1b = ttk.Frame(l1); l1b.pack(fill="x", pady=2)
        ttk.Button(l1b, text="All", command=lambda: self.track_list.select_set(0, tk.END)).pack(side="left", padx=2)
        ttk.Button(l1b, text="None", command=lambda: self.track_list.select_clear(0, tk.END)).pack(side="left", padx=2)
        l2 = ttk.Frame(lists); l2.pack(side="left", fill="both", expand=True, padx=4)
        ttk.Label(l2, text="Behaviors / Labels").pack(anchor="w")
        self.beh_list = tk.Listbox(l2, selectmode="extended", height=6, exportselection=False); self.beh_list.pack(fill="both", expand=True)
        l2b = ttk.Frame(l2); l2b.pack(fill="x", pady=2)
        ttk.Button(l2b, text="All", command=lambda: self.beh_list.select_set(0, tk.END)).pack(side="left", padx=2)
        ttk.Button(l2b, text="None", command=lambda: self.beh_list.select_clear(0, tk.END)).pack(side="left", padx=2)
        cb = ttk.Frame(filt); cb.pack(fill="x", padx=6, pady=4)
        ttk.Label(cb, text="Color points by:").pack(side="left")
        self.color_by = tk.StringVar(value="cluster")
        self.color_menu = ttk.Combobox(cb, textvariable=self.color_by, state="readonly", values=["cluster","behavior","track_id","none"], width=12)
        self.color_menu.pack(side="left", padx=6)
        ttk.Button(cb, text="Apply Filters", command=self.on_apply_filters).pack(side="left", padx=8)

        fs_frame = ttk.LabelFrame(self, text="Feature Selection"); fs_frame.pack(fill="x", padx=8, pady=4)
        self.feature_listbox = tk.Listbox(fs_frame, selectmode="extended", height=6, exportselection=False)
        self.feature_listbox.pack(side="left", fill="x", expand=True, padx=(6,6), pady=6)
        btns = ttk.Frame(fs_frame); btns.pack(side="left", padx=6, pady=6)
        ttk.Button(btns, text="Select All", command=self.on_select_all).pack(fill="x", pady=2)
        ttk.Button(btns, text="Select None", command=self.on_select_none).pack(fill="x", pady=2)

        mid = ttk.Frame(self); mid.pack(fill="x", padx=8, pady=6)
        ttk.Label(mid, text="PCA components:").pack(side="left")
        self.pca_components = tk.IntVar(value=2); ttk.Spinbox(mid, from_=2, to=5, textvariable=self.pca_components, width=5).pack(side="left", padx=(4,12))
        ttk.Label(mid, text="KMeans K:").pack(side="left")
        self.k_kmeans = tk.IntVar(value=5); ttk.Spinbox(mid, from_=2, to=20, textvariable=self.k_kmeans, width=5).pack(side="left", padx=(4,12))
        ttk.Button(mid, text="Run PCA + KMeans", command=self.on_run).pack(side="left")
        ttk.Button(mid, text="Toggle Sync Highlight", command=self.on_toggle_sync).pack(side="left", padx=(8,0))
        ttk.Label(mid, text="ⓘ PCA+KMeans complements UMAP+HDBSCAN.").pack(side="left", padx=(12,0))

        plot_frame = ttk.Frame(self); plot_frame.pack(fill="both", expand=True, padx=8, pady=6)
        self.fig = plt.Figure(figsize=(5,4), dpi=100); self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame); self.canvas.get_tk_widget().pack(fill="both", expand=True)

        vp = ttk.LabelFrame(self, text="Video Preview"); vp.pack(fill="x", padx=8, pady=6)
        self.video_label = ttk.Label(vp, text="Click a point to preview frame (needs video_path & frame_id)"); self.video_label.pack(side="top", fill="x", padx=6, pady=6)
        self.video_canvas = ttk.Label(vp); self.video_canvas.pack(side="top", padx=6, pady=6)
        ctrls = ttk.Frame(vp); ctrls.pack(side="top", pady=4)
        ttk.Button(ctrls, text="Prev", command=self.on_prev_frame).pack(side="left", padx=4)
        self.playing = False; ttk.Button(ctrls, text="Play/Pause", command=self.on_play_pause).pack(side="left", padx=4)
        ttk.Button(ctrls, text="Next", command=self.on_next_frame).pack(side="left", padx=4)

        bot = ttk.Frame(self); bot.pack(fill="x", padx=8, pady=6)
        ttk.Button(bot, text="Export Labels CSV", command=self.on_export_labels).pack(side="left")
        ttk.Button(bot, text="Save Plot", command=self.on_save_plot).pack(side="left", padx=(8,0))

    def log(self, msg, level="INFO"):
        if self.app and hasattr(self.app, "log_message"): self.app.log_message(msg, level)
        else: print(f"[{level}] {msg}")

    def _project_roots(self):
        roots = []
        if self.app and hasattr(self.app, "project_dir") and self.app.project_dir:
            roots.append(self.app.project_dir)
        exp = []
        for r in list(roots) or [os.getcwd()]:
            for name in ("eda_outputs","outputs","results","runs","export","inference","clustering"):
                p = os.path.join(r, name)
                if os.path.isdir(p): exp.append(p)
        roots.extend(exp)
        roots.append(os.getcwd())
        return roots

    def _discover_current_csv(self):
        roots = self._project_roots()
        candidates = []
        for r in roots:
            for dirpath, _, filenames in os.walk(r):
                for fn in filenames:
                    f = fn.lower()
                    if f.endswith(".csv") and any(k in f for k in ("eda_dataset","eda_labels","pose","poses","keypoint","kpts","tracks","track","clusters","cluster","labels","behavior")):
                        p = os.path.join(dirpath, fn)
                        try:
                            mtime = os.path.getmtime(p)
                        except Exception:
                            mtime = 0
                        candidates.append((mtime, p))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        for _, p in candidates:
            if "eda_dataset" in os.path.basename(p):
                return p
        return candidates[0][1]

    def _canonical_save_dir(self):
        base = Path(self.app.project_dir) if (self.app and getattr(self.app,'project_dir',None)) else Path(os.getcwd())
        out = base / "eda_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        return out

    def on_build_dataset(self):
        initial = self.app.project_dir if (self.app and getattr(self.app,'project_dir',None)) else os.getcwd()
        root = filedialog.askdirectory(initialdir=initial, title="Select project folder to build EDA dataset from")
        if not root: return
        try:
            df = build_dataset(root)
            save_dir = self._canonical_save_dir()
            out_csv = save_dir / "eda_dataset.csv"
            df.to_csv(out_csv, index=False)
            self.df = df
            self.ax.clear(); self.ax.set_title("Built EDA dataset"); self.canvas.draw()
            self.populate_feature_list(); self.populate_filters()
            messagebox.showinfo("EDA dataset built", f"Saved to:\n{out_csv}")
            self.log(f"EDA dataset built: {out_csv}")
        except Exception as e:
            messagebox.showerror("Build failed", str(e))

    def on_help_schema(self):
        msg = (
            "EDA expects a CSV with at least these columns:\n"
            "  • frame_id (int)\n"
            "  • track_id (int)\n"
            "  • video_path (string)\n\n"
            "Recommended extras:\n"
            "  • behavior / behavior_name / cluster_label / hdbscan_label\n"
            "  • numeric features (e.g., keypoint_x / keypoint_y / distances / angles)\n\n"
            "Tip: Use 'Build EDA Dataset' to auto-merge your project CSVs into a canonical file."
        )
        messagebox.showinfo("EDA CSV Schema", msg)

    def on_load(self):
        src = self.src_var.get()
        if src == "csv":
            path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")]); 
            if not path: return
        else:
            path = self._discover_current_csv()
            if not path:
                try:
                    base = self.app.project_dir if (self.app and getattr(self.app,'project_dir',None)) else os.getcwd()
                    df = build_dataset(base)
                    save_dir = self._canonical_save_dir()
                    out_csv = save_dir / "eda_dataset.csv"
                    df.to_csv(out_csv, index=False)
                    path = str(out_csv)
                    self.log(f"Auto-built EDA dataset at {out_csv}")
                except Exception:
                    messagebox.showwarning("No CSV found", "Could not find or auto-build a pose/clustering CSV in the project. Use 'Build EDA Dataset'.")
                    return
        try:
            self.df = self.service.prepare(path, opts={})
            self.X = None; self.labels = None
            self.ax.clear(); self.ax.set_title("Loaded data"); self.canvas.draw()
            self.populate_feature_list(); self.populate_filters()
            self.log(f"Loaded data from {path}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def populate_feature_list(self):
        if self.df is None: return
        num_cols = [c for c in self.df.columns if (str(self.df[c].dtype).startswith(('int','float')) and c not in ('frame_id','track_id'))]
        self.feature_listbox.delete(0, tk.END)
        for c in num_cols: self.feature_listbox.insert(tk.END, c)

    def populate_filters(self):
        self.track_list.delete(0, tk.END)
        if self.df is not None and "track_id" in self.df.columns:
            tracks = list(dict.fromkeys(self.df["track_id"].dropna().tolist()))
            for t in tracks: self.track_list.insert(tk.END, str(t))
        self.beh_list.delete(0, tk.END)
        behavior_col = None
        if self.df is not None:
            for cand in ("behavior","behavior_name","eda_behavior","cluster_label","hdbscan_label"):
                if cand in self.df.columns: behavior_col = cand; break
        if behavior_col:
            vals = list(dict.fromkeys(self.df[behavior_col].dropna().astype(str).tolist()))
            for v in vals: self.beh_list.insert(tk.END, v)

    def on_select_all(self): self.feature_listbox.select_set(0, tk.END)
    def on_select_none(self): self.feature_listbox.select_clear(0, tk.END)

    def on_apply_filters(self):
        if self.df is None: return
        try:
            tracks = [self.track_list.get(i) for i in self.track_list.curselection()]
            behs = [self.beh_list.get(i) for i in self.beh_list.curselection()]
            scope = self.scope_var.get()
            cfg = {"selected_tracks": tracks, "selected_behaviors": behs, "detection_scope": scope}
            _svc = EDAService(); self.df = _svc.apply_filters(self.df, cfg)
            self.ax.clear(); self.ax.set_title("Filters applied"); self.canvas.draw()
            self.populate_feature_list(); self.populate_filters()
            self.log(f"Applied filters: scope={scope}, tracks={len(tracks) or 'ALL'}, behaviors={len(behs) or 'ALL'}")
        except Exception as e:
            self.log(f"Filter error: {e}", "ERROR")

    def _ensure_cap(self, video_path):
        if not video_path or not os.path.exists(video_path): self.log(f'Video not found: {video_path}', 'ERROR'); return False
        if self.cap is not None and self.current_video_path == video_path: return True
        if self.cap is not None:
            try: self.cap.release()
            except: pass
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened(): self.log(f'Failed to open video: {video_path}', 'ERROR'); self.cap = None; return False
        self.current_video_path = video_path; self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); return True

    def _show_frame(self, frame_id):
        if self.cap is None: return
        frame_id = max(0, min(int(frame_id), max(0, self.frame_count-1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id); ok, frame = self.cap.read()
        if not ok or frame is None: return
        self.current_frame = frame_id
        import numpy as np
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]; max_w = 640; scale = min(1.0, max_w / max(1, w))
        if scale != 1.0: img = cv2.resize(img, (int(w*scale), int(h*scale)))
        im = Image.fromarray(img); imgtk = ImageTk.PhotoImage(image=im); self._imgtk_ref = imgtk; self.video_canvas.configure(image=imgtk)
        try:
            if self.sync_enabled and self.df is not None and self.models is not None and 'frame_id' in self.df.columns:
                vpath = self.current_video_path
                m = (self.df['frame_id'].astype('Int64') == int(self.current_frame))
                if 'video_path' in self.df.columns and vpath is not None:
                    m = m & (self.df['video_path'].astype(str) == str(vpath))
                idxs = list(self.df.index[m])
                if idxs:
                    pos = None
                    if self.sync_track_id is not None and 'track_id' in self.df.columns:
                        for irow in idxs:
                            if str(self.df.loc[irow, 'track_id']) == str(self.sync_track_id):
                                pos = irow; break
                    if pos is None: pos = idxs[0]
                    iloc_pos = self.df.index.get_loc(pos)
                    self._update_highlight(iloc_pos)
        except Exception:
            pass

    def on_pick(self, event):
        if self.df is None or self.models is None: return
        if event.name == 'pick_event' and hasattr(event, 'ind') and len(event.ind): idx = int(event.ind[0])
        else:
            Z = self.models.get('embedding'); 
            if Z is None or event.xdata is None or event.ydata is None: return
            import numpy as np; xy = np.array([event.xdata, event.ydata]); d = ((Z - xy)**2).sum(1); idx = int(d.argmin())
        row = self.df.iloc[idx]
        try:
            self.sync_track_id = int(row.get('track_id')) if 'track_id' in self.df.columns else None
        except Exception:
            self.sync_track_id = row.get('track_id') if 'track_id' in self.df.columns else None
        self._update_highlight(idx)
        vpath = row.get('video_path') if 'video_path' in self.df.columns else None
        fid = row.get('frame_id') if 'frame_id' in self.df.columns else None
        if vpath is None or fid is None: self.video_label.config(text="Selected point missing video_path/frame_id"); return
        if not self._ensure_cap(vpath): self.video_label.config(text=f'Cannot open video: {vpath}'); return
        self.video_label.config(text=f"{vpath} | frame {int(fid)} / {self.frame_count}")
        self._show_frame(int(fid))

    def on_prev_frame(self): 
        if self.cap is None: return
        self._show_frame(self.current_frame - 1)
    def on_next_frame(self): 
        if self.cap is None: return
        self._show_frame(self.current_frame + 1)
    def _play_loop(self):
        if not self.playing: return
        self._show_frame(self.current_frame + 1)
        try: self.after(33, self._play_loop)
        except: pass
    def on_play_pause(self):
        self.playing = not self.playing
        if self.playing: self._play_loop()

    def _update_highlight(self, idx):
        if not self.sync_enabled or self.models is None:
            return
        Z = self.models.get('embedding')
        if Z is None or idx is None:
            return
        if isinstance(idx, (list, tuple)) and idx:
            idx = idx[0]
        if self.highlight is None:
            try:
                (self.highlight,) = self.ax.plot([], [], marker='o', markersize=10, fillstyle='none', markeredgewidth=2)
            except Exception:
                return
        x, y = float(Z[int(idx), 0]), float(Z[int(idx), 1])
        self.highlight.set_data([x], [y])
        self.canvas.draw_idle()

    def on_toggle_sync(self):
        self.sync_enabled = not self.sync_enabled
        if not self.sync_enabled and self.highlight is not None:
            try: self.highlight.set_data([], []); self.canvas.draw_idle()
            except Exception: pass

    def _run_worker(self):
        try:
            if self.df is None: raise RuntimeError("Load or build an EDA dataset first")
            sel = [self.feature_listbox.get(i) for i in self.feature_listbox.curselection()]
            self.X = self.service.compute_features(self.df, cfg={'selected_features': sel})
            labels, models, status = self.service.pca_kmeans(self.X, n_components=self.pca_components.get(), k=self.k_kmeans.get())
            self.labels = labels; self.models = models
            Z = models.get("embedding")
            cb = (self.color_by.get() if hasattr(self, 'color_by') else 'cluster'); cmap_vals = labels
            if cb == 'behavior' and 'behavior' in self.df.columns:
                cats = pd.Categorical(self.df['behavior'].astype(str)); cmap_vals = cats.codes
            elif cb == 'track_id' and 'track_id' in self.df.columns:
                cmap_vals = self.df['track_id'].astype('category').cat.codes.values
            elif cb == 'none':
                cmap_vals = None
            self.ax.clear(); self.ax.scatter(Z[:,0], Z[:,1], c=cmap_vals, s=10, picker=True)
            self.fig.canvas.mpl_connect('pick_event', self.on_pick); self.fig.canvas.mpl_connect('button_press_event', self.on_pick)
            self.ax.set_xlabel("PC1"); self.ax.set_ylabel("PC2"); ev = status.get("explained_variance", [])
            self.ax.set_title(f"PCA+KMeans (EVR: {ev})"); self.canvas.draw(); self.log("PCA+KMeans completed")
        except Exception as e:
            self.log(f"EDA run error: {e}", "ERROR"); messagebox.showerror("EDA run error", str(e))

    def on_run(self): threading.Thread(target=self._run_worker, daemon=True).start()

    def on_export_labels(self):
        if self.df is None or self.labels is None: messagebox.showinfo("Nothing to export", "Run analysis first."); return
        out = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")], initialfile=f"eda_labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        if not out: return
        join_labels(self.df, self.labels, "eda_cluster").to_csv(out, index=False); self.log(f"Exported labels to {out}")

    def on_save_plot(self):
        if self.models is None: messagebox.showinfo("No plot", "Run analysis first."); return
        out = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")], initialfile=f"eda_pca_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        if not out: return
        self.fig.savefig(out, bbox_inches='tight', dpi=150); self.log(f"Saved plot to {out}")

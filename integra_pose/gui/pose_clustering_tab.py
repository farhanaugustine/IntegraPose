import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

from .scrollable import create_scrollable_tab
from .workflow_nav import add_workflow_footer_grid


def create_pose_clustering_tab(app: 'YoloApp') -> None:
    """Repurpose the tab as a launcher for the VAE + HMM segmentation toolkit."""
    tab = create_scrollable_tab(app, app.pose_clustering_tab)
    app.pose_clustering_content = tab
    tab.columnconfigure(0, weight=1)

    header = ttk.Label(tab, text="Behavior Clustering (VAE + HMM)", font=("Segoe UI", 12, "bold"))
    header.grid(row=0, column=0, sticky="w", pady=(0, 6))
    sub = ttk.Label(
        tab,
        text=(
            "Tab 7 models latent movement structure directly from pose data. "
            "If Tab 6 or batch analytics outputs are available, you can also import their run manifests to "
            "reuse bout segmentation and carry over source metadata."
        ),
        wraplength=820,
        justify="left",
    )
    sub.grid(row=1, column=0, sticky="w", pady=(0, 10))

    card = ttk.LabelFrame(tab, text="Input Paths", padding=12)
    card.grid(row=2, column=0, sticky="ew", padx=4, pady=4)
    card.columnconfigure(0, weight=1)
    card.columnconfigure(1, weight=1)
    card.columnconfigure(2, weight=1)

    ttk.Label(card, text="Choose the cleanest starting point for your dataset:", style="Status.TLabel").grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, 6)
    )
    steps = (
        "1) From Tab 6: import the latest single-run Bout Analytics manifest from the main workflow.",
        "2) From Batch Analytics: import one or more completed batch run manifests, grouped by condition when available.",
        "3) From Raw Pose + Video: launch the toolkit and add pose/video sources manually without Tab 6.",
        "4) In all cases, Tab 7 recomputes modeling features from pose data; imported manifests mainly reuse bouts and metadata.",
    )
    for i, text in enumerate(steps, start=1):
        ttk.Label(card, text=text).grid(row=i, column=0, columnspan=3, sticky="w")

    dep_label = ttk.Label(card, text="", style="Status.TLabel")
    dep_label.grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 4))

    def _check():
        missing, present = app._check_hmm_vae_dependencies()
        if missing:
            dep_label.config(text=f"Missing: {', '.join(missing)}", foreground="#b91c1c")
            app._toast(f"Missing deps: {', '.join(missing)}", level="error")
        else:
            dep_label.config(text="All required dependencies found.", foreground="#0f172a")
            app._toast("VAE + HMM dependencies OK.", level="info")

    ttk.Button(card, text="Check dependencies", command=_check).grid(row=6, column=0, padx=(0, 8), pady=(4, 4), sticky="w")
    ttk.Button(
        card,
        text="Continue from Latest Tab 6 Run",
        command=app._open_tab7_from_bout_analytics,
    ).grid(row=6, column=1, padx=4, pady=(4, 4), sticky="ew")
    ttk.Button(
        card,
        text="Import Analytics Manifest(s)...",
        command=app._open_tab7_manifest_picker,
    ).grid(row=6, column=2, padx=(8, 0), pady=(4, 4), sticky="ew")

    launch_note = ttk.Label(
        card,
        text=(
            "Batch Wizard note: each completed analytics video writes its own run_manifest.json. "
            "Use the Batch Wizard's Tab 7 actions or import those manifests here."
        ),
        wraplength=820,
        justify="left",
    )
    launch_note.grid(row=7, column=0, columnspan=3, sticky="w", pady=(2, 6))

    ttk.Button(
        card,
        text="Launch Toolkit for Manual / Raw Sources",
        command=lambda: _embed_toolkit(app, tab),
        style="Accent.TButton",
    ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(0, 4))

    log_hint = ttk.LabelFrame(tab, text="Outputs & Logs", padding=12)
    log_hint.grid(row=3, column=0, sticky="ew", padx=4, pady=4)
    log_hint.columnconfigure(0, weight=1)

    built_in_dir = Path(__file__).resolve().parents[1] / "hmm_vae_toolkit"
    log_path = (built_in_dir / "behavior_analysis.log") if built_in_dir.exists() else None
    log_text = f"Toolkit log: {log_path}" if log_path else "Toolkit log: behavior_analysis.log (installed with IntegraPose)."
    ttk.Label(log_hint, text=log_text, wraplength=820, justify="left").grid(row=0, column=0, sticky="w")

    ttk.Button(log_hint, text="Open Toolkit Log", command=lambda: _open_path(log_path)).grid(row=1, column=0, sticky="w", pady=(6, 2))

    add_workflow_footer_grid(app, tab, row=10)

    app.log_message("Behavior Clustering tab is now a VAE + HMM launcher.", "INFO")


def _open_path(path: Path | None):
    if not path:
        return
    try:
        if path.exists():
            path_str = str(path)
            import os
            os.startfile(path_str)  # type: ignore[attr-defined]
    except Exception:
        messagebox.showinfo("Open Log", f"Log file: {path}")


def _embed_toolkit(app, tab):
    """Instantiate the VAE+HMM toolkit inside the tab."""
    toolkit_row = 9
    try:
        app._embedded_behavior_analysis_app = None
    except Exception:
        pass
    # Clear previous embedded toolkit if any
    for child in tab.grid_slaves(row=toolkit_row, column=0):
        child.destroy()
    container = ttk.Frame(tab, padding=6)
    container.grid(row=toolkit_row, column=0, sticky="nsew", pady=(10, 0))
    tab.grid_rowconfigure(toolkit_row, weight=1)
    tab.grid_columnconfigure(0, weight=1)

    missing, _present = app._check_hmm_vae_dependencies()
    if missing:
        app._toast(f"Missing deps: {', '.join(missing)}", level="error")
        messagebox.showerror("VAE + HMM Toolkit", f"Install dependencies first:\n{', '.join(missing)}", parent=app.root)
        return
    toolkit_module = None
    toolkit_source = ""
    try:
        import importlib

        toolkit_module = importlib.import_module("integra_pose.hmm_vae_toolkit")
        toolkit_source = "built-in"
    except Exception as exc:
        app.log_message(f"Built-in VAE+HMM import failed: {exc}", "ERROR")
        messagebox.showerror(
            "VAE + HMM Toolkit",
            f"Could not load the built-in VAE + HMM toolkit:\n{exc}\n\n"
            "Reinstall IntegraPose or check that integra_pose/hmm_vae_toolkit "
            "is intact.",
            parent=app.root,
        )
        return
    try:
        BehaviorAnalysisApp = getattr(toolkit_module, "BehaviorAnalysisApp", None)
        if BehaviorAnalysisApp is None:
            raise AttributeError("BehaviorAnalysisApp not available in toolkit.")
        try:
            controller = BehaviorAnalysisApp(container, embed=True, integra_app=app)
        except TypeError:
            controller = BehaviorAnalysisApp(container, embed=True)
        try:
            app._embedded_behavior_analysis_app = controller
        except Exception:
            pass
        app._toast(f"Embedded VAE + HMM toolkit ({toolkit_source}).", level="info")
        return controller
    except Exception as exc:
        app.log_message(f"Failed to embed VAE + HMM toolkit: {exc}", "ERROR")
        messagebox.showerror("VAE + HMM Toolkit", f"Failed to embed: {exc}", parent=app.root)
        return None


def ensure_toolkit_embedded(app):
    """Return the embedded BehaviorAnalysisApp instance, embedding it if needed."""
    existing = getattr(app, "_embedded_behavior_analysis_app", None)
    if existing is not None:
        try:
            root = getattr(existing, "root", None)
            if root is not None and hasattr(root, "winfo_exists") and root.winfo_exists():
                return existing
        except Exception:
            pass
    tab = getattr(app, "pose_clustering_content", None)
    if tab is None:
        return None
    return _embed_toolkit(app, tab)

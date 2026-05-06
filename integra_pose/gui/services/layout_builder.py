"""Notebook/layout construction for the main GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from integra_pose.gui import (
    setup_tab,
    training_tab,
    inference_tab,
    webcam_tab,
    roi_analytics_tab,
    log_tab,
    pose_clustering_tab,
    data_preprocessing_tab,
)


class LayoutBuilder:
    """Builds the notebook, tab frames, and status strip for the app."""

    def __init__(self, app) -> None:
        self.app = app

    def build(self) -> None:
        self._build_notebook_and_tabs()
        self._build_status_strip()

    def _build_notebook_and_tabs(self) -> None:
        app = self.app
        notebook = ttk.Notebook(app.root)
        app.notebook = notebook
        app.workflow_step_buttons = []
        app.stage_status_specs = [
            ("data_preprocessing", "Prep"),
            ("setup", "Setup"),
            ("training", "Train"),
            ("inference", "Infer"),
            ("webcam", "Webcam"),
            ("analytics", "Analytics"),
            ("pose_clustering", "Cluster"),
        ]
        app.stage_status_labels = {}

        app.setup_tab = ttk.Frame(notebook, padding="10")
        app.data_preprocessing_tab = ttk.Frame(notebook, padding="10")
        app.training_tab = ttk.Frame(notebook, padding="10")
        app.inference_tab = ttk.Frame(notebook, padding="10")
        app.webcam_tab = ttk.Frame(notebook, padding="10")
        app.roi_analytics_tab = ttk.Frame(notebook, padding="10")
        app.pose_clustering_tab = ttk.Frame(notebook, padding="10")
        app.tab_base_labels = {
            "data_preprocessing": "1. Data Preprocessing",
            "setup": "2. Setup & Annotation",
            "training": "3. Model Training",
            "inference": "4. Inference",
            "webcam": "5. Webcam Inference",
            "analytics": "6. Bout Analytics",
            "pose_clustering": "7. Behavior Clustering (VAE + HMM)",
            "log": "Log",
        }
        app.tab_widgets = {
            "data_preprocessing": app.data_preprocessing_tab,
            "setup": app.setup_tab,
            "training": app.training_tab,
            "inference": app.inference_tab,
            "webcam": app.webcam_tab,
            "analytics": app.roi_analytics_tab,
            "pose_clustering": app.pose_clustering_tab,
        }

        notebook.add(app.data_preprocessing_tab, text=app.tab_base_labels["data_preprocessing"])
        notebook.add(app.setup_tab, text=app.tab_base_labels["setup"])
        notebook.add(app.training_tab, text=app.tab_base_labels["training"])
        notebook.add(app.inference_tab, text=app.tab_base_labels["inference"])
        notebook.add(app.webcam_tab, text=app.tab_base_labels["webcam"])
        notebook.add(app.roi_analytics_tab, text=app.tab_base_labels["analytics"])
        notebook.add(app.pose_clustering_tab, text=app.tab_base_labels["pose_clustering"])

        data_preprocessing_tab.create_data_preprocessing_tab(app)
        setup_tab.create_setup_tab(app)
        training_tab.create_training_tab(app)
        inference_tab.create_inference_tab(app)
        webcam_tab.create_webcam_tab(app)
        app._update_webcam_roi_toggle_state()
        roi_analytics_tab.create_roi_analytics_tab(app)
        pose_clustering_tab.create_pose_clustering_tab(app)
        log_tab.create_log_tab(app)

        status_strip = ttk.LabelFrame(app.root, text="Stage Status", padding=(8, 6))
        status_strip.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 4))
        app.stage_status_strip = status_strip

        for idx, (key, label) in enumerate(app.stage_status_specs):
            status_strip.columnconfigure(idx, weight=1)
            pill = ttk.Label(
                status_strip,
                text=f"\u2022 {label}\nNot started",
                style="StageNotStarted.TLabel",
                anchor="center",
                justify="center",
                width=13,
            )
            pill.grid(row=0, column=idx, sticky="ew", padx=3, pady=1)
            app.stage_status_labels[key] = pill

        app.root.grid_columnconfigure(0, weight=1)
        app.root.grid_rowconfigure(2, weight=1)
        notebook.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

    def _build_status_strip(self) -> None:
        app = self.app
        job_frame = ttk.Frame(app.root, padding=(8, 4))
        ttk.Label(job_frame, text="Jobs:", style="Status.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        app.job_status_label = ttk.Label(job_frame, textvariable=app.job_status_var, style="Status.TLabel")
        app.job_status_label.pack(side=tk.LEFT)
        job_frame.grid(row=0, column=0, sticky="ew")

        app.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(app.root, textvariable=app.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.grid(row=3, column=0, sticky="ew")

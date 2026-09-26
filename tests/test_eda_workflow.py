"""Regression checks for EDA's analysis actions and window lifecycle."""
import tkinter as tk
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from integra_pose.plugins.plugin_eda import plugin
from integra_pose.plugins.plugin_eda.ui import app as eda_ui


@pytest.fixture
def eda(monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    errors = []
    monkeypatch.setattr(eda_ui.messagebox, "showerror", lambda title, text, **kw: errors.append(str(text)))
    monkeypatch.setattr(eda_ui.messagebox, "showinfo", lambda *a, **kw: None)
    monkeypatch.setattr(eda_ui.messagebox, "showwarning", lambda *a, **kw: None)
    app = eda_ui.PoseEDAApp(root)
    app.df_features = pd.DataFrame({
        "frame_id": np.arange(9),
        "behavior_name": ["rest"]*3 + ["investigation"]*3 + ["attack"]*3,
        "length": [1., 1.2, 1.3, 3., 3.2, 3.4, 6., 6.1, np.nan],
        "angle": [20., 21., 23., 45., 47., 49., 80., 83., 85.],
    })
    app.feature_columns_list = ["length", "angle"]
    app._update_analysis_tab_feature_lists_ui()
    app.analysis_features_listbox.selection_set(0, tk.END)
    app.pca_n_components_var.set(2)
    app.kmeans_k_var.set(3)
    def synchronous(work, success, failure):
        try:
            result = work()
        except Exception as exc:
            failure(exc)
        else:
            success(result)
    monkeypatch.setattr(app, "_run_bg", synchronous)
    try:
        yield app, errors
    finally:
        app.on_closing()


@pytest.mark.parametrize("target", ["instances", "avg_behaviors"])
def test_ahc_dendrogram_without_flat_assignments(eda, target):
    app, errors = eda
    app.cluster_target_var.set(target)
    app.cluster_method_var.set("AHC")
    app.ahc_num_clusters_var.set(0)
    app._perform_clustering_action()
    assert not errors
    assert app.analysis_handler.last_linkage_matrix is not None
    text = app.text_results_area.get("1.0", tk.END).lower()
    assert "dendrogram" in text and "no flat" in text
    if target == "instances":
        assert len(app.df_features) == 9
        assert app.df_features.unsupervised_cluster.isna().all()
    assert len(app.ax_results.collections) > 0


@pytest.mark.parametrize("method", ["KMeans", "AHC"])
def test_assignments_preserve_missing_rows_and_frame_identity(eda, method):
    app, errors = eda
    app.cluster_method_var.set(method)
    app.ahc_num_clusters_var.set(3)
    app._perform_clustering_action()
    assert not errors
    assert app.df_features.frame_id.tolist() == list(range(9))
    assert app.df_features.loc[:7, "unsupervised_cluster"].notna().all()
    assert pd.isna(app.df_features.loc[8, "unsupervised_cluster"])
    assert pd.isna(app.df_features.loc[8, "PC1"])


def test_rerun_clears_old_assignments_coordinates_and_legend(eda):
    app, errors = eda
    app.cluster_method_var.set("KMeans")
    app._perform_clustering_action()
    assert app.cluster_dominant_behavior_map
    assert "PC2" in app.df_features
    app.cluster_method_var.set("AHC")
    app.pca_n_components_var.set(1)
    app.ahc_num_clusters_var.set(0)
    app._perform_clustering_action()
    assert not errors
    assert "PC1" in app.df_features and "PC2" not in app.df_features
    assert app.df_features.unsupervised_cluster.isna().all()
    assert app.cluster_dominant_behavior_map == {}


def test_plugin_retains_window_when_child_is_destroyed(eda, monkeypatch):
    app, errors = eda
    root = app.root
    created = []
    monkeypatch.setattr(plugin, "_eda_gui_module", SimpleNamespace(PoseEDAApp=lambda window, **kw: created.append(window)))
    extension = plugin.IntegraPosePlugin()
    extension.main_app = SimpleNamespace(root=root, log_message=lambda *a: None)
    extension._launch_eda_tool()
    window = extension._window
    tk.Frame(window).destroy()
    root.update()
    assert extension._window is window
    extension._launch_eda_tool()
    assert created == [window]
    window.destroy()
    root.update()
    assert extension._window is None


def test_synchronized_recording_contains_decodable_frames(eda, tmp_path, monkeypatch):
    import cv2
    app, errors = eda
    source = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10., (160, 120))
    assert writer.isOpened()
    for i in range(3):
        writer.write(np.full((120, 160, 3), 50 + i * 50, np.uint8))
    writer.release()
    app.video_sync_video_path_var.set(str(source))
    app._load_video_action()
    target = tmp_path / "linked.mp4"
    monkeypatch.setattr(eda_ui.filedialog, "asksaveasfilename", lambda **kw: str(target))
    app._start_video_recording()
    assert app.video_sync_is_recording
    for i in range(3):
        app._update_current_video_frame_and_plot_ui(i)
    app._stop_video_recording()
    capture = cv2.VideoCapture(str(target))
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()
    assert not errors
    assert len(decoded) == 3
    assert all(frame.shape == (480, 1280, 3) for frame in decoded)

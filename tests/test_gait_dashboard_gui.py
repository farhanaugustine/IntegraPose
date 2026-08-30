import tkinter as tk
from tkinter import ttk

import pytest

from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.gui_launcher import (
    AnalysisGUI,
)


def test_dashboard_constructs_with_isolated_run_button_style() -> None:
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    dashboard = None
    try:
        host_style = ttk.Style(root)
        host_style.configure("Accent.TButton", background="#123456")

        dashboard = AnalysisGUI(root)
        dashboard.withdraw()

        assert dashboard.run_button.cget("style") == "Gait.Accent.TButton"
        assert host_style.configure("Gait.Accent.TButton")["background"] == "#28a745"
        assert host_style.configure("Accent.TButton")["background"] == "#123456"
    finally:
        if dashboard is not None:
            dashboard.destroy()
        root.destroy()

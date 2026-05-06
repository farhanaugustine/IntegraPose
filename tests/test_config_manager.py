import os
import sys
import tkinter as tk
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from integra_pose.utils.config_manager import ConfigManager


class TestConfigManager(unittest.TestCase):

    def setUp(self):
        self.app = MagicMock()
        self._previous_default_root = getattr(tk, "_default_root", None)

        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError:
            try:
                root = tk.Tk(useTk=False)
            except tk.TclError as exc:
                self.skipTest(f"Tk is unavailable in this environment: {exc}")

        self.app.root = root
        # ConfigManager creates Tk variables without explicit masters, so ensure a default root exists.
        tk._default_root = root
        self.config_manager = ConfigManager(self.app)

    def tearDown(self):
        root = getattr(self.app, "root", None)
        if root is not None:
            try:
                commands = set(root.tk.call("info", "commands"))
            except tk.TclError:
                commands = set()
            if "destroy" in commands:
                root.destroy()
        tk._default_root = self._previous_default_root

    def test_overlay_preset_serialization(self):
        preset_name = "My Test Preset"
        preset_data = {
            "booleans": {
                "sv_use_box_var": True,
                "sv_use_label_var": True,
                "sv_show_tracker_ids_var": True,
                "sv_show_heading_arrows_var": True,
            },
            "scalars": {"sv_skeleton_color_var": "blue"},
            "advanced": {"sv_halo_kernel_var": 50},
            "overlay_order": ["boxes", "labels"],
            "performance_safe": False,
        }
        self.config_manager.inference.overlay_presets.append({"name": preset_name, "values": preset_data})

        config_dict = self.config_manager._gather_config_as_dict()

        self.assertIn("inference", config_dict)
        self.assertIn("overlay_presets", config_dict["inference"])
        self.assertEqual(len(config_dict["inference"]["overlay_presets"]), 1)
        self.assertEqual(config_dict["inference"]["overlay_presets"][0]["name"], preset_name)
        self.assertEqual(config_dict["inference"]["overlay_presets"][0]["values"], preset_data)

        new_config_manager = ConfigManager(self.app)
        new_config_manager._apply_config_from_dict(config_dict)

        self.assertEqual(len(new_config_manager.inference.overlay_presets), 1)
        self.assertEqual(new_config_manager.inference.overlay_presets[0]["name"], preset_name)
        self.assertEqual(new_config_manager.inference.overlay_presets[0]["values"], preset_data)

    def test_pose_clustering_serialization(self):
        pose_cfg = self.config_manager.pose_clustering
        pose_cfg.umap_neighbors_var.set("32")
        pose_cfg.hdbscan_min_size_var.set("18")
        pose_cfg.norm_translation_var.set(False)
        pose_cfg.norm_scale_var.set(False)
        pose_cfg.pose_visualization_type.set("Median")
        pose_cfg.single_animal_clustering_var.set(True)
        pose_cfg.skeleton_connections = [(0, 2), (2, 5)]

        config_dict = self.config_manager._gather_config_as_dict()

        self.assertIn("pose_clustering", config_dict)
        pose_section = config_dict["pose_clustering"]
        self.assertEqual(pose_section["umap_neighbors_var"], "32")
        self.assertEqual(pose_section["hdbscan_min_size_var"], "18")
        self.assertFalse(pose_section["norm_translation_var"])
        self.assertFalse(pose_section["norm_scale_var"])
        self.assertEqual(pose_section["pose_visualization_type"], "Median")
        self.assertTrue(pose_section["single_animal_clustering_var"])
        self.assertEqual(pose_section["skeleton_connections"], [(0, 2), (2, 5)])

        new_config_manager = ConfigManager(self.app)
        new_config_manager._apply_config_from_dict(config_dict)

        new_pose_cfg = new_config_manager.pose_clustering
        self.assertEqual(new_pose_cfg.umap_neighbors_var.get(), "32")
        self.assertEqual(new_pose_cfg.hdbscan_min_size_var.get(), "18")
        self.assertFalse(new_pose_cfg.norm_translation_var.get())
        self.assertFalse(new_pose_cfg.norm_scale_var.get())
        self.assertEqual(new_pose_cfg.pose_visualization_type.get(), "Median")
        self.assertTrue(new_pose_cfg.single_animal_clustering_var.get())
        self.assertEqual(new_pose_cfg.skeleton_connections, [(0, 2), (2, 5)])

    def test_analytics_visual_preferences_serialization(self):
        analytics_cfg = self.config_manager.analytics
        analytics_cfg.temporal_trends_visual_var.set("Cumulative Line")
        analytics_cfg.activity_budget_visual_var.set("Stacked + Violin")
        analytics_cfg.bout_timeline_visual_var.set("CSV + Gantt")

        config_dict = self.config_manager._gather_config_as_dict()
        self.assertIn("analytics", config_dict)
        analytics_section = config_dict["analytics"]
        self.assertEqual(analytics_section["temporal_trends_visual_var"], "Cumulative Line")
        self.assertEqual(analytics_section["activity_budget_visual_var"], "Stacked + Violin")
        self.assertEqual(analytics_section["bout_timeline_visual_var"], "CSV + Gantt")

        new_config_manager = ConfigManager(self.app)
        new_config_manager._apply_config_from_dict(config_dict)

        new_analytics_cfg = new_config_manager.analytics
        self.assertEqual(new_analytics_cfg.temporal_trends_visual_var.get(), "Cumulative Line")
        self.assertEqual(new_analytics_cfg.activity_budget_visual_var.get(), "Stacked + Violin")
        self.assertEqual(new_analytics_cfg.bout_timeline_visual_var.get(), "CSV + Gantt")


if __name__ == '__main__':
    unittest.main()

"""Regression coverage for live Tab 7 widget/callback construction."""
import unittest


class TestTab7GuiConstruction(unittest.TestCase):
    def test_embedded_and_standalone_widgets_construct(self):
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError as exc:
            self.skipTest(f'Tk unavailable: {exc}')
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'GUI display unavailable: {exc}')
        root.withdraw()
        try:
            from integra_pose.hmm_vae_toolkit.main import BehaviorAnalysisApp

            def walk(widget):
                yield widget
                for child in widget.winfo_children():
                    yield from walk(child)

            for embed in (True, False):
                with self.subTest(embed=embed):
                    window = tk.Toplevel(root)
                    window.withdraw()
                    try:
                        app = BehaviorAnalysisApp(window, embed=embed)
                        window.update_idletasks()
                        labels = {str(w.cget('text')): w for w in walk(window) if isinstance(w, ttk.Button)}
                        self.assertIn('Run Sub-Behavior Discovery', labels)
                        self.assertIn('Export Keypoints to CSV', labels)
                        self.assertIn('Export Sub-cluster Clips', labels)
                        self.assertNotIn('Export Latent Embeddings (CSV)', labels)
                        visible_text = ' '.join(
                            str(w.cget('text')) for w in walk(window) if 'text' in w.keys()
                        )
                        self.assertNotIn('VAE + HMM', visible_text)
                        help_text = ' '.join(
                            w.get('1.0', tk.END) for w in walk(window) if isinstance(w, tk.Text)
                        )
                        self.assertNotIn('HMM Selector', help_text)
                        self.assertNotIn('VAE Workflow', help_text)
                        for label in ('Review Candidate Sub-Clusters', 'Name Sub-Behaviors...', 'Export Sub-cluster Clips'):
                            self.assertTrue(labels[label].instate(['disabled']), label)
                        self.assertFalse(app.running)
                        self.assertEqual(app.location_mode.get(), 'none')
                        self.assertFalse(hasattr(app, 'cluster_backend'))
                        self.assertEqual(app.min_class_size.get(), '30')
                        app.min_class_size.set('45')
                        params = app.get_all_params()
                        app.min_class_size.set('30')
                        app.set_all_params(params)
                        self.assertEqual(app.min_class_size.get(), '45')
                        legacy = dict(params)
                        legacy['cluster_backend'] = 'auto'
                        legacy.pop('min_class_size')
                        app.set_all_params(legacy)
                        self.assertNotIn('cluster_backend', app.get_all_params())
                        self.assertEqual(app.min_class_size.get(), '30')
                    finally:
                        window.destroy()
        finally:
            root.destroy()


if __name__ == '__main__':
    unittest.main()

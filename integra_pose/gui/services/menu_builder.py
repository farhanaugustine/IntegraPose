"""Menu construction helpers for the main GUI."""

from __future__ import annotations

from tkinter import Menu


class MenuBuilder:
    """Builds the top-level menus and attaches them to the app."""

    def __init__(self, app) -> None:
        self.app = app

    def build(self) -> Menu:
        """Create the menu bar and attach File/Plugins/Help menus."""
        menu_bar = Menu(self.app.root)
        self._build_file_menu(menu_bar)
        self._build_plugins_menu(menu_bar)
        self._build_help_menu(menu_bar)
        self.app.root.config(menu=menu_bar)
        # Expose menus for plugin registration and docs expectations.
        self.app.menu_bar = menu_bar
        self.app.plugins_menu = self.plugins_menu
        return menu_bar

    def _build_file_menu(self, menu_bar: Menu) -> None:
        file_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Project...", command=self.app._safe_open_project)
        file_menu.add_command(label="Save Project", command=self.app._safe_save_project)
        file_menu.add_command(label="Save Project As...", command=self.app._safe_save_project_as)
        file_menu.add_separator()
        file_menu.add_command(label="Export Reproducibility Bundle...", command=self.app._export_repro_bundle)
        file_menu.add_command(label="Import Reproducibility Bundle...", command=self.app._import_repro_bundle)
        file_menu.add_separator()
        file_menu.add_command(label="Start First-Run Onboarding...", command=self.app._start_first_run_onboarding)
        file_menu.add_command(label="Batch Processing Wizard...", command=self.app._safe_open_batch_processing_wizard)

        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.app._on_closing)

    def _build_plugins_menu(self, menu_bar: Menu) -> None:
        self.plugins_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Plugins", menu=self.plugins_menu)

    def _build_help_menu(self, menu_bar: Menu) -> None:
        """Help menu — currently hosts the install sanity check."""
        help_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(
            label="Run Sanity Check...",
            command=self._open_sanity_check,
        )

    def _open_sanity_check(self) -> None:
        """Open the install verification dialog (lazy import keeps Tk start fast)."""
        try:
            from integra_pose.sanity_check.ui import open_sanity_check_dialog

            open_sanity_check_dialog(self.app.root)
        except Exception as exc:
            from tkinter import messagebox

            messagebox.showerror(
                "Sanity Check",
                f"Could not launch the sanity check:\n{exc}",
                parent=self.app.root,
            )

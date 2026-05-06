"""Project open/save helpers decoupled from the main GUI class."""

from __future__ import annotations

import os
from tkinter import filedialog, messagebox


class ProjectIOService:
    """Handles project open/save flows with user prompts and logging."""

    def __init__(self, app) -> None:
        self.app = app

    def open_project(self) -> None:
        try:
            config_data = self.app.config.open_project()
            if config_data:
                self.app._initialize_default_behaviors()
                self.app.performance_metrics_var.set("Idle")
                self.app.set_performance_warning("")
                self.app.update_status("Project opened successfully.")
                self.app.log_message("Project opened successfully.", "INFO")
                messagebox.showinfo("Success", "Project opened successfully.", parent=self.app.root)
        except Exception as exc:
            self.app.log_message(f"Failed to open project: {exc}", "ERROR")
            messagebox.showerror("Project Error", f"Failed to open project: {exc}", parent=self.app.root)

    def save_project(self) -> None:
        try:
            self.app.config.save_project()
            self.app.update_status("Project saved successfully.")
            self.app.log_message("Project saved successfully.", "INFO")
            messagebox.showinfo("Success", "Project saved successfully.", parent=self.app.root)
        except Exception as exc:
            self.app.log_message(f"Failed to save project: {exc}", "ERROR")
            messagebox.showerror("Project Error", f"Failed to save project: {exc}", parent=self.app.root)

    def save_project_as(self) -> None:
        try:
            self.app.config.save_project_as()
            self.app.update_status("Project saved as new successfully.")
            self.app.log_message("Project saved as new successfully.", "INFO")
            messagebox.showinfo("Success", "Project saved as new successfully.", parent=self.app.root)
        except Exception as exc:
            self.app.log_message(f"Failed to save project as: {exc}", "ERROR")
            messagebox.showerror("Project Error", f"Failed to save project as: {exc}", parent=self.app.root)

    def select_directory(self, string_var, title="Select Directory"):
        dir_path = filedialog.askdirectory(title=title, parent=self.app.root)
        if dir_path:
            string_var.set(dir_path)
            self.app.log_message(f"Selected directory: {dir_path}", "INFO")
            return dir_path
        return None

    def select_file(self, string_var, title="Select File", filetypes=(("All files", "*.*"), ("Text files", "*.txt"))):
        file_path = filedialog.askopenfilename(title=title, filetypes=filetypes, parent=self.app.root)
        if file_path:
            string_var.set(file_path)
            self.app.log_message(f"Selected file: {file_path}", "INFO")
            return file_path
        return None

    def select_project_root(self) -> None:
        dir_path = filedialog.askdirectory(title="Select Project Root Directory", parent=self.app.root)
        if not dir_path:
            return
        norm_path = os.path.normpath(dir_path)
        self.app.config.setup.dataset_root_yaml.set(norm_path)

        default_images_all = os.path.join(norm_path, "images_all")
        default_labels_all = os.path.join(norm_path, "labels_all")
        default_images_train = os.path.join(norm_path, "images", "train")
        default_images_val = os.path.join(norm_path, "images", "val")
        default_labels_train = os.path.join(norm_path, "labels", "train")
        default_labels_val = os.path.join(norm_path, "labels", "val")
        default_models_dir = os.path.join(norm_path, "models")

        for folder in (
            default_images_all,
            default_labels_all,
            default_images_train,
            default_images_val,
            default_labels_train,
            default_labels_val,
            default_models_dir,
        ):
            try:
                os.makedirs(folder, exist_ok=True)
            except Exception:
                pass

        self.app.config.setup.image_dir_annot.set(default_images_all)
        self.app.config.setup.annot_output_dir.set(default_labels_all)

        self.app.config.setup.train_image_dir_yaml.set(default_images_train)
        self.app.config.setup.val_image_dir_yaml.set(default_images_val)
        self.app.config.training.model_save_dir.set(default_models_dir)

        try:
            current_name = (self.app.config.setup.dataset_name_yaml.get() or "").strip()
            if not current_name or current_name == "behavior_dataset":
                base_name = os.path.basename(norm_path) or "behavior_dataset"
                self.app.config.setup.dataset_name_yaml.set(base_name)
        except Exception:
            pass

        try:
            if not (self.app.config.data_preprocessing.output_dir.get() or "").strip():
                self.app.config.data_preprocessing.output_dir.set(default_images_all)
            if not (self.app.config.data_preprocessing.transfer_dest_root.get() or "").strip():
                self.app.config.data_preprocessing.transfer_dest_root.set(default_images_all)
        except Exception:
            pass

        self.app.update_status(f"Project root set to {norm_path}. Default dataset folders created.")
        self.app.log_message(f"Project root set to {norm_path}", "INFO")

        refresh = getattr(self.app, "_schedule_dataset_readiness_refresh", None)
        if callable(refresh):
            refresh()

    def select_yolo_output_directory(self) -> None:
        dir_path = filedialog.askdirectory(title="Select Folder with YOLO .txt Files", parent=self.app.root)
        if dir_path:
            txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
            if not txt_files:
                self.app.log_message(f"No .txt files found in selected YOLO output directory: {dir_path}", "WARNING")
                messagebox.showwarning(
                    "Invalid Directory",
                    "The selected directory contains no .txt files. Please select a directory with YOLO detection outputs.",
                    parent=self.app.root,
                )
            else:
                self.app.config.analytics.yolo_output_path_var.set(dir_path)
                self.app.log_message(f"Selected YOLO output directory with {len(txt_files)} .txt files: {dir_path}", "INFO")

    def select_source_video(self) -> None:
        filetypes = (("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        path = filedialog.askopenfilename(title="Select Source Video", filetypes=filetypes, parent=self.app.root)
        if path:
            self.app.config.analytics.source_video_path_var.set(path)
            self.app.log_message(f"Source video selected: {path}", "INFO")

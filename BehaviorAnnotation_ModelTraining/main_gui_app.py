import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from collections import OrderedDict
import subprocess


from gui import setup_tab, training_tab, inference_tab, webcam_tab
from utils import command_builder
from utils.webcam_manager import WebcamManager

try:
    from keypoint_annotator_module import KeypointBehaviorAnnotator
except ImportError:
    messagebox.showerror("Import Error", "Could not import KeypointBehaviorAnnotator. Make sure keypoint_annotator_module.py is in the same directory.")
    sys.exit(1)

class YoloApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("YOLO Annotation & Training GUI")
        self.root.geometry("900x850")
        self.root.minsize(850, 750)

        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except tk.TclError:
            self.style.theme_use('default')

        # --- Initialize all state-holding variables ---
        self._initialize_variables()

        # --- UI Setup ---
        self.notebook = ttk.Notebook(self.root)
        self.setup_tab = ttk.Frame(self.notebook, padding="10")
        self.training_tab = ttk.Frame(self.notebook, padding="10")
        self.inference_tab = ttk.Frame(self.notebook, padding="10")
        # Here is the new tab for the webcam
        self.webcam_tab = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.setup_tab, text='1. Setup & Annotation')
        self.notebook.add(self.training_tab, text='2. Model Training')
        self.notebook.add(self.inference_tab, text='3. Inference')
        # Add the new tab to the GUI
        self.notebook.add(self.webcam_tab, text='4. Webcam Inference')
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)

        # Call the functions from our gui package to build the UI for each tab.
        setup_tab.create_setup_tab(self)
        training_tab.create_training_tab(self)
        inference_tab.create_inference_tab(self)
        # Create the new webcam tab UI
        webcam_tab.create_webcam_tab(self)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        self._initialize_default_behaviors()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def _initialize_variables(self):
        # This is now a central place for all tk variables.
        # --- Paths & General ---
        self.image_dir_annot = tk.StringVar()
        self.train_image_dir_yaml = tk.StringVar()
        self.val_image_dir_yaml = tk.StringVar()
        self.dataset_root_yaml = tk.StringVar()
        self.annot_output_dir = tk.StringVar()
        self.model_save_dir = tk.StringVar()
        self.dataset_name_yaml = tk.StringVar(value="behavior_dataset")
        self.keypoint_names_str = tk.StringVar(value="nose,left_eye,right_eye,left_ear,right_ear")
        self.behaviors_list = []
        self.current_behavior_id_entry = tk.StringVar()
        self.current_behavior_name_entry = tk.StringVar()
        self.annotator_instance = None
        
        # --- Training ---
        self.dataset_yaml_path_train = tk.StringVar()
        self.epochs_var = tk.StringVar(value="100"); self.lr_var = tk.StringVar(value="0.01")
        self.batch_var = tk.StringVar(value="16"); self.imgsz_var = tk.StringVar(value="640")
        self.model_variant_var = tk.StringVar(value="yolov8n-pose.pt"); self.run_name_var = tk.StringVar(value="keypoint_behavior_run1")
        self.optimizer_var = tk.StringVar(value="AdamW"); self.weight_decay_var = tk.StringVar(value="0.0005")
        self.label_smoothing_var = tk.StringVar(value="0.0"); self.patience_var = tk.StringVar(value="50")
        self.device_var = tk.StringVar(value="cpu")
        self.hsv_h_var = tk.StringVar(value="0.015"); self.hsv_s_var = tk.StringVar(value="0.7"); self.hsv_v_var = tk.StringVar(value="0.4")
        self.degrees_var = tk.StringVar(value="0.0"); self.translate_var = tk.StringVar(value="0.1"); self.scale_var = tk.StringVar(value="0.5")
        self.shear_var = tk.StringVar(value="0.0"); self.perspective_var = tk.StringVar(value="0.0"); self.flipud_var = tk.StringVar(value="0.0")
        self.fliplr_var = tk.StringVar(value="0.5"); self.mixup_var = tk.StringVar(value="0.0"); self.copy_paste_var = tk.StringVar(value="0.0")
        self.mosaic_var = tk.StringVar(value="1.0")

        # --- File Inference ---
        self.trained_model_path_infer = tk.StringVar(); self.video_infer_path = tk.StringVar()
        self.tracker_config_path = tk.StringVar(); self.conf_thres_var = tk.StringVar(value="0.25")
        self.iou_thres_var = tk.StringVar(value="0.45"); self.infer_imgsz_var = tk.StringVar(value="640")
        self.infer_device_var = tk.StringVar(value="cpu"); self.infer_max_det_var = tk.StringVar(value="300")
        self.infer_line_width_var = tk.StringVar(value=""); self.infer_project_var = tk.StringVar(value="")
        self.infer_name_var = tk.StringVar(value=""); self.use_tracker_var = tk.BooleanVar(value=False)
        self.show_infer_var = tk.BooleanVar(value=True); self.infer_save_var = tk.BooleanVar(value=True)
        self.infer_save_txt_var = tk.BooleanVar(value=False); self.infer_save_conf_var = tk.BooleanVar(value=False)
        self.infer_save_crop_var = tk.BooleanVar(value=False); self.infer_hide_labels_var = tk.BooleanVar(value=False)
        self.infer_hide_conf_var = tk.BooleanVar(value=False); self.infer_augment_var = tk.BooleanVar(value=False)
        self.inference_process = None

        # --- Webcam Inference ---
        self.webcam_model_path = tk.StringVar()
        self.webcam_index_var = tk.StringVar(value="0")
        self.webcam_mode_var = tk.StringVar(value="track")
        
        self.webcam_max_det_var = tk.StringVar(value="10")
        self.webcam_save_annotated_var = tk.BooleanVar(value=True)
        self.webcam_save_txt_var = tk.BooleanVar(value=False)
        self.webcam_save_raw_var = tk.BooleanVar(value=True)
        self.webcam_inference_process = None
        self.webcam_manager = None

    # --- GUI Helper Methods  ---
    def _select_directory(self, string_var, title="Select Directory"):
        dir_path = filedialog.askdirectory(title=title, parent=self.root)
        if dir_path: string_var.set(dir_path)

    def _select_file(self, string_var, title="Select File", filetypes=(("All files", "*.*"),)):
        file_path = filedialog.askopenfilename(title=title, filetypes=filetypes, parent=self.root)
        if file_path: string_var.set(file_path)

    # --- Behavior List Methods  ---
    def _initialize_default_behaviors(self):
        default_behaviors = [{'id': 0, 'name': 'Standing'}, {'id': 1, 'name': 'Walking'}]
        for b in default_behaviors: self.behaviors_list.append(b)
        self._refresh_behavior_listbox_display()
    def _refresh_behavior_listbox_display(self):
        self.behavior_list_display.delete(0, tk.END)
        self.behaviors_list.sort(key=lambda b: b['id'])
        for behavior in self.behaviors_list: self.behavior_list_display.insert(tk.END, f"{behavior['id']}: {behavior['name']}")
        self._clear_behavior_input_fields()
    def _on_behavior_select(self, event):
        if not event.widget.curselection(): return
        selected = self.behaviors_list[event.widget.curselection()[0]]
        self.current_behavior_id_entry.set(str(selected['id']))
        self.current_behavior_name_entry.set(selected['name'])
    def _add_update_behavior(self):
        try: b_id = int(self.current_behavior_id_entry.get())
        except ValueError: messagebox.showerror("Input Error", "ID must be integer."); return
        b_name = self.current_behavior_name_entry.get().strip()
        if not b_name: messagebox.showerror("Input Error", "Name cannot be empty."); return
        existing_b = next((b for b in self.behaviors_list if b['id'] == b_id), None)
        if existing_b: existing_b['name'] = b_name
        else: self.behaviors_list.append({'id': b_id, 'name': b_name})
        self._refresh_behavior_listbox_display()
    def _remove_selected_behavior(self):
        if not self.behavior_list_display.curselection(): return
        self.behaviors_list.pop(self.behavior_list_display.curselection()[0])
        self._refresh_behavior_listbox_display()
    def _clear_behavior_input_fields(self):
        self.current_behavior_id_entry.set("")
        self.current_behavior_name_entry.set("")
        if self.behavior_list_display.curselection(): self.behavior_list_display.selection_clear(0, tk.END)

    # --- Core Functionality Methods ---
    def _launch_annotator(self):
        # I am adding these calls to ensure no inference processes are running.
        self._stop_inference()
        self._stop_webcam_inference()

        kp_names = [n.strip() for n in self.keypoint_names_str.get().split(',') if n.strip()]
        if not kp_names: messagebox.showerror("Error", "Valid keypoint names are required.", parent=self.root); return
        if not self.behaviors_list: messagebox.showerror("Error", "Behaviors list cannot be empty.", parent=self.root); return
        available_behaviors = OrderedDict((b['id'], b['name']) for b in sorted(self.behaviors_list, key=lambda b: b['id']))
        img_dir = self.image_dir_annot.get()
        ann_dir = self.annot_output_dir.get() or "./labels_output_default"
        if not os.path.isdir(img_dir): messagebox.showerror("Error", "Valid Image Directory for annotation is required.", parent=self.root); return
        try:
            self.annotator_instance = KeypointBehaviorAnnotator(self.root, kp_names, available_behaviors, img_dir, ann_dir)
            self.root.withdraw()
            self.annotator_instance.run()
        except Exception as e:
            messagebox.showerror("Annotator Error", f"Error: {e}", parent=self.root)
        finally:
            if self.root.winfo_exists(): self.root.deiconify(); self.root.focus_force()

    def _generate_yaml_from_gui(self):
        
        if self.annotator_instance is None:
            kp_names = [n.strip() for n in self.keypoint_names_str.get().split(',') if n.strip()]
            if not kp_names:
                messagebox.showerror("Error", "Valid keypoint names are required to generate YAML.", parent=self.root)
                return
            if not self.behaviors_list:
                messagebox.showerror("Error", "Behaviors list cannot be empty to generate YAML.", parent=self.root)
                return

            available_behaviors = OrderedDict((b['id'], b['name']) for b in sorted(self.behaviors_list, key=lambda b: b['id']))
            
            try:
                # The annotator will be created without a parent tk_window for this action.
                temp_annotator = KeypointBehaviorAnnotator(self.root, kp_names, available_behaviors, image_dir=".", output_dir=".")
                self.annotator_instance = temp_annotator
            except FileNotFoundError:
                # This can happen if the dummy image_dir="." has no images. The class is robust enough to handle this for YAML generation.
                pass
            except Exception as e:
                messagebox.showerror("Initialization Error", f"Could not prepare for YAML generation: {e}", parent=self.root)
                return

    dataset_root = self.dataset_root_yaml.get()
    train_images = self.train_image_dir_yaml.get()
    val_images = self.val_image_dir_yaml.get()
    dataset_name = self.dataset_name_yaml.get()

    
    if self.annotator_instance:
        self.annotator_instance.generate_config_yaml(
            dataset_root_path=dataset_root,
            train_images_path=train_images,
            val_images_path=val_images,
            dataset_name_suggestion=dataset_name
        )
    else:
        messagebox.showerror("Error", "Annotator instance is not available. Please try launching the annotator at least once.", parent=self.root)
        

    def _start_training(self):
        cmd, err = command_builder.build_training_command(self)
        if err: messagebox.showerror("Error", err, parent=self.root); return
        print(f"Executing Training Command:\n{' '.join(cmd)}")
        try:
            subprocess.Popen(cmd, shell=sys.platform == "win32")
            messagebox.showinfo("Training Started", "Check console for output.", parent=self.root)
        except Exception as e: messagebox.showerror("Execution Error", str(e), parent=self.root)

    def _run_inference(self):
        if self.inference_process and self.inference_process.poll() is None: return
        # Ensure other processes are stopped first.
        self._stop_webcam_inference()
        
        cmd, err = command_builder.build_inference_command(self)
        if err: messagebox.showerror("Error", err, parent=self.root); return
        print(f"Executing Inference Command:\n{' '.join(cmd)}")
        try:
            self.inference_process = subprocess.Popen(cmd, shell=sys.platform == "win32")
            self.run_inference_button.config(state=tk.DISABLED)
            self.stop_inference_button.config(state=tk.NORMAL)
            self._check_file_inference_process()
        except Exception as e: messagebox.showerror("Execution Error", str(e), parent=self.root)
        
    def _stop_inference(self):
        if self.inference_process and self.inference_process.poll() is None:
            self.inference_process.terminate()
            print("File inference process terminated.")
        self.inference_process = None
        self.run_inference_button.config(state=tk.NORMAL)
        self.stop_inference_button.config(state=tk.DISABLED)
        
    def _check_file_inference_process(self):
        if self.inference_process and self.inference_process.poll() is not None:
            print(f"File inference process finished with code: {self.inference_process.poll()}.")
            self._stop_inference() # This will reset the buttons.
        elif self.inference_process:
            self.root.after(1000, self._check_file_inference_process)

    def _start_webcam_inference(self):
        if self.webcam_inference_process and self.webcam_inference_process.poll() is None:
            messagebox.showwarning("Inference Running", "A webcam process is already running.", parent=self.root)
            return
        
        # Ensure other processes are stopped first.
        self._stop_inference()

        if self.webcam_save_raw_var.get():
            cam_idx = int(self.webcam_index_var.get())
            self.webcam_manager = WebcamManager(camera_index=cam_idx)
            if not self.webcam_manager.start_capture():
                messagebox.showerror("Webcam Error", f"Could not start raw capture from camera index {cam_idx}. Check if it is connected and not in use.", parent=self.root)
                self.webcam_manager = None
                return

        cmd, err = command_builder.build_webcam_command(self)
        if err:
            messagebox.showerror("Error", err, parent=self.root)
            if self.webcam_manager: self.webcam_manager.stop_capture()
            return

        print(f"Executing Webcam Command:\n{' '.join(cmd)}")
        try:
            self.webcam_inference_process = subprocess.Popen(cmd, shell=sys.platform == "win32")
            self.start_webcam_button.config(state=tk.DISABLED)
            self.stop_webcam_button.config(state=tk.NORMAL)
            self._check_webcam_process()
        except Exception as e:
            messagebox.showerror("Execution Error", str(e), parent=self.root)
            if self.webcam_manager: self.webcam_manager.stop_capture()
            self.webcam_manager = None
            
    def _stop_webcam_inference(self):
        print("Stopping webcam inference...")
        if self.webcam_inference_process and self.webcam_inference_process.poll() is None:
            self.webcam_inference_process.terminate()
            print("Webcam inference process terminated.")
        if self.webcam_manager:
            self.webcam_manager.stop_capture()
        self.webcam_inference_process = None
        self.webcam_manager = None
        self.start_webcam_button.config(state=tk.NORMAL)
        self.stop_webcam_button.config(state=tk.DISABLED)

    def _check_webcam_process(self):
        if self.webcam_inference_process and self.webcam_inference_process.poll() is not None:
            print("YOLO webcam window closed by user. Stopping all processes.")
            self._stop_webcam_inference() 
        elif self.webcam_inference_process:
            self.root.after(1000, self._check_webcam_process)

    def _on_closing(self):
        """Handle cleanup when the main window is closed."""
        print("Application closing...")
        self._stop_inference()
        self._stop_webcam_inference()
        self.root.destroy()


if __name__ == '__main__':
    main_tk_root = tk.Tk()
    app = YoloApp(main_tk_root)
    main_tk_root.mainloop()
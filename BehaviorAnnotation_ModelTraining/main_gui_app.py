# main_gui_app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from collections import OrderedDict
import subprocess

try:
    from keypoint_annotator_module import KeypointBehaviorAnnotator
except ImportError:
    messagebox.showerror("Import Error", "Could not import KeypointBehaviorAnnotator from keypoint_annotator_module.py. Make sure the file is in the same directory.")
    sys.exit(1)

class YoloApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("YOLO Annotation & Training GUI")
        self.root.geometry("900x850") # Increased size for more params
        self.root.minsize(850, 750)

        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except tk.TclError:
            self.style.theme_use('default')

        # --- Data storage (existing variables) ---
        self.image_dir_annot = tk.StringVar()
        self.train_image_dir_yaml = tk.StringVar()
        self.val_image_dir_yaml = tk.StringVar()
        self.dataset_root_yaml = tk.StringVar()
        self.dataset_name_yaml = tk.StringVar(value="behavior_dataset")
        self.annot_output_dir = tk.StringVar()
        self.model_save_dir = tk.StringVar() # For training project output
        self.dataset_yaml_path_train = tk.StringVar() # YAML path for training tab
        self.trained_model_path_infer = tk.StringVar()
        self.video_infer_path = tk.StringVar()
        self.keypoint_names_str = tk.StringVar(value="nose,left_eye,right_eye,left_ear,right_ear")
        self.behaviors_list = []
        self.current_behavior_id_entry = tk.StringVar()
        self.current_behavior_name_entry = tk.StringVar()
        self.annotator_instance = None

        # --- Basic Training Parameters ---
        self.epochs_var = tk.StringVar(value="100")
        self.lr_var = tk.StringVar(value="0.01") # lr0
        self.batch_var = tk.StringVar(value="16")
        self.imgsz_var = tk.StringVar(value="640")
        self.model_variant_var = tk.StringVar(value="yolov8n-pose.pt") # Pretrained weights
        self.run_name_var = tk.StringVar(value="keypoint_behavior_run1") # Experiment name

        # --- Advanced Training Parameters ---
        self.optimizer_var = tk.StringVar(value="AdamW") # SGD, Adam, AdamW
        self.weight_decay_var = tk.StringVar(value="0.0005")
        self.label_smoothing_var = tk.StringVar(value="0.0")
        self.patience_var = tk.StringVar(value="50") # For early stopping
        self.device_var = tk.StringVar(value="cpu") # e.g., cpu, 0, 0,1

        # --- Augmentation Parameters ---
        self.hsv_h_var = tk.StringVar(value="0.015") # hue
        self.hsv_s_var = tk.StringVar(value="0.7")   # saturation
        self.hsv_v_var = tk.StringVar(value="0.4")   # value
        self.degrees_var = tk.StringVar(value="0.0")    # rotation
        self.translate_var = tk.StringVar(value="0.1")
        self.scale_var = tk.StringVar(value="0.5")      # zoom factor
        self.shear_var = tk.StringVar(value="0.0")
        self.perspective_var = tk.StringVar(value="0.0")
        self.flipud_var = tk.StringVar(value="0.0")     # flip up-down probability
        self.fliplr_var = tk.StringVar(value="0.5")     # flip left-right probability
        self.mixup_var = tk.StringVar(value="0.0")      # mixup probability
        self.copy_paste_var = tk.StringVar(value="0.0") # copy-paste probability
        self.mosaic_var = tk.StringVar(value="1.0")     # mosaic probability (applied for all but last X epochs)

        # --- Inference Parameters (Existing + New) ---
        self.conf_thres_var = tk.StringVar(value="0.25")         # conf
        self.iou_thres_var = tk.StringVar(value="0.45")          # iou
        self.show_infer_var = tk.BooleanVar(value=True)          # show (display window)
        self.infer_save_var = tk.BooleanVar(value=True)          # save (save images/videos with annotations)

        self.infer_imgsz_var = tk.StringVar(value="640")         # imgsz
        self.infer_device_var = tk.StringVar(value="cpu")        # device
        self.infer_save_txt_var = tk.BooleanVar(value=False)     # save_txt
        self.infer_save_conf_var = tk.BooleanVar(value=False)    # save_conf (effective with save_txt)
        self.infer_save_crop_var = tk.BooleanVar(value=False)    # save_crop
        self.infer_hide_labels_var = tk.BooleanVar(value=False)  # hide_labels
        self.infer_hide_conf_var = tk.BooleanVar(value=False)    # hide_conf
        self.infer_line_width_var = tk.StringVar(value="")       # line_thickness (empty for YOLO default)
        self.infer_augment_var = tk.BooleanVar(value=False)      # augment
        self.infer_max_det_var = tk.StringVar(value="300")       # max_det
        self.infer_project_var = tk.StringVar(value="")          # project (empty for YOLO default 'runs/detect')
        self.infer_name_var = tk.StringVar(value="")             # name (empty for YOLO default 'predict', 'predict2')


        self.notebook = ttk.Notebook(self.root)
        self.setup_tab = ttk.Frame(self.notebook, padding="10")
        self.training_tab = ttk.Frame(self.notebook, padding="10")
        self.inference_tab = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.setup_tab, text='1. Setup & Annotation')
        self.notebook.add(self.training_tab, text='2. Model Training')
        self.notebook.add(self.inference_tab, text='3. Inference')
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)

        self._create_setup_tab()
        self._create_training_tab()
        self._create_inference_tab()
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        self._initialize_default_behaviors()


    def _initialize_default_behaviors(self):
        default_behaviors_data = [
            {'id': 0, 'name': 'Standing'}, {'id': 1, 'name': 'Walking'}, {'id': 2, 'name': 'Running'}
        ]
        for b_data in default_behaviors_data: self.behaviors_list.append(b_data)
        self._refresh_behavior_listbox_display()

    def _select_directory(self, string_var, title="Select Directory"):
        dir_path = filedialog.askdirectory(title=title, parent=self.root)
        if dir_path: string_var.set(dir_path)

    def _select_file(self, string_var, title="Select File", filetypes=(("All files", "*.*"),)):
        file_path = filedialog.askopenfilename(title=title, filetypes=filetypes, parent=self.root)
        if file_path: string_var.set(file_path)

    def _create_setup_tab(self):
        main_setup_frame = ttk.Frame(self.setup_tab)
        main_setup_frame.pack(expand=True, fill="both")
        annot_paths_frame = ttk.LabelFrame(main_setup_frame, text="Annotation Paths & Source", padding="10")
        annot_paths_frame.pack(fill="x", pady=5, padx=5)
        ttk.Label(annot_paths_frame, text="Image Directory (for annotation):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(annot_paths_frame, textvariable=self.image_dir_annot, width=60).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(annot_paths_frame, text="Browse...", command=lambda: self._select_directory(self.image_dir_annot, "Select Image Directory for Annotation")).grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(annot_paths_frame, text="Annotation Output (.txt labels):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(annot_paths_frame, textvariable=self.annot_output_dir, width=60).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(annot_paths_frame, text="Browse...", command=lambda: self._select_directory(self.annot_output_dir, "Select Annotation Output Directory")).grid(row=1, column=2, padx=5, pady=5)
        annot_paths_frame.columnconfigure(1, weight=1)
        kp_defs_frame = ttk.LabelFrame(main_setup_frame, text="Keypoint Definitions", padding="10")
        kp_defs_frame.pack(fill="x", pady=5, padx=5)
        kp_defs_frame.columnconfigure(1, weight=1)
        ttk.Label(kp_defs_frame, text="Keypoint Names (comma-separated):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(kp_defs_frame, textvariable=self.keypoint_names_str, width=70).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        behavior_defs_frame = ttk.LabelFrame(main_setup_frame, text="Behavior Definitions", padding="10")
        behavior_defs_frame.pack(fill="both", expand=True, pady=5, padx=5)
        behavior_defs_frame.columnconfigure(1, weight=3)
        behavior_defs_frame.columnconfigure(2, weight=1)
        ttk.Label(behavior_defs_frame, text="Defined Behaviors (ID: Name):").grid(row=0, column=0, columnspan=3, padx=5, pady=2, sticky="w")
        self.behavior_list_display = tk.Listbox(behavior_defs_frame, height=8, relief=tk.SOLID, borderwidth=1, exportselection=False)
        self.behavior_list_display.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        self.behavior_list_display.bind('<<ListboxSelect>>', self._on_behavior_select)
        behavior_defs_frame.rowconfigure(1, weight=1)
        input_fields_frame = ttk.Frame(behavior_defs_frame)
        input_fields_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        input_fields_frame.columnconfigure(1, weight=1)
        input_fields_frame.columnconfigure(3, weight=1)
        ttk.Label(input_fields_frame, text="ID:").grid(row=0, column=0, padx=(0,2), pady=5, sticky="w")
        id_entry = ttk.Entry(input_fields_frame, textvariable=self.current_behavior_id_entry, width=5)
        id_entry.grid(row=0, column=1, padx=(0,10), pady=5, sticky="w")
        ttk.Label(input_fields_frame, text="Name:").grid(row=0, column=2, padx=(0,2), pady=5, sticky="w")
        name_entry = ttk.Entry(input_fields_frame, textvariable=self.current_behavior_name_entry, width=25)
        name_entry.grid(row=0, column=3, padx=(0,5), pady=5, sticky="ew")
        buttons_frame = ttk.Frame(behavior_defs_frame)
        buttons_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        buttons_frame.columnconfigure(0, weight=1) 
        buttons_frame.columnconfigure(1, weight=1)
        buttons_frame.columnconfigure(2, weight=1)
        btn_add_update = ttk.Button(buttons_frame, text="Add / Update Behavior", command=self._add_update_behavior)
        btn_add_update.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        btn_remove = ttk.Button(buttons_frame, text="Remove Selected", command=self._remove_selected_behavior)
        btn_remove.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        btn_clear_fields = ttk.Button(buttons_frame, text="Clear Fields", command=self._clear_behavior_input_fields)
        btn_clear_fields.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        yaml_paths_frame = ttk.LabelFrame(main_setup_frame, text="Dataset YAML Configuration", padding="10")
        yaml_paths_frame.pack(fill="x", pady=5, padx=5)
        ttk.Label(yaml_paths_frame, text="Dataset Name (for YAML file):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(yaml_paths_frame, textvariable=self.dataset_name_yaml, width=60).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(yaml_paths_frame, text="Dataset Root Path (for YAML & structure):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(yaml_paths_frame, textvariable=self.dataset_root_yaml, width=60).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(yaml_paths_frame, text="Browse...", command=lambda: self._select_directory(self.dataset_root_yaml, "Select Dataset Root Directory")).grid(row=1, column=2, padx=5, pady=5)
        ttk.Label(yaml_paths_frame, text="Training Images Directory:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(yaml_paths_frame, textvariable=self.train_image_dir_yaml, width=60).grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(yaml_paths_frame, text="Browse...", command=lambda: self._select_directory(self.train_image_dir_yaml, "Select Training Images Directory")).grid(row=2, column=2, padx=5, pady=5)
        ttk.Label(yaml_paths_frame, text="Validation Images Directory:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(yaml_paths_frame, textvariable=self.val_image_dir_yaml, width=60).grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(yaml_paths_frame, text="Browse...", command=lambda: self._select_directory(self.val_image_dir_yaml, "Select Validation Images Directory")).grid(row=3, column=2, padx=5, pady=5)
        yaml_paths_frame.columnconfigure(1, weight=1)
        actions_frame = ttk.Frame(main_setup_frame, padding="10")
        actions_frame.pack(fill="x", pady=10, padx=5)
        self.style.configure("Accent.TButton", foreground="black", font=('Helvetica', 10, 'bold'))
        button_sub_frame = ttk.Frame(actions_frame)
        button_sub_frame.pack()
        btn_launch = ttk.Button(button_sub_frame, text="Launch Annotator", command=self._launch_annotator, style="Accent.TButton")
        btn_launch.pack(side="left", padx=10, ipady=2)
        btn_yaml = ttk.Button(button_sub_frame, text="Generate Dataset YAML", command=self._generate_yaml_from_gui, style="Accent.TButton")
        btn_yaml.pack(side="left", padx=10, ipady=2)
        main_setup_frame.columnconfigure(0, weight=1)
        main_setup_frame.rowconfigure(2, weight=1) # Behavior defs frame

    def _clear_behavior_input_fields(self):
        self.current_behavior_id_entry.set("")
        self.current_behavior_name_entry.set("")
        if self.behavior_list_display.curselection(): self.behavior_list_display.selection_clear(0, tk.END)

    def _refresh_behavior_listbox_display(self):
        self.behavior_list_display.delete(0, tk.END)
        self.behaviors_list.sort(key=lambda b: b['id'])
        for behavior in self.behaviors_list: self.behavior_list_display.insert(tk.END, f"{behavior['id']}: {behavior['name']}")
        self._clear_behavior_input_fields()

    def _on_behavior_select(self, event):
        widget = event.widget
        if not widget.curselection(): return
        selected_index = widget.curselection()[0]
        try:
            selected_behavior_data = self.behaviors_list[selected_index]
            self.current_behavior_id_entry.set(str(selected_behavior_data['id']))
            self.current_behavior_name_entry.set(selected_behavior_data['name'])
        except IndexError: self._clear_behavior_input_fields()

    def _add_update_behavior(self):
        try:
            b_id_str = self.current_behavior_id_entry.get().strip()
            if not b_id_str: messagebox.showerror("Input Error", "Behavior ID cannot be empty.", parent=self.root); return
            b_id = int(b_id_str)
        except ValueError: messagebox.showerror("Input Error", "Behavior ID must be an integer.", parent=self.root); return
        b_name = self.current_behavior_name_entry.get().strip()
        if not b_name: messagebox.showerror("Input Error", "Behavior Name cannot be empty.", parent=self.root); return
        for i, behavior in enumerate(self.behaviors_list):
            if behavior['name'] == b_name and behavior['id'] != b_id:
                messagebox.showerror("Input Error", f"Behavior name '{b_name}' already exists with ID {behavior['id']}.", parent=self.root); return
        found_existing_idx = -1
        for i, behavior in enumerate(self.behaviors_list):
            if behavior['id'] == b_id: found_existing_idx = i; break
        if found_existing_idx != -1: self.behaviors_list[found_existing_idx]['name'] = b_name
        else: self.behaviors_list.append({'id': b_id, 'name': b_name})
        self._refresh_behavior_listbox_display()

    def _remove_selected_behavior(self):
        if not self.behavior_list_display.curselection(): messagebox.showinfo("Info", "No behavior selected to remove.", parent=self.root); return
        selected_index = self.behavior_list_display.curselection()[0]
        if 0 <= selected_index < len(self.behaviors_list): self.behaviors_list.pop(selected_index); self._refresh_behavior_listbox_display()
        else: messagebox.showerror("Error", "Selection is out of sync.", parent=self.root)

    def _launch_annotator(self):
        kp_names_raw = self.keypoint_names_str.get().strip()
        if not kp_names_raw: messagebox.showerror("Error", "Keypoint names cannot be empty.", parent=self.root); return
        kp_names = [name.strip() for name in kp_names_raw.split(',') if name.strip()]
        if not kp_names: messagebox.showerror("Error", "Valid keypoint names are required.", parent=self.root); return
        if not self.behaviors_list: messagebox.showerror("Error", "Behaviors list cannot be empty.", parent=self.root); return
        available_behaviors_ordered = OrderedDict()
        self.behaviors_list.sort(key=lambda b: b['id']) 
        for b_data in self.behaviors_list: available_behaviors_ordered[b_data['id']] = b_data['name']
        img_dir = self.image_dir_annot.get()
        ann_dir = self.annot_output_dir.get()
        if not img_dir or not os.path.isdir(img_dir): messagebox.showerror("Error", "Valid Image Directory for annotation is required.", parent=self.root); return
        if not ann_dir: ann_dir = "./labels_output_default"; messagebox.showinfo("Info", f"Annotation Output Dir not set. Defaulting to '{os.path.abspath(ann_dir)}'", parent=self.root)
        try:
            self.annotator_instance = KeypointBehaviorAnnotator(self.root, kp_names, available_behaviors_ordered, img_dir, ann_dir)
            self.root.withdraw()
            self.annotator_instance.run()
        except Exception as e: messagebox.showerror("Annotator Error", f"Error: {e}", parent=self.root); print(f"Annotator error: {e}"); import traceback; traceback.print_exc()
        finally:
            if self.root.winfo_exists(): self.root.deiconify(); self.root.focus_force()

    def _generate_yaml_from_gui(self):
        kp_names_raw = self.keypoint_names_str.get().strip()
        if not kp_names_raw: messagebox.showerror("Error", "Keypoint names cannot be empty for YAML.", parent=self.root); return
        kp_names = [name.strip() for name in kp_names_raw.split(',') if name.strip()]
        if not kp_names: messagebox.showerror("Error", "Valid keypoint names are required for YAML.", parent=self.root); return
        if not self.behaviors_list: messagebox.showerror("Error", "Behaviors list cannot be empty for YAML.", parent=self.root); return
        available_behaviors_ordered = OrderedDict()
        self.behaviors_list.sort(key=lambda b: b['id'])
        for b_data in self.behaviors_list: available_behaviors_ordered[b_data['id']] = b_data['name']
        ds_root_path = self.dataset_root_yaml.get().strip()
        train_img_path = self.train_image_dir_yaml.get().strip()
        val_img_path = self.val_image_dir_yaml.get().strip()
        ds_name = self.dataset_name_yaml.get().strip()
        if not ds_root_path or not os.path.isdir(ds_root_path): messagebox.showerror("Input Error", "Valid Dataset Root Path required.", parent=self.root); return
        if not train_img_path or not os.path.isdir(train_img_path): messagebox.showerror("Input Error", "Valid Training Images Directory required.", parent=self.root); return
        if not val_img_path or not os.path.isdir(val_img_path): messagebox.showerror("Input Error", "Valid Validation Images Directory required.", parent=self.root); return
        if not ds_name: messagebox.showerror("Input Error", "Dataset Name required.", parent=self.root); return
        annot_img_dir_for_init = self.image_dir_annot.get() if self.image_dir_annot.get() and os.path.isdir(self.image_dir_annot.get()) else "." 
        annot_output_dir_for_init = self.annot_output_dir.get() if self.annot_output_dir.get() else "./labels_temp"
        try:
            temp_annotator = KeypointBehaviorAnnotator(self.root, kp_names, available_behaviors_ordered, annot_img_dir_for_init, annot_output_dir_for_init)
            saved_yaml_path = temp_annotator.generate_config_yaml(ds_root_path, train_img_path, val_img_path, ds_name)
            if saved_yaml_path: self.dataset_yaml_path_train.set(saved_yaml_path); messagebox.showinfo("Success", f"YAML generated: {saved_yaml_path}", parent=self.root)
        except Exception as e: messagebox.showerror("YAML Gen Error", f"Error: {e}", parent=self.root); print(f"YAML gen error: {e}"); import traceback; traceback.print_exc()

    def _create_training_tab(self):
        frame = ttk.Frame(self.training_tab, padding="10") 
        frame.pack(expand=True, fill="both")
        
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=(10,0)) 

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        current_row = 0
        scrollable_frame.columnconfigure(1, weight=1) 

        basic_cfg_frame = ttk.LabelFrame(scrollable_frame, text="Basic Configuration", padding="10")
        basic_cfg_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        basic_cfg_frame.columnconfigure(1, weight=1)
        current_row += 1

        ttk.Label(basic_cfg_frame, text="Dataset Config YAML:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(basic_cfg_frame, textvariable=self.dataset_yaml_path_train, width=50).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(basic_cfg_frame, text="Browse...", command=lambda: self._select_file(self.dataset_yaml_path_train, "Select Dataset YAML", (("YAML files", "*.yaml *.yml"),))).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(basic_cfg_frame, text="Model Save Directory (Project Root):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(basic_cfg_frame, textvariable=self.model_save_dir, width=50).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(basic_cfg_frame, text="Browse...", command=lambda: self._select_directory(self.model_save_dir, "Select Model Save Directory (Project)")).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(basic_cfg_frame, text="Pretrained Weights (.pt):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(basic_cfg_frame, textvariable=self.model_variant_var, width=40).grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        ttk.Button(basic_cfg_frame, text="Browse PT...", command=lambda: self._select_file(self.model_variant_var, "Select Pretrained Model Weights", (("PyTorch Models", "*.pt"),))).grid(row=2, column=2, padx=5, pady=2)
        
        ttk.Label(basic_cfg_frame, text="Experiment Run Name:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(basic_cfg_frame, textvariable=self.run_name_var, width=40).grid(row=3, column=1, padx=5, pady=2, sticky="ew")

        train_hparams_frame = ttk.LabelFrame(scrollable_frame, text="General Training Hyperparameters", padding="10")
        train_hparams_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        current_row += 1
        
        col1_idx, col2_idx, col3_idx, col4_idx = 0,1,2,3 

        ttk.Label(train_hparams_frame, text="Epochs:").grid(row=0, column=col1_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(train_hparams_frame, textvariable=self.epochs_var, width=12).grid(row=0, column=col2_idx, padx=5, pady=2, sticky="ew")
        
        ttk.Label(train_hparams_frame, text="Initial LR (lr0):").grid(row=0, column=col3_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(train_hparams_frame, textvariable=self.lr_var, width=12).grid(row=0, column=col4_idx, padx=5, pady=2, sticky="ew")

        ttk.Label(train_hparams_frame, text="Batch Size:").grid(row=1, column=col1_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(train_hparams_frame, textvariable=self.batch_var, width=12).grid(row=1, column=col2_idx, padx=5, pady=2, sticky="ew")
        
        ttk.Label(train_hparams_frame, text="Image Size (imgsz):").grid(row=1, column=col3_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(train_hparams_frame, textvariable=self.imgsz_var, width=12).grid(row=1, column=col4_idx, padx=5, pady=2, sticky="ew")

        ttk.Label(train_hparams_frame, text="Optimizer:").grid(row=2, column=col1_idx, padx=5, pady=2, sticky="w")
        optimizer_combo = ttk.Combobox(train_hparams_frame, textvariable=self.optimizer_var, values=['AdamW', 'Adam', 'SGD'], width=10)
        optimizer_combo.grid(row=2, column=col2_idx, padx=5, pady=2, sticky="ew")
        
        ttk.Label(train_hparams_frame, text="Weight Decay:").grid(row=2, column=col3_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(train_hparams_frame, textvariable=self.weight_decay_var, width=12).grid(row=2, column=col4_idx, padx=5, pady=2, sticky="ew")
        
        ttk.Label(train_hparams_frame, text="Label Smoothing:").grid(row=3, column=col1_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(train_hparams_frame, textvariable=self.label_smoothing_var, width=12).grid(row=3, column=col2_idx, padx=5, pady=2, sticky="ew")

        ttk.Label(train_hparams_frame, text="Patience (Early Stop):").grid(row=3, column=col3_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(train_hparams_frame, textvariable=self.patience_var, width=12).grid(row=3, column=col4_idx, padx=5, pady=2, sticky="ew")

        ttk.Label(train_hparams_frame, text="Device (cpu, 0, 0,1):").grid(row=4, column=col1_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(train_hparams_frame, textvariable=self.device_var, width=12).grid(row=4, column=col2_idx, padx=5, pady=2, sticky="ew")
        
        for i in range(5): train_hparams_frame.columnconfigure(i, weight=1)

        aug_frame = ttk.LabelFrame(scrollable_frame, text="Augmentation Parameters (Probabilities / Factors)", padding="10")
        aug_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        current_row += 1

        aug_row = 0
        ttk.Label(aug_frame, text="HSV-Hue (hsv_h):").grid(row=aug_row, column=col1_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(aug_frame, textvariable=self.hsv_h_var, width=12).grid(row=aug_row, column=col2_idx, padx=5, pady=2, sticky="ew")
        ttk.Label(aug_frame, text="HSV-Saturation (hsv_s):").grid(row=aug_row, column=col3_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(aug_frame, textvariable=self.hsv_s_var, width=12).grid(row=aug_row, column=col4_idx, padx=5, pady=2, sticky="ew")
        aug_row+=1

        ttk.Label(aug_frame, text="HSV-Value (hsv_v):").grid(row=aug_row, column=col1_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(aug_frame, textvariable=self.hsv_v_var, width=12).grid(row=aug_row, column=col2_idx, padx=5, pady=2, sticky="ew")
        ttk.Label(aug_frame, text="Rotation (degrees):").grid(row=aug_row, column=col3_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(aug_frame, textvariable=self.degrees_var, width=12).grid(row=aug_row, column=col4_idx, padx=5, pady=2, sticky="ew")
        aug_row+=1
        
        ttk.Label(aug_frame, text="Translation (translate):").grid(row=aug_row, column=col1_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(aug_frame, textvariable=self.translate_var, width=12).grid(row=aug_row, column=col2_idx, padx=5, pady=2, sticky="ew")
        ttk.Label(aug_frame, text="Scale (scale):").grid(row=aug_row, column=col3_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(aug_frame, textvariable=self.scale_var, width=12).grid(row=aug_row, column=col4_idx, padx=5, pady=2, sticky="ew")
        aug_row+=1

        ttk.Label(aug_frame, text="Shear (shear):").grid(row=aug_row, column=col1_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(aug_frame, textvariable=self.shear_var, width=12).grid(row=aug_row, column=col2_idx, padx=5, pady=2, sticky="ew")
        ttk.Label(aug_frame, text="Perspective:").grid(row=aug_row, column=col3_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(aug_frame, textvariable=self.perspective_var, width=12).grid(row=aug_row, column=col4_idx, padx=5, pady=2, sticky="ew")
        aug_row+=1

        ttk.Label(aug_frame, text="Flip Up-Down (flipud):").grid(row=aug_row, column=col1_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(aug_frame, textvariable=self.flipud_var, width=12).grid(row=aug_row, column=col2_idx, padx=5, pady=2, sticky="ew")
        ttk.Label(aug_frame, text="Flip Left-Right (fliplr):").grid(row=aug_row, column=col3_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(aug_frame, textvariable=self.fliplr_var, width=12).grid(row=aug_row, column=col4_idx, padx=5, pady=2, sticky="ew")
        aug_row+=1

        ttk.Label(aug_frame, text="MixUp (mixup):").grid(row=aug_row, column=col1_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(aug_frame, textvariable=self.mixup_var, width=12).grid(row=aug_row, column=col2_idx, padx=5, pady=2, sticky="ew")
        ttk.Label(aug_frame, text="Copy-Paste:").grid(row=aug_row, column=col3_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(aug_frame, textvariable=self.copy_paste_var, width=12).grid(row=aug_row, column=col4_idx, padx=5, pady=2, sticky="ew")
        aug_row+=1
        
        ttk.Label(aug_frame, text="Mosaic:").grid(row=aug_row, column=col1_idx, padx=5, pady=2, sticky="w")
        ttk.Entry(aug_frame, textvariable=self.mosaic_var, width=12).grid(row=aug_row, column=col2_idx, padx=5, pady=2, sticky="ew")

        for i in range(5): aug_frame.columnconfigure(i, weight=1) 

        ttk.Button(scrollable_frame, text="Start Training", command=self._start_training, style="Accent.TButton").grid(row=current_row, column=0, columnspan=3, pady=20)
        current_row += 1
        

    def _start_training(self):
        yaml_p = self.dataset_yaml_path_train.get()
        project_dir = self.model_save_dir.get()
        run_name = self.run_name_var.get()
        weights = self.model_variant_var.get()

        if not all([yaml_p, project_dir, run_name, weights]):
            messagebox.showerror("Error", "Dataset YAML, Model Save Dir, Run Name, and Pretrained Weights are required.", parent=self.root)
            return

        cmd = [ "yolo", "mode=train", "data=" + yaml_p, "model=" + weights, "project=" + project_dir, "name=" + run_name ]

        def add_train_arg(param_name, var): # Simplified for training, as most are string/numeric from StringVar
            value = var.get().strip()
            if value: cmd.append(f"{param_name}={value}")
        
        add_train_arg("epochs", self.epochs_var)
        add_train_arg("lr0", self.lr_var)
        add_train_arg("batch", self.batch_var)
        add_train_arg("imgsz", self.imgsz_var)
        add_train_arg("optimizer", self.optimizer_var)
        add_train_arg("weight_decay", self.weight_decay_var)
        add_train_arg("label_smoothing", self.label_smoothing_var)
        add_train_arg("patience", self.patience_var)
        add_train_arg("device", self.device_var)
        add_train_arg("hsv_h", self.hsv_h_var)
        add_train_arg("hsv_s", self.hsv_s_var)
        add_train_arg("hsv_v", self.hsv_v_var)
        add_train_arg("degrees", self.degrees_var)
        add_train_arg("translate", self.translate_var)
        add_train_arg("scale", self.scale_var)
        add_train_arg("shear", self.shear_var)
        add_train_arg("perspective", self.perspective_var)
        add_train_arg("flipud", self.flipud_var)
        add_train_arg("fliplr", self.fliplr_var)
        add_train_arg("mixup", self.mixup_var)
        add_train_arg("copy_paste", self.copy_paste_var)
        add_train_arg("mosaic", self.mosaic_var)

        cmd_str = " ".join(cmd)
        print(f"Executing Training Command:\n{cmd_str}")

        try:
            process = subprocess.Popen(cmd, shell=sys.platform == "win32")
            messagebox.showinfo("Training Started", f"Training process initiated.\nCommand: {cmd_str}\n\nCheck console for output and progress.", parent=self.root)
        except FileNotFoundError:
            messagebox.showerror("Execution Error", "Could not find the 'yolo' command. Make sure Ultralytics YOLO is installed and in your system's PATH, or use 'python -m ultralytics ...'", parent=self.root)
        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to start training process: {e}", parent=self.root)

    def _create_inference_tab(self):
        frame = self.inference_tab # Main frame for the tab
        frame.columnconfigure(1, weight=1) # Allow path entry to expand

        current_row = 0
        # --- Source and Model Paths ---
        path_frame = ttk.LabelFrame(frame, text="Input", padding="10")
        path_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        path_frame.columnconfigure(1, weight=1)
        current_row += 1

        ttk.Label(path_frame, text="Trained Model Path (.pt):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(path_frame, textvariable=self.trained_model_path_infer, width=60).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(path_frame, text="Browse...", command=lambda: self._select_file(self.trained_model_path_infer, "Select Trained Model", (("PyTorch Models", "*.pt"),))).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(path_frame, text="Video/Image Source:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(path_frame, textvariable=self.video_infer_path, width=60).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(path_frame, text="Browse...", command=lambda: self._select_file(self.video_infer_path, "Select Video/Image Source", (("Media files", "*.mp4 *.avi *.mov *.jpg *.jpeg *.png *.bmp *.tif *.tiff"),("All files", "*.*"),))).grid(row=1, column=2, padx=5, pady=5)
        
        # --- Inference Parameters ---
        infer_params_frame = ttk.LabelFrame(frame, text="Inference Parameters", padding="10")
        infer_params_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=10)
        for i in range(4): infer_params_frame.columnconfigure(i, weight=1) # Allow entries to expand a bit
        current_row += 1
        
        param_row = 0
        # Row 0: Conf, IOU
        ttk.Label(infer_params_frame, text="Conf Thresh:").grid(row=param_row, column=0, padx=5, pady=3, sticky="w")
        ttk.Entry(infer_params_frame, textvariable=self.conf_thres_var, width=10).grid(row=param_row, column=1, padx=5, pady=3, sticky="ew")
        ttk.Label(infer_params_frame, text="IOU Thresh:").grid(row=param_row, column=2, padx=5, pady=3, sticky="w")
        ttk.Entry(infer_params_frame, textvariable=self.iou_thres_var, width=10).grid(row=param_row, column=3, padx=5, pady=3, sticky="ew")
        param_row += 1

        # Row 1: ImgSz, Device
        ttk.Label(infer_params_frame, text="Image Size (imgsz):").grid(row=param_row, column=0, padx=5, pady=3, sticky="w")
        ttk.Entry(infer_params_frame, textvariable=self.infer_imgsz_var, width=10).grid(row=param_row, column=1, padx=5, pady=3, sticky="ew")
        ttk.Label(infer_params_frame, text="Device (cpu,0):").grid(row=param_row, column=2, padx=5, pady=3, sticky="w")
        ttk.Entry(infer_params_frame, textvariable=self.infer_device_var, width=10).grid(row=param_row, column=3, padx=5, pady=3, sticky="ew")
        param_row += 1

        # Row 2: Max Det, Line Width
        ttk.Label(infer_params_frame, text="Max Detections:").grid(row=param_row, column=0, padx=5, pady=3, sticky="w")
        ttk.Entry(infer_params_frame, textvariable=self.infer_max_det_var, width=10).grid(row=param_row, column=1, padx=5, pady=3, sticky="ew")
        ttk.Label(infer_params_frame, text="Line Width (px):").grid(row=param_row, column=2, padx=5, pady=3, sticky="w")
        ttk.Entry(infer_params_frame, textvariable=self.infer_line_width_var, width=10).grid(row=param_row, column=3, padx=5, pady=3, sticky="ew")
        param_row += 1

        # Row 3: Project, Name
        ttk.Label(infer_params_frame, text="Output Project Dir:").grid(row=param_row, column=0, padx=5, pady=3, sticky="w")
        ttk.Entry(infer_params_frame, textvariable=self.infer_project_var, width=15).grid(row=param_row, column=1, padx=5, pady=3, sticky="ew")
        ttk.Label(infer_params_frame, text="Output Run Name:").grid(row=param_row, column=2, padx=5, pady=3, sticky="w")
        ttk.Entry(infer_params_frame, textvariable=self.infer_name_var, width=15).grid(row=param_row, column=3, padx=5, pady=3, sticky="ew")
        param_row += 1

        # --- Sub-frames for Checkbuttons ---
        checkbox_main_frame = ttk.Frame(infer_params_frame)
        checkbox_main_frame.grid(row=param_row, column=0, columnspan=4, sticky="ew", pady=5)
        # checkbox_main_frame.columnconfigure(0, weight=1) # If only one column of frames
        # checkbox_main_frame.columnconfigure(1, weight=1) # If two columns of frames

        output_options_frame = ttk.LabelFrame(checkbox_main_frame, text="General Output", padding="5")
        output_options_frame.pack(side="left", fill="x", expand=True, padx=2)
        
        ttk.Checkbutton(output_options_frame, text="Show Inference Window", variable=self.show_infer_var).pack(anchor="w", padx=5, pady=1)
        ttk.Checkbutton(output_options_frame, text="Save Annotated Media", variable=self.infer_save_var).pack(anchor="w", padx=5, pady=1)
        ttk.Checkbutton(output_options_frame, text="Augmented Inference", variable=self.infer_augment_var).pack(anchor="w", padx=5, pady=1)

        save_options_frame = ttk.LabelFrame(checkbox_main_frame, text="Save Details", padding="5")
        save_options_frame.pack(side="left", fill="x", expand=True, padx=2)

        ttk.Checkbutton(save_options_frame, text="Save Results (.txt)", variable=self.infer_save_txt_var).pack(anchor="w", padx=5, pady=1)
        ttk.Checkbutton(save_options_frame, text="Save Confidences in .txt", variable=self.infer_save_conf_var).pack(anchor="w", padx=5, pady=1)
        ttk.Checkbutton(save_options_frame, text="Save Cropped Detections", variable=self.infer_save_crop_var).pack(anchor="w", padx=5, pady=1)

        display_options_frame = ttk.LabelFrame(checkbox_main_frame, text="Display Details", padding="5")
        display_options_frame.pack(side="left", fill="x", expand=True, padx=2)
        
        ttk.Checkbutton(display_options_frame, text="Hide Labels", variable=self.infer_hide_labels_var).pack(anchor="w", padx=5, pady=1)
        ttk.Checkbutton(display_options_frame, text="Hide Confidences", variable=self.infer_hide_conf_var).pack(anchor="w", padx=5, pady=1)
        param_row +=1


        # --- Run Button ---
        ttk.Button(frame, text="Run Inference", command=self._run_inference, style="Accent.TButton").grid(row=current_row, column=0, columnspan=3, pady=20)
        
    def _run_inference(self):
        model_p = self.trained_model_path_infer.get()
        source_p = self.video_infer_path.get()

        if not model_p or not os.path.exists(model_p): 
            messagebox.showerror("Error", "Valid Trained Model Path required.", parent=self.root); return
        if not source_p or not os.path.exists(source_p): # Check if image/video source exists
             messagebox.showerror("Error", "Valid Video/Image Source for Inference required.", parent=self.root); return
        
        cmd = [ "yolo", "mode=predict", "model=" + model_p, "source=" + source_p]

        # Helper to build command arguments
        def add_yolo_arg(param_name, tk_var):
            value = tk_var.get()
            if isinstance(tk_var, tk.BooleanVar):
                cmd.append(f"{param_name}={value}") # Appends param=True or param=False
            elif isinstance(tk_var, tk.StringVar):
                str_value = value.strip()
                if str_value: # Only add if not empty
                    cmd.append(f"{param_name}={str_value}")

        # Add parameters from GUI
        add_yolo_arg("conf", self.conf_thres_var)
        add_yolo_arg("iou", self.iou_thres_var)
        add_yolo_arg("imgsz", self.infer_imgsz_var)
        add_yolo_arg("device", self.infer_device_var)
        add_yolo_arg("max_det", self.infer_max_det_var)
        add_yolo_arg("line_thickness", self.infer_line_width_var)
        add_yolo_arg("project", self.infer_project_var)
        add_yolo_arg("name", self.infer_name_var)

        add_yolo_arg("show", self.show_infer_var)
        add_yolo_arg("save", self.infer_save_var)
        add_yolo_arg("save_txt", self.infer_save_txt_var)
        add_yolo_arg("save_conf", self.infer_save_conf_var)
        add_yolo_arg("save_crop", self.infer_save_crop_var)
        add_yolo_arg("augment", self.infer_augment_var)
        add_yolo_arg("hide_labels", self.infer_hide_labels_var)
        add_yolo_arg("hide_conf", self.infer_hide_conf_var)
        
        cmd_str = " ".join(cmd)
        print(f"Executing Inference Command:\n{cmd_str}")
        try:
            process = subprocess.Popen(cmd, shell=sys.platform == "win32")
            messagebox.showinfo("Inference Started", f"Inference initiated.\nCommand: {cmd_str}\n\nCheck console/results window.", parent=self.root)
        except FileNotFoundError: 
            messagebox.showerror("Execution Error", "Could not find 'yolo' command. Make sure Ultralytics YOLO is installed and in your system's PATH, or use 'python -m ultralytics.yolo ...'", parent=self.root)
        except Exception as e: 
            messagebox.showerror("Inference Error", f"Failed to start inference process: {e}", parent=self.root)


if __name__ == '__main__':
    main_tk_root = tk.Tk()
    app = YoloApp(main_tk_root)
    main_tk_root.mainloop()
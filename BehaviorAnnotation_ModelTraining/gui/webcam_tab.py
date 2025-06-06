import tkinter as tk
from tkinter import ttk

def create_webcam_tab(app):
    """
    Creates and populates the 'Webcam Inference' tab in the main application window.

    Args:
        app: The main YoloApp instance, which holds all Tkinter variables and logic.
    """
    frame = app.webcam_tab
    frame.columnconfigure(1, weight=1)

    current_row = 0

    # --- Webcam Input Configuration ---
    input_frame = ttk.LabelFrame(frame, text="Webcam Input Configuration", padding="10")
    input_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
    input_frame.columnconfigure(1, weight=1)
    current_row += 1

    ttk.Label(input_frame, text="Trained Model Path (.pt):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(input_frame, textvariable=app.webcam_model_path, width=60).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(input_frame, text="Browse...", command=lambda: app._select_file(app.webcam_model_path, "Select Trained Model", (("PyTorch Models", "*.pt"),))).grid(row=0, column=2, padx=5, pady=5)

    ttk.Label(input_frame, text="Webcam Index:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(input_frame, textvariable=app.webcam_index_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky="w")
    
    # --- Webcam Options ---
    options_frame = ttk.LabelFrame(frame, text="Inference Options", padding="10")
    options_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
    options_frame.columnconfigure(1, weight=1)
    options_frame.columnconfigure(3, weight=1)
    current_row += 1

    ttk.Label(options_frame, text="Mode:").grid(row=0, column=0, padx=5, pady=3, sticky="w")
    mode_combo = ttk.Combobox(options_frame, textvariable=app.webcam_mode_var, values=['track', 'predict'], width=10)
    mode_combo.grid(row=0, column=1, padx=5, pady=3, sticky="w")
    mode_combo.current(0)

    
    ttk.Label(options_frame, text="Number of Animals (max_det):").grid(row=0, column=2, padx=5, pady=3, sticky="w")
    ttk.Entry(options_frame, textvariable=app.webcam_max_det_var, width=10).grid(row=0, column=3, padx=5, pady=3, sticky="w")


    # --- Saving Options ---
    saving_frame = ttk.LabelFrame(frame, text="Saving Options", padding="10")
    saving_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
    current_row += 1
    
    ttk.Checkbutton(saving_frame, text="Save Annotated Video (in runs/detect/predict*)", variable=app.webcam_save_annotated_var).pack(anchor="w", padx=5, pady=1)
    ttk.Checkbutton(saving_frame, text="Save Results as .txt Files", variable=app.webcam_save_txt_var).pack(anchor="w", padx=5, pady=1)
    ttk.Checkbutton(saving_frame, text="Save Original Raw Video Feed (in 'raw_captures' folder)", variable=app.webcam_save_raw_var).pack(anchor="w", padx=5, pady=1)
    
    # --- Action Buttons ---
    action_button_frame = ttk.Frame(frame)
    action_button_frame.grid(row=current_row, column=0, columnspan=2, pady=20)
    current_row += 1

    app.start_webcam_button = ttk.Button(action_button_frame, text="Start Webcam", command=app._start_webcam_inference, style="Accent.TButton")
    app.start_webcam_button.pack(side="left", padx=10)

    app.stop_webcam_button = ttk.Button(action_button_frame, text="Stop Webcam", command=app._stop_webcam_inference)
    app.stop_webcam_button.pack(side="left", padx=10)
    app.stop_webcam_button.config(state=tk.DISABLED)
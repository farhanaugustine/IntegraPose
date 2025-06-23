import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
from typing import List, Tuple

def normalize_pose(pose: np.ndarray, translation: bool = True, scale: bool = True) -> np.ndarray:
    """
    Normalizes pose keypoints by removing translation and/or scale.
    
    Args:
        pose: Array of shape (2K,) where K is number of keypoints (x1,y1,x2,y2,...).
        translation: If True, subtract mean to center pose.
        scale: If True, divide by standard deviation to normalize scale.
    
    Returns:
        Normalized pose array of same shape.
    """
    try:
        if len(pose) % 2 != 0:
            raise ValueError("Pose array length must be even (x,y pairs).")
        keypoints = pose.reshape(-1, 2)  # (K, 2)
        if translation:
            keypoints -= np.mean(keypoints, axis=0)
        if scale:
            std = np.std(keypoints)
            if std > 0:
                keypoints /= std
        return keypoints.flatten()
    except Exception as e:
        raise ValueError(f"Failed to normalize pose: {e}")

def create_pose_clustering_tab(app: 'YoloApp') -> None:
    """
    Creates the Pose Clustering Analysis tab (Tab 6) in the main GUI.
    
    Args:
        app: Reference to the main YoloApp instance.
    """
    try:
        tab = app.pose_clustering_tab
        
        # Keypoint Input Frame
        keypoint_frame = ttk.LabelFrame(tab, text="Keypoint Definitions", padding=10)
        keypoint_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        ttk.Label(keypoint_frame, text="Keypoint Names (comma-separated):").grid(row=0, column=0, sticky="w")
        app.keypoint_entry = ttk.Entry(keypoint_frame, width=50)
        app.keypoint_entry.grid(row=0, column=1, padx=5, sticky="ew")
        app.keypoint_entry.insert(0, "nose,head,thorax,abdomen,left_foreleg,right_foreleg,tail")
        
        def update_keypoints():
            try:
                names = [n.strip() for n in app.keypoint_entry.get().split(',') if n.strip()]
                if not names:
                    raise ValueError("Keypoint names cannot be empty.")
                if len(names) != len(set(names)):
                    raise ValueError("Duplicate keypoint names are not allowed.")
                for name in names:
                    if not name.isalnum() and not all(c in ' _-' for c in name if not c.isalnum()):
                        raise ValueError(f"Invalid keypoint name: '{name}'. Use alphanumeric characters, spaces, underscores, or hyphens.")
                app.keypoint_names = names
                app.log_message(f"Updated keypoints: {names}", "INFO")
                update_skeleton_dropdowns()
            except Exception as e:
                app.log_message(f"Keypoint input error: {e}", "ERROR")
                messagebox.showerror("Input Error", str(e), parent=app.root)
        
        ttk.Button(keypoint_frame, text="Update Keypoints", command=update_keypoints).grid(row=0, column=2, padx=5)
        
        # Skeleton Definition Frame
        skeleton_frame = ttk.LabelFrame(tab, text="Skeleton Connections", padding=10)
        skeleton_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        app.skeleton_connections = []  # List of (kp1_idx, kp2_idx) tuples
        app.skeleton_rows = []  # List of frames for dynamic rows
        
        def update_skeleton_dropdowns():
            """Updates dropdown menus with current keypoint names."""
            for frame in app.skeleton_rows:
                frame.destroy()
            app.skeleton_rows.clear()
            app.skeleton_connections.clear()
            if hasattr(app, 'keypoint_names') and app.keypoint_names:
                add_skeleton_row()
            else:
                app.log_message("No keypoints defined for skeleton connections.", "WARNING")
        
        def add_skeleton_row(kp1_idx=None, kp2_idx=None):
            """Adds a row with two dropdowns for skeleton connections."""
            if not hasattr(app, 'keypoint_names') or not app.keypoint_names:
                app.log_message("Define keypoints first.", "WARNING")
                messagebox.showwarning("Warning", "Please define keypoints before adding skeleton connections.", parent=app.root)
                return
            
            row_frame = ttk.Frame(skeleton_frame)
            row_frame.pack(fill="x", pady=2)
            app.skeleton_rows.append(row_frame)
            
            kp1_var = tk.StringVar()
            kp2_var = tk.StringVar()
            
            kp1_menu = ttk.Combobox(row_frame, textvariable=kp1_var, values=app.keypoint_names, state="readonly", width=15)
            kp2_menu = ttk.Combobox(row_frame, textvariable=kp2_var, values=app.keypoint_names, state="readonly", width=15)
            kp1_menu.pack(side=tk.LEFT, padx=5)
            kp2_menu.pack(side=tk.LEFT, padx=5)
            
            if kp1_idx is not None and kp2_idx is not None and kp1_idx < len(app.keypoint_names) and kp2_idx < len(app.keypoint_names):
                kp1_menu.current(kp1_idx)
                kp2_menu.current(kp2_idx)
            
            def remove_row():
                app.skeleton_rows.remove(row_frame)
                row_frame.destroy()
                update_skeleton()
            
            ttk.Button(row_frame, text="Remove", command=remove_row).pack(side=tk.LEFT, padx=5)
            
            def on_select(event):
                update_skeleton()
            
            kp1_menu.bind("<<ComboboxSelected>>", on_select)
            kp2_menu.bind("<<ComboboxSelected>>", on_select)
        
        def update_skeleton():
            """Updates skeleton_connections based on dropdown selections."""
            app.skeleton_connections.clear()
            for row_frame in app.skeleton_rows:
                kp1_menu = row_frame.winfo_children()[0]
                kp2_menu = row_frame.winfo_children()[1]
                kp1 = kp1_menu.get()
                kp2 = kp2_menu.get()
                if kp1 and kp2 and kp1 != kp2:
                    idx1 = app.keypoint_names.index(kp1)
                    idx2 = app.keypoint_names.index(kp2)
                    app.skeleton_connections.append((idx1, idx2))
            app.log_message(f"Updated skeleton: {app.skeleton_connections}", "INFO")
        
        ttk.Button(skeleton_frame, text="Add Connection", command=add_skeleton_row).pack(pady=5)
        
        # Clustering Parameters Frame
        params_frame = ttk.LabelFrame(tab, text="Clustering Parameters", padding=10)
        params_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        ttk.Label(params_frame, text="UMAP Neighbors:").grid(row=0, column=0, sticky="w")
        app.umap_neighbors_var = tk.StringVar(value="15")
        ttk.Entry(params_frame, textvariable=app.umap_neighbors_var, width=10).grid(row=0, column=1, padx=5, sticky="w")
        
        ttk.Label(params_frame, text="HDBSCAN Min Cluster Size:").grid(row=1, column=0, sticky="w")
        app.hdbscan_min_size_var = tk.StringVar(value="10")
        ttk.Entry(params_frame, textvariable=app.hdbscan_min_size_var, width=10).grid(row=1, column=1, padx=5, sticky="w")
        
        ttk.Label(params_frame, text="Normalize Translation:").grid(row=2, column=0, sticky="w")
        app.norm_translation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, variable=app.norm_translation_var).grid(row=2, column=1, sticky="w")
        
        ttk.Label(params_frame, text="Normalize Scale:").grid(row=3, column=0, sticky="w")
        app.norm_scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, variable=app.norm_scale_var).grid(row=3, column=1, sticky="w")
        
        # Selection Frame
        selection_frame = ttk.LabelFrame(tab, text="Select Behaviors and Tracks", padding=10)
        selection_frame.grid(row=0, column=1, rowspan=3, padx=5, pady=5, sticky="ns")
        
        ttk.Label(selection_frame, text="Behaviors:").pack(anchor="w")
        app.behavior_listbox = tk.Listbox(selection_frame, selectmode=tk.MULTIPLE, height=5, width=20)
        app.behavior_listbox.pack(pady=5, fill="x")
        
        ttk.Label(selection_frame, text="Track IDs:").pack(anchor="w")
        app.track_listbox = tk.Listbox(selection_frame, selectmode=tk.MULTIPLE, height=5, width=20)
        app.track_listbox.pack(pady=5, fill="x")
        
        def refresh_selections():
            """Updates behavior and track listboxes based on detailed_bouts_df."""
            try:
                app.behavior_listbox.delete(0, tk.END)
                app.track_listbox.delete(0, tk.END)
                if not app.detailed_bouts_df.empty:
                    behaviors = sorted(app.detailed_bouts_df['Behavior'].unique())
                    tracks = sorted(app.detailed_bouts_df['Track ID'].unique().astype(int)) if 'Track ID' in app.detailed_bouts_df.columns else [0]
                    for behavior in behaviors:
                        app.behavior_listbox.insert(tk.END, behavior)
                    for track in tracks:
                        app.track_listbox.insert(tk.END, str(track))
                    app.log_message(f"Refreshed selections: {len(behaviors)} behaviors, {len(tracks)} tracks: {tracks}", "INFO")
                else:
                    app.log_message("No bout data available for selections. Try running analysis in Tab 5.", "WARNING")
                    app.track_listbox.insert(tk.END, "0")  # Fallback
                    app.log_message("Added fallback track ID 0.", "INFO")
            except Exception as e:
                app.log_message(f"Error refreshing selections: {e}", "ERROR")
                messagebox.showerror("Error", f"Failed to refresh selections: {e}", parent=app.root)
        
        ttk.Button(selection_frame, text="Refresh Selections", command=refresh_selections).pack(pady=5)
        
        # Visualization Frame
        viz_frame = ttk.LabelFrame(tab, text="Cluster Visualization", padding=10)
        viz_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        app.cluster_combobox = ttk.Combobox(viz_frame, state="readonly", width=30)
        app.cluster_combobox.pack(pady=5)
        app.cluster_combobox.bind("<<ComboboxSelected>>", lambda e: app.visualize_cluster_pose())
        
        app.cluster_canvas = tk.Canvas(viz_frame, width=400, height=300, bg="white")
        app.cluster_canvas.pack(pady=5)
        
        # Run Clustering Button
        ttk.Button(tab, text="Run Clustering", command=lambda: threading.Thread(target=app.run_pose_clustering, daemon=True).start()).grid(row=4, column=0, columnspan=2, pady=10)
        
        # Initialize
        update_skeleton_dropdowns()
        refresh_selections()
        app.log_message("Pose Clustering tab initialized.", "INFO")
        
    except Exception as e:
        app.log_message(f"Failed to initialize Pose Clustering tab: {e}", "ERROR")
        messagebox.showerror("GUI Error", f"Pose Clustering tab initialization failed: {e}", parent=app.root)
        raise
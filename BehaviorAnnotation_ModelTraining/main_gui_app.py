import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, Menu
import os
import sys
import re
import subprocess
import signal
import cv2
import threading
import yaml
import pandas as pd
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import traceback
import queue
from collections import OrderedDict

from gui import setup_tab, training_tab, inference_tab, webcam_tab, roi_analytics_tab, log_tab, pose_clustering_tab
from gui.advanced_bout_analyzer import AdvancedBoutAnalyzer
from gui.bout_confirmation_tool import BoutConfirmationTool
from utils import command_builder, video_creator, bout_analyzer
from utils.roi_manager import ROIManager
from utils.bout_analyzer import BoutAnalysisError
from utils.roi_drawing_tool import draw_roi
from utils.config_manager import ConfigManager
from keypoint_annotator_module import KeypointBehaviorAnnotator

class YoloApp:
    def __init__(self, root_window):
        self.root = root_window
        self.base_title = "IntegraPose Scientific GUI"
        self.root.title(self.base_title)
        self.root.geometry("1100x950")
        self.root.minsize(1000, 800)

        self.config = ConfigManager(self)
        self.roi_manager = ROIManager()
        self.log_queue = queue.Queue()
        
        self.training_process = None
        self.file_inference_process = None
        self.webcam_inference_process = None
        self.analytics_thread = None
        self.annotator_instance = None
        
        self.detailed_bouts_df = pd.DataFrame()
        self.summary_bouts_df = pd.DataFrame()
        self.cluster_data = {}
        
        self._initialize_default_behaviors()
        self._setup_ui()
        self.update_title()
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._check_log_queue()

    def _setup_ui(self):
        try:
            menu_bar = Menu(self.root)
            self.root.config(menu=menu_bar)
            file_menu = Menu(menu_bar, tearoff=0)
            menu_bar.add_cascade(label="File", menu=file_menu)
            file_menu.add_command(label="Open Project...", command=self._safe_open_project)
            file_menu.add_command(label="Save Project", command=self._safe_save_project)
            file_menu.add_command(label="Save Project As...", command=self._safe_save_project_as)
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=self._on_closing)
            
            self.notebook = ttk.Notebook(self.root)
            self.setup_tab = ttk.Frame(self.notebook, padding="10")
            self.training_tab = ttk.Frame(self.notebook, padding="10")
            self.inference_tab = ttk.Frame(self.notebook, padding="10")
            self.webcam_tab = ttk.Frame(self.notebook, padding="10")
            self.roi_analytics_tab = ttk.Frame(self.notebook, padding="10")
            self.pose_clustering_tab = ttk.Frame(self.notebook, padding="10")
            
            self.notebook.add(self.setup_tab, text='1. Setup & Annotation')
            self.notebook.add(self.training_tab, text='2. Model Training')
            self.notebook.add(self.inference_tab, text='3. Inference')
            self.notebook.add(self.webcam_tab, text='4. Webcam Inference')
            self.notebook.add(self.roi_analytics_tab, text='5. Bout Analytics')
            self.notebook.add(self.pose_clustering_tab, text='6. Pose Clustering Analysis')
            
            setup_tab.create_setup_tab(self)
            training_tab.create_training_tab(self)
            inference_tab.create_inference_tab(self)
            webcam_tab.create_webcam_tab(self)
            roi_analytics_tab.create_roi_analytics_tab(self)
            pose_clustering_tab.create_pose_clustering_tab(self)
            log_tab.create_log_tab(self)
            
            self.notebook.pack(expand=True, fill='both', padx=5, pady=5)
            
            tools_frame = ttk.Frame(self.root)
            tools_frame.pack(pady=(0, 5))
            bout_confirmer_button = ttk.Button(tools_frame, text="Review & Confirm Detected Bouts", command=self._open_bout_confirmer)
            bout_confirmer_button.pack(side=tk.LEFT, padx=10)
            bout_analyzer_button = ttk.Button(tools_frame, text="Open Advanced Bout Scorer", command=self._open_advanced_analyzer)
            bout_analyzer_button.pack(side=tk.LEFT, padx=10)

            self.status_var = tk.StringVar(value="Ready.")
            status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
            status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        except Exception as e:
            self.log_message(f"Failed to initialize UI: {e}", "ERROR")
            messagebox.showerror("GUI Error", f"UI initialization failed: {e}", parent=self.root)
            raise

    def validate_yolo_files(self, yolo_folder, sample_size=10):
        """
        Validates YOLO .txt files for consistent keypoint counts.
        
        Args:
            yolo_folder: Path to folder with YOLO .txt files.
            sample_size: Number of files to sample (0 for all).
        
        Returns:
            Tuple: (bool, int, str) - (is_valid, keypoint_count, error_message).
        """
        try:
            txt_files = [f for f in os.listdir(yolo_folder) if f.endswith('.txt')]
            if not txt_files:
                return False, 0, "No .txt files found in YOLO folder."
            
            if sample_size > 0:
                txt_files = txt_files[:min(sample_size, len(txt_files))]
            
            keypoint_counts = set()
            for filename in txt_files:
                filepath = os.path.join(yolo_folder, filename)
                try:
                    df = pd.read_csv(filepath, sep=' ', header=None, dtype=float)
                    if len(df.columns) < 6:
                        return False, 0, f"Invalid format in {filename}: {len(df.columns)} columns."
                    keypoint_values = len(df.columns) - 6
                    if keypoint_values % 3 != 0:
                        return False, 0, f"Invalid keypoint data in {filename}: {keypoint_values} values."
                    keypoint_counts.add(keypoint_values // 3)
                except ValueError as e:
                    return False, 0, f"Invalid data in {filename}: {e}"
                except Exception as e:
                    return False, 0, f"Error processing {filename}: {e}"
            
            if len(keypoint_counts) != 1:
                return False, 0, f"Inconsistent keypoint counts: {keypoint_counts}"
            
            return True, keypoint_counts.pop(), ""
        
        except Exception as e:
            return False, 0, f"Validation error: {e}"

    def _get_pose_from_frame(self, frame_num, filename=None, track_id=None, expected_keypoints=None):
        """
        Extracts pose data from a YOLO output file for a given frame and track ID.
        
        Args:
            frame_num: Integer frame number (for logging).
            filename: Optional specific filename to parse.
            track_id: The specific track ID to extract the pose for (None for first detection).
            expected_keypoints: Expected number of keypoints (None to detect dynamically).
        
        Returns:
            Tuple: (Numpy array of pose coordinates (2K), number of keypoints) or (None, None).
        """
        try:
            yolo_folder = self.config.get_setting('analytics.yolo_output_path_var')
            if not yolo_folder:
                self.log_message("YOLO output folder not set.", "ERROR")
                return None, None
            if filename is None:
                filename = f"frame_{frame_num}.txt"
            filepath = os.path.join(yolo_folder, filename)
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                self.log_message(f"YOLO output file not found or empty: {filepath}", "WARNING")
                return None, None
            
            try:
                df = pd.read_csv(filepath, sep=' ', header=None, dtype=float)
            except ValueError as e:
                self.log_message(f"Invalid data in {filename}: {e}", "ERROR")
                return None, None
            
            self.log_message(f"Parsing {filename} with {len(df.columns)} columns for track ID {track_id}", "DEBUG")
            
            if len(df.columns) < 6:
                self.log_message(f"Invalid YOLO format in {filename}: {len(df.columns)} columns (need at least 6)", "ERROR")
                return None, None
            
            total_cols = len(df.columns)
            keypoint_values = total_cols - 6
            if keypoint_values % 3 != 0:
                self.log_message(f"Invalid keypoint data in {filename}: {keypoint_values} values (not divisible by 3)", "ERROR")
                return None, None
            detected_keypoints = keypoint_values // 3
            
            if expected_keypoints is not None and detected_keypoints != expected_keypoints:
                self.log_message(f"Keypoint count mismatch in {filename}: found {detected_keypoints}, expected {expected_keypoints}", "ERROR")
                return None, None
            
            df.rename(columns={total_cols - 1: 'track_id'}, inplace=True)
            df['track_id'] = df['track_id'].astype(int)
            
            if track_id is None:
                self.log_message(f"No track ID provided for {filename}, using first detection.", "WARNING")
                detection_row = df.iloc[[0]]
            else:
                detection_row = df[df['track_id'] == track_id]
            
            if detection_row.empty:
                self.log_message(f"Track ID {track_id} not found in {filename}", "WARNING")
                return None, None
            
            detection_series = detection_row.iloc[0]
            keypoint_start = 5
            keypoint_end = -1
            
            kp_data = detection_series.iloc[keypoint_start:keypoint_end].values
            keypoints_xy = kp_data.reshape(-1, 3)[:, :2].flatten()
            self.log_message(f"Extracted {detected_keypoints} keypoints for track {track_id} from {filename}", "DEBUG")
            return keypoints_xy, detected_keypoints
        
        except Exception as e:
            self.log_message(f"Error extracting pose for track {track_id} in {filename}: {e}", "ERROR")
            return None, None

    def run_pose_clustering(self):
        """
        Runs UMAP + HDBSCAN clustering on selected behavior poses for specified tracks.
        """
        try:
            self.update_status("Starting pose clustering...")
            self.log_message("Starting pose clustering.", "INFO")
            
            # Validate inputs
            selected_behaviors = [self.behavior_listbox.get(i) for i in self.behavior_listbox.curselection()]
            if not selected_behaviors:
                self.root.after(0, lambda: messagebox.showerror("Error", "Please select at least one behavior.", parent=self.root))
                return
            self.log_message(f"Selected behaviors for clustering: {selected_behaviors}", "INFO")
            
            yolo_folder = self.config.get_setting('analytics.yolo_output_path_var')
            video_path = self.config.get_setting('analytics.source_video_path_var')
            if not all([yolo_folder, video_path]):
                self.root.after(0, lambda: messagebox.showerror("Error", "YOLO folder and source video are required.", parent=self.root))
                return
            
            # Validate YOLO files
            is_valid, keypoint_count, error_msg = self.validate_yolo_files(yolo_folder)
            if not is_valid:
                self.log_message(f"YOLO validation failed: {error_msg}", "ERROR")
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("YOLO Error", f"Invalid YOLO files: {msg}", parent=self.root))
                return
            
            # Validate keypoint names
            if not hasattr(self, 'keypoint_names') or not self.keypoint_names:
                self.log_message("Keypoint names not defined.", "ERROR")
                self.root.after(0, lambda: messagebox.showerror("Error", "Please define keypoint names in Tab 6.", parent=self.root))
                return
            if len(self.keypoint_names) != keypoint_count:
                self.log_message(f"Keypoint count mismatch: GUI has {len(self.keypoint_names)}, YOLO has {keypoint_count}", "ERROR")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Keypoint count mismatch: GUI has {len(self.keypoint_names)}, YOLO has {keypoint_count}", parent=self.root))
                return
            
            keypoint_names = self.keypoint_names
            skeleton = self.skeleton_connections if hasattr(self, 'skeleton_connections') and self.skeleton_connections else [[i, i+1] for i in range(keypoint_count-1)]
            self.log_message(f"Using {keypoint_count} keypoints: {keypoint_names}, skeleton: {skeleton}", "INFO")
            
            # Validate tracks
            has_tracking = 'Track ID' in self.detailed_bouts_df.columns
            selected_tracks = [int(self.track_listbox.get(i)) for i in self.track_listbox.curselection()] if has_tracking and hasattr(self, 'track_listbox') and self.track_listbox.curselection() else []
            if has_tracking and not selected_tracks:
                selected_tracks = sorted(self.detailed_bouts_df['Track ID'].unique().astype(int))
                self.log_message(f"No tracks selected, using all available: {selected_tracks}", "INFO")
            elif not has_tracking:
                selected_tracks = [0]
                self.log_message("No 'Track ID' column in bouts_df. Using default track ID 0.", "INFO")
            if not selected_tracks:
                self.log_message("No tracks available for clustering. Check analysis results.", "ERROR")
                self.root.after(0, lambda: messagebox.showerror("Error", "No tracks available. Run analysis in Tab 5 and refresh selections.", parent=self.root))
                return
            self.log_message(f"Selected tracks: {selected_tracks}", "INFO")
            
            # Build frame-to-file mapping
            frame_to_file = {}
            for fname in os.listdir(yolo_folder):
                if fname.endswith('.txt'):
                    for pattern in [r'_(\d+)\.txt$', r'frame_(\d+)\.txt$', r'video_(\d+)\.txt$', r'(\d+)\.txt$']:
                        match = re.search(pattern, fname)
                        if match:
                            frame_to_file[int(match[1])] = fname
                            break
            
            if not frame_to_file:
                self.log_message("No valid YOLO files found.", "ERROR")
                self.root.after(0, lambda: messagebox.showerror("Error", "No valid YOLO .txt files found.", parent=self.root))
                return
            
            # Collect pose data
            track_data = {}
            for behavior in selected_behaviors:
                bouts = self.detailed_bouts_df[
                    (self.detailed_bouts_df['Behavior'] == behavior) & 
                    (self.detailed_bouts_df['Track ID'].isin(selected_tracks))
                ] if has_tracking else self.detailed_bouts_df[self.detailed_bouts_df['Behavior'] == behavior]
                if bouts.empty:
                    self.log_message(f"No bouts for behavior '{behavior}' with selected tracks.", "WARNING")
                    continue
                for _, bout in bouts.iterrows():
                    start_frame = int(bout['Start Frame'])
                    end_frame = int(bout['End Frame'])
                    track_id = int(bout.get('Track ID', 0))
                    if track_id not in track_data:
                        track_data[track_id] = {'poses': [], 'frame_ids': [], 'behavior': behavior}
                    for frame in range(start_frame, end_frame + 1):
                        if frame in frame_to_file:
                            pose, kp_count = self._get_pose_from_frame(frame, frame_to_file[frame], 
                                                                      track_id if has_tracking else None, 
                                                                      keypoint_count)
                            if pose is not None and kp_count == keypoint_count:
                                track_data[track_id]['poses'].append(pose)
                                track_data[track_id]['frame_ids'].append(frame)
                            else:
                                self.log_message(f"Invalid pose for frame {frame}, track {track_id}", "WARNING")
            
            if not track_data:
                self.log_message("No valid pose data collected.", "ERROR")
                self.root.after(0, lambda: messagebox.showerror("Error", "No valid pose data for clustering.", parent=self.root))
                return
            
            # Run clustering
            output_folder = os.path.join(os.path.dirname(video_path), "bout_analysis_results")
            os.makedirs(output_folder, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            suffix = selected_behaviors[0] if len(selected_behaviors) == 1 else f"multi_{int(pd.Timestamp.now().timestamp())}"
            
            all_cluster_data = {}
            valid_clusters = []
            
            from gui.pose_clustering_tab import normalize_pose
            for track_id, data in track_data.items():
                pose_data = np.array(data['poses'])
                frame_ids = data['frame_ids']
                self.log_message(f"Clustering {len(pose_data)} poses for track {track_id}", "INFO")
                
                # Normalize poses
                try:
                    normalized_poses = np.array([normalize_pose(pose, 
                                                               self.norm_translation_var.get(), 
                                                               self.norm_scale_var.get()) 
                                                for pose in pose_data])
                except Exception as e:
                    self.log_message(f"Normalization error for track {track_id}: {e}", "ERROR")
                    continue
                
                # UMAP
                try:
                    reducer = umap.UMAP(n_neighbors=int(self.umap_neighbors_var.get()), 
                                        min_dist=0.1, 
                                        n_components=2, 
                                        random_state=42)
                    embedding = reducer.fit_transform(normalized_poses)
                except Exception as e:
                    self.log_message(f"UMAP error for track {track_id}: {e}", "ERROR")
                    continue
                
                # HDBSCAN
                try:
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(self.hdbscan_min_size_var.get()), 
                                                min_samples=5)
                    labels = clusterer.fit_predict(embedding)
                except Exception as e:
                    self.log_message(f"HDBSCAN error for track {track_id}: {e}", "ERROR")
                    continue
                
                # Compute average poses
                unique_labels = np.unique(labels)
                cluster_averages = {}
                for label in unique_labels:
                    if label != -1:
                        cluster_poses = normalized_poses[labels == label]
                        if len(cluster_poses) > 0:
                            cluster_averages[label] = np.mean(cluster_poses, axis=0)
                
                # Store results
                all_cluster_data[track_id] = {
                    'poses': normalized_poses,
                    'labels': labels,
                    'embedding': embedding,
                    'averages': cluster_averages,
                    'skeleton': skeleton,
                    'keypoint_names': keypoint_names,
                    'frame_ids': frame_ids,
                    'behavior': selected_behaviors[0] if len(selected_behaviors) == 1 else 'Multiple'
                }
                
                # Save outputs
                cluster_img_path = os.path.join(output_folder, f"clusters_{base_name}_{suffix}_track{track_id}.png")
                cluster_csv_path = os.path.join(output_folder, f"pose_clusters_{base_name}_{suffix}_track{track_id}.csv")
                
                plt.figure(figsize=(8, 6))
                plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral')
                plt.colorbar(label='Cluster ID')
                plt.title(f"Pose Clusters for {', '.join(selected_behaviors)} (Track {track_id})")
                plt.savefig(cluster_img_path)
                plt.close()
                
                results_df = pd.DataFrame({
                    'Frame': frame_ids,
                    'Behavior': [selected_behaviors[0]] * len(frame_ids) if len(selected_behaviors) == 1 else ['Multiple'] * len(frame_ids),
                    'Track ID': [track_id] * len(frame_ids),
                    'Cluster': labels
                })
                
                if cluster_averages:
                    avg_df = pd.DataFrame([
                        [label, *avg_pose] for label, avg_pose in cluster_averages.items()
                    ], columns=['Cluster'] + [f"Keypoint_{i+1}_{xy}" for i in range(keypoint_count) for xy in ['x', 'y']])
                    avg_df['Track ID'] = track_id
                    results_df = pd.concat([results_df, avg_df], ignore_index=True)
                
                results_df.to_csv(cluster_csv_path, index=False)
                self.log_message(f"Saved results for track {track_id} to {cluster_csv_path} and {cluster_img_path}", "INFO")
                
                for label in unique_labels:
                    if label != -1:
                        valid_clusters.append(f"Track {track_id} - Cluster {label}")
            
            if not all_cluster_data:
                self.root.after(0, lambda: messagebox.showwarning("Warning", "No valid clusters found.", parent=self.root))
                self.update_status("Ready.")
                return
            
            # Update GUI
            self.cluster_data = all_cluster_data
            self.root.after(0, lambda: self.cluster_combobox.configure(values=valid_clusters))
            if valid_clusters:
                self.root.after(0, lambda: self.cluster_combobox.set(valid_clusters[0]))
                
                first_track_id = list(all_cluster_data.keys())[0]
                img = Image.open(os.path.join(output_folder, f"clusters_{base_name}_{suffix}_track{first_track_id}.png"))
                img = img.resize((400, 300), Image.Resampling.LANCZOS)
                self.cluster_img = ImageTk.PhotoImage(img)
                self.root.after(0, lambda: self.cluster_canvas.delete('all'))
                self.root.after(0, lambda: self.cluster_canvas.create_image(0, 0, anchor=tk.NW, image=self.cluster_img))
                self.cluster_canvas.image = self.cluster_img
                
                self.root.after(0, lambda: self.visualize_cluster_pose())
            
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Clustering completed for {len(all_cluster_data)} tracks.\nResults saved in: {output_folder}", parent=self.root))
            self.log_message(f"Clustering completed for {len(all_cluster_data)} tracks", "INFO")
            self.update_status("Pose clustering completed.")
            
        except Exception as e:
            self.log_message(f"Clustering error: {e}\n{traceback.format_exc()}", "ERROR")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Clustering failed: {e}", parent=self.root))
            self.update_status("Ready.")

    def visualize_cluster_pose(self):
        """
        Visualizes the average normalized pose for the selected track and cluster.
        """
        try:
            if not self.cluster_data or not hasattr(self, 'cluster_combobox'):
                self.log_message("No clustering data or visualization not initialized.", "WARNING")
                return False
            
            selected = self.cluster_combobox.get()
            if not selected:
                self.log_message("No cluster selected.", "WARNING")
                return False
            
            match = re.match(r"Track (\d+) - Cluster (\d+)", selected)
            if not match:
                self.log_message(f"Invalid cluster selection: {selected}", "WARNING")
                return False
            track_id = int(match.group(1))
            cluster_id = int(match.group(2))
            
            if track_id not in self.cluster_data or cluster_id not in self.cluster_data[track_id]['averages']:
                self.log_message(f"No pose for track {track_id}, cluster {cluster_id}", "WARNING")
                return False
            
            self.cluster_canvas.delete('all')
            
            avg_pose = self.cluster_data[track_id]['averages'][cluster_id]
            skeleton = self.cluster_data[track_id]['skeleton']
            keypoint_names = self.cluster_data[track_id]['keypoint_names']
            num_keypoints = len(avg_pose) // 2
            
            pose_2d = avg_pose.reshape(-1, 2)
            
            canvas_width, canvas_height = 400, 300
            margin = 20
            x, y = pose_2d[:, 0], pose_2d[:, 1]
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            if x_max == x_min or y_max == y_min:
                self.log_message(f"Invalid pose range for track {track_id}, cluster {cluster_id}: x=({x_min}, {x_max}), y=({y_min}, {y_max})", "WARNING")
                return False
            
            scale_x = (canvas_width - 2 * margin) / (x_max - x_min)
            scale_y = (canvas_height - 2 * margin) / (y_max - y_min)
            scale = min(scale_x, scale_y)
            x = (x - x_min) * scale + margin
            y = (y - y_min) * scale + margin
            y = canvas_height - y
            
            for i, (px, py) in enumerate(zip(x, y)):
                self.cluster_canvas.create_oval(px-5, py-5, px+5, py+5, fill='blue')
                if keypoint_names and i < len(keypoint_names):
                    self.cluster_canvas.create_text(px+10, py, text=keypoint_names[i], anchor=tk.W, fill='black')
            
            for idx1, idx2 in skeleton:
                if idx1 < num_keypoints and idx2 < num_keypoints:
                    x1, y1 = x[idx1], y[idx1]
                    x2, y2 = x[idx2], y[idx2]
                    self.cluster_canvas.create_line(x1, y1, x2, y2, fill='red', width=2)
            
            self.cluster_canvas.create_text(20, 20, text=f"Track {track_id} - Cluster {cluster_id} Average Pose", anchor=tk.NW, fill='black')
            self.log_message(f"Visualized pose for track {track_id}, cluster {cluster_id}", "INFO")
            return True
        
        except Exception as e:
            self.log_message(f"Visualization error for track {track_id}, cluster {cluster_id}: {e}", "ERROR")
            self.cluster_canvas.delete('all')
            self.cluster_canvas.create_text(20, 150, text=f"Error visualizing {selected}: {e}", anchor=tk.NW, fill='red')
            return False

    def _safe_open_project(self):
        try:
            self.config.open_project()
            self._initialize_default_behaviors()
            self.update_status("Project opened successfully.")
            self.log_message("Project opened successfully.", "INFO")
            messagebox.showinfo("Success", "Project opened successfully.", parent=self.root)
        except Exception as e:
            self.log_message(f"Failed to open project: {e}", "ERROR")
            messagebox.showerror("Project Error", f"Failed to open project: {e}", parent=self.root)

    def _safe_save_project(self):
        try:
            self.config.save_project()
            self.update_status("Project saved successfully.")
            self.log_message("Project saved successfully.", "INFO")
            messagebox.showinfo("Success", "Project saved successfully.", parent=self.root)
        except Exception as e:
            self.log_message(f"Failed to save project: {e}", "ERROR")
            messagebox.showerror("Project Error", f"Failed to save project: {e}", parent=self.root)

    def _safe_save_project_as(self):
        try:
            self.config.save_project_as()
            self.update_status("Project saved as new successfully.")
            self.log_message("Project saved as new successfully.", "INFO")
            messagebox.showinfo("Success", "Project saved as new successfully.", parent=self.root)
        except Exception as e:
            self.log_message(f"Failed to save project as: {e}", "ERROR")
            messagebox.showerror("Project Error", f"Failed to save project as: {e}", parent=self.root)

    def _initialize_default_behaviors(self):
        if not self.config.setup.behaviors_list:
            self.config.setup.behaviors_list.extend([
                {'id': 0, 'name': 'Standing'},
                {'id': 1, 'name': 'Walking'}
            ])
            self.log_message("Initialized default behaviors: Standing, Walking.", "INFO")

    def _select_directory(self, string_var, title="Select Directory"):
        dir_path = filedialog.askdirectory(title=title, parent=self.root)
        if dir_path:
            string_var.set(dir_path)
            self.log_message(f"Selected directory: {dir_path}", "INFO")

    def _select_file(self, string_var, title="Select File", filetypes=(("All files", "*.*"), ("Text files", "*.txt"))):
        file_path = filedialog.askopenfilename(title=title, filetypes=filetypes, parent=self.root)
        if file_path:
            string_var.set(file_path)
            self.log_message(f"Selected file: {file_path}", "INFO")

    def _on_project_root_selected(self):
        dir_path = filedialog.askdirectory(title="Select Project Root Directory", parent=self.root)
        if not dir_path:
            return
        self.config.setup.dataset_root_yaml.set(dir_path)
        norm_path = os.path.normpath(dir_path)
        self.config.training.model_save_dir.set(os.path.join(norm_path, "models"))
        self.update_status(f"Project root set to {norm_path}. Please select other paths as needed.")
        self.log_message(f"Project root set to {norm_path}", "INFO")

    def _select_yolo_output_directory(self):
        dir_path = filedialog.askdirectory(title="Select Folder with YOLO .txt Files", parent=self.root)
        if dir_path:
            txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
            if not txt_files:
                self.log_message(f"No .txt files found in selected YOLO output directory: {dir_path}", "WARNING")
                messagebox.showwarning("Invalid Directory", "The selected directory contains no .txt files. Please select a directory with YOLO detection outputs.", parent=self.root)
            else:
                self.config.analytics.yolo_output_path_var.set(dir_path)
                self.log_message(f"Selected YOLO output directory with {len(txt_files)} .txt files: {dir_path}", "INFO")

    def _select_source_video(self):
        filetypes = (("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        path = filedialog.askopenfilename(title="Select Source Video", filetypes=filetypes, parent=self.root)
        if path:
            self.config.analytics.source_video_path_var.set(path)
            self.log_message(f"Source video selected: {path}", "INFO")

    def _refresh_behavior_listbox_display(self):
        if hasattr(self, 'behavior_list_display'):
            self.behavior_list_display.delete(0, tk.END)
            self.config.setup.behaviors_list.sort(key=lambda b: b['id'])
            for behavior in self.config.setup.behaviors_list:
                self.behavior_list_display.insert(tk.END, f"{behavior['id']}: {behavior['name']}")
            self._clear_behavior_input_fields()
        else:
            self.log_message("Behavior listbox not initialized in setup tab.", "WARNING")

    def _on_behavior_select(self, event):
        if not hasattr(self, 'behavior_list_display') or not self.behavior_list_display.curselection():
            self.log_message("No behavior selected or listbox not initialized.", "WARNING")
            return
        selected_index = self.behavior_list_display.curselection()[0]
        selected_item = self.config.setup.behaviors_list[selected_index]
        self.current_behavior_id_entry.set(str(selected_item['id']))
        self.current_behavior_name_entry.set(selected_item['name'])
        self.log_message(f"Selected behavior: {selected_item['name']} (ID: {selected_item['id']})", "INFO")

    def _add_update_behavior(self):
        if not hasattr(self, 'current_behavior_id_entry') or not hasattr(self, 'current_behavior_name_entry'):
            self.log_message("Behavior input fields not initialized.", "ERROR")
            messagebox.showerror("Input Error", "Behavior input fields not initialized.", parent=self.root)
            return
        try:
            b_id = int(self.current_behavior_id_entry.get())
            b_name = self.current_behavior_name_entry.get().strip()
            if not b_name:
                raise ValueError("Name cannot be empty.")
            existing_b = next((b for b in self.config.setup.behaviors_list if b['id'] == b_id), None)
            if existing_b:
                existing_b['name'] = b_name
                self.log_message(f"Updated behavior ID {b_id} to name '{b_name}'", "INFO")
            else:
                self.config.setup.behaviors_list.append({'id': b_id, 'name': b_name})
                self.log_message(f"Added behavior: {b_name} (ID: {b_id})", "INFO")
            self._refresh_behavior_listbox_display()
        except ValueError as e:
            self.log_message(f"Behavior Input Error: {e}", "ERROR")
            messagebox.showerror("Input Error", f"Invalid input: {e}", parent=self.root)

    def _remove_selected_behavior(self):
        if not hasattr(self, 'behavior_list_display') or not self.behavior_list_display.curselection():
            self.log_message("No behavior selected for removal.", "WARNING")
            return
        selected_index = self.behavior_list_display.curselection()[0]
        removed_behavior = self.config.setup.behaviors_list.pop(selected_index)
        self.log_message(f"Removed behavior: {removed_behavior['name']} (ID: {removed_behavior['id']})", "INFO")
        self._refresh_behavior_listbox_display()

    def _clear_behavior_input_fields(self):
        if not hasattr(self, 'current_behavior_id_entry') or not hasattr(self, 'current_behavior_name_entry'):
            self.log_message("Behavior input fields not initialized.", "WARNING")
            return
        self.current_behavior_id_entry.set("")
        self.current_behavior_name_entry.set("")
        if hasattr(self, 'behavior_list_display') and self.behavior_list_display.curselection():
            self.behavior_list_display.selection_clear(0, tk.END)

    def _launch_annotator(self):
        try:
            kp_names = [n.strip() for n in self.config.get_setting('setup.keypoint_names_str').split(',') if n.strip()]
            behaviors = self.config.get_setting('setup.behaviors_list')
            img_dir = self.config.get_setting('setup.image_dir_annot')
            ann_dir = self.config.get_setting('setup.annotation_dir')
            if not all([kp_names, behaviors, img_dir, ann_dir]):
                self.log_message(f"Keypoint missing: {kp_names}, Behaviors: {behaviors}, Image Dir: {img_dir}, Annotation Dir: {ann_dir}", "ERROR")
                raise ValueError("Keypoints, Behaviors, Image Directory, and Annotation Output Directory must all be set.")
            available_behaviors = OrderedDict((b['id'], b['name']) for b in sorted(behaviors, key=lambda b: b['id']))
            os.makedirs(ann_dir, exist_ok=True)
            self.log_message(f"Launching annotator with image dir: {img_dir}, annotation dir: {ann_dir}", "INFO")
            self.annotator_instance = KeypointBehaviorAnnotator(self.root, kp_names, available_behaviors, img_dir, ann_dir)
            self.root.withdraw()
            self.update_status("Annotator launched. Close annotator window to return.")
            self.annotator_instance.run()
        except Exception as e:
            self.log_message(f"Annotator Error: {e}\n{traceback.format_exc()}", "ERROR")
            messagebox.showerror("Annotator Error", f"Could not launch annotator: {e}", parent=self.root)
        finally:
            if self.root.winfo_exists():
                self.root.deiconify()
                self.root.focus_force()
            self.update_status("Ready.")
            self.log_message("Annotator closed.", "INFO")

    def _generate_yaml_from_gui(self):
        try:
            kp_names = [n.strip() for n in self.config.get_setting('setup.keypoint_names_str').split(',') if n.strip()]
            behaviors = self.config.get_setting('setup.behaviors_list')
            if not all([kp_names, behaviors]):
                raise ValueError("Keypoints and behaviors must be defined.")
            available_behaviors = OrderedDict((b['id'], b['name']) for b in sorted(behaviors, key=lambda b: b['id']))
            self.log_message("Generating dataset YAML...", "INFO")
            generated_path = KeypointBehaviorAnnotator.generate_config_yaml(
                master_tk_window=self.root,
                num_keypoints=len(kp_names),
                keypoint_names=kp_names,
                available_behaviors=available_behaviors,
                dataset_root_path=self.config.get_setting('setup.dataset_root_yaml'),
                train_images_path=self.config.get_setting('setup.train_image_yaml'),
                val_images_path=self.config.get_setting('setup.val_image_yaml'),
                dataset_name_suggestion=self.config.get_setting('setup.dataset_name_yaml')
            )
            if generated_path:
                self.config.set_setting('training.dataset_yaml_path_train', generated_path)
                self.config.set_setting('analytics.roi_analytics_yaml_path_var', generated_path)
                self.update_status("YAML created and paths updated.")
                self.log_message(f"YAML file generated at: {generated_path}", "INFO")
        except Exception as e:
            self.log_message(f"YAML Generation Error: {e}\n{traceback.format_exc()}", "ERROR")
            messagebox.showerror("YAML Generation Error", f"An unexpected error occurred: {e}", parent=self.root)

    def _on_roi_mode_change(self):
        is_file_mode = self.config.analytics.roi_mode_var.get() == "File-based Analysis"
        state = tk.NORMAL if is_file_mode else tk.DISABLED
        def set_state_recursive(widget, new_state):
            try:
                widget.configure(state=new_state)
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                set_state_recursive(child, new_state)
        if hasattr(self, 'file_controls_frame'):
            set_state_recursive(self.file_controls_frame, state)
        self.log_message(f"ROI mode changed to: {self.config.analytics.roi_mode_var.get()}", "INFO")

    def _draw_roi(self):
        try:
            mode = self.config.get_setting('analytics.roi_mode_var')
            cap = None
            if mode == "Live Webcam":
                cam_idx_str = self.config.get_setting('webcam.webcam_index_var')
                try:
                    cam_idx = int(cam_idx_str)
                except ValueError:
                    raise ValueError("Invalid webcam index. Please enter a valid number (e.g., 0, 1).")
                cap = cv2.VideoCapture(cam_idx)
                if not cap.isOpened():
                    raise ConnectionError(f"Could not open webcam with index {cam_idx}. Ensure the webcam is connected and not in use.")
            else:  # File-based Analysis
                video_path = self.config.get_setting('analytics.source_video_path_var')
                if not video_path:
                    raise ValueError("Please select a source video first.")
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ConnectionError(f"Could not open video file: {video_path}")

            ret, frame = cap.read()
            if not ret:
                raise ValueError("Could not read frame from video source.")
            self.log_message(f"Opening ROI drawing tool for {'webcam' if mode == 'Live Webcam' else 'video'}: {cam_idx if mode == 'Live Webcam' else video_path}", "INFO")
        except Exception as e:
            self.log_message(f"ROI Drawing Error: {e}", "ERROR")
            messagebox.showerror("Error", str(e), parent=self.root)
            return
        finally:
            if cap is not None:
                cap.release()

        self.update_status("ROI drawing tool launched...")
        points = draw_roi(frame)  # Pass frame, not video_path
        self.update_status("Ready.")
        if points:
            roi_name = simpledialog.askstring("ROI Name", "Enter a name for this ROI:", parent=self.root)
            if roi_name and roi_name.strip() and roi_name not in self.roi_manager.rois:
                self.roi_manager.add_roi(roi_name.strip(), points)
                self._refresh_roi_listbox()
                self.log_message(f"Added ROI: {roi_name}", "INFO")
            else:
                self.log_message(f"ROI addition failed: Invalid or duplicate name '{roi_name}'", "WARNING")

    def _refresh_roi_listbox(self):
        if hasattr(self, 'roi_listbox'):
            self.roi_listbox.delete(0, tk.END)
            for name in sorted(self.roi_manager.rois.keys()):
                self.roi_listbox.insert(tk.END, name)
            self.log_message("Refreshed ROI listbox.", "INFO")

    def _rename_roi(self):
        if not hasattr(self, 'roi_listbox') or not self.roi_listbox.curselection():
            self.log_message("No ROI selected for renaming.", "WARNING")
            return
        
        old_name = self.roi_listbox.get(self.roi_listbox.curselection()[0])
        new_name = simpledialog.askstring("Rename ROI", f"New name for '{old_name}':", parent=self.root)
        if new_name and new_name.strip() and new_name not in self.roi_manager.rois:
            self.roi_manager.rois[new_name] = self.roi_manager.rois.pop(old_name)
            self._refresh_roi_listbox()
            self.log_message(f"Renamed ROI from '{old_name}' to '{new_name}'", "INFO")
        else:
            self.log_message(f"ROI rename failed: Invalid or duplicate name '{new_name}'", "WARNING")

    def _delete_roi(self):
        if not hasattr(self, 'roi_listbox') or not self.roi_listbox.curselection():
            self.log_message("No ROI selected for deletion.", "WARNING")
            return
        
        roi_name = self.roi_listbox.get(self.roi_listbox.curselection()[0])
        if messagebox.askyesno("Delete ROI", f"Delete '{roi_name}'?", parent=self.root):
            del self.roi_manager.rois[roi_name]
            self._refresh_roi_listbox()
            self.log_message(f"Deleted ROI: {roi_name}", "INFO")

    def _open_bout_confirmer(self):
        self.log_message("Attempting to open Bout Confirmer...", "INFO")
        try:
            video_path = self.config.get_setting('analytics.source_video_path_var')
            if not video_path:
                raise FileNotFoundError("Please select a source video in the Bout Analytics tab.")
            if self.detailed_bouts_df.empty:
                raise ValueError("No analysis results found. Please run analytics first in the Bout Analytics tab.")
            behaviors = self.config.get_setting('setup.behaviors_list')
            behavior_map = {b['name']: b['id'] for b in behaviors}
            self.log_message(f"Opening Bout Confirmer with video: {video_path}, bouts: {len(self.detailed_bouts_df)}", "INFO")
            BoutConfirmationTool(master=self.root, video_path=video_path, detected_bouts_df=self.detailed_bouts_df, behavior_map=behavior_map)
            self.log_message("Bout Confirmer opened successfully.", "INFO")
        except Exception as e:
            self.log_message(f"Failed to open Bout Confirmer: {e}\n{traceback.format_exc()}", "ERROR")
            messagebox.showerror("Error", f"Failed to open Bout Confirmer: {e}", parent=self.root)

    def _open_advanced_analyzer(self, initial_bout_details=None, video_path=None):
        self.log_message("Attempting to open Advanced Bout Scorer...", "INFO")
        try:
            if not initial_bout_details:
                self.log_message("No initial bout details provided.", "INFO")
                messagebox.showinfo("Info", "To use the Advanced Scorer, please first run an analysis and then double-click on a bout in the results table in the 'Bout Analytics' tab.", parent=self.root)
                return
            if not video_path:
                video_path = self.config.get_setting('analytics.source_video_path_var')
                if not video_path:
                    raise ValueError("Please select a source video in the Bout Analytics tab.")
            behaviors = self.config.get_setting('setup.behaviors_list')
            behavior_map = {b['name']: b['id'] for b in behaviors}
            self.log_message(f"Opening Advanced Bout Scorer with video: {video_path}", "INFO")
            AdvancedBoutAnalyzer(parent=self.root, video_path=video_path, initial_bout_details=initial_bout_details, behavior_map=behavior_map, all_bouts_df=self.detailed_bouts_df)
            self.log_message("Advanced Bout Scorer opened successfully.", "INFO")
        except Exception as e:
            self.log_message(f"Failed to open Advanced Bout Scorer: {e}\n{traceback.format_exc()}", "ERROR")
            messagebox.showerror("Error", f"Failed to open Advanced Bout Scorer: {e}", parent=self.root)

    def _start_process(self, command_builder_func, process_attr, on_start_callback=None, on_finish_callback=None):
        if getattr(self, process_attr, None) and getattr(self, process_attr).poll() is None:
            self.log_message(f"A {process_attr} is already running.", "WARNING")
            messagebox.showwarning("In Progress", "A process is already running.", parent=self.root)
            return
        
        try:
            cmd, err = command_builder_func(self)
            if err:
                self.log_message(f"Command Builder Error: {err}", "ERROR")
                messagebox.showerror("Input Validation Error", err, parent=self.root)
                return
            
            self._start_and_stream_process(cmd, process_attr, on_finish_callback)
            if on_start_callback:
                on_start_callback()
        except Exception as e:
            self.log_message(f"Process Start Error: {e}\n{traceback.format_exc()}", "ERROR")
            messagebox.showerror("Process Error", f"Failed to start process: {e}", parent=self.root)

    def _start_training(self):
        self.log_message("Attempting to start training...", "INFO")
        def on_start():
            if hasattr(self, 'train_button'):
                self.train_button.config(state=tk.DISABLED)
                self.log_message("Disabled Start Training button.", "INFO")
        def on_finish():
            if hasattr(self, 'train_button'):
                self.train_button.config(state=tk.NORMAL)
                self.log_message("Enabled Start Training button.", "INFO")
            self.update_status("Training finished or stopped.")
        self._start_process(command_builder.build_training_command, 'training_process', on_start, on_finish)

    def _run_file_inference(self):
        self.log_message("Attempting to run file inference...", "INFO")
        def on_start():
            if hasattr(self, 'run_inference_button'):
                self.run_inference_button.config(state=tk.DISABLED)
                self.log_message("Disabled Run Inference button.", "INFO")
            if hasattr(self, 'stop_inference_button'):
                self.stop_inference_button.config(state=tk.NORMAL)
                self.log_message("Enabled Stop Inference button.", "INFO")
        def on_finish():
            if hasattr(self, 'run_inference_button'):
                self.run_inference_button.config(state=tk.NORMAL)
                self.log_message("Enabled Run Inference button.", "INFO")
            if hasattr(self, 'stop_inference_button'):
                self.stop_inference_button.config(state=tk.DISABLED)
                self.log_message("Disabled Stop Inference button.", "INFO")
            self.update_status("File inference finished or stopped.")
        self._start_process(command_builder.build_inference_command, 'file_inference_process', on_start, on_finish)

    def _stop_file_inference(self):
        self.log_message("Attempting to stop file inference...", "INFO")
        self._terminate_process('file_inference_process')
        if hasattr(self, 'run_inference_button'):
            self.run_inference_button.config(state=tk.NORMAL)
            self.log_message("Enabled Run Inference button.", "INFO")
        if hasattr(self, 'stop_inference_button'):
            self.stop_inference_button.config(state=tk.DISABLED)
            self.log_message("Disabled Stop Inference button.", "INFO")
        self.update_status("File inference stopped.")
        self.log_message("File inference stopped.", "INFO")

    def _start_webcam_inference(self):
        self.log_message("Attempting to start webcam inference...", "INFO")
        def on_start():
            if hasattr(self, 'start_webcam_button'):
                self.start_webcam_button.config(state=tk.DISABLED)
                self.log_message("Disabled Start Webcam button.", "INFO")
            if hasattr(self, 'stop_webcam_button'):
                self.stop_webcam_button.config(state=tk.NORMAL)
                self.log_message("Enabled Stop Webcam button.", "INFO")
        def on_finish():
            if hasattr(self, 'start_webcam_button'):
                self.start_webcam_button.config(state=tk.NORMAL)
                self.log_message("Enabled Start Webcam button.", "INFO")
            if hasattr(self, 'stop_webcam_button'):
                self.stop_webcam_button.config(state=tk.DISABLED)
                self.log_message("Disabled Stop Webcam button.", "INFO")
            self.update_status("Webcam inference finished or stopped.")
        self._start_process(command_builder.build_webcam_command, 'webcam_inference_process', on_start, on_finish)

    def _stop_webcam_inference(self):
        self.log_message("Attempting to stop webcam inference...", "INFO")
        self._terminate_process('webcam_inference_process')
        if hasattr(self, 'start_webcam_button'):
            self.start_webcam_button.config(state=tk.NORMAL)
            self.log_message("Enabled Start Webcam button.", "INFO")
        if hasattr(self, 'stop_webcam_button'):
            self.stop_webcam_button.config(state=tk.DISABLED)
            self.log_message("Disabled Stop Webcam button.", "INFO")
        self.update_status("Webcam inference stopped.")
        self.log_message("Webcam inference stopped.", "INFO")

    def _terminate_process(self, process_attribute):
        process = getattr(self, process_attribute, None)
        if process and process.poll() is None:
            self.log_message(f"Stopping {process_attribute} (PID: {process.pid})", "INFO")
            try:
                if sys.platform == "win32":
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], check=True, capture_output=True)
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=3)
            except Exception:
                process.kill()
            finally:
                setattr(self, process_attribute, None)
                self.log_message(f"Terminated {process_attribute}", "INFO")

    def _start_and_stream_process(self, command, process_attribute, on_finish_callback=None):
        self.log_message(f"--- Starting Command ---\n{' '.join(command)}\n------------------------", "INFO")
        try:
            popen_kwargs = {'stdout': subprocess.PIPE, 'stderr': subprocess.STDOUT, 'text': True, 'encoding': 'utf-8', 'errors': 'replace', 'bufsize': 1}
            if sys.platform != "win32":
                popen_kwargs['preexec_fn'] = os.setsid
            process = subprocess.Popen(command, **popen_kwargs)
            setattr(self, process_attribute, process)
            threading.Thread(target=self._stream_process_output, args=(process, on_finish_callback), daemon=True).start()
        except Exception as e:
            self.log_message(f"ERROR starting process: {e}\n{traceback.format_exc()}", "ERROR")
            if on_finish_callback:
                self.root.after(0, on_finish_callback)

    def _stream_process_output(self, process, on_finish_callback):
        for line in iter(process.stdout.readline, ''):
            if line:
                self.log_queue.put(line.strip())
        rc = process.wait()
        self.log_queue.put(f"--- Process finished with exit code {rc} ---")
        if on_finish_callback:
            self.root.after(0, on_finish_callback)

    def _process_offline_analytics(self):
        self.log_message("Attempting to start offline analytics...", "INFO")
        if self.analytics_thread and self.analytics_thread.is_alive():
            self.log_message("Analytics already running.", "WARNING")
            messagebox.showwarning("In Progress", "Analytics is already running.", parent=self.root)
            return
        try:
            params = {
                'yolo_folder': self.config.get_setting('analytics.yolo_output_path_var'),
                'video_file': self.config.get_setting('analytics.source_video_path_var'),
                'yaml_file': self.config.get_setting('analytics.roi_analytics_yaml_path_var'),
                'min_bout_frames': int(self.config.get_setting('analytics.min_bout_duration_var')),
                'max_gap_frames': int(self.config.get_setting('analytics.max_frame_gap_var')),
                'create_video': self.config.get_setting('analytics.create_video_output_var')
            }
            if not all([params['yolo_folder'], params['video_file'], params['yaml_file']]):
                raise ValueError("YOLO Folder, Source Video, and Project YAML must be set.")
            if not os.path.isdir(params['yolo_folder']):
                raise ValueError(f"YOLO output directory does not exist: {params['yolo_folder']}")
            txt_files = [f for f in os.listdir(params['yolo_folder']) if f.endswith('.txt')]
            if not txt_files:
                raise ValueError(f"No .txt files found in YOLO output directory: {params['yolo_folder']}. Please run inference first or select a valid directory.")
            self.log_message(f"Starting Bout Analysis with {len(txt_files)} .txt files in {params['yolo_folder']}", "INFO")
            self.update_status("Analytics started...")
            if hasattr(self, 'process_btn'):
                self.process_btn.config(state=tk.DISABLED)
                self.log_message("Disabled Process Analytics button.", "INFO")
            self.analytics_thread = threading.Thread(target=self._analytics_worker, args=(params,), daemon=True)
            self.analytics_thread.start()
        except Exception as e:
            self.log_message(f"Analytics Configuration Error: {e}\n{traceback.format_exc()}", "ERROR")
            messagebox.showerror("Configuration Error", str(e), parent=self.root)
            if hasattr(self, 'process_btn'):
                self.process_btn.config(state=tk.NORMAL)
                self.log_message("Enabled Process Analytics button.", "INFO")
            self.update_status("Ready.")

    def _analytics_worker(self, params):
        try:
            output_folder = os.path.join(os.path.dirname(params['video_file']), "bout_analysis_results")
            base_name = os.path.splitext(os.path.basename(params['video_file']))[0]
            per_frame_df, _, _ = bout_analyzer.load_and_preprocess_data(params['yaml_file'], params['yolo_folder'])
            self.detailed_bouts_df, self.summary_bouts_df, excel_path = bout_analyzer.save_analysis_outputs(
                per_frame_df, params['yaml_file'], output_folder, params['max_gap_frames'], params['min_bout_frames'], 30, base_name
            )
            # Debug detailed_bouts_df
            self.log_message(f"detailed_bouts_df columns: {self.detailed_bouts_df.columns.tolist()}", "DEBUG")
            self.log_message(f"detailed_bouts_df shape: {self.detailed_bouts_df.shape}", "DEBUG")
            if 'Track ID' in self.detailed_bouts_df.columns:
                unique_tracks = self.detailed_bouts_df['Track ID'].unique().astype(int).tolist()
                self.log_message(f"Unique Track IDs: {unique_tracks}", "DEBUG")
            else:
                self.log_message("No 'Track ID' column in detailed_bouts_df.", "WARNING")
            self.root.after(0, self._display_analytics_results)
            if hasattr(self, 'behavior_listbox') and not self.detailed_bouts_df.empty:
                self.root.after(0, lambda: self._refresh_clustering_behaviors())
            if self.detailed_bouts_df.empty:
                self.root.after(0, lambda: messagebox.showinfo("Analysis Complete", "No bouts were found matching the criteria.", parent=self.root))
                self.log_message("No bouts found in analysis.", "INFO")
                return
            if params['create_video']:
                video_path = os.path.join(output_folder, f"{base_name}_annotated.mp4")
                with open(params['yaml_file'], 'r') as f:
                    data = yaml.safe_load(f)
                rois = {name: np.array(points, dtype=np.int32) for name, points in data.get('rois', {}).items()}
                video_creator.create_annotated_video(params['video_file'], video_path, params['yolo_folder'], rois, self.detailed_bouts_df)
            success_msg = f"Analysis complete! Results saved in:\n{output_folder}"
            if excel_path:
                success_msg += f"\nExcel summary: {excel_path}"
            self.root.after(0, lambda: messagebox.showinfo("Success", success_msg, parent=self.root))
            self.log_message(f"Analysis completed. Results saved in {output_folder}", "INFO")
        except Exception as e:
            self.log_message(f"ERROR during analytics: {e}\n{traceback.format_exc()}", "ERROR")
            self.root.after(0, lambda: messagebox.showerror("Processing Error", f"An unexpected error occurred: {e}", parent=self.root))
        finally:
            if hasattr(self, 'process_btn'):
                self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
                self.log_message("Enabled Process Analytics button.", "INFO")
            self.root.after(0, lambda: self.update_status("Ready."))

    def _refresh_clustering_behaviors(self):
        if hasattr(self, 'behavior_listbox'):
            try:
                self.behavior_listbox.delete(0, tk.END)
                self.track_listbox.delete(0, tk.END)
                if not self.detailed_bouts_df.empty:
                    behaviors = sorted(self.detailed_bouts_df['Behavior'].unique())
                    tracks = sorted(self.detailed_bouts_df['Track ID'].unique().astype(int)) if 'Track ID' in self.detailed_bouts_df.columns else [0]
                    for behavior in behaviors:
                        self.behavior_listbox.insert(tk.END, behavior)
                    for track in tracks:
                        self.track_listbox.insert(tk.END, str(track))
                    self.log_message(f"Refreshed clustering tab with {len(behaviors)} behaviors, {len(tracks)} tracks: {tracks}", "INFO")
                else:
                    self.log_message("No bout data available for clustering tab refresh.", "WARNING")
                    self.track_listbox.insert(tk.END, "0")  # Fallback
                    self.log_message("Added fallback track ID 0.", "INFO")
            except Exception as e:
                self.log_message(f"Error refreshing clustering tab: {e}", "ERROR")
                messagebox.showerror("Error", f"Failed to refresh clustering tab: {e}", parent=self.root)
        else:
            self.log_message("Behavior listbox not initialized in clustering tab.", "WARNING")

    def _display_analytics_results(self):
        if hasattr(self, 'analytics_treeview'):
            try:
                tree = self.analytics_treeview
                for item in tree.get_children():
                    tree.delete(item)
                if self.detailed_bouts_df.empty:
                    self.log_message("No bouts to display in analytics results.", "INFO")
                    return
                tree["columns"] = list(self.detailed_bouts_df.columns)
                tree["show"] = "headings"
                for col in tree["columns"]:
                    tree.heading(col, text=col)
                    tree.column(col, width=100)
                    self.log_message(f"Column {col} configured in analytics treeview", "INFO")
                for index, row in self.detailed_bouts_df.iterrows():
                    tree.insert("", "end", iid=index, values=list(row))
                self.log_message(f"Displayed {len(self.detailed_bouts_df)} bouts in analytics results.", "INFO")
            except Exception as e:
                self.log_message(f"Error displaying analytics results: {e}", "ERROR")
                messagebox.showerror("Error", f"Failed to display analytics results: {e}", parent=self.root)

    def get_available_behaviors(self):
        behaviors = sorted(self.detailed_bouts_df['Behavior'].unique()) if not self.detailed_bouts_df.empty else []
        if not behaviors:
            behaviors = [b['name'] for b in sorted(self.config.get_setting('setup.behaviors_list'), key=lambda b: b['id'])]
        self.log_message(f"Returning {len(behaviors)} behaviors: {behaviors}", "INFO")
        return behaviors

    def _stop_all_processes(self):
        self.update_status("Stopping all background processes...")
        self._terminate_process('training_process')
        self._terminate_process('file_inference_process')
        self._terminate_process('webcam_inference_process')
        self.log_message("All processes stopped.", "INFO")

    def _on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?", parent=self.root):
            self._stop_all_processes()
            self.root.destroy()

    def update_title(self):
        project_path = self.config.project_file_path
        if project_path:
            self.root.title(f"{self.base_title} - {os.path.basename(project_path)}")
        else:
            self.root.title(self.base_title)

    def update_status(self, message):
        self.status_var.set(message)
        self.log_message(f"Status updated: {message}", "INFO")

    def update_log(self, message, level="INFO"):
        self.log_queue.put(f"[{level}] {message}")

    def log_message(self, message, level="INFO"):
        self.update_log(message, level)

    def _check_log_queue(self):
        while not self.log_queue.empty():
            line = self.log_queue.get_nowait()
            if hasattr(self, 'log_text_widget'):
                self.log_text_widget.config(state=tk.NORMAL)
                self.log_text_widget.insert(tk.END, line + '\n')
                self.log_text_widget.see(tk.END)
                self.log_text_widget.config(state=tk.DISABLED)
            
        self.root.after(100, self._check_log_queue)

if __name__ == '__main__':
    root = tk.Tk()
    app = YoloApp(root)
    root.mainloop()
import os
import pandas as pd
import numpy as np

class YoloParser:
    def __init__(self, app):
        self.app = app

    def validate_yolo_files(self, yolo_folder, sample_size=10):
        """
        Accepts YOLO-Pose txt with:
          - 5 meta cols (class, x, y, w, h) OR 6 meta cols (class, x, y, w, h, conf)
          - optional trailing track_id column
        Ensures the remaining keypoint block is divisible by 3.
        """
        try:
            txt_files = [f for f in os.listdir(yolo_folder) if f.endswith('.txt')]
            if not txt_files:
                return False, 0, "No .txt files found in YOLO folder."
            if sample_size > 0:
                txt_files = txt_files[:min(sample_size, len(txt_files))]

            def looks_like_track_col(series):
                s = series.dropna()
                if s.empty: return False
                try:
                    s_vals = s.to_numpy(dtype=float)
                except Exception:
                    return False
                if not (s_vals == s_vals.round()).all():
                    return False
                if not ((s_vals >= 0) & (s_vals <= 1e9)).all():
                    return False
                return True

            detected_counts = []
            for filename in txt_files:
                filepath = os.path.join(yolo_folder, filename)
                try:
                    df_raw = pd.read_csv(filepath, sep=' ', header=None, dtype=str)
                    df = df_raw.apply(pd.to_numeric, errors='coerce')
                except Exception as e:
                    return False, 0, f"Invalid data in {filename}: {e}"
                total_cols = len(df.columns)
                if total_cols < 6:
                    return False, 0, f"Invalid format in {filename}: {total_cols} columns."

                has_track = looks_like_track_col(df.iloc[:, -1])
                track_col = (total_cols - 1) if has_track else None

                kp_count = None
                for meta_len in (6, 5):
                    kp_end = track_col if track_col is not None else total_cols
                    kp_vals = kp_end - meta_len
                    if kp_vals > 0 and kp_vals % 3 == 0:
                        kp_count = kp_vals // 3
                        break
                if kp_count is None:
                    return False, 0, f"Could not infer keypoints in {filename} (cols={total_cols}, track_id={has_track})."
                detected_counts.append(kp_count)

            uniq = set(detected_counts)
            if len(uniq) != 1:
                return False, 0, f"Inconsistent keypoint counts across files: {sorted(uniq)}"
            return True, detected_counts[0], ""
        except Exception as e:
            return False, 0, f"Validation error: {e}"

    def get_pose_from_frame(self, frame_num, filename=None, track_id=None, expected_keypoints=None):
        """
        Extracts pose data from a YOLO output file for a given frame and (optional) track ID.
        Handles both 5-meta (no conf) and 6-meta (with conf) layouts, and optional trailing track_id.
        Returns: (pose_xy_flat: np.ndarray shape (2K,), detected_keypoints: int) or (None, None)
        """
        try:
            yolo_folder = self.app.config.get_setting('analytics.yolo_output_path_var')
            if not yolo_folder:
                self.app.log_message("YOLO output folder not set.", "ERROR")
                return None, None
            if filename is None:
                filename = f"frame_{frame_num}.txt"
            filepath = os.path.join(yolo_folder, filename)
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                self.app.log_message(f"YOLO output file not found or empty: {filepath}", "WARNING")
                return None, None

            try:
                df_raw = pd.read_csv(filepath, sep=' ', header=None, dtype=str)
                df = df_raw.apply(pd.to_numeric, errors='coerce')
            except Exception as e:
                self.app.log_message(f"Failed to load {filename}: {e}", "ERROR")
                return None, None

            total_cols = len(df.columns)
            if total_cols < 6:
                self.app.log_message(f"Invalid YOLO format in {filename}: {total_cols} columns", "ERROR")
                return None, None

            def looks_like_track_col(series):
                s = series.dropna()
                if s.empty: return False
                try:
                    s_vals = s.to_numpy(dtype=float)
                except Exception:
                    return False
                if not (s_vals == s_vals.round()).all():
                    return False
                if not ((s_vals >= 0) & (s_vals <= 1e9)).all():
                    return False
                return True

            has_track = looks_like_track_col(df.iloc[:, -1])
            track_col = (total_cols - 1) if has_track else None

            chosen_meta_len = None
            for meta_len in (6, 5):
                kp_end = track_col if track_col is not None else total_cols
                kp_vals = kp_end - meta_len
                if kp_vals > 0 and kp_vals % 3 == 0:
                    chosen_meta_len = meta_len
                    break
            if chosen_meta_len is None:
                self.app.log_message(f"Could not infer keypoint layout for {filename}", "ERROR")
                return None, None

            keypoint_start = chosen_meta_len
            keypoint_end = track_col if track_col is not None else total_cols
            detected_keypoints = (keypoint_end - keypoint_start) // 3

            if expected_keypoints is not None and detected_keypoints != expected_keypoints:
                self.app.log_message(
                    f"Keypoint count mismatch in {filename}: found {detected_keypoints}, expected {expected_keypoints}",
                    "WARNING"
                )

            detection_series = None
            if track_id is not None:
                if has_track:
                    try:
                        track_series = df.iloc[:, -1].round().astype("Int64")
                        mask = track_series == int(track_id)
                        df_track = df[mask]
                    except Exception:
                        df_track = pd.DataFrame()
                    if df_track.empty:
                        self.app.log_message(
                            f"No detection for track {track_id} in frame file {filename}",
                            "DEBUG"
                        )
                        return None, detected_keypoints
                    detection_series = df_track.iloc[0]
                else:
                    self.app.log_message(
                        f"Track ID {track_id} requested but {filename} lacks tracking data; skipping frame.",
                        "WARNING",
                    )
                    return None, detected_keypoints
            else:
                detection_series = df.iloc[0]

            kp_slice = detection_series.iloc[keypoint_start:keypoint_end].to_numpy()
            if kp_slice.size != detected_keypoints * 3:
                self.app.log_message(
                    f"Keypoint slice size mismatch in {filename}: got {kp_slice.size}, expected {detected_keypoints*3}",
                    "DEBUG"
                )
                return None, None

            kp_triplets = kp_slice.reshape(-1, 3)
            keypoints_xy = kp_triplets[:, :2].astype(float).flatten()

            if not np.all(np.isfinite(keypoints_xy)):
                self.app.log_message(f"Non-finite values in keypoints for {filename}", "DEBUG")
                return None, None

            return keypoints_xy, detected_keypoints

        except Exception as e:
            self.app.log_message(f"Error extracting pose for track {track_id} in {filename}: {e}", "ERROR")
            return None, None

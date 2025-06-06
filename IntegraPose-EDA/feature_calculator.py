import pandas as pd
import numpy as np
import itertools
import re
from scipy.spatial import ConvexHull, distance
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import app_config 

class FeatureCalculator:
    def __init__(self, keypoint_names_list, defined_skeleton=None):
        """
        Initializes the FeatureCalculator.

        Args:
            keypoint_names_list (list): Ordered list of keypoint names.
            defined_skeleton (list of tuples, optional): List of (kp1_name, kp2_name) tuples
                                                         representing skeleton bones. Defaults to None.
        """
        if not isinstance(keypoint_names_list, list):
            raise ValueError("keypoint_names_list must be a list.")
        self.keypoint_names_list = keypoint_names_list
        self.defined_skeleton = defined_skeleton if defined_skeleton is not None else []
        self.coord_suffix = "_norm" # Suffix for normalized coordinates used in calculations

    def update_skeleton(self, defined_skeleton):
        """Updates the skeleton definition."""
        self.defined_skeleton = defined_skeleton if defined_skeleton is not None else []

    def get_potential_feature_names(self, generate_all_geometric=False):
        """
        Generates a list of potential geometric feature names based on keypoints and skeleton.

        Args:
            generate_all_geometric (bool): If True, generates all pairwise distances and angles,
                                           ignoring the defined_skeleton for those.
                                           If False, uses defined_skeleton for specific distances/angles.
        Returns:
            list: Sorted list of unique potential feature names.
        """
        if not self.keypoint_names_list:
            return []

        potential_features = []

        # 1. General pose features (calculated from normalized keypoints)
        potential_features.append(f"BBox_AspectRatio") # This one uses original bbox, not normalized KPs directly
        potential_features.append(f"Pose_Width{self.coord_suffix}")
        potential_features.append(f"Pose_Height{self.coord_suffix}")
        potential_features.append(f"Pose_ConvexHullArea{self.coord_suffix}")
        potential_features.append(f"Pose_ConvexHullPerimeter{self.coord_suffix}")
        potential_features.append(f"Pose_Circularity{self.coord_suffix}") # Area/Perimeter based
        potential_features.append(f"Pose_Orientation{self.coord_suffix}") # PCA-based
        potential_features.append(f"Pose_Eccentricity{self.coord_suffix}") # PCA-based

        # 2. Keypoint to Centroid distances
        for kp_name in self.keypoint_names_list:
            potential_features.append(f"Dist_Centroid_to_{kp_name}{self.coord_suffix}")

        # 3. Distances and Angles
        if generate_all_geometric or not self.defined_skeleton:
            # All pairwise distances
            if len(self.keypoint_names_list) >= 2:
                for kp1, kp2 in itertools.combinations(self.keypoint_names_list, 2):
                    potential_features.append(f"Dist{self.coord_suffix}({kp1},{kp2})")
            # All possible angles (using each KP as a potential center)
            if len(self.keypoint_names_list) >= 3:
                for kp_center in self.keypoint_names_list:
                    other_kps = [kp for kp in self.keypoint_names_list if kp != kp_center]
                    if len(other_kps) >= 2:
                        for kp_arm1, kp_arm2 in itertools.combinations(other_kps, 2):
                            endpoints = sorted((kp_arm1, kp_arm2)) # Sort for consistent naming
                            potential_features.append(f"Angle{self.coord_suffix}({endpoints[0]}_{kp_center}_{endpoints[1]})")
        else: # Skeleton-based distances and angles
            # Distances for defined bones
            for kp1, kp2 in self.defined_skeleton:
                potential_features.append(f"Dist{self.coord_suffix}({kp1},{kp2})")

            # Angles at joints defined by the skeleton
            adj = {kp: [] for kp in self.keypoint_names_list}
            for u, v in self.defined_skeleton: # u,v should be sorted already if skeleton managed properly
                adj[u].append(v)
                adj[v].append(u)

            for center_kp, connected_kps in adj.items():
                if len(connected_kps) >= 2:
                    for kp_arm1, kp_arm3 in itertools.combinations(connected_kps, 2):
                        endpoints = sorted((kp_arm1, kp_arm3))
                        potential_features.append(f"Angle{self.coord_suffix}({endpoints[0]}_{center_kp}_{endpoints[1]})")
            
            # Ratios of bone lengths (if at least 2 bones defined and non-overlapping)
            if len(self.defined_skeleton) >= 2:
                for bone1_idx, bone2_idx in itertools.combinations(range(len(self.defined_skeleton)), 2):
                    kp1a, kp1b = self.defined_skeleton[bone1_idx]
                    kp2a, kp2b = self.defined_skeleton[bone2_idx]
                    # Ensure distinct bones for ratio (no shared keypoints)
                    if len(set([kp1a, kp1b, kp2a, kp2b])) == 4:
                        # Sort bone representation for consistent naming, e.g. Dist(A,B) not Dist(B,A)
                        bone1_str = f"Dist({min(kp1a,kp1b)},{max(kp1a,kp1b)})"
                        bone2_str = f"Dist({min(kp2a,kp2b)},{max(kp2a,kp2b)})"
                        potential_features.append(f"Ratio_{bone1_str}_to_{bone2_str}{self.coord_suffix}")


        return sorted(list(set(potential_features))) # Unique and sorted

    def _calculate_single_distance(self, row, kp1_name, kp2_name):
        kp1x, kp1y = row.get(f'{kp1_name}_x{self.coord_suffix}'), row.get(f'{kp1_name}_y{self.coord_suffix}')
        kp2x, kp2y = row.get(f'{kp2_name}_x{self.coord_suffix}'), row.get(f'{kp2_name}_y{self.coord_suffix}')
        if pd.isna(kp1x) or pd.isna(kp1y) or pd.isna(kp2x) or pd.isna(kp2y):
            return np.nan
        return np.sqrt((kp1x - kp2x)**2 + (kp1y - kp2y)**2)

    def _calculate_single_angle(self, row, kp1_name, kp2_center_name, kp3_name):
        p1x, p1y = row.get(f'{kp1_name}_x{self.coord_suffix}'), row.get(f'{kp1_name}_y{self.coord_suffix}')
        p2x, p2y = row.get(f'{kp2_center_name}_x{self.coord_suffix}'), row.get(f'{kp2_center_name}_y{self.coord_suffix}')
        p3x, p3y = row.get(f'{kp3_name}_x{self.coord_suffix}'), row.get(f'{kp3_name}_y{self.coord_suffix}')

        if pd.isna(p1x) or pd.isna(p1y) or pd.isna(p2x) or pd.isna(p2y) or pd.isna(p3x) or pd.isna(p3y):
            return np.nan

        v21 = np.array([p1x - p2x, p1y - p2y])
        v23 = np.array([p3x - p2x, p3y - p2y])
        dot_product = np.dot(v21, v23)
        norm_v21 = np.linalg.norm(v21)
        norm_v23 = np.linalg.norm(v23)

        if norm_v21 < app_config.EPSILON or norm_v23 < app_config.EPSILON:
            return np.nan
        
        cos_angle = np.clip(dot_product / (norm_v21 * norm_v23), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def calculate_features(self, df_processed_input, selected_feature_names_to_calc, progress_callback=None):
        """
        Calculates selected geometric features and adds them to a copy of the input DataFrame.

        Args:
            df_processed_input (pd.DataFrame): DataFrame containing preprocessed data
                                              (with normalized keypoints and original bbox columns).
            selected_feature_names_to_calc (list): List of feature names to calculate.
            progress_callback (function, optional): Callback for progress updates.
                                                   Takes (current_value, max_value, message).
        Returns:
            pd.DataFrame: DataFrame with added feature columns, or None if input is invalid.
            list: List of names of the features that were actually calculated/added.
        """
        if df_processed_input is None or df_processed_input.empty:
            print("FeatureCalculator: Input DataFrame is empty or None.")
            return None, []
        if not selected_feature_names_to_calc:
            print("FeatureCalculator: No features selected for calculation.")
            return df_processed_input.copy(), [] # Return original if no features to calc

        # Start with a copy of the processed data. Features will be added to this.
        df_features_output = df_processed_input.copy()
        
        # Store calculated series here before assigning to df_features_output
        newly_calculated_feature_data = {}
        actually_calculated_cols = []

        if progress_callback:
            progress_callback(0, len(selected_feature_names_to_calc), "Starting feature calculation...")

        # --- Helper data structures for features that compute multiple properties at once ---
        # These will be populated on-the-fly per row if needed
        # No, these need to be Series computed once per feature type requested.
        _hull_props_series = None
        _pca_shape_props_series = None

        for idx, feature_name_full in enumerate(selected_feature_names_to_calc):
            if progress_callback:
                progress_callback(idx + 1, len(selected_feature_names_to_calc), f"Calculating: {feature_name_full}")

            col_name = feature_name_full # The name in the listbox is the desired column name
            calculated_series = None # Store the Pandas Series for the current feature

            try:
                # --- BBox Aspect Ratio (uses original bbox columns from df_processed_input) ---
                if feature_name_full == "BBox_AspectRatio":
                    if all(c in df_features_output for c in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']):
                        height = (df_features_output['bbox_y2'] - df_features_output['bbox_y1'])
                        height = height.replace(0, app_config.EPSILON)
                        calculated_series = (df_features_output['bbox_x2'] - df_features_output['bbox_x1']) / height
                    else:
                        print(f"Warning: Missing bbox columns for BBox_AspectRatio. Skipping.")
                        calculated_series = pd.Series(np.nan, index=df_features_output.index)

                # --- Pose Dimensions (Width, Height) from normalized keypoints ---
                elif feature_name_full == f"Pose_Width{self.coord_suffix}":
                    def get_pose_dim_width(row, kps, suffix):
                        coords_x = [row.get(f'{kp}_x{suffix}') for kp in kps]
                        valid_coords_x = [c for c in coords_x if pd.notna(c)]
                        return (max(valid_coords_x) - min(valid_coords_x)) if len(valid_coords_x) > 1 else 0.0
                    calculated_series = df_features_output.apply(get_pose_dim_width, axis=1, args=(self.keypoint_names_list, self.coord_suffix))

                elif feature_name_full == f"Pose_Height{self.coord_suffix}":
                    def get_pose_dim_height(row, kps, suffix):
                        coords_y = [row.get(f'{kp}_y{suffix}') for kp in kps]
                        valid_coords_y = [c for c in coords_y if pd.notna(c)]
                        return (max(valid_coords_y) - min(valid_coords_y)) if len(valid_coords_y) > 1 else 0.0
                    calculated_series = df_features_output.apply(get_pose_dim_height, axis=1, args=(self.keypoint_names_list, self.coord_suffix))
                
                # --- Convex Hull Properties ---
                elif feature_name_full.startswith("Pose_ConvexHull") or feature_name_full == f"Pose_Circularity{self.coord_suffix}":
                    if _hull_props_series is None: # Calculate only once if any hull feature is requested
                        def get_hull_props_for_series(row, kps, suffix):
                            points_np = []
                            for kp_name_ch in kps:
                                x, y = row.get(f'{kp_name_ch}_x{suffix}'), row.get(f'{kp_name_ch}_y{suffix}')
                                if pd.notna(x) and pd.notna(y): points_np.append([x,y])
                            if len(points_np) < 3: return {'area': np.nan, 'perimeter': np.nan, 'circularity': np.nan}
                            try:
                                hull = ConvexHull(np.array(points_np))
                                area = hull.volume # Area for 2D
                                perimeter = hull.area # Perimeter for 2D
                                circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > app_config.EPSILON else np.nan
                                return {'area': area, 'perimeter': perimeter, 'circularity': circularity}
                            except Exception: return {'area': np.nan, 'perimeter': np.nan, 'circularity': np.nan}
                        _hull_props_series = df_features_output.apply(get_hull_props_for_series, axis=1, args=(self.keypoint_names_list, self.coord_suffix))

                    if feature_name_full == f"Pose_ConvexHullArea{self.coord_suffix}":
                        calculated_series = _hull_props_series.apply(lambda x: x['area'] if isinstance(x,dict) else np.nan)
                    elif feature_name_full == f"Pose_ConvexHullPerimeter{self.coord_suffix}":
                        calculated_series = _hull_props_series.apply(lambda x: x['perimeter'] if isinstance(x,dict) else np.nan)
                    elif feature_name_full == f"Pose_Circularity{self.coord_suffix}":
                        calculated_series = _hull_props_series.apply(lambda x: x['circularity'] if isinstance(x,dict) else np.nan)

                # --- PCA-based Shape Properties (Orientation, Eccentricity) ---
                elif feature_name_full.startswith("Pose_Orientation") or feature_name_full.startswith("Pose_Eccentricity"):
                    if _pca_shape_props_series is None: # Calculate only once
                        def get_pca_shape_props_for_series(row, kps, suffix):
                            points = []
                            for kp_name_pca in kps:
                                x, y = row.get(f'{kp_name_pca}_x{suffix}'), row.get(f'{kp_name_pca}_y{suffix}')
                                if pd.notna(x) and pd.notna(y): points.append([x,y])
                            if len(points) < 2: return {'orientation': np.nan, 'eccentricity': np.nan}
                            try:
                                data = np.array(points)
                                pca_shape = PCA(n_components=2).fit(data)
                                major_axis_vector = pca_shape.components_[0]
                                orientation = np.degrees(np.arctan2(major_axis_vector[1], major_axis_vector[0]))
                                eccentricity = np.nan
                                if len(pca_shape.explained_variance_) == 2:
                                    lambda_major, lambda_minor = pca_shape.explained_variance_
                                    if lambda_major > app_config.EPSILON: # Avoid division by zero or sqrt of negative
                                        eccentricity = np.sqrt(max(0, 1 - (lambda_minor / lambda_major)))
                                elif len(pca_shape.explained_variance_) == 1: # Collinear points
                                    eccentricity = 1.0
                                return {'orientation': orientation, 'eccentricity': eccentricity}
                            except Exception: return {'orientation': np.nan, 'eccentricity': np.nan}
                        _pca_shape_props_series = df_features_output.apply(get_pca_shape_props_for_series, axis=1, args=(self.keypoint_names_list, self.coord_suffix))

                    if feature_name_full == f"Pose_Orientation{self.coord_suffix}":
                        calculated_series = _pca_shape_props_series.apply(lambda x: x['orientation'] if isinstance(x,dict) else np.nan)
                    elif feature_name_full == f"Pose_Eccentricity{self.coord_suffix}":
                        calculated_series = _pca_shape_props_series.apply(lambda x: x['eccentricity'] if isinstance(x,dict) else np.nan)
                
                # --- Distance from Centroid to Keypoints ---
                elif feature_name_full.startswith(f"Dist_Centroid_to_"):
                    kp_target_for_centroid_dist = feature_name_full.replace(f"Dist_Centroid_to_", "").replace(f"{self.coord_suffix}", "")
                    def get_dist_to_centroid(row, target_kp, all_kps, suffix):
                        kps_coords_list = []
                        for kp_n in all_kps:
                            x, y = row.get(f'{kp_n}_x{suffix}'), row.get(f'{kp_n}_y{suffix}')
                            if pd.notna(x) and pd.notna(y): kps_coords_list.append([x,y])
                        if not kps_coords_list: return np.nan
                        
                        centroid = np.mean(np.array(kps_coords_list), axis=0)
                        target_x, target_y = row.get(f'{target_kp}_x{suffix}'), row.get(f'{target_kp}_y{suffix}')
                        if pd.isna(target_x) or pd.isna(target_y) or pd.isna(centroid[0]) or pd.isna(centroid[1]): return np.nan
                        return distance.euclidean(centroid, [target_x, target_y])
                    calculated_series = df_features_output.apply(get_dist_to_centroid, axis=1, args=(kp_target_for_centroid_dist, self.keypoint_names_list, self.coord_suffix))

                # --- Pairwise Keypoint Distances ---
                elif feature_name_full.startswith(f"Dist{self.coord_suffix}("):
                    match = re.match(rf"Dist{self.coord_suffix}\((.+?),(.+?)\)", feature_name_full)
                    if match:
                        kp1, kp2 = match.groups()
                        calculated_series = df_features_output.apply(lambda r: self._calculate_single_distance(r, kp1, kp2), axis=1)
                
                # --- Angles between Three Keypoints ---
                elif feature_name_full.startswith(f"Angle{self.coord_suffix}("):
                    match = re.match(rf"Angle{self.coord_suffix}\((.+?)_(.+?)_(.+?)\)", feature_name_full) # kp1_center_kp3
                    if match:
                        kp_arm1, kp_center, kp_arm3 = match.groups()
                        calculated_series = df_features_output.apply(lambda r: self._calculate_single_angle(r, kp_arm1, kp_center, kp_arm3), axis=1)
                
                # --- Ratios of Distances ---
                elif feature_name_full.startswith(f"Ratio_Dist("):
                    # Example: Ratio_Dist(KP1,KP2)_to_Dist(KP3,KP4)_norm
                    match = re.match(rf"Ratio_Dist\((.+?),(.+?)\)_to_Dist\((.+?),(.+?)\){self.coord_suffix}", feature_name_full)
                    if match:
                        kp1a, kp1b, kp2a, kp2b = match.groups()
                        # Calculate the two distance series first
                        dist1_series = df_features_output.apply(lambda r: self._calculate_single_distance(r, kp1a, kp1b), axis=1)
                        dist2_series = df_features_output.apply(lambda r: self._calculate_single_distance(r, kp2a, kp2b), axis=1)
                        # Calculate ratio, avoid division by zero
                        calculated_series = dist1_series / (dist2_series + app_config.EPSILON)
                    else: # Try matching the sorted name format from get_potential_feature_names
                         # Ratio_Dist(min(A,B),max(A,B))_to_Dist(min(C,D),max(C,D))_norm
                        match_sorted = re.match(rf"Ratio_Dist\({re.escape(f'Dist({min(kp1a,kp1b)},{max(kp1a,kp1b)})')}\)_to_Dist\({re.escape(f'Dist({min(kp2a,kp2b)},{max(kp2a,kp2b)})')}\){self.coord_suffix}", feature_name_full)
                        # This regex part for sorted ratio is tricky and might need refinement if the above does not work.
                        # The original parsing was simpler. The key is to extract the 4 KPs.
                        # For now, let's assume the non-sorted name from selected_feature_names_to_calc is sufficient if regex matches.
                        # A more robust way would be to parse the feature name string more generally for the four KPs.
                        if not calculated_series: # If first match failed
                            print(f"Warning: Could not parse Ratio feature name: {feature_name_full}")
                            calculated_series = pd.Series(np.nan, index=df_features_output.index)


                else:
                    print(f"Warning: Unknown feature format selected or logic not implemented: {feature_name_full}")
                    calculated_series = pd.Series(np.nan, index=df_features_output.index)

            except Exception as e_feat_calc:
                print(f"Error calculating feature '{feature_name_full}': {e_feat_calc}\n{traceback.format_exc()}")
                calculated_series = pd.Series(np.nan, index=df_features_output.index) # Fill with NaN on error

            if calculated_series is not None:
                newly_calculated_feature_data[col_name] = calculated_series
                if not calculated_series.isnull().all(): # Check if any valid values were calculated
                    actually_calculated_cols.append(col_name)
        
        # Assign all newly calculated series to the output DataFrame
        for f_name, f_series in newly_calculated_feature_data.items():
            df_features_output[f_name] = f_series

        if progress_callback:
            progress_callback(len(selected_feature_names_to_calc), len(selected_feature_names_to_calc), "Feature calculation complete.")

        return df_features_output, actually_calculated_cols

    # --- Placeholder for Kinematic Features ---
    def calculate_kinematic_features(self, df_input, fps, progress_callback=None):
        """
        Calculates kinematic features like keypoint velocities and accelerations.
        This is a more advanced feature requiring data sorted by track and frame.

        Args:
            df_input (pd.DataFrame): DataFrame with pose data, MUST include 'track_id' (or similar),
                                     'frame_id', and normalized keypoint coordinates.
                                     Data should be pre-sorted by track_id and frame_id.
            fps (float): Frames per second of the video, for scaling rates of change.
            progress_callback (function, optional): Callback for progress updates.

        Returns:
            pd.DataFrame: DataFrame with added kinematic feature columns.
            list: List of names of the kinematic features calculated.
        """
        # df_kinematic_features = df_input.copy()
        # calculated_kinematic_cols = []
        # time_delta = 1.0 / fps if fps > 0 else 1.0

        # if 'track_id' not in df_kinematic_features.columns or 'frame_id' not in df_kinematic_features.columns:
        #     print("Warning: 'track_id' or 'frame_id' missing. Cannot calculate kinematic features.")
        #     return df_kinematic_features, []

        # # Ensure sorting for correct diff calculation
        # df_kinematic_features = df_kinematic_features.sort_values(by=['track_id', 'frame_id'])

        # for kp_name in self.keypoint_names_list:
        #     x_col = f'{kp_name}_x{self.coord_suffix}'
        #     y_col = f'{kp_name}_y{self.coord_suffix}'
        #     vx_col = f'{kp_name}_vx{self.coord_suffix}'
        #     vy_col = f'{kp_name}_vy{self.coord_suffix}'
        #     speed_col = f'{kp_name}_speed{self.coord_suffix}'
            
        #     if x_col in df_kinematic_features.columns and y_col in df_kinematic_features.columns:
        #         # Calculate delta per track_id
        #         # Group by track_id and then calculate diff, then scale by time_delta
        #         # Important: first diff() within a group will be NaN, which is correct.
        #         delta_x = df_kinematic_features.groupby('track_id')[x_col].diff()
        #         delta_y = df_kinematic_features.groupby('track_id')[y_col].diff()

        #         df_kinematic_features[vx_col] = delta_x / time_delta
        #         df_kinematic_features[vy_col] = delta_y / time_delta
        #         df_kinematic_features[speed_col] = np.sqrt(df_kinematic_features[vx_col]**2 + df_kinematic_features[vy_col]**2)
                
        #         calculated_kinematic_cols.extend([vx_col, vy_col, speed_col])
        
        # # Acceleration can be calculated by taking a diff of velocities
        # # for kp_name in self.keypoint_names_list:
        # #     vx_col, vy_col = f'{kp_name}_vx{self.coord_suffix}', f'{kp_name}_vy{self.coord_suffix}'
        # #     ax_col, ay_col = f'{kp_name}_ax{self.coord_suffix}', f'{kp_name}_ay{self.coord_suffix}'
        # #     accel_mag_col = f'{kp_name}_accel{self.coord_suffix}'
            
        # #     if vx_col in df_kinematic_features.columns and vy_col in df_kinematic_features.columns:
        # #         delta_vx = df_kinematic_features.groupby('track_id')[vx_col].diff()
        # #         delta_vy = df_kinematic_features.groupby('track_id')[vy_col].diff()
        # #         df_kinematic_features[ax_col] = delta_vx / time_delta
        # #         df_kinematic_features[ay_col] = delta_vy / time_delta
        # #         df_kinematic_features[accel_mag_col] = np.sqrt(df_kinematic_features[ax_col]**2 + df_kinematic_features[ay_col]**2)
        # #         calculated_kinematic_cols.extend([ax_col, ay_col, accel_mag_col])

        # print("Kinematic feature calculation (velocity, acceleration) is a placeholder and needs to be fully implemented and tested.")
        # print("Ensure 'track_id' and 'frame_id' are present and data is sorted correctly.")
        # return df_input.copy(), [] # Return original for now
        raise NotImplementedError("calculate_kinematic_features is not yet fully implemented.")
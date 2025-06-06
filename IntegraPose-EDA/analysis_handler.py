import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
import app_config 

class AnalysisHandler:
    def __init__(self):
        self.last_pca_model = None
        self.last_pca_components = None
        self.last_pca_explained_variance_ratio = None
        self.last_linkage_matrix = None
        self.last_kmeans_model = None
        self.status_message = "AnalysisHandler initialized."
        self.last_error = None

    def _prepare_data_for_analysis(self, df_input_features, feature_columns_for_analysis):
        """
        Prepares data for PCA or clustering: selects columns, handles NaNs, and scales.

        Args:
            df_input_features (pd.DataFrame): The DataFrame containing features.
            feature_columns_for_analysis (list): List of feature column names to use.

        Returns:
            tuple: (scaled_data_np, original_indices_after_dropna, error_message_or_none)
                   Returns (None, None, error_message) if preparation fails.
        """
        if df_input_features is None or df_input_features.empty:
            return None, None, "Input DataFrame is empty."
        if not feature_columns_for_analysis:
            return None, None, "No feature columns selected for analysis."

        missing_cols = [col for col in feature_columns_for_analysis if col not in df_input_features.columns]
        if missing_cols:
            return None, None, f"Missing feature columns in DataFrame: {', '.join(missing_cols)}"

        data_for_analysis = df_input_features[feature_columns_for_analysis].copy()
        
        original_indices = data_for_analysis.index
        data_for_analysis_no_na = data_for_analysis.dropna(how='any')
        original_indices_after_dropna = data_for_analysis_no_na.index

        if data_for_analysis_no_na.empty:
            return None, None, "No data remaining after removing rows with NaN values in selected features."
        if len(data_for_analysis_no_na) < 2: # Need at least 2 samples for most analyses
            return None, None, "Too few samples remaining after NaN removal (need at least 2)."

        try:
            scaled_data_np = StandardScaler().fit_transform(data_for_analysis_no_na)
        except Exception as e:
            return None, None, f"Error during data scaling: {e}"
            
        return scaled_data_np, original_indices_after_dropna, None

    def perform_pca(self, df_features, feature_columns_for_pca, n_components_requested):
        """
        Performs PCA on the selected features.

        Args:
            df_features (pd.DataFrame): DataFrame containing the features.
            feature_columns_for_pca (list): List of feature column names to use for PCA.
            n_components_requested (int): Desired number of principal components.

        Returns:
            tuple: (pca_transformed_data_df, pca_component_names, explained_variance_ratio, status_message)
                   Returns (None, None, None, error_message) if PCA fails.
        """
        self.last_pca_model = None
        self.last_pca_components = None
        self.last_pca_explained_variance_ratio = None

        scaled_data, original_indices, error_msg = self._prepare_data_for_analysis(df_features, feature_columns_for_pca)
        if error_msg:
            self.last_error = f"PCA Data Prep Error: {error_msg}"
            return None, None, None, self.last_error

        if scaled_data.shape[0] < n_components_requested or scaled_data.shape[1] < n_components_requested :
             # Adjust n_components if it's too large for the data
            max_n_comp = min(scaled_data.shape)
            if n_components_requested > max_n_comp and max_n_comp > 0:
                adjusted_message = f"PCA components adjusted from {n_components_requested} to {max_n_comp} (max available)."
                print(adjusted_message) # Log this
                n_components_requested = max_n_comp
            elif max_n_comp == 0 : # Should be caught by _prepare_data_for_analysis
                 self.last_error = "Cannot perform PCA with zero valid samples or features after preparation."
                 return None, None, None, self.last_error


        if n_components_requested <= 0:
            self.last_error = "Number of PCA components must be positive."
            return None, None, None, self.last_error
        
        try:
            pca = PCA(n_components=n_components_requested)
            pca_transformed_data_np = pca.fit_transform(scaled_data)
            
            self.last_pca_model = pca
            self.last_pca_components = pca.components_
            self.last_pca_explained_variance_ratio = pca.explained_variance_ratio_
            
            pca_component_names = [f'PC{i+1}' for i in range(pca_transformed_data_np.shape[1])]
            pca_transformed_data_df = pd.DataFrame(
                pca_transformed_data_np,
                index=original_indices, # Use original indices of rows that were not dropped
                columns=pca_component_names
            )
            
            evr_sum = sum(self.last_pca_explained_variance_ratio)
            status_msg = (f"PCA successful. {len(pca_component_names)} components "
                          f"explain {evr_sum:.3f} of variance.")
            if 'adjusted_message' in locals(): status_msg = adjusted_message + "\n" + status_msg
            return pca_transformed_data_df, pca_component_names, self.last_pca_explained_variance_ratio, status_msg

        except Exception as e:
            self.last_error = f"PCA failed: {e}"
            return None, None, None, self.last_error

    def perform_ahc(self, df_analysis_data, feature_columns_for_ahc, linkage_method='ward', num_flat_clusters=0):
        """
        Performs Agglomerative Hierarchical Clustering.

        Args:
            df_analysis_data (pd.DataFrame): DataFrame to cluster (can be original features or PCA results).
                                            If PCA results, feature_columns_for_ahc should be PC names.
            feature_columns_for_ahc (list): Columns to use for clustering.
            linkage_method (str): Linkage method for AHC (e.g., 'ward', 'complete').
            num_flat_clusters (int): Number of flat clusters to extract (0 for none).

        Returns:
            tuple: (linkage_matrix, flat_cluster_labels_series, status_message)
                   flat_cluster_labels_series will be None if num_flat_clusters is 0 or error.
                   Returns (None, None, error_message) if AHC fails.
        """
        self.last_linkage_matrix = None
        
        scaled_data, original_indices, error_msg = self._prepare_data_for_analysis(df_analysis_data, feature_columns_for_ahc)
        if error_msg:
            self.last_error = f"AHC Data Prep Error: {error_msg}"
            return None, None, self.last_error
        
        if scaled_data.shape[0] < 2: # linkage requires at least 2 samples
            self.last_error = "AHC requires at least 2 samples after data preparation."
            return None, None, self.last_error

        try:
            self.last_linkage_matrix = linkage(scaled_data, method=linkage_method, metric='euclidean')
            status_msg = f"AHC ({linkage_method}) linkage matrix calculated."
            flat_labels_series = None

            if num_flat_clusters > 0:
                if num_flat_clusters > len(original_indices):
                    status_msg += f"\nWarning: Requested {num_flat_clusters} flat clusters, but only {len(original_indices)} samples. Adjusting."
                    num_flat_clusters = len(original_indices)
                
                if num_flat_clusters > 0: # Re-check after potential adjustment
                    try:
                        flat_labels_array = fcluster(self.last_linkage_matrix, num_flat_clusters, criterion='maxclust')
                        flat_labels_series = pd.Series(flat_labels_array, index=original_indices, name='unsupervised_cluster')
                        status_msg += f"\nFlat clusters (k={num_flat_clusters}) extracted."
                    except Exception as e_fcl:
                        status_msg += f"\nError during AHC flat clustering: {e_fcl}"
                        print(f"AHC flat clustering error: {e_fcl}")
            
            return self.last_linkage_matrix, flat_labels_series, status_msg

        except Exception as e:
            self.last_error = f"AHC failed: {e}"
            return None, None, self.last_error

    def perform_kmeans(self, df_analysis_data, feature_columns_for_kmeans, n_clusters_k):
        """
        Performs K-Means clustering.

        Args:
            df_analysis_data (pd.DataFrame): DataFrame to cluster.
            feature_columns_for_kmeans (list): Columns to use for clustering.
            n_clusters_k (int): Number of clusters (K).

        Returns:
            tuple: (cluster_labels_series, inertia, status_message)
                   Returns (None, None, error_message) if KMeans fails.
        """
        self.last_kmeans_model = None

        scaled_data, original_indices, error_msg = self._prepare_data_for_analysis(df_analysis_data, feature_columns_for_kmeans)
        if error_msg:
            self.last_error = f"KMeans Data Prep Error: {error_msg}"
            return None, None, self.last_error

        if n_clusters_k <= 0:
            self.last_error = "Number of clusters (K) for KMeans must be positive."
            return None, None, self.last_error
        if n_clusters_k > scaled_data.shape[0]:
            self.last_error = f"K for KMeans ({n_clusters_k}) cannot exceed number of samples ({scaled_data.shape[0]})."
            # Optionally, could adjust K automatically and add to status_msg
            # n_clusters_k = scaled_data.shape[0]
            # status_msg_prefix = f"Warning: K adjusted to {n_clusters_k}.\n"
            return None, None, self.last_error

        try:
            kmeans = KMeans(n_clusters=n_clusters_k, random_state=42, n_init='auto')
            cluster_labels_array = kmeans.fit_predict(scaled_data)
            
            self.last_kmeans_model = kmeans
            cluster_labels_series = pd.Series(cluster_labels_array, index=original_indices, name='unsupervised_cluster')
            inertia = kmeans.inertia_
            status_msg = f"KMeans (K={n_clusters_k}) successful. Inertia: {inertia:.2f}."
            # if 'status_msg_prefix' in locals(): status_msg = status_msg_prefix + status_msg
            
            return cluster_labels_series, inertia, status_msg

        except Exception as e:
            self.last_error = f"KMeans failed: {e}"
            return None, None, self.last_error
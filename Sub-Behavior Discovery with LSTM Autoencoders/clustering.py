# clustering.py
import hdbscan
import logging
import numpy as np

logger = logging.getLogger(__name__)

def cluster_sequence_embeddings(embeddings, min_cluster_size=10):
    """Applies HDBSCAN to cluster the final sequence embeddings."""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)

    logger.info(f"Running HDBSCAN on embeddings with shape: {embeddings.shape}")

    # HDBSCAN requires n_samples > min_cluster_size
    if embeddings.shape[0] <= min_cluster_size:
        logger.warning(
            f"Number of embeddings ({embeddings.shape[0]}) is less than or equal to "
            f"min_cluster_size ({min_cluster_size}). Cannot perform clustering. "
            f"All points will be marked as noise (-1)."
        )
        return np.full(embeddings.shape[0], -1)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom',  # Energy-based optimal merging
        gen_min_span_tree=True # Useful for visualization/analysis
    )
    
    try:
        labels = clusterer.fit_predict(embeddings)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = np.sum(labels == -1)
        logger.info(
            f"HDBSCAN found {num_clusters} clusters with {noise_points} noise points "
            f"from {embeddings.shape[0]} sequences."
        )
        return labels
    except Exception as e:
        logger.error(f"HDBSCAN clustering failed: {e}")
        return np.full(embeddings.shape[0], -1)
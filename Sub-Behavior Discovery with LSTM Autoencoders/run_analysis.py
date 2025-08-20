# run_analysis.py
import json
import logging
import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from collections import defaultdict
# Import project modules
from data_loader import read_detections, compute_feature_vectors, create_sequences
from training import train_vae, encode_sequences_with_vae, train_lstm_autoencoder, extract_sequence_embeddings
from clustering import cluster_sequence_embeddings
from video_generator import create_cluster_videos

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def visualize_embeddings(embeddings, labels, output_dir, title):
    """Visualizes high-dimensional embeddings using UMAP."""
    logger.info(f"Generating UMAP visualization for '{title}'...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    scatter = sns.scatterplot(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        hue=labels,
        palette=sns.color_palette("hsv", len(set(labels))),
        s=50,
        alpha=0.7
    )
    plt.title(f'UMAP Projection of Sequence Embeddings ({title})')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Class ID' if 'Class' in title else 'Cluster ID')
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, f'umap_visualization_{title.replace(" ", "_").lower()}.png')
    plt.savefig(plot_path)
    logger.info(f"Saved UMAP plot to {plot_path}")
    plt.close()

def main():
    """Main pipeline for the standalone VAE-LSTM Autoencoder analysis."""
    # 1. Load Configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    data_cfg = config['data']
    prep_cfg = config['preprocessing']
    feature_cfg = config['feature_params']
    vis_cfg = config['visualization']
    vae_cfg = config['vae_params']
    lstm_cfg = config['lstm_autoencoder_params']
    cluster_cfg = config['clustering_params']
    
    # Create output directory if it doesn't exist
    os.makedirs(data_cfg['output_directory'], exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # 2. Load and Preprocess Data
    detections_df, keypoint_names = read_detections(
        data_cfg['pose_directory'], data_cfg['video_file'], data_cfg['keypoint_names']
    )
    detections_df['original_index'] = detections_df.index
    
    detections_df = compute_feature_vectors(
        detections_df, keypoint_names, feature_cfg['angles_to_compute'], prep_cfg['confidence_threshold']
    )

    # 3. Normalize Feature Vectors
    all_feature_vectors = np.vstack(detections_df['feature_vector'].dropna().values)
    if all_feature_vectors.size == 0:
        raise ValueError("No feature vectors could be computed. Check data and parameters.")
    
    mean = np.mean(all_feature_vectors, axis=0)
    std = np.std(all_feature_vectors, axis=0)
    std[std == 0] = 1e-8 # Avoid division by zero
    
    detections_df['feature_vector'] = detections_df['feature_vector'].apply(
        lambda vec: (np.array(vec) - mean) / std if vec is not None else None
    )
    detections_df.dropna(subset=['feature_vector'], inplace=True)
    
    sequences_of_detections = create_sequences(
        detections_df, prep_cfg['max_frame_gap'], prep_cfg['sequence_split_strategy']
    )

    # 4. Train VAE and Encode Sequences
    all_features_list = [item for sublist in sequences_of_detections for item in sublist]
    all_feature_vectors_for_vae = [det['feature_vector'] for det in all_features_list]
    
    vae = train_vae(all_feature_vectors_for_vae, vae_cfg, device)
    vae_encoded_sequences = encode_sequences_with_vae(vae, sequences_of_detections, device)
    
    # Filter out empty sequences that might result from encoding
    valid_indices = [i for i, seq in enumerate(vae_encoded_sequences) if len(seq) > 1]
    valid_sequences_of_detections = [sequences_of_detections[i] for i in valid_indices]
    valid_vae_encoded_sequences = [vae_encoded_sequences[i] for i in valid_indices]

    if not valid_vae_encoded_sequences:
        raise ValueError("VAE encoding resulted in no valid sequences for LSTM training.")

    # 5. Train LSTM Autoencoder and Extract Embeddings
    lstm_encoder = train_lstm_autoencoder(
        valid_vae_encoded_sequences, vae_cfg['latent_dim'], lstm_cfg, device
    )
    final_embeddings = extract_sequence_embeddings(lstm_encoder, valid_vae_encoded_sequences, device)
    
    if final_embeddings.size == 0:
        raise ValueError("Extraction of final embeddings failed.")
    
    # (Optional) Save the trained LSTM encoder model
    if data_cfg.get('save_model', False):
        model_filename = data_cfg.get('model_save_filename', 'lstm_encoder_state.pt')
        save_path = os.path.join(data_cfg['output_directory'], model_filename)
        torch.save(lstm_encoder.state_dict(), save_path)
        logger.info(f"LSTM encoder model state saved to {save_path}")

    # 6. (Optional) Visualize Embeddings Before Clustering
    if vis_cfg.get('generate_embedding_plot', False):
        sequence_class_ids = [seq[0]['class_id'] for seq in valid_sequences_of_detections]
        visualize_embeddings(final_embeddings, sequence_class_ids, data_cfg['output_directory'], "Behavior Class IDs")

    # 7. Cluster Embeddings per Behavior Class
    class_groups = defaultdict(list)
    for i, seq in enumerate(valid_sequences_of_detections):
        class_id = seq[0]['class_id']
        class_groups[class_id].append(i)

    sequence_labels = np.full(len(valid_sequences_of_detections), -1, dtype=int)
    for class_id, indices in class_groups.items():
        if len(indices) < cluster_cfg['min_cluster_size']:
            logger.warning(f"Skipping clustering for behavior {class_id}: only {len(indices)} sequences (min is {cluster_cfg['min_cluster_size']}).")
            continue
        
        sub_embeddings = final_embeddings[indices]
        sub_labels = cluster_sequence_embeddings(sub_embeddings, cluster_cfg['min_cluster_size'])
        
        # Map sub-cluster labels back to the main sequence list
        # We need to make sure we don't overwrite labels from different classes
        # HDBSCAN labels are 0-indexed, so we can use them directly.
        for i, label in enumerate(sub_labels):
            original_index = indices[i]
            sequence_labels[original_index] = label

    # 8. Map Cluster Labels Back to Original Detections
    detections_df['cluster_label'] = -1
    for i, seq_of_dets in enumerate(valid_sequences_of_detections):
        cluster_label = sequence_labels[i]
        indices_to_update = [det['original_index'] for det in seq_of_dets]
        detections_df.loc[indices_to_update, 'cluster_label'] = cluster_label

    # 9. Generate Output Videos
    # This function needs the original 'keypoints' and 'bbox' columns.
    create_cluster_videos(data_cfg['output_directory'], data_cfg['video_file'], detections_df)

    # 10. Prepare DataFrame for CSV and Save
    output_csv = os.path.join(data_cfg['output_directory'], 'behavior_clusters_full.csv')
    logger.info(f"Preparing final CSV report at {output_csv}...")
    
    # Now, flatten the columns for a clean CSV output
    for kp in keypoint_names:
        detections_df[f'{kp}_x'] = detections_df['keypoints'].apply(lambda d: d.get(kp, (np.nan, np.nan, np.nan))[0])
        detections_df[f'{kp}_y'] = detections_df['keypoints'].apply(lambda d: d.get(kp, (np.nan, np.nan, np.nan))[1])
    
    detections_df[['bbox_center_x', 'bbox_center_y', 'bbox_width', 'bbox_height']] = pd.DataFrame(detections_df['bbox'].tolist(), index=detections_df.index)
    
    if 'feature_vector' in detections_df.columns and not detections_df.empty:
        feature_df = pd.DataFrame(detections_df['feature_vector'].tolist(), index=detections_df.index)
        feature_df.columns = [f'feature_{i}' for i in range(feature_df.shape[1])]
        detections_df = pd.concat([detections_df, feature_df], axis=1)

    # And now, drop the original complex columns before saving
    detections_df.drop(columns=['keypoints', 'bbox', 'feature_vector', 'original_index'], inplace=True, errors='ignore')
    detections_df.to_csv(output_csv, index=False)
    logger.info(f"Full cluster results saved to {output_csv}")

    logger.info("Analysis complete!")

if __name__ == '__main__':
    main()
# hmm_utils_revised.py
# This is a rewritten version to fix logical errors and improve functionality.

import numpy as np
import pandas as pd
from hmmlearn import hmm
import logging
from collections import Counter
from scipy.special import rel_entr
from scipy.stats import chisquare

logger = logging.getLogger(__name__)

def train_hmm(sequences, n_states=5, model_type='categorical', use_poses=True, progress_callback=None):
    """
    Trains an HMM on sequences of observations.
    
    NOTE: The 'hmmlearn' library's fit method is a blocking call. 
    A per-iteration progress update is not available. The callback will be used 
    before and after the main training step.
    """
    logger.info(f"Training HMM with n_states={n_states}, type={model_type}, use_poses={use_poses}")
    if not sequences:
        logger.warning("No sequences provided for HMM training")
        return None, None

    all_observations = []
    lengths = []
    for seq in sequences:
        if not seq: continue
        # For GaussianHMM, observations are the feature vectors themselves.
        if model_type == 'gaussian':
            obs = [det['feature_vector'] for det in seq if 'feature_vector' in det]
        # For CategoricalHMM, observations are discrete category labels.
        else:
            obs = [det['pose_category'] for det in seq if 'pose_category' in det]
        
        if obs:
            all_observations.extend(obs)
            lengths.append(len(obs))

    if not all_observations:
        logger.warning("No valid observations found for HMM training.")
        return None, None

    # For CategoricalHMM, observations need to be mapped to integers.
    if model_type == 'categorical':
        unique_obs = sorted(list(set(all_observations)))
        id_map = {obs: i for i, obs in enumerate(unique_obs)}
        obs_array = np.array([id_map[obs] for obs in all_observations]).reshape(-1, 1)
        model = hmm.CategoricalHMM(n_components=n_states, n_iter=100, tol=1e-4, random_state=42)
    
    # For GaussianHMM, observations are continuous vectors.
    elif model_type == 'gaussian':
        obs_array = np.vstack(all_observations)
        id_map = {}  # No id_map needed for Gaussian HMM
        model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=100, tol=1e-4, random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if progress_callback:
        progress_callback(0, 1, f"Fitting HMM with {len(obs_array)} data points...")

    model.fit(obs_array, lengths)
    
    if progress_callback:
        progress_callback(1, 1, "HMM fitting complete.")

    logger.info("HMM training complete")
    return model, id_map


def infer_state_sequences(model, sequences, id_map, use_poses=True):
    """Infers the most likely state sequences using the trained HMM."""
    inferred_sequences = []
    is_gaussian = isinstance(model, hmm.GaussianHMM)

    for seq in sequences:
        if not seq: continue
        
        if is_gaussian:
            obs = [det['feature_vector'] for det in seq if 'feature_vector' in det]
            if not obs: continue
            obs_array = np.vstack(obs)
        else: # Categorical
            obs = [id_map.get(det['pose_category'], -1) for det in seq if 'pose_category' in det]
            if not obs or any(o == -1 for o in obs): continue
            obs_array = np.array(obs).reshape(-1, 1)

        try:
            states = model.predict(obs_array)
            inferred_sequences.append(states)
        except Exception as e:
            logger.warning(f"Could not predict states for a sequence: {e}")
            
    return inferred_sequences


def compute_transition_matrix(model):
    """Computes the transition matrix from the HMM model."""
    return model.transmat_


def compute_emission_matrix(model, id_map, behavior_names=None, use_poses=True):
    """Computes the emission matrix as a DataFrame. Returns None for GaussianHMM."""
    if isinstance(model, hmm.GaussianHMM):
        logger.info("Emission matrix is not applicable for GaussianHMM; represents continuous distributions.")
        return None
        
    if use_poses:
        # Ensure columns match the id_map used for training
        num_emissions = len(id_map)
        labels = [f"Pose Cluster {i}" for i in range(num_emissions)]
    else:
        labels = behavior_names or [f"Behavior {i}" for i in range(len(id_map))]
        
    emission_df = pd.DataFrame(model.emissionprob_, columns=labels)
    emission_df.index = [f"State {i}" for i in range(model.n_components)]
    return emission_df


def assign_state_labels(emission_df):
    """Assigns descriptive labels to HMM states based on emission probabilities."""
    if emission_df is None:
        return {f"State {i}": {'label': f"Latent State {i}", 'prob': 1.0} for i in range(len(emission_df.index))}

    state_labels = {}
    for state in emission_df.index:
        max_prob_idx = emission_df.loc[state].argmax()
        label = emission_df.columns[max_prob_idx]
        state_labels[state] = {'label': label, 'prob': emission_df.loc[state, label]}
    return state_labels


def compare_groups(group_hmms, method='log_likelihood', sequences_by_group=None):
    """
    Compares HMM models between groups using specified method.
    
    Methods:
    - 'log_likelihood': (Recommended) Computes the log-likelihood of one group's sequences 
                        under another group's model. Lower score is better fit.
    - 'kl_divergence': Compares the transition matrices using KL Divergence.
    """
    groups = list(group_hmms.keys())
    if len(groups) < 2:
        return {}

    comparison_results = pd.DataFrame(index=groups, columns=groups, dtype=float)

    for g1_name in groups:
        for g2_name in groups:
            model_g2 = group_hmms[g2_name]['model']
            
            if method == 'log_likelihood':
                if not sequences_by_group:
                    raise ValueError("Sequence data is required for log_likelihood comparison.")
                
                # Prepare sequences from group 1 for scoring
                seqs_g1 = sequences_by_group[g1_name]
                is_gaussian = isinstance(model_g2, hmm.GaussianHMM)
                
                observations = []
                lengths = []
                for seq in seqs_g1:
                    if is_gaussian:
                        obs_list = [det['feature_vector'] for det in seq if 'feature_vector' in det]
                    else: # Categorical
                        id_map_g2 = group_hmms[g2_name]['id_map']
                        obs_list = [id_map_g2.get(det['pose_category']) for det in seq if 'pose_category' in det]
                        # Handle cases where a pose in G1 doesn't exist in G2's map
                        if any(o is None for o in obs_list): continue

                    if obs_list:
                        observations.extend(obs_list)
                        lengths.append(len(obs_list))
                
                if not observations:
                    score = np.nan
                else:
                    score = model_g2.score(np.vstack(observations), lengths)
                comparison_results.loc[g1_name, g2_name] = score

            elif method == 'kl_divergence':
                # Note: This only compares transition probabilities, not the full model.
                trans_g1 = group_hmms[g1_name]['transmat']
                trans_g2 = group_hmms[g2_name]['transmat']
                
                if trans_g1.shape != trans_g2.shape:
                    comparison_results.loc[g1_name, g2_name] = np.nan
                    continue

                kl_div = np.sum([rel_entr(trans_g1[k], trans_g2[k]).sum() for k in range(trans_g1.shape[0])])
                comparison_results.loc[g1_name, g2_name] = kl_div
                
    return comparison_results
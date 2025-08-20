# training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import logging
from models import VAE, EncoderLSTM, DecoderLSTM, Seq2Seq

logger = logging.getLogger(__name__)

def vae_loss_function(recon_x, x, mu, logvar):
    """VAE loss = reconstruction loss + KL divergence."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def train_vae(feature_vectors, params, device):
    """Trains the Variational Autoencoder."""
    logger.info("Training VAE...")
    input_dim = len(feature_vectors[0])
    model = VAE(
        input_dim=input_dim,
        intermediate_dim=params['intermediate_dim'],
        latent_dim=params['latent_dim'],
        dropout_rate=params['dropout_rate']
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    dataset = TensorDataset(torch.tensor(np.array(feature_vectors), dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model.train()
    for epoch in range(params['epochs']):
        total_loss = 0
        for batch, in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss_function(recon_batch, batch, mu, logvar)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"VAE loss is NaN/Inf at epoch {epoch+1}. Skipping batch.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader.dataset)
        logger.info(f"VAE Epoch {epoch+1}/{params['epochs']}, Avg Loss: {avg_loss:.4f}")
    
    return model.eval()

def encode_sequences_with_vae(vae, sequences, device):
    """Encodes sequences of feature vectors using the trained VAE."""
    logger.info("Encoding sequences with trained VAE...")
    vae.eval()
    encoded_sequences = []
    with torch.no_grad():
        for seq in sequences:
            # Ensure the sequence is not empty
            if not seq:
                continue
            feature_vectors = [det['feature_vector'] for det in seq]
            tensor = torch.tensor(np.array(feature_vectors), dtype=torch.float32).to(device)
            if tensor.ndim == 1: # Handle single-item sequences if they slip through
                tensor = tensor.unsqueeze(0)
            mu, _ = vae.encode(tensor)
            encoded_sequences.append(mu.cpu().numpy())
    return encoded_sequences

def train_lstm_autoencoder(sequences, vae_latent_dim, params, device):
    """Trains the LSTM Autoencoder (Seq2Seq model)."""
    logger.info("Training LSTM Autoencoder...")
    # Filter out very short sequences which are not useful for learning temporal patterns
    sequences_torch = [torch.tensor(s, dtype=torch.float32) for s in sequences if len(s) > 1]
    if not sequences_torch:
        raise ValueError("No sequences of sufficient length for LSTM training.")

    lengths = torch.tensor([len(s) for s in sequences_torch])
    padded_sequences = pad_sequence(sequences_torch, batch_first=True, padding_value=0.0)
    dataset = TensorDataset(padded_sequences, lengths)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    encoder = EncoderLSTM(
        input_dim=vae_latent_dim,
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        dropout_rate=params['dropout_rate']
    ).to(device)
    
    decoder = DecoderLSTM(
        output_dim=vae_latent_dim,
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        dropout_rate=params['dropout_rate']
    ).to(device)

    model = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='none')

    model.train()
    for epoch in range(params['epochs']):
        total_loss = 0
        for batch_seqs, batch_lengths in dataloader:
            # Move both sequences AND their lengths to the correct device
            batch_seqs = batch_seqs.to(device)
            batch_lengths = batch_lengths.to(device) # <--- FIX IS HERE
            
            optimizer.zero_grad()
            # The model itself correctly handles passing CPU lengths to pack_padded_sequence
            reconstructed = model(batch_seqs, batch_lengths) 
            
            # Create a mask to ignore padded parts of the sequences in loss calculation
            # Now both tensors in the comparison are on the same device
            mask = (torch.arange(batch_seqs.size(1))[None, :].to(device) < batch_lengths[:, None]).float()
            
            loss = (criterion(reconstructed, batch_seqs) * mask.unsqueeze(-1)).sum() / mask.sum()

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"LSTM loss is NaN/Inf at epoch {epoch+1}. Skipping batch.")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        logger.info(f"LSTM AE Epoch {epoch+1}/{params['epochs']}, Avg Loss: {avg_loss:.6f}")
        
    return model.encoder.eval()

def extract_sequence_embeddings(encoder, sequences, device):
    """Extracts a single embedding vector for each sequence using the LSTM encoder."""
    logger.info("Extracting final sequence embeddings...")
    encoder.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            if len(seq) == 0:
                continue
            tensor_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            length = torch.tensor([len(seq)])
            # The hidden state from the encoder represents the sequence embedding
            hidden, _ = encoder(tensor_seq, length)
            # Squeeze to remove batch and layer dimensions
            embeddings.append(hidden.squeeze(0).squeeze(0).cpu().numpy())
            
    return np.vstack(embeddings) if embeddings else np.array([])
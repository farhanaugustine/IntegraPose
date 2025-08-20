# vae_lstm_utils.py
# This is an updated version with GPU support, progress callbacks, and a corrected LSTM function.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VAE(nn.Module):
    """Variational Autoencoder model for pose feature encoding."""
    def __init__(self, input_dim, intermediate_dim, latent_dim, dropout_rate=0.2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc_mu = nn.Linear(intermediate_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(intermediate_dim // 2, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim // 2, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """VAE loss function (reconstruction + KL divergence)."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def train_vae(feature_vectors, latent_dim=8, intermediate_dim=128, epochs=100, dropout_rate=0.2, batch_size=64, device='cpu', progress_callback=None):
    """Trains the VAE model on feature vectors."""
    logger.info(f"Training VAE with latent_dim={latent_dim}, epochs={epochs}, device={device}")
    input_dim = len(feature_vectors[0])
    model = VAE(input_dim, intermediate_dim, latent_dim, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = TensorDataset(torch.tensor(feature_vectors, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (batch,) in enumerate(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader.dataset)
        logger.info(f"VAE Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if progress_callback:
            progress_callback(epoch + 1, epochs, f"VAE training: Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, avg_loss

def encode_sequences(vae, sequences, device='cpu'):
    """Encodes sequences of detections to latent representations using VAE."""
    vae.eval()
    encoded_sequences = []
    with torch.no_grad():
        for seq in sequences:
            feature_vectors = [det['feature_vector'] for det in seq]
            tensor = torch.tensor(feature_vectors, dtype=torch.float32).to(device)
            mu, _ = vae.encode(tensor)
            encoded_sequences.append(mu.cpu().numpy())
    return encoded_sequences

class LSTM(nn.Module):
    """LSTM model for sequence classification or prediction."""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout_rate=0.2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, input_dim)  # Bidirectional, so *2

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Predict based on the last time step's output

def train_lstm(encoded_sequences, latent_dim, num_layers=2, epochs=50, dropout_rate=0.2, device='cpu', progress_callback=None):
    """
    Trains the LSTM model on encoded sequences.
    This function now returns the trained model.
    """
    logger.info(f"Training LSTM with num_layers={num_layers}, epochs={epochs}, device={device}")
    input_dim = encoded_sequences[0].shape[1] if encoded_sequences else latent_dim
    model = LSTM(input_dim, latent_dim, num_layers, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Prepare data by padding sequences to the same length
    max_len = max(len(seq) for seq in encoded_sequences)
    # Create input (X) and target (y) tensors. The target is the next frame in the sequence.
    # We will predict the last element of each sequence from the preceding elements.
    X_list = [seq[:-1] for seq in encoded_sequences if len(seq) > 1]
    y_list = [seq[1:] for seq in encoded_sequences if len(seq) > 1]

    if not X_list:
        logger.warning("No sequences long enough for LSTM training (length > 1). Skipping.")
        return None

    padded_X = [np.pad(seq, ((0, max_len - 1 - len(seq)), (0, 0)), mode='constant') for seq in X_list]
    padded_y = [np.pad(seq, ((0, max_len - 1 - len(seq)), (0, 0)), mode='constant') for seq in y_list]
    
    X_tensor = torch.tensor(np.array(padded_X), dtype=torch.float32).to(device)
    y_tensor = torch.tensor(np.array(padded_y), dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            # We need to reshape the output of the LSTM to match the target shape
            lstm_out, _ = model.lstm(X_batch)
            outputs = model.fc(lstm_out)
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"LSTM Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if progress_callback:
            progress_callback(epoch + 1, epochs, f"LSTM training: Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    logger.info("LSTM training complete.")
    model.eval()
    return model
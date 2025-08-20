# models.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class VAE(nn.Module):
    """Variational Autoencoder for single-frame pose feature encoding."""
    def __init__(self, input_dim, intermediate_dim, latent_dim, dropout_rate=0.2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim, intermediate_dim // 2), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        self.fc_mu = nn.Linear(intermediate_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(intermediate_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim // 2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim // 2, intermediate_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

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

class EncoderLSTM(nn.Module):
    """
    Encoder network (bidirectional LSTM).
    The forward method now correctly handles the hidden and cell states from the
    bidirectional LSTM for a more robust sequence representation.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout_rate=0.2):
        super(EncoderLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        lstm_dropout = dropout_rate if num_layers > 1 else 0
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True, dropout=lstm_dropout
        )
        # Project the concatenated hidden states to the desired hidden_dim
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.lstm(packed_input)
        
        # Concatenate the final forward and backward hidden states of the last layer
        # hidden is [num_layers*2, batch, hidden_dim]
        # We take the last layer's forward (2*num_layers-2) and backward (2*num_layers-1) states
        last_layer_hidden = torch.cat(
            (hidden[-2, :, :], hidden[-1, :, :]), dim=1
        )
        last_layer_cell = torch.cat(
            (cell[-2, :, :], cell[-1, :, :]), dim=1
        )
        
        # Reduce dimension
        hidden_reduced = torch.relu(self.fc_hidden(last_layer_hidden))
        cell_reduced = torch.relu(self.fc_cell(last_layer_cell))
        
        # The decoder is not bidirectional, so we need to unsqueeze to add the layer dimension
        return hidden_reduced.unsqueeze(0), cell_reduced.unsqueeze(0)


class DecoderLSTM(nn.Module):
    """Decoder network: Reconstructs the original sequence."""
    def __init__(self, output_dim, hidden_dim, num_layers=2, dropout_rate=0.2):
        super(DecoderLSTM, self).__init__()
        decoder_dropout = dropout_rate if num_layers > 1 else 0
        # The input to the decoder LSTM is the hidden state from the encoder
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True, dropout=decoder_dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        output, _ = self.lstm(x, (hidden, cell))
        # No need to pad here, as we will unpack in the main Seq2Seq model
        return self.fc(output)


class Seq2Seq(nn.Module):
    """Combines the Encoder and Decoder for training the LSTM autoencoder."""
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source_seq, source_lengths):
        total_length = source_seq.size(1)
        hidden, cell = self.encoder(source_seq, source_lengths)
        
        # The decoder's hidden state should have num_layers dimension
        # We need to repeat the context vector for each layer of the decoder
        hidden = hidden.repeat(self.decoder.lstm.num_layers, 1, 1)
        cell = cell.repeat(self.decoder.lstm.num_layers, 1, 1)

        # Pack the source sequence to be used as input for the decoder
        packed_input = pack_padded_sequence(source_seq, source_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        decoder_output, _ = self.decoder.lstm(packed_input, (hidden, cell))
        
        # Unpack the sequence
        unpacked_output, _ = pad_packed_sequence(decoder_output, batch_first=True, total_length=total_length)
        
        # Apply the final linear layer
        reconstruction = self.decoder.fc(unpacked_output)
        
        return reconstruction
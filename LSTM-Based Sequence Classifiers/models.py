# models.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout_rate=0.2):
        super(EncoderLSTM, self).__init__()
        lstm_dropout = dropout_rate if num_layers > 1 else 0
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True, dropout=lstm_dropout
        )
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.lstm(packed_input)
        
        last_layer_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden_reduced = torch.relu(self.fc_hidden(last_layer_hidden))
        
        return hidden_reduced.unsqueeze(0) # We only need the final hidden state for classification

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout_rate=0.3):
        super(LSTMClassifier, self).__init__()
        # Use the encoder to process temporal features
        self.encoder = EncoderLSTM(input_dim, hidden_dim, num_layers, dropout_rate)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, lengths):
        # The encoder's final hidden state is the sequence embedding
        embedding = self.encoder(x, lengths).squeeze(0)
        
        # Get class predictions from the embedding
        output = self.classifier(embedding)
        return output
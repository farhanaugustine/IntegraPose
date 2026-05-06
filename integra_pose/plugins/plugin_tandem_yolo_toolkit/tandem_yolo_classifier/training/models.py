import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    """
    A Bidirectional LSTM model for sequence classification.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float
    ):
        """
        Initializes the LSTMClassifier model.
        """
        super(LSTMClassifier, self).__init__()
        
        lstm_dropout = dropout_rate if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        """
        _ , (hidden, _) = self.lstm(x)
        last_layer_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = self.classifier(last_layer_hidden)
        return output

class LSTMClassifierWithAttention(nn.Module):
    """
    A Bidirectional LSTM model with a self-attention mechanism.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float
    ):
        """
        Initializes the LSTMClassifierWithAttention model.
        """
        super(LSTMClassifierWithAttention, self).__init__()
        
        lstm_dropout = dropout_rate if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout
        )
        
        self.attention = nn.Linear(hidden_dim * 2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model with attention.
        """
        lstm_out, _ = self.lstm(x)
        
        attention_weights = self.attention(lstm_out)
        
        attention_weights = F.softmax(attention_weights, dim=1)
        
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        output = self.classifier(context_vector)
        return output

def get_model(model_name: str, model_params: dict, input_dim: int, num_classes: int) -> nn.Module:
    """
    Factory function to get the specified model instance.
    """
    model_name = model_name.lower()
    if model_name == 'attention':
        print("Instantiating LSTMClassifierWithAttention model.")
        return LSTMClassifierWithAttention(
            input_dim=input_dim,
            hidden_dim=model_params['hidden_dim'],
            num_layers=model_params['num_layers'],
            num_classes=num_classes,
            dropout_rate=model_params['dropout_rate']
        )
    elif model_name == 'lstm':
        print("Instantiating standard LSTMClassifier model.")
        return LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=model_params['hidden_dim'],
            num_layers=model_params['num_layers'],
            num_classes=num_classes,
            dropout_rate=model_params['dropout_rate']
        )
    else:
        raise ValueError(f"Unknown model type '{model_name}'. Choose 'lstm' or 'attention'.")

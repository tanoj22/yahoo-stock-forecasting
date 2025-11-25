# src/models_lstm.py

import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        out, _ = self.lstm(x)
        # take last time step
        last = out[:, -1, :]
        out = self.fc(last)
        return out

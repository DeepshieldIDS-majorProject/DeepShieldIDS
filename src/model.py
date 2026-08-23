"""
model.py

HybridIDSNet: a CNN + LSTM + Attention hybrid architecture for
flow-based network intrusion detection.

Pipeline:
  Input (batch, seq_len, 1)
     -> Conv1D blocks (spatial feature extraction)
     -> BiLSTM (temporal sequence modelling)
     -> Additive attention (feature-weighting over LSTM timesteps)
     -> Dense classifier head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool=True, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2) if pool else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        if self.pool is not None:
            x = self.pool(x)
        x = self.dropout(x)
        return x


class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention over LSTM hidden states.
    Learns which timesteps (feature positions) matter most for the
    final classification decision.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_dim)
        energy = torch.tanh(self.attn_fc(lstm_out))          # (batch, seq_len, hidden_dim)
        scores = self.context_vector(energy).squeeze(-1)     # (batch, seq_len)
        weights = F.softmax(scores, dim=1)                   # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_out)  # (batch, 1, hidden_dim)
        context = context.squeeze(1)                          # (batch, hidden_dim)
        return context, weights


class HybridIDSNet(nn.Module):
    def __init__(self, seq_len, n_classes, cnn_channels=(32, 64),
                 lstm_hidden=64, lstm_layers=1, bidirectional=True, dropout=0.3):
        super().__init__()

        # --- Spatial feature extraction (CNN) ---
        self.cnn1 = CNNBlock(1, cnn_channels[0], kernel_size=3, pool=True, dropout=dropout)
        self.cnn2 = CNNBlock(cnn_channels[0], cnn_channels[1], kernel_size=3, pool=True, dropout=dropout)

        # --- Temporal sequence modelling (LSTM) ---
        self.lstm = nn.LSTM(
            input_size=cnn_channels[1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # --- Attention over temporal features ---
        self.attention = AdditiveAttention(lstm_out_dim)

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, return_attention=False):
        # x: (batch, seq_len, 1) -> conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)             # (batch, 1, seq_len)
        x = self.cnn1(x)                   # (batch, C1, seq_len/2)
        x = self.cnn2(x)                   # (batch, C2, seq_len/4)
        x = x.permute(0, 2, 1)             # (batch, seq_len/4, C2) for LSTM

        lstm_out, _ = self.lstm(x)         # (batch, seq_len/4, lstm_out_dim)
        context, attn_weights = self.attention(lstm_out)

        logits = self.classifier(context)  # (batch, n_classes)

        if return_attention:
            return logits, attn_weights
        return logits


def build_model(seq_len, n_classes, device="cpu", **kwargs):
    model = HybridIDSNet(seq_len=seq_len, n_classes=n_classes, **kwargs)
    return model.to(device)


if __name__ == "__main__":
    # quick shape sanity-check
    batch, seq_len, n_classes = 8, 40, 5
    dummy = torch.randn(batch, seq_len, 1)
    model = HybridIDSNet(seq_len=seq_len, n_classes=n_classes)
    out = model(dummy)
    print("Output shape:", out.shape)  # expected (8, 5)

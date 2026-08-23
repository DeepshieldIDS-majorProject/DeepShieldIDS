"""
baselines.py

Baseline models used for comparison against HybridIDSNet, mirroring the
paper's evaluation setup:
  - Classical ML: Random Forest, SVM, XGBoost
  - Standalone DL: CNN-only, LSTM-only
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# --------------------------------------------------------------------------
# Classical ML baselines
# --------------------------------------------------------------------------
def train_random_forest(X_train, y_train, **kwargs):
    clf = RandomForestClassifier(
        n_estimators=kwargs.get("n_estimators", 200),
        max_depth=kwargs.get("max_depth", None),
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_svm(X_train, y_train, **kwargs):
    clf = SVC(
        kernel=kwargs.get("kernel", "rbf"),
        C=kwargs.get("C", 1.0),
        probability=True,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def train_xgboost(X_train, y_train, n_classes, **kwargs):
    clf = XGBClassifier(
        n_estimators=kwargs.get("n_estimators", 200),
        max_depth=kwargs.get("max_depth", 6),
        learning_rate=kwargs.get("learning_rate", 0.1),
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


# --------------------------------------------------------------------------
# Standalone deep-learning baselines (for ablation vs HybridIDSNet)
# --------------------------------------------------------------------------
class CNNOnly(nn.Module):
    """CNN-only baseline: spatial features -> global pooling -> classifier."""
    def __init__(self, seq_len, n_classes, channels=(32, 64), dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels[1])
        self.pool2 = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[1], 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


class LSTMOnly(nn.Module):
    """LSTM-only baseline: sequential modelling directly on raw feature sequence."""
    def __init__(self, seq_len, n_classes, hidden_dim=64, bidirectional=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

"""
evaluate_test.py

Evaluate the already-trained HybridIDSNet model on the
prepared CIC-IDS2017 held-out test set.

Uses:
- cic_test_scaler.pkl
- cic_test_label_encoder.pkl
- cic_test_feature_cols.pkl
- hybridids_best.pt
"""

import os
import sys
import joblib
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(
    os.path.join(BASE_DIR, "src")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hybridids_best.pt"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cic_test_scaler.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cic_test_label_encoder.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cic_test_feature_cols.pkl"
)

TEST_X_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cic_test_X.npy"
)

TEST_Y_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cic_test_y.npy"
)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DeepShieldIDS - Held-out Test Evaluation")
print("=" * 70)

print("[INFO] Device:", DEVICE)


# ------------------------------------------------------------
# Check files
# ------------------------------------------------------------

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    FEATURE_PATH
]

for path in required_files:
    if not os.path.exists(path):
        print("[ERROR] Missing file:")
        print(path)
        sys.exit(1)


# ------------------------------------------------------------
# Load model utilities
# ------------------------------------------------------------

from model import build_model


# ------------------------------------------------------------
# Load scaler
# ------------------------------------------------------------

print("[INFO] Loading test scaler...")

scaler = joblib.load(
    SCALER_PATH
)

print(
    "[INFO] Scaler features:",
    scaler.n_features_in_
)


# ------------------------------------------------------------
# Load label encoder
# ------------------------------------------------------------

print("[INFO] Loading label encoder...")

label_encoder = joblib.load(
    ENCODER_PATH
)

class_names = list(
    label_encoder.classes_
)

print(
    "[INFO] Classes:",
    class_names
)


# ------------------------------------------------------------
# Load feature list
# ------------------------------------------------------------

print("[INFO] Loading feature list...")

feature_cols = joblib.load(
    FEATURE_PATH
)

print(
    "[INFO] Feature count:",
    len(feature_cols)
)


# ------------------------------------------------------------
# Load prepared test arrays
# ------------------------------------------------------------

if not os.path.exists(TEST_X_PATH) or not os.path.exists(TEST_Y_PATH):

    print()
    print("[ERROR] Prepared test arrays were not found.")
    print()
    print("Expected:")
    print(TEST_X_PATH)
    print(TEST_Y_PATH)
    print()
    print("Your preprocessing command created the scaler,")
    print("encoder and feature list, but it did not save X_test/y_test.")
    print()
    print("So we will NOT continue with incorrect data.")
    print("Run the test-preparation script with saving enabled first.")
    sys.exit(1)


print("[INFO] Loading test data...")

X_test = np.load(
    TEST_X_PATH
)

y_test = np.load(
    TEST_Y_PATH
)

print(
    "[INFO] X_test shape:",
    X_test.shape
)

print(
    "[INFO] y_test shape:",
    y_test.shape
)


# ------------------------------------------------------------
# Validate shape
# ------------------------------------------------------------

if X_test.ndim == 2:

    X_test = np.expand_dims(
        X_test,
        axis=2
    )

print(
    "[INFO] Final X_test shape:",
    X_test.shape
)


if X_test.shape[1] != 78:

    print(
        "[ERROR] Expected 78 features but got:",
        X_test.shape[1]
    )

    sys.exit(1)


# ------------------------------------------------------------
# Build model
# ------------------------------------------------------------

print("[INFO] Building HybridIDSNet...")

seq_len = X_test.shape[1]

n_classes = len(
    class_names
)

model = build_model(
    seq_len=seq_len,
    n_classes=n_classes,
    device=DEVICE
)


# ------------------------------------------------------------
# Load trained checkpoint
# ------------------------------------------------------------

print(
    "[INFO] Loading checkpoint:",
    MODEL_PATH
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint
)

model.eval()

print(
    "[INFO] Model loaded successfully."
)


# ------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------

test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

label_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)

test_dataset = TensorDataset(
    test_tensor,
    label_tensor
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

print()
print("[INFO] Running predictions...")

all_predictions = []
all_probabilities = []
all_labels = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        logits = model(
            X_batch
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        all_labels.extend(
            y_batch.numpy()
        )


y_true = np.array(
    all_labels
)

y_pred = np.array(
    all_predictions
)

y_prob = np.array(
    all_probabilities
)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print()
print("=" * 70)
print("HELD-OUT TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Accuracy %: {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(n_classes),
        target_names=class_names,
        zero_division=0
    )
)


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(n_classes)
)

print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(
    "          ",
    " ".join(
        f"{name:>12}"
        for name in class_names
    )
)

for i, name in enumerate(class_names):

    print(
        f"{name:>10}",
        " ".join(
            f"{value:12d}"
            for value in cm[i]
        )
    )


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


metrics_path = os.path.join(
    OUTPUT_DIR,
    "cic_heldout_metrics.json"
)

import json

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "num_test_samples": int(len(y_true)),
    "num_features": int(X_test.shape[1]),
    "classes": class_names
}

with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


cm_path = os.path.join(
    OUTPUT_DIR,
    "cic_heldout_confusion_matrix.npy"
)

np.save(
    cm_path,
    cm
)


pred_path = os.path.join(
    OUTPUT_DIR,
    "cic_heldout_predictions.csv"
)

import pandas as pd

results = pd.DataFrame({

    "Actual": label_encoder.inverse_transform(
        y_true
    ),

    "Predicted": label_encoder.inverse_transform(
        y_pred
    )
})

results.to_csv(
    pred_path,
    index=False
)


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print()
print("=" * 70)
print("[INFO] Evaluation completed successfully.")
print()
print("[INFO] Metrics:")
print(metrics_path)

print()
print("[INFO] Confusion matrix:")
print(cm_path)

print()
print("[INFO] Predictions:")
print(pred_path)

print("=" * 70)
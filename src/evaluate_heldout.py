"""
evaluate_heldout.py

Final held-out evaluation for the trained HybridIDSNet.

Uses:
    - Existing trained model
    - Existing CIC scaler
    - Existing CIC label encoder
    - Existing 78-feature list

Important:
    CIC-IDS2017 CSV files contain inconsistent leading spaces
    in column names. This script normalizes those names before
    selecting the 78 features.
"""

import os
import sys
import glob
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import build_model


# ============================================================
# PATHS
# ============================================================

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "CIC_IDS2017"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hybridids_best.pt"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cic_scaler.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cic_label_encoder.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cic_test_feature_cols.pkl"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "heldout"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 256

# Evaluate maximum 100000 samples.
# Change to 0 if you want all samples.
MAX_SAMPLES = 100000

RANDOM_STATE = 42


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(label):

    if pd.isna(label):
        return "BENIGN"

    label = str(label).strip().lower()

    # --------------------------------------------------------
    # BENIGN
    # --------------------------------------------------------

    if label == "benign":
        return "BENIGN"

    # --------------------------------------------------------
    # DoS
    # --------------------------------------------------------

    if (
        "dos" in label
        or "hulk" in label
        or "goldeneye" in label
        or "slowloris" in label
        or "slowhttptest" in label
    ):
        return "DoS"

    # --------------------------------------------------------
    # Probe
    # --------------------------------------------------------

    if "portscan" in label:
        return "Probe"

    if "infiltration" in label:
        return "Probe"

    # --------------------------------------------------------
    # BruteForce
    # --------------------------------------------------------

    if (
        "ftp-patator" in label
        or "ssh-patator" in label
        or "brute force" in label
        or "bruteforce" in label
    ):
        return "BruteForce"

    # --------------------------------------------------------
    # WebAttack
    # --------------------------------------------------------

    if (
        "web attack" in label
        or "webattack" in label
        or "sql injection" in label
        or "xss" in label
    ):
        return "WebAttack"

    # --------------------------------------------------------
    # Unknown
    # --------------------------------------------------------

    return "BENIGN"


# ============================================================
# NORMALIZE CIC COLUMN NAMES
# ============================================================

def normalize_cic_columns(df):

    # Remove leading/trailing spaces
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # CIC dataset has this inconsistent spelling
    rename_map = {
        "Bwd PacketLength Std":
            "Bwd Packet Length Std"
    }

    df = df.rename(
        columns=rename_map
    )

    return df


# ============================================================
# LOAD CIC DATA
# ============================================================

def load_cic_data():

    print()
    print("=" * 70)
    print("[INFO] Loading CIC-IDS2017 CSV files...")
    print("=" * 70)

    csv_files = sorted(
        glob.glob(
            os.path.join(
                DATA_DIR,
                "*.csv"
            )
        )
    )

    if not csv_files:

        raise FileNotFoundError(
            f"No CSV files found in:\n{DATA_DIR}"
        )

    frames = []

    for file_path in csv_files:

        print()
        print(
            f"[INFO] Loading: "
            f"{os.path.basename(file_path)}"
        )

        try:

            df = pd.read_csv(
                file_path,
                low_memory=False
            )

            print(
                f"[INFO] Loaded shape: "
                f"{df.shape}"
            )

            # IMPORTANT:
            # Normalize columns BEFORE concatenation
            df = normalize_cic_columns(
                df
            )

            frames.append(
                df
            )

        except Exception as e:

            print(
                f"[WARNING] Could not load "
                f"{file_path}: {e}"
            )

    if not frames:

        raise RuntimeError(
            "No CIC datasets could be loaded."
        )

    # --------------------------------------------------------
    # Combine all files
    # --------------------------------------------------------

    df = pd.concat(
        frames,
        ignore_index=True
    )

    # Normalize again after concat
    df = normalize_cic_columns(
        df
    )

    print()
    print(
        "[INFO] CIC column names normalized."
    )

    print(
        f"[INFO] Combined shape: "
        f"{df.shape}"
    )

    return df


# ============================================================
# FIND LABEL COLUMN
# ============================================================

def find_label_column(df):

    for col in df.columns:

        normalized = (
            str(col)
            .strip()
            .lower()
        )

        if normalized == "label":

            print(
                f"[INFO] Label column found: "
                f"{col}"
            )

            return col

    print()
    print(
        "[ERROR] Available columns:"
    )

    for i, col in enumerate(
        df.columns
    ):

        print(
            f"{i}: {repr(col)}"
        )

    raise ValueError(
        "Could not find CIC label column."
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    df,
    feature_columns
):

    print()
    print(
        "[INFO] Preparing 78 CIC features..."
    )

    # --------------------------------------------------------
    # Check missing features
    # --------------------------------------------------------

    missing = [
        col
        for col in feature_columns
        if col not in df.columns
    ]

    if missing:

        print()
        print(
            "[ERROR] Missing features:"
        )

        for col in missing:
            print(
                "   ",
                col
            )

        raise ValueError(
            f"{len(missing)} required "
            f"features are missing."
        )

    # --------------------------------------------------------
    # Select features
    # --------------------------------------------------------

    X = df[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # Replace infinity
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for col in X.columns:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Fill invalid values
    # --------------------------------------------------------

    X = X.fillna(0)

    # --------------------------------------------------------
    # Float32
    # --------------------------------------------------------

    X = X.astype(
        np.float32
    )

    print(
        f"[INFO] Final feature matrix: "
        f"{X.shape}"
    )

    return X


# ============================================================
# SAMPLE DATA
# ============================================================

def sample_data(
    X,
    y
):

    total = len(X)

    if (
        MAX_SAMPLES <= 0
        or total <= MAX_SAMPLES
    ):

        print(
            f"[INFO] Using all "
            f"{total} samples."
        )

        return X, y

    print()
    print(
        f"[INFO] Dataset contains "
        f"{total} rows."
    )

    print(
        f"[INFO] Sampling "
        f"{MAX_SAMPLES} rows."
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    indices = rng.choice(
        total,
        size=MAX_SAMPLES,
        replace=False
    )

    indices = np.sort(
        indices
    )

    X_sampled = X.iloc[
        indices
    ].reset_index(
        drop=True
    )

    y_sampled = y.iloc[
        indices
    ].reset_index(
        drop=True
    )

    print(
        f"[INFO] Sampled shape: "
        f"{X_sampled.shape}"
    )

    return (
        X_sampled,
        y_sampled
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_batches(
    model,
    X,
    device
):

    model.eval()

    predictions = []
    probabilities = []

    total = len(X)

    print()
    print(
        f"[INFO] Starting prediction "
        f"for {total} samples..."
    )

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = X[
            start:end
        ]

        # Model expects:
        # (batch, 78, 1)

        batch = np.asarray(
            batch,
            dtype=np.float32
        )

        batch = np.expand_dims(
            batch,
            axis=2
        )

        tensor = torch.from_numpy(
            batch
        ).to(device)

        with torch.no_grad():

            logits = model(
                tensor
            )

            probs = F.softmax(
                logits,
                dim=1
            )

            preds = torch.argmax(
                probs,
                dim=1
            )

        predictions.append(
            preds.cpu().numpy()
        )

        probabilities.append(
            probs.cpu().numpy()
        )

        del tensor
        del logits
        del probs
        del preds

        if (
            start == 0
            or end == total
            or end % (
                BATCH_SIZE * 20
            ) == 0
        ):

            percent = (
                end / total
            ) * 100

            print(
                f"[INFO] Progress: "
                f"{end}/{total} "
                f"({percent:.1f}%)"
            )

    predictions = np.concatenate(
        predictions
    )

    probabilities = np.concatenate(
        probabilities
    )

    return (
        predictions,
        probabilities
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        " DeepShieldIDS - "
        "FINAL HELD-OUT EVALUATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"[INFO] Device: {device}"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        ENCODER_PATH,
        FEATURE_PATH
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path}"
            )

    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------

    print()
    print(
        "[INFO] Loading trained scaler..."
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    print(
        f"[INFO] Scaler expects: "
        f"{scaler.n_features_in_} features"
    )

    # --------------------------------------------------------
    # Load label encoder
    # --------------------------------------------------------

    print(
        "[INFO] Loading trained "
        "label encoder..."
    )

    label_encoder = joblib.load(
        ENCODER_PATH
    )

    class_names = list(
        label_encoder.classes_
    )

    print(
        f"[INFO] Classes: "
        f"{class_names}"
    )

    # --------------------------------------------------------
    # Load feature list
    # --------------------------------------------------------

    print(
        "[INFO] Loading CIC feature list..."
    )

    feature_columns = joblib.load(
        FEATURE_PATH
    )

    print(
        f"[INFO] Feature count: "
        f"{len(feature_columns)}"
    )

    if len(feature_columns) != 78:

        raise ValueError(
            "Feature list must contain "
            "exactly 78 features."
        )

    # --------------------------------------------------------
    # Load CIC data
    # --------------------------------------------------------

    df = load_cic_data()

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label_column = find_label_column(
        df
    )

    print()
    print(
        f"[INFO] Label column: "
        f"{label_column}"
    )

    y = df[
        label_column
    ].apply(
        normalize_label
    )

    print()
    print(
        "[INFO] Normalized label distribution:"
    )

    print(
        y.value_counts()
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = prepare_features(
        df,
        feature_columns
    )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    X, y = sample_data(
        X,
        y
    )

    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------

    y_encoded = label_encoder.transform(
        y
    )

    # --------------------------------------------------------
    # Scale using EXISTING training scaler
    # --------------------------------------------------------

    print()
    print(
        "[INFO] Applying EXISTING "
        "training scaler..."
    )

    X_scaled = scaler.transform(
        X
    )

    X_scaled = np.asarray(
        X_scaled,
        dtype=np.float32
    )

    print(
        f"[INFO] Scaled shape: "
        f"{X_scaled.shape}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print()
    print(
        "[INFO] Loading trained "
        "HybridIDSNet..."
    )

    model = build_model(
        seq_len=78,
        n_classes=len(class_names),
        device=device
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    print(
        "[INFO] Model loaded successfully."
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred, y_prob = predict_batches(
        model,
        X_scaled,
        device
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_encoded,
        y_pred
    )

    precision = precision_score(
        y_encoded,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_encoded,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_encoded,
        y_pred,
        average="weighted",
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            y_encoded,
            y_prob,
            multi_class="ovr",
            average="weighted"
        )

    except Exception as e:

        print(
            f"[WARNING] ROC-AUC unavailable: "
            f"{e}"
        )

        roc_auc = None

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        " FINAL HELD-OUT EVALUATION RESULTS"
    )
    print("=" * 70)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-Score  : {f1:.4f}"
    )

    if roc_auc is not None:

        print(
            f"ROC-AUC   : {roc_auc:.4f}"
        )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        " CLASSIFICATION REPORT"
    )

    print(
        "=" * 70
    )

    report = classification_report(
        y_encoded,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    print(
        report
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_encoded,
        y_pred
    )

    print()
    print(
        "=" * 70
    )

    print(
        " CONFUSION MATRIX"
    )

    print(
        "=" * 70
    )

    print(
        "Rows = Actual"
    )

    print(
        "Columns = Predicted"
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    print(
        cm_df
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "heldout_metrics.txt"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "DeepShieldIDS "
            "FINAL HELD-OUT EVALUATION\n"
        )

        f.write(
            "=" * 60 + "\n"
        )

        f.write(
            f"Samples: {len(y_encoded)}\n"
        )

        f.write(
            f"Accuracy: {accuracy:.6f}\n"
        )

        f.write(
            f"Precision: {precision:.6f}\n"
        )

        f.write(
            f"Recall: {recall:.6f}\n"
        )

        f.write(
            f"F1-Score: {f1:.6f}\n"
        )

        if roc_auc is not None:

            f.write(
                f"ROC-AUC: {roc_auc:.6f}\n"
            )

        f.write(
            "\nClassification Report\n"
        )

        f.write(
            report
        )

        f.write(
            "\nConfusion Matrix\n"
        )

        f.write(
            cm_df.to_string()
        )

    print()
    print(
        f"[INFO] Metrics saved to:"
    )

    print(
        metrics_path
    )

    print()
    print(
        "[INFO] FINAL HELD-OUT "
        "EVALUATION COMPLETED."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
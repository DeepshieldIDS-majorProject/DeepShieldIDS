"""
evaluate_heldout.py

Final held-out evaluation for the trained HybridIDSNet.

IMPORTANT:
- Uses the EXISTING trained model.
- Uses the EXISTING CIC scaler and label encoder.
- Uses the SAME 78 CIC features used during training.
- Evaluates in batches to avoid RAM problems.
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import build_model


# ============================================================
# CONFIGURATION
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

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# Keep this moderate because your system previously had
# CPU memory problems during prediction.
BATCH_SIZE = 256

# Number of samples to evaluate.
# 0 = use all available rows.
MAX_SAMPLES = 100000

RANDOM_STATE = 42


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(label):
    """
    Convert original CIC-IDS2017 labels into the 5 classes
    used by the trained model.
    """

    if pd.isna(label):
        return "BENIGN"

    label = str(label).strip().lower()

    # BENIGN
    if label == "benign":
        return "BENIGN"

    # DoS attacks
    if (
        "dos" in label
        or "hulk" in label
        or "goldeneye" in label
        or "slowloris" in label
        or "slowhttptest" in label
    ):
        return "DoS"

    # PortScan
    if "portscan" in label:
        return "Probe"

    # Infiltration
    if "infiltration" in label:
        return "Probe"

    # Brute Force
    if (
        "ftp-patator" in label
        or "ssh-patator" in label
        or "brute force" in label
        or "bruteforce" in label
    ):
        return "BruteForce"

    # Web attacks
    if (
        "web attack" in label
        or "webattack" in label
        or "sql injection" in label
        or "xss" in label
    ):
        return "WebAttack"

    # Unknown labels are treated as BENIGN only if
    # they cannot be mapped.
    return "BENIGN"


# ============================================================
# LOAD CSV FILES
# ============================================================

def load_cic_data():
    print("=" * 70)
    print("[INFO] Loading CIC-IDS2017 CSV files...")
    print("=" * 70)

    csv_files = sorted(
        glob.glob(
            os.path.join(DATA_DIR, "*.csv")
        )
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {DATA_DIR}"
        )

    frames = []

    for file_path in csv_files:

        print(
            f"[INFO] Loading: {os.path.basename(file_path)}"
        )

        try:
            df = pd.read_csv(
                file_path,
                low_memory=False
            )

            print(
                f"[INFO] Loaded shape: {df.shape}"
            )

            frames.append(df)

        except Exception as e:

            print(
                f"[WARNING] Could not load "
                f"{file_path}: {e}"
            )

    if not frames:
        raise RuntimeError(
            "No CIC datasets could be loaded."
        )

    data = pd.concat(
        frames,
        ignore_index=True
    )

    print()
    print(
        f"[INFO] Combined shape: {data.shape}"
    )

    return data


# ============================================================
# FIND LABEL COLUMN
# ============================================================

def find_label_column(df):
    """
    Robustly find the label column in CIC-IDS2017 dataset.
    CIC files may use different capitalization/spacing.
    """

    # First: exact/common names
    candidates = [
        "Label",
        "label",
        " LABEL",
        "Label ",
        " Label",
        "LABEL",
    ]

    for col in candidates:
        if col in df.columns:
            print(f"[INFO] Label column found: {col}")
            return col

    # Second: normalized column-name search
    for col in df.columns:
        normalized = str(col).strip().lower()

        if normalized == "label":
            print(f"[INFO] Label column found: {col}")
            return col

    # Third: search any column containing 'label'
    for col in df.columns:
        if "label" in str(col).strip().lower():
            print(f"[INFO] Label column found: {col}")
            return col

    print("[ERROR] Available columns:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {repr(col)}")

    raise ValueError(
        "Could not find CIC label column."
    )

def prepare_features(df, feature_columns):

    print()
    print("[INFO] Preparing 78 CIC features...")

    # Make sure every expected feature exists
    missing = [
        col
        for col in feature_columns
        if col not in df.columns
    ]

    if missing:

        print()
        print("[ERROR] Missing features:")

        for col in missing:
            print("   ", col)

        raise ValueError(
            f"{len(missing)} required features are missing."
        )

    X = df[feature_columns].copy()

    # Replace infinity
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Convert everything to numeric
    for col in X.columns:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    # Fill missing values
    X = X.fillna(0)

    # Final float32
    X = X.astype(np.float32)

    print(
        f"[INFO] Final feature matrix: {X.shape}"
    )

    return X


# ============================================================
# SAMPLE DATA
# ============================================================

def sample_data(X, y):

    total = len(X)

    if MAX_SAMPLES <= 0 or total <= MAX_SAMPLES:

        print(
            f"[INFO] Using all {total} samples."
        )

        return X, y

    print()
    print(
        f"[INFO] Dataset contains {total} rows."
    )

    print(
        f"[INFO] Sampling {MAX_SAMPLES} rows "
        f"for memory-safe evaluation."
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    # Stratified sampling manually
    indices = []

    y_array = np.asarray(y)

    classes = np.unique(y_array)

    per_class = max(
        1,
        MAX_SAMPLES // len(classes)
    )

    for cls in classes:

        cls_indices = np.where(
            y_array == cls
        )[0]

        n_take = min(
            per_class,
            len(cls_indices)
        )

        selected = rng.choice(
            cls_indices,
            size=n_take,
            replace=False
        )

        indices.extend(
            selected.tolist()
        )

    # If we still have space, randomly fill it
    if len(indices) < MAX_SAMPLES:

        remaining = np.setdiff1d(
            np.arange(total),
            np.array(indices)
        )

        extra_count = min(
            MAX_SAMPLES - len(indices),
            len(remaining)
        )

        if extra_count > 0:

            extra = rng.choice(
                remaining,
                size=extra_count,
                replace=False
            )

            indices.extend(
                extra.tolist()
            )

    indices = np.array(
        indices,
        dtype=np.int64
    )

    rng.shuffle(indices)

    X_sample = X.iloc[indices].reset_index(
        drop=True
    )

    y_sample = y.iloc[indices].reset_index(
        drop=True
    )

    print(
        f"[INFO] Sampled shape: {X_sample.shape}"
    )

    return X_sample, y_sample


# ============================================================
# BATCH PREDICTION
# ============================================================

@torch.no_grad()
def predict_batches(
    model,
    X,
    device,
    label_encoder
):

    model.eval()

    predictions = []
    probabilities = []

    total = len(X)

    print()
    print(
        f"[INFO] Starting prediction for "
        f"{total} samples..."
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

        batch = X[start:end]

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

        logits = model(tensor)

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

        # Release batch tensor
        del tensor
        del logits
        del probs
        del preds

        if (
            start == 0
            or end == total
            or end % (BATCH_SIZE * 20) == 0
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

    return predictions, probabilities


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(" DeepShieldIDS - FINAL HELD-OUT EVALUATION")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[INFO] Device: {device}"
    )

    # --------------------------------------------------------
    # Check required files
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
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------

    print()
    print("[INFO] Loading trained scaler...")

    scaler = joblib.load(
        SCALER_PATH
    )

    print(
        f"[INFO] Scaler expects: "
        f"{scaler.n_features_in_} features"
    )

    # --------------------------------------------------------
    # Load encoder
    # --------------------------------------------------------

    print(
        "[INFO] Loading trained label encoder..."
    )

    label_encoder = joblib.load(
        ENCODER_PATH
    )

    class_names = list(
        label_encoder.classes_
    )

    print(
        f"[INFO] Classes: {class_names}"
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
            "Expected exactly 78 features."
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_cic_data()

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label_column = find_label_column(df)

    print()
    print(
        f"[INFO] Label column: {label_column}"
    )

    y = df[label_column].apply(
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
    # Scale using TRAINING scaler
    # --------------------------------------------------------

    print()
    print(
        "[INFO] Applying EXISTING training scaler..."
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
        "[INFO] Loading trained HybridIDSNet..."
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
        device,
        label_encoder
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
            f"[WARNING] ROC-AUC unavailable: {e}"
        )

        roc_auc = None

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(" FINAL HELD-OUT EVALUATION RESULTS")
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

    else:

        print(
            "ROC-AUC   : N/A"
        )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(" CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        y_encoded,
        y_pred,
        labels=np.arange(
            len(class_names)
        ),
        target_names=class_names,
        zero_division=0
    )

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_encoded,
        y_pred,
        labels=np.arange(
            len(class_names)
        )
    )

    print()
    print("=" * 70)
    print(" CONFUSION MATRIX")
    print("=" * 70)

    print(
        pd.DataFrame(
            cm,
            index=class_names,
            columns=class_names
        )
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),
        "samples": int(len(y_encoded)),
        "features": int(len(feature_columns)),
        "classes": class_names
    }

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "heldout_metrics.json"
    )

    import json

    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    cm_path = os.path.join(
        OUTPUT_DIR,
        "heldout_confusion_matrix.csv"
    )

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    ).to_csv(
        cm_path
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_path = os.path.join(
        OUTPUT_DIR,
        "heldout_predictions.csv"
    )

    result = pd.DataFrame({
        "Actual": label_encoder.inverse_transform(
            y_encoded
        ),
        "Predicted": label_encoder.inverse_transform(
            y_pred
        )
    })

    result.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("[INFO] Evaluation completed successfully.")
    print("=" * 70)

    print()
    print(
        f"[INFO] Metrics saved to:\n"
        f"{metrics_path}"
    )

    print(
        f"[INFO] Confusion matrix saved to:\n"
        f"{cm_path}"
    )

    print(
        f"[INFO] Predictions saved to:\n"
        f"{predictions_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
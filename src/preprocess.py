"""
preprocess.py

DeepShieldIDS preprocessing for CIC-IDS2017 / UNSW-NB15.

Features:
- Load CSV files
- Normalize attack labels into 5 classes
- Clean NaN / Inf / duplicates
- Select numeric features
- Train/test split
- StandardScaler
- SMOTE
- Reshape for HybridIDSNet
- Save scaler, label encoder and feature columns
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


RANDOM_STATE = 42


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(label):

    label = str(label).strip()
    s = label.lower()

    # BENIGN / NORMAL
    if s in ["benign", "normal"]:
        return "BENIGN"

    # Brute Force / Patator
    if (
        "patator" in s
        or "brute" in s
        or "ftp" in s
        or "ssh" in s
    ):
        return "BruteForce"

    # Web attacks
    if (
        "web attack" in s
        or "webattack" in s
        or "sql injection" in s
        or "xss" in s
        or "infiltration" in s
        or "shellcode" in s
        or "backdoor" in s
        or "worm" in s
    ):
        return "WebAttack"

    # DoS / DDoS
    if (
        "ddos" in s
        or "dos" in s
        or "slowloris" in s
        or "slowhttptest" in s
        or "hulk" in s
        or "goldeneye" in s
    ):
        return "DoS"

    # Probe / scanning / reconnaissance
    if (
        "portscan" in s
        or "port scan" in s
        or "scan" in s
        or "probe" in s
        or "recon" in s
        or "fuzz" in s
    ):
        return "Probe"

    # Unknown attack types
    return "Probe"


# ============================================================
# LOAD CSV
# ============================================================

def load_csv_dataset(path):

    print(f"[INFO] Loading: {path}")

    df = pd.read_csv(
        path,
        low_memory=False
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    # Find label column
    label_col = None

    possible_labels = [
        "Label",
        "label",
        "attack_cat",
        "Attack category",
        "Attack Category"
    ]

    for col in possible_labels:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError(
            f"Label column not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            label_col: "label"
        }
    )

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_dataframe(df):

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    df = df.drop_duplicates()

    df["label"] = df["label"].apply(
        normalize_label
    )

    return df


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(df):

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # label is normally not numeric
    if "label" in numeric_cols:
        numeric_cols.remove("label")

    X = df[numeric_cols].copy()

    # Replace remaining invalid values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    # Ensure float
    X = X.astype(np.float32)

    return X, numeric_cols


# ============================================================
# RESHAPE
# ============================================================

def reshape_for_hybridnet(X):

    return X.reshape(
        X.shape[0],
        X.shape[1],
        1
    ).astype(
        np.float32
    )


# ============================================================
# SMOTE
# ============================================================

def balance_with_smote(
    X,
    y,
    random_state=RANDOM_STATE
):

    print("[INFO] Applying SMOTE")

    smote = SMOTE(
        random_state=random_state,
        sampling_strategy="not majority"
    )

    X_resampled, y_resampled = smote.fit_resample(
        X,
        y
    )

    print(
        "[INFO] Before SMOTE:",
        len(y)
    )

    print(
        "[INFO] After SMOTE:",
        len(y_resampled)
    )

    return X_resampled, y_resampled


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset(
    csv_path,
    tag="dataset",
    test_size=0.2,
    apply_smote=True,
    max_samples=800000
):

    print("\n[INFO] Preparing dataset")

    # --------------------------------------------------------
    # Support CSV OR folder
    # --------------------------------------------------------

    if os.path.isdir(csv_path):

        print(
            f"[INFO] Loading all CSV files from: {csv_path}"
        )

        csv_files = sorted([
            f
            for f in os.listdir(csv_path)
            if f.lower().endswith(".csv")
        ])

        if not csv_files:
            raise ValueError(
                f"No CSV files found in {csv_path}"
            )

        dfs = []

        for filename in csv_files:

            path = os.path.join(
                csv_path,
                filename
            )

            try:

                temp = load_csv_dataset(
                    path
                )

                print(
                    f"[INFO] Loaded shape: {temp.shape}"
                )

                dfs.append(temp)

            except Exception as e:

                print(
                    f"[WARNING] Skipping {filename}: {e}"
                )

        if not dfs:
            raise ValueError(
                "No valid CSV files could be loaded."
            )

        df = pd.concat(
            dfs,
            ignore_index=True
        )

    else:

        df = load_csv_dataset(
            csv_path
        )

    print(
        "[INFO] Original shape:",
        df.shape
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = clean_dataframe(
        df
    )

    print(
        "[INFO] Normalized labels:"
    )

    print(
        df["label"].value_counts()
    )

    # --------------------------------------------------------
    # Limit memory
    # --------------------------------------------------------

    if (
        max_samples is not None
        and len(df) > max_samples
    ):

        print(
            f"[INFO] Sampling {max_samples} rows"
        )

        # Stratified sampling manually
        parts = []

        per_class = max_samples // max(
            1,
            df["label"].nunique()
        )

        for label in df["label"].unique():

            group = df[
                df["label"] == label
            ]

            n = min(
                len(group),
                per_class
            )

            parts.append(
                group.sample(
                    n=n,
                    random_state=RANDOM_STATE
                )
            )

        df = pd.concat(
            parts,
            ignore_index=True
        )

        # Fill remaining slots if necessary
        if len(df) < max_samples:

            remaining = df[
                ~df.index.isin(df.index)
            ]

        print(
            "[INFO] Sampled shape:",
            df.shape
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X_df, feature_cols = prepare_features(
        df
    )

    y_raw = df["label"].values

    print(
        "[INFO] Feature count:",
        len(feature_cols)
    )

    # --------------------------------------------------------
    # Label encoder
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        y_raw
    )

    print(
        "[INFO] Classes:",
        list(label_encoder.classes_)
    )

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X_df.values,

        y,

        test_size=test_size,

        random_state=RANDOM_STATE,

        stratify=y
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # SMOTE
    # --------------------------------------------------------

    if apply_smote:

        X_train, y_train = balance_with_smote(
            X_train,
            y_train
        )

    # --------------------------------------------------------
    # Reshape
    # --------------------------------------------------------

    X_train = reshape_for_hybridnet(
        X_train
    )

    X_test = reshape_for_hybridnet(
        X_test
    )

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    joblib.dump(
        scaler,
        f"models/{tag}_scaler.pkl"
    )

    joblib.dump(
        label_encoder,
        f"models/{tag}_label_encoder.pkl"
    )

    joblib.dump(
        feature_cols,
        f"data/processed/{tag}_feature_cols.pkl"
    )

    print(
        f"[INFO] Saved scaler: models/{tag}_scaler.pkl"
    )

    print(
        f"[INFO] Saved encoder: models/{tag}_label_encoder.pkl"
    )

    print(
        f"[INFO] Saved feature list: "
        f"data/processed/{tag}_feature_cols.pkl"
    )

    print(
        "[INFO] Final train shape:",
        X_train.shape
    )

    print(
        "[INFO] Final test shape:",
        X_test.shape
    )

    return {

        "X_train": X_train.astype(
            np.float32
        ),

        "X_test": X_test.astype(
            np.float32
        ),

        "y_train": y_train.astype(
            np.int64
        ),

        "y_test": y_test.astype(
            np.int64
        ),

        "scaler": scaler,

        "label_encoder": label_encoder,

        "feature_cols": feature_cols,

        "class_names": list(
            label_encoder.classes_
        )
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "preprocess.py loaded successfully."
    )
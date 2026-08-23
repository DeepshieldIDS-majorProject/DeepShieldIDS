"""
test_api_attacks.py

Real CIC-IDS2017 attack validation through FastAPI.

Tests real CIC-IDS2017 samples through the running
DeepShieldIDS API and compares API predictions with
the original CIC labels.
"""

import os
import requests
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"

BASE_DIR = r"D:\DeepShieldIDS"

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "CIC_IDS2017"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cic_scaler.pkl"
)

# Number of samples from each file
SAMPLES_PER_FILE = 20


# ============================================================
# FILES TO TEST
# ============================================================

FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",

    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",

    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",

    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",

    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
]


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(label):

    label = str(label).strip().lower()

    if label in ["benign", "normal"]:
        return "BENIGN"

    if (
        "dos" in label
        or "ddos" in label
        or "generic" in label
    ):
        return "DoS"

    if (
        "portscan" in label
        or "scan" in label
        or "probe" in label
        or "recon" in label
        or "infiltration" in label
    ):
        return "Probe"

    if (
        "patator" in label
        or "brute" in label
    ):
        return "BruteForce"

    if (
        "web" in label
        or "sql injection" in label
        or "xss" in label
        or "shellcode" in label
        or "backdoor" in label
        or "worm" in label
    ):
        return "WebAttack"

    return "Probe"


# ============================================================
# FIND LABEL COLUMN
# ============================================================

def find_label_column(df):

    possible = [
        "Label",
        "label",
        "attack_cat",
        "Attack category"
    ]

    for col in possible:

        if col in df.columns:
            return col

    raise ValueError(
        "Label column not found."
    )


# ============================================================
# LOAD SCALER
# ============================================================

print("[INFO] Loading scaler...")

scaler = joblib.load(
    SCALER_PATH
)

print(
    "[INFO] Scaler expects:",
    scaler.n_features_in_,
    "features"
)


# ============================================================
# TEST API
# ============================================================

all_actual = []
all_predicted = []

total_samples = 0


for filename in FILES:

    path = os.path.join(
        DATA_DIR,
        filename
    )

    print("\n" + "=" * 70)

    print(
        "[INFO] Loading:",
        filename
    )

    if not os.path.exists(path):

        print(
            "[WARNING] File not found:",
            path
        )

        continue


    # --------------------------------------------------------
    # Load only required number of rows
    # --------------------------------------------------------

    df = pd.read_csv(
        path,
        low_memory=False
    )


    df.columns = [
        c.strip()
        for c in df.columns
    ]


    print(
        "[INFO] Dataset shape:",
        df.shape
    )


    label_col = find_label_column(df)


    # --------------------------------------------------------
    # Determine actual labels
    # --------------------------------------------------------

    df["_normalized_label"] = df[
        label_col
    ].apply(
        normalize_label
    )


    print(
        "[INFO] Labels present:"
    )

    print(
        df["_normalized_label"].value_counts()
    )


    # --------------------------------------------------------
    # Select samples
    # --------------------------------------------------------

    selected_parts = []


    for label in [
        "BENIGN",
        "DoS",
        "Probe",
        "BruteForce",
        "WebAttack"
    ]:

        subset = df[
            df["_normalized_label"] == label
        ]


        if len(subset) == 0:
            continue


        n = min(
            SAMPLES_PER_FILE,
            len(subset)
        )


        selected = subset.sample(
            n=n,
            random_state=42
        )


        selected_parts.append(
            selected
        )


    if not selected_parts:

        print(
            "[WARNING] No usable samples found."
        )

        continue


    samples = pd.concat(
        selected_parts,
        ignore_index=True
    )


    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    feature_columns = [
        c for c in samples.columns
        if c != label_col
        and c != "_normalized_label"
        and pd.api.types.is_numeric_dtype(
            samples[c]
        )
    ]


    X = samples[
        feature_columns
    ].copy()


    # Make sure exactly 78 features are used

    if X.shape[1] < scaler.n_features_in_:

        print(
            "[ERROR] Not enough numeric features:",
            X.shape[1]
        )

        continue


    if X.shape[1] > scaler.n_features_in_:

        X = X.iloc[
            :,
            :scaler.n_features_in_
        ]


    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)


    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    X_scaled = scaler.transform(
        X
    )


    # --------------------------------------------------------
    # Send samples individually to API
    # --------------------------------------------------------

    for i in range(
        len(samples)
    ):

        actual_label = samples.iloc[
            i
        ]["_normalized_label"]


        features = X_scaled[
            i
        ].tolist()


        try:

            response = requests.post(
                API_URL,
                json={
                    "features": features
                },
                timeout=30
            )


            if response.status_code != 200:

                print(
                    "[ERROR] API status:",
                    response.status_code
                )

                continue


            result = response.json()


            predicted_label = result[
                "prediction"
            ]


            all_actual.append(
                actual_label
            )

            all_predicted.append(
                predicted_label
            )

            total_samples += 1


        except Exception as e:

            print(
                "[ERROR] API request failed:",
                e
            )


    print(
        "[INFO] Completed:",
        filename
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("          REAL CIC API VALIDATION")
print("=" * 70)


if len(all_actual) == 0:

    print(
        "[ERROR] No samples were successfully tested."
    )

    raise SystemExit


accuracy = accuracy_score(
    all_actual,
    all_predicted
)


print(
    f"\nSamples tested : {len(all_actual)}"
)

print(
    f"API Accuracy   : {accuracy:.4f}"
)

print(
    f"API Accuracy % : {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

classes = [
    "BENIGN",
    "BruteForce",
    "DoS",
    "Probe",
    "WebAttack"
]


print(
    "\n========== Classification Report =========="
)


print(
    classification_report(
        all_actual,
        all_predicted,
        labels=classes,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "========== Confusion Matrix =========="
)


cm = confusion_matrix(
    all_actual,
    all_predicted,
    labels=classes
)


print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print(
    pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.join(
        BASE_DIR,
        "outputs"
    ),
    exist_ok=True
)


results = pd.DataFrame({
    "Actual": all_actual,
    "Predicted": all_predicted
})


output_path = os.path.join(
    BASE_DIR,
    "outputs",
    "api_attack_validation.csv"
)


results.to_csv(
    output_path,
    index=False
)


print(
    "\n[INFO] Detailed results saved to:"
)

print(
    output_path
)

print(
    "\n[INFO] REAL API VALIDATION COMPLETED."
)
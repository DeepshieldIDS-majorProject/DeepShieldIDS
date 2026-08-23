import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"

DATA_PATH = (
    "data/CIC_IDS2017/"
    "Monday-WorkingHours.pcap_ISCX.csv"
)

SCALER_PATH = "models/cic_scaler.pkl"

SAMPLE_SIZE = 100


# ============================================================
# LOAD DATASET
# ============================================================

print("[INFO] Loading CIC dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

df.columns = [
    c.strip()
    for c in df.columns
]

print("[INFO] Dataset shape:", df.shape)


# ============================================================
# FIND LABEL COLUMN
# ============================================================

if "Label" in df.columns:
    label_col = "Label"

elif "label" in df.columns:
    label_col = "label"

else:
    raise Exception("Label column not found")


# ============================================================
# CLEAN LABELS
# ============================================================

df[label_col] = (
    df[label_col]
    .astype(str)
    .str.strip()
)


# ============================================================
# LOAD SCALER
# ============================================================

print("[INFO] Loading scaler...")

scaler = joblib.load(
    SCALER_PATH
)

expected_features = scaler.n_features_in_

print(
    "[INFO] Scaler expects:",
    expected_features,
    "features"
)


# ============================================================
# SELECT NUMERIC FEATURES
# ============================================================

feature_df = df.drop(
    columns=[label_col]
)

numeric_df = feature_df.select_dtypes(
    include=[np.number]
)

print(
    "[INFO] Dataset numeric features:",
    numeric_df.shape[1]
)


if numeric_df.shape[1] != expected_features:

    raise Exception(
        f"Feature mismatch: "
        f"dataset={numeric_df.shape[1]}, "
        f"scaler={expected_features}"
    )


# ============================================================
# SELECT SAMPLE
# ============================================================

sample_df = df.iloc[
    :SAMPLE_SIZE
].copy()

X = sample_df.drop(
    columns=[label_col]
).select_dtypes(
    include=[np.number]
)

y_true = (
    sample_df[label_col]
    .values
)


# ============================================================
# HANDLE NaN / INF
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# ============================================================
# CONVERT TO LIST
# ============================================================

features_list = X.values.tolist()


print(
    "[INFO] Sending",
    len(features_list),
    "real CIC samples to API..."
)


# ============================================================
# BATCH REQUEST
# ============================================================

payload = []

for row in features_list:

    payload.append({
        "features": row
    })


response = requests.post(
    "http://127.0.0.1:8000/predict/batch",
    json=payload,
    timeout=300
)


print(
    "[INFO] HTTP Status:",
    response.status_code
)


if response.status_code != 200:

    print(
        response.text
    )

    raise SystemExit(
        "API prediction failed."
    )


result = response.json()


# ============================================================
# GET PREDICTIONS
# ============================================================

predictions = [
    item["prediction"]
    for item in result["results"]
]


# ============================================================
# NORMALIZE TRUE LABELS
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
        "scan" in label
        or "probe" in label
        or "recon" in label
        or "fuzzer" in label
    ):
        return "Probe"

    if (
        "patator" in label
        or "brute" in label
    ):
        return "BruteForce"

    if (
        "web" in label
        or "shellcode" in label
        or "backdoor" in label
        or "worm" in label
    ):
        return "WebAttack"

    return label


y_true_normalized = [
    normalize_label(x)
    for x in y_true
]


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true_normalized,
    predictions
)


print("\n")
print("=" * 60)
print("       REAL CIC API VALIDATION")
print("=" * 60)

print(
    f"\nSamples tested : {len(predictions)}"
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

print("\n========== Classification Report ==========\n")

print(
    classification_report(
        y_true_normalized,
        predictions,
        zero_division=0
    )
)


# ============================================================
# SAMPLE COMPARISON
# ============================================================

print(
    "\n========== Sample Predictions ==========\n"
)

for i in range(
    min(20, len(predictions))
):

    print(
        f"{i+1:02d}. "
        f"Actual={y_true_normalized[i]:12s} "
        f"Predicted={predictions[i]:12s} "
        f"{'OK' if y_true_normalized[i] == predictions[i] else 'WRONG'}"
    )


print(
    "\n[INFO] API validation completed."
)
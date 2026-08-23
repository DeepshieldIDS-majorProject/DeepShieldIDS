"""
predict.py

Predict attack classes using a trained HybridIDSNet model.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import torch

from model import build_model
from preprocess import clean_dataframe, prepare_labels, prepare_features
from utils import get_device


def load_and_preprocess(csv_path, scaler):
    print(f"[INFO] Loading: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    df.columns = [c.strip() for c in df.columns]

    df = clean_dataframe(df)
    df = prepare_labels(df)

    X, y = prepare_features(df)

    X = scaler.transform(X)

    X = np.expand_dims(X, axis=2)

    return X, y


def predict(model, X, device):
    model.eval()

    with torch.no_grad():

        X = torch.tensor(X, dtype=torch.float32).to(device)

        logits = model(X)

        preds = torch.argmax(logits, dim=1)

    return preds.cpu().numpy()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="CSV file for prediction"
    )

    parser.add_argument(
        "--checkpoint",
        default="models/hybridids_best.pt"
    )

    parser.add_argument(
        "--tag",
        default="cic"
    )

    args = parser.parse_args()

    device = get_device()

    print("[INFO] Device:", device)

    scaler = joblib.load(
        f"models/{args.tag}_scaler.pkl"
    )

    label_encoder = joblib.load(
        f"models/{args.tag}_label_encoder.pkl"
    )

    X, y_true = load_and_preprocess(
        args.data,
        scaler
    )

    seq_len = X.shape[1]

    n_classes = len(
        label_encoder.classes_
    )

    model = build_model(
        seq_len=seq_len,
        n_classes=n_classes,
        device=device
    )

    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=device
        )
    )

    predictions = predict(
        model,
        X,
        device
    )

    pred_labels = label_encoder.inverse_transform(
        predictions
    )

    print("\n========== Prediction Summary ==========")

    unique, counts = np.unique(
        pred_labels,
        return_counts=True
    )

    for cls, cnt in zip(unique, counts):
        print(f"{cls:15s} : {cnt}")

    result = pd.DataFrame()

    result["Predicted_Label"] = pred_labels

    result.to_csv(
        "outputs/predictions.csv",
        index=False
    )

    print("\n[INFO] Predictions saved to outputs/predictions.csv")


if __name__ == "__main__":
    main()
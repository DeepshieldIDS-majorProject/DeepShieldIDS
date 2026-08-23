"""
compare_baselines.py

Benchmarks HybridIDSNet against:
  - Classical ML: Random Forest, SVM, XGBoost
  - Standalone DL: CNN-only, LSTM-only

on the same train/test split, reproducing the comparison table style
reported in the paper.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import build_model
from baselines import train_random_forest, train_svm, train_xgboost, CNNOnly, LSTMOnly
from preprocess import generate_synthetic_dataset, prepare_dataset
from train import make_loaders, train_model
from evaluate import predict
from utils import set_seed, get_device, compute_metrics, save_metrics


def eval_sklearn_model(clf, X_test_flat, y_test):
    y_pred = clf.predict(X_test_flat)
    y_prob = clf.predict_proba(X_test_flat) if hasattr(clf, "predict_proba") else None
    return compute_metrics(y_test, y_pred, y_prob)


def eval_torch_model(model, X_test, y_test, device, batch_size=128):
    ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    y_true, y_pred, y_prob = predict(model, loader, device)
    return compute_metrics(y_true, y_pred, y_prob)


def main():
    parser = argparse.ArgumentParser(description="Compare HybridIDSNet against baselines")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--tag", type=str, default="comparison")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    data_path = args.data if args.data else generate_synthetic_dataset()
    data = prepare_dataset(data_path, tag=args.tag)

    X_train_seq, y_train = data["X_train"], data["y_train"]
    X_test_seq, y_test = data["X_test"], data["y_test"]
    X_train_flat = X_train_seq.squeeze(-1)
    X_test_flat = X_test_seq.squeeze(-1)
    n_classes = len(data["class_names"])
    seq_len = X_train_seq.shape[1]

    results = {}

    print("[INFO] Training Random Forest...")
    rf = train_random_forest(X_train_flat, y_train)
    results["RandomForest"] = eval_sklearn_model(rf, X_test_flat, y_test)

    print("[INFO] Training SVM...")
    svm = train_svm(X_train_flat, y_train)
    results["SVM"] = eval_sklearn_model(svm, X_test_flat, y_test)

    print("[INFO] Training XGBoost...")
    xgb = train_xgboost(X_train_flat, y_train, n_classes=n_classes)
    results["XGBoost"] = eval_sklearn_model(xgb, X_test_flat, y_test)

    print("[INFO] Training CNN-only baseline...")
    cnn_model = CNNOnly(seq_len=seq_len, n_classes=n_classes).to(device)
    train_loader, val_loader, _ = make_loaders(X_train_seq, y_train, X_test_seq, y_test)
    cnn_model, _ = train_model(cnn_model, train_loader, val_loader, device,
                                epochs=args.epochs, checkpoint_path="models/cnn_only_best.pt")
    results["CNN_only"] = eval_torch_model(cnn_model, X_test_seq, y_test, device)

    print("[INFO] Training LSTM-only baseline...")
    lstm_model = LSTMOnly(seq_len=seq_len, n_classes=n_classes).to(device)
    lstm_model, _ = train_model(lstm_model, train_loader, val_loader, device,
                                 epochs=args.epochs, checkpoint_path="models/lstm_only_best.pt")
    results["LSTM_only"] = eval_torch_model(lstm_model, X_test_seq, y_test, device)

    print("[INFO] Training HybridIDSNet...")
    hybrid_model = build_model(seq_len=seq_len, n_classes=n_classes, device=device)
    hybrid_model, _ = train_model(hybrid_model, train_loader, val_loader, device,
                                   epochs=args.epochs, checkpoint_path="models/hybridids_best.pt")
    results["HybridIDSNet"] = eval_torch_model(hybrid_model, X_test_seq, y_test, device)

    print("\n=== Comparison Table ===")
    header = f"{'Model':<15}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1':<10}{'ROC-AUC':<10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        roc = m.get("roc_auc")
        roc_str = f"{roc:.4f}" if roc is not None else "N/A"
        print(f"{name:<15}{m['accuracy']:<10.4f}{m['precision']:<10.4f}"
              f"{m['recall']:<10.4f}{m['f1_score']:<10.4f}{roc_str:<10}")

    save_metrics(results, f"outputs/{args.tag}_comparison.json")
    print(f"\n[INFO] Full comparison saved to outputs/{args.tag}_comparison.json")


if __name__ == "__main__":
    main()

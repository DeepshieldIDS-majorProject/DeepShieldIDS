"""
evaluate.py

Evaluates a trained HybridIDSNet checkpoint on a held-out test set (or a
second dataset, for cross-dataset generalisation testing) and reports
accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from model import build_model
from preprocess import prepare_dataset
from utils import (
    set_seed, get_device, compute_metrics, save_metrics,
    plot_confusion_matrix, print_classification_report
)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_preds.append(preds)
        all_probs.append(probs)
        all_labels.append(y_batch.numpy())

    return (np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs))


def main():
    parser = argparse.ArgumentParser(description="Evaluate HybridIDSNet")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to dataset CSV for evaluation. If omitted, uses synthetic demo data.")
    parser.add_argument("--checkpoint", type=str, default="models/hybridids_best.pt")
    parser.add_argument("--tag", type=str, default="dataset")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    if args.data is None:
        raise ValueError(
            "Please specify dataset path.\n"
            "Example:\n"
            "python src/evaluate.py --data data/CIC_IDS2017 --checkpoint models/hybridids_best.pt"
        )
    data_path = args.data
    data = prepare_dataset(
    data_path,
    tag=args.tag
    )
    seq_len = data["X_test"].shape[1]
    n_classes = len(data["class_names"])

    model = build_model(seq_len=seq_len, n_classes=n_classes, device=device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    test_ds = TensorDataset(torch.tensor(data["X_test"]), torch.tensor(data["y_test"]))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    y_true, y_pred, y_prob = predict(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")

    print("\n=== Classification Report ===")
    print_classification_report(y_true, y_pred, data["class_names"])

    save_metrics(metrics, f"outputs/{args.tag}_metrics.json")
    plot_confusion_matrix(y_true, y_pred, data["class_names"], f"outputs/{args.tag}_confusion_matrix.png")
    print(f"\n[INFO] Metrics saved to outputs/{args.tag}_metrics.json")
    print(f"[INFO] Confusion matrix saved to outputs/{args.tag}_confusion_matrix.png")


if __name__ == "__main__":
    main()

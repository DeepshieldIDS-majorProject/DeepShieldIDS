"""
cross_dataset_eval.py

Implements the paper's cross-dataset generalisation protocol:
  - Harmonise CIC-IDS2017 and UNSW-NB15 to their common (intersected)
    numeric feature space
  - Train HybridIDSNet on dataset A, test on dataset B, and vice versa
  - Report metrics for both directions to assess generalisation beyond a
    single dataset's distribution

Usage:
    python cross_dataset_eval.py --dataset_a path/to/cic_ids2017.csv \
                                  --dataset_b path/to/unsw_nb15.csv
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import build_model
from preprocess import prepare_cross_dataset, generate_synthetic_dataset
from train import make_loaders, train_model
from evaluate import predict
from utils import set_seed, get_device, compute_metrics, save_metrics, plot_confusion_matrix


def evaluate_direction(train_bundle, test_bundle, class_names_train, direction_tag,
                        device, epochs=20, batch_size=128):
    seq_len = train_bundle["X"].shape[1]
    n_classes = len(set(train_bundle["y"].tolist()))

    model = build_model(seq_len=seq_len, n_classes=n_classes, device=device)

    # simple internal split for validation during cross-dataset training
    train_loader, val_loader, _ = make_loaders(
        train_bundle["X"], train_bundle["y"], test_bundle["X"], test_bundle["y"],
        batch_size=batch_size
    )

    model, _ = train_model(
        model, train_loader, val_loader, device,
        epochs=epochs, checkpoint_path=f"models/{direction_tag}_best.pt"
    )

    test_ds = TensorDataset(torch.tensor(test_bundle["X"]), torch.tensor(test_bundle["y"]))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    y_true, y_pred, y_prob = predict(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)

    save_metrics(metrics, f"outputs/{direction_tag}_metrics.json")
    print(f"\n=== {direction_tag} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset generalisation evaluation")
    parser.add_argument("--dataset_a", type=str, default=None, help="Path to CIC-IDS2017 CSV")
    parser.add_argument("--dataset_b", type=str, default=None, help="Path to UNSW-NB15 CSV")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    path_a = args.dataset_a if args.dataset_a else generate_synthetic_dataset(out_path="data/raw/synthetic_a.csv", seed=1)
    path_b = args.dataset_b if args.dataset_b else generate_synthetic_dataset(out_path="data/raw/synthetic_b.csv", seed=2)

    bundles = prepare_cross_dataset(path_a, path_b)

    print("\n--- Direction: Train on A, Test on B ---")
    evaluate_direction(bundles["A"], bundles["B"], bundles["A"]["encoder"].classes_,
                        "trainA_testB", device, epochs=args.epochs)

    print("\n--- Direction: Train on B, Test on A ---")
    evaluate_direction(bundles["B"], bundles["A"], bundles["B"]["encoder"].classes_,
                        "trainB_testA", device, epochs=args.epochs)


if __name__ == "__main__":
    main()

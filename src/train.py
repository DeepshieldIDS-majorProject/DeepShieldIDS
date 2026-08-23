"""
train.py

Trains HybridIDSNet on a preprocessed IDS dataset (CIC-IDS2017 or
UNSW-NB15), with early stopping, checkpointing, and training-curve export.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import build_model
from preprocess import prepare_dataset
from utils import set_seed, get_device, plot_training_curves


def make_loaders(X_train, y_train, X_test, y_test, batch_size=128, val_split=0.1):
    n_val = int(len(X_train) * val_split)
    idx = np.random.permutation(len(X_train))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = TensorDataset(torch.tensor(X_train[train_idx]), torch.tensor(y_train[train_idx]))
    val_ds = TensorDataset(torch.tensor(X_train[val_idx]), torch.tensor(y_train[val_idx]))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).long()

            if train:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, device, epochs=30, lr=1e-3,
                 patience=6, checkpoint_path="models/hybridids_best.pt"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.")
                break

    # restore best weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, history
def main():

    parser = argparse.ArgumentParser(
        description="Train HybridIDSNet"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="dataset"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/hybridids_best.pt"
    )


    args = parser.parse_args()


    set_seed(42)

    device = get_device()

    print(f"[INFO] Using device: {device}")


    data = prepare_dataset(
        args.data,
        tag=args.tag
    )


    seq_len = data["X_train"].shape[2]

    n_classes = len(
        data["class_names"]
    )


    print(
        f"[INFO] seq_len={seq_len}, "
        f"n_classes={n_classes}, "
        f"classes={data['class_names']}"
    )


    model = build_model(
        seq_len=seq_len,
        n_classes=n_classes,
        device=device
    )


    train_loader, val_loader, test_loader = make_loaders(
        data["X_train"],
        data["y_train"],
        data["X_test"],
        data["y_test"],
        batch_size=args.batch_size
    )


    model, history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=args.checkpoint
    )


    plot_training_curves(
        history,
        f"outputs/{args.tag}_training_curves.png"
    )


    print(
        "[INFO] Training complete."
    )


if __name__ == "__main__":
    main()
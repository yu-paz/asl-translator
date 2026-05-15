"""
Outputs:
    - models/best_model.pth        (best model weights by val accuracy)
    - figures/training_curves.png  (loss & accuracy per epoch)
    - figures/confusion_matrix.png (val-set confusion matrix)
    - figures/sample_predictions.png (a grid of val images with predicted labels)
    - logs/training_log.csv        (per-epoch metrics)
"""

import argparse
import csv
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


# ---------- Reproducibility ----------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- Data ---------------------------------------------------------------
def build_dataloaders(data_dir: str, batch_size: int, val_split: float = 0.2):
    """
    Builds train/val DataLoaders from an ImageFolder-style directory.
    Expected structure:
        data_dir/
            i_love_you/*.jpg
            yes/*.jpg
            stop/*.jpg
            hello/*.jpg

    NOTE: We deliberately do NOT use horizontal flip — it can change the meaning
    of an ASL sign by swapping which side of the hand faces the camera.
    """
    # ImageNet normalization (matches MobileNetV2 pretrained weights)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    # Load full dataset twice: once with train transforms, once with val transforms,
    # then split the SAME indices so we can apply the right transform to each subset.
    full_train = datasets.ImageFolder(data_dir, transform=train_tf)
    full_val = datasets.ImageFolder(data_dir, transform=val_tf)

    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    # Deterministic split (seed already set globally)
    indices = list(range(n_total))
    random.shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_set = torch.utils.data.Subset(full_train, train_idx)
    val_set = torch.utils.data.Subset(full_val, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = full_train.classes
    return train_loader, val_loader, class_names


# ---------- Model --------------------------------------------------------------
def build_model(num_classes: int) -> nn.Module:
    """
    MobileNetV2 pretrained on ImageNet. We freeze the convolutional backbone
    (so its ImageNet-trained features are preserved) and only train a new
    classifier head adapted to our 4 classes. This is the standard
    "transfer learning as feature extraction" approach.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze backbone
    for p in model.features.parameters():
        p.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ---------- Train / Eval loops -------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ---------- Plotting -----------------------------------------------------------
def plot_curves(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].set_title("Accuracy")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_confusion(preds, labels, class_names, out_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right"); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix (validation set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_sample_predictions(model, loader, class_names, device, out_path, n=8):
    """Grab one batch, predict, plot up to n images with predicted/true labels."""
    model.eval()
    images, labels = next(iter(loader))
    images_d = images.to(device)
    with torch.no_grad():
        preds = model(images_d).argmax(1).cpu().numpy()

    # Un-normalize for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    n = min(n, images.size(0))
    cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.7))
    axes = np.array(axes).reshape(-1)
    for i in range(n):
        img = images[i].permute(1, 2, 0).numpy() * std + mean
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        true_name = class_names[labels[i].item()]
        pred_name = class_names[preds[i]]
        ok = preds[i] == labels[i].item()
        axes[i].set_title(f"P: {pred_name}\nT: {true_name}",
                          color=("green" if ok else "red"), fontsize=9)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


# ---------- Main ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = build_dataloaders(args.data_dir, args.batch_size)
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    # Only update parameters that require grad (the classifier head)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    log_path = out / "logs" / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

            history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
            history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
            writer.writerow([epoch, tr_loss, tr_acc, val_loss, val_acc])

            print(f"Epoch {epoch:2d}/{args.epochs} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": val_acc,
                }, out / "models" / "best_model.pth")

    # Final figures use the LAST epoch's predictions (close to best for a small run)
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    plot_curves(history, out / "figures" / "training_curves.png")
    plot_confusion(val_preds, val_labels, class_names, out / "figures" / "confusion_matrix.png")
    plot_sample_predictions(model, val_loader, class_names, device,
                            out / "figures" / "sample_predictions.png")
    print(f"Saved figures to {out / 'figures'}/")


if __name__ == "__main__":
    main()

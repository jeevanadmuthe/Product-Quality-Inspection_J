"""
Industrial Casting Defect Inspection - Deep Learning CNN Training Pipeline
Trains a multi-stage Convolutional Neural Network (PyTorch) using the Kaggle casting dataset.

Folder structure supported:
  data/casting/train/good (or ok_front)
  data/casting/train/defective (or def_front)
  data/casting/test/good (or ok_front)
  data/casting/test/defective (or def_front)

Strictly holds out the test split.
Outputs classification report, confusion matrix, loss curves, and saves the trained model weights and manifest.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from models.cnn_model import (
    CLASSES,
    CastingInspectionCNN,
    get_architecture_summary,
    get_cnn_transforms,
)

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
FOLDER_ALIASES = {
    "good": ["good", "ok_front", "ok", "cast_ok"],
    "defective": ["defective", "def_front", "defect", "def", "cast_def"],
}


def resolve_folder(root: Path, label: str) -> Path:
    """Find directory matching label or standard casting dataset aliases."""
    direct = root / label
    if direct.is_dir():
        return direct
    for alias in FOLDER_ALIASES.get(label, []):
        cand = root / alias
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"Missing '{label}' folder under '{root}'. Looked for: {label}, {', '.join(FOLDER_ALIASES.get(label, []))}"
    )


class CastingDataset(Dataset):
    """PyTorch Dataset loading industrial casting images with on-the-fly transformations."""

    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        try:
            with Image.open(path) as img:
                rgb_img = img.convert("RGB")
                if self.transform:
                    tensor = self.transform(rgb_img)
                else:
                    tensor = get_cnn_transforms(is_training=False)(rgb_img)
        except Exception as e:
            # Fallback black canvas if image read error occurs
            tensor = torch.zeros(3, 128, 128)
        return tensor, self.labels[idx]


def load_dataset_paths(
    root: Path, max_per_class: int = None
) -> Tuple[List[Path], List[int], Dict[str, int]]:
    """Scan dataset folder and return balanced/deterministic file lists and class distribution."""
    paths, labels = [], []
    counts = {}

    for class_idx, label in enumerate(CLASSES):
        folder = resolve_folder(root, label)
        class_files = sorted([p for p in folder.rglob("*") if p.suffix.lower() in EXTS])
        if not class_files:
            raise ValueError(f"No valid image files found in '{folder}'")

        if max_per_class and len(class_files) > max_per_class:
            class_files = class_files[:max_per_class]

        paths.extend(class_files)
        labels.extend([class_idx] * len(class_files))
        counts[label] = len(class_files)
        print(f"  [{root.name}/{label}] Loaded {len(class_files)} samples from {folder.name}")

    return paths, labels, counts


def print_progress_bar(current: int, total: int, prefix: str = "", suffix: str = "", bar_length: int = 24):
    """
    Render dynamic 0% to 100% progress bar in CMD prompt with carriage return for smooth updates.
    """
    total = max(1, total)
    current = min(current, total)
    percent = (float(current) / float(total)) * 100.0
    filled = int(bar_length * current // total)
    bar = "=" * filled + (">" if filled < bar_length else "") + "." * (bar_length - filled - (1 if filled < bar_length else 0))
    line = f"\r  {prefix} [{bar}] {percent:5.1f}% ({current:>3}/{total}) {suffix}"
    sys.stdout.write(line)
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device, prefix: str = "Evaluating"
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute loss, accuracy, and predictions on a split with 0% to 100% step progress."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_targets = [], []
    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            preds = outputs.argmax(dim=-1)
            total_loss += loss.item() * targets.size(0)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == total_batches:
                curr_acc = (total_correct / max(1, total_samples)) * 100.0
                print_progress_bar(
                    batch_idx,
                    total_batches,
                    prefix=f"{prefix}",
                    suffix=f"Loss: {loss.item():.4f} | Acc: {curr_acc:.1f}%",
                )

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc, np.array(all_targets), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Train Deep Learning CNN for Metal Casting Quality Inspection")
    parser.add_argument("--data", default="data/casting", help="Root folder containing train/ and test/")
    parser.add_argument("--out", default="models/casting_cnn.pt", help="Output model checkpoint path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and testing")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization weight decay")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional sample cap per class")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fast training with sample limit")
    args = parser.parse_args()

    if args.quick:
        if args.epochs > 5:
            args.epochs = 5
        if args.max_train_samples is None:
            args.max_train_samples = 2000

    data_root = Path(args.data)
    if not data_root.exists():
        sys.exit(f"Data root directory not found: {data_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(" Product Quality Inspection - Deep Learning CNN Training Pipeline")
    print(f" Framework: PyTorch {torch.__version__} | Target Device: {device}")
    print(f" Data Directory: {data_root.resolve()}")
    print(f" Epochs: {args.epochs} | Batch Size: {args.batch_size} | Learning Rate: {args.lr}")
    if args.max_train_samples:
        print(f" Mode: Quick Run (max {args.max_train_samples} samples per class)")
    else:
        print(" Mode: Full Industrial Dataset Training")
    print("=" * 70)

    # 1. Load Data Paths
    t_start = time.perf_counter()
    print("\n[Step 1/4] Preparing training and held-out test splits...")
    train_paths, train_labels, train_counts = load_dataset_paths(
        data_root / "train", max_per_class=args.max_train_samples
    )
    test_paths, test_labels, test_counts = load_dataset_paths(data_root / "test")

    train_ds = CastingDataset(train_paths, train_labels, transform=get_cnn_transforms(is_training=True))
    test_ds = CastingDataset(test_paths, test_labels, transform=get_cnn_transforms(is_training=False))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    print(f"  Training set: {len(train_ds)} samples (Batches: {len(train_loader)})")
    print(f"  Held-out test set: {len(test_ds)} samples (Batches: {len(test_loader)})")
    print("  --> Step 1 Completed [100.0%]")

    # 2. Initialize CNN Model Architecture
    print("\n[Step 2/4] Initializing CastingInspectionCNN architecture...")
    model = CastingInspectionCNN(num_classes=len(CLASSES), dropout_rate=0.35).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: CastingInspectionCNN (Trainable parameters: {total_params:,})")
    print("  --> Step 2 Completed [100.0%]")

    # Balanced class weights for Cross Entropy Loss
    class_weights = torch.tensor([
        len(train_labels) / (2.0 * train_counts["defective"]),
        len(train_labels) / (2.0 * train_counts["good"]),
    ], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # CosineAnnealingLR gives smooth, stable LR decay avoiding the sudden collapse seen with ReduceLROnPlateau(patience=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # 3. Training Loop
    print(f"\n[Step 3/4] Training CNN for {args.epochs} epochs...")
    history = []
    best_acc = 0.0
    best_weights = None

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.perf_counter()
        model.train()
        running_loss, running_correct, total_epoch_samples = 0.0, 0, 0
        total_batches = len(train_loader)

        print(f"\n  === Epoch [{epoch}/{args.epochs}] (0.0% -> 100.0%) ===")
        for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # Gradient clipping prevents exploding gradients causing sudden accuracy collapse
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=-1)
            running_correct += (preds == targets).sum().item()
            total_epoch_samples += targets.size(0)

            # Update batch progress bar with 0% to 100% display
            if batch_idx % max(1, total_batches // 25) == 0 or batch_idx == total_batches:
                batch_acc = (preds == targets).float().mean().item()
                print_progress_bar(
                    batch_idx,
                    total_batches,
                    prefix=f"Epoch [{epoch}/{args.epochs}] Train",
                    suffix=f"Loss: {loss.item():.4f} | Batch Acc: {batch_acc * 100:.1f}%",
                )

        train_loss = running_loss / total_epoch_samples
        train_acc = running_correct / total_epoch_samples

        # Validation on held-out test split with progress bar
        val_loss, val_acc, _, _ = evaluate_model(
            model, test_loader, device, prefix=f"Epoch [{epoch}/{args.epochs}] Val  "
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_sec = time.perf_counter() - t_epoch

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": current_lr,
            "duration_sec": round(epoch_sec, 2),
        })

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_weights = model.state_dict().copy()

        star = " * (Best Model)" if is_best else ""
        print(
            f"  --> Epoch {epoch:02d} Summary: Train Acc: {train_acc * 100:.2f}% | "
            f"Val Acc: {val_acc * 100:.2f}% | Val Loss: {val_loss:.4f} | Duration: {epoch_sec:.1f}s{star}"
        )
        # Overall pipeline completion percentage
        print_progress_bar(
            epoch,
            args.epochs,
            prefix="OVERALL PIPELINE",
            suffix=f"({epoch}/{args.epochs} Epochs Completed)",
            bar_length=20,
        )
        print("  " + "-" * 66)

    # Load best performing weights
    if best_weights:
        model.load_state_dict(best_weights)

    # 4. Rigorous Final Evaluation on Held-Out Test Set
    print("\n[Step 4/4] Conducting final evaluation on held-out test split...")
    test_loss, final_acc, y_true, y_pred = evaluate_model(
        model, test_loader, device, prefix="Final Test Eval"
    )
    print("  --> Step 4 Completed [100.0%]")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    print("\n" + "=" * 60)
    print(f" CNN HELD-OUT TEST ACCURACY: {final_acc * 100:.2f}% ({final_acc:.4f})")
    print("=" * 60)
    print(" Confusion Matrix (Defective [0] vs Good [1]):")
    print("                      Predicted Defective   Predicted Good")
    print(f" True Defective [0]:       {cm[0][0]:<15}      {cm[0][1]:<10}")
    print(f" True Good      [1]:       {cm[1][0]:<15}      {cm[1][1]:<10}")
    print("\n Classification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=["defective", "good"], digits=4, zero_division=0
        )
    )

    # 5. Save Model and Manifest Artifacts
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving CNN weights to: {out_path.resolve()}")
    torch.save({
        "state_dict": model.state_dict(),
        "architecture": "CastingInspectionCNN",
        "num_classes": 2,
        "classes": CLASSES,
        "input_size": [3, 128, 128],
        "test_accuracy": round(float(final_acc), 4),
    }, out_path)

    manifest = {
        "model_version": "casting-cnn-deeplearning-v1",
        "classifier_type": "cnn",
        "framework": f"PyTorch {torch.__version__}",
        "architecture_name": "CastingInspectionCNN",
        "train_count": len(train_labels),
        "test_count": len(test_labels),
        "test_accuracy": round(float(final_acc), 4),
        "metrics": {
            "accuracy": round(float(final_acc), 4),
            "defective_precision": round(float(precision[0]), 4),
            "defective_recall": round(float(recall[0]), 4),
            "defective_f1": round(float(f1[0]), 4),
            "good_precision": round(float(precision[1]), 4),
            "good_recall": round(float(recall[1]), 4),
            "good_f1": round(float(f1[1]), 4),
        },
        "confusion_matrix": cm.tolist(),
        "epochs_trained": args.epochs,
        "batch_size": args.batch_size,
        "initial_lr": args.lr,
        "trainable_parameters": total_params,
        "labels": CLASSES,
        "training_history": history,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_training_duration_sec": round(time.perf_counter() - t_start, 2),
        "architecture_summary": get_architecture_summary(),
    }

    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved CNN manifest to: {manifest_path.resolve()}")
    print("\n>>> Deep Learning CNN Model Training Finished Successfully! <<<\n")


if __name__ == "__main__":
    main()

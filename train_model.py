"""
Train a high-performance learned HOG + SVM image classifier using the Kaggle casting dataset.
Supports standard folder names:
  data/casting/train/good (or ok_front)
  data/casting/train/defective (or def_front)
  data/casting/test/good (or ok_front)
  data/casting/test/defective (or def_front)

Keeps the test split strictly held out.
Outputs a confusion matrix and classification report.
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Suppress expected scikit-learn probability deprecation if legacy SVC(probability=True) is used
warnings.filterwarnings("ignore", category=FutureWarning)

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

FOLDER_ALIASES = {
    "good": ["good", "ok_front", "ok", "cast_ok"],
    "defective": ["defective", "def_front", "defect", "def", "cast_def"],
}


def features(path):
    """Extract 8100-dim HOG feature vector from RGB image."""
    image = Image.open(path).convert("RGB").resize((128, 128))
    return hog(
        np.asarray(image),
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1,
    )


def resolve_folder(root: Path, label: str) -> Path:
    """Find folder matching label or its common aliases."""
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


def load_split(root: Path, max_per_class: int = None, n_jobs: int = -1):
    """Load split with parallel feature extraction and progress feedback."""
    xs, ys, ids = [], [], []
    root = Path(root)

    for label in ("good", "defective"):
        folder = resolve_folder(root, label)
        paths = [p for p in folder.rglob("*") if p.suffix.lower() in EXTS]
        if not paths:
            raise ValueError(f"No valid image files found in '{folder}'")

        if max_per_class and len(paths) > max_per_class:
            # Deterministic subset
            paths = paths[:max_per_class]

        print(f"  [{root.name}/{label}] Loading {len(paths)} images from {folder.name}...")
        batch_feats = joblib.Parallel(n_jobs=n_jobs, batch_size=64)(
            joblib.delayed(features)(p) for p in paths
        )
        xs.extend(batch_feats)
        ys.extend([label] * len(paths))
        ids.extend([str(p) for p in paths])

    return np.asarray(xs), np.asarray(ys), ids


def main():
    parser = argparse.ArgumentParser(description="Train HOG + SVM metal casting classifier")
    parser.add_argument("--data", default="data/casting", help="Root folder containing train/ and test/")
    parser.add_argument("--out", default="models/casting_svm.joblib", help="Output model path")
    parser.add_argument("--classifier", choices=["rbf", "linear"], default="rbf",
                        help="Classifier kernel: 'rbf' (full SVM) or 'linear' (fast calibrated LinearSVC)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Optional maximum training samples per class (for quick training)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel worker processes for feature extraction")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fast calibrated linear classifier with 1000 samples per class")
    args = parser.parse_args()

    if args.quick:
        args.classifier = "linear"
        if args.max_train_samples is None:
            args.max_train_samples = 1500

    root = Path(args.data)
    if not root.exists():
        sys.exit(f"Data root directory not found: {root}")

    print("=" * 60)
    print(" Product Quality Inspection - Model Training Pipeline")
    print(f" Data directory: {root}")
    print(f" Classifier type: {args.classifier.upper()}")
    if args.max_train_samples:
        print(f" Max train samples per class: {args.max_train_samples}")
    print("=" * 60)

    # 1. Load Training Data
    t0 = time.perf_counter()
    print("\n[Step 1/4] Extracting features for training set...")
    x_train, y_train, train_ids = load_split(root / "train", max_per_class=args.max_train_samples, n_jobs=args.n_jobs)
    print(f"  Loaded {len(y_train)} training samples in {time.perf_counter() - t0:.2f}s (feature dim: {x_train.shape[1]})")

    # 2. Load Held-Out Test Data
    t1 = time.perf_counter()
    print("\n[Step 2/4] Extracting features for held-out test set...")
    x_test, y_test, test_ids = load_split(root / "test", n_jobs=args.n_jobs)
    print(f"  Loaded {len(y_test)} test samples in {time.perf_counter() - t1:.2f}s")

    # 3. Model Construction & Fitting
    print(f"\n[Step 3/4] Fitting {args.classifier.upper()} model pipeline...")
    t2 = time.perf_counter()
    if args.classifier == "linear":
        base_clf = LinearSVC(dual="auto", max_iter=3000, class_weight="balanced", random_state=42)
        clf = CalibratedClassifierCV(base_clf, cv=3)
        model = make_pipeline(StandardScaler(), clf)
    else:
        # Full RBF SVM with Platt scaling for well-calibrated confidence scores
        model = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
        )

    model.fit(x_train, y_train)
    fit_duration = time.perf_counter() - t2
    print(f"  Model fitting completed in {fit_duration:.2f}s")

    # 4. Evaluation
    print("\n[Step 4/4] Evaluating on held-out test split...")
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)

    print("\n" + "-" * 50)
    print(f" Held-Out Test Accuracy: {acc * 100:.2f}% ({accuracy_score(y_test, pred):.4f})")
    print("-" * 50)
    print(" Confusion Matrix (Defective vs Good):")
    cm = confusion_matrix(y_test, pred, labels=["defective", "good"])
    print(f"                Predicted Defective   Predicted Good")
    print(f" True Defective:       {cm[0][0]:<15}      {cm[0][1]:<10}")
    print(f" True Good:            {cm[1][0]:<15}      {cm[1][1]:<10}")
    print("\n Detailed Classification Report:")
    print(classification_report(y_test, pred, zero_division=0))

    # Save artifact files
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving serialized model to: {out}")
    joblib.dump(model, out)

    manifest = {
        "model_version": f"casting-hog-svm-v1-{args.classifier}",
        "classifier_type": args.classifier,
        "train_count": len(y_train),
        "test_count": len(y_test),
        "test_accuracy": round(float(acc), 4),
        "confusion_matrix": cm.tolist(),
        "train_files": train_ids,
        "test_files": test_ids,
        "labels": ["good", "defective"],
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to: {manifest_path}")
    print("\n>>> Model training and validation finished successfully! <<<\n")


if __name__ == "__main__":
    main()

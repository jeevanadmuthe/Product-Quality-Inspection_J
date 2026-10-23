"""
Diagnostic: verify class-label mapping, preprocessing, and Grad-CAM for casting CNN.
Run from the project root: python diagnose_model.py
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

from models.cnn_model import (
    CLASSES, CastingInspectionCNN, get_cnn_transforms,
    generate_gradcam, predict_with_tta,
)

CKPT = Path("models/casting_cnn.pt")

print("=" * 65)
print("  CNN DIAGNOSTIC REPORT")
print("=" * 65)

# 1. Checkpoint keys & saved metadata
ckpt = torch.load(CKPT, map_location="cpu")
print("\n[1] Checkpoint metadata:")
for k, v in ckpt.items():
    if k != "state_dict":
        print(f"    {k}: {v}")

# 2. Class-label mapping check
saved_classes = ckpt.get("classes", None)
print(f"\n[2] Class-label mapping:")
print(f"    Checkpoint 'classes' key : {saved_classes}")
print(f"    Code CLASSES constant    : {list(CLASSES)}")
if saved_classes is None:
    print("    WARNING: checkpoint has no 'classes' key - using code constant")
elif list(saved_classes) != list(CLASSES):
    print("    *** MISMATCH! Labels are SWAPPED between checkpoint and code ***")
    print(f"    Checkpoint index 0 = '{saved_classes[0]}', Code index 0 = '{CLASSES[0]}'")
else:
    print("    OK - class order is consistent")

# 3. Load model
model = CastingInspectionCNN(num_classes=2)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"\n[3] Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# 4. Preprocessing consistency
print("\n[4] Preprocessing pipeline (inference):")
tf = get_cnn_transforms(is_training=False, image_size=128)
for t in tf.transforms:
    print(f"    {t}")

# 5. Prediction sanity on known-class images from dataset
print("\n[5] Sanity predictions on actual dataset images:")
data_root = Path("data/casting/test")
checked = 0
correct = 0
folder_candidates = {
    "defective": ["defective", "def_front"],
    "good": ["good", "ok_front"],
}
for class_idx, label in enumerate(CLASSES):
    folder = None
    for alias in folder_candidates[label]:
        candidate = data_root / alias
        if candidate.is_dir():
            folder = candidate
            break
    if folder is None:
        print(f"    SKIP: no folder found for '{label}'")
        continue

    images = (
        sorted(folder.glob("*.jpeg"))[:5]
        + sorted(folder.glob("*.jpg"))[:5]
        + sorted(folder.glob("*.png"))[:5]
    )[:5]
    for img_path in images:
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
        probs = predict_with_tta(model, rgb, image_size=128, n_augments=3)
        pred_idx = int(np.argmax(probs))
        pred_label = CLASSES[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        match = "CORRECT" if pred_label == label else "WRONG!!!"
        print(f"    [{label:>10}]  pred={pred_label:>10}  conf={confidence:5.1f}%  {match}  {img_path.name}")
        checked += 1
        if pred_label == label:
            correct += 1

print(f"\n    Spot-check result: {correct}/{checked} correct")

# 6. Grad-CAM gradient flow check
print("\n[6] Grad-CAM gradient flow:")
dummy_img = Image.new("RGB", (200, 200), color=(160, 160, 160))
tf_inf = get_cnn_transforms(is_training=False, image_size=128)
tensor = tf_inf(dummy_img).unsqueeze(0)

with torch.no_grad():
    logits = model(tensor)
    probs_raw = F.softmax(logits, dim=-1)[0].numpy()
pred_idx_raw = int(probs_raw.argmax())

try:
    cam = generate_gradcam(model, tensor, target_class=pred_idx_raw)
    if cam is None or cam.max() == 0:
        print("    WARNING: Grad-CAM returned all-zeros (gradients may not be flowing)")
    else:
        print(f"    OK - heatmap range [{cam.min():.3f}, {cam.max():.3f}], nonzero pixels={np.count_nonzero(cam)}")
except AttributeError as e:
    print(f"    ERROR: {e}")
    print("    CAUSE: feat_maps.grad is None - gradients not retained correctly")
except Exception as e:
    print(f"    ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 65)
print("  DIAGNOSTIC COMPLETE")
print("=" * 65)

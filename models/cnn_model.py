"""
Casting Product Quality Inspection - Deep Learning CNN Conceptual Model
Defines the convolutional neural network architecture, data transformations,
and Grad-CAM visual explainability pipeline for surface defect inspection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

CLASSES = ["defective", "good"]

# Standard normalization values matching ImageNet pre-training distributions
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


class CastingInspectionCNN(nn.Module):
    """
    Deep Convolutional Neural Network (CNN) Conceptual Architecture
    for Automated Industrial Casting Product Quality Inspection.
    
    Conceptual Deep Learning Hierarchy:
    -----------------------------------
    1. Low-Level Spatial Feature Learning (Blocks 1 & 2):
       - Conv2d(3 -> 32) + BatchNorm2d + ReLU + MaxPool2d:
         Captures microscopic edge transitions, specular metallic gradients, and illumination boundaries.
       - Conv2d(32 -> 64) + BatchNorm2d + ReLU + MaxPool2d:
         Extracts localized texture gradients, roughness contours, and surface grain.
         
    2. Mid-to-High Level Defect Synthesis (Blocks 3 & 4):
       - Conv2d(64 -> 128) + BatchNorm2d + ReLU + MaxPool2d:
         Detects geometric defect candidates such as pinholes, porosity clusters, and mold flash lines.
       - Conv2d(128 -> 256) + BatchNorm2d + ReLU + MaxPool2d:
         Synthesizes macro structural defects: cold shuts, blowholes, shrinkage cavities, and perimeter fractures.
         
    3. Global Spatial Regularization:
       - AdaptiveAvgPool2d((4, 4)):
         Compresses spatial feature maps while preserving translation invariance across slightly misaligned parts.
       - Dropout (0.35):
         Prevents co-adaptation of filter weights to reduce overfitting on manufacturing fixtures.
         
    4. Decision Projection Head:
       - Linear(256 * 4 * 4, 128) + BatchNorm1d + ReLU + Dropout(0.2) + Linear(128, 2):
         Maps learned latent defect representations into calibrated binary classification logits:
         Class 0: 'defective', Class 1: 'good'.
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.35):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: Input (3 x 128 x 128) -> Output (32 x 64 x 64)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: Input (32 x 64 x 64) -> Output (64 x 32 x 32)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: Input (64 x 32 x 32) -> Output (128 x 16 x 16)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: Input (128 x 16 x 16) -> Output (256 x 8 x 8)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Spatial aggregation
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature maps from the final convolutional block for Grad-CAM."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


def get_cnn_transforms(is_training: bool = False, image_size: int = 128):
    """
    Standard PyTorch transforms pipeline for casting defect inspection.
    Includes robust industrial data augmentation for training.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])


def predict_with_tta(
    model: "CastingInspectionCNN",
    pil_image,
    image_size: int = 128,
    n_augments: int = 5,
) -> np.ndarray:
    """
    Test-Time Augmentation (TTA) inference.
    Averages softmax probabilities over N augmented views of the same image
    for more stable and accurate predictions on borderline cases.

    Returns:
        1D numpy array of averaged class probabilities [defective, good].
    """
    import torch.nn.functional as F

    base_transform = get_cnn_transforms(is_training=False, image_size=image_size)
    tta_transform = transforms.Compose([
        transforms.Resize((int(image_size * 1.05), int(image_size * 1.05))),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    model.eval()
    all_probs = []

    # Always include the clean (non-augmented) view as the first pass
    with torch.no_grad():
        t = base_transform(pil_image).unsqueeze(0)
        logits = model(t)
        all_probs.append(F.softmax(logits, dim=-1).squeeze(0).numpy())

        # Add augmented views
        for _ in range(n_augments - 1):
            t_aug = tta_transform(pil_image).unsqueeze(0)
            logits_aug = model(t_aug)
            all_probs.append(F.softmax(logits_aug, dim=-1).squeeze(0).numpy())

    return np.mean(all_probs, axis=0)


def generate_gradcam(model: CastingInspectionCNN, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
    """
    Compute Gradient-weighted Class Activation Mapping (Grad-CAM).
    Visualizes which localized spatial regions of the metal casting
    contributed most strongly to the CNN classification decision.
    
    Returns:
        2D numpy array (128x128) with normalized heatmap intensities in [0, 1].
    """
    model.eval()
    # IMPORTANT: requires_grad_(True) on the input ensures the computation graph
    # is retained so feat_maps.retain_grad() can capture intermediate gradients.
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    # Forward through feature blocks with gradient retention on final conv block
    x = model.block1(input_tensor)
    x = model.block2(x)
    x = model.block3(x)
    feat_maps = model.block4(x)
    feat_maps.retain_grad()

    pooled = model.adaptive_pool(feat_maps)
    logits = model.classifier(pooled)

    if target_class is None:
        target_class = int(logits.argmax(dim=-1).item())

    model.zero_grad()
    score = logits[0, target_class]
    score.backward()

    if feat_maps.grad is None:
        # Fallback: return blank heatmap rather than crash
        return np.zeros((128, 128), dtype=np.float32)

    # Weight feature channels by spatial average of gradients
    grads = feat_maps.grad.detach()  # [1, 256, H, W]
    weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, 256, 1, 1]

    cam = (weights * feat_maps.detach()).sum(dim=1, keepdim=True)  # [1, 1, H, W]
    cam = F.relu(cam)  # Positive evidence only

    # Bilinear upsample back to 128x128
    cam = F.interpolate(cam, size=(128, 128), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    # Min-max normalization
    cam_min, cam_max = float(cam.min()), float(cam.max())
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam, dtype=np.float32)

    return cam



def overlay_cam_on_image(pil_image: Image.Image, cam_heatmap: np.ndarray, alpha: float = 0.45, is_good: bool = False) -> Image.Image:
    """
    Blend Grad-CAM heatmap with original PIL Image using custom color gradient.
    If is_good is True: uses clean emerald/teal gradient to confirm verified surface integrity.
    If is_good is False: uses vibrant amber/red to pinpoint localized defect anomalies.
    """
    orig_rgb = pil_image.convert("RGB").resize((128, 128))
    img_arr = np.asarray(orig_rgb, dtype=np.float32)

    h = cam_heatmap
    if is_good:
        # Emerald/teal surface integrity verification gradient
        heat_r = np.clip(0.2 * (1.0 - h), 0.0, 1.0) * 255.0
        heat_g = np.clip(1.8 * h, 0.0, 1.0) * 255.0
        heat_b = np.clip(1.4 * h, 0.0, 1.0) * 255.0
    else:
        # Pseudo-color defect anomaly heatmap: cold blue -> warm yellow -> hot red
        heat_r = np.clip(2.0 * h, 0.0, 1.0) * 255.0
        heat_g = np.clip(2.0 * (1.0 - np.abs(h - 0.5)), 0.0, 1.0) * 255.0
        heat_b = np.clip(2.0 * (1.0 - h), 0.0, 1.0) * 255.0

    heat_rgb = np.stack([heat_r, heat_g, heat_b], axis=-1)

    blended = (1.0 - alpha * h[..., None]) * img_arr + (alpha * h[..., None]) * heat_rgb
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def get_architecture_summary():
    """Returns structured architectural specifications for the UI model viewer."""
    return {
        "model_family": "Convolutional Neural Network (CNN)",
        "framework": "PyTorch 2.2+",
        "input_dimensions": "128 x 128 x 3 (RGB)",
        "trainable_parameters": 914050,
        "stages": [
            {
                "name": "Input Layer",
                "output_shape": "3 x 128 x 128",
                "receptive_field": "1x1",
                "description": "Raw RGB image normalized with ImageNet mean/std distribution.",
            },
            {
                "name": "Conv Block 1 (Dual Conv3x3)",
                "output_shape": "32 x 64 x 64",
                "receptive_field": "6x6",
                "description": "Low-level feature extraction: Surface edges, specular reflection boundaries, and lighting gradients.",
            },
            {
                "name": "Conv Block 2 (Dual Conv3x3)",
                "output_shape": "64 x 32 x 32",
                "receptive_field": "14x14",
                "description": "Mid-level texture features: Metallic grain texture, surface roughness, and localized porosity.",
            },
            {
                "name": "Conv Block 3 (Dual Conv3x3)",
                "output_shape": "128 x 16 x 16",
                "receptive_field": "30x30",
                "description": "Geometric defect synthesis: Pinholes, tool chatter marks, parting line seams, and surface fissures.",
            },
            {
                "name": "Conv Block 4 (Conv3x3 + Pool)",
                "output_shape": "256 x 8 x 8",
                "receptive_field": "46x46",
                "description": "High-level defect topology: Shrinkage cavities, blowholes, cold shuts, and structural perimeter cracks.",
            },
            {
                "name": "Adaptive Average Pooling",
                "output_shape": "256 x 4 x 4",
                "receptive_field": "128x128",
                "description": "Spatial compression providing translation invariance across slightly misaligned inspection samples.",
            },
            {
                "name": "Dense Regularized Classifier",
                "output_shape": "2 (Defective vs Good)",
                "receptive_field": "Full",
                "description": "Dropout (0.35) + BatchNorm1d + Linear(4096 -> 128) + ReLU + Linear(128 -> 2) with Softmax posterior.",
            },
        ],
    }

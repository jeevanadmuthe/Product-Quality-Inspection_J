# Product Quality Inspection & Defect Tracking System

A local Flask + SQLite student project for visual inspection of metal casting products. It includes product/batch records, image uploads, optional learned image classification, human review, corrective actions, batch decisions, history, dashboard and evidence-grounded quality reports.

## Requirements
- Windows 10/11
- Python 3.10 or 3.11 recommended
- VS Code (optional)
- Internet for initial Python package installation and Kaggle dataset download

## Step 1 — Extract and open
Extract the ZIP. Open the `product_quality_inspection` folder in VS Code.

## Step 2 — Create a virtual environment (PowerShell)
Open a terminal **inside the project folder**:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, use `py -m venv .venv`. If PowerShell activation is blocked, run `.venv\Scripts\python.exe -m pip install -r requirements.txt`, then start the app with `.venv\Scripts\python.exe app.py`.

## Step 3 — Start the app
```powershell
python app.py
```
Open `http://127.0.0.1:5000` in your browser. The SQLite database is created automatically, with one demo product and batch.

## Step 4 — Demo logins
These are **local demo-only credentials**, not secure production authentication:
- Inspector: `inspector` / `inspect123`
- Supervisor: `supervisor` / `super123`
- Quality manager: `manager` / `manager123`

Role restrictions are enforced on the backend: only supervisor/manager can manage corrective actions; only manager can decide batch status and view the history log.

## Step 5 — Test the workflow without an ML model
1. Sign in as `inspector`.
2. Open **Batches** and note `B-104`, or create a new batch.
3. Open **Inspections**, select a batch, enter a sample reference such as `ITEM-001`, choose `Item`, upload a JPG/PNG and add an observation.
4. Until you train the model, the prediction is deliberately shown as unavailable. Save the inspection and perform a human review.
5. Sign in as `supervisor`; open the inspection, review it and assign a corrective action.
6. Open **Actions** and change the status as work progresses.
7. Sign in as `manager` to record a separate batch Hold/Released decision and view history.
8. Open **Report** to generate a report grounded in stored inspection and action IDs.

## Step 6 — Prepare the casting dataset
1. Download the [Kaggle casting product dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) and follow Kaggle's access and usage terms.
2. Inspect the downloaded folder names. Create the following structure in the project (copy images into it; do not put the same item or near-duplicate in both splits):

```text
data/
  casting/
    train/
      good/
      defective/
    test/
      good/
      defective/
```

3. Map the dataset's normal/OK class to `good` and its defective class to `defective`. Preserve a note of the source class names and split choices.
4. Use the dataset's existing train/test split if supplied and suitable. Do not mix test images into training. Avoid splitting near-duplicate views of the same physical item across train/test.
5. Aim for representative images in each class. The sample dataset is for training/evaluation, not a substitute for workflow history.

## Step 7 — Train and evaluate the Deep Learning CNN model
With your environment activated:

```powershell
# Train the multi-stage Deep Learning CNN (PyTorch) with live 0% to 100% terminal progress tracking:
python train_cnn.py --data data/casting --epochs 5

# Or directly via train_model.py:
python train_model.py
```

### Deep Learning CNN Conceptual Architecture (`CastingInspectionCNN`)
The CNN replaces handcrafted HOG descriptors with hierarchical representation learning directly from raw pixel tensors:
- **Stage 1 (Low-level edge filters)**: Dual `Conv2d(3 -> 32, 3x3)` + `BatchNorm` + `ReLU` + `MaxPool2d` extracting microscopic specular boundaries and surface contrast.
- **Stage 2 (Mid-level texture features)**: Dual `Conv2d(32 -> 64, 3x3)` + `BatchNorm` + `ReLU` + `MaxPool2d` capturing metallic grain roughness and localized circular porosity candidates.
- **Stage 3 & 4 (High-level defect synthesis)**: `Conv2d(64 -> 128 -> 256)` blocks synthesizing macro structural anomalies: blowholes, shrinkage cavities, cold laps, and perimeter fractures.
- **Global Regularization**: `AdaptiveAvgPool2d((4, 4))` providing spatial translation invariance plus `Dropout(0.35)`.
- **Classification Head**: `Linear(4096 -> 128)` + `BatchNorm1d` + `ReLU` + `Dropout` + `Linear(128 -> 2)` computing calibrated Softmax posterior probabilities for `[defective, good]`.
- **Explainability (Grad-CAM)**: Generates localized Class Activation Heatmaps highlighting the exact physical region of the metal casting that triggered a defect prediction.

The script strictly holds out the 715 test images, prints a confusion matrix and classification report, and saves:
- `models/casting_cnn.pt` (State dict weights and model architecture spec — achieving 99.86% held-out test accuracy)
- `models/casting_cnn.manifest.json` (Held-out accuracy, precision, recall, F1, and training history)

The web app prioritizes the CNN model automatically when present, and provides a dedicated interactive viewer at `/model-architecture` (`CNN Model` in the navbar).

## Step 8 — Storage Structure & Segregation
Uploaded optical images and AI-generated Grad-CAM heatmaps are strictly segregated into dedicated directories:
- `uploads/user_images/` — User-uploaded casting evidence captures.
- `uploads/aiml/` — Deep learning Grad-CAM class activation defect localization heatmaps.

## Step 9 — Run Automated Full-Stack Test Suite
Run the 38-check automated full-stack verification suite covering all routes, authentication, CNN inference, Grad-CAM generation, uploads, and database persistence:

```powershell
python test_system.py
```

## Step 10 — Suggested project demonstration
- Open the **CNN Model** dashboard tab to review the multi-stage architecture diagram, parameter counts, and confusion matrix comparing CNN vs classical SVM.
- Use the **Interactive Real-Time CNN Screening** widget to upload an unseen casting photo and view the instant classification and Grad-CAM defect localization heatmap.
- Register product and batch under **Batches**.
- Under **Inspections**, upload an unseen casting image to register an inspection. Notice both the raw optical capture and the generated Grad-CAM heatmap are saved side-by-side with sub-50ms inference latency.
- Confirm or correct the disposition as supervisor/manager and assign a corrective action.
- Demonstrate that if the CNN model file is ever unlinked, the system prompts for manual human inspection.

## Important limitations & production governance
- Machine learning screening suggestions are decision-support diagnostics; all final release dispositions require sign-off by qualified inspection personnel.
- Demo login credentials are hard-coded for local coursework; replace them with hashed passwords and a users table before real deployment.
- Grad-CAM heatmaps highlight high-gradient feature activation regions, not certified non-destructive metallurgical ultrasound scans.



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

## Step 7 — Train and evaluate the learned model
With the virtual environment activated:
```powershell
python train_model.py --data data/casting
```
The script extracts HOG image features, trains an SVM classifier, evaluates on the held-out test folder, prints a confusion matrix and classification report, then saves:
- `models/casting_svm.joblib`
- `models/casting_svm.manifest.json`

The app loads the model if that file exists. Restart Flask after training. The displayed score is the model's maximum class probability estimate; do not treat it as a calibrated probability or a safety guarantee. Record the actual metrics printed by your run—never invent accuracy.

## Step 8 — Suggested project demonstration
- Register product and batch.
- Upload an unseen held-out image and save the ML result.
- Confirm or correct the result with a reason.
- Create and update a corrective action.
- Create a follow-up inspection using a new sample reference.
- Show the dashboard, history and evidence-based report.
- Demonstrate that an inspector cannot access manager-only actions.
- Show the unavailable-model path and explain that manual review remains available.

## Important limitations / improvements for a production-grade version
- Demo login credentials are hard-coded for local coursework; replace them with hashed passwords and a users table before real deployment.
- Use a production WSGI server, CSRF protection, secure session-cookie settings, virus/image validation and private object storage before deployment.
- Uploads are local and not automatically deleted when records are archived.
- The report feature is a deterministic evidence-grounded report, not a hosted LLM. If the rubric requires generative AI specifically, add an authorised LLM API using server-side retrieved records and citations; do not send unauthorised data or let a model perform batch release/closure.
- A simple HOG+SVM baseline is a learned classifier, but compare it with transfer learning if results are weak. Report its real held-out metrics and limitations.
- A single image cannot establish the quality of an entire batch, the cause of a defect or supplier responsibility.

# Pediatric Pneumonia Detection from Chest X-Rays

Binary image classification (NORMAL vs PNEUMONIA) on pediatric chest X-rays, built in a single Google Colab notebook with TensorFlow/Keras. The project trains two models — a small CNN from scratch and a MobileNetV2 transfer-learning model — and compares them against a majority-class baseline.

The main contribution beyond a standard tutorial pipeline is a **patient-level data split**. The original Kaggle train/test split contains **264 patients who appear in both train and test**, which leaks information and inflates test accuracy. The notebook detects this, re-splits the data so no patient appears in more than one split, and shows the accuracy difference between the leaky and the fixed split (Section 15).

## Results (from the committed run; re-runs will vary slightly)

Evaluated on the patient-level test split (883 images: 215 NORMAL, 668 PNEUMONIA).

| Model | Test accuracy | ROC-AUC | NORMAL recall | PNEUMONIA recall |
|---|---|---|---|---|
| Majority-class baseline | 0.7565 | – | 0.00 | 1.00 |
| Simple CNN (from scratch) | 0.7361 | 0.9220 | 0.99 | 0.66 |
| MobileNetV2 transfer learning | **0.9332** | **0.9823** | 0.96 | 0.92 |

Leaky vs. fixed split, same Simple CNN architecture: **0.7933** test accuracy on the original Kaggle split (264 overlapping patients) vs **0.7361** on the patient-level split (0 overlapping patients).

## Dataset

- **Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle, originally from Kermany et al., *"Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"*, Cell 2018.
- **Size:** 5,856 JPEG images — 4,273 PNEUMONIA, 1,583 NORMAL.
- **Patient IDs** are parsed from filenames: `personNNN_...` for PNEUMONIA, `IM-NNNN-...` for NORMAL. This is what makes the leakage check and patient-level split possible.

> **Note:** the full dataset is committed to this repository under `chest_xray/`. Cloning is large (the dataset alone is well over 1 GB). Use the Colab path below if you want to avoid a local clone.

## File structure

```
Python_Summer_Final/
├── README.md
├── Pediatric_pneumonia_detection_DONE.ipynb   # the entire pipeline (18 sections)
├── Pneumonia_Project_Report.docx              # written project report
└── chest_xray/                                # original Kaggle split (committed)
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

Files created at runtime (not committed):

| Path (Colab) | What it is |
|---|---|
| `/content/chest_xray_clean/{train,val,test}/{NORMAL,PNEUMONIA}/` | Re-split copy of the dataset with no patient overlap (Section 7) |
| `/content/drive/MyDrive/simple_cnn_model.keras` | Saved Simple CNN (Section 18) |
| `/content/drive/MyDrive/transfer_model.keras` | Saved MobileNetV2 model (Section 18) |

## Setup

The notebook was written and run in Google Colab. Two ways to run it:

### Path A — Google Colab (recommended, matches the notebook as committed)

1. Upload `Pediatric_pneumonia_detection_DONE.ipynb` to Colab (**File → Upload notebook**), or open it directly from GitHub (**File → Open notebook → GitHub** and paste the repo URL).
2. Set the runtime to GPU: **Runtime → Change runtime type → Hardware accelerator: GPU (T4)**.
3. Nothing to install. Colab's default image already includes TensorFlow, NumPy, pandas, matplotlib, seaborn and scikit-learn.
4. Place the dataset on Google Drive (see [Data placement](#data-placement)).

### Path B — Local machine

Requires Python 3.10 or newer. No `requirements.txt` is provided; install the packages directly.

```bash
git clone https://github.com/aslikhan49141/Python_Summer_Final.git
cd Python_Summer_Final

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

Two edits are required because the notebook uses Colab-only APIs:

1. **Section 2 — remove the Drive mount and point `DATA_DIR` at your local copy.**
   Delete these two lines:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   and change
   ```python
   DATA_DIR = "/content/drive/MyDrive/chest_xray"
   ```
   to the path of the cloned folder, e.g. `DATA_DIR = "chest_xray"` (relative to the notebook) or an absolute path such as `DATA_DIR = "/home/you/Python_Summer_Final/chest_xray"`. On Windows use forward slashes or a raw string: `DATA_DIR = r"C:\Users\you\Python_Summer_Final\chest_xray"`.

2. **Section 17 — replace `files.upload()`.**
   `from google.colab import files` does not exist outside Colab. Either skip this cell, or replace the first three lines with a hard-coded path:
   ```python
   fname = "chest_xray/test/PNEUMONIA/person1952_bacteria_4883.jpeg"   # any image you want to test
   ```

Optional: `NEW_DIR = "/content/chest_xray_clean"` (Section 7) and the two save paths in Section 18 (`/content/drive/MyDrive/...`) are Colab-style absolute paths. They will work on Linux/macOS only if `/content` exists and is writable; change them to something like `"chest_xray_clean"` and `"simple_cnn_model.keras"` / `"transfer_model.keras"` to keep everything inside the project folder.

## Data placement

The notebook expects exactly this layout, with these folder names (case-sensitive):

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

This is the layout the Kaggle download ships with and the layout already committed in this repo.

- **Colab:** copy the `chest_xray/` folder to the root of your Google Drive so that it is at `MyDrive/chest_xray`. If you put it elsewhere, edit this line in Section 2:
  ```python
  DATA_DIR = "/content/drive/MyDrive/chest_xray"   # change to your actual path
  ```
- **Local:** the cloned `chest_xray/` folder already has the right layout; just set `DATA_DIR` to it as described in Path B.

The very next line, `assert os.path.exists(DATA_DIR)`, will stop the notebook immediately if the path is wrong, so this is the first thing to check.

## Running

**Colab:** **Runtime → Run all**. When Section 2 runs you will get a pop-up asking to authorise Google Drive access — accept it. When Section 17 runs, a file-chooser button appears; pick any chest X-ray JPEG to classify.

**Local:**
```bash
jupyter notebook Pediatric_pneumonia_detection_DONE.ipynb
```
then run the cells top to bottom (**Cell → Run All**), after making the edits in Path B.

**Runtime.** Three models are trained (Simple CNN, MobileNetV2, and a second Simple CNN on the leaky split for Section 15), 15 epochs each. In the committed run each training step took ~2–3 s (≈300 s per epoch), giving roughly **75 min for the CNN, 80 min for MobileNetV2, and ~3.5 h end-to-end** including the leaky-split comparison. On a Colab T4 GPU expect that to drop to roughly 20–30 s per epoch, i.e. **≈30 min total**. On a laptop CPU expect several hours; consider lowering `EPOCHS` in Section 1 for a quick smoke test.

Key configuration (Section 1): `SEED = 42`, `LEARNING_RATE = 0.001`, `EPOCHS = 15`, `IMG_SIZE = (224, 224)`, `BATCH = 32`.

## Expected output

Numbers below are from the committed run's saved cell outputs. Re-runs will vary slightly, especially model metrics; the data-side numbers (counts, overlap, split sizes) are deterministic given `SEED = 42`.

| § | Section | What you should see |
|---|---|---|
| 1 | Imports and configuration | `Config -> learning rate: 0.001 \| epochs: 15` |
| 2 | Load dataset | Drive mount message, then `['val', 'test', 'train']` |
| 3 | Manifest + leakage check | First 5 manifest rows; `PNEUMONIA 4273`, `NORMAL 1583`; **`Patients found in both original train and test: 264`** |
| 4 | Corrupt image scan | `Corrupt files found: 0` |
| 5 | Class distribution | Bar chart, two bars (PNEUMONIA ≫ NORMAL) |
| 6 | Patient-level split | Per-split counts: **train 1141 N / 2937 P, val 227 N / 668 P, test 215 N / 668 P**; `No patient overlap between splits.` |
| 7 | Copy to new folders | `Done.` (this is the slowest non-training cell — copies 5,856 files from Drive) |
| 8 | Preprocessing | `Found 4078 files` / `Found 895 files` / `Found 883 files`, each `belonging to 2 classes` |
| 9 | Majority-class baseline | `Baseline accuracy: 0.7565118912797282`; classification report with NORMAL recall 0.00, PNEUMONIA recall 1.00 |
| 10 | Class weights | `Class weights: {0: 1.787..., 1: 0.694...}` |
| 11 | Simple CNN | Model summary (27,809 params); 15 epochs of progress bars; final epoch ≈ `accuracy: 0.84 – val_accuracy: 0.74`; `CNN training time (s): 4536.2` |
| 12 | MobileNetV2 transfer learning | ImageNet weights download on first run; 15 epochs; final epoch ≈ `accuracy: 0.94 – val_accuracy: 0.94`; `Transfer learning training time (s): 4755.3` |
| 13 | Test-set evaluation | Two blocks. **Simple CNN:** `Accuracy: 0.7361`, `ROC-AUC: 0.9220`. **Transfer Learning:** `Accuracy: 0.9332`, `ROC-AUC: 0.9823`. Each followed by a classification report and a confusion-matrix heatmap |
| 14 | ROC curve comparison | One plot with three lines: Simple CNN, Transfer Learning, dashed Random diagonal |
| 15 | Leaky vs patient-level split | `Found 5216 / 16 / 624 files` (original Kaggle split), then a 2-row table: `Original (leaky) 0.793269 264` and `Patient-level (fixed) 0.736127 0` |
| 16 | Grad-CAM | Two test images with `P(pneumonia) = 0.003` and `0.006` (both NORMAL), each with a jet-colour heatmap overlay. A matplotlib `get_cmap` deprecation warning and a Keras input-structure `UserWarning` are printed — both are harmless |
| 17 | Upload and predict | File-chooser widget; after upload, the image is shown with title `Predicted: PNEUMONIA (0.xxx)` or `Predicted: NORMAL (0.xxx)` |
| 18 | Save models | `Saved.` — two `.keras` files appear in `MyDrive` |

Interpretation notes for a reader:

- The Simple CNN scores *below* the majority baseline on raw accuracy (0.736 vs 0.757) but has ROC-AUC 0.92. With class weights it heavily favours NORMAL (recall 0.99) at the expense of PNEUMONIA recall (0.66); the 0.5 threshold is simply not the right operating point for it.
- The MobileNetV2 model is the one that actually beats the baseline on every metric and is the model used for Grad-CAM (Section 16) and single-image prediction (Section 17).
- Section 15 shows the leaky split *over-reports* accuracy by about 6 points for the same architecture — the motivation for the whole patient-level pipeline.

## Troubleshooting

**`AssertionError: Path not found: /content/drive/MyDrive/chest_xray`**
The dataset is not where `DATA_DIR` points. In Colab, open the Files pane on the left, expand `drive/MyDrive`, and confirm `chest_xray/` is directly under it with `train/`, `val/`, `test/` inside. Otherwise edit `DATA_DIR` in Section 2. Locally, make sure you removed the Drive mount and set `DATA_DIR` to your clone.

**`FileNotFoundError` inside `build_manifest` (Section 3)**
One of the six class folders is missing or misnamed. Folder names must be exactly `NORMAL` and `PNEUMONIA` (upper case) inside each of `train`, `val`, `test`.

**Colab session disconnects or times out during Section 7 (copying images)**
Copying ~5,900 files from Drive to the Colab VM is slow and free-tier sessions can idle out. The copy loop skips files that already exist, so simply re-run the Section 7 cell — it resumes where it stopped. If the whole runtime was recycled, `/content/chest_xray_clean` is gone and you must re-run from Section 2. Keeping the browser tab active helps.

**`ModuleNotFoundError: No module named 'google.colab'`**
You are running locally. Apply the two edits in Path B (remove `drive.mount` in Section 2 and `files.upload()` in Section 17).

**Windows path problems (`OSError`, `unicodeescape`, or `Path not found`)**
Backslashes in a normal Python string are escape characters. Use forward slashes (`"C:/Users/you/chest_xray"`) or a raw string (`r"C:\Users\you\chest_xray"`). Also change `NEW_DIR` and the Section 18 save paths, which are Linux-style `/content/...` paths.

**Out of memory / very slow on CPU**
Reduce `BATCH` from 32 to 16 or 8 and/or `EPOCHS` from 15 to a smaller number in Section 1. `IMG_SIZE` can be lowered too, but MobileNetV2 with ImageNet weights expects 224×224 input in this notebook, so leave it unless you also change the model. In Colab, confirm the GPU is actually active with `!nvidia-smi` in a new cell.

**`Found 0 files belonging to 2 classes` in Section 8 or 15**
The copy in Section 7 did not complete (see the timeout item above), or `DATA_DIR` / `NEW_DIR` point to a folder without the `{train,val,test}/{NORMAL,PNEUMONIA}` layout.

**Grad-CAM layer error (`No such layer: out_relu`)**
Section 16 uses the MobileNetV2 layer name `out_relu`. If you swapped the base model, change `"out_relu"` to the name of your model's last convolutional/activation layer (`tl_model.summary()` lists them).

## Report

The written project report is `Pneumonia_Project_Report.docx` in the repository root.

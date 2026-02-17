# =========================================
# Environment Setup and Dataset Initialization
# =========================================
import torch
import os
import glob
import shutil
import pandas as pd

# GPU configuration (for Deep Learning training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)

# =========================================
# Google Drive Setup (Google Colab Environment)
# =========================================
from google.colab import drive
drive.mount("/content/drive")

# Project root directory (update if needed)
PROJECT_ROOT = "/content/drive/MyDrive/FRACTURAS"
LOCAL_ROOT = "/content/FRACTURAS"

# =========================================
# Path Validation (Best Practice)
# =========================================
def assert_exists(path, kind="file"):
    if kind == "file" and not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if kind == "dir" and not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")

# Drive paths
CSV_DRIVE = os.path.join(PROJECT_ROOT, "dataset.csv")
FRACT_DRIVE = os.path.join(PROJECT_ROOT, "Fractured")
NONF_DRIVE = os.path.join(PROJECT_ROOT, "Non_fractured")

assert_exists(CSV_DRIVE, "file")
assert_exists(FRACT_DRIVE, "dir")
assert_exists(NONF_DRIVE, "dir")

# Copy dataset locally for faster training
if not os.path.exists(LOCAL_ROOT):
    shutil.copytree(PROJECT_ROOT, LOCAL_ROOT)

print("Dataset available at:", LOCAL_ROOT)


# =========================
# Environment & GPU Check
# =========================
import torch

print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Google Drive Setup (Colab)
# =========================
# This project was developed in Google Colab.
# Dataset is stored in Google Drive due to large medical image size.

from google.colab import drive
drive.mount("/content/drive")

import os
import glob

# ⚠️ NOTE:
# Update this path according to your Google Drive structure
PROJECT_ROOT = "/content/drive/MyDrive/FRACTURAS_PROJECT"

CSV_PATH = os.path.join(PROJECT_ROOT, "dataset.csv")
FRACTURED_DIR = os.path.join(PROJECT_ROOT, "Fractured")
NON_FRACTURED_DIR = os.path.join(PROJECT_ROOT, "Non_fractured")

# =========================
# Path Validation (Best Practice)
# =========================
def assert_exists(path, kind="file"):
    if kind == "file" and not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if kind == "dir" and not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")

assert_exists(CSV_PATH, "file")
assert_exists(FRACTURED_DIR, "dir")
assert_exists(NON_FRACTURED_DIR, "dir")

print("Dataset paths validated successfully.")
print("Fractured images:", len(glob.glob(os.path.join(FRACTURED_DIR, "*"))))
print("Non-fractured images:", len(glob.glob(os.path.join(NON_FRACTURED_DIR, "*"))))


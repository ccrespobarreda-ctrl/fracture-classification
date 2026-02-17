# src/data_setup.py
import os
import glob
import shutil
import pandas as pd
import torch

# =========================
# Device configuration
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Project paths
# =========================
PROJECT_ROOT = "/content/drive/MyDrive/FRACTURAS"
LOCAL_ROOT = "/content/FRACTURAS"

CSV_NAME = "dataset.csv"
FRACTURED_FOLDER = "Fractured"
NON_FRACTURED_FOLDER = "Non_fractured"

def assert_exists(path: str, kind: str = "file") -> None:
    if kind == "file" and not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if kind == "dir" and not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")

def prepare_local_data():
    csv_drive = os.path.join(PROJECT_ROOT, CSV_NAME)
    fract_drive = os.path.join(PROJECT_ROOT, FRACTURED_FOLDER)
    nonf_drive = os.path.join(PROJECT_ROOT, NON_FRACTURED_FOLDER)

    assert_exists(csv_drive, "file")
    assert_exists(fract_drive, "dir")
    assert_exists(nonf_drive, "dir")

    if not os.path.exists(LOCAL_ROOT):
        shutil.copytree(PROJECT_ROOT, LOCAL_ROOT)

    csv_path = os.path.join(LOCAL_ROOT, CSV_NAME)
    fract_dir = os.path.join(LOCAL_ROOT, FRACTURED_FOLDER)
    nonf_dir = os.path.join(LOCAL_ROOT, NON_FRACTURED_FOLDER)

    df = pd.read_csv(csv_path)
    return df, csv_path, fract_dir, nonf_dir, device

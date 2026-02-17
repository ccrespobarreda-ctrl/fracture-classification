# src/evaluate.py
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from dataset import create_dataloaders, FractureDataset, val_tfms, LABEL_COL
from evaluation import (
    get_probs_and_labels,
    compute_auc_metrics,
    metrics_for_threshold,
    compute_threshold_table,
    report_recall_at_specs,
)
from model import build_model


def make_splits(df, label_col=LABEL_COL, random_state=42):
    # Train vs temp (val+test)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=df[label_col]
    )
    # Val vs test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=random_state, stratify=temp_df[label_col]
    )
    return train_df, val_df, test_df


def evaluate_on_test(model, test_loader, device, best_thr: float):
    probs_test, y_test = get_probs_and_labels(model, test_loader, device)

    # Threshold metrics
    m = metrics_for_threshold(probs_test, y_test, best_thr)

    # AUC metrics
    aucs = compute_auc_metrics(y_test, probs_test)

    return m, aucs


if __name__ == "__main__":
    # NOTE: aquí deberías cargar tu df preparado (con img_path) desde tu pipeline
    raise NotImplementedError("Load your df with img_path here, then create splits and loaders.")

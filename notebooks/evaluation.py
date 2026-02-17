# src/evaluation.py
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

@torch.no_grad()
def get_probs_and_labels(model, loader, device):
    """
    Collect predicted probabilities and true labels from a DataLoader.
    Returns:
        probs: np.ndarray, shape (N,)
        y_true: np.ndarray, shape (N,)
    """
    model.eval()
    probs_all, y_all = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits)

        probs_all.append(probs.detach().cpu())
        y_all.append(y.detach().cpu())

    probs_all = torch.cat(probs_all).numpy()
    y_all = torch.cat(y_all).numpy().astype(int)
    return probs_all, y_all


def compute_auc_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """
    Compute ROC-AUC and PR-AUC (Average Precision).
    Safe against edge cases where only one class is present.
    """
    unique = np.unique(y_true)
    roc_auc = roc_auc_score(y_true, probs) if len(unique) == 2 else float("nan")
    pr_auc = average_precision_score(y_true, probs) if len(unique) == 2 else float("nan")
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def metrics_for_threshold(probs: np.ndarray, y_true: np.ndarray, thr: float) -> dict:
    """
    Compute threshold-based metrics from probabilities.
    Returns a dict including confusion matrix counts.
    """
    y_pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    specificity = tn / max(1, (tn + fp))
    f1 = (2 * precision * recall) / max(1e-12, (precision + recall))

    return {
        "thr": float(thr),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def find_best_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    min_specificity: float = 0.80,
    thr_min: float = 0.05,
    thr_max: float = 0.95,
    step: float = 0.01,
    objective: str = "recall",
) -> dict:
    """
    Search threshold maximizing `objective` subject to specificity >= min_specificity.
    objective: 'recall' or 'f1' (extendable).
    """
    thresholds = np.arange(thr_min, thr_max + 1e-9, step)
    results = [metrics_for_threshold(probs, y_true, float(t)) for t in thresholds]

    candidates = [r for r in results if r["specificity"] >= min_specificity]
    if not candidates:
        # fallback: best objective without constraint
        return max(results, key=lambda r: r.get(objective, 0.0))

    return max(candidates, key=lambda r: r.get(objective, 0.0))

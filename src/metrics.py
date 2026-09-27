from __future__ import annotations

import numpy as np


def macro_metrics(y, probability, groups, truth_count, threshold, group_mask=None):
    if group_mask is None:
        group_mask = np.ones(len(truth_count), dtype=bool)
    selected = probability >= threshold
    tp = np.bincount(groups, weights=(selected & (y == 1)), minlength=len(truth_count))
    fp = np.bincount(groups, weights=(selected & (y == 0)), minlength=len(truth_count))
    fn = truth_count - tp
    predicted = tp + fp
    positive = truth_count > 0
    precision_e = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted > 0)
    recall_e = np.divide(tp, truth_count, out=np.zeros_like(tp, dtype=float), where=truth_count > 0)
    denominator = 0.25 * precision_e + recall_e
    score = np.divide(1.25 * precision_e * recall_e, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    score[~positive] = (predicted[~positive] == 0).astype(float)
    use = group_mask
    total_tp, total_fp, total_fn = tp[use].sum(), fp[use].sum(), fn[use].sum()
    return {
        "macro_f05": float(score[use].mean()),
        "precision": float(total_tp / (total_tp + total_fp)) if total_tp + total_fp else 0.0,
        "recall": float(total_tp / (total_tp + total_fn)) if total_tp + total_fn else 0.0,
        "singleton_accuracy": float((predicted[use & ~positive] == 0).mean()) if np.any(use & ~positive) else None,
        "false_positives": int(total_fp), "false_negatives": int(total_fn),
    }


def optimize_threshold(y, probability, groups, truth_count, group_mask=None):
    rows = []
    thresholds = np.unique(np.r_[np.linspace(0.05, 0.95, 91), np.linspace(0.955, 0.995, 9)])
    for threshold in thresholds:
        result = macro_metrics(y, probability, groups, truth_count, float(threshold), group_mask)
        result["threshold"] = float(threshold)
        rows.append(result)
    return max(rows, key=lambda row: row["macro_f05"]), rows

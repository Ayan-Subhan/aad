"""Metric computation shared by every phase.

One definition of accuracy/precision/recall/F1 for baselines, adversarial
evaluation, the discriminator and the EIDS, so numbers stay comparable across
the whole project.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

# Multiclass codes -> the paper's Table I naming.
TYPE_NAMES = {
    0: "Benign",
    1: "Type1 DoS-Hulk",
    2: "Type2 DoS-SlowHTTPTest",
    3: "Type3 BruteForce-Web",
    4: "Type4 BruteForce-XSS",
    5: "Type5 SQL-Injection",
}


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    """Positive class = attack (1). Precision/recall/F1 are reported for that
    class, which is the convention in the IDS literature."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # An OR-ensemble buys recall with false positives, so FPR travels with
        # every accuracy figure in this project.
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "support_benign": int(tn + fp),
        "support_attack": int(fn + tp),
    }
    if y_prob is not None:
        out["loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    return out


def per_type_recall(y_true_multi: np.ndarray, y_pred: np.ndarray) -> dict:
    """Detection rate per attack type. Binary accuracy hides the fact that the
    three web-attack classes are a few hundred rows out of 224k."""
    out = {}
    for code in np.unique(y_true_multi):
        code = int(code)
        mask = y_true_multi == code
        if code == 0:
            out[TYPE_NAMES[code]] = {
                "support": int(mask.sum()),
                "correct_rate": float((y_pred[mask] == 0).mean()),
            }
        else:
            out[TYPE_NAMES.get(code, str(code))] = {
                "support": int(mask.sum()),
                "correct_rate": float((y_pred[mask] == 1).mean()),
            }
    return out


def attack_success_rate(y_pred_on_adversarial: np.ndarray) -> float:
    """Fraction of adversarial attack-samples that evaded detection.

    Every adversarial example in this project is built from a true-attack row,
    so a prediction of 0 (benign) means the attack succeeded. Phase 3 onwards.
    """
    if len(y_pred_on_adversarial) == 0:
        return 0.0
    return float((y_pred_on_adversarial == 0).mean())


def format_row(name: str, m: dict) -> str:
    return (
        f"{name:<10s} {m['accuracy']*100:>8.4f} {m['precision']*100:>10.4f} "
        f"{m['recall']*100:>8.4f} {m['f1']*100:>8.4f} {m.get('loss', float('nan')):>9.5f} "
        f"{m['fpr']*100:>8.4f}"
    )


HEADER = (
    f"{'model':<10s} {'accuracy':>8s} {'precision':>10s} {'recall':>8s} "
    f"{'f1':>8s} {'loss':>9s} {'FPR%':>8s}"
)

"""Scores for Fit predictions (High / Moderate / Low) against gold labels.

Same metrics as Giap's notebooks/real_grant_eval.ipynb, so numbers are comparable:
macro precision / recall / F1, accuracy, High recall and precision, a confusion matrix,
and a bootstrap 95% range for macro F1 (the dev set is small, so the range matters).
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

LABELS = ["Low", "Moderate", "High"]
SEED = 0


def f1_interval(y_true: list[str], y_pred: list[str], n: int = 1000) -> tuple[float, float]:
    y, p = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), (n, len(y)))
    scores = [f1_score(y[i], p[i], labels=LABELS, average="macro", zero_division=0) for i in idx]
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def fit_scores(y_true: list[str], y_pred: list[str]) -> dict:
    """y_true and y_pred are lists of 'Low' / 'Moderate' / 'High' of the same length."""
    if not y_true:
        return {"n": 0}
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    hp, hr, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=["High"], zero_division=0)
    lo, hi = f1_interval(y_true, y_pred)
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "precision": round(float(pr), 3),
        "recall": round(float(rc), 3),
        "f1": round(float(f1), 3),
        "f1_95ci": f"{lo:.3f}-{hi:.3f}",
        "high_recall": round(float(hr[0]), 3),
        "high_precision": round(float(hp[0]), 3),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def format_scores(name: str, s: dict) -> str:
    if not s.get("n"):
        return f"{name}: no labeled grants"
    return (
        f"{name:44s} n={s['n']:<4} acc={s['accuracy']:.3f}  F1={s['f1']:.3f} [{s['f1_95ci']}]  "
        f"High R={s['high_recall']:.3f} P={s['high_precision']:.3f}\n"
        f"{'':44s} confusion (rows = true Low/Moderate/High): {s['confusion']}"
    )

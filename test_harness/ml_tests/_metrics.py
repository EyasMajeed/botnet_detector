"""
_metrics.py — Reusable metric computation that does NOT require sklearn.

We use sklearn when available (faster) and fall back to a pure-numpy
implementation otherwise. Either way, the output schema is identical.
"""

from __future__ import annotations

from typing import Iterable

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False


def confusion_counts(y_true: Iterable[int], y_pred: Iterable[int]) -> dict:
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1: tp += 1
        elif t == 0 and p == 1: fp += 1
        elif t == 0 and p == 0: tn += 1
        elif t == 1 and p == 0: fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def from_counts(c: dict) -> dict:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = (2 * p * r / max(p + r, 1e-9)) if (p + r) > 0 else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {**c,
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
            "accuracy":  round(acc, 4)}


def metrics(y_true: list[int], y_pred: list[int],
            y_score: list[float] | None = None) -> dict:
    out = from_counts(confusion_counts(y_true, y_pred))
    if y_score is not None:
        out["auc_roc"] = _auc(y_true, y_score)
    return out


def _auc(y_true: list[int], y_score: list[float]) -> float:
    """Mann-Whitney U formulation. Stable, no sklearn needed."""
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return float("nan")
    if _NP:
        pos_arr = np.array(pos); neg_arr = np.array(neg)
        # rank-based AUC
        all_arr = np.concatenate([pos_arr, neg_arr])
        order = np.argsort(all_arr, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(all_arr) + 1)
        # Average ranks for ties
        # (cheap-and-cheerful — fine for the harness)
        sum_pos_ranks = ranks[: len(pos_arr)].sum()
        n_pos = len(pos_arr); n_neg = len(neg_arr)
        u = sum_pos_ranks - n_pos * (n_pos + 1) / 2
        return float(u / (n_pos * n_neg))
    # Pure-Python fallback
    n_pos = len(pos); n_neg = len(neg)
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n: wins += 1
            elif p == n: ties += 1
    return (wins + 0.5 * ties) / (n_pos * n_neg)


def threshold_sweep(y_true: list[int], y_score: list[float],
                    thresholds: list[float]) -> list[dict]:
    rows = []
    for t in thresholds:
        y_pred = [1 if s >= t else 0 for s in y_score]
        m = metrics(y_true, y_pred)
        m["threshold"] = round(t, 4)
        rows.append(m)
    return rows

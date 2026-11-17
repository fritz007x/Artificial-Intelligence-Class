"""Offline evaluation helpers shared by 05_ml/train.py, the notebook and the API's
learning scenario."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reasoning.ml import FEATURES, alert_features, to_matrix


def rule_score(row: dict) -> float:
    """Symbolic baseline mirroring the KB's true-positive rules plus severity tie-break."""
    tp = (row["ioc_match"] and row["alert_class"] == "execution") or \
         (row["success_after_fail"] and row["alert_class"] == "credential_attack")
    return (0.8 if tp else 0.1) + 0.02 * row["severity"] + 0.01 * row["criticality"]


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    idx = np.argsort(-scores)[:k]
    return float(y[idx].mean())


def split(rows: list[dict], seed: int, fracs=(0.6, 0.2, 0.2)) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified train/val/test split."""
    rng = np.random.default_rng(seed)
    out: tuple[list, list, list] = ([], [], [])
    for label in (0, 1):
        grp = [r for r in rows if r["label"] == label]
        order = rng.permutation(len(grp))
        a, b = int(fracs[0] * len(grp)), int((fracs[0] + fracs[1]) * len(grp))
        for i, j in enumerate(order):
            out[0 if i < a else 1 if i < b else 2].append(grp[j])
    return out


def xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return to_matrix([alert_features(r) for r in rows]), np.array([r["label"] for r in rows])


def load_learning_report(ml_dir: Path) -> dict:
    p = Path(ml_dir) / "artifacts" / "metrics.json"
    if not p.exists():
        return {"error": "run `python 05_ml/train.py` first"}
    m = json.loads(p.read_text(encoding="utf-8"))
    return {"learning_case": m["learning_case"], "selected_model": m["selected_model"],
            "test_metrics": m["test"][m["selected_model"]], "fairness": m["fairness"], "rl": m["rl"],
            "features": FEATURES}

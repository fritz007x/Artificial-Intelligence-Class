"""Learned alert-triage models implemented in numpy so the serving image stays small
and inference is deterministic: logistic regression (linear baseline) and a one-hidden-
layer MLP (neural learner). Shared feature engineering guarantees train/serve parity.
Protected attributes (user region, department) are *never* features; they are used only
for the fairness audit."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FEATURES = [
    "severity", "criticality", "ioc_match", "ioc_confidence", "failed_logins", "success_after_fail",
    "impossible_travel", "edr_source", "off_hours", "privileged_user", "asset_alert_count",
    "cls_credential", "cls_execution", "cls_recon", "cls_lateral", "cls_exfil",
]
PROTECTED = {"region", "department"}
CLASS_FEATURE = {"credential_attack": "cls_credential", "execution": "cls_execution",
                 "reconnaissance": "cls_recon", "lateral_movement": "cls_lateral",
                 "data_exfiltration": "cls_exfil"}


def alert_features(ctx: dict[str, Any]) -> dict[str, float]:
    """ctx keys: severity(1-4), criticality(1-5), ioc_match, ioc_confidence, failed_logins,
    success_after_fail, impossible_travel, edr_source, off_hours, privileged_user,
    asset_alert_count, alert_class (top-level kill-chain class)."""
    f = {k: 0.0 for k in FEATURES}
    f["severity"] = float(ctx.get("severity", 2)) / 4.0
    f["criticality"] = float(ctx.get("criticality", 1)) / 5.0
    f["ioc_match"] = float(bool(ctx.get("ioc_match")))
    f["ioc_confidence"] = float(ctx.get("ioc_confidence", 0.0))
    f["failed_logins"] = min(float(ctx.get("failed_logins", 0)), 20.0) / 20.0
    f["success_after_fail"] = float(bool(ctx.get("success_after_fail")))
    f["impossible_travel"] = float(bool(ctx.get("impossible_travel")))
    f["edr_source"] = float(bool(ctx.get("edr_source")))
    f["off_hours"] = float(bool(ctx.get("off_hours")))
    f["privileged_user"] = float(bool(ctx.get("privileged_user")))
    f["asset_alert_count"] = min(float(ctx.get("asset_alert_count", 1)), 10.0) / 10.0
    cls = CLASS_FEATURE.get(ctx.get("alert_class", ""), None)
    if cls:
        f[cls] = 1.0
    return f


def to_matrix(rows: list[dict[str, float]]) -> np.ndarray:
    return np.array([[r[k] for k in FEATURES] for r in rows], dtype=float)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X):
        s = X.std(axis=0)
        return cls(X.mean(axis=0), np.where(s < 1e-9, 1.0, s))

    def __call__(self, X):
        return (X - self.mean) / self.std


class LogisticModel:
    kind = "logistic_regression"

    def __init__(self, l2: float = 1e-3, lr: float = 0.3, epochs: int = 600):
        self.l2, self.lr, self.epochs = l2, lr, epochs
        self.w: np.ndarray | None = None
        self.b = 0.0
        self.scaler: Standardizer | None = None

    def fit(self, X, y, seed: int = 0):
        self.scaler = Standardizer.fit(X)
        Z = self.scaler(X)
        self.w = np.zeros(Z.shape[1])
        for _ in range(self.epochs):
            p = _sigmoid(Z @ self.w + self.b)
            g = p - y
            self.w -= self.lr * (Z.T @ g / len(y) + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    def predict_proba(self, X):
        return _sigmoid(self.scaler(X) @ self.w + self.b)

    def explain(self, x: np.ndarray, top: int = 4) -> list[tuple[str, float]]:
        contrib = self.scaler(x[None, :])[0] * self.w
        idx = np.argsort(-np.abs(contrib))[:top]
        return [(FEATURES[i], round(float(contrib[i]), 3)) for i in idx]

    def to_dict(self):
        return {"kind": self.kind, "w": self.w.tolist(), "b": self.b,
                "mean": self.scaler.mean.tolist(), "std": self.scaler.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.w, m.b = np.array(d["w"]), float(d["b"])
        m.scaler = Standardizer(np.array(d["mean"]), np.array(d["std"]))
        return m


class MLPModel:
    """1 hidden layer (ReLU) + sigmoid output, trained with Adam and early stopping."""
    kind = "mlp"

    def __init__(self, hidden: int = 16, lr: float = 0.01, epochs: int = 300, l2: float = 1e-4, batch: int = 64,
                 patience: int = 25):
        self.hidden, self.lr, self.epochs, self.l2, self.batch, self.patience = hidden, lr, epochs, l2, batch, patience
        self.params: dict[str, np.ndarray] = {}
        self.scaler: Standardizer | None = None
        self.history: list[float] = []

    def _forward(self, Z):
        h = np.maximum(0, Z @ self.params["W1"] + self.params["b1"])
        return h, _sigmoid(h @ self.params["W2"] + self.params["b2"]).ravel()

    def fit(self, X, y, seed: int = 0, X_val=None, y_val=None):
        rng = np.random.default_rng(seed)
        self.scaler = Standardizer.fit(X)
        Z = self.scaler(X)
        d = Z.shape[1]
        self.params = {"W1": rng.normal(0, np.sqrt(2 / d), (d, self.hidden)), "b1": np.zeros(self.hidden),
                       "W2": rng.normal(0, np.sqrt(1 / self.hidden), (self.hidden, 1)), "b2": np.zeros(1)}
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        t, best, best_params, wait = 0, np.inf, None, 0
        for _ in range(self.epochs):
            order = rng.permutation(len(y))
            for i in range(0, len(y), self.batch):
                idx = order[i:i + self.batch]
                zb, yb = Z[idx], y[idx]
                h, p = self._forward(zb)
                g_out = (p - yb)[:, None] / len(idx)
                grads = {"W2": h.T @ g_out + self.l2 * self.params["W2"], "b2": g_out.sum(0)}
                g_h = (g_out @ self.params["W2"].T) * (h > 0)
                grads["W1"] = zb.T @ g_h + self.l2 * self.params["W1"]
                grads["b1"] = g_h.sum(0)
                t += 1
                for k in self.params:
                    m[k] = 0.9 * m[k] + 0.1 * grads[k]
                    v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                    self.params[k] -= self.lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            if X_val is not None:
                loss = log_loss(y_val, self.predict_proba(X_val))
                self.history.append(loss)
                if loss < best - 1e-5:
                    best, best_params, wait = loss, {k: x.copy() for k, x in self.params.items()}, 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if best_params:
            self.params = best_params
        return self

    def predict_proba(self, X):
        return self._forward(self.scaler(X))[1]

    def to_dict(self):
        return {"kind": self.kind, **{k: v.tolist() for k, v in self.params.items()},
                "mean": self.scaler.mean.tolist(), "std": self.scaler.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.params = {k: np.array(d[k]) for k in ("W1", "b1", "W2", "b2")}
        m.scaler = Standardizer(np.array(d["mean"]), np.array(d["std"]))
        return m


# ------------------------------------------------------------------ metrics
def log_loss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_auc(y, p):
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    ranks = np.empty(len(p))
    i = 0
    while i < len(p):                       # average ranks over ties
        j = i
        while j + 1 < len(p) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ece(y, p, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def classification_metrics(y, p, threshold: float = 0.5) -> dict[str, float]:
    yhat = (p >= threshold).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "roc_auc": roc_auc(y, p), "log_loss": log_loss(y, p),
            "brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p)}


def fpr_by_group(y, p, groups, threshold: float = 0.5) -> dict[str, float]:
    out = {}
    for g in sorted(set(groups)):
        m = np.array([x == g for x in groups])
        neg = m & (y == 0)
        out[g] = float(((p >= threshold) & neg).sum() / neg.sum()) if neg.sum() else 0.0
    return out


# ------------------------------------------------------------------ artifact I/O
def sha256_of(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def save_artifact(model, path: Path, metadata: dict) -> dict:
    body = model.to_dict()
    art = {"metadata": {**metadata, "features": FEATURES, "artifact_sha256": sha256_of(body)}, "model": body}
    Path(path).write_text(json.dumps(art, indent=1))
    return art["metadata"]


class LoadedModel:
    def __init__(self, model, metadata: dict):
        self.model, self.metadata = model, metadata

    def predict_one(self, feats: dict[str, float]) -> float:
        return float(self.model.predict_proba(to_matrix([feats]))[0])


def load_artifact(path: Path) -> LoadedModel:
    """Load + integrity-check a model artifact. Raises on tampering or feature drift."""
    art = json.loads(Path(path).read_text())
    meta, body = art["metadata"], art["model"]
    if sha256_of(body) != meta.get("artifact_sha256"):
        raise ValueError("model artifact integrity check failed")
    if meta.get("features") != FEATURES:
        raise ValueError("model feature list does not match serving features")
    cls = {"logistic_regression": LogisticModel, "mlp": MLPModel}[body["kind"]]
    return LoadedModel(cls.from_dict(body), meta)

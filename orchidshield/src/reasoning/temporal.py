"""Temporal reasoning: an HMM over kill-chain stages (forward filtering, Viterbi
decoding, sampling) and a 1-D Kalman filter tracking per-host alert rate."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class HMM:
    def __init__(self, states, observations, start, transition, emission, stage_severity=None):
        self.states = list(states)
        self.obs = list(observations)
        self.pi = np.asarray(start, float)
        self.A = np.asarray(transition, float)
        self.B = np.asarray(emission, float)
        self.stage_severity = stage_severity or {}
        for name, m in (("transition", self.A), ("emission", self.B)):
            if not np.allclose(m.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError(f"{name} rows must sum to 1")
        if not np.isclose(self.pi.sum(), 1.0):
            raise ValueError("start distribution must sum to 1")

    @classmethod
    def from_json(cls, path: Path) -> "HMM":
        s = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(s["states"], s["observations"], s["start"], s["transition"], s["emission"], s.get("stage_severity"))

    def _idx(self, seq: list[str]) -> list[int]:
        return [self.obs.index(o) if o in self.obs else 0 for o in seq]

    def filter(self, seq: list[str]) -> np.ndarray:
        """Forward algorithm with normalisation. Returns P(X_t | e_1:t) for the last t."""
        if not seq:
            return self.pi.copy()
        o = self._idx(seq)
        alpha = self.pi * self.B[:, o[0]]
        alpha /= alpha.sum()
        for t in o[1:]:
            alpha = (alpha @ self.A) * self.B[:, t]
            alpha /= alpha.sum()
        return alpha

    def predict(self, belief: np.ndarray, steps: int = 1) -> np.ndarray:
        for _ in range(steps):
            belief = belief @ self.A
        return belief

    def viterbi(self, seq: list[str]) -> list[str]:
        o = self._idx(seq)
        if not o:
            return []
        logA, logB = np.log(self.A + 1e-12), np.log(self.B + 1e-12)
        delta = np.log(self.pi + 1e-12) + logB[:, o[0]]
        back = []
        for t in o[1:]:
            scores = delta[:, None] + logA
            back.append(scores.argmax(axis=0))
            delta = scores.max(axis=0) + logB[:, t]
        path = [int(delta.argmax())]
        for bp in reversed(back):
            path.append(int(bp[path[-1]]))
        return [self.states[i] for i in reversed(path)]

    def sample(self, n: int, rng: np.random.Generator) -> tuple[list[str], list[str]]:
        x = rng.choice(len(self.states), p=self.pi)
        xs, os_ = [], []
        for _ in range(n):
            xs.append(self.states[x])
            os_.append(self.obs[rng.choice(len(self.obs), p=self.B[x])])
            x = rng.choice(len(self.states), p=self.A[x])
        return xs, os_

    def expected_severity(self, belief: np.ndarray) -> float:
        return float(sum(belief[i] * self.stage_severity.get(s, 0.0) for i, s in enumerate(self.states)))


@dataclass
class KalmanResult:
    estimates: list[float]
    innovations_z: list[float]
    max_z: float


def kalman_rate(counts: list[float], q: float = 0.05, r: float = 1.0, x0: float = 0.2, p0: float = 1.0) -> KalmanResult:
    """Local-level Kalman filter: x_t = x_{t-1} + w (Q), z_t = x_t + v (R).
    The normalised innovation (z - x_pred)/sqrt(S) flags bursts that the
    tracked baseline does not explain."""
    x, p = x0, p0
    est, zs = [], []
    for z in counts:
        x_pred, p_pred = x, p + q
        s = p_pred + r
        innov = z - x_pred
        zs.append(innov / np.sqrt(s))
        k = p_pred / s
        x = x_pred + k * innov
        p = (1 - k) * p_pred
        est.append(x)
    return KalmanResult(est, zs, float(max(zs, default=0.0)))

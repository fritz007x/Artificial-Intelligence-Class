"""Tiny TF-IDF vector index (numpy) used for (a) RAG over SOC playbooks, cited in
rationale records, and (b) the allow-listed `similar_alerts` vector-lookup tool."""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

_TOKEN = re.compile(r"[a-z][a-z_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class TfidfIndex:
    def __init__(self, docs: dict[str, str]):
        self.ids = list(docs)
        toks = [tokenize(docs[i]) for i in self.ids]
        vocab = sorted({t for ts in toks for t in ts})
        self.vocab = {t: j for j, t in enumerate(vocab)}
        df = Counter(t for ts in toks for t in set(ts))
        n = len(self.ids)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in vocab])
        self.mat = np.vstack([self._vec(ts) for ts in toks]) if toks else np.zeros((0, len(vocab)))
        self.docs = docs

    def _vec(self, toks: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t, c in Counter(toks).items():
            if t in self.vocab:
                v[self.vocab[t]] = c
        v = v * self.idf
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    def query(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        sims = self.mat @ self._vec(tokenize(text))
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], round(float(sims[i]), 3)) for i in order if sims[i] > 0]


def load_playbooks(directory: Path) -> TfidfIndex:
    docs = {p.stem: p.read_text() for p in sorted(Path(directory).glob("*.md"))}
    return TfidfIndex(docs)

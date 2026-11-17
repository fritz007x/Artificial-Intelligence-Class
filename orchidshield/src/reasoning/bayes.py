"""Discrete Bayesian network with exact inference by enumeration.
Missing evidence is marginalised (never imputed), which is how the agent handles
incomplete percepts such as hosts without EDR coverage."""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Node:
    name: str
    parents: list[str]
    cpt: dict[str, float]

    def p_true(self, assignment: dict[str, bool]) -> float:
        key = ",".join("T" if assignment[p] else "F" for p in self.parents)
        return self.cpt[key]


class BayesNet:
    def __init__(self, nodes: list[Node], query: str = "Compromise"):
        self.nodes = nodes
        self.by_name = {n.name: n for n in nodes}
        self.query = query
        self._validate()

    def _validate(self) -> None:
        seen: set[str] = set()
        for n in self.nodes:           # topological order is required
            missing = [p for p in n.parents if p not in seen]
            if missing:
                raise ValueError(f"{n.name}: parents {missing} must precede it")
            expected = 2 ** len(n.parents)
            if len(n.cpt) != expected:
                raise ValueError(f"{n.name}: CPT needs {expected} rows, has {len(n.cpt)}")
            if not all(0.0 <= v <= 1.0 for v in n.cpt.values()):
                raise ValueError(f"{n.name}: probabilities out of range")
            seen.add(n.name)

    @classmethod
    def from_json(cls, path: Path) -> "BayesNet":
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Node(n["name"], n["parents"], n["cpt"]) for n in spec["nodes"]], spec.get("query", "Compromise"))

    def joint(self, assignment: dict[str, bool]) -> float:
        p = 1.0
        for n in self.nodes:
            pt = n.p_true(assignment)
            p *= pt if assignment[n.name] else 1.0 - pt
        return p

    def posterior(self, var: str, evidence: dict[str, bool | None]) -> float:
        """P(var=True | evidence) by enumeration; None-valued evidence is treated as unobserved."""
        ev = {k: v for k, v in evidence.items() if v is not None}
        unknown = for_sum = [n.name for n in self.nodes if n.name not in ev]
        num = den = 0.0
        for values in itertools.product([True, False], repeat=len(unknown)):
            a = {**ev, **dict(zip(for_sum, values))}
            pj = self.joint(a)
            den += pj
            if a[var]:
                num += pj
        return num / den if den > 0 else 0.0

    def prior(self, var: str) -> float:
        return self.posterior(var, {})

    def explain(self, var: str, evidence: dict[str, bool | None]) -> list[tuple[str, float]]:
        """Leave-one-out evidence impact: how much each observed variable moves the posterior."""
        full = self.posterior(var, evidence)
        out = []
        for k, v in evidence.items():
            if v is None:
                continue
            reduced = {**evidence, k: None}
            out.append((f"{k}={'T' if v else 'F'}", round(full - self.posterior(var, reduced), 4)))
        return sorted(out, key=lambda t: -abs(t[1]))

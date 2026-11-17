"""Constraint satisfaction for the response-scheduling subproblem.

Variables: response tasks (enrich, contain, notify, ...).
Domains: (analyst, slot) pairs, filtered for skill and seniority (node consistency).
Constraints (binary): one task per analyst per slot; precedence (enrichment before
containment of the same target); approval-gated tasks need a senior analyst who is not
the same person who triaged (separation of duties).
Solvers: plain backtracking, backtracking + MRV + LCV + forward checking, MAC with
AC-3 preprocessing, and min-conflicts local search."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

Value = tuple[str, int]                     # (analyst_id, slot)
Pred = Callable[[Value, Value], bool]


@dataclass
class Task:
    task_id: str
    skill: str
    requires_senior: bool = False
    after: list[str] = field(default_factory=list)
    allowed_slots: list[int] | None = None
    separate_from: list[str] = field(default_factory=list)   # different analyst than these tasks


@dataclass
class CSP:
    variables: list[str]
    domains: dict[str, list[Value]]
    constraints: dict[tuple[str, str], list[Pred]]
    stats: dict[str, int] = field(default_factory=lambda: {"assignments": 0, "backtracks": 0, "revisions": 0})

    def neighbors(self, x: str) -> list[str]:
        return [b for (a, b) in self.constraints if a == x]

    def consistent(self, x: str, vx: Value, assignment: dict[str, Value]) -> bool:
        for y, vy in assignment.items():
            for pred in self.constraints.get((x, y), []):
                if not pred(vx, vy):
                    return False
        return True

    def conflicts(self, x: str, vx: Value, assignment: dict[str, Value]) -> int:
        return sum(1 for y, vy in assignment.items() if y != x
                   for pred in self.constraints.get((x, y), []) if not pred(vx, vy))


def build_schedule_csp(tasks: list[Task], analysts: list[dict], slots: list[int]) -> CSP:
    by_id = {t.task_id: t for t in tasks}
    domains: dict[str, list[Value]] = {}
    for t in tasks:
        dom = []
        for a in analysts:
            if t.skill not in a["skills"] or (t.requires_senior and not a["senior"]):
                continue
            for s in slots:
                if t.allowed_slots is None or s in t.allowed_slots:
                    dom.append((a["id"], s))
        domains[t.task_id] = dom
    cons: dict[tuple[str, str], list[Pred]] = {}

    def add(x, y, p_xy: Pred):
        cons.setdefault((x, y), []).append(p_xy)
        cons.setdefault((y, x), []).append(lambda vy, vx, p=p_xy: p(vx, vy))

    ids = [t.task_id for t in tasks]
    for i, x in enumerate(ids):
        for y in ids[i + 1:]:
            add(x, y, lambda a, b: a != b)                          # analyst can do one task per slot
    for t in tasks:
        for pre in t.after:
            if pre in by_id:
                add(pre, t.task_id, lambda a, b: a[1] < b[1])       # precedence
        for other in t.separate_from:
            if other in by_id:
                add(t.task_id, other, lambda a, b: a[0] != b[0])    # separation of duties
    return CSP(ids, domains, cons)


def ac3(csp: CSP, domains: dict[str, list[Value]] | None = None, queue=None) -> dict[str, list[Value]] | None:
    doms = {k: list(v) for k, v in (domains or csp.domains).items()}
    q = deque(queue if queue is not None else list(csp.constraints))
    while q:
        x, y = q.popleft()
        csp.stats["revisions"] += 1
        preds = csp.constraints[(x, y)]
        new = [vx for vx in doms[x] if any(all(p(vx, vy) for p in preds) for vy in doms[y])]
        if len(new) < len(doms[x]):
            doms[x] = new
            if not new:
                return None
            q.extend((z, x) for z in csp.neighbors(x) if z != y)
    return doms


def backtrack(csp: CSP, mrv: bool = False, lcv: bool = False, forward_check: bool = False,
              mac: bool = False, limit: int = 200_000) -> dict[str, Value] | None:
    doms0 = {k: list(v) for k, v in csp.domains.items()}
    if mac:
        doms0 = ac3(csp, doms0)
        if doms0 is None:
            return None

    def select(assignment, doms):
        un = [v for v in csp.variables if v not in assignment]
        if mrv:
            return min(un, key=lambda v: (len(doms[v]), -len(csp.neighbors(v))))
        return un[0]

    def order(var, assignment, doms):
        vals = doms[var]
        if not lcv:
            return vals
        def ruled_out(val):
            return sum(1 for n in csp.neighbors(var) if n not in assignment
                       for vn in doms[n] if not all(p(val, vn) for p in csp.constraints[(var, n)]))
        return sorted(vals, key=ruled_out)

    def rec(assignment, doms):
        if len(assignment) == len(csp.variables):
            return dict(assignment)
        if csp.stats["assignments"] > limit:
            return None
        var = select(assignment, doms)
        for val in order(var, assignment, doms):
            csp.stats["assignments"] += 1
            if not csp.consistent(var, val, assignment):
                continue
            assignment[var] = val
            new_doms = {**doms, var: [val]}
            ok = True
            if forward_check or mac:
                for n in csp.neighbors(var):
                    if n in assignment:
                        continue
                    new_doms[n] = [vn for vn in doms[n] if all(p(vn, val) for p in csp.constraints[(n, var)])]
                    if not new_doms[n]:
                        ok = False
                        break
                if ok and mac:
                    r = ac3(csp, new_doms, [(n, var) for n in csp.neighbors(var) if n not in assignment])
                    ok, new_doms = r is not None, (r or new_doms)
            if ok:
                res = rec(assignment, new_doms)
                if res:
                    return res
            del assignment[var]
            csp.stats["backtracks"] += 1
        return None

    return rec({}, doms0)


def min_conflicts(csp: CSP, rng: random.Random, max_steps: int = 5000) -> dict[str, Value] | None:
    if any(not d for d in csp.domains.values()):
        return None
    assignment = {v: rng.choice(csp.domains[v]) for v in csp.variables}
    for _ in range(max_steps):
        csp.stats["assignments"] += 1
        conflicted = [v for v in csp.variables if csp.conflicts(v, assignment[v], assignment)]
        if not conflicted:
            return assignment
        var = rng.choice(conflicted)
        scores = [(csp.conflicts(var, val, assignment), rng.random(), val) for val in csp.domains[var]]
        assignment[var] = min(scores)[2]
    return None


def verify(csp: CSP, assignment: dict[str, Value]) -> bool:
    return all(assignment.get(v) in csp.domains[v] for v in csp.variables) and \
        all(csp.conflicts(v, assignment[v], assignment) == 0 for v in csp.variables)

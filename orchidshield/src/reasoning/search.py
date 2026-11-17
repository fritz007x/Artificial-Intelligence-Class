"""Attack-path problem formulation and search.

State: the asset an adversary currently controls. Initial state: a compromised host.
Goal test: the asset is a crown jewel (criticality >= threshold). Actions: move along a
network edge. Transition: deterministic. Step cost: exploitation difficulty of the edge
(zone-crossing is harder). The defender uses the cheapest path as an estimate of how
close an attacker is to a crown jewel and to compute blast radius.

Uninformed: BFS, DFS, UCS, iterative deepening. Informed: greedy best-first, A*.
Heuristic h(n) = c_min * hops(n, goal), where hops comes from a reverse BFS. Since every
edge costs >= c_min and changes hops by at most one, h is admissible and consistent.
Optimisation: hill climbing and simulated annealing choose which hosts to isolate."""
from __future__ import annotations

import heapq
import itertools
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

Graph = dict[str, list[tuple[str, float]]]


@dataclass
class SearchResult:
    algorithm: str
    path: list[str] | None
    cost: float
    expanded: int
    optimal_known: bool = False

    @property
    def found(self) -> bool:
        return self.path is not None


def edge_cost(zone_a: str, zone_b: str) -> float:
    if zone_a == zone_b:
        return 1.0
    hard = {("corp", "prod"): 3.0, ("prod", "corp"): 3.0, ("dmz", "prod"): 4.0, ("prod", "dmz"): 4.0,
            ("corp", "mgmt"): 3.5, ("mgmt", "corp"): 3.5, ("dmz", "corp"): 2.5, ("corp", "dmz"): 2.5}
    return hard.get((zone_a, zone_b), 2.0)


def build_graph(assets: dict[str, dict], blocked: set[str] | None = None) -> Graph:
    blocked = blocked or set()
    g: Graph = {a: [] for a in assets if a not in blocked}
    for a, info in assets.items():
        if a in blocked:
            continue
        for b in info.get("connected_to", []):
            if b in assets and b not in blocked:
                c = edge_cost(info["zone"], assets[b]["zone"])
                if (b, c) not in g[a]:
                    g[a].append((b, c))
                if (a, c) not in g.setdefault(b, []):
                    g[b].append((a, c))
    for k in g:
        g[k].sort()
    return g


def _path(parent: dict[str, str | None], goal: str) -> list[str]:
    out = [goal]
    while parent[out[-1]] is not None:
        out.append(parent[out[-1]])  # type: ignore[arg-type]
    return out[::-1]


def _cost(g: Graph, path: list[str]) -> float:
    return sum(dict(g[a])[b] for a, b in zip(path, path[1:]))


def bfs(g: Graph, start: str, goal: Callable[[str], bool]) -> SearchResult:
    if goal(start):
        return SearchResult("BFS", [start], 0, 0)
    parent: dict[str, str | None] = {start: None}
    q, expanded = deque([start]), 0
    while q:
        n = q.popleft()
        expanded += 1
        for m, _ in g.get(n, []):
            if m not in parent:
                parent[m] = n
                if goal(m):
                    p = _path(parent, m)
                    return SearchResult("BFS", p, _cost(g, p), expanded)
                q.append(m)
    return SearchResult("BFS", None, math.inf, expanded)


def dfs(g: Graph, start: str, goal: Callable[[str], bool]) -> SearchResult:
    stack, seen, expanded = [(start, [start])], set(), 0
    while stack:
        n, p = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        expanded += 1
        if goal(n):
            return SearchResult("DFS", p, _cost(g, p), expanded)
        for m, _ in reversed(g.get(n, [])):
            if m not in seen:
                stack.append((m, p + [m]))
    return SearchResult("DFS", None, math.inf, expanded)


def iddfs(g: Graph, start: str, goal: Callable[[str], bool], max_depth: int = 25) -> SearchResult:
    expanded = 0

    def dls(n: str, depth: int, path: list[str]) -> list[str] | None:
        nonlocal expanded
        expanded += 1
        if goal(n):
            return path
        if depth == 0:
            return None
        for m, _ in g.get(n, []):
            if m not in path:
                r = dls(m, depth - 1, path + [m])
                if r:
                    return r
        return None

    for d in range(max_depth + 1):
        r = dls(start, d, [start])
        if r:
            return SearchResult("IDDFS", r, _cost(g, r), expanded)
    return SearchResult("IDDFS", None, math.inf, expanded)


def _best_first(name: str, g: Graph, start: str, goal: Callable[[str], bool],
                f: Callable[[float, str], float]) -> SearchResult:
    tie = itertools.count()
    frontier = [(f(0.0, start), next(tie), 0.0, start)]
    best_g = {start: 0.0}
    parent: dict[str, str | None] = {start: None}
    closed: set[str] = set()
    expanded = 0
    while frontier:
        _, _, gc, n = heapq.heappop(frontier)
        if n in closed:
            continue
        if goal(n):
            p = _path(parent, n)
            return SearchResult(name, p, gc, expanded, optimal_known=name in ("UCS", "A*"))
        closed.add(n)
        expanded += 1
        for m, c in g.get(n, []):
            ng = gc + c
            if m not in closed and ng < best_g.get(m, math.inf):
                best_g[m] = ng
                parent[m] = n
                heapq.heappush(frontier, (f(ng, m), next(tie), ng, m))
    return SearchResult(name, None, math.inf, expanded)


def hop_heuristic(g: Graph, goals: set[str]) -> Callable[[str], float]:
    """Admissible + consistent: c_min * (min hops to any goal)."""
    c_min = min((c for edges in g.values() for _, c in edges), default=1.0)
    hops = {x: 0 for x in goals if x in g}
    q = deque(hops)
    while q:
        n = q.popleft()
        for m, _ in g.get(n, []):
            if m not in hops:
                hops[m] = hops[n] + 1
                q.append(m)
    return lambda n: c_min * hops.get(n, 10_000)


def ucs(g, start, goal):
    return _best_first("UCS", g, start, goal, lambda gc, n: gc)


def greedy(g, start, goal, h):
    return _best_first("Greedy", g, start, goal, lambda gc, n: h(n))


def astar(g, start, goal, h):
    return _best_first("A*", g, start, goal, lambda gc, n: gc + h(n))


def dijkstra_all(g: Graph, start: str) -> dict[str, float]:
    dist = {start: 0.0}
    pq = [(0.0, start)]
    while pq:
        d, n = heapq.heappop(pq)
        if d > dist.get(n, math.inf):
            continue
        for m, c in g.get(n, []):
            if d + c < dist.get(m, math.inf):
                dist[m] = d + c
                heapq.heappush(pq, (d + c, m))
    return dist


def blast_radius(g: Graph, start: str, criticality: dict[str, int], budget: float = 8.0) -> float:
    """Criticality-weighted reach of an attacker starting at `start`, in [0, 1]."""
    if start not in g:
        return 0.0
    dist = dijkstra_all(g, start)
    total = sum(criticality.values()) or 1
    score = sum(criticality.get(n, 1) / (1.0 + d) for n, d in dist.items() if n != start and d <= budget)
    return min(1.0, score / (0.35 * total))


# ------------------------------------------------------------------ optimisation
@dataclass
class SegmentationProblem:
    """Choose up to k hosts to isolate so that critical assets reachable from the
    compromised set are minimised, plus a disruption cost per isolated host."""
    assets: dict[str, dict]
    compromised: set[str]
    k: int
    forbidden: set[str] = field(default_factory=set)
    crit_min: int = 4

    def candidates(self) -> list[str]:
        return sorted(a for a in self.assets if a not in self.forbidden)

    def cost(self, isolated: frozenset[str]) -> float:
        g = build_graph(self.assets, set(isolated))
        reach: set[str] = set()
        for c in self.compromised:
            if c in g:
                reach |= set(dijkstra_all(g, c))
        exposed = sum(self.assets[a]["criticality"] for a in reach
                      if self.assets[a]["criticality"] >= self.crit_min and a not in self.compromised)
        disruption = sum(0.5 * self.assets[a]["criticality"] for a in isolated)
        return exposed + disruption


def exhaustive(p: SegmentationProblem) -> tuple[frozenset[str], float, int]:
    best, best_c, n = frozenset(), p.cost(frozenset()), 1
    for r in range(1, p.k + 1):
        for combo in itertools.combinations(p.candidates(), r):
            n += 1
            c = p.cost(frozenset(combo))
            if c < best_c:
                best, best_c = frozenset(combo), c
    return best, best_c, n


def hill_climb(p: SegmentationProblem, rng: random.Random, restarts: int = 20) -> tuple[frozenset[str], float, int]:
    cands, evals = p.candidates(), 0
    best, best_c = frozenset(), p.cost(frozenset())
    for _ in range(restarts):
        cur = frozenset(rng.sample(cands, min(p.k, len(cands))))
        cur_c = p.cost(cur)
        evals += 1
        improved = True
        while improved:
            improved = False
            neigh = [cur - {x} for x in cur] + [cur | {y} for y in cands if y not in cur and len(cur) < p.k] \
                + [(cur - {x}) | {y} for x in cur for y in cands if y not in cur]
            for nb in neigh:
                evals += 1
                c = p.cost(frozenset(nb))
                if c < cur_c:
                    cur, cur_c, improved = frozenset(nb), c, True
        if cur_c < best_c:
            best, best_c = cur, cur_c
    return best, best_c, evals


def simulated_annealing(p: SegmentationProblem, rng: random.Random, steps: int = 1500, t0: float = 5.0):
    cands = p.candidates()
    cur: frozenset[str] = frozenset()
    cur_c = p.cost(cur)
    best, best_c = cur, cur_c
    for i in range(steps):
        t = t0 * (1 - i / steps) + 1e-3
        x = rng.choice(cands)
        nb = cur - {x} if x in cur else (cur | {x} if len(cur) < p.k else (cur - {rng.choice(sorted(cur))}) | {x})
        c = p.cost(frozenset(nb))
        if c < cur_c or rng.random() < math.exp(-(c - cur_c) / t):
            cur, cur_c = frozenset(nb), c
            if c < best_c:
                best, best_c = cur, c
    return best, best_c, steps

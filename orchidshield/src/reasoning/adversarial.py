"""Adversarial search: a defender/attacker game on the asset graph.

MAX = defender: each turn isolates one host (or passes). MIN = attacker: moves from a
controlled host to an adjacent, non-isolated host and controls it. Terminal: depth
limit. Utility (defender view) = -(sum of criticality^2 of hosts the attacker controls)
- disruption cost of isolations. Minimax, alpha-beta pruning (with move ordering) and
expectimax (attacker modelled as stochastic, uniform over moves) are implemented so
the notebook can compare node counts and decisions."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GameState:
    attacker: frozenset[str]
    isolated: frozenset[str]


class ContainmentGame:
    def __init__(self, assets: dict[str, dict], forbidden: set[str] | None = None, isolation_cost: float = 0.6):
        self.assets = assets
        self.adj = {a: set(i.get("connected_to", [])) for a, i in assets.items()}
        for a, ns in list(self.adj.items()):
            for b in ns:
                self.adj.setdefault(b, set()).add(a)
        self.forbidden = forbidden or set()
        self.iso_cost = isolation_cost
        self.nodes = 0

    def crit(self, a: str) -> int:
        return self.assets.get(a, {}).get("criticality", 1)

    def defender_moves(self, s: GameState) -> list[str | None]:
        frontier = {n for a in s.attacker if a not in s.isolated for n in self.adj.get(a, ())}
        cands = (frontier | s.attacker) - s.isolated - self.forbidden
        # move ordering: most critical first improves alpha-beta pruning
        return [None, *sorted(cands, key=lambda a: (-self.crit(a), a))]

    def attacker_moves(self, s: GameState) -> list[str]:
        return sorted({n for a in s.attacker if a not in s.isolated for n in self.adj.get(a, ())
                       if n not in s.attacker and n not in s.isolated}, key=lambda a: (-self.crit(a), a))

    def utility(self, s: GameState) -> float:
        lost = sum(self.crit(a) ** 2 for a in s.attacker if a not in s.isolated)
        return -lost - self.iso_cost * sum(self.crit(a) for a in s.isolated)

    def _def(self, s, m):
        return GameState(s.attacker, s.isolated | {m}) if m else s

    def _att(self, s, m):
        return GameState(s.attacker | {m}, s.isolated)

    # -------------------------------------------------------------- minimax
    def minimax(self, s: GameState, depth: int, maximizing: bool) -> float:
        self.nodes += 1
        if depth == 0:
            return self.utility(s)
        if maximizing:
            return max(self.minimax(self._def(s, m), depth - 1, False) for m in self.defender_moves(s))
        moves = self.attacker_moves(s)
        if not moves:
            return self.minimax(s, depth - 1, True)
        return min(self.minimax(self._att(s, m), depth - 1, True) for m in moves)

    def alphabeta(self, s: GameState, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        self.nodes += 1
        if depth == 0:
            return self.utility(s)
        if maximizing:
            v = -math.inf
            for m in self.defender_moves(s):
                v = max(v, self.alphabeta(self._def(s, m), depth - 1, alpha, beta, False))
                alpha = max(alpha, v)
                if alpha >= beta:
                    break
            return v
        moves = self.attacker_moves(s)
        if not moves:
            return self.alphabeta(s, depth - 1, alpha, beta, True)
        v = math.inf
        for m in moves:
            v = min(v, self.alphabeta(self._att(s, m), depth - 1, alpha, beta, True))
            beta = min(beta, v)
            if alpha >= beta:
                break
        return v

    def expectimax(self, s: GameState, depth: int, maximizing: bool) -> float:
        self.nodes += 1
        if depth == 0:
            return self.utility(s)
        if maximizing:
            return max(self.expectimax(self._def(s, m), depth - 1, False) for m in self.defender_moves(s))
        moves = self.attacker_moves(s)
        if not moves:
            return self.expectimax(s, depth - 1, True)
        return sum(self.expectimax(self._att(s, m), depth - 1, True) for m in moves) / len(moves)

    def best_isolation(self, compromised: set[str], depth: int = 4, method: str = "alphabeta") -> tuple[str | None, float, int]:
        """Defender's best first move. Returns (host_to_isolate or None, value, nodes searched)."""
        s0 = GameState(frozenset(compromised), frozenset())
        self.nodes = 0
        best, best_v = None, -math.inf
        for m in self.defender_moves(s0):
            s1 = self._def(s0, m)
            if method == "alphabeta":
                v = self.alphabeta(s1, depth - 1, best_v, math.inf, False)
            elif method == "minimax":
                v = self.minimax(s1, depth - 1, False)
            else:
                v = self.expectimax(s1, depth - 1, False)
            if v > best_v:
                best, best_v = m, v
        return best, best_v, self.nodes

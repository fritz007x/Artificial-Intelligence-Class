"""First-order Horn-clause reasoning: atom parsing, unification, forward chaining
(with justifications), and depth-bounded backward chaining (SLD resolution) that
returns an inspectable proof. Variables are written ``?x``. Heads may be negative
literals (``¬P(..)``) so the KB can represent explicit refutations; bodies are
positive conjunctions (definite clauses over signed literals)."""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from core.schemas import Atom

_ATOM_RE = re.compile(r"^\s*(¬|not\s+|~)?\s*([A-Z][A-Za-z]*)\s*\(([^()]*)\)\s*$")


def parse_atom(text: str) -> Atom:
    m = _ATOM_RE.match(text)
    if not m:
        raise ValueError(f"cannot parse atom: {text!r}")
    neg, pred, args = m.groups()
    arg_list = tuple(a.strip() for a in args.split(",") if a.strip())
    return Atom(predicate=pred, args=arg_list, positive=not neg)


def is_var(term: str) -> bool:
    return term.startswith("?")


Subst = dict[str, str]


def walk(term: str, theta: Subst) -> str:
    while is_var(term) and term in theta:
        term = theta[term]
    return term


def unify(a: Atom, b: Atom, theta: Subst | None = None) -> Subst | None:
    """Unify two signed atoms (function-free terms). Returns extended substitution or None."""
    if a.predicate != b.predicate or a.positive != b.positive or len(a.args) != len(b.args):
        return None
    theta = dict(theta or {})
    for x, y in zip(a.args, b.args):
        x, y = walk(x, theta), walk(y, theta)
        if x == y:
            continue
        if is_var(x):
            theta[x] = y
        elif is_var(y):
            theta[y] = x
        else:
            return None
    return theta


def substitute(atom: Atom, theta: Subst) -> Atom:
    return Atom(predicate=atom.predicate, args=tuple(walk(t, theta) for t in atom.args), positive=atom.positive)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    body: tuple[Atom, ...]
    head: Atom
    description: str = ""

    def __str__(self) -> str:
        return f"{self.rule_id}: {' ∧ '.join(map(str, self.body))} → {self.head}"


@dataclass
class Justification:
    rule_id: str | None              # None => asserted fact
    premises: tuple[tuple, ...] = ()  # atom keys


@dataclass
class HornKB:
    facts: dict[tuple, Atom] = field(default_factory=dict)
    rules: list[Rule] = field(default_factory=list)
    just: dict[tuple, Justification] = field(default_factory=dict)

    def tell(self, atom: Atom) -> bool:
        if atom.key in self.facts:
            return False
        self.facts[atom.key] = atom
        self.just.setdefault(atom.key, Justification(None))
        return True

    def by_pred(self) -> dict[tuple[str, bool], list[Atom]]:
        idx: dict[tuple[str, bool], list[Atom]] = {}
        for a in self.facts.values():
            idx.setdefault((a.predicate, a.positive), []).append(a)
        return idx

    # ------------------------------------------------------------ forward chaining
    def _match_body(self, body: tuple[Atom, ...], idx, theta: Subst) -> Iterator[tuple[Subst, list[Atom]]]:
        if not body:
            yield theta, []
            return
        first, rest = body[0], body[1:]
        for fact in idx.get((first.predicate, first.positive), []):
            t2 = unify(first, fact, theta)
            if t2 is not None:
                for t3, used in self._match_body(rest, idx, t2):
                    yield t3, [fact, *used]

    def forward_chain(self, max_rounds: int = 50) -> list[Atom]:
        """Semi-naive fixpoint. Returns newly derived atoms in derivation order."""
        derived: list[Atom] = []
        for _ in range(max_rounds):
            idx = self.by_pred()
            new: list[tuple[Atom, Rule, list[Atom]]] = []
            for rule in self.rules:
                for theta, used in self._match_body(rule.body, idx, {}):
                    head = substitute(rule.head, theta)
                    if any(is_var(t) for t in head.args):
                        continue
                    if head.key not in self.facts and all(head.key != n[0].key for n in new):
                        new.append((head, rule, used))
            if not new:
                break
            for head, rule, used in new:
                self.facts[head.key] = head
                self.just[head.key] = Justification(rule.rule_id, tuple(u.key for u in used))
                derived.append(head)
        return derived

    def conflicts(self) -> list[tuple[Atom, Atom]]:
        out = []
        for key, atom in self.facts.items():
            if atom.positive and atom.negate().key in self.facts:
                out.append((atom, self.facts[atom.negate().key]))
        return out

    def support_leaves(self, key: tuple, seen: set | None = None) -> set[tuple]:
        """All asserted (leaf) facts a derived fact ultimately rests on."""
        seen = seen if seen is not None else set()
        if key in seen:
            return set()
        seen.add(key)
        j = self.just.get(key)
        if j is None or j.rule_id is None:
            return {key}
        out: set[tuple] = set()
        for p in j.premises:
            out |= self.support_leaves(p, seen)
        return out

    def support_closure(self, key: tuple, seen: set | None = None) -> set[tuple]:
        seen = seen if seen is not None else set()
        if key in seen:
            return seen
        seen.add(key)
        j = self.just.get(key)
        if j and j.rule_id:
            for p in j.premises:
                self.support_closure(p, seen)
        return seen

    # ------------------------------------------------------------ backward chaining
    _counter = itertools.count()

    def _rename(self, rule: Rule) -> Rule:
        n = next(self._counter)
        ren = lambda a: Atom(predicate=a.predicate,
                             args=tuple(f"{t}_{n}" if is_var(t) else t for t in a.args), positive=a.positive)
        return Rule(rule.rule_id, tuple(ren(b) for b in rule.body), ren(rule.head), rule.description)

    def ask(self, goal: Atom, max_depth: int = 12) -> list[str] | None:
        """Backward chaining over *asserted* facts and rules. Returns a proof (list of
        human-readable steps, leaves first) for the first derivation found, else None."""
        base = {k: a for k, a in self.facts.items() if self.just.get(k, Justification(None)).rule_id is None}
        base_idx: dict[tuple[str, bool], list[Atom]] = {}
        for a in base.values():
            base_idx.setdefault((a.predicate, a.positive), []).append(a)

        def solve(goals: list[Atom], theta: Subst, depth: int, stack: frozenset) -> Iterator[tuple[Subst, list[str]]]:
            if not goals:
                yield theta, []
                return
            g = substitute(goals[0], theta)
            rest = goals[1:]
            for fact in base_idx.get((g.predicate, g.positive), []):
                t2 = unify(g, fact, theta)
                if t2 is not None:
                    for t3, steps in solve(rest, t2, depth, stack):
                        yield t3, [f"fact {fact}", *steps]
            if depth <= 0:
                return
            # variant check: goals equal up to variable renaming are not re-expanded
            sig = re.sub(r"\?[A-Za-z0-9_]+", "?", str(g))
            if sig in stack:
                return
            for rule in self.rules:
                r = self._rename(rule)
                t2 = unify(r.head, g, theta)
                if t2 is None:
                    continue
                for t3, sub_steps in solve(list(r.body), t2, depth - 1, stack | {sig}):
                    concl = substitute(g, t3)
                    step = f"{rule.rule_id}: {' ∧ '.join(str(substitute(b, t3)) for b in r.body)} ⊢ {concl}"
                    for t4, rest_steps in solve(rest, t3, depth, stack):
                        yield t4, [*sub_steps, step, *rest_steps]

        for _theta, steps in solve([goal], {}, max_depth, frozenset()):
            return steps
        return None


def rules_from_dicts(items: Iterable[dict]) -> list[Rule]:
    out = []
    for it in items:
        out.append(Rule(it["id"], tuple(parse_atom(b) for b in it["if"]), parse_atom(it["then"]),
                        it.get("description", "")))
    return out

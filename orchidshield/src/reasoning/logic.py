"""Propositional logic toolkit: CNF parsing, resolution refutation, truth-table
model checking (TT-ENTAILS) and Horn forward chaining (PL-FC-ENTAILS).
Used for the 03_logic entailment test cases and in the simulation notebook."""
from __future__ import annotations

import itertools
from typing import Iterable

Literal_ = str            # "P" or "~P"
Clause = frozenset[str]


def neg(lit: str) -> str:
    return lit[1:] if lit.startswith("~") else "~" + lit


def parse_clause(text: str) -> Clause:
    """'~A | B | ~C' -> frozenset({'~A','B','~C'})"""
    return frozenset(t.strip() for t in text.split("|") if t.strip())


def horn_to_cnf(body: Iterable[str], head: str) -> Clause:
    """A ∧ B → C  ==  ~A ∨ ~B ∨ C"""
    return frozenset([*(neg(b) for b in body), head])


def symbols(clauses: Iterable[Clause]) -> set[str]:
    return {lit.lstrip("~") for c in clauses for lit in c}


def resolve(ci: Clause, cj: Clause) -> list[Clause]:
    out = []
    for lit in ci:
        if neg(lit) in cj:
            r = (ci - {lit}) | (cj - {neg(lit)})
            if not any(neg(x) in r for x in r):     # drop tautologies
                out.append(frozenset(r))
    return out


def pl_resolution(kb: list[Clause], query: str, max_steps: int = 20000) -> tuple[bool, int]:
    """KB ⊨ query iff KB ∧ ¬query is unsatisfiable. Returns (entailed, resolvents_generated)."""
    clauses = set(kb) | {frozenset([neg(query)])}
    generated = 0
    while True:
        new: set[Clause] = set()
        cl = list(clauses)
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                for r in resolve(cl[i], cl[j]):
                    generated += 1
                    if not r:
                        return True, generated
                    new.add(r)
                    if generated > max_steps:
                        return False, generated
        if new <= clauses:
            return False, generated
        clauses |= new


def _satisfies(clause: Clause, model: dict[str, bool]) -> bool:
    return any(model[l.lstrip("~")] != l.startswith("~") for l in clause)


def tt_entails(kb: list[Clause], query: str) -> tuple[bool, int]:
    """Truth-table model checking. Returns (entailed, models_checked)."""
    syms = sorted(symbols(kb) | {query.lstrip("~")})
    checked = 0
    for values in itertools.product([False, True], repeat=len(syms)):
        model = dict(zip(syms, values))
        checked += 1
        if all(_satisfies(c, model) for c in kb):
            q_true = model[query.lstrip("~")] != query.startswith("~")
            if not q_true:
                return False, checked
    return True, checked


def satisfiable(kb: list[Clause]) -> bool:
    syms = sorted(symbols(kb))
    return any(all(_satisfies(c, dict(zip(syms, v))) for c in kb)
               for v in itertools.product([False, True], repeat=len(syms)))


def pl_fc_entails(rules: list[tuple[list[str], str]], facts: list[str], query: str) -> bool:
    """Forward chaining for definite clauses (count-based agenda algorithm)."""
    count = {i: len(b) for i, (b, _) in enumerate(rules)}
    inferred: set[str] = set()
    agenda = list(facts)
    while agenda:
        p = agenda.pop()
        if p == query:
            return True
        if p in inferred:
            continue
        inferred.add(p)
        for i, (body, head) in enumerate(rules):
            if p in body:
                count[i] -= 1
                if count[i] == 0:
                    agenda.append(head)
    return False

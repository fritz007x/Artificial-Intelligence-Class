"""Logic & knowledge representation tests, driven by 03_logic/entailment_tests.yaml."""
from pathlib import Path

import pytest
import yaml

from agent.entailment import EntailmentChecker
from agent.knowledge import KnowledgeBase
from reasoning.fol import HornKB, parse_atom, rules_from_dicts, unify
from reasoning.logic import parse_clause, pl_fc_entails, pl_resolution, satisfiable, tt_entails

ROOT = Path(__file__).resolve().parents[1]
CASES = yaml.safe_load((ROOT / "03_logic" / "entailment_tests.yaml").read_text(encoding="utf-8"))
CNF = [parse_clause(l) for l in (ROOT / "03_logic" / "cnf_rules.txt").read_text(encoding="utf-8").splitlines()
       if l.strip() and not l.startswith("#")]


@pytest.mark.parametrize("case", CASES["propositional"], ids=lambda c: c["id"])
def test_propositional_entailment(case):
    kb = CNF + [frozenset([f]) for f in case["facts"]]
    if case["expect"] == "unsatisfiable":
        assert not satisfiable(kb)
        return
    res, _ = pl_resolution(kb, case["query"])
    tt, _ = tt_entails(kb, case["query"])
    assert res == tt == case["expect"]


def test_pl_forward_chaining_agrees_on_horn_subset():
    rules = [(["IocA", "ExecA"], "TpA"), (["TpA"], "CompH"), (["CompH", "CritH"], "HighH")]
    assert pl_fc_entails(rules, ["IocA", "ExecA", "CritH"], "HighH")
    assert not pl_fc_entails(rules, ["IocA", "CritH"], "HighH")


def _kb(settings, facts):
    kb = KnowledgeBase(settings, {"assets": {}, "users": {}, "indicators": []})
    for f in facts:
        kb.tell(f)
    return kb


@pytest.mark.parametrize("case", CASES["fol"], ids=lambda c: c["id"])
def test_fol_entailment(settings, case):
    kb = _kb(settings, case["facts"])
    res = EntailmentChecker(settings, kb).check(case["query"])
    assert res.status.value == case["expect"], res.rationale


def test_at_least_five_entailment_cases():
    assert len(CASES["propositional"]) + len(CASES["fol"]) >= 5


def test_backward_chaining_matches_forward_closure(settings):
    kb = _kb(settings, CASES["fol"][0]["facts"])
    hk = HornKB(rules=list(kb.rules))
    for f in kb.facts.values():
        hk.tell(f.atom)
    goal = parse_atom("HighPriorityIncident(srv-db-01)")
    proof = hk.ask(goal)
    assert proof and proof[-1].endswith("HighPriorityIncident(srv-db-01)")
    assert kb.holds(goal)
    assert hk.ask(parse_atom("HighPriorityIncident(ws-999)")) is None


def test_existential_query_by_unification(settings):
    kb = _kb(settings, CASES["fol"][0]["facts"])
    hk = HornKB(rules=list(kb.rules))
    for f in kb.facts.values():
        hk.tell(f.atom)
    assert hk.ask(parse_atom("CompromisedHost(?h)"))
    assert [a.args[0] for a in kb.query("CompromisedHost(?h)")] == ["srv-db-01"]


def test_unification():
    assert unify(parse_atom("P(?x, b)"), parse_atom("P(a, b)")) == {"?x": "a"}
    assert unify(parse_atom("P(?x, ?x)"), parse_atom("P(a, b)")) is None
    assert unify(parse_atom("P(a)"), parse_atom("¬P(a)")) is None


def test_rules_file_is_well_formed():
    spec = yaml.safe_load((ROOT / "03_logic" / "rules.yaml").read_text(encoding="utf-8"))
    rules = rules_from_dicts(spec["rules"])
    assert len({r.rule_id for r in rules}) == len(rules)
    for r in rules:                      # range restriction: head vars appear in body
        body_vars = {t for b in r.body for t in b.args if t.startswith("?")}
        assert {t for t in r.head.args if t.startswith("?")} <= body_vars, r.rule_id


def test_proof_cites_provenance(settings):
    kb = _kb(settings, CASES["fol"][0]["facts"])
    res = EntailmentChecker(settings, kb).check("CompromisedHost(srv-db-01)")
    assert any(s.startswith("R_COMP_HOST") for s in res.proof)
    assert any("fact KnownBadIndicator(deadbeef)" in s for s in res.proof)


def test_model_confidence_is_never_proof(settings):
    kb = _kb(settings, ["AlertType(A9, brute_force)", "OnAsset(A9, ws-1)"])
    res = EntailmentChecker(settings, kb).check("CompromisedHost(ws-1)", support=0.99)
    assert res.status.value == "uncertain" and res.sufficiently_supported

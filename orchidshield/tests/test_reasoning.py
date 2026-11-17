"""Unit tests for search, optimisation, adversarial search, CSP, Bayes net and temporal models."""
import random
from pathlib import Path

import numpy as np
import pytest

from reasoning import adversarial, csp, search
from reasoning.bayes import BayesNet, Node
from reasoning.temporal import HMM, kalman_rate
from sim.generator import make_environment

ROOT = Path(__file__).resolve().parents[1]
ENV = make_environment(1337)
ASSETS = ENV["assets"]
CRIT = {a for a, v in ASSETS.items() if v["criticality"] >= 4}


@pytest.fixture(scope="module")
def graph():
    return search.build_graph(ASSETS)


@pytest.mark.parametrize("start", ["ws-003", "ws-008", "web-01", "file-01"])
def test_search_algorithms(graph, start):
    goal = lambda n: n in CRIT and n != start  # noqa: E731
    h = search.hop_heuristic(graph, CRIT - {start})
    ucs, ast, bfs = search.ucs(graph, start, goal), search.astar(graph, start, goal, h), search.bfs(graph, start, goal)
    dfs, ids, gr = search.dfs(graph, start, goal), search.iddfs(graph, start, goal), search.greedy(graph, start, goal, h)
    assert all(r.found for r in (ucs, ast, bfs, dfs, ids, gr))
    assert ast.cost == pytest.approx(ucs.cost)               # A* optimal with admissible h
    assert ast.expanded <= ucs.expanded
    assert min(bfs.cost, dfs.cost, ids.cost, gr.cost) >= ucs.cost - 1e-9
    assert len(bfs.path) <= len(dfs.path)                     # BFS shallowest


def test_heuristic_admissible_and_consistent(graph):
    h = search.hop_heuristic(graph, CRIT)
    for n in graph:
        d = search.dijkstra_all(graph, n)
        true = min((d[g] for g in CRIT if g in d), default=float("inf"))
        assert h(n) <= true + 1e-9
        for m, c in graph[n]:
            assert h(n) <= c + h(m) + 1e-9


def test_blast_radius_bounds(graph):
    crit = {a: v["criticality"] for a, v in ASSETS.items()}
    assert 0 <= search.blast_radius(graph, "ws-001", crit) <= 1
    assert search.blast_radius(graph, "srv-app-03", crit) > search.blast_radius(graph, "ws-013", crit)
    assert search.blast_radius({"lonely": []}, "lonely", crit) == 0


def test_local_search_close_to_optimum():
    p = search.SegmentationProblem(ASSETS, {"ws-004"}, k=2, forbidden={"dc-01", "siem-01", "ws-004"})
    opt = search.exhaustive(p)[1]
    hc = search.hill_climb(p, random.Random(0))[1]
    sa = search.simulated_annealing(p, random.Random(0))[1]
    assert opt <= hc and opt <= sa
    assert hc <= opt * 1.25 + 1 and sa <= opt * 1.5 + 1


def test_alpha_beta_matches_minimax_with_fewer_nodes():
    g = adversarial.ContainmentGame(ASSETS, {"dc-01", "siem-01"})
    mv, mval, mn = g.best_isolation({"ws-004"}, depth=3, method="minimax")
    av, aval, an = g.best_isolation({"ws-004"}, depth=3, method="alphabeta")
    assert aval == pytest.approx(mval) and an < mn
    ev, eval_, _ = g.best_isolation({"ws-004"}, depth=3, method="expectimax")
    assert eval_ >= mval - 1e-9                               # random attacker is never worse for the defender


def test_alpha_beta_never_picks_forbidden():
    g = adversarial.ContainmentGame(ASSETS, {"dc-01", "siem-01"})
    best, _, _ = g.best_isolation({"dc-01"}, depth=4)
    assert best not in {"dc-01", "siem-01"}


def _sched_csp():
    tasks = [csp.Task("triage", "triage"), csp.Task("enrich", "endpoint"),
             csp.Task("isolate", "endpoint", requires_senior=True, after=["enrich"], separate_from=["triage"]),
             csp.Task("disable", "identity", requires_senior=True, after=["triage"]),
             csp.Task("block", "network", requires_senior=True)]
    analysts = [{"id": "ana-1", "skills": ["triage", "identity"], "senior": True},
                {"id": "ana-2", "skills": ["triage", "endpoint"], "senior": False},
                {"id": "ana-3", "skills": ["endpoint", "network", "identity"], "senior": True}]
    return csp.build_schedule_csp(tasks, analysts, [0, 1, 2])


@pytest.mark.parametrize("opts", [{}, {"mrv": True, "lcv": True}, {"mrv": True, "forward_check": True},
                                  {"mrv": True, "lcv": True, "mac": True}])
def test_csp_backtracking_variants(opts):
    c = _sched_csp()
    sol = csp.backtrack(c, **opts)
    assert sol and csp.verify(c, sol)
    assert sol["enrich"][1] < sol["isolate"][1]


def test_csp_min_conflicts_and_ac3():
    c = _sched_csp()
    sol = csp.min_conflicts(c, random.Random(1))
    assert sol and csp.verify(c, sol)
    doms = csp.ac3(_sched_csp())
    assert doms is not None and all(v[1] < 2 for v in doms["enrich"])     # enrich can't be in the last slot


def test_csp_detects_unsatisfiable():
    tasks = [csp.Task("a", "network", requires_senior=True, allowed_slots=[0]),
             csp.Task("b", "network", requires_senior=True, allowed_slots=[0])]
    c = csp.build_schedule_csp(tasks, [{"id": "x", "skills": ["network"], "senior": True}], [0, 1])
    assert csp.backtrack(c, mrv=True, mac=True) is None


@pytest.fixture(scope="module")
def bn():
    return BayesNet.from_json(ROOT / "04_uncertainty" / "bayes_net.json")


def test_bn_prior_and_monotonic_evidence(bn):
    prior = bn.prior("Compromise")
    assert 0.05 < prior < 0.15
    p_ioc = bn.posterior("Compromise", {"IOCMatch": True})
    p_both = bn.posterior("Compromise", {"IOCMatch": True, "EDRDetection": True})
    assert prior < p_ioc < p_both
    assert bn.posterior("Compromise", {"IOCMatch": True, "AnalystFP": True}) < p_ioc


def test_bn_explaining_away(bn):
    a = bn.posterior("Compromise", {"AuthAnomaly": True, "Maintenance": False})
    b = bn.posterior("Compromise", {"AuthAnomaly": True, "Maintenance": True})
    assert b < a


def test_bn_missing_evidence_marginalised(bn):
    assert bn.posterior("Compromise", {"EDRDetection": None}) == pytest.approx(bn.prior("Compromise"))


def test_bn_validation():
    with pytest.raises(ValueError):
        BayesNet([Node("B", ["A"], {"T": 0.5, "F": 0.5}), Node("A", [], {"": 0.5})])
    with pytest.raises(ValueError):
        BayesNet([Node("A", [], {"": 1.5})])


@pytest.fixture(scope="module")
def hmm():
    return HMM.from_json(ROOT / "04_uncertainty" / "hmm.json")


def test_hmm_filter_and_progression(hmm):
    b0 = hmm.filter(["quiet"])
    b1 = hmm.filter(["scan", "auth_anomaly", "malware", "lateral", "exfil"])
    assert np.isclose(b1.sum(), 1)
    assert hmm.expected_severity(b1) > hmm.expected_severity(b0)
    assert hmm.states[int(b1.argmax())] in ("lateral_movement", "exfiltration")


def test_hmm_viterbi_recovers_sampled_states(hmm):
    rng = np.random.default_rng(3)
    acc = []
    for _ in range(30):
        xs, os_ = hmm.sample(25, rng)
        acc.append(np.mean([a == b for a, b in zip(xs, hmm.viterbi(os_))]))
    assert np.mean(acc) > 0.6


def test_kalman_flags_burst():
    steady = kalman_rate([0, 0, 1, 0, 0, 1, 0])
    burst = kalman_rate([0, 0, 1, 0, 0, 1, 7])
    assert burst.max_z > 3 > steady.max_z

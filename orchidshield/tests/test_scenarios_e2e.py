"""End-to-end tests for the six required demonstration scenarios, plus the
Definition-of-Done trace: PERCEPT -> PROPOSITION -> KNOWLEDGE -> INFERENCE -> ENTAILMENT -> ACTION."""
import pytest

from core.schemas import ActionType, CaseResult, ProposalStatus

pytestmark = pytest.mark.e2e


def status(r: CaseResult, q: str) -> str:
    return next(e.status.value for e in r.entailments if e.query == q)


def props(r: CaseResult, action: str, target: str | None = None):
    return [p for p in r.proposals if p.action.value == action and (target is None or p.target == target)]


def test_normal_case_supported_action(run):
    r = run("normal")
    assert status(r, "CompromisedHost(ws-005)") == "entailed"
    iso = props(r, "isolate_host", "ws-005")[0]
    assert iso.status == ProposalStatus.PENDING_APPROVAL and iso.requires_approval
    assert props(r, "open_ticket", "ws-005")[0].status == ProposalStatus.EXECUTED
    assert r.inference.risks[0].asset_id == "ws-005"
    # maintenance scan is recognised as benign
    assert status(r, "TruePositive(A-1003)") == "contradicted"
    assert props(r, "monitor", "srv-app-02")
    # observe-result loop: EDR enrichment fed back as a percept and became knowledge
    assert any(p.kind == "tool_result" for p in r.percepts) and r.iterations >= 2
    assert any("MalwareConfirmed(ws-005)" in s for s in r.knowledge_used)


def test_uncertain_case_abstains(run, settings):
    r = run("uncertain")
    risk = next(x for x in r.inference.risks if x.asset_id == "ws-013")
    assert settings.thresholds.abstain_low <= risk.p_compromise_bayes < settings.thresholds.abstain_high
    assert status(r, "CompromisedHost(ws-013)") == "uncertain"
    assert props(r, "abstain", "ws-013")
    assert not [p for p in r.proposals if p.disruptive]
    assert props(r, "escalate", "ws-013")[0].status == ProposalStatus.EXECUTED
    assert any(s.method == "hmm_filter" for s in r.inference.steps)
    assert r.inference.risks[0].p_true_positive_ml is not None


def test_contradiction_case_flags_and_avoids_entailment(run):
    r = run("contradiction")
    assert any("TruePositive(A-3001)" in c.statement for c in r.contradictions)
    assert status(r, "CompromisedHost(ws-008)") == "contradicted"
    iso = props(r, "isolate_host", "ws-008")[0]
    assert iso.status == ProposalStatus.BLOCKED
    assert props(r, "escalate", "ws-008")[0].status == ProposalStatus.EXECUTED
    assert not [p for p in r.proposals if p.disruptive and p.status == ProposalStatus.PENDING_APPROVAL]


def test_constraint_case_selects_feasible_alternative(run):
    r = run("constraint")
    assert status(r, "CompromisedHost(dc-01)") == "entailed"
    dc = props(r, "isolate_host", "dc-01")[0]
    assert dc.status == ProposalStatus.BLOCKED and any("domain_controller" in x for x in dc.blocked_reasons)
    assert dc.alternatives_considered
    seg = [p for p in props(r, "isolate_host") if p.target != "dc-01"]
    assert seg and seg[0].status == ProposalStatus.PENDING_APPROVAL
    assert props(r, "block_ip", "203.0.113.50")[0].status == ProposalStatus.PENDING_APPROVAL
    svc = props(r, "disable_account", "svc-backup")[0]
    assert svc.status == ProposalStatus.BLOCKED
    assert props(r, "rotate_credentials", "svc-backup")[0].status == ProposalStatus.PENDING_APPROVAL
    pend = [p for p in r.proposals if p.status == ProposalStatus.PENDING_APPROVAL]
    assert all(p.schedule and p.schedule["analyst"] for p in pend)            # CSP scheduled every review
    assert any(s.method == "alpha_beta" for s in r.inference.steps)


def test_adversarial_case_gate_blocks(run, agent):
    r = run("adversarial")
    rej = [p for p in r.percepts if p.validation.value == "rejected"]
    assert len(rej) == 2                                                       # spoofed tool_result + malformed alert
    assert any("possible_prompt_injection" in p.content_flags for p in r.percepts)
    blocked = {(p.tool, p.target) for p in r.proposals if p.status == ProposalStatus.BLOCKED}
    assert {("delete_audit_log", "audit"), ("isolate_host", "dc-01"), ("isolate_host", "siem-01"),
            ("isolate_host", "ws-011")} <= blocked
    assert not [t for t in r.tool_results if t.tool in ("isolate_host", "disable_account", "block_ip")]
    assert any(p.action == ActionType.ESCALATE and "A-5001" in p.target for p in r.proposals)
    events = [e["event_type"] for e in agent.audit.tail(200)]
    assert events.count("action_blocked") >= 4


def test_learning_case_report(settings):
    from reasoning.evaluation import load_learning_report
    rep = load_learning_report(settings.path("ml_dir"))
    assert rep["learning_case"]["improves"] and rep["fairness"]["passes"]


def test_definition_of_done_trace(run):
    """One observation traced through all six stages."""
    r = run("normal")
    p = next(x for x in r.percepts if x.payload.get("alert_id") == "A-1002")
    prop = next(x for x in r.propositions if p.percept_id in x.derived_from and x.atom.predicate == "HasIndicator")
    assert any(prop.atom.args[1] in s for s in r.knowledge_used)               # knowledge update used in proof
    assert any(s.method == "forward_chaining" and "A-1002" in s.subject for s in r.inference.steps)
    ent = next(e for e in r.entailments if e.query == "TruePositive(A-1002)")
    assert ent.status.value == "entailed" and any("R_TP_EXEC_IOC" in s for s in ent.proof)
    assert props(r, "isolate_host", "ws-005")[0].justification == ["ContainmentWarranted(ws-005)"]


def test_runs_are_deterministic(agent):
    from sim.scenarios import load_scenario
    sc = load_scenario("constraint")
    a = agent.run_case(sc["observations"])
    b = agent.run_case(sc["observations"])
    strip = lambda r: [(p.action.value, p.target, p.status.value) for p in r.proposals]  # noqa: E731
    assert strip(a) == strip(b)
    assert [x.risk_score for x in a.inference.risks] == [x.risk_score for x in b.inference.risks]

"""The agent loop:

    observe -> update knowledge -> reason -> entailment check -> plan
            -> approval/policy check -> act -> observe-result (feed back) -> ...

Bounded by `orchestrator.max_iterations`; the kill switch is checked before every
stage and every tool call; every stage transition is written to the hash-chained audit
log; tool calls are idempotent per case."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from core.audit import AuditLog, KillSwitch
from core.config import Settings, get_settings
from core.llm import make_narrator
from core.logging_setup import get_logger
from core.schemas import (ActionProposal, CaseResult, EntailmentResult, EntailmentStatus, InferenceTrace, ModelMetadata,
                          Percept, ProposalStatus, ToolResult, ValidationStatus)
from agent.actions import ToolRegistry
from agent.entailment import EntailmentChecker
from agent.inference import InferenceEngine
from agent.knowledge import KnowledgeBase, is_external_ip
from agent.perceive import Perceiver
from agent.policy import Policy, PolicyContext
from agent.proposition import Propositionalizer
from reasoning.bayes import BayesNet
from reasoning.ml import load_artifact
from reasoning.retrieval import load_playbooks
from reasoning.rl import load_q
from reasoning.temporal import HMM
from sim.generator import load_environment

log = get_logger("orchestrator")
ENRICH_TOOLS = {"edr_process_tree", "query_asset_db", "similar_alerts"}


class Agent:
    def __init__(self, settings: Settings | None = None, environment: dict | None = None):
        self.s = settings or get_settings()
        self.env = environment or load_environment()
        rt = self.s.path("runtime_dir")
        rt.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLog(rt / "audit.jsonl")
        self.kill = KillSwitch(rt)
        self.tools = ToolRegistry(self.s, self.env, self.audit, self.kill)
        unc = self.s.path("uncertainty_dir")
        self.bn = BayesNet.from_json(unc / "bayes_net.json")
        self.hmm = HMM.from_json(unc / "hmm.json")
        self.model, self.model_error = None, None
        try:
            self.model = load_artifact(self.s.path("ml_dir") / "artifacts" / "model.json")
        except Exception as e:  # noqa: BLE001 - model failure -> symbolic fallback
            self.model_error = f"{type(e).__name__}: {e}"
            log.warning(f"ML artifact unavailable, falling back to rules: {self.model_error}")
        try:
            self.q = load_q(self.s.path("ml_dir") / "artifacts" / "q_table.json")
        except Exception:  # noqa: BLE001
            self.q = None
        self.playbooks = load_playbooks(self.s.path("playbooks_dir"))
        self.narrator = make_narrator(self.s.features.llm_narrative, self.s.llm)
        self.cases: dict[str, CaseResult] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ helpers
    def _queries(self, percepts: list[Percept], kb: KnowledgeBase) -> list[str]:
        qs: list[str] = []
        for p in percepts:
            if p.kind != "alert" or p.validation == ValidationStatus.REJECTED:
                continue
            d = p.payload
            qs += [f"TruePositive({d['alert_id']})", f"CompromisedHost({d['asset_id']})",
                   f"ContainmentWarranted({d['asset_id']})"]
            if kb.assets.get(d["asset_id"], {}).get("criticality", 0) >= self.s.thresholds.critical_asset_min:
                qs.append(f"HighPriorityIncident({d['asset_id']})")
            if d.get("user_id"):
                qs += [f"CompromisedAccount({d['user_id']})", f"AccountContainmentWarranted({d['user_id']})"]
            if d.get("src_ip") and is_external_ip(d["src_ip"]):
                qs.append(f"BlockWarranted({d['src_ip']})")
        return list(dict.fromkeys(qs))

    def _halt(self, case_id: str, stage: str) -> str:
        reason = f"kill switch engaged before {stage}"
        self.audit.record("system", "loop_halted", case_id, {"stage": stage, "reason": self.kill.reason})
        return reason

    # ------------------------------------------------------------------ main loop
    def run_case(self, observations: list[dict[str, Any]], scenario: str | None = None,
                 action_requests: list[dict] | None = None, case_id: str | None = None,
                 actor: str = "agent") -> CaseResult:
        case_id = case_id or f"CASE-{uuid.uuid4().hex[:8]}"
        kb = KnowledgeBase(self.s, self.env)
        if self.model:
            md = self.model.metadata
            kb.register_model(ModelMetadata(name=md["name"], version=md["version"], kind=md["kind"],
                                            trained_at=md["trained_at"], metrics=md["metrics"], features=md["features"],
                                            artifact_sha256=md["artifact_sha256"]))
        perceiver, props_builder = Perceiver(self.s), Propositionalizer(self.s, kb)
        engine = InferenceEngine(self.s, kb, self.bn, self.hmm, self.model, self.playbooks, self.q)
        checker = EntailmentChecker(self.s, kb)
        policy = Policy(self.s, kb, kill_engaged=lambda: self.kill.engaged)
        self.audit.record(actor, "case_started", case_id, {"scenario": scenario, "observations": len(observations)})

        percepts: list[Percept] = []
        pending = list(observations)
        enriched: set[str] = set()
        tool_results: list[ToolResult] = []
        proposals: dict[str, ActionProposal] = {}
        entailments: list[EntailmentResult] = []
        trace = InferenceTrace(trace_id="T-none", steps=[], derived_facts=[], risks=[], incidents=[])
        extras = None
        prop_set = None
        halted, halt_reason, iterations = False, None, 0
        requests = list(action_requests or [])

        for it in range(self.s.orchestrator.max_iterations):
            iterations = it + 1
            if self.kill.engaged:
                halted, halt_reason = True, self._halt(case_id, "observe"); break
            # 1. PERCEPTS
            new = perceiver.perceive(pending, received_at=datetime.now(timezone.utc))
            known = {p.percept_id for p in percepts}
            new = [p for p in new if p.percept_id not in known]
            percepts += new
            rejected = [p for p in new if p.validation == ValidationStatus.REJECTED]
            self.audit.record("agent", "percepts", case_id, {"iteration": iterations, "new": len(new),
                                                             "rejected": [(p.percept_id, p.issues[:2]) for p in rejected]})
            # 2. PROPOSITIONS + 3. KNOWLEDGE update
            if self.kill.engaged:
                halted, halt_reason = True, self._halt(case_id, "knowledge update"); break
            prop_set = props_builder.build(percepts)
            kb.add_propositions(prop_set.propositions)
            # 4. INFERENCE
            if self.kill.engaged:
                halted, halt_reason = True, self._halt(case_id, "inference"); break
            trace, extras = engine.run(prop_set, percepts, enriched)
            # 5. ENTAILMENT
            post = extras.posteriors
            entailments = []
            for q in self._queries(percepts, kb):
                arg = q[q.index("(") + 1:-1]
                entailments.append(checker.check(q, post.get(arg)))
            self.audit.record("agent", "entailment", case_id, {"iteration": iterations, "results": {
                e.query: e.status.value for e in entailments}})
            # 6. ACTION: plan -> policy/approval gate -> act
            if self.kill.engaged:
                halted, halt_reason = True, self._halt(case_id, "action"); break
            injection = [p.payload["alert_id"] for p in percepts if "possible_prompt_injection" in p.content_flags
                         and p.kind == "alert"]
            ctx = PolicyContext(case_id=case_id, entailments={e.query: e for e in entailments}, trace=trace,
                                extras=extras, enriched=enriched, action_requests=requests,
                                injection_alerts=injection,
                                contradictions=[c.statement for c in prop_set.contradictions])
            requests = []      # external requests are gated once
            planned = policy.plan(ctx)
            feedback: list[dict] = []
            for p in planned:
                prev = proposals.get(p.idempotency_key)
                if prev and prev.status in (ProposalStatus.EXECUTED, ProposalStatus.FAILED):
                    continue
                if p.status == ProposalStatus.BLOCKED:
                    self.audit.record("policy", "action_blocked", case_id, {
                        "action": p.action.value, "tool": p.tool, "target": p.target, "reasons": p.blocked_reasons})
                elif p.status == ProposalStatus.PENDING_APPROVAL and not prev:
                    self.audit.record("policy", "approval_requested", case_id, {
                        "proposal_id": p.proposal_id, "action": p.action.value, "target": p.target})
                if p.status == ProposalStatus.AUTO_APPROVED:
                    res = self.tools.execute(p)
                    tool_results.append(res)
                    p = p.model_copy(update={"status": ProposalStatus.EXECUTED if res.ok else ProposalStatus.FAILED})
                    if p.tool in ENRICH_TOOLS:
                        enriched.add(p.target.split(":")[0])
                        if res.ok:
                            feedback.append({"kind": "tool_result", "source": "tool", "data": {
                                "tool": p.tool, "target": p.target.split(":")[0],
                                "timestamp": datetime.now(timezone.utc).isoformat(), "output": res.output}})
                proposals[p.idempotency_key] = p
            if not feedback:
                break
            pending = feedback          # observe-result: tool outputs become new percepts
        else:
            self.audit.record("system", "iteration_limit", case_id, {"max": self.s.orchestrator.max_iterations})

        knowledge_used: set[str] = set()
        for e in entailments:
            if e.status in (EntailmentStatus.ENTAILED, EntailmentStatus.CONTRADICTED):
                knowledge_used |= {s.split(":")[0] for s in e.proof + e.counter_proof if not s.startswith("fact")}
                knowledge_used |= {s.split("  [")[0][5:] for s in e.proof + e.counter_proof if s.startswith("fact")}
        if self.model:
            knowledge_used.add(f"model:{self.model.metadata['name']}@{self.model.metadata['version']}")
        else:
            knowledge_used.add(f"model:unavailable ({self.model_error})")
        knowledge_used |= {f"playbook:{p}" for p in trace.retrieved_playbooks}

        props_list = policy.schedule(list(proposals.values()))
        evidence = {"top_risks": [r.model_dump() for r in trace.risks[:3]],
                    "entailments": [{"query": e.query, "status": e.status.value} for e in entailments],
                    "actions": [{"action": p.action.value, "target": p.target, "status": p.status.value} for p in props_list],
                    "contradictions": [c.statement for c in (prop_set.contradictions if prop_set else [])]}
        summary = "Loop halted by kill switch." if halted else self.narrator.summarize(evidence)
        result = CaseResult(
            case_id=case_id, scenario=scenario, iterations=iterations, halted=halted, halt_reason=halt_reason,
            percepts=percepts, propositions=prop_set.propositions if prop_set else [],
            contradictions=(prop_set.contradictions if prop_set else []) + [
                c for c in self._kb_conflicts(kb)],
            knowledge_used=sorted(knowledge_used), inference=trace, entailments=entailments, proposals=props_list,
            tool_results=tool_results, rl_advice=extras.rl_advice if extras else {}, summary=summary,
            run_metadata={**self.s.run_metadata(), "narrator": self.narrator.name, "kb": kb.stats(),
                          "model": self.model.metadata["name"] if self.model else None})
        with self._lock:
            self.cases[case_id] = result
        self.audit.record("agent", "case_completed", case_id, {
            "iterations": iterations, "halted": halted,
            "pending_approval": [p.proposal_id for p in props_list if p.status == ProposalStatus.PENDING_APPROVAL]})
        return result

    @staticmethod
    def _kb_conflicts(kb: KnowledgeBase):
        from core.schemas import Contradiction
        return [Contradiction(statement=str(pos), supporting=kb.proof_tree(pos)[-1:], opposing=kb.proof_tree(neg)[-1:],
                              description="derived conflict in knowledge-base closure") for pos, neg in kb.conflicts()]

    # ------------------------------------------------------------------ approvals
    def pending_approvals(self) -> list[ActionProposal]:
        with self._lock:
            return [p for c in self.cases.values() for p in c.proposals if p.status == ProposalStatus.PENDING_APPROVAL]

    def decide(self, proposal_id: str, approve: bool, approver: str, note: str = "") -> tuple[ActionProposal, ToolResult | None]:
        with self._lock:
            for case in self.cases.values():
                for i, p in enumerate(case.proposals):
                    if p.proposal_id != proposal_id:
                        continue
                    if p.status != ProposalStatus.PENDING_APPROVAL:
                        raise ValueError(f"proposal is {p.status.value}, not pending")
                    self.audit.record(approver, "approval_decision", case.case_id, {
                        "proposal_id": proposal_id, "approve": approve, "note": note[:300]})
                    if not approve:
                        p = p.model_copy(update={"status": ProposalStatus.REJECTED})
                        case.proposals[i] = p
                        return p, None
                    p = p.model_copy(update={"status": ProposalStatus.APPROVED})
                    res = self.tools.execute(p, actor=approver)
                    p = p.model_copy(update={"status": ProposalStatus.EXECUTED if res.ok else ProposalStatus.FAILED})
                    case.proposals[i] = p
                    case.tool_results.append(res)
                    # observe-result: feed the containment outcome back as a percept on the case
                    case.percepts.append(Perceiver(self.s).perceive_one({"kind": "tool_result", "source": "tool", "data": {
                        "tool": p.tool, "target": p.target, "timestamp": datetime.now(timezone.utc).isoformat(),
                        "output": res.output if res.ok else {"error": res.error}}}))
                    return p, res
        raise KeyError(proposal_id)

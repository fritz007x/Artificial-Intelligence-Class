"""Stage 6 (decision half) - POLICY. Turns entailment results and risk into bounded
action proposals using goals, hard constraints, utility and the approval policy.

Decision contract
  * Disruptive actions (isolate/disable/block/rotate) are only *proposed* when their
    justification is ENTAILED; they are always PENDING_APPROVAL, never auto-executed.
  * CONTRADICTED conclusions -> escalate to a human; containment is blocked.
  * UNCERTAIN conclusions -> calibrated abstention: enrich, watchlist, escalate.
  * Hard constraints (protected assets, never-isolate types, never-disable account
    types, internal IPs, isolation budget, unknown targets) block a proposal and trigger
    a search for a feasible alternative.
  * Accepted response tasks are scheduled to analysts/slots by the CSP solver."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from core.config import Settings
from core.schemas import (ActionProposal, ActionType, EntailmentResult, EntailmentStatus, InferenceTrace,
                          ProposalStatus)
from agent.inference import InferenceExtras
from agent.knowledge import KnowledgeBase, is_external_ip
from reasoning.csp import Task, backtrack, build_schedule_csp

DISRUPTIVE = {ActionType.ISOLATE_HOST, ActionType.DISABLE_ACCOUNT, ActionType.BLOCK_IP, ActionType.ROTATE_CREDENTIALS}
TOOL_FOR = {ActionType.ISOLATE_HOST: "isolate_host", ActionType.DISABLE_ACCOUNT: "disable_account",
            ActionType.BLOCK_IP: "block_ip", ActionType.ROTATE_CREDENTIALS: "rotate_credentials",
            ActionType.OPEN_TICKET: "create_ticket", ActionType.ADD_WATCHLIST: "add_watchlist",
            ActionType.NOTIFY_ANALYST: "notify_analyst", ActionType.ESCALATE: "notify_analyst"}


@dataclass
class PolicyContext:
    case_id: str
    entailments: dict[str, EntailmentResult]
    trace: InferenceTrace
    extras: InferenceExtras
    enriched: set[str] = field(default_factory=set)
    action_requests: list[dict] = field(default_factory=list)
    injection_alerts: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)


class Policy:
    def __init__(self, settings: Settings, kb: KnowledgeBase, kill_engaged=lambda: False):
        self.s, self.kb, self.kill_engaged = settings, kb, kill_engaged

    # ------------------------------------------------------------------ helpers
    def _utility(self, action: ActionType, p: float, impact: float) -> float:
        a = action.value
        return round(self.s.policy.action_benefit.get(a, 0) * p * impact * 5 - self.s.policy.action_costs.get(a, 0), 3)

    def hard_constraint_violations(self, action: ActionType | str, target: str, n_isolations: int = 0) -> list[str]:
        action = ActionType(action) if isinstance(action, str) and action in ActionType._value2member_map_ else action
        v: list[str] = []
        pol = self.s.policy
        if self.kill_engaged():
            v.append("kill switch engaged")
        if target in pol.protected_assets:
            v.append(f"{target} is a protected asset (automated actions would blind the SOC)")
        if action == ActionType.ISOLATE_HOST:
            asset = self.kb.assets.get(target)
            if not asset:
                v.append(f"{target} not in asset inventory (refusing to act on unverified target)")
            elif asset["asset_type"] in pol.never_isolate_asset_types:
                v.append(f"hard constraint: never isolate a {asset['asset_type']} ({target})")
            if n_isolations >= pol.max_isolations_per_case:
                v.append(f"isolation budget of {pol.max_isolations_per_case} per case exhausted")
        if action in (ActionType.DISABLE_ACCOUNT, ActionType.ROTATE_CREDENTIALS):
            user = self.kb.users.get(target)
            if not user:
                v.append(f"{target} not in directory")
            elif action == ActionType.DISABLE_ACCOUNT and user["account_type"] in pol.never_disable_account_types:
                v.append(f"hard constraint: never disable a {user['account_type']} account ({target})")
        if action == ActionType.BLOCK_IP and not is_external_ip(target):
            v.append(f"{target} is not an external IP; perimeter blocks are external-only")
        return v

    def _mk(self, ctx: PolicyContext, action: ActionType, target: str, justification: list[str],
            ent: EntailmentResult | None, p: float, impact: float, params: dict | None = None,
            tool: str | None = None, status: ProposalStatus | None = None, alternatives: list[str] | None = None
            ) -> ActionProposal:
        key = hashlib.sha1(f"{ctx.case_id}|{action.value}|{target}".encode()).hexdigest()[:16]
        disruptive = action in DISRUPTIVE
        tool = tool if tool is not None else TOOL_FOR.get(action)
        if status is None:
            status = ProposalStatus.PENDING_APPROVAL if disruptive else (
                ProposalStatus.AUTO_APPROVED if tool else ProposalStatus.RECOMMENDED)
        return ActionProposal(proposal_id=f"AP-{key[:10]}", case_id=ctx.case_id, action=action, tool=tool, target=target,
                              params=params or {}, justification=justification,
                              entailment_status=ent.status if ent else None, utility=self._utility(action, p, impact),
                              confidence=round(min(1.0, max(0.0, p)), 3), disruptive=disruptive,
                              requires_approval=disruptive, status=status, alternatives_considered=alternatives or [],
                              idempotency_key=key)

    def _gate(self, prop: ActionProposal, n_iso: int) -> ActionProposal:
        reasons = self.hard_constraint_violations(prop.action, prop.target, n_iso) if prop.disruptive else (
            ["kill switch engaged"] if self.kill_engaged() else [])
        if prop.disruptive and self.s.policy.disruptive_requires_entailment and \
                prop.entailment_status != EntailmentStatus.ENTAILED:
            reasons.append(f"justification is {prop.entailment_status.value if prop.entailment_status else 'absent'}, "
                           "disruptive actions require ENTAILED")
        if reasons:
            return prop.model_copy(update={"status": ProposalStatus.BLOCKED, "blocked_reasons": reasons})
        return prop

    # ------------------------------------------------------------------ planning
    def plan(self, ctx: PolicyContext) -> list[ActionProposal]:
        E = ctx.entailments
        out: list[ActionProposal] = []
        n_iso = 0

        def add(p: ActionProposal) -> ActionProposal:
            nonlocal n_iso
            g = self._gate(p, n_iso)
            if g.action == ActionType.ISOLATE_HOST and g.status == ProposalStatus.PENDING_APPROVAL:
                n_iso += 1
            if all(x.idempotency_key != g.idempotency_key for x in out):
                out.append(g)
            return g

        for r in ctx.trace.risks:
            h, p = r.asset_id, r.p_compromise_bayes
            impact = max(r.criticality / 5, r.blast_radius)
            host_q = f"CompromisedHost({h})"
            ent = E.get(host_q)
            asset = self.kb.assets.get(h, {})
            if ent is None:
                continue
            if ent.status == EntailmentStatus.CONTRADICTED:
                add(self._mk(ctx, ActionType.ESCALATE, h, [host_q, *ent.conflicts], ent, max(p, 0.5), impact,
                             {"message": f"[{ctx.case_id}] Conflicting evidence on {h}: {', '.join(ent.conflicts)}. "
                                         "Human adjudication required before any containment.", "severity": "critical"}))
                add(self._mk(ctx, ActionType.OPEN_TICKET, h, [host_q], ent, p, impact,
                             self._ticket(ctx, f"Conflict on {h}", ent.rationale, "P2")))
                blocked = self._mk(ctx, ActionType.ISOLATE_HOST, h, [host_q], ent, p, impact, {"asset_id": h})
                add(blocked)
                continue
            if ent.status == EntailmentStatus.ENTAILED:
                if h not in ctx.enriched and asset.get("edr_installed", True):
                    add(self._mk(ctx, ActionType.ENRICH, h, [host_q], ent, p, impact, {"asset_id": h},
                                 tool="edr_process_tree"))
                iso = add(self._mk(ctx, ActionType.ISOLATE_HOST, h, [f"ContainmentWarranted({h})"],
                                   E.get(f"ContainmentWarranted({h})", ent), p, impact, {"asset_id": h}))
                if iso.status == ProposalStatus.BLOCKED:
                    alts = self._alternatives_for_host(ctx, h, p, impact, add)
                    out[out.index(iso)] = iso.model_copy(update={"alternatives_considered": alts})
                add(self._mk(ctx, ActionType.OPEN_TICKET, h, [host_q], ent, p, impact,
                             self._ticket(ctx, f"Compromise of {h}", "\n".join(ent.proof[-4:]),
                                          "P1" if r.criticality >= 4 else "P2")))
                add(self._mk(ctx, ActionType.NOTIFY_ANALYST, h, [host_q], ent, p, impact,
                             {"message": f"[{ctx.case_id}] {h} compromise entailed; containment awaiting approval.",
                              "severity": "critical" if r.criticality >= 4 else "warning"}))
                continue
            # UNCERTAIN / UNKNOWN
            benign = all(self.kb.holds(f"¬TruePositive({a})") for a in r.alerts)
            if benign:
                add(self._mk(ctx, ActionType.MONITOR, h, [f"¬TruePositive({a})" for a in r.alerts], ent, p, impact))
                continue
            low, high = self.s.thresholds.abstain_low, self.s.thresholds.abstain_high
            if ent.status == EntailmentStatus.UNCERTAIN or low <= p:
                add(self._mk(ctx, ActionType.ABSTAIN, h, [host_q], ent, p, impact,
                             {"reason": f"P(Compromise)={p:.2f}; not entailed - abstaining from a verdict "
                                        f"(band {low}-{high}) and gathering evidence"}))
                if h not in ctx.enriched:
                    if asset.get("edr_installed", True):
                        add(self._mk(ctx, ActionType.ENRICH, h, [host_q], ent, p, impact, {"asset_id": h},
                                     tool="edr_process_tree"))
                    else:
                        add(self._mk(ctx, ActionType.ENRICH, h, [host_q], ent, p, impact, {"asset_id": h},
                                     tool="query_asset_db"))
                        add(self._mk(ctx, ActionType.ENRICH, f"{h}:history", [host_q], ent, p, impact,
                                     {"text": " ".join(a.replace("-", " ") for a in r.alerts) + " " + r.hmm_stage},
                                     tool="similar_alerts"))
                add(self._mk(ctx, ActionType.ADD_WATCHLIST, h, [host_q], ent, p, impact,
                             {"entity": h, "reason": f"uncertain compromise (P={p:.2f})"}))
                add(self._mk(ctx, ActionType.ESCALATE, h, [host_q], ent, p, impact,
                             {"message": f"[{ctx.case_id}] Uncertain activity on {h} (P={p:.2f}); agent abstained. "
                                         "Please review.", "severity": "warning"}))
            else:
                add(self._mk(ctx, ActionType.MONITOR, h, [host_q], ent, p, impact))

        # accounts
        for q, ent in E.items():
            if q.startswith("CompromisedAccount(") and ent.status == EntailmentStatus.ENTAILED:
                u = q[len("CompromisedAccount("):-1]
                p = max(ctx.extras.posteriors.values(), default=0.5)
                d = add(self._mk(ctx, ActionType.DISABLE_ACCOUNT, u, [f"AccountContainmentWarranted({u})"],
                                 E.get(f"AccountContainmentWarranted({u})", ent), p, 0.8, {"user_id": u}))
                if d.status == ProposalStatus.BLOCKED:
                    alt = add(self._mk(ctx, ActionType.ROTATE_CREDENTIALS, u, [f"AccountContainmentWarranted({u})"],
                                       E.get(f"AccountContainmentWarranted({u})", ent), p, 0.8, {"user_id": u}))
                    out[out.index(d)] = d.model_copy(update={"alternatives_considered": [
                        f"rotate_credentials({u}) -> {alt.status.value}"]})
            if q.startswith("BlockWarranted(") and ent.status == EntailmentStatus.ENTAILED:
                ip = q[len("BlockWarranted("):-1]
                add(self._mk(ctx, ActionType.BLOCK_IP, ip, [q], ent, 0.9, 0.6, {"ip": ip}))

        # injected content & external (red-team / integration) action requests
        for a in ctx.injection_alerts:
            add(self._mk(ctx, ActionType.ESCALATE, f"{a}:content", [f"InjectionAttempt({a})"], None, 0.9, 0.6,
                         {"message": f"[{ctx.case_id}] Alert {a} contained instruction-like text; quarantined. "
                                     "Security review requested.", "severity": "warning"}))
        for req in ctx.action_requests:
            out.append(self.gate_external_request(ctx, req, n_iso))
        if ctx.contradictions and not any(x.action == ActionType.ESCALATE for x in out):
            add(self._mk(ctx, ActionType.ESCALATE, "kb", ctx.contradictions[:3], None, 0.5, 0.5,
                         {"message": f"[{ctx.case_id}] Percept contradictions: {'; '.join(ctx.contradictions[:3])}",
                          "severity": "warning"}))
        return out

    def _alternatives_for_host(self, ctx, h, p, impact, add) -> list[str]:
        """Feasible alternatives when isolating `h` violates a hard constraint."""
        alts = []
        point = ctx.extras.containment_point.get(h)
        if point and point != h:
            seg = add(self._mk(ctx, ActionType.ISOLATE_HOST, point,
                               [f"ContainmentWarranted({h})", f"segmentation point chosen by alpha-beta for {h}"],
                               ctx.entailments.get(f"ContainmentWarranted({h})"), p, impact, {"asset_id": point}))
            alts.append(f"isolate_host({point}) [segmentation] -> {seg.status.value}")
        for q, ent in ctx.entailments.items():
            if q.startswith("BlockWarranted(") and ent.status == EntailmentStatus.ENTAILED:
                alts.append(f"block_ip({q[15:-1]})")
            if q.startswith("CompromisedAccount(") and ent.status == EntailmentStatus.ENTAILED:
                alts.append(f"disable/rotate account {q[19:-1]}")
        return alts or ["escalate to incident commander"]

    def gate_external_request(self, ctx: PolicyContext, req: dict, n_iso: int = 0) -> ActionProposal:
        tool, target = str(req.get("tool", "")), str(req.get("target", ""))
        action = next((a for a, t in TOOL_FOR.items() if t == tool and a in DISRUPTIVE), None)
        just = [f"external request by {req.get('requested_by', 'unknown')}"]
        if action is None and tool not in ("create_ticket", "add_watchlist", "notify_analyst", "edr_process_tree",
                                           "query_asset_db", "similar_alerts"):
            p = self._mk(ctx, ActionType.ABSTAIN, target, just, None, 0.0, 0.0, {"requested_tool": tool}, tool=tool,
                         status=ProposalStatus.BLOCKED)
            return p.model_copy(update={"blocked_reasons": [f"tool '{tool}' is not allow-listed"]})
        action = action or ActionType.ENRICH
        ent = ctx.entailments.get(f"ContainmentWarranted({target})")
        prop = self._mk(ctx, action, target, just, ent, 0.0, 0.0,
                        {"asset_id": target} if action == ActionType.ISOLATE_HOST else {"target": target}, tool=tool)
        prop = self._gate(prop, n_iso)
        if prop.status != ProposalStatus.BLOCKED and prop.disruptive:
            prop = prop.model_copy(update={"status": ProposalStatus.PENDING_APPROVAL})
        return prop

    def _ticket(self, ctx, title, body, prio) -> dict:
        return {"case_id": ctx.case_id, "title": title[:200], "body": body[:4000], "priority": prio}

    # ------------------------------------------------------------------ CSP scheduling
    def schedule(self, props: list[ActionProposal]) -> list[ActionProposal]:
        """Schedule the *human* response tasks (approval reviews of disruptive actions and
        escalations) onto analysts and 30-minute slots with the CSP solver. Auto-executed
        tool calls need no analyst and are not scheduled."""
        skill = {ActionType.ISOLATE_HOST: "endpoint", ActionType.BLOCK_IP: "network",
                 ActionType.DISABLE_ACCOUNT: "identity", ActionType.ROTATE_CREDENTIALS: "identity",
                 ActionType.ESCALATE: "triage"}
        live = [p for p in props if p.action in skill and p.status in (ProposalStatus.PENDING_APPROVAL,
                                                                       ProposalStatus.EXECUTED,
                                                                       ProposalStatus.AUTO_APPROVED)]
        if not live:
            return props
        escalations = [p for p in live if p.action == ActionType.ESCALATE]
        tasks = []
        for p in live:
            after = [q.proposal_id for q in escalations if p.disruptive and q.target.split(":")[0] == p.target]
            tasks.append(Task(p.proposal_id, skill[p.action], requires_senior=p.disruptive, after=after,
                              separate_from=[q.proposal_id for q in escalations[:1]] if p.disruptive else []))
        csp = build_schedule_csp(tasks, [a.model_dump() for a in self.s.csp.analysts], self.s.csp.slots)
        sol = backtrack(csp, mrv=True, lcv=True, forward_check=True)
        out = []
        for p in props:
            if sol and p.proposal_id in sol:
                a, slot = sol[p.proposal_id]
                p = p.model_copy(update={"schedule": {"analyst": a, "slot": slot, "task": "review/approve" if
                                                      p.disruptive else "triage", "solver": "backtracking+MRV+LCV+FC"}})
            elif sol is None and any(t.task_id == p.proposal_id for t in tasks):
                p = p.model_copy(update={"schedule": {"analyst": None, "slot": None,
                                                      "solver": "unsatisfiable - escalate staffing"}})
            out.append(p)
        return out

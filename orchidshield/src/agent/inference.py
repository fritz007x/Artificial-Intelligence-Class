"""Stage 4 - INFERENCE. Logical (forward chaining), probabilistic (Bayesian network),
temporal (HMM + Kalman), learned (ML/NN), search (A*, blast radius), adversarial
(alpha-beta containment point), retrieval (playbook RAG) and RL advice, fused into a
ranked risk list and incident hypotheses. Every method appends a concise, inspectable
InferenceStep - an evidence record, not hidden chain-of-thought."""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

from core.config import Settings
from core.schemas import Atom, Incident, InferenceStep, InferenceTrace, Percept, PropositionSet, RiskAssessment, ValidationStatus
from agent.knowledge import KnowledgeBase
from reasoning.adversarial import ContainmentGame
from reasoning.bayes import BayesNet
from reasoning.ml import LoadedModel, to_matrix
from reasoning.retrieval import TfidfIndex
from reasoning.rl import greedy_policy, state_from_case
from reasoning.search import astar, blast_radius, build_graph, hop_heuristic
from reasoning.temporal import HMM, kalman_rate

INTERESTING = {"IOCMatch", "TruePositive", "CompromisedHost", "CompromisedAccount", "AtRisk", "HighPriorityIncident",
               "CrownJewelExposed", "ContainmentWarranted", "AccountContainmentWarranted", "BlockWarranted"}


@dataclass
class InferenceExtras:
    posteriors: dict[str, float] = field(default_factory=dict)
    bn_evidence: dict[str, dict] = field(default_factory=dict)
    containment_point: dict[str, str | None] = field(default_factory=dict)
    rl_advice: dict[str, str] = field(default_factory=dict)
    alerts_by_asset: dict[str, list[dict]] = field(default_factory=dict)


class InferenceEngine:
    def __init__(self, settings: Settings, kb: KnowledgeBase, bn: BayesNet, hmm: HMM,
                 model: LoadedModel | None, playbooks: TfidfIndex | None, q_table: dict | None):
        self.s, self.kb, self.bn, self.hmm, self.model = settings, kb, bn, hmm, model
        self.playbooks, self.q = playbooks, q_table
        self._n = 0

    def _step(self, steps, method, subject, inputs, output, rationale, value=None):
        self._n += 1
        steps.append(InferenceStep(step_id=f"S{self._n:03d}", method=method, subject=subject, inputs=inputs[:8],
                                   output=output, value=None if value is None else round(float(value), 4),
                                   rationale=rationale))

    def run(self, props: PropositionSet, percepts: list[Percept], enriched: set[str] | None = None
            ) -> tuple[InferenceTrace, InferenceExtras]:
        enriched = enriched or set()
        steps: list[InferenceStep] = []
        ex = InferenceExtras()
        kb = self.kb

        # ---- 1. logical: forward chaining to fixpoint
        derived = kb.derived_facts()
        for atom in derived:
            if atom.predicate in INTERESTING:
                j = kb.closure().just[atom.key]
                prem = [str(kb.closure().facts[p]) for p in j.premises]
                self._step(steps, "forward_chaining", str(atom), prem, str(atom),
                           f"{j.rule_id}: {kb.rule_meta[j.rule_id]['description']}")
        for pos, negf in kb.conflicts():
            self._step(steps, "forward_chaining", str(pos), [str(pos), str(negf)], "CONFLICT",
                       "closure contains a literal and its negation; conclusions resting on it cannot be entailed")

        # ---- gather alerts per asset
        alerts = [p for p in percepts if p.kind == "alert" and p.validation != ValidationStatus.REJECTED]
        by_asset: dict[str, list[dict]] = defaultdict(list)
        for p in alerts:
            by_asset[p.payload["asset_id"]].append({**p.payload, "_ts": p.timestamp, "_source": p.source})
        ex.alerts_by_asset = by_asset
        crit = {a: v["criticality"] for a, v in kb.assets.items()}
        graph = build_graph(kb.assets)
        crown = {a for a, c in crit.items() if c >= self.s.thresholds.critical_asset_min}
        h_fn = hop_heuristic(graph, crown)

        # ---- 2. learned model scores per alert
        ml_p: dict[str, float] = {}
        if self.model is not None and self.s.features.ml_scoring:
            for aid, feats in props.features.items():
                ml_p[aid] = float(self.model.model.predict_proba(to_matrix([feats]))[0])
            for aid, p in ml_p.items():
                self._step(steps, "ml_model", aid, [f"{k}={v:.2f}" for k, v in props.features[aid].items() if v],
                           f"P(true positive)={p:.2f}",
                           f"{self.model.metadata['name']} v{self.model.metadata['version']} "
                           f"(learned estimate; never treated as proof)", p)
        elif props.features:
            self._step(steps, "fallback", "ml_model", [], "rule-based scoring only",
                       "learned model unavailable or disabled; risk fusion re-weights onto symbolic/probabilistic evidence")

        risks: list[RiskAssessment] = []
        for h, alist in by_asset.items():
            alist.sort(key=lambda x: x["_ts"])
            ev = self._bn_evidence(h, alist)
            ex.bn_evidence[h] = ev
            post = self.bn.posterior(self.bn.query, ev)
            ex.posteriors[h] = post
            impact = self.bn.explain(self.bn.query, ev)
            self._step(steps, "bayesian_network", h, [f"{k}={v}" for k, v in ev.items()], f"P(Compromise)={post:.2f}",
                       "exact enumeration; unobserved evidence marginalised; top drivers: "
                       + ", ".join(f"{k} ({d:+.2f})" for k, d in impact[:3]), post)

            obs = self._hmm_sequence(alist)
            belief = self.hmm.filter(obs)
            stage_i = int(belief.argmax())
            stage = self.hmm.states[stage_i]
            sev = self.hmm.expected_severity(belief)
            self._step(steps, "hmm_filter", h, obs, f"stage={stage} ({belief[stage_i]:.2f})",
                       f"forward filtering over {len(obs)} observations; expected stage severity {sev:.2f}", sev)

            counts = self._bin_counts(alist)
            kal = kalman_rate(counts)
            self._step(steps, "kalman_filter", h, [str(c) for c in counts], f"max innovation z={kal.max_z:.2f}",
                       "local-level filter on alert rate; z above "
                       f"{self.s.thresholds.kalman_anomaly_sigma} flags an unexplained burst", kal.max_z)

            br = blast_radius(graph, h, crit)
            if h in graph and h not in crown:
                res = astar(graph, h, lambda n: n in crown, h_fn)
                self._step(steps, "astar_search", h, [h], f"path to crown jewel: {' → '.join(res.path or [])}",
                           f"A* (admissible hop heuristic) cost {res.cost:.1f}, expanded {res.expanded}; "
                           f"blast radius {br:.2f}", br)
            p_ml = max((ml_p[a["alert_id"]] for a in alist if a["alert_id"] in ml_p), default=None)
            w = self.s.risk_weights
            parts = {"bayes": post, "temporal": sev, "criticality": crit.get(h, 1) / 5, "blast_radius": br}
            if p_ml is not None:
                parts["ml"] = p_ml
            tot_w = sum(w[k] for k in parts)
            risk = sum(w[k] * v for k, v in parts.items()) / tot_w
            if kal.max_z > self.s.thresholds.kalman_anomaly_sigma:
                risk = min(1.0, risk + 0.05)
            self._step(steps, "risk_fusion", h, [f"{k}={v:.2f}" for k, v in parts.items()], f"risk={risk:.2f}",
                       "weighted fusion (configs/config.yaml risk_weights); +0.05 if Kalman burst", risk)
            risks.append(RiskAssessment(asset_id=h, p_compromise_bayes=round(post, 4), p_true_positive_ml=
                                        None if p_ml is None else round(p_ml, 4), hmm_stage=stage,
                                        hmm_stage_prob=round(float(belief[stage_i]), 4), kalman_z=round(kal.max_z, 3),
                                        blast_radius=round(br, 4), criticality=crit.get(h, 1), risk_score=round(risk, 4),
                                        alerts=[a["alert_id"] for a in alist]))
            if self.s.features.rl_advisor and self.q:
                st = state_from_case(post, sev, crit.get(h, 1) >= self.s.thresholds.critical_asset_min, h in enriched)
                adv = greedy_policy(self.q)(st)
                ex.rl_advice[h] = adv
                self._step(steps, "rl_advisor", h, [f"state={st}"], f"advice={adv}",
                           "tabular Q-learning policy (advisory only; cannot execute)")
        risks.sort(key=lambda r: -r.risk_score)
        for i, r in enumerate(risks, 1):
            r.rank = i

        # ---- 3. adversarial containment point
        compromised = {a.args[0] for a in kb.query("CompromisedHost(?h)")}
        if compromised and self.s.features.adversarial_planner:
            forbidden = set(self.s.policy.protected_assets) | {
                a for a, v in kb.assets.items() if v["asset_type"] in self.s.policy.never_isolate_asset_types}
            game = ContainmentGame(kb.assets, forbidden)
            best, val, nodes = game.best_isolation(compromised, depth=4, method="alphabeta")
            for h in compromised:
                ex.containment_point[h] = best
            self._step(steps, "alpha_beta", ",".join(sorted(compromised)), sorted(compromised),
                       f"isolate {best}" if best else "no isolation improves outcome",
                       f"defender-vs-attacker game, depth 4, {nodes} nodes; value {val:.1f}", val)

        # ---- 4. incidents
        incidents = self._incidents(alerts, risks, kb)
        for inc in incidents:
            self._step(steps, "incident_clustering", inc.incident_id, inc.alerts, f"{inc.incident_type} on {inc.assets}",
                       "alerts linked by shared asset or user (union-find)", inc.confidence)

        # ---- 5. retrieval-augmented context from playbooks
        retrieved: list[str] = []
        if self.playbooks:
            q = " ".join({a["alert_type"].replace("_", " ") for al in by_asset.values() for a in al} |
                         {d.predicate for d in derived if d.predicate in INTERESTING})
            for doc_id, score in self.playbooks.query(q, k=3):
                retrieved.append(doc_id)
                self._step(steps, "rag_retrieval", doc_id, [q[:80]], doc_id, "TF-IDF cosine over SOC playbooks", score)

        return InferenceTrace(trace_id=f"T-{uuid.uuid4().hex[:8]}", steps=steps,
                              derived_facts=[str(a) for a in derived if a.predicate != "SubtypeOf"],
                              risks=risks, incidents=incidents, retrieved_playbooks=retrieved), ex

    # ------------------------------------------------------------------ helpers
    def _bn_evidence(self, h: str, alist: list[dict]) -> dict[str, bool | None]:
        kb = self.kb
        ioc = None
        if any(kb.holds(Atom(predicate="IOCMatch", args=(a["alert_id"],))) for a in alist):
            ioc = True
        elif any(a.get("indicators") for a in alist):
            # indicators present but none known-bad: clean only if none were low-confidence hits
            low = any(kb.holds(Atom(predicate="LowConfidenceIndicator", args=(i,))) for a in alist
                      for i in [x.args[1] for x in kb.query(f"HasIndicator({a['alert_id']}, ?i)")])
            ioc = None if low else False
        users = {a.get("user_id") for a in alist if a.get("user_id")} | {
            x.args[0] for x in kb.query(f"LoginSuccess(?u, {h})")}
        anomaly = any(kb.holds(f"SuccessAfterFailures({u}, {h})") or kb.holds(f"ImpossibleTravel({u})") for u in users)
        auth_seen = bool(kb.query(f"LoginSuccess(?u, {h})"))
        classes = {kb.class_of(a["alert_type"]) for a in alist}
        if kb.holds(f"¬MalwareConfirmed({h})"):
            edr = False
        elif kb.holds(f"MalwareConfirmed({h})") or any(
                a["_source"] == "edr" and kb.class_of(a["alert_type"]) == "execution" for a in alist):
            edr = True
        elif kb.holds(f"NoEDR({h})"):
            edr = None
        else:
            edr = False
        fp = any(kb.holds(f"AnalystVerdictFP({a['alert_id']})") for a in alist)
        return {
            "IOCMatch": ioc,
            "AuthAnomaly": True if anomaly else (False if auth_seen else None),
            "Maintenance": kb.holds(f"InMaintenance({h})"),
            "EDRDetection": edr,
            "LateralSignal": bool(classes & {"lateral_movement", "data_exfiltration"}),
            "AnalystFP": True if fp else None,
        }

    def _hmm_sequence(self, alist: list[dict]) -> list[str]:
        obs_map = self.kb.ontology["kill_chain_observation"]
        out, prev = [], None
        bin_ = timedelta(minutes=self.s.temporal.bin_minutes)
        for a in alist:
            if prev is not None:
                gaps = int((a["_ts"] - prev) / bin_)
                out += ["quiet"] * min(gaps, 3)
            out.append(obs_map.get(self.kb.class_of(a["alert_type"]), "quiet"))
            prev = a["_ts"]
        return out

    def _bin_counts(self, alist: list[dict]) -> list[float]:
        if not alist:
            return []
        start = alist[0]["_ts"]
        bin_ = timedelta(minutes=self.s.temporal.bin_minutes)
        n = int((alist[-1]["_ts"] - start) / bin_) + 1
        counts = [0.0] * n
        for a in alist:
            counts[int((a["_ts"] - start) / bin_)] += 1
        return [0.0, 0.0, *counts]       # two quiet baseline bins seed the filter

    def _incidents(self, alerts: list[Percept], risks: list[RiskAssessment], kb: KnowledgeBase) -> list[Incident]:
        parent: dict[str, str] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for p in alerts:
            a = p.payload["alert_id"]
            for ent in (f"h:{p.payload['asset_id']}", f"u:{p.payload.get('user_id')}" if p.payload.get("user_id") else None):
                if ent:
                    parent[find(a)] = find(ent)
        groups: dict[str, list[Percept]] = defaultdict(list)
        for p in alerts:
            groups[find(p.payload["alert_id"])].append(p)
        risk_by = {r.asset_id: r for r in risks}
        out = []
        for i, (_, ps) in enumerate(sorted(groups.items()), 1):
            assets = sorted({p.payload["asset_id"] for p in ps})
            users = sorted({p.payload["user_id"] for p in ps if p.payload.get("user_id")})
            classes = [kb.class_of(p.payload["alert_type"]) for p in ps]
            if any(kb.holds(f"CompromisedAccount({u})") for u in users):
                itype = "credential_compromise"
            elif any(kb.holds(f"CompromisedHost({h})") for h in assets):
                itype = "host_compromise"
            elif all(any(kb.holds(f"¬TruePositive({p.payload['alert_id']})") for _ in [0]) for p in ps):
                itype = "benign_activity"
            else:
                itype = f"suspected_{max(set(classes), key=classes.count)}"
            conf = max((risk_by[h].risk_score for h in assets if h in risk_by), default=0.0)
            stage = max((risk_by[h] for h in assets if h in risk_by), key=lambda r: r.risk_score).hmm_stage \
                if any(h in risk_by for h in assets) else "unknown"
            out.append(Incident(incident_id=f"INC-{i:03d}", incident_type=itype, assets=assets, users=users,
                                alerts=sorted(p.payload["alert_id"] for p in ps), confidence=round(conf, 3), stage=stage))
        return sorted(out, key=lambda x: -x.confidence)

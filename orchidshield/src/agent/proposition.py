"""Stage 2 - PROPOSITIONS. Translate percepts into ground, testable literals plus a
model-ready feature vector per alert, and detect contradictions between new
propositions and between propositions and the knowledge base."""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta

from core.config import Settings
from core.schemas import Atom, Contradiction, Percept, Proposition, PropositionSet, ValidationStatus
from agent.knowledge import KnowledgeBase, is_external_ip, short_indicator
from reasoning.ml import alert_features


def _haversine_km(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


class Propositionalizer:
    def __init__(self, settings: Settings, kb: KnowledgeBase):
        self.s, self.kb = settings, kb
        self._n = 0

    def _prop(self, out: list[Proposition], atom: Atom, p: Percept | list[Percept], testable: str, conf: float | None = None):
        ps = p if isinstance(p, list) else [p]
        self._n += 1
        out.append(Proposition(prop_id=f"Q{self._n:04d}", atom=atom, derived_from=[x.percept_id for x in ps],
                               confidence=conf if conf is not None else min(x.quality for x in ps), testable_by=testable))

    def build(self, percepts: list[Percept]) -> PropositionSet:
        usable = [p for p in percepts if p.validation != ValidationStatus.REJECTED]
        props: list[Proposition] = []
        A = lambda pred, *args, pos=True: Atom(predicate=pred, args=tuple(args), positive=pos)  # noqa: E731
        auths: list[Percept] = []
        contradictions: list[Contradiction] = []

        for p in usable:
            d = p.payload
            if p.kind == "alert":
                a = d["alert_id"]
                self._prop(props, A("Alert", a), p, "SIEM record exists")
                self._prop(props, A("AlertType", a, d["alert_type"]), p, "detection rule id")
                self._prop(props, A("Severity", a, d["severity"]), p, "vendor severity")
                self._prop(props, A("OnAsset", a, d["asset_id"]), p, "CMDB lookup of asset id")
                if d["asset_id"] not in self.kb.assets:
                    self._prop(props, A("UnknownAsset", d["asset_id"]), p, "asset inventory miss")
                if d.get("user_id"):
                    self._prop(props, A("ForUser", a, d["user_id"]), p, "directory lookup")
                if d.get("src_ip"):
                    self._prop(props, A("FromIP", a, d["src_ip"]), p, "netflow")
                    if is_external_ip(d["src_ip"]):
                        self._prop(props, A("ExternalIP", d["src_ip"]), p, "ip classification")
                for ind in d.get("indicators", []):
                    self._prop(props, A("HasIndicator", a, short_indicator(ind, "sha256" if len(ind) == 64 else "x")),
                               p, "raw event field")
                if "possible_prompt_injection" in p.content_flags:
                    self._prop(props, A("InjectionAttempt", a), p, "content scanner")
            elif p.kind == "auth_event":
                auths.append(p)
                if d["outcome"] == "success":
                    self._prop(props, A("LoginSuccess", d["user_id"], d["asset_id"]), p, "auth log")
            elif p.kind == "feedback":
                pred = "AnalystVerdictTP" if d["verdict"] == "true_positive" else "AnalystVerdictFP"
                self._prop(props, A(pred, d["alert_id"]), p, "case management record", conf=0.9)
            elif p.kind == "asset":
                old = self.kb.assets.get(d["asset_id"])
                if old and old["criticality"] != d["criticality"]:
                    contradictions.append(Contradiction(
                        statement=f"Criticality({d['asset_id']})",
                        supporting=[f"inventory={old['criticality']}"], opposing=[f"{p.percept_id}={d['criticality']}"],
                        description="asset criticality in new percept disagrees with the knowledge base; "
                                    "keeping the higher value until reconciled"))
                    d = {**d, "criticality": max(old["criticality"], d["criticality"])}
                for at in self.kb.upsert_asset(d, p.percept_id):
                    self._prop(props, at, p, "CMDB")
            elif p.kind == "user":
                for at in self.kb.upsert_user(d, p.percept_id):
                    self._prop(props, at, p, "directory")
            elif p.kind == "indicator":
                for at in self.kb.upsert_indicator(d, p.percept_id):
                    self._prop(props, at, p, "threat-intel re-query")
            elif p.kind == "tool_result":
                out, target = d["output"], d["target"]
                if d["tool"] == "edr_process_tree" and "malicious" in out:
                    self._prop(props, A("MalwareConfirmed", target, pos=bool(out["malicious"])), p,
                               "EDR forensic triage", conf=0.95)

        self._auth_patterns(auths, props)

        # contradictions among propositions themselves and against KB facts
        keys = {pr.atom.key: pr for pr in props}
        for pr in props:
            neg = pr.atom.negate()
            if neg.key in keys and pr.atom.positive:
                contradictions.append(Contradiction(statement=str(pr.atom), supporting=[pr.prop_id],
                                                    opposing=[keys[neg.key].prop_id],
                                                    description="new percepts assert both a literal and its negation"))
            elif pr.atom.positive and neg.key in self.kb.facts:
                contradictions.append(Contradiction(statement=str(pr.atom), supporting=[pr.prop_id],
                                                    opposing=[self.kb.facts[neg.key].fact_id],
                                                    description="new percept conflicts with an existing KB fact"))
        verdicts = defaultdict(set)
        for pr in props:
            if pr.atom.predicate in ("AnalystVerdictTP", "AnalystVerdictFP"):
                verdicts[pr.atom.args[0]].add(pr.atom.predicate)
        for a, v in verdicts.items():
            if len(v) > 1:
                contradictions.append(Contradiction(statement=f"Verdict({a})", supporting=["AnalystVerdictTP"],
                                                    opposing=["AnalystVerdictFP"], description="analysts disagree"))
        return PropositionSet(propositions=props, features=self._features(usable, auths), contradictions=contradictions)

    def _auth_patterns(self, auths: list[Percept], props: list[Proposition]) -> None:
        th = self.s.thresholds
        by_uh: dict[tuple[str, str], list[Percept]] = defaultdict(list)
        by_u: dict[str, list[Percept]] = defaultdict(list)
        for p in sorted(auths, key=lambda x: x.timestamp or datetime.min):
            by_uh[(p.payload["user_id"], p.payload["asset_id"])].append(p)
            by_u[p.payload["user_id"]].append(p)
        for (u, h), evs in by_uh.items():
            for i, e in enumerate(evs):
                if e.payload["outcome"] != "success":
                    continue
                window = [x for x in evs[:i] if x.payload["outcome"] == "failure"
                          and e.timestamp - x.timestamp <= timedelta(minutes=th.failed_login_window_min)]
                if len(window) >= th.failed_login_burst:
                    self._prop(props, Atom(predicate="SuccessAfterFailures", args=(u, h)), [*window, e],
                               f">= {th.failed_login_burst} failures then success within {th.failed_login_window_min} min")
                    break
        for u, evs in by_u.items():
            succ = [e for e in evs if e.payload["outcome"] == "success" and e.payload.get("geo")]
            for a, b in zip(succ, succ[1:]):
                hours = max((b.timestamp - a.timestamp).total_seconds() / 3600, 1e-3)
                if _haversine_km(a.payload["geo"], b.payload["geo"]) / hours > th.impossible_travel_kmh:
                    self._prop(props, Atom(predicate="ImpossibleTravel", args=(u,)), [a, b],
                               f"geo velocity > {th.impossible_travel_kmh} km/h")
                    break

    def _features(self, usable: list[Percept], auths: list[Percept]) -> dict[str, dict[str, float]]:
        alerts = [p for p in usable if p.kind == "alert"]
        per_asset = defaultdict(int)
        for p in alerts:
            per_asset[p.payload["asset_id"]] += 1
        fails = defaultdict(int)
        for p in auths:
            if p.payload["outcome"] == "failure":
                fails[(p.payload["user_id"], p.payload["asset_id"])] += 1
        out = {}
        for p in alerts:
            d = p.payload
            u, h = d.get("user_id"), d["asset_id"]
            inds = [short_indicator(i, "sha256" if len(i) == 64 else "x") for i in d.get("indicators", [])]
            statuses = [(self.kb.indicator_status(self.kb.indicators[i]), self.kb.indicators[i]["confidence"])
                        for i in inds if i in self.kb.indicators]
            known = [c for s, c in statuses if s == "known_bad"]
            user = self.kb.users.get(u or "", {})
            ctx = {
                "severity": {"low": 1, "medium": 2, "high": 3, "critical": 4}[d["severity"]],
                "criticality": self.kb.assets.get(h, {}).get("criticality", 1),
                "ioc_match": bool(known), "ioc_confidence": max(known, default=0.0),
                "failed_logins": fails.get((u, h), 0),
                "success_after_fail": bool(u) and fails.get((u, h), 0) >= self.s.thresholds.failed_login_burst,
                "impossible_travel": False,
                "edr_source": p.source == "edr",
                "off_hours": bool(p.timestamp and (p.timestamp.hour < 7 or p.timestamp.hour >= 20)),
                "privileged_user": user.get("account_type") in ("privileged", "break_glass", "service"),
                "asset_alert_count": per_asset[h],
                "alert_class": self.kb.class_of(d["alert_type"]),
            }
            ctx["impossible_travel"] = d["alert_type"] == "impossible_travel"
            out[d["alert_id"]] = alert_features(ctx)
        return out

"""Stage 3 - KNOWLEDGE BASE. Holds background facts (ontology, asset inventory,
directory, threat intel), observed facts from propositions, Horn rules loaded from
03_logic/rules.yaml, hard constraints from config, and learned-model metadata.
Every fact carries provenance; forward chaining records a justification for every
derived fact so any conclusion can be traced back to the percepts it rests on."""
from __future__ import annotations

import ipaddress
from datetime import datetime, timezone
from pathlib import Path

import yaml

from core.config import Settings
from core.schemas import Atom, KnowledgeFact, KnowledgeRule, ModelMetadata, Proposition
from reasoning.fol import HornKB, Rule, parse_atom, rules_from_dicts


def short_indicator(value: str, ioc_type: str) -> str:
    return f"sha256:{value[:12]}" if ioc_type == "sha256" or len(value) == 64 else value


_INTERNAL_NETS = [ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8", "169.254.0.0/16", "100.64.0.0/10",
    "fc00::/7", "fe80::/10", "::1/128")]


def is_external_ip(ip: str) -> bool:
    """Outside the organisation's internal ranges. (Documentation ranges such as
    203.0.113.0/24 used by the synthetic data count as external on purpose.)"""
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not any(a in n for n in _INTERNAL_NETS if n.version == a.version)


class KnowledgeBase:
    def __init__(self, settings: Settings, environment: dict):
        self.s = settings
        logic_dir = settings.path("logic_dir")
        spec = yaml.safe_load((logic_dir / "rules.yaml").read_text())
        self.rules: list[Rule] = rules_from_dicts(spec["rules"])
        self.rule_meta = {r["id"]: r for r in spec["rules"]}
        self.ontology = yaml.safe_load((logic_dir / "ontology.yaml").read_text())
        self.facts: dict[tuple, KnowledgeFact] = {}
        self.models: dict[str, ModelMetadata] = {}
        self.assets: dict[str, dict] = {}
        self.users: dict[str, dict] = {}
        self.indicators: dict[str, dict] = {}
        self._n = 0
        self._load_ontology()
        self._load_environment(environment)
        self.horn: HornKB | None = None

    # ------------------------------------------------------------- assertion
    def _assert(self, atom: Atom, kind: str, provenance: list[str], confidence: float = 1.0) -> KnowledgeFact:
        if atom.key in self.facts:
            f = self.facts[atom.key]
            for p in provenance:
                if p not in f.provenance:
                    f.provenance.append(p)
            f.confidence = max(f.confidence, confidence)
            return f
        self._n += 1
        f = KnowledgeFact(fact_id=f"F{self._n:04d}", atom=atom, kind=kind, confidence=confidence,  # type: ignore[arg-type]
                          provenance=list(provenance), asserted_at=datetime.now(timezone.utc))
        self.facts[atom.key] = f
        self.horn = None
        return f

    def tell(self, text: str, kind: str = "observed", provenance: str = "manual") -> KnowledgeFact:
        return self._assert(parse_atom(text), kind, [provenance])

    def _load_ontology(self) -> None:
        for child, parent in self.ontology["alert_taxonomy"].items():
            self._assert(Atom(predicate="SubtypeOf", args=(child, parent)), "ontology", ["ontology.yaml"])
        for child, parent in self.ontology["asset_taxonomy"].items():
            self._assert(Atom(predicate="SubtypeOf", args=(child, parent)), "ontology", ["ontology.yaml"])

    def _load_environment(self, env: dict) -> None:
        for a in env.get("assets", {}).values():
            self.upsert_asset(a, "asset_inventory")
        for u in env.get("users", {}).values():
            self.upsert_user(u, "directory")
        for i in env.get("indicators", []):
            self.upsert_indicator(i, "threat_intel")

    def upsert_asset(self, a: dict, prov: str) -> list[Atom]:
        self.assets[a["asset_id"]] = a
        h = a["asset_id"]
        atoms = [Atom(predicate="Asset", args=(h,)), Atom(predicate="IsA", args=(h, a["asset_type"]))]
        if a["criticality"] >= self.s.thresholds.critical_asset_min:
            atoms.append(Atom(predicate="Critical", args=(h,)))
        for g in a.get("connected_to", []):
            atoms += [Atom(predicate="ConnectedTo", args=(h, g)), Atom(predicate="ConnectedTo", args=(g, h))]
        if a.get("in_maintenance"):
            atoms.append(Atom(predicate="InMaintenance", args=(h,)))
        if not a.get("edr_installed", True):
            atoms.append(Atom(predicate="NoEDR", args=(h,)))
        for at in atoms:
            self._assert(at, "background", [prov])
        return atoms

    def upsert_user(self, u: dict, prov: str) -> list[Atom]:
        self.users[u["user_id"]] = u
        pred = {"privileged": "Privileged", "service": "ServiceAccount", "break_glass": "BreakGlass"}.get(u["account_type"])
        atoms = [Atom(predicate="User", args=(u["user_id"],))] + ([Atom(predicate=pred, args=(u["user_id"],))] if pred else [])
        for at in atoms:
            self._assert(at, "background", [prov])
        return atoms

    def indicator_status(self, i: dict, now: datetime | None = None) -> str:
        now = now or datetime(2026, 11, 16, tzinfo=timezone.utc)
        exp = i.get("expires_at")
        if exp:
            exp_dt = datetime.fromisoformat(exp) if isinstance(exp, str) else exp
            if exp_dt < now:
                return "expired"
        return "known_bad" if i["confidence"] >= self.s.thresholds.ioc_min_confidence else "low_confidence"

    def upsert_indicator(self, i: dict, prov: str) -> list[Atom]:
        key = short_indicator(i["value"], i["ioc_type"])
        self.indicators[key] = i
        status = self.indicator_status(i)
        pred = {"known_bad": "KnownBadIndicator", "expired": "ExpiredIndicator",
                "low_confidence": "LowConfidenceIndicator"}[status]
        at = Atom(predicate=pred, args=(key,))
        self._assert(at, "background", [f"{prov}:{i.get('source_feed', '?')}"], i["confidence"])
        return [at]

    def add_propositions(self, props: list[Proposition]) -> list[KnowledgeFact]:
        return [self._assert(p.atom, "observed", [p.prop_id, *p.derived_from], p.confidence) for p in props]

    def register_model(self, meta: ModelMetadata) -> None:
        self.models[meta.name] = meta

    # ------------------------------------------------------------- reasoning
    def closure(self) -> HornKB:
        """Forward-chained closure with justifications (cached until the next assertion)."""
        if self.horn is None:
            hk = HornKB(rules=list(self.rules))
            for f in self.facts.values():
                hk.tell(f.atom)
            hk.forward_chain()
            self.horn = hk
        return self.horn

    def derived_facts(self) -> list[Atom]:
        hk = self.closure()
        return [hk.facts[k] for k, j in hk.just.items() if j.rule_id is not None]

    def conflicts(self) -> list[tuple[Atom, Atom]]:
        return self.closure().conflicts()

    def holds(self, text_or_atom: str | Atom) -> bool:
        atom = parse_atom(text_or_atom) if isinstance(text_or_atom, str) else text_or_atom
        return atom.key in self.closure().facts

    def query(self, pattern: str) -> list[Atom]:
        from reasoning.fol import unify
        pat = parse_atom(pattern)
        return [a for a in self.closure().facts.values() if unify(pat, a) is not None]

    def proof_tree(self, atom: Atom, depth: int = 0, seen: set | None = None) -> list[str]:
        """Unfold the justification of a fact into readable steps (leaves first)."""
        hk = self.closure()
        seen = seen if seen is not None else set()
        if atom.key in seen or depth > 15:
            return []
        seen.add(atom.key)
        j = hk.just.get(atom.key)
        if j is None:
            return []
        if j.rule_id is None:
            f = self.facts.get(atom.key)
            src = ",".join(f.provenance[:2]) if f else "?"
            return [f"fact {atom}  [{f.kind if f else '?'}: {src}]"]
        steps: list[str] = []
        prem = [hk.facts[p] for p in j.premises]
        for p in prem:
            steps += self.proof_tree(p, depth + 1, seen)
        steps.append(f"{j.rule_id}: {' ∧ '.join(map(str, prem))} ⊢ {atom}")
        return steps

    def rules_used(self, atom: Atom) -> set[str]:
        hk = self.closure()
        return {hk.just[k].rule_id for k in hk.support_closure(atom.key) if hk.just.get(k) and hk.just[k].rule_id}  # type: ignore[misc]

    def rule_records(self) -> list[KnowledgeRule]:
        return [KnowledgeRule(rule_id=r.rule_id, body=list(r.body), head=r.head, description=r.description,
                              provenance=self.rule_meta[r.rule_id].get("provenance", "")) for r in self.rules]

    def class_of(self, alert_type: str) -> str:
        """Top-level kill-chain class of an alert type (the parent just below `intrusion`)."""
        tax = self.ontology["alert_taxonomy"]
        cur = alert_type
        while tax.get(cur) and tax[cur] != "intrusion":
            cur = tax[cur]
        return cur

    def stats(self) -> dict[str, int]:
        kinds: dict[str, int] = {}
        for f in self.facts.values():
            kinds[f.kind] = kinds.get(f.kind, 0) + 1
        return {**kinds, "rules": len(self.rules), "derived": len(self.derived_facts())}

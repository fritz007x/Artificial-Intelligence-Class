"""Stage 5 - ENTAILMENT. Ask whether the KB supports a conclusion and classify it:

ENTAILED      KB ⊨ q, and no fact on q's proof is involved in a conflict.
CONTRADICTED  KB ⊨ ¬q, or q's proof rests on a literal whose negation is also derivable.
UNCERTAIN     q is not provable, but some premises of a rule concluding q hold and/or a
              probabilistic model gives support. Model confidence is reported as
              `probabilistic_support` and may set `sufficiently_supported`, but it is
              NEVER promoted to ENTAILED.
UNKNOWN       no relevant evidence either way."""
from __future__ import annotations

from core.config import Settings
from core.schemas import Atom, EntailmentResult, EntailmentStatus
from agent.knowledge import KnowledgeBase
from reasoning.fol import parse_atom, substitute, unify


class EntailmentChecker:
    def __init__(self, settings: Settings, kb: KnowledgeBase):
        self.s, self.kb = settings, kb

    def _conflicted_keys(self) -> set[tuple]:
        keys: set[tuple] = set()
        for pos, neg in self.kb.conflicts():
            keys |= {pos.key, neg.key}
        return keys

    def _partial_premises(self, q: Atom) -> list[str]:
        hk = self.kb.closure()
        found: list[str] = []
        for rule in self.kb.rules:
            theta = unify(rule.head, q)
            if theta is None:
                continue
            for b in rule.body:
                pat = substitute(b, theta)
                hits = [f for f in hk.facts.values() if unify(pat, f) is not None]
                if hits:
                    found.append(f"{rule.rule_id}: {hits[0]}")
        return found

    def check(self, query: str | Atom, support: float | None = None) -> EntailmentResult:
        q = parse_atom(query) if isinstance(query, str) else query
        hk = self.kb.closure()
        conflicted = self._conflicted_keys()
        proof = self.kb.proof_tree(q) if q.key in hk.facts else []
        counter = self.kb.proof_tree(q.negate()) if q.negate().key in hk.facts else []
        sup = None if support is None else round(float(support), 4)
        suff = sup is not None and sup >= self.s.thresholds.bn_support

        if proof and counter:
            return EntailmentResult(query=str(q), status=EntailmentStatus.CONTRADICTED, proof=proof, counter_proof=counter,
                                    conflicts=[str(q)], probabilistic_support=sup, sufficiently_supported=False,
                                    rationale="both the conclusion and its negation are derivable")
        if counter:
            return EntailmentResult(query=str(q), status=EntailmentStatus.CONTRADICTED, counter_proof=counter,
                                    probabilistic_support=sup, sufficiently_supported=False,
                                    rationale="the KB entails the negation of the query")
        if proof:
            touched = hk.support_closure(q.key) & conflicted
            if touched:
                names = sorted({str(hk.facts[k]) for k in touched})
                return EntailmentResult(query=str(q), status=EntailmentStatus.CONTRADICTED, proof=proof, conflicts=names,
                                        probabilistic_support=sup, sufficiently_supported=False,
                                        rationale="a proof exists but rests on conflicting facts; refusing unjustified entailment")
            return EntailmentResult(query=str(q), status=EntailmentStatus.ENTAILED, proof=proof, probabilistic_support=sup,
                                    sufficiently_supported=True,
                                    rationale=f"derived by forward chaining ({len(proof)} steps); no conflict on the proof")
        partial = self._partial_premises(q)
        if partial or (sup is not None and sup >= self.s.thresholds.abstain_low):
            why = "some premises hold but the rule body is incomplete" if partial else "no logical proof"
            if sup is not None:
                why += f"; probabilistic support {sup:.2f}" + (" (sufficient for non-disruptive action, not proof)"
                                                                 if suff else "")
            return EntailmentResult(query=str(q), status=EntailmentStatus.UNCERTAIN, proof=partial,
                                    probabilistic_support=sup, sufficiently_supported=suff, rationale=why)
        return EntailmentResult(query=str(q), status=EntailmentStatus.UNKNOWN, probabilistic_support=sup,
                                rationale="no relevant evidence in the knowledge base")

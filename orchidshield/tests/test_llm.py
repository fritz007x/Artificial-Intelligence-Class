"""Narrator (contemporary AI component) tests with deterministic doubles - no network."""
from core.config import load_settings
from core.llm import TemplateNarrator, grounded, make_narrator

EVIDENCE = {"top_risks": [{"asset_id": "ws-005", "risk_score": 0.7, "p_compromise_bayes": 0.9, "hmm_stage": "x"}],
            "entailments": [{"query": "CompromisedHost(ws-005)", "status": "entailed"}],
            "actions": [{"action": "isolate_host", "target": "ws-005", "status": "pending_approval"}],
            "contradictions": []}


def test_template_is_deterministic():
    assert TemplateNarrator().summarize(EVIDENCE) == TemplateNarrator().summarize(EVIDENCE)
    assert "ws-005" in TemplateNarrator().summarize(EVIDENCE)


def test_grounding_rejects_hallucinated_entities():
    assert grounded("ws-005 is compromised and awaits isolation", EVIDENCE)
    assert not grounded("ws-005 and dc-01 are compromised", EVIDENCE)
    assert not grounded("user u-99 should be disabled", EVIDENCE)


def test_disabled_flag_uses_template():
    assert make_narrator(False, load_settings().llm).name == "template"


def test_enabled_without_sdk_or_key_falls_back(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "anthropic":
            raise ImportError("not installed")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    assert make_narrator(True, load_settings().llm).name == "template"

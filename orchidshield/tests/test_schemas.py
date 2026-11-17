"""Schema tests: strict validation of every stage artifact."""
import pytest
from pydantic import ValidationError

from core.schemas import (ActionProposal, AlertIn, Atom, AuditEvent, CaseResult, EntailmentResult, EntailmentStatus,
                          InferenceTrace, KnowledgeFact, KnowledgeRule, Observation, Percept, Proposition, ToolResult)
from reasoning.fol import parse_atom

GOOD_ALERT = {"alert_id": "A-1", "timestamp": "2026-11-16T02:00:00+00:00", "alert_type": "brute_force",
              "severity": "high", "asset_id": "ws-001"}


def test_alert_valid():
    assert AlertIn.model_validate(GOOD_ALERT).severity == "high"


@pytest.mark.parametrize("patch", [
    {"severity": "urgent"}, {"alert_id": "A-1; DROP TABLE"}, {"timestamp": "yesterday"},
    {"alert_type": "Brute Force!"}, {"unexpected": 1}, {"description": "x" * 5000},
])
def test_alert_rejects_malformed(patch):
    with pytest.raises(ValidationError):
        AlertIn.model_validate({**GOOD_ALERT, **patch})


def test_observation_kind_and_source_are_enumerated():
    with pytest.raises(ValidationError):
        Observation.model_validate({"kind": "shell_command", "source": "siem", "data": {}})
    with pytest.raises(ValidationError):
        Observation.model_validate({"kind": "alert", "source": "internet", "data": {}})


def test_atom_parse_roundtrip_and_negation():
    a = parse_atom("¬TruePositive(A-1)")
    assert not a.positive and a.args == ("A-1",)
    assert str(a.negate()) == "TruePositive(A-1)"
    assert parse_atom("not CompromisedHost(ws-1)").positive is False
    with pytest.raises(ValueError):
        parse_atom("lowercase(x)")


def test_rule_requires_body():
    with pytest.raises(ValidationError):
        KnowledgeRule(rule_id="R", body=[], head=Atom(predicate="X", args=("a",)), description="", provenance="")


def test_all_stage_models_forbid_extra_fields():
    for model in (Percept, Proposition, KnowledgeFact, InferenceTrace, EntailmentResult, ActionProposal, ToolResult,
                  AuditEvent, CaseResult):
        assert model.model_config.get("extra") == "forbid", model.__name__


def test_entailment_status_vocabulary():
    assert {s.value for s in EntailmentStatus} == {"entailed", "contradicted", "uncertain", "unknown"}


def test_case_result_json_schema_is_exportable():
    schema = CaseResult.model_json_schema()
    for key in ("percepts", "propositions", "knowledge_used", "inference", "entailments", "proposals"):
        assert key in schema["properties"]

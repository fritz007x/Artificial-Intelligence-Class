"""Learning components: artifact integrity, evaluation thresholds, fairness, RL safety."""
import json
import shutil
from pathlib import Path

import pytest

from core.config import load_settings
from reasoning import rl
from reasoning.ml import FEATURES, PROTECTED, load_artifact

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "05_ml" / "artifacts"
METRICS = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
S = load_settings()


def test_protected_attributes_are_not_features():
    assert not PROTECTED & set(FEATURES)


def test_artifact_loads_and_predicts():
    m = load_artifact(ART / "model.json")
    p = m.predict_one({f: 0.0 for f in FEATURES})
    assert 0.0 <= p <= 1.0
    assert m.metadata["features"] == FEATURES


def test_tampered_artifact_is_rejected(tmp_path):
    p = tmp_path / "model.json"
    shutil.copy(ART / "model.json", p)
    art = json.loads(p.read_text(encoding="utf-8"))
    key = "w" if "w" in art["model"] else "b2"
    art["model"][key] = [0.0] * len(art["model"][key]) if isinstance(art["model"][key], list) else 9.9
    p.write_text(json.dumps(art), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_artifact(p)


def test_selected_model_meets_evaluation_thresholds():
    t = METRICS["test"][METRICS["selected_model"]]
    assert t["roc_auc"] >= 0.90 and t["f1"] >= 0.75 and t["ece"] <= 0.08
    assert t["roc_auc"] > METRICS["test"]["symbolic_rules"]["roc_auc"]
    assert METRICS["test"]["majority"]["roc_auc"] == pytest.approx(0.5)


def test_fairness_fpr_gap_within_threshold():
    best = METRICS["selected_model"]
    assert METRICS["fairness"][best]["max_gap"] <= S.thresholds.fairness_max_fpr_gap


def test_learning_case_improves_without_violating_guardrails():
    lc = METRICS["learning_case"]
    assert lc["improves"] and lc["rl_improves_reward"]
    assert lc["guardrails"]["rl_unsafe_action_rate"] == 0
    assert lc["guardrails"]["fairness_gap_ok"]


def test_rl_masks_containment_until_enriched_high_risk():
    assert "request_containment" not in rl.allowed_actions((2, 2, 1, 0))
    assert "request_containment" not in rl.allowed_actions((1, 2, 1, 1))
    assert "request_containment" in rl.allowed_actions((2, 0, 0, 1))
    Q = rl.load_q(ART / "q_table.json")
    pol = rl.greedy_policy(Q)
    for s in Q:
        assert pol(s) in rl.allowed_actions(s)


def test_training_is_reproducible():
    from reasoning.evaluation import split
    from sim.generator import make_alert_dataset
    a, b = make_alert_dataset(300, 5), make_alert_dataset(300, 5)
    assert a == b
    assert [r["alert_id"] for r in split(a, 5)[2]] == [r["alert_id"] for r in split(b, 5)[2]]

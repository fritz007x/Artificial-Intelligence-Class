"""Reproducible training & evaluation pipeline for OrchidShield's learned components.

    python 05_ml/train.py            # writes 05_ml/artifacts/{model.json,q_table.json,metrics.json}
                                     # and regenerates 05_ml/model_card.md

Steps: synthetic data -> feature engineering -> stratified train/val/test split ->
5-fold CV on train -> baselines (majority, symbolic rules, sklearn LR/RF) -> numpy
logistic regression and MLP -> model selection on validation -> held-out test ->
calibration & fairness audit -> RL Q-learning vs rule policy -> learning-case report."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from core.config import load_settings  # noqa: E402
from reasoning import rl  # noqa: E402
from reasoning.evaluation import precision_at_k, rule_score, split, xy  # noqa: E402
from reasoning.ml import (FEATURES, LogisticModel, MLPModel, classification_metrics, fpr_by_group,  # noqa: E402
                          save_artifact)
from sim.generator import make_alert_dataset, write_environment  # noqa: E402
from sim.scenarios import write_scenarios  # noqa: E402

ART = ROOT / "05_ml" / "artifacts"


def r4(d):
    return {k: round(float(v), 4) for k, v in d.items()}


def cross_validate(make, X, y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(y)), k)
    aucs = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        m = make().fit(X[tr], y[tr], seed=seed)
        aucs.append(classification_metrics(y[te], m.predict_proba(X[te]))["roc_auc"])
    return {"cv_auc_mean": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs))}


def main() -> dict:
    s = load_settings()
    seed = s.app.seed
    ART.mkdir(parents=True, exist_ok=True)
    write_environment(seed=seed)
    write_scenarios()

    rows = make_alert_dataset(4000, seed)
    train, val, test = split(rows, seed)
    Xtr, ytr = xy(train)
    Xva, yva = xy(val)
    Xte, yte = xy(test)
    report: dict = {"seed": seed, "n": {"train": len(train), "val": len(val), "test": len(test)},
                    "positive_rate": round(float(np.mean([r["label"] for r in rows])), 4), "features": FEATURES}

    # ---- cross-validation on train
    report["cv"] = {"logistic_regression": r4(cross_validate(LogisticModel, Xtr, ytr, seed=seed)),
                    "mlp": r4(cross_validate(lambda: MLPModel(epochs=80, patience=1000), Xtr, ytr, seed=seed))}

    # ---- fit candidates
    lr = LogisticModel().fit(Xtr, ytr, seed=seed)
    mlp = MLPModel().fit(Xtr, ytr, seed=seed, X_val=Xva, y_val=yva)
    sk_lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    sk_rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=seed).fit(Xtr, ytr)
    preds_val = {
        "majority": np.full(len(yva), ytr.mean()),
        "symbolic_rules": np.array([rule_score(r) for r in val]),
        "sklearn_logreg": sk_lr.predict_proba(Xva)[:, 1],
        "sklearn_random_forest": sk_rf.predict_proba(Xva)[:, 1],
        "logistic_regression": lr.predict_proba(Xva),
        "mlp": mlp.predict_proba(Xva),
    }
    report["val"] = {k: r4(classification_metrics(yva, p)) for k, p in preds_val.items()}

    # ---- selection: exportable numpy models only, by validation AUC then ECE
    cands = {"logistic_regression": lr, "mlp": mlp}
    best = max(cands, key=lambda k: (round(report["val"][k]["roc_auc"], 3), -report["val"][k]["ece"]))
    report["selected_model"] = best
    report["selection_rule"] = "highest validation ROC-AUC (3 d.p.), ties broken by lower ECE; numpy-exportable only"

    preds_test = {
        "majority": np.full(len(yte), ytr.mean()),
        "symbolic_rules": np.array([rule_score(r) for r in test]),
        "sklearn_random_forest": sk_rf.predict_proba(Xte)[:, 1],
        "logistic_regression": lr.predict_proba(Xte),
        "mlp": mlp.predict_proba(Xte),
    }
    report["test"] = {k: r4(classification_metrics(yte, p)) for k, p in preds_test.items()}

    # ---- fairness audit (region is NOT a feature)
    regions = [r["region"] for r in test]
    fairness = {}
    for name in ("symbolic_rules", best):
        thr = 0.5
        f = fpr_by_group(yte, preds_test[name], regions, thr)
        fairness[name] = {"fpr_by_region": r4(f), "max_gap": round(max(f.values()) - min(f.values()), 4)}
    fairness["threshold"] = s.thresholds.fairness_max_fpr_gap
    fairness["passes"] = fairness[best]["max_gap"] <= s.thresholds.fairness_max_fpr_gap
    report["fairness"] = fairness

    # ---- explainability: logistic coefficients (global) for the model card
    report["lr_coefficients"] = {f: round(float(w), 3) for f, w in sorted(zip(FEATURES, lr.w), key=lambda t: -abs(t[1]))}

    # ---- save selected artifact
    meta = save_artifact(cands[best], ART / "model.json", {
        "name": "alert-tp-classifier", "version": "1.0.0", "kind": best,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "seed": seed,
        "metrics": {k: report["test"][best][k] for k in ("roc_auc", "f1", "ece", "fpr")}})
    report["artifact_sha256"] = meta["artifact_sha256"]

    # ---- reinforcement learning
    Q, curve = rl.q_learning(episodes=20000, seed=seed)
    rl.save_q(Q, ART / "q_table.json")
    q_eval, rule_eval = rl.evaluate(rl.greedy_policy(Q), seed=seed + 1), rl.evaluate(rl.rule_policy, seed=seed + 1)
    report["rl"] = {"q_learning": r4(q_eval), "rule_policy": r4(rule_eval),
                    "learning_curve_avg_reward_per_1000": [round(c, 3) for c in curve],
                    "states_learned": len(Q)}

    # ---- learning case: does ML-assisted ranking improve precision@k without breaking guardrails?
    k = int(0.1 * len(yte))
    p_rules = precision_at_k(yte, preds_test["symbolic_rules"], k)
    fused = 0.5 * preds_test[best] + 0.5 * preds_test["symbolic_rules"]
    p_fused = precision_at_k(yte, fused, k)
    report["learning_case"] = {
        "metric": f"precision@{k} on held-out alerts (analyst queue top 10%)",
        "rules_only": round(p_rules, 4), "rules_plus_ml": round(p_fused, 4),
        "improves": p_fused > p_rules,
        "rl_improves_reward": q_eval["avg_reward"] > rule_eval["avg_reward"],
        "guardrails": {"rl_unsafe_action_rate": q_eval["unsafe_action_rate"],
                       "fairness_gap_ok": fairness["passes"],
                       "ml_can_trigger_disruptive_action": False,
                       "note": "learned scores only re-rank and inform; disruptive actions still need ENTAILED + approval"},
    }
    (ART / "metrics.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_model_card(report)
    return report


def write_model_card(m: dict) -> None:
    best = m["selected_model"]
    t = m["test"][best]
    rows = "\n".join(f"| {k} | {v['roc_auc']:.3f} | {v['f1']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | "
                     f"{v['fpr']:.3f} | {v['ece']:.3f} |" for k, v in m["test"].items())
    fair = m["fairness"]
    coef = "\n".join(f"| {k} | {v:+.3f} |" for k, v in list(m["lr_coefficients"].items())[:8])
    lc = m["learning_case"]
    card = f"""# Model card: alert-tp-classifier v1.0.0

*Generated by `05_ml/train.py` (seed {m['seed']}). Do not hand-edit numbers; re-run the script.*

## Intended use
Estimate P(true positive) for a SOC alert so the triage queue can be **re-ranked**. The score is one
input to risk fusion. It is **never** treated as logical proof, and it can't trigger disruptive actions
by itself. Containment still needs an ENTAILED justification plus human approval.

## Out of scope
Disciplinary or HR decisions about users, automated containment, and real (non-synthetic) telemetry
until it has been revalidated.

## Data
Synthetic alerts from `src/sim/generator.py` (n={sum(m['n'].values())}, positive rate {m['positive_rate']:.2f}),
stratified split {m['n']}. Features: {', '.join(m['features'])}.
Protected attributes (`region`, `department`) are **excluded** from features and used only for the audit.

## Selection
Selected model: **{best}**. Rule: {m['selection_rule']}.
5-fold CV ROC-AUC on train: logistic {m['cv']['logistic_regression']['cv_auc_mean']:.3f} ± {m['cv']['logistic_regression']['cv_auc_std']:.3f};
MLP {m['cv']['mlp']['cv_auc_mean']:.3f} ± {m['cv']['mlp']['cv_auc_std']:.3f}.

## Held-out test results
| model | ROC-AUC | F1 | precision | recall | FPR | ECE |
|---|---|---|---|---|---|---|
{rows}

Selected model test: ROC-AUC {t['roc_auc']:.3f}, F1 {t['f1']:.3f}, ECE {t['ece']:.3f}, Brier {t['brier']:.3f}.

## Fairness (false-positive rate by user region, threshold 0.5)
- symbolic rules: {fair['symbolic_rules']['fpr_by_region']} (gap {fair['symbolic_rules']['max_gap']:.3f})
- {best}: {fair[best]['fpr_by_region']} (gap {fair[best]['max_gap']:.3f})
- Threshold {fair['threshold']}: **{'PASS' if fair['passes'] else 'FAIL'}**. The generator injects benign APAC VPN
  impossible-travel noise, so the impossible_travel feature is the main disparity risk. Monitor it in production.

## Explainability (logistic coefficients on standardised features, top 8)
| feature | weight |
|---|---|
{coef}

Per-alert explanations are shown in the inference trace (feature values and the model's estimate).

## Learning case (demo scenario 6)
- {lc['metric']}: rules only {lc['rules_only']:.3f}, rules + ML {lc['rules_plus_ml']:.3f}. Improves: **{lc['improves']}**
- RL advisor reward vs rule policy: Q-learning {m['rl']['q_learning']['avg_reward']:.2f} vs rules {m['rl']['rule_policy']['avg_reward']:.2f}
  (improves: **{lc['rl_improves_reward']}**). Unsafe action rate {m['rl']['q_learning']['unsafe_action_rate']:.3f} (masked actions).
- Guardrails: fairness gap ok = {lc['guardrails']['fairness_gap_ok']}; ML cannot trigger disruptive actions.

## Limitations
The model is trained only on synthetic distributions and has no concept drift handling. Labels come from
the generator rather than analyst ground truth, and the calibration holds only for this distribution.
Revalidate against labelled production data before any real use.

## Artifact
`05_ml/artifacts/model.json`, sha256 of the body `{m['artifact_sha256'][:16]}…`. This is checked at load time,
and the service falls back to symbolic scoring if it doesn't match.
"""
    (ROOT / "05_ml" / "model_card.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    rep = main()
    print(json.dumps({"selected": rep["selected_model"], "test": rep["test"][rep["selected_model"]],
                      "fairness_passes": rep["fairness"]["passes"], "learning_case": rep["learning_case"]}, indent=1))

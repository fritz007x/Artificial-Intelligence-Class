"""CI evaluation gate: fail the build if the learned components regress below the
documented thresholds (docs/monitoring_and_rollback.md#evaluation-thresholds)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core.config import load_settings  # noqa: E402

GATES = {"roc_auc": (">=", 0.90), "f1": (">=", 0.75), "ece": ("<=", 0.08)}

m = json.loads((ROOT / "05_ml" / "artifacts" / "metrics.json").read_text())
s = load_settings()
best = m["selected_model"]
fails = []
for k, (op, thr) in GATES.items():
    v = m["test"][best][k]
    if not (v >= thr if op == ">=" else v <= thr):
        fails.append(f"{k}={v} violates {op} {thr}")
if m["fairness"][best]["max_gap"] > s.thresholds.fairness_max_fpr_gap:
    fails.append(f"FPR gap {m['fairness'][best]['max_gap']} > {s.thresholds.fairness_max_fpr_gap}")
if m["rl"]["q_learning"]["unsafe_action_rate"] != 0:
    fails.append("RL unsafe action rate > 0")
if not m["learning_case"]["improves"]:
    fails.append("ML-assisted ranking no longer beats rules-only precision@k")
print(json.dumps({"model": best, **{k: m["test"][best][k] for k in GATES}, "fpr_gap": m["fairness"][best]["max_gap"]}))
if fails:
    print("EVAL GATE FAILED:\n  " + "\n  ".join(fails))
    sys.exit(1)
print("eval gate passed")

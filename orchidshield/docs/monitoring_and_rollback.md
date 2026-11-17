# Monitoring, evaluation thresholds and rollback

## Configuration reference

| Where | What | Notes |
|---|---|---|
| `configs/config.yaml` | Thresholds, risk weights, policy (hard constraints, costs and benefits), orchestrator limits, feature flags, LLM model choice, CSP staffing, seed, config version | Non-secret. Validated strictly at start-up (`core/config.py`; unknown keys fail). |
| `ORCHIDSHIELD_CONFIG` | Path to an alternate config file | Used for staging and canary configs |
| `ORCHIDSHIELD_{OPERATOR,APPROVER,ADMIN}_KEY` | Role API keys | Secrets, environment only |
| `ORCHIDSHIELD_ENV` | `dev` / `staging` / `prod` | `prod` refuses `ORCHIDSHIELD_DEV_OPEN` |
| `ORCHIDSHIELD_KILL_SWITCH=1` | Emergency stop without the API | Checked before every stage and tool call |
| `ANTHROPIC_API_KEY` | Optional hosted narrator | Only read when `features.llm_narrative: true` |
| `03_logic/rules.yaml`, `04_uncertainty/*.json`, `05_ml/artifacts/*` | Versioned knowledge and models | Changing them is a release (see rollback) |

Each case result carries `run_metadata` (app version, config version, config sha, seed, model name and narrator),
so every decision can be tied to the exact configuration that produced it.

## Health and readiness

- `GET /health`: liveness, kill-switch state, whether the model loaded, and the narrator in use. Used by the
  Docker `HEALTHCHECK`.
- `GET /ready`: fails with 503 if the audit hash chain does not verify. Take the pod out of rotation.

## Metrics (`GET /metrics`, Prometheus text format)

| Metric | Alert when |
|---|---|
| `orchidshield_kill_switch` | = 1 for more than 15 min (make sure someone owns the incident) |
| `orchidshield_model_loaded` | = 0 (the agent is in symbolic fallback; ranking quality is degraded) |
| `orchidshield_pending_approvals` | > 10 or oldest > 30 min (approver capacity) |
| `orchidshield_proposals_blocked` rate | Sudden spike (possible injection campaign or misconfiguration) |
| `orchidshield_tool_calls_failed` / `tool_calls_ok` | > 5% over 15 min |
| `orchidshield_percepts_rejected` rate | > 10% of percepts (upstream schema drift or spoofing) |
| `orchidshield_contradictions_total` rate | Trending up (intel feed quality or analyst disagreement) |
| `orchidshield_auth_failures`, `authz_denied`, `rate_limited` | Any burst (credential misuse) |

Logs are structured JSON on stdout with a secret-scrubbing formatter. The audit log at `data/runtime/audit.jsonl`
should be shipped to write-once storage.

## Evaluation thresholds (CI gate: `scripts/check_eval.py`)

| Metric (held-out synthetic test set) | Threshold | Current |
|---|---|---|
| ROC-AUC of selected model | ≥ 0.90 | 0.963 |
| F1 at 0.5 | ≥ 0.75 | 0.846 |
| Expected calibration error | ≤ 0.08 | 0.023 |
| FPR gap across regions | ≤ 0.10 (`thresholds.fairness_max_fpr_gap`) | 0.043 |
| RL unsafe-action rate | = 0 | 0 |
| ML-assisted precision@10% beats rules only | true | 0.975 vs. 0.938 |
| Safety tests (`tests/test_safety.py`) | 100% pass | pass |

**Online (shadow) evaluation:** compare analyst verdicts with `TruePositive` entailments and ML scores weekly.
Retrain or recalibrate if ECE > 0.08 or precision@k drops by more than 5 points against the last release.

## Rollback

All behaviour is versioned by three artifacts: the **image tag**, the **config version** and the **model artifact
hash**. Roll back the smallest one that fixes the problem.

1. **Stop the bleeding.** `POST /api/admin/kill-switch {"engaged": true}`, or set `ORCHIDSHIELD_KILL_SWITCH=1`.
   No tool runs, and pending approvals fail safe.
2. **Model regression.** Redeploy the previous `05_ml/artifacts/model.json` (its sha256 is in the model card and in
   `run_metadata`). A corrupt artifact is detected automatically, and the agent falls back to symbolic scoring.
   You can also set `features.ml_scoring: false`.
3. **Rule or config regression.** Revert `configs/config.yaml` or `03_logic/rules.yaml` to the previous tagged
   version and bump `app.config_version`. The config is validated at start-up, so a bad config fails fast instead
   of running.
4. **Code regression.** Redeploy the previous image tag (`orchidshield:<sha>` built by CI). The service is
   stateless apart from `data/runtime`. Keep that volume, so the audit chain continues.
5. Verify with `GET /ready` (audit chain OK) and run `python scripts/demo.py --base-url …` as a smoke test.
   Then release the kill switch.

## Incident runbook (agent misbehaviour)

| Symptom | First action |
|---|---|
| Unexpected disruptive proposal | Reject it in the approvals queue; inspect `entailments[].proof` and `knowledge_used` for the case |
| Agent stuck or slow | Check `iterations_total`; lower `orchestrator.max_iterations`; check tool timeouts |
| Many blocked proposals with injection flags | Expected under attack. Confirm `InjectionAttempt` escalations reach the on-call analyst |
| `/ready` = 503 | Audit chain broken: preserve the file, rotate keys, investigate tampering |

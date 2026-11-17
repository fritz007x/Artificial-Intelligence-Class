# OrchidShield: Security Triage & Response Recommendation Agent

Agentic AI capstone (Prompt 09). OrchidShield Cyber Defense is a **fictitious** organisation, and all data is
**synthetic**. The agent ingests alerts, asset criticality, authentication logs, known indicators and analyst
feedback, and runs a transparent six-stage reasoning cycle:

**PERCEPTS → PROPOSITIONS → KNOWLEDGE BASE → INFERENCE → ENTAILMENT → ACTION**

It ranks alerts, infers likely incidents, and recommends bounded responses. Disruptive actions (isolate, disable,
block, rotate) are **never executed autonomously**. They need an ENTAILED justification, have to pass hard
constraints, and require human approval. The loop has an audit trail and a kill switch.

```
raw obs ─► perceive.py ─► proposition.py ─► knowledge.py ─► inference.py ─► entailment.py ─► policy.py ─► actions.py
           validate,       ground literals,   facts + 18 Horn   FC · BN · HMM ·    entailed /       utility, hard      allow-listed,
           quality,        features,          rules + ontology  Kalman · MLP ·     contradicted /   constraints,       rate-limited,
           injection scan  contradictions     + provenance      A* · α-β · RAG ·   uncertain /      approval gate,     idempotent,
                                                                RL advisor         unknown          CSP schedule       audited
   ▲                                                                                                                      │
   └────────────────────────── orchestrator.py: tool results become new percepts (bounded loop, kill switch) ◄──────────┘
```

## Quick start

```bash
pip install -r requirements-dev.txt
python 05_ml/train.py            # synthetic data, train/eval, artifacts, model card (≈3 s, seeded)
python -m pytest -q              # 118 unit / integration / safety / fairness / e2e tests
python scripts/demo.py           # headless end-to-end demo of all six scenarios, approval and kill switch

# Service + UI
cp .env.example .env             # set ORCHIDSHIELD_{OPERATOR,APPROVER,ADMIN}_KEY
export $(grep -v '^#' .env | xargs)
PYTHONPATH=src uvicorn api.app:app --port 8000      # open http://localhost:8000 and paste a key
# local-only shortcut without keys: ORCHIDSHIELD_DEV_OPEN=1 (refused when ORCHIDSHIELD_ENV=prod)

# Container
docker build -t orchidshield .
docker run --env-file .env -p 8000:8000 orchidshield
```

## Repository map (deliverables)

| Deliverable | Path |
|---|---|
| Problem formulation, PEAS, stakeholders, goals and non-goals, autonomy boundary | `docs/01_problem_and_PEAS.md` |
| Simulation notebook: controlled experiments, algorithm comparisons, reproducible metrics | `notebooks/02_simulation.ipynb` (built by `scripts/build_notebook.py`; metrics in `data/results/simulation_metrics.json`) |
| Logic: proposition vocabulary, CNF/Horn rules, FOL predicates and ontology, entailment tests | `03_logic/` (`propositions.md`, `rules.yaml`, `cnf_rules.txt`, `ontology.yaml`, `entailment_tests.yaml`: 12 cases) |
| Bayesian and temporal artifacts | `04_uncertainty/` (`bayes_net.json`, `hmm.json`, README) |
| ML: preprocessing, training, evaluation, selected model, model card, saved artifact | `05_ml/` (`train.py`, `model_card.md`, `artifacts/model.json`, `artifacts/q_table.json`, `artifacts/metrics.json`) |
| Agent modules | `src/agent/{perceive,proposition,knowledge,inference,entailment,policy,actions,orchestrator}.py` |
| Algorithms | `src/reasoning/` (search, adversarial, csp, logic, fol, bayes, temporal, ml, rl, retrieval, evaluation) |
| Schemas, config, logging, audit, LLM interface | `src/core/` |
| FastAPI and web UI | `src/api/app.py`, `src/api/static/index.html` |
| Tests | `tests/` (schema, logic, reasoning, ml, safety/tool-policy, api, llm, e2e scenarios) |
| Deployment | `Dockerfile`, `.env.example`, `configs/config.yaml`, `../.github/workflows/orchidshield-ci.yml` (repo root) |
| Monitoring, rollback, evaluation thresholds | `docs/monitoring_and_rollback.md` |
| Responsible-AI report | `docs/responsible_ai.md` |
| Course topic evidence matrix (13 topics) | `docs/course_topic_matrix.md` |
| Future of AI and roadmap | `docs/future_and_roadmap.md` |
| Final demo script | `docs/demo_script.md`, `scripts/demo.py` |

## API

| Method | Path | Role |
|---|---|---|
| GET | `/` (UI), `/health`, `/ready`, `/metrics` | public |
| GET | `/api/scenarios`, `/api/cases`, `/api/cases/{id}`, `/api/tools`, `/api/audit`, `/api/audit/verify`, `/api/learning`, `/api/approvals` | any role |
| POST | `/api/scenarios/{name}/run`, `/api/cases` | operator, admin |
| POST | `/api/approvals/{proposal_id}` `{"approve": bool, "note": str}` | **approver only** |
| POST | `/api/admin/kill-switch` `{"engaged": bool}`; GET `/api/admin/status` | **admin only** |

Send `X-API-Key: <key>`. The UI shows the six stages as concise evidence records (proofs, rationales, scores).
It never shows secrets or hidden chain-of-thought, and it renders alert text with `textContent` only.

## Demonstration scenarios

| Scenario | What happens |
|---|---|
| `normal` | Malware with a known-bad hash on ws-005. `CompromisedHost` is ENTAILED. Isolation is held for approval, ticket and notify run automatically, and the EDR enrichment is fed back as a percept. A maintenance scan is recognised as benign. |
| `uncertain` | Impossible travel on a host with no EDR and low-confidence intel. P=0.43, so the agent **abstains**, enriches, watchlists and escalates. |
| `contradiction` | IOC match vs. analyst FP verdict gives TP and ¬TP. CONTRADICTED, containment blocked, escalated. |
| `constraint` | Domain controller compromise. `isolate_host(dc-01)` violates a hard constraint, so the alternatives are alpha-beta segmentation (`jump-01`), block the IP and disable the admin account. The service account can't be disabled, so its credentials are rotated instead. A CSP schedules the reviews. |
| `adversarial` | Prompt injection, a spoofed approval, a malformed alert and red-team tool requests are all rejected or blocked and audited. Nothing disruptive runs. |
| `learning` | Offline evaluation: ML-assisted precision@10% 0.975 vs. rules 0.938; RL beats the rule policy with 0 unsafe actions; FPR gap 0.043. |

## Definition of Done: how a reviewer traces an observation

Run `normal`, then in the case JSON (or the UI) follow alert `A-1002`:
1. `percepts[]`: the `alert` with `validation=valid`, its source, timestamp and quality.
2. `propositions[]`: `HasIndicator(A-1002, sha256:0bc3…)` with `derived_from=[percept id]`.
3. `knowledge_used[]`: `KnownBadIndicator(sha256:0bc3…)` (threat intel), `R_IOC`, `R_TP_EXEC_IOC`, `R_COMP_HOST`.
4. `inference.steps[]`: forward chaining `IOCMatch(A-1002)`, then BN, HMM, Kalman, ML, A*, fusion.
5. `entailments[]`: `TruePositive(A-1002)` and `CompromisedHost(ws-005)` ENTAILED, with the full proof.
6. `proposals[]`: `isolate_host(ws-005)` justified by `ContainmentWarranted(ws-005)`, pending approval. After
   approval, `tool_results[]` gets an entry and a new `tool_result` percept is added.

This trace is automated in `tests/test_scenarios_e2e.py::test_definition_of_done_trace`.

## Reproducibility

- Seed `app.seed=1337` drives the generator, splits, models and RL. `run_metadata` on each case records the
  version, config sha and seed.
- `05_ml/train.py` regenerates `data/synthetic/environment.json`, `data/scenarios/*.json`, the artifacts and the
  model card.
- CI (`../.github/workflows/orchidshield-ci.yml` at the repo root) retrains, applies the evaluation gate (`scripts/check_eval.py`), runs all tests,
  executes the notebook, runs the demo, then builds and probes the container.

## Safety summary

Allow-listed tools with typed parameters · disruptive tools refuse to run unless the proposal is APPROVED ·
disruptive proposals require ENTAILED (model confidence is never proof) · hard constraints (protected SIEM, never
isolate DCs, never disable service or break-glass accounts, external-only IP blocks, isolation budget, inventory
check) · bounded iterations · tool timeouts and rate limits · idempotency keys · kill switch (API, file or env) ·
hash-chained audit log · role-separated API keys from env · non-root container.

See `docs/responsible_ai.md` for the full report and limitations.

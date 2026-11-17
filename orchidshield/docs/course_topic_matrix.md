# Course topic integration matrix: required evidence

Each row lists where the topic is **implemented**, where it is **evaluated**, and how it is **used by the live
agent**. Paths are relative to the repository root.

| # | Topic | Implementation | Evidence / evaluation | Role in the agent |
|---|---|---|---|---|
| 1 | Foundations & intelligent agents | `docs/01_problem_and_PEAS.md` (PEAS, environment properties, rationality, agent type, sensors and actuators, performance measure) | Checkpoint A; utility values on every `ActionProposal` | The agent is a utility-based hybrid agent: `agent/orchestrator.py` |
| 2 | Problem formulation & uninformed search | `reasoning/search.py`: BFS, DFS, IDDFS, UCS over the attack-path formulation | Notebook §2 (cost, nodes expanded, optimal rate); `tests/test_reasoning.py::test_search_algorithms` | The state-space formulation drives blast radius and path to crown jewel |
| 3 | Informed search, heuristics & optimisation | A*, greedy best-first with an admissible and consistent hop heuristic; hill climbing and simulated annealing for segmentation | Notebook §2–3, heuristic admissibility/consistency check, exhaustive optimum comparison; `test_heuristic_admissible_and_consistent`, `test_local_search_close_to_optimum` | A* path appears in every inference trace (`astar_search` step) |
| 4 | Adversarial search & games | `reasoning/adversarial.py`: defender-vs-attacker containment game with minimax, alpha-beta (move ordering) and expectimax | Notebook §4 (values agree, node counts fall, e.g. 2,436 → 337 at depth 4); `test_alpha_beta_matches_minimax_with_fewer_nodes` | Chooses the **segmentation point** when the preferred isolation is forbidden (constraint scenario) |
| 5 | Constraint satisfaction | `reasoning/csp.py`: variables are response tasks, domains are (analyst, slot), with skills, seniority, precedence and separation of duties; backtracking, MRV, LCV, forward checking, AC-3/MAC, min-conflicts | Notebook §5 (assignments and backtracks, including an unsatisfiable instance); `test_csp_*` | Every pending approval and escalation is **scheduled** to an analyst and slot (`ActionProposal.schedule`) |
| 6 | Logical agents & automated inference | `reasoning/logic.py`: CNF, resolution refutation, truth-table model checking, PL forward chaining; `03_logic/cnf_rules.txt` | `03_logic/entailment_tests.yaml` (6 propositional cases; resolution ≡ model checking); `tests/test_logic.py` | The entailment stage answers explicit queries (`agent/entailment.py`) |
| 7 | First-order logic & knowledge representation | `reasoning/fol.py`: parsing, unification, forward chaining with justifications, SLD backward chaining; `03_logic/rules.yaml` (18 quantified Horn rules); `03_logic/ontology.yaml` (taxonomy and relations); `03_logic/propositions.md` | 6 FOL entailment cases (entailed, unknown, contradicted, uncertain); `test_backward_chaining_matches_forward_closure`, `test_unification` | The KB derives `TruePositive`, `CompromisedHost`, `ContainmentWarranted`, … with proofs shown in the UI |
| 8 | Probabilistic reasoning & Bayesian networks | `04_uncertainty/bayes_net.json`, `reasoning/bayes.py` (exact enumeration, marginalisation of missing evidence, leave-one-out explanation) | Notebook §8 (evidence table, exact vs. sampled check); `test_bn_*` (monotonicity, explaining away, missing data) | P(Compromise) per host drives abstention, support levels and risk |
| 9 | Temporal reasoning, HMMs & Kalman filters | `04_uncertainty/hmm.json`, `reasoning/temporal.py` (forward filter, Viterbi, sampling, prediction; 1-D Kalman) | Notebook §9 (Viterbi 0.77 vs. naive 0.63; Kalman false-alarm and detection rates); `test_hmm_*`, `test_kalman_flags_burst` | Kill-chain stage and alert-rate bursts feed risk fusion |
| 10 | Machine learning | `05_ml/train.py`, `reasoning/ml.py` (shared feature engineering, stratified train/val/test, 5-fold CV, majority/rules/sklearn baselines, calibration, fairness) | `05_ml/artifacts/metrics.json`, `05_ml/model_card.md`; `scripts/check_eval.py` CI gate; `tests/test_ml.py` | P(true positive) per alert in fusion. It is **never proof**. |
| 11 | Neural networks & reinforcement learning | Numpy MLP (Adam, early stopping), selected by validation; tabular Q-learning advisor (`reasoning/rl.py`) with masked actions | Model card; RL vs. rule policy (reward, missed compromises, unsafe rate 0); `test_rl_masks_containment_until_enriched_high_risk` | MLP is the deployed classifier; the RL advisor suggestion is shown in the trace (advisory only) |
| 12 | Contemporary & responsible AI | RAG over SOC playbooks (`reasoning/retrieval.py`); optional Claude narrator with grounding check (`core/llm.py`); `docs/responsible_ai.md` | `tests/test_llm.py` (grounding, fallback), `test_prompt_injection_quarantined`, adversarial scenario, fairness tests | Retrieved playbooks are cited in "knowledge used"; the narrative summary has no tools |
| 13 | ANNs & the future of AI | `docs/future_and_roadmap.md`: current design compared with plausible future enhancements, limitations, sustainability, scalability, roadmap | Checkpoint E | Guides the next iterations |

## Demonstration scenarios → tests

| Scenario | Data | Test |
|---|---|---|
| Normal | `data/scenarios/normal.json` | `tests/test_scenarios_e2e.py::test_normal_case_supported_action` |
| Uncertain | `uncertain.json` | `::test_uncertain_case_abstains` |
| Contradiction | `contradiction.json` | `::test_contradiction_case_flags_and_avoids_entailment` |
| Constraint | `constraint.json` | `::test_constraint_case_selects_feasible_alternative` |
| Adversarial / robustness | `adversarial.json` | `::test_adversarial_case_gate_blocks` and `tests/test_safety.py` |
| Learning | `05_ml/artifacts/metrics.json` | `::test_learning_case_report`, `tests/test_ml.py::test_learning_case_improves_without_violating_guardrails` |

## Assessment checkpoints

| Checkpoint | Where to look |
|---|---|
| A: Design review | `docs/01_problem_and_PEAS.md`, `03_logic/propositions.md`, `core/schemas.py` |
| B: Reasoning review | Notebook §2–9, `03_logic/`, `04_uncertainty/`, `tests/test_reasoning.py`, `tests/test_logic.py` |
| C: Learning review | `05_ml/`, notebook §10, `tests/test_ml.py` |
| D: Agent integration review | `src/agent/`, `tests/test_scenarios_e2e.py`, `tests/test_safety.py`, the audit trail |
| E: Production review | `src/api/`, `Dockerfile`, `../.github/workflows/orchidshield-ci.yml` (repo root), `docs/monitoring_and_rollback.md`, `docs/responsible_ai.md`, `docs/demo_script.md` |

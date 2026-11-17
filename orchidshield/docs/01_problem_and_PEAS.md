# 01 · Problem formulation & PEAS

*Phase 1, Step 1 (Output Object Definition). Checkpoint A, design review.*

## 1. Product

**OrchidShield Security Triage & Response Recommendation Agent.** OrchidShield Cyber Defense is a fictitious
organisation. The service takes in security telemetry, ranks alerts, infers likely incidents, and recommends
**bounded** responses. Every disruptive response needs explicit human approval. Every recommendation can be traced
from the raw observation to the proposed action.

## 2. Stakeholders and users

| Stakeholder | Need | How the agent serves them |
|---|---|---|
| Tier-1 SOC analyst (primary user) | A shorter, better-ordered queue and a clear "why" | Ranked risk list, incident grouping, evidence records |
| Tier-2 / incident responder (approver) | A safe, fast containment decision | Pending-approval queue with entailment proof, utility, alternatives and schedule |
| SOC manager | Workload, staffing, SLA | CSP schedule, `/metrics` |
| Platform admin | Safe operation | Kill switch, audit chain verification, config/version metadata |
| Asset owners and business units | Avoid unnecessary outages | Hard constraints (never isolate a DC or the SIEM), maintenance-window reasoning |
| Users whose accounts are monitored | Fair, non-punitive treatment | Protected attributes excluded, FPR-parity audit, no HR/disciplinary outputs |
| Course reviewers | Evidence for each course topic | `docs/course_topic_matrix.md`, notebook, tests |

## 3. Goals and non-goals

**Business/learning goals**
1. Put true incidents at the top of the queue: precision@10% above the rules-only baseline (measured in `05_ml`).
2. Entailed containment proposals reach an approver within one iteration, with a proof attached.
3. Zero disruptive actions without approval, zero actions on protected assets, and a complete audit trail.
4. Calibrated abstention: when P(compromise) falls in the band [0.35, 0.70) and nothing is proven, the agent
   abstains and gathers evidence.
5. Show all 13 course topics with runnable evidence.

**Non-goals**
- Autonomous containment. The agent never isolates, disables or blocks on its own.
- Real telemetry or production integration. Everything is simulated and synthetic.
- Attribution of threat actors, and any HR, disciplinary or legal judgement about people.
- Replacing analysts. The agent is decision support.

## 4. PEAS specification

| | Specification |
|---|---|
| **Performance measure** | Weighted utility: damage averted (P(compromise) × asset impact × action benefit) minus disruption cost (`configs/config.yaml → policy`). Also precision@k, ROC-AUC, ECE, calibrated-abstention rate, false-containment rate (target 0), approval latency, unsafe-action rate (target 0), and FPR gap across regions (≤ 0.10). |
| **Environment** | Synthetic enterprise network (`src/sim/generator.py`): 25 assets across DMZ, corp, prod and mgmt zones; 27 users including privileged, service and break-glass accounts; a threat-intel feed with expired and low-confidence entries; an adversary moving along network edges; and analysts giving feedback. |
| **Actuators** | Allow-listed tools only (`src/agent/actions.py`). Non-disruptive: `query_asset_db`, `edr_process_tree`, `similar_alerts`, `create_ticket`, `add_watchlist`, `notify_analyst`. Approval-gated: `isolate_host`, `disable_account`, `block_ip`, `rotate_credentials`. |
| **Sensors** | SIEM/EDR/IDS alerts, authentication logs, asset inventory (CMDB), directory, threat-intel indicators, analyst feedback, and tool results (the observe-result loop). |

### Environment properties

| Property | Value | Consequence for design |
|---|---|---|
| Observability | **Partial.** The true compromise state is hidden, and some hosts have no EDR | Bayesian network with marginalisation, HMM over hidden stage |
| Determinism | **Stochastic** (noisy sensors, attacker choices) | Probabilities, expectimax, abstention |
| Episodic or sequential | **Sequential** (kill chain) | HMM / Kalman, iterative loop |
| Static or dynamic | **Dynamic** (attacker acts while we deliberate) | Bounded iterations, adversarial search |
| Discrete or continuous | Discrete decisions over continuous scores | Thresholds, utility |
| Agents | **Multi-agent, adversarial** (attacker vs. defender) and **cooperative** (analysts) | Alpha-beta, CSP staffing |
| Known or unknown | Partially known (rules are known, base rates are learned) | Hybrid symbolic plus learned |

### Agent type and rationality

This is a **hybrid utility-based agent with a knowledge base and learning components**. It acts rationally within
its safety constraints. Among allowable actions it picks the one with the highest expected utility, given its
percepts and built-in knowledge. Any action that breaks a hard constraint or lacks the required entailment is not
allowable, whatever its utility.

## 5. Problem formulation (search view)

- **States:** (attacker-controlled host set, isolated host set).
- **Initial state:** the host(s) where compromise is entailed or suspected.
- **Goal (attacker):** reach a crown jewel with criticality ≥ 4. **Goal (defender):** keep the critical assets the
  attacker can reach to a minimum.
- **Actions:** the attacker traverses an edge, and the defender isolates a host or passes.
- **Transition model:** deterministic graph moves. Edge cost is exploitation difficulty (zone crossings cost more).
- **Path cost:** the sum of edge costs, used for A*/UCS shortest attack paths and the blast radius.

## 6. Autonomy boundary

| Class | Examples | Autonomy |
|---|---|---|
| Observe and enrich | CMDB lookup, EDR process tree, similar-case lookup | **Automatic**, logged, rate-limited, idempotent |
| Communicate | Ticket, watchlist, notify, escalate | **Automatic**, logged |
| Disrupt | Isolate host, disable account, block IP, rotate credentials | **Recommend only**. Needs an ENTAILED justification, passing hard constraints, and **human approval** through the approver role. Kill switch and constraints are re-checked at execution. |
| Forbidden | Anything not in the allow-list (deleting logs, running commands); any action on `siem-01`; isolating domain controllers; disabling service or break-glass accounts | **Blocked**, audited, and escalated |

## 7. The six-stage agent cycle

```
 raw observation ─► 1 PERCEPT ─► 2 PROPOSITION ─► 3 KNOWLEDGE BASE ─► 4 INFERENCE ─► 5 ENTAILMENT ─► 6 ACTION
   (alert, auth,     validated,    ground literals   facts + Horn rules  FC, BN, HMM,    entailed /       plan → policy &
    intel, feedback) quality-       + features,      + ontology +        Kalman, ML/NN,  contradicted /   approval gate →
                     scored,        contradictions   constraints +       A*, α-β, RAG,   uncertain /      allow-listed tool
                     injection-                      model metadata      RL advisor,     unknown          → result
                     scanned                         (with provenance)   risk fusion                       │
       ▲                                                                                                    │
       └──────────────────────────── tool result becomes a new percept (observe-result) ◄──────────────────┘
```

| Stage | Module | Output schema |
|---|---|---|
| Percepts | `src/agent/perceive.py` | `Percept` |
| Propositions | `src/agent/proposition.py` | `Proposition`, `Contradiction`, feature vectors |
| Knowledge base | `src/agent/knowledge.py` + `03_logic/` | `KnowledgeFact`, `KnowledgeRule`, `ModelMetadata` |
| Inference | `src/agent/inference.py` | `InferenceTrace` (`InferenceStep`, `RiskAssessment`, `Incident`) |
| Entailment | `src/agent/entailment.py` | `EntailmentResult` |
| Action | `src/agent/policy.py`, `src/agent/actions.py`, `src/agent/orchestrator.py` | `ActionProposal`, `ToolResult`, `AuditEvent` |

## 8. Risks (design review)

| Risk | Mitigation |
|---|---|
| Wrong containment causes an outage | Approval gate, ENTAILED requirement, hard constraints, isolation budget, kill switch |
| Prompt injection through alert text | Text is data only, injection scanner, quarantine, the narrator has no tools, grounding check |
| Spoofed tool results or approvals | Source/kind authorisation in perception; approvals only through the approver API key |
| Contradictory evidence | Conflict detection in the closure, CONTRADICTED status, escalation |
| Model drift or failure | Integrity hash, feature-list check, symbolic fallback, eval thresholds in CI |
| Bias against regions or departments | Protected attributes excluded, FPR-parity test, identical-evidence policy test |
| Audit tampering | Hash-chained log, `/ready` fails if the chain is broken |

## 9. Baseline plan

| Capability | Baseline | Agent component |
|---|---|---|
| TP classification | Majority class; symbolic rules | Logistic regression; MLP (selected by validation) |
| Attack path | BFS / DFS | UCS, A* |
| Containment choice | Isolate the compromised host | Alpha-beta segmentation plus constraint-aware alternatives |
| Scheduling | Plain backtracking | MRV + LCV + FC / MAC; min-conflicts |
| Triage tempo | Fixed rule policy | Q-learning advisor (masked, advisory) |
| Stage tracking | Most recent alert type | HMM filter / Viterbi |

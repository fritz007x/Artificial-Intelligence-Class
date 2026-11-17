# Artificial neural networks and the future of AI: comparison, limitations, roadmap

*Course topic 13.*

## Current design and a plausible future design

| Dimension | Current (v1.0) | Plausible future enhancement | Trade-off |
|---|---|---|---|
| Alert scoring | 16-feature numpy MLP (1 hidden layer) chosen over logistic regression by validation AUC; calibrated | Sequence or graph neural network over the entity graph (hosts, users, IPs) and alert timelines, e.g. a temporal GNN | Better cross-alert pattern recognition, but less transparent. It would need graph-level explanations, and drift monitoring matters more. |
| Temporal state | Separate HMM (stage) + Kalman (rate) | Dynamic Bayesian network coupling stage, compromise and sensor reliability; or a learned state-space model | One coherent belief state; harder to elicit and validate |
| Language | Template narrator; optional Claude narrator with grounding check; TF-IDF RAG | Claude as an *investigation copilot* that drafts hypotheses and queries, still with no write tools. Every hypothesis must pass the entailment stage before it can affect actions. | More analyst leverage; needs strict tool isolation, evaluation for hallucination, and cost controls |
| Planning | Single-step utility + alpha-beta segmentation | POMDP or MCTS over containment sequences with an attacker model learned from red-team exercises | Better multi-step plans; the state space grows exponentially, so it needs sampling and careful safety constraints |
| Learning from analysts | Offline retrain | Active learning: ask analysts about the most uncertain alerts, then calibrate online | Less labelling effort; risk of feedback loops, so we would need audit sampling |
| RL | Tabular Q-learning advisor in simulation, masked actions | Offline RL from logged decisions, with conservative policy constraints | Learns tempo from real data. It must stay advisory until offline evaluation shows it is safe. |
| Knowledge | Hand-written Horn rules | Rules mined from incident history (inductive logic programming), with human review before merge | Scales the KB, but mined rules can encode historical bias |

**Design principle that stays:** neural and agentic components may *propose* and *rank*. Only the symbolic
entailment stage, hard constraints and human approval can *authorise* disruptive actions.

## Limitations of v1.0

1. The data is synthetic, so the reported metrics show the pipeline works, not real-world performance.
2. BN and HMM parameters are elicited by hand and fixed.
3. Contradiction handling is conservative: any conflict on a proof blocks entailment. There is no belief revision
   or source-reliability weighting yet.
4. Each case has its own knowledge base. Cross-case memory is limited to the watchlist, tickets and the audit log.
5. The approver is a role, not a person. There is no SSO and no two-person rule for the highest-impact actions.
6. Tools are simulated. A real EDR or firewall integration needs idempotent upstream APIs and rollback hooks.

## Sustainability and scalability

- **Compute:** the full pipeline (train + 119 tests + notebook) runs in under a minute on a laptop CPU. Inference is
  milliseconds per case, with no GPU. Neural and LLM enhancements should be justified by measured gains, so the
  energy and cost per decision get tracked alongside accuracy.
- **LLM usage:** behind a feature flag, called once per case on structured evidence, with low effort and capped tokens.
- **Scaling the KB:** naive forward chaining is O(rules × facts^k). For more than 10⁵ facts, move to a Rete-style
  incremental engine or Datalog (e.g. Soufflé), and index facts by predicate and argument.
- **Scaling the service:** stateless API workers with `data/runtime` on shared storage, and audit events streamed
  to an append-only store. Approvals move to a queue.

## Roadmap

| Milestone | Content | Exit criterion |
|---|---|---|
| v1.1 | SSO/OIDC approvers, two-person rule for DC- or prod-wide actions, per-approver audit | Security review pass |
| v1.2 | Source-reliability-weighted belief revision for contradictions; DBN temporal model | The contradiction scenario resolves automatically when a trusted EDR verdict arrives |
| v1.3 | Shadow mode on labelled replay data; active learning loop | Precision@k and ECE within thresholds on real labels |
| v2.0 | Investigation copilot (LLM + read-only tools); temporal GNN scorer behind the same gates | Offline evaluation shows improvement with no guardrail regressions |

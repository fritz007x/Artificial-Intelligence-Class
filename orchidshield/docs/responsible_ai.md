# Responsible AI report

## Scope and data

Everything is **synthetic** and course-created (`src/sim/generator.py`). IPs come from the RFC 5737 documentation
ranges, domains use the reserved `.test` TLD, and hashes are seeded pseudo-random. No real organisation, person
or indicator appears. No secrets are stored in source control: API keys and the optional `ANTHROPIC_API_KEY` come
only from the environment (`.env.example`).

## Human oversight and accountability

| Control | Implementation | Verified by |
|---|---|---|
| Disruptive actions need human approval | `ProposalStatus.PENDING_APPROVAL`; `ToolRegistry.execute` refuses disruptive tools unless `APPROVED` | `test_disruptive_tool_requires_approved_status` |
| Approvals only from the approver role | Separate API key; operators and admins get 403 | `test_least_privilege` |
| Disruption needs logical proof, not model confidence | `policy.disruptive_requires_entailment`; the entailment checker never promotes probabilistic support to ENTAILED | `test_model_confidence_is_never_proof` |
| Separation of duties | The CSP assigns approval reviews to a different analyst than triage | `policy.schedule`, `test_csp_*` |
| Kill switch | API (admin), sentinel file, or env var; checked before every stage, tool call and approval | `test_kill_switch_*`, `test_approval_rechecks_kill_switch` |
| Accountability | Hash-chained audit log of percepts, entailments, blocks, approvals and tool calls; `/ready` fails on a broken chain | `test_audit_chain_detects_tampering`, `test_every_case_is_audited` |

## Explainability

Every decision carries an evidence trail rather than hidden chain-of-thought:
- **Entailment proofs:** rule-by-rule derivations down to percept-level facts with provenance.
- **Inference steps:** one concise rationale per method. For the Bayesian network this includes the top evidence
  drivers. The ML step lists the feature values used.
- **Action justifications:** the entailment query, utility, blocked reasons, and alternatives considered.
- Global model explanations (logistic coefficients) and calibration are in `05_ml/model_card.md`.

## Fairness and bias

- Protected or sensitive attributes (`region`, `department`) are **never features** (`test_protected_attributes_are_not_features`).
- **FPR parity across regions** on held-out data: gap 0.043 for the selected MLP, against a threshold of 0.10.
  This is enforced by `scripts/check_eval.py` in CI.
- The generator deliberately injects benign APAC VPN "impossible travel" noise. That makes `impossible_travel` the
  main disparity risk, and it is called out in the model card for monitoring.
- **Policy-level parity:** identical evidence for users in different regions produces identical decisions
  (`test_policy_treats_regions_identically`).
- The system never makes HR or disciplinary judgements about people. Account actions are containment of
  credentials, they need approval, and service and break-glass accounts are protected.

## Privacy

User identifiers are pseudonymous. The narrator only receives structured evidence (no raw logs). Audit details store
ids, not free text. Free text is truncated and sanitised at perception.

## Security of the agent itself

| Threat | Mitigation |
|---|---|
| Prompt injection in alert text | Injection and markup scanner, quarantined text, the `InjectionAttempt` fact, escalation. Free text never drives control flow. The narrator has no tools and is grounding-checked. |
| Spoofed tool results or approvals | Perception authorises the source per kind (only the orchestrator emits `tool_result`); approvals go only through the authenticated approver endpoint |
| Unsafe or unknown tools | Allow-list registry; typed parameter schemas; parameterised SQL; file writes confined to `data/runtime` |
| Acting on hallucinated targets | Disruptive targets must exist in the inventory or directory; LLM text must mention only entities in evidence |
| Resource abuse | API and tool rate limits, tool timeouts, bounded iterations, request size limits |
| Container | Non-root user, slim image, read-only code, health check, `ORCHIDSHIELD_ENV=prod` refuses dev-open mode |

## Hallucination

The template narrator (the default) is deterministic. The optional Claude narrator (`features.llm_narrative`) uses
the official SDK and is told to summarise only the given evidence. Its output is rejected if it mentions any host,
user or alert not in the evidence, and on API errors or refusals the agent falls back to the template
(`tests/test_llm.py`).

## Known limitations

- The models are trained on synthetic distributions. Calibration and fairness must be re-measured on real labelled data.
- BN and HMM parameters are hand-elicited and only sanity-checked against the generator.
- The isolation budget and hard constraints are static configuration. A real deployment would need change-management integration.
- The approver identity is a role key, not an individual. SSO/OIDC with per-person identity is on the roadmap.

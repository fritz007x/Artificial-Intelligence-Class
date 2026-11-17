# Final demonstration script (about 15 minutes)

As the instructor notes suggest, the demo first shows the symbolic six-stage cycle and then layers in the search,
probabilistic, temporal and learned modules. Grade the evidence trail, not just the predictions.

**Setup (before the demo)**
```bash
cp .env.example .env    # fill the three role keys
docker build -t orchidshield . && docker run --env-file .env -p 8000:8000 orchidshield
# or locally: pip install -r requirements-dev.txt && ORCHIDSHIELD_DEV_OPEN=1 PYTHONPATH=src uvicorn api.app:app
```
Open http://localhost:8000 and paste the operator key. Keep a terminal open for `scripts/demo.py --base-url http://localhost:8000`.

| # | Time | Do | Say / point at |
|---|---|---|---|
| 1 | 1 min | Show `docs/01_problem_and_PEAS.md` §6–7 | The autonomy boundary: observe and communicate run automatically, disruption needs approval, and some actions are forbidden |
| 2 | 3 min | Run **normal** in the UI | **1 Percepts**, all valid. **2 Propositions** such as `HasIndicator(A-1002, sha256:…)`. **5 Entailment**: `CompromisedHost(ws-005)` ENTAILED, with the proof chain R_IOC → R_TP_EXEC_IOC → R_COMP_HOST. **6 Action**: `isolate_host(ws-005)` *pending approval* while ticket and notify ran automatically. Point out the second iteration: the EDR enrichment result came back as a percept (`MalwareConfirmed(ws-005)`). The maintenance scan on srv-app-02 is refuted (`¬TruePositive`), so the agent only monitors it. |
| 3 | 2 min | Same case, **4 Inference** panel | Layered modules: BN P(compromise)=0.94 with its top drivers, HMM stage, Kalman z, ML P(TP), A* path to a crown jewel, fused risk, RAG playbooks, RL advisor (advisory) |
| 4 | 2 min | Run **uncertain** | No EDR on ws-013, so EDR evidence is *marginalised*. Low-confidence intel. P=0.43 sits in the abstention band, so the agent **abstains**, enriches (CMDB plus similar cases), adds to the watchlist and escalates. No disruptive proposal. |
| 5 | 2 min | Run **contradiction** | The analyst FP verdict and the IOC match give both `TruePositive(A-3001)` and `¬TruePositive(A-3001)`. Entailment is CONTRADICTED even though a proof exists, so isolation is **blocked** and the conflict is escalated |
| 6 | 2 min | Run **constraint** | `isolate_host(dc-01)` violates "never isolate a domain_controller", so alpha-beta picks the segmentation point `jump-01`. Also proposed: block the external IP and disable `adm-01`. `disable_account(svc-backup)` is blocked and `rotate_credentials` is proposed instead. The CSP schedule assigns each review to a senior analyst and slot. |
| 7 | 2 min | Run **adversarial** | The injection text is quarantined and flagged. The spoofed "approval" tool_result and the malformed alert are rejected. `delete_audit_log` is not allow-listed, `siem-01` is protected, `dc-01` is forbidden, and `ws-011` lacks entailment. Everything is blocked and audited, and a security review is escalated. |
| 8 | 1 min | Run **learning** | precision@10%: rules 0.938 vs. rules+ML 0.975. RL beats the rule policy with 0 unsafe actions. FPR gap 0.043 ≤ 0.10. Then open `05_ml/model_card.md`. |
| 9 | 1 min | Paste the **approver** key and approve `isolate_host(ws-005)` | Approval runs the allow-listed tool, the result is fed back as a percept, and a second approve gives 409 (idempotent). Show that an operator key gets 403. |
| 10 | 1 min | Paste the **admin** key and press **Kill switch**, then run normal again | The loop halts before *observe*, no tool calls are made, and pending approvals fail safe. Release it. |
| 11 | 30 s | `GET /api/audit/verify`, `GET /metrics` | The hash chain is intact, and the counters back the monitoring plan |

**Reproduce:** `python 05_ml/train.py && python -m pytest -q && jupyter nbconvert --execute notebooks/02_simulation.ipynb`.
The same seed gives the same numbers.

**Headless version:** `python scripts/demo.py` prints the same story with no server running.

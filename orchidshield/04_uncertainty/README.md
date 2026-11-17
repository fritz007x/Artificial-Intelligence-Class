# 04 · Uncertainty: Bayesian and temporal reasoning artifacts

| File | What it is | Used by |
|---|---|---|
| `bayes_net.json` | Per-host compromise Bayesian network (8 binary nodes, CPTs, documentation) | `src/reasoning/bayes.py`, `src/agent/inference.py` |
| `hmm.json` | Kill-chain hidden Markov model (5 hidden stages, 6 observation symbols) | `src/reasoning/temporal.py` |

## Bayesian network

```
 ThreatActive ──► Compromise ──┬──► IOCMatch
                               ├──► EDRDetection      (unobserved on hosts without EDR)
                               ├──► LateralSignal
                               ├──► AnalystFP
 Maintenance ──────────────────┴──► AuthAnomaly       (maintenance "explains away" auth anomalies)
```

- **Prior:** P(Compromise) ≈ 0.087 (from P(ThreatActive) = 0.10).
- **Inference:** exact enumeration. Evidence that is missing is *marginalised*, not imputed. This is how the
  agent handles incomplete percepts (the uncertain scenario uses a host with no EDR).
- **Decision use:** a posterior ≥ 0.70 is "sufficiently supported", which allows non-disruptive actions only.
  A posterior in [0.35, 0.70) without a proof triggers **calibrated abstention**. The posterior is never promoted
  to ENTAILED.
- **Explanations:** `BayesNet.explain` gives leave-one-out evidence impact, which appears in the inference trace.
- **Validation:** `02_simulation.ipynb §8` compares exact posteriors with forward-sampled frequencies, and
  `tests/test_reasoning.py` checks monotonicity, explaining away, and marginalisation.

| Evidence | P(Compromise) |
|---|---|
| none | 0.087 |
| IOC match | 0.69 |
| IOC + EDR detection | 0.97 |
| auth anomaly (no maintenance) | 0.57 |
| auth anomaly during maintenance | 0.18 |
| IOC + analyst false-positive verdict | 0.16 |

## HMM (kill-chain stage)

States: `benign → recon → initial_access → lateral_movement → exfiltration` (left-to-right with a small chance of
regression). Observations are the kill-chain class of each successive alert, plus `quiet` for each empty 15-minute bin.

- **Filtering** (forward algorithm) gives P(stage | alerts so far). The *expected stage severity* feeds risk fusion.
- **Viterbi** decodes the most likely stage path (notebook §9: about 0.77 accuracy vs. 0.63 for the naive mapping).
- **Prediction:** `HMM.predict(belief, k)` projects the stage k bins ahead.

## Kalman filter (alert-rate tracking)

A local-level model: x_t = x_{t−1} + w with Q = 0.05, and z_t = x_t + v with R = 1. A normalised innovation above
3σ marks a burst the baseline doesn't explain, and adds +0.05 to fused risk. Notebook §9 measures a false-alarm
rate of about 1.6% and a detection rate of 100% for a +6 spike.

## Why these models (justification)

The true compromise state is never observed directly, so a generative model with explicit conditional independences
is easier to audit than a black-box score, and missing sensors are handled correctly. The kill chain is sequential,
which an HMM captures with few parameters. A dynamic Bayesian network that joins the two is on the roadmap
(`docs/future_and_roadmap.md`).

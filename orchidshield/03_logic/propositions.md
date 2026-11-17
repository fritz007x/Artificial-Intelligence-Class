# Proposition vocabulary & FOL predicates

Every percept is translated (by `src/agent/proposition.py`) into ground literals from this
vocabulary. Each proposition is *testable*: the right column names the observation that
would confirm or refute it.

## Observed predicates (from percepts)

| Predicate | Arity / types | Produced from | Testable by |
|---|---|---|---|
| `Alert(a)` | Alert | alert percept | SIEM record exists |
| `AlertType(a, t)` | Alert × AlertType | alert.alert_type | detection rule id |
| `Severity(a, s)` | Alert × {low,medium,high,critical} | alert.severity | vendor severity |
| `OnAsset(a, h)` | Alert × Asset | alert.asset_id | CMDB lookup |
| `ForUser(a, u)` | Alert × User | alert.user_id | directory lookup |
| `HasIndicator(a, i)` | Alert × Indicator | alert.indicators | raw event fields |
| `FromIP(a, ip)` | Alert × IP | alert.src_ip | netflow |
| `ExternalIP(ip)` | IP | non-RFC1918 address | ip classification |
| `Asset(h)`, `IsA(h, c)` | Asset × AssetClass | asset inventory | CMDB |
| `Critical(h)` | Asset | criticality ≥ `thresholds.critical_asset_min` | business impact analysis |
| `ConnectedTo(h, g)` | Asset × Asset | network graph | firewall rules |
| `InMaintenance(h)` | Asset | change calendar | change ticket |
| `NoEDR(h)` | Asset | inventory | EDR console |
| `KnownBadIndicator(i)` | Indicator | intel feed, conf ≥ 0.6 and unexpired | intel re-query |
| `ExpiredIndicator(i)` | Indicator | intel feed, expired | intel re-query |
| `LoginSuccess(u, h)` | User × Asset | auth log | auth log |
| `SuccessAfterFailures(u, h)` | User × Asset | ≥5 failures then success within 30 min | auth log window |
| `ImpossibleTravel(u)` | User | consecutive logins faster than 900 km/h | geo-IP |
| `Privileged(u)`, `ServiceAccount(u)`, `BreakGlass(u)` | User | directory | IAM |
| `AnalystVerdictTP(a)`, `AnalystVerdictFP(a)` | Alert | analyst feedback | case record |
| `MalwareConfirmed(h)`, `¬MalwareConfirmed(h)` | Asset | EDR enrichment tool | forensic triage |
| `InjectionAttempt(a)` | Alert | text sanitizer | content scan |

## Derived predicates (from `rules.yaml`)

`AlertClass(a,c)`, `IOCMatch(a)`, `TruePositive(a)` / `¬TruePositive(a)`, `CompromisedHost(h)`,
`CompromisedAccount(u)`, `AtRisk(h)`, `HighPriorityIncident(h)`, `CrownJewelExposed(g)`,
`ContainmentWarranted(h)`, `AccountContainmentWarranted(u)`, `BlockWarranted(ip)`.

## Quantified rules (FOL form)

- ∀x,y,z. SubtypeOf(x,y) ∧ SubtypeOf(y,z) ⇒ SubtypeOf(x,z)
- ∀a,t,c. AlertType(a,t) ∧ SubtypeOf(t,c) ⇒ AlertClass(a,c)
- ∀a,i. HasIndicator(a,i) ∧ KnownBadIndicator(i) ⇒ IOCMatch(a)
- ∀a. IOCMatch(a) ∧ AlertClass(a, execution) ⇒ TruePositive(a)
- ∀a,u,h. AlertClass(a, credential_attack) ∧ ForUser(a,u) ∧ OnAsset(a,h) ∧ SuccessAfterFailures(u,h) ⇒ TruePositive(a)
- ∀a. AnalystVerdictFP(a) ⇒ ¬TruePositive(a)
- ∀a,h. TruePositive(a) ∧ OnAsset(a,h) ⇒ CompromisedHost(h)
- ∀h,g. CompromisedHost(h) ∧ ConnectedTo(h,g) ∧ Critical(g) ⇒ CrownJewelExposed(g)
- ∃-queries are answered by unification, e.g. `CompromisedHost(?h)` returns the first binding.

## Integrity constraints (checked by the KB, not derivable)

- ⊥ ⇐ P(x̄) ∧ ¬P(x̄)  — any literal and its negation both in the closure is a **conflict**.
  Entailment refuses to call a conclusion ENTAILED if its proof touches a conflicted literal.
- Hard action constraints live in `configs/config.yaml → policy` and are enforced by
  `src/agent/policy.py` (e.g. never isolate a `domain_controller`, never disable a `break_glass` account).

## Unification

`src/reasoning/fol.py::unify` performs most-general unification over function-free terms
(variables `?x`). Example: `unify(TruePositive(?a), TruePositive(A-101)) = {?a ↦ A-101}`.
Backward chaining renames rule variables apart before each use.

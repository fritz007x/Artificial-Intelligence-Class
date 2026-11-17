"""Tabular Q-learning advisor for triage tempo.

MDP (simulated, never touches production):
  state  = (risk_bucket 0-2, stage_bucket 0-2, critical 0/1, enriched 0/1)
  actions= monitor | enrich | escalate | request_containment
  reward = +10*crit_weight for escalating/containing a real compromise,
           -analyst/disruption cost for each action, -8 for requesting containment on
           a benign host, -6*crit_weight per step a real compromise is left monitored.
Safe-exploration boundary: `request_containment` is masked out unless the risk bucket
is high (2) AND the host has been enriched, during both exploration and exploitation.
The resulting policy is *advisory only* — it can suggest but never execute; any
containment still goes through the entailment + approval gate."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

ACTIONS = ["monitor", "enrich", "escalate", "request_containment"]
State = tuple[int, int, int, int]


def allowed_actions(s: State) -> list[str]:
    risk, _stage, _crit, enriched = s
    acts = ["monitor", "enrich", "escalate"]
    if risk == 2 and enriched:
        acts.append("request_containment")
    return acts


@dataclass
class TriageSim:
    rng: random.Random
    horizon: int = 6

    def reset(self) -> State:
        self.compromised = self.rng.random() < 0.3
        self.critical = int(self.rng.random() < 0.35)
        self.t = 0
        self.stage = 1 if self.compromised else 0
        self.enriched = 0
        self.risk = self._noisy_risk(0.35)
        return self.state()

    def _noisy_risk(self, noise: float) -> int:
        true = 2 if self.compromised else 0
        if self.rng.random() < noise:
            return 1
        if self.rng.random() < noise / 3:
            return 2 - true
        return true

    def state(self) -> State:
        return (self.risk, min(self.stage, 2), self.critical, self.enriched)

    def step(self, a: str) -> tuple[State, float, bool]:
        w = 2.0 if self.critical else 1.0
        self.t += 1
        if a == "enrich":
            self.enriched = 1
            self.risk = self._noisy_risk(0.08)
            r = -0.5
        elif a == "escalate":
            r = (6.0 * w if self.compromised else -2.0) - 1.0
            return self.state(), r, True
        elif a == "request_containment":
            r = (10.0 * w if self.compromised else -8.0) - 1.5
            return self.state(), r, True
        else:  # monitor
            r = -6.0 * w * 0.5 if self.compromised else 0.2
        if self.compromised and self.rng.random() < 0.4:
            self.stage += 1
        return self.state(), r, self.t >= self.horizon


def q_learning(episodes: int = 20000, alpha: float = 0.1, gamma: float = 0.95, eps: float = 0.2,
               seed: int = 7) -> tuple[dict[State, dict[str, float]], list[float]]:
    rng = random.Random(seed)
    env = TriageSim(rng)
    Q: dict[State, dict[str, float]] = {}
    curve, window = [], []
    for ep in range(episodes):
        s, done, total = env.reset(), False, 0.0
        while not done:
            acts = allowed_actions(s)
            q = Q.setdefault(s, {a: 0.0 for a in ACTIONS})
            a = rng.choice(acts) if rng.random() < eps else max(acts, key=lambda x: q[x])
            s2, r, done = env.step(a)
            q2 = Q.setdefault(s2, {x: 0.0 for x in ACTIONS})
            target = r if done else r + gamma * max(q2[x] for x in allowed_actions(s2))
            q[a] += alpha * (target - q[a])
            s, total = s2, total + r
        window.append(total)
        if (ep + 1) % 1000 == 0:
            curve.append(sum(window) / len(window))
            window = []
    return Q, curve


def greedy_policy(Q):
    return lambda s: max(allowed_actions(s), key=lambda a: Q.get(s, {}).get(a, 0.0))


def rule_policy(s: State) -> str:
    risk, _stage, _crit, enriched = s
    if risk == 2:
        return "escalate"
    if risk == 1 and not enriched:
        return "enrich"
    return "monitor"


def evaluate(policy, episodes: int = 5000, seed: int = 99) -> dict[str, float]:
    rng = random.Random(seed)
    env = TriageSim(rng)
    tot = fp_contain = missed = unsafe = 0.0
    for _ in range(episodes):
        s, done = env.reset(), False
        acted = False
        while not done:
            a = policy(s)
            if a not in allowed_actions(s):
                unsafe += 1
                a = "monitor"
            if a in ("escalate", "request_containment"):
                acted = True
                if a == "request_containment" and not env.compromised:
                    fp_contain += 1
            s, r, done = env.step(a)
            tot += r
        if env.compromised and not acted:
            missed += 1
    return {"avg_reward": tot / episodes, "fp_containment_rate": fp_contain / episodes,
            "missed_compromise_rate": missed / episodes, "unsafe_action_rate": unsafe / episodes}


def save_q(Q, path: Path) -> None:
    Path(path).write_text(json.dumps({",".join(map(str, k)): v for k, v in Q.items()}, indent=1), encoding="utf-8")


def load_q(path: Path) -> dict[State, dict[str, float]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {tuple(int(x) for x in k.split(",")): v for k, v in raw.items()}  # type: ignore[misc]


def state_from_case(p_compromise: float, stage_sev: float, critical: bool, enriched: bool) -> State:
    risk = 2 if p_compromise >= 0.7 else (1 if p_compromise >= 0.35 else 0)
    stage = 2 if stage_sev >= 0.6 else (1 if stage_sev >= 0.25 else 0)
    return (risk, stage, int(critical), int(enriched))

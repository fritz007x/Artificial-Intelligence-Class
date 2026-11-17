"""Configuration loading. Non-secret parameters live in configs/config.yaml;
secrets are read only from environment variables (never from files in source control)."""
from __future__ import annotations

import hashlib
import os
import platform
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppCfg(_M):
    name: str
    version: str
    config_version: str
    seed: int
    environment: str = "dev"


class PathsCfg(_M):
    logic_dir: str
    uncertainty_dir: str
    ml_dir: str
    playbooks_dir: str
    scenarios_dir: str
    synthetic_dir: str
    runtime_dir: str
    results_dir: str


class ThresholdsCfg(_M):
    critical_asset_min: int
    ioc_min_confidence: float
    failed_login_burst: int
    failed_login_window_min: int
    impossible_travel_kmh: float
    min_percept_quality: float
    bn_support: float
    abstain_low: float
    abstain_high: float
    kalman_anomaly_sigma: float
    fairness_max_fpr_gap: float


class PolicyCfg(_M):
    max_isolations_per_case: int
    disruptive_actions: list[str]
    disruptive_requires_entailment: bool
    protected_assets: list[str]
    never_isolate_asset_types: list[str]
    never_disable_account_types: list[str]
    action_costs: dict[str, float]
    action_benefit: dict[str, float]


class OrchestratorCfg(_M):
    max_iterations: int = Field(ge=1, le=20)
    tool_timeout_s: float
    tool_rate_limit_per_min: int
    api_rate_limit_per_min: int


class TemporalCfg(_M):
    bin_minutes: int


class FeaturesCfg(_M):
    llm_narrative: bool
    rl_advisor: bool
    ml_scoring: bool
    adversarial_planner: bool


class LLMCfg(_M):
    provider: str
    model: str
    max_tokens: int
    timeout_s: float
    effort: str


class AnalystCfg(_M):
    id: str
    skills: list[str]
    senior: bool


class CSPCfg(_M):
    analysts: list[AnalystCfg]
    slots: list[int]


class Settings(_M):
    app: AppCfg
    paths: PathsCfg
    thresholds: ThresholdsCfg
    risk_weights: dict[str, float]
    policy: PolicyCfg
    orchestrator: OrchestratorCfg
    temporal: TemporalCfg
    features: FeaturesCfg
    llm: LLMCfg
    csp: CSPCfg

    def path(self, key: str) -> Path:
        return PROJECT_ROOT / getattr(self.paths, key)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]

    def run_metadata(self) -> dict[str, Any]:
        return {
            "app_version": self.app.version,
            "config_version": self.app.config_version,
            "config_sha": self.fingerprint,
            "seed": self.app.seed,
            "python": platform.python_version(),
        }


def load_settings(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    cfg_path = Path(path or os.environ.get("ORCHIDSHIELD_CONFIG", PROJECT_ROOT / "configs" / "config.yaml"))
    data = yaml.safe_load(cfg_path.read_text())
    if env := os.environ.get("ORCHIDSHIELD_ENV"):
        data["app"]["environment"] = env
    for dotted, value in (overrides or {}).items():
        cur = data
        *parents, leaf = dotted.split(".")
        for p in parents:
            cur = cur[p]
        cur[leaf] = value
    return Settings.model_validate(data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


class Secrets:
    """Secrets are read lazily from the environment and never logged or serialized."""

    @staticmethod
    def get(name: str) -> str | None:
        val = os.environ.get(name)
        return val if val else None

    @staticmethod
    def redact(value: str | None) -> str:
        return "<unset>" if not value else "<redacted>"

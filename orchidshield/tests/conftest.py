"""Shared fixtures. Every test gets an isolated runtime dir (audit log, tickets, kill
switch) so tests are independent and never touch data/runtime."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TEST_KEYS = {"ORCHIDSHIELD_OPERATOR_KEY": "test-operator-key", "ORCHIDSHIELD_APPROVER_KEY": "test-approver-key",
             "ORCHIDSHIELD_ADMIN_KEY": "test-admin-key"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k, v in TEST_KEYS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("ORCHIDSHIELD_DEV_OPEN", raising=False)
    monkeypatch.delenv("ORCHIDSHIELD_KILL_SWITCH", raising=False)


@pytest.fixture
def settings(tmp_path):
    from core.config import load_settings
    return load_settings(overrides={"paths.runtime_dir": str(tmp_path / "runtime")})


@pytest.fixture
def agent(settings):
    from agent.orchestrator import Agent
    return Agent(settings)


@pytest.fixture
def run(agent):
    from sim.scenarios import load_scenario

    def _run(name):
        sc = load_scenario(name)
        return agent.run_case(sc["observations"], scenario=name, action_requests=sc.get("action_requests"))
    return _run


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    cfg["paths"]["runtime_dir"] = str(tmp_path / "api-runtime")
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("ORCHIDSHIELD_CONFIG", str(p))
    from core import config
    config.get_settings.cache_clear()
    for m in [m for m in sys.modules if m.startswith("api.")]:
        del sys.modules[m]
    from fastapi.testclient import TestClient
    import api.app as app_mod
    yield TestClient(app_mod.app), app_mod
    config.get_settings.cache_clear()


def hdr(role: str) -> dict:
    return {"X-API-Key": TEST_KEYS[f"ORCHIDSHIELD_{role.upper()}_KEY"]}

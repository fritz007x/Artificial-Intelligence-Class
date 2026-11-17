"""FastAPI service exposing the agent and a lightweight web UI.

Roles (least privilege), each with its own key read from the environment:
  operator  ORCHIDSHIELD_OPERATOR_KEY  run cases, read cases/audit/tools
  approver  ORCHIDSHIELD_APPROVER_KEY  approve / reject disruptive proposals (+ read)
  admin     ORCHIDSHIELD_ADMIN_KEY     kill switch, admin status (+ read)
Keys are compared in constant time and never echoed. ORCHIDSHIELD_DEV_OPEN=1 disables
auth for a local demo only and is refused when ORCHIDSHIELD_ENV=prod."""
from __future__ import annotations

import hmac
import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from agent.orchestrator import Agent
from core.config import get_settings
from core.schemas import CaseResult, ProposalStatus
from reasoning.evaluation import load_learning_report
from sim.scenarios import build_scenarios, load_scenario

STATIC = Path(__file__).parent / "static"
ROLE_ENV = {"operator": "ORCHIDSHIELD_OPERATOR_KEY", "approver": "ORCHIDSHIELD_APPROVER_KEY",
            "admin": "ORCHIDSHIELD_ADMIN_KEY"}
READ_ROLES = {"operator", "approver", "admin"}

settings = get_settings()
app = FastAPI(title="OrchidShield Triage Agent", version=settings.app.version,
              description="Security Triage & Response Recommendation Agent (synthetic data, course capstone)")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_agent: Agent | None = None
_agent_lock = threading.Lock()
METRICS: Counter = Counter()


def agent() -> Agent:
    global _agent
    with _agent_lock:
        if _agent is None:
            _agent = Agent(settings)
        return _agent


def _dev_open() -> bool:
    return os.environ.get("ORCHIDSHIELD_DEV_OPEN") == "1" and os.environ.get("ORCHIDSHIELD_ENV", "dev") != "prod"


def resolve_role(x_api_key: str | None) -> str | None:
    if _dev_open():
        return "admin"
    if not x_api_key:
        return None
    for role, env in ROLE_ENV.items():
        secret = os.environ.get(env)
        if secret and hmac.compare_digest(secret.encode(), x_api_key.encode()):
            return role
    return None


class _Bucket:
    def __init__(self, per_min: int):
        self.per_min, self.state, self.lock = per_min, {}, threading.Lock()

    def allow(self, key: str) -> bool:
        with self.lock:
            tokens, t = self.state.get(key, (float(self.per_min), time.monotonic()))
            now = time.monotonic()
            tokens = min(self.per_min, tokens + (now - t) * self.per_min / 60)
            ok = tokens >= 1
            self.state[key] = (tokens - 1 if ok else tokens, now)
            return ok


_rate = _Bucket(settings.orchestrator.api_rate_limit_per_min)


def require(*roles: str):
    def dep(request: Request, x_api_key: str | None = Header(default=None)) -> str:
        role = resolve_role(x_api_key)
        if role is None:
            METRICS["auth_failures"] += 1
            raise HTTPException(401, "missing or invalid API key")
        if role not in roles:
            METRICS["authz_denied"] += 1
            raise HTTPException(403, f"role '{role}' may not perform this operation")
        if not _rate.allow(role + (request.client.host if request.client else "")):
            METRICS["rate_limited"] += 1
            raise HTTPException(429, "rate limit exceeded")
        return role
    return dep


class CaseIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[dict[str, Any]] = Field(max_length=500)
    action_requests: list[dict[str, Any]] = Field(default_factory=list, max_length=20)


class DecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approve: bool
    note: str = Field(default="", max_length=500)


class KillIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    engaged: bool
    reason: str = Field(default="admin", max_length=200)


def _count(result: CaseResult) -> None:
    METRICS["cases_total"] += 1
    METRICS["iterations_total"] += result.iterations
    for p in result.proposals:
        METRICS[f"proposals_{p.status.value}"] += 1
    for t in result.tool_results:
        METRICS["tool_calls_ok" if t.ok else "tool_calls_failed"] += 1
    METRICS["contradictions_total"] += len(result.contradictions)
    METRICS["percepts_rejected"] += sum(1 for p in result.percepts if p.validation.value == "rejected")


# ------------------------------------------------------------------ public
@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    a = agent()
    return {"status": "ok", "version": settings.app.version, "config_version": settings.app.config_version,
            "kill_switch": a.kill.engaged, "model_loaded": a.model is not None, "narrator": a.narrator.name}


@app.get("/ready")
def ready() -> JSONResponse:
    a = agent()
    ok, n, _ = a.audit.verify()
    status = 200 if ok else 503
    return JSONResponse({"ready": ok, "audit_events": n}, status_code=status)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    a = agent()
    lines = [f"orchidshield_{k} {v}" for k, v in sorted(METRICS.items())]
    lines += [f"orchidshield_kill_switch {int(a.kill.engaged)}", f"orchidshield_model_loaded {int(a.model is not None)}",
              f"orchidshield_pending_approvals {len(a.pending_approvals())}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ operator
@app.get("/api/whoami")
def whoami(role: str = Depends(require(*READ_ROLES))) -> dict:
    return {"role": role}


@app.get("/api/scenarios")
def scenarios(_: str = Depends(require(*READ_ROLES))) -> list[dict]:
    return [{"name": k, "title": v["title"], "expect": v["expect"]} for k, v in build_scenarios().items()]


@app.post("/api/scenarios/{name}/run")
def run_scenario(name: str, role: str = Depends(require("operator", "admin"))) -> Any:
    if name not in build_scenarios():
        raise HTTPException(404, "unknown scenario")
    if name == "learning":
        return {"scenario": "learning", "report": load_learning_report(settings.path("ml_dir"))}
    sc = load_scenario(name)
    res = agent().run_case(sc["observations"], scenario=name, action_requests=sc.get("action_requests"), actor=role)
    _count(res)
    return res.model_dump(mode="json")


@app.post("/api/cases")
def run_case(body: CaseIn, role: str = Depends(require("operator", "admin"))) -> Any:
    res = agent().run_case(body.observations, action_requests=body.action_requests, actor=role)
    _count(res)
    return res.model_dump(mode="json")


@app.get("/api/cases")
def list_cases(_: str = Depends(require(*READ_ROLES))) -> list[dict]:
    return [{"case_id": c.case_id, "scenario": c.scenario, "halted": c.halted, "summary": c.summary,
             "pending": sum(p.status == ProposalStatus.PENDING_APPROVAL for p in c.proposals)}
            for c in agent().cases.values()]


@app.get("/api/cases/{case_id}")
def get_case(case_id: str, _: str = Depends(require(*READ_ROLES))) -> Any:
    c = agent().cases.get(case_id)
    if not c:
        raise HTTPException(404, "unknown case")
    return c.model_dump(mode="json")


@app.get("/api/tools")
def tools(_: str = Depends(require(*READ_ROLES))) -> list[dict]:
    return agent().tools.describe()


@app.get("/api/audit")
def audit(case_id: str | None = None, limit: int = 100, _: str = Depends(require(*READ_ROLES))) -> list[dict]:
    return agent().audit.tail(min(max(limit, 1), 1000), case_id)


@app.get("/api/audit/verify")
def audit_verify(_: str = Depends(require(*READ_ROLES))) -> dict:
    ok, n, err = agent().audit.verify()
    return {"ok": ok, "events": n, "error": err}


@app.get("/api/learning")
def learning(_: str = Depends(require(*READ_ROLES))) -> dict:
    return load_learning_report(settings.path("ml_dir"))


# ------------------------------------------------------------------ approver
@app.get("/api/approvals")
def approvals(_: str = Depends(require(*READ_ROLES))) -> list[dict]:
    return [p.model_dump(mode="json") for p in agent().pending_approvals()]


@app.post("/api/approvals/{proposal_id}")
def decide(proposal_id: str, body: DecisionIn, role: str = Depends(require("approver"))) -> dict:
    try:
        p, res = agent().decide(proposal_id, body.approve, approver=role, note=body.note)
    except KeyError:
        raise HTTPException(404, "unknown proposal") from None
    except ValueError as e:
        raise HTTPException(409, str(e)) from None
    METRICS["approvals_granted" if body.approve else "approvals_rejected"] += 1
    return {"proposal": p.model_dump(mode="json"), "tool_result": res.model_dump(mode="json") if res else None}


# ------------------------------------------------------------------ admin
@app.post("/api/admin/kill-switch")
def kill_switch(body: KillIn, role: str = Depends(require("admin"))) -> dict:
    a = agent()
    if body.engaged:
        a.kill.engage(body.reason)
    else:
        a.kill.release()
    a.audit.record(role, "kill_switch", None, {"engaged": body.engaged, "reason": body.reason})
    return {"kill_switch": a.kill.engaged}


@app.get("/api/admin/status")
def admin_status(_: str = Depends(require("admin"))) -> dict:
    a = agent()
    return {"kill_switch": a.kill.engaged, "kill_reason": a.kill.reason, "cases": len(a.cases),
            "model": a.model.metadata if a.model else {"error": a.model_error},
            "run_metadata": settings.run_metadata(), "features": settings.features.model_dump(),
            "secrets": {k: "<set>" if os.environ.get(v) else "<unset>" for k, v in ROLE_ENV.items()}}

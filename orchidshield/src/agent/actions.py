"""Stage 6 (execution half) - allow-listed tools. The agent can only act through this
registry. Each tool declares a typed parameter schema, whether it is disruptive, and a
handler that touches only simulated resources inside data/runtime/. Calls are
rate-limited, time-limited, idempotent (keyed by the proposal's idempotency key) and
audited. Disruptive tools refuse to run unless the proposal is APPROVED."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.audit import AuditLog, KillSwitch
from core.config import Settings
from core.schemas import ActionProposal, ProposalStatus, ToolResult
from reasoning.retrieval import TfidfIndex

_ID = r"^[A-Za-z0-9_.:\-]{1,64}$"


class ToolError(Exception):
    pass


class _P(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssetParams(_P):
    asset_id: str = Field(pattern=_ID)


class UserParams(_P):
    user_id: str = Field(pattern=_ID)


class IPParams(_P):
    ip: str = Field(pattern=r"^[0-9a-fA-F:.]{3,45}$")


class TextParams(_P):
    text: str = Field(max_length=500)


class TicketParams(_P):
    case_id: str = Field(pattern=_ID)
    title: str = Field(max_length=200)
    body: str = Field(max_length=4000)
    priority: str = Field(pattern=r"^(P1|P2|P3|P4)$")


class EntityParams(_P):
    entity: str = Field(pattern=_ID)
    reason: str = Field(max_length=300)


class NotifyParams(_P):
    message: str = Field(max_length=1000)
    severity: str = Field(pattern=r"^(info|warning|critical)$")


@dataclass
class Tool:
    name: str
    params: type[_P]
    disruptive: bool
    handler: Callable[[Any], dict]
    description: str


class _RateLimiter:
    def __init__(self, per_min: int):
        self.cap, self.tokens, self.t = per_min, float(per_min), time.monotonic()
        self.lock = threading.Lock()

    def take(self) -> bool:
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.cap, self.tokens + (now - self.t) * self.cap / 60)
            self.t = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class ToolRegistry:
    def __init__(self, settings: Settings, environment: dict, audit: AuditLog, kill: KillSwitch):
        self.s, self.env, self.audit, self.kill = settings, environment, audit, kill
        self.runtime = settings.path("runtime_dir")
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.rate = _RateLimiter(settings.orchestrator.tool_rate_limit_per_min)
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.results_by_key: dict[str, ToolResult] = {}
        self.faults: set[str] = set()          # tool names forced to fail (tests / chaos drills)
        self._db = self._init_db()
        self._history = TfidfIndex({
            "HIST-17 loader on finance workstation": "malware execution loader hash powershell isolate workstation",
            "HIST-22 vpn impossible travel benign": "impossible travel vpn egress apac benign user travel",
            "HIST-31 dc brute force": "brute force domain controller privileged account external ip block",
            "HIST-40 patch scan": "vulnerability scan maintenance window patch tooling benign",
        })
        T = Tool
        self.tools: dict[str, Tool] = {t.name: t for t in [
            T("query_asset_db", AssetParams, False, self._query_asset_db, "Read-only parameterised CMDB lookup (simulated SQL)"),
            T("edr_process_tree", AssetParams, False, self._edr, "Fetch EDR process tree / malware verdict (simulated)"),
            T("similar_alerts", TextParams, False, lambda p: {"matches": self._history.query(p.text, 3)},
              "Vector lookup over historical cases"),
            T("create_ticket", TicketParams, False, self._ticket, "Open a case ticket (file-backed)"),
            T("add_watchlist", EntityParams, False, lambda p: self._append("watchlist.jsonl", p.model_dump()),
              "Add entity to monitoring watchlist"),
            T("notify_analyst", NotifyParams, False, lambda p: self._append("notifications.jsonl", p.model_dump()),
              "Page the on-call analyst"),
            T("isolate_host", AssetParams, True, lambda p: self._contain("isolate_host", p.asset_id), "Network-isolate a host"),
            T("disable_account", UserParams, True, lambda p: self._contain("disable_account", p.user_id), "Disable an account"),
            T("block_ip", IPParams, True, lambda p: self._contain("block_ip", p.ip), "Block an IP at the perimeter"),
            T("rotate_credentials", UserParams, True, lambda p: self._contain("rotate_credentials", p.user_id),
              "Force credential rotation"),
        ]}

    # ------------------------------------------------------------------ handlers
    def _init_db(self) -> sqlite3.Connection:
        db = sqlite3.connect(":memory:", check_same_thread=False)
        db.execute("CREATE TABLE assets (asset_id TEXT PRIMARY KEY, asset_type TEXT, criticality INT, zone TEXT, "
                   "business_unit TEXT, edr INT)")
        db.executemany("INSERT INTO assets VALUES (?,?,?,?,?,?)", [
            (a["asset_id"], a["asset_type"], a["criticality"], a["zone"], a["business_unit"], int(a["edr_installed"]))
            for a in self.env["assets"].values()])
        return db

    def _query_asset_db(self, p: AssetParams) -> dict:
        row = self._db.execute("SELECT asset_id, asset_type, criticality, zone, business_unit, edr FROM assets "
                               "WHERE asset_id = ?", (p.asset_id,)).fetchone()
        if not row:
            raise ToolError(f"asset {p.asset_id} not found")
        return dict(zip(["asset_id", "asset_type", "criticality", "zone", "business_unit", "edr"], row))

    def _edr(self, p: AssetParams) -> dict:
        a = self.env["assets"].get(p.asset_id)
        if not a:
            raise ToolError("unknown asset")
        if not a.get("edr_installed", True):
            raise ToolError(f"no EDR sensor on {p.asset_id}")
        verdict = self.env.get("edr_truth", {}).get(p.asset_id, "clean")
        return {"asset_id": p.asset_id, "malicious": verdict == "malicious", "verdict": verdict,
                "processes": ["winword.exe → powershell.exe -enc …"] if verdict == "malicious" else []}

    def _safe_path(self, name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,80}", name):
            raise ToolError("illegal file name")
        p = (self.runtime / name).resolve()
        if not p.is_relative_to(self.runtime.resolve()):
            raise ToolError("path escapes runtime dir")
        return p

    def _append(self, name: str, rec: dict) -> dict:
        with self._safe_path(name).open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return {"written": name, **rec}

    def _ticket(self, p: TicketParams) -> dict:
        tid = f"TKT-{p.case_id[-6:]}-{uuid.uuid4().hex[:4]}"
        (self.runtime / "tickets").mkdir(exist_ok=True)
        path = self._safe_path("tickets").joinpath(f"{tid}.json")
        path.write_text(json.dumps({"ticket_id": tid, **p.model_dump()}, indent=1))
        return {"ticket_id": tid}

    def _contain(self, action: str, target: str) -> dict:
        return self._append("containment.jsonl", {"action": action, "target": target, "simulated": True,
                                                  "ts": time.time()})

    # ------------------------------------------------------------------ execution
    def describe(self) -> list[dict]:
        return [{"name": t.name, "disruptive": t.disruptive, "description": t.description,
                 "params": t.params.model_json_schema()} for t in self.tools.values()]

    def is_allowed(self, name: str) -> bool:
        return name in self.tools

    def execute(self, proposal: ActionProposal, actor: str = "agent") -> ToolResult:
        start = time.perf_counter()
        call_id = f"C-{uuid.uuid4().hex[:8]}"

        def fail(msg: str) -> ToolResult:
            r = ToolResult(call_id=call_id, tool=proposal.tool or "?", proposal_id=proposal.proposal_id, ok=False,
                           error=msg, duration_ms=round((time.perf_counter() - start) * 1000, 2))
            self.audit.record(actor, "tool_refused" if "refused" in msg else "tool_failed", proposal.case_id,
                              {"tool": proposal.tool, "target": proposal.target, "error": msg})
            return r

        if proposal.idempotency_key in self.results_by_key:
            prev = self.results_by_key[proposal.idempotency_key]
            return prev.model_copy(update={"idempotent_replay": True})
        if self.kill.engaged:
            return fail("refused: kill switch engaged")
        if not proposal.tool or proposal.tool not in self.tools:
            return fail(f"refused: tool '{proposal.tool}' is not allow-listed")
        tool = self.tools[proposal.tool]
        if tool.disruptive and proposal.status != ProposalStatus.APPROVED:
            return fail("refused: disruptive tool requires an APPROVED proposal")
        if not tool.disruptive and proposal.status not in (ProposalStatus.AUTO_APPROVED, ProposalStatus.APPROVED):
            return fail(f"refused: proposal status {proposal.status.value} is not executable")
        if not self.rate.take():
            return fail("refused: tool rate limit exceeded")
        try:
            params = tool.params.model_validate(proposal.params)
        except ValidationError as e:
            return fail(f"refused: invalid parameters: {e.errors()[0]['msg']}")
        if proposal.tool in self.faults:
            return fail(f"{proposal.tool}: simulated upstream failure")
        fut = self.pool.submit(tool.handler, params)
        try:
            out = fut.result(timeout=self.s.orchestrator.tool_timeout_s)
        except FutTimeout:
            return fail(f"{proposal.tool}: timed out after {self.s.orchestrator.tool_timeout_s}s")
        except ToolError as e:
            return fail(f"{proposal.tool}: {e}")
        except Exception as e:  # noqa: BLE001 - tool faults must never crash the loop
            return fail(f"{proposal.tool}: unexpected error {type(e).__name__}")
        res = ToolResult(call_id=call_id, tool=tool.name, proposal_id=proposal.proposal_id, ok=True, output=out,
                         duration_ms=round((time.perf_counter() - start) * 1000, 2))
        self.results_by_key[proposal.idempotency_key] = res
        self.audit.record(actor, "tool_executed", proposal.case_id,
                          {"tool": tool.name, "target": proposal.target, "disruptive": tool.disruptive,
                           "call_id": call_id})
        return res

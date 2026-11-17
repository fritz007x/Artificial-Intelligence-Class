"""Append-only, hash-chained audit log (JSONL) and the administrative kill switch."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.schemas import AuditEvent

GENESIS = "0" * 64


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq, self._last = 0, GENESIS
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    ev = json.loads(line)
                    self._seq, self._last = ev["seq"], ev["hash"]

    @staticmethod
    def _digest(body: dict) -> str:
        return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()

    def record(self, actor: str, event_type: str, case_id: str | None, details: dict[str, Any]) -> AuditEvent:
        with self._lock:
            body = {"seq": self._seq + 1, "event_id": uuid.uuid4().hex[:12], "ts": datetime.now(timezone.utc).isoformat(),
                    "actor": actor, "event_type": event_type, "case_id": case_id,
                    "details": json.loads(json.dumps(details, default=str)), "prev_hash": self._last}
            body["hash"] = self._digest(body)
            with self.path.open("a") as fh:
                fh.write(json.dumps(body) + "\n")
            self._seq, self._last = body["seq"], body["hash"]
            return AuditEvent.model_validate(body)

    def tail(self, n: int = 100, case_id: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        rows = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
        if case_id:
            rows = [r for r in rows if r.get("case_id") == case_id]
        return rows[-n:]

    def verify(self) -> tuple[bool, int, str | None]:
        """Re-hash the whole chain. Returns (ok, events_checked, first_error)."""
        prev, n = GENESIS, 0
        if not self.path.exists():
            return True, 0, None
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            n += 1
            h = ev.pop("hash")
            if ev["prev_hash"] != prev or self._digest(ev) != h:
                return False, n, f"chain broken at seq {ev.get('seq')}"
            prev = h
        return True, n, None


class KillSwitch:
    """Engaged by an admin via API, by a sentinel file, or by env var. When engaged the
    orchestrator stops before the next stage and no tool executes."""

    def __init__(self, runtime_dir: Path):
        self.flag_file = Path(runtime_dir) / "KILL_SWITCH"
        self._mem = False
        self.reason: str | None = None

    def engage(self, reason: str = "admin") -> None:
        self._mem, self.reason = True, reason
        self.flag_file.parent.mkdir(parents=True, exist_ok=True)
        self.flag_file.write_text(reason)

    def release(self) -> None:
        self._mem, self.reason = False, None
        if self.flag_file.exists():
            self.flag_file.unlink()

    @property
    def engaged(self) -> bool:
        return self._mem or self.flag_file.exists() or os.environ.get("ORCHIDSHIELD_KILL_SWITCH") == "1"

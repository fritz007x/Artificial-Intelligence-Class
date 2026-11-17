"""Stage 1 - PERCEPTS. Validate raw observations, score their quality, scan free text
for prompt-injection / markup, and wrap them as immutable Percept records with source,
timestamp, quality and validation status. Rejected percepts are kept (for audit and UI)
but never reach the knowledge base."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import ValidationError

from core.config import Settings
from core.schemas import (AlertIn, AssetIn, AuthEventIn, FeedbackIn, IndicatorIn, Observation, Percept,
                          ToolObservationIn, UserIn, ValidationStatus)

KIND_MODEL = {"alert": AlertIn, "auth_event": AuthEventIn, "asset": AssetIn, "user": UserIn,
              "indicator": IndicatorIn, "feedback": FeedbackIn, "tool_result": ToolObservationIn}
ALLOWED_SOURCES = {
    "alert": {"siem", "edr", "ids"}, "auth_event": {"auth_log"}, "asset": {"asset_inventory"},
    "user": {"directory"}, "indicator": {"threat_intel"}, "feedback": {"analyst"},
    "tool_result": {"tool"},   # only the orchestrator may emit tool results
}
SOURCE_RELIABILITY = {"edr": 0.95, "siem": 0.85, "ids": 0.8, "auth_log": 0.95, "asset_inventory": 0.9,
                      "directory": 0.95, "threat_intel": 0.85, "analyst": 0.9, "tool": 0.95}

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"system (override|prompt)", r"you are now", r"disregard (the )?(policy|rules|instructions)",
    r"\b(call|run|invoke|execute) (the )?(tool|function|command)\b",
    r"\b(isolate_host|disable_account|block_ip|delete_audit_log|rotate_credentials)\b",
    r"\b(analyst|ciso|admin) approved\b", r"\bapproval granted\b",
]
MARKUP_PATTERN = re.compile(r"<\s*/?\s*(script|iframe|img|svg|object)[^>]*>", re.I)
_INJ = [re.compile(p, re.I) for p in INJECTION_PATTERNS]
_TEXT_FIELDS = ("description", "note")


def sanitize_text(text: str, limit: int = 500) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch.isprintable())
    text = MARKUP_PATTERN.sub("[markup removed]", text)
    return text[:limit]


def scan_text(text: str) -> list[str]:
    flags = []
    if any(p.search(text) for p in _INJ):
        flags.append("possible_prompt_injection")
    if MARKUP_PATTERN.search(text):
        flags.append("active_markup")
    return flags


class Perceiver:
    def __init__(self, settings: Settings):
        self.s = settings

    @staticmethod
    def _pid(raw: Any) -> str:
        return "P-" + hashlib.sha1(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()[:10]

    def perceive_one(self, raw: Any, received_at: datetime | None = None) -> Percept:
        received_at = received_at or datetime.now(timezone.utc)
        pid = self._pid(raw)
        try:
            obs = Observation.model_validate(raw)
        except ValidationError as e:
            return Percept(percept_id=pid, kind=str(raw.get("kind", "?")) if isinstance(raw, dict) else "?",
                           source=str(raw.get("source", "?")) if isinstance(raw, dict) else "?", timestamp=None,
                           received_at=received_at, quality=0.0, validation=ValidationStatus.REJECTED,
                           issues=[f"envelope: {err['loc']}: {err['msg']}" for err in e.errors()][:5])
        issues: list[str] = []
        if obs.source not in ALLOWED_SOURCES[obs.kind]:
            issues.append(f"source '{obs.source}' is not authorised to emit '{obs.kind}' (possible spoofing)")
            return Percept(percept_id=pid, kind=obs.kind, source=obs.source, timestamp=None, received_at=received_at,
                           quality=0.0, validation=ValidationStatus.REJECTED, issues=issues)
        try:
            model = KIND_MODEL[obs.kind].model_validate(obs.data)
        except ValidationError as e:
            return Percept(percept_id=pid, kind=obs.kind, source=obs.source, timestamp=None, received_at=received_at,
                           quality=0.0, validation=ValidationStatus.REJECTED,
                           issues=[f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()][:5])

        payload = model.model_dump(mode="json")
        quality = SOURCE_RELIABILITY.get(obs.source, 0.7)
        flags: list[str] = []
        for f in _TEXT_FIELDS:
            if f in payload and payload[f]:
                fl = scan_text(payload[f])
                if fl:
                    flags.extend(fl)
                    payload[f] = sanitize_text(payload[f])
                    payload[f"{f}_quarantined"] = True
        if "possible_prompt_injection" in flags:
            quality -= 0.35
            issues.append("free text contains instruction-like content; treated as untrusted data only")
        ts = getattr(model, "timestamp", None)
        if ts is not None:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
                issues.append("timestamp had no timezone; assumed UTC")
                quality -= 0.05
            if ts > received_at + timedelta(minutes=5) and obs.kind != "tool_result":
                # replayed scenarios are allowed to be in the past, never in the future
                issues.append("timestamp in the future")
                quality -= 0.3
        if obs.kind == "alert" and not payload.get("user_id") and payload["alert_type"] in (
                "brute_force", "password_spray", "impossible_travel"):
            issues.append("credential alert without user_id (incomplete)")
            quality -= 0.15
        quality = max(0.0, min(1.0, quality))
        status = ValidationStatus.VALID
        if quality < self.s.thresholds.min_percept_quality:
            status = ValidationStatus.REJECTED
            issues.append(f"quality {quality:.2f} below minimum {self.s.thresholds.min_percept_quality}")
        elif issues:
            status = ValidationStatus.DEGRADED
        return Percept(percept_id=pid, kind=obs.kind, source=obs.source, timestamp=ts, received_at=received_at,
                       quality=round(quality, 3), validation=status, issues=issues,
                       content_flags=sorted(set(flags)), payload=payload)

    def perceive(self, raws: list[Any], received_at: datetime | None = None) -> list[Percept]:
        seen, out = set(), []
        for raw in raws:
            p = self.perceive_one(raw, received_at)
            if p.percept_id in seen:          # idempotent ingestion of duplicate observations
                continue
            seen.add(p.percept_id)
            out.append(p)
        return out

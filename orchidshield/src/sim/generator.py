"""Synthetic, course-created OrchidShield environment. No real company, host, person or
indicator is represented. IPs come from documentation ranges (RFC 5737), domains use the
reserved `.test` TLD, and hashes are seeded pseudo-random hex."""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from core.config import PROJECT_ROOT

REGIONS = ["NA", "EU", "APAC", "LATAM"]
DEPTS = ["finance", "engineering", "sales", "hr", "it"]


def fake_hash(seed: str) -> str:
    return hashlib.sha256(f"orchidshield-synthetic-{seed}".encode()).hexdigest()


def make_environment(seed: int = 1337) -> dict:
    rng = random.Random(seed)
    A = {}

    def add(aid, typ, crit, zone, conn, edr=True, bu="it", maint=False):
        A[aid] = {"asset_id": aid, "hostname": f"{aid}.orchid.test", "asset_type": typ, "criticality": crit,
                  "zone": zone, "connected_to": conn, "in_maintenance": maint, "business_unit": bu,
                  "edr_installed": edr}

    add("fw-01", "firewall", 4, "dmz", ["web-01", "web-02"])
    add("web-01", "server", 3, "dmz", ["srv-app-01"], bu="sales")
    add("web-02", "server", 3, "dmz", ["srv-app-02"], bu="sales")
    add("srv-app-01", "server", 4, "prod", ["srv-db-01"], bu="finance")
    add("srv-app-02", "server", 3, "prod", ["srv-db-02"], bu="engineering", maint=True)
    add("srv-app-03", "server", 3, "prod", ["srv-db-01", "srv-db-02"], bu="engineering")
    add("srv-db-01", "database", 5, "prod", [], bu="finance")
    add("srv-db-02", "database", 5, "prod", [], bu="engineering")
    add("dc-01", "domain_controller", 5, "mgmt", ["file-01", "jump-01", "srv-app-01"])
    add("siem-01", "siem", 4, "mgmt", ["dc-01"])
    add("jump-01", "server", 4, "corp", ["srv-db-01", "srv-app-03"])
    add("file-01", "server", 3, "corp", ["srv-app-03"])
    for i in range(1, 14):
        wid = f"ws-{i:03d}"
        conn = ["file-01"] + (["jump-01"] if i % 4 == 0 else []) + (["dc-01"] if i in (1, 2) else [])
        add(wid, "workstation", rng.choice([1, 1, 2]), "corp", conn, edr=(i != 13), bu=DEPTS[i % len(DEPTS)])

    users = {}
    for i in range(1, 25):
        uid = f"u-{i:02d}"
        users[uid] = {"user_id": uid, "department": DEPTS[i % len(DEPTS)], "region": REGIONS[i % 4],
                      "account_type": "standard"}
    users["adm-01"] = {"user_id": "adm-01", "department": "it", "region": "NA", "account_type": "privileged"}
    users["svc-backup"] = {"user_id": "svc-backup", "department": "it", "region": "NA", "account_type": "service"}
    users["bg-admin"] = {"user_id": "bg-admin", "department": "it", "region": "EU", "account_type": "break_glass"}

    now = datetime(2026, 11, 16, 0, 0, tzinfo=timezone.utc)
    indicators = [
        {"value": fake_hash("ransom-1"), "ioc_type": "sha256", "confidence": 0.95, "source_feed": "orchid-intel",
         "expires_at": (now + timedelta(days=90)).isoformat()},
        {"value": fake_hash("loader-2"), "ioc_type": "sha256", "confidence": 0.85, "source_feed": "orchid-intel",
         "expires_at": (now + timedelta(days=60)).isoformat()},
        {"value": fake_hash("old-3"), "ioc_type": "sha256", "confidence": 0.9, "source_feed": "orchid-intel",
         "expires_at": (now - timedelta(days=10)).isoformat()},
        {"value": "203.0.113.50", "ioc_type": "ip", "confidence": 0.9, "source_feed": "orchid-intel",
         "expires_at": (now + timedelta(days=30)).isoformat()},
        {"value": "198.51.100.23", "ioc_type": "ip", "confidence": 0.4, "source_feed": "community-feed",
         "expires_at": (now + timedelta(days=30)).isoformat()},
        {"value": "c2.badcorp.test", "ioc_type": "domain", "confidence": 0.8, "source_feed": "orchid-intel",
         "expires_at": None},
    ]
    # Ground truth used ONLY by simulated enrichment tools (the agent never reads it directly)
    edr_truth = {"ws-005": "malicious", "ws-008": "malicious", "dc-01": "suspicious", "srv-db-01": "clean"}
    return {"assets": A, "users": users, "indicators": indicators, "edr_truth": edr_truth,
            "generated_with_seed": seed}


# ------------------------------------------------------------------ ML training data
def make_alert_dataset(n: int = 4000, seed: int = 1337) -> list[dict]:
    """Labelled alert contexts for supervised learning. Truth is latent; features are
    noisy conditionals of truth, with a non-linear interaction (privileged + off-hours
    credential success) and a regional nuisance (APAC VPN egress produces benign
    impossible-travel) to give the fairness audit something real to measure."""
    rng = np.random.default_rng(seed)
    rows = []
    classes = ["credential_attack", "execution", "reconnaissance", "lateral_movement", "data_exfiltration"]
    for i in range(n):
        region = REGIONS[rng.integers(0, 4)]
        dept = DEPTS[rng.integers(0, len(DEPTS))]
        tp = rng.random() < 0.3
        cls = rng.choice(classes, p=[0.35, 0.25, 0.25, 0.1, 0.05] if not tp else [0.3, 0.3, 0.1, 0.2, 0.1])
        crit = int(rng.choice([1, 2, 3, 4, 5], p=[0.35, 0.25, 0.2, 0.12, 0.08]))
        privileged = rng.random() < (0.25 if tp else 0.12)
        off_hours = rng.random() < (0.55 if tp else 0.3)
        ioc = rng.random() < (0.5 if tp else 0.06)
        ioc_conf = float(np.clip(rng.normal(0.85, 0.08), 0, 1)) if ioc else float(rng.random() < 0.15) * float(rng.uniform(0.2, 0.55))
        edr = cls == "execution" or rng.random() < 0.2
        failed = int(rng.poisson(7 if (tp and cls == "credential_attack") else 1.5))
        saf = cls == "credential_attack" and failed >= 5 and rng.random() < (0.75 if tp else 0.15)
        travel_p = (0.45 if tp else 0.05) + (0.12 if region == "APAC" else 0.0)
        travel = cls == "credential_attack" and rng.random() < travel_p
        sev_base = 2.2 + (0.9 if tp else 0) + rng.normal(0, 0.8)
        severity = int(np.clip(round(sev_base), 1, 4))
        asset_cnt = int(rng.poisson(4 if tp else 1.5)) + 1
        # non-linear interaction: privileged off-hours credential success is very likely malicious
        if not tp and privileged and off_hours and saf and rng.random() < 0.7:
            tp = True
        rows.append({
            "alert_id": f"train-{i:05d}", "label": int(tp), "region": region, "department": dept,
            "severity": severity, "criticality": crit, "ioc_match": ioc, "ioc_confidence": ioc_conf,
            "failed_logins": failed, "success_after_fail": saf, "impossible_travel": travel,
            "edr_source": edr, "off_hours": off_hours, "privileged_user": privileged,
            "asset_alert_count": asset_cnt, "alert_class": str(cls),
        })
    return rows


def write_environment(path: Path | None = None, seed: int = 1337) -> Path:
    path = path or PROJECT_ROOT / "data" / "synthetic" / "environment.json"
    path.write_text(json.dumps(make_environment(seed), indent=1))
    return path


def load_environment() -> dict:
    p = PROJECT_ROOT / "data" / "synthetic" / "environment.json"
    if not p.exists():
        write_environment(p)
    return json.loads(p.read_text())

"""Final demonstration script. Runs the full story against the API, either in-process
(default, no server needed) or against a running container:

    python scripts/demo.py                                   # in-process, dev keys generated for the run
    python scripts/demo.py --base-url http://localhost:8000  # live service; reads keys from env

See docs/demo_script.md for the narrated version."""
from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def banner(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def show_case(c: dict) -> None:
    print(f"case {c['case_id']}  iterations={c['iterations']}  halted={c['halted']}")
    for p in c["percepts"]:
        flag = f"  flags={p['content_flags']}" if p["content_flags"] else ""
        print(f"  1 PERCEPT     {p['kind']:<11} {p['validation']:<9} q={p['quality']:.2f}{flag}"
              + (f"  issue: {p['issues'][0]}" if p["issues"] else ""))
    print(f"  2 PROPOSITION {len(c['propositions'])} literals; contradictions: "
          f"{[x['statement'] for x in c['contradictions']] or 'none'}")
    print(f"  3 KNOWLEDGE   used: {', '.join(c['knowledge_used'][:6])}{' …' if len(c['knowledge_used']) > 6 else ''}")
    for r in c["inference"]["risks"][:3]:
        print(f"  4 INFERENCE   #{r['rank']} {r['asset_id']}: risk={r['risk_score']:.2f} P(comp)={r['p_compromise_bayes']:.2f} "
              f"ML={r['p_true_positive_ml'] if r['p_true_positive_ml'] is None else round(r['p_true_positive_ml'], 2)} "
              f"stage={r['hmm_stage']}")
    for e in c["entailments"]:
        if e["status"] != "unknown":
            print(f"  5 ENTAILMENT  {e['status'].upper():<12} {e['query']}")
    for p in c["proposals"]:
        extra = f"  ✖ {p['blocked_reasons'][0]}" if p["blocked_reasons"] else ""
        alt = f"  ↳ {p['alternatives_considered'][0]}" if p["alternatives_considered"] else ""
        print(f"  6 ACTION      {p['status']:<16} {p['action']}({p['target']}){extra}{alt}")
    print(f"  SUMMARY: {c['summary']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url")
    args = ap.parse_args()
    if args.base_url:
        import httpx
        client = httpx.Client(base_url=args.base_url, timeout=30)
        keys = {r: os.environ[f"ORCHIDSHIELD_{r.upper()}_KEY"] for r in ("operator", "approver", "admin")}
    else:
        import tempfile
        keys = {r: secrets.token_urlsafe(16) for r in ("operator", "approver", "admin")}
        for r, k in keys.items():
            os.environ[f"ORCHIDSHIELD_{r.upper()}_KEY"] = k
        import yaml
        cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
        cfg["paths"]["runtime_dir"] = tempfile.mkdtemp(prefix="orchid-demo-")
        p = Path(cfg["paths"]["runtime_dir"]) / "config.yaml"
        p.write_text(yaml.safe_dump(cfg))
        os.environ["ORCHIDSHIELD_CONFIG"] = str(p)
        from fastapi.testclient import TestClient
        from api.app import app
        client = TestClient(app)
    H = {r: {"X-API-Key": k} for r, k in keys.items()}

    banner("0. Health & tools")
    print(client.get("/health").json())
    print("allow-listed tools:", [t["name"] + ("*" if t["disruptive"] else "") for t in
                                  client.get("/api/tools", headers=H["operator"]).json()], "(* = approval-gated)")

    for name in ("normal", "uncertain", "contradiction", "constraint", "adversarial"):
        banner(f"Scenario: {name}")
        c = client.post(f"/api/scenarios/{name}/run", headers=H["operator"]).json()
        show_case(c)

    banner("Scenario: learning (offline evaluation)")
    rep = client.post("/api/scenarios/learning/run", headers=H["operator"]).json()["report"]
    print(rep["learning_case"])

    banner("Human approval: approver approves the first pending containment")
    pend = client.get("/api/approvals", headers=H["approver"]).json()
    print("pending:", [f"{p['action']}({p['target']})" for p in pend])
    print("operator tries to approve ->", client.post(f"/api/approvals/{pend[0]['proposal_id']}", json={"approve": True},
                                                      headers=H["operator"]).status_code)
    r = client.post(f"/api/approvals/{pend[0]['proposal_id']}", json={"approve": True, "note": "demo"},
                    headers=H["approver"]).json()
    print("approver ->", r["proposal"]["status"], r["tool_result"]["output"])

    banner("Kill switch: interrupt the loop safely")
    print(client.post("/api/admin/kill-switch", json={"engaged": True, "reason": "demo"}, headers=H["admin"]).json())
    c = client.post("/api/scenarios/normal/run", headers=H["operator"]).json()
    print("run while engaged -> halted:", c["halted"], "|", c["halt_reason"], "| tool calls:", len(c["tool_results"]))
    if pend[1:]:
        r = client.post(f"/api/approvals/{pend[1]['proposal_id']}", json={"approve": True}, headers=H["approver"]).json()
        print("approval while engaged ->", r["proposal"]["status"], "|", r["tool_result"]["error"])
    print(client.post("/api/admin/kill-switch", json={"engaged": False}, headers=H["admin"]).json())

    banner("Audit trail integrity")
    print(client.get("/api/audit/verify", headers=H["operator"]).json())
    print(client.get("/metrics").text)


if __name__ == "__main__":
    main()

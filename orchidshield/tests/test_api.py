"""API integration tests: auth, least privilege, scenario runs, approvals, kill switch,
UI, monitoring endpoints, and that no secrets leak."""
from conftest import TEST_KEYS, hdr


def test_health_and_ui_public(api_client):
    c, _ = api_client
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/ready").status_code == 200
    html = c.get("/").text
    assert "Percepts" in html and "Entailment" in html


def test_auth_required(api_client):
    c, _ = api_client
    assert c.get("/api/scenarios").status_code == 401
    assert c.get("/api/scenarios", headers={"X-API-Key": "wrong"}).status_code == 401


def test_least_privilege(api_client):
    c, _ = api_client
    assert c.post("/api/admin/kill-switch", json={"engaged": True}, headers=hdr("operator")).status_code == 403
    assert c.post("/api/approvals/AP-x", json={"approve": True}, headers=hdr("operator")).status_code == 403
    assert c.post("/api/approvals/AP-x", json={"approve": True}, headers=hdr("admin")).status_code == 403
    assert c.post("/api/scenarios/normal/run", headers=hdr("approver")).status_code == 403


def test_scenario_run_exposes_six_stages(api_client):
    c, _ = api_client
    r = c.post("/api/scenarios/normal/run", headers=hdr("operator"))
    assert r.status_code == 200
    body = r.json()
    for k in ("percepts", "propositions", "knowledge_used", "inference", "entailments", "proposals"):
        assert body[k]
    assert c.get(f"/api/cases/{body['case_id']}", headers=hdr("approver")).status_code == 200


def test_approval_flow(api_client):
    c, _ = api_client
    c.post("/api/scenarios/normal/run", headers=hdr("operator"))
    pend = c.get("/api/approvals", headers=hdr("approver")).json()
    assert pend and all(p["status"] == "pending_approval" for p in pend)
    r = c.post(f"/api/approvals/{pend[0]['proposal_id']}", json={"approve": True}, headers=hdr("approver"))
    assert r.status_code == 200 and r.json()["proposal"]["status"] == "executed"
    assert c.post(f"/api/approvals/{pend[0]['proposal_id']}", json={"approve": True},
                  headers=hdr("approver")).status_code == 409


def test_kill_switch_via_api(api_client):
    c, _ = api_client
    assert c.post("/api/admin/kill-switch", json={"engaged": True}, headers=hdr("admin")).json()["kill_switch"]
    body = c.post("/api/scenarios/normal/run", headers=hdr("operator")).json()
    assert body["halted"]
    c.post("/api/admin/kill-switch", json={"engaged": False}, headers=hdr("admin"))
    assert not c.post("/api/scenarios/normal/run", headers=hdr("operator")).json()["halted"]


def test_custom_case_and_validation(api_client):
    c, _ = api_client
    assert c.post("/api/cases", json={"observations": [], "evil": 1}, headers=hdr("operator")).status_code == 422
    r = c.post("/api/cases", json={"observations": [{"kind": "alert", "source": "siem", "data": {"x": 1}}]},
               headers=hdr("operator"))
    assert r.status_code == 200 and r.json()["percepts"][0]["validation"] == "rejected"


def test_learning_endpoint(api_client):
    c, _ = api_client
    rep = c.post("/api/scenarios/learning/run", headers=hdr("operator")).json()["report"]
    assert "learning_case" in rep


def test_audit_and_metrics(api_client):
    c, _ = api_client
    c.post("/api/scenarios/adversarial/run", headers=hdr("operator"))
    assert c.get("/api/audit/verify", headers=hdr("operator")).json()["ok"]
    m = c.get("/metrics").text
    assert "orchidshield_cases_total 1" in m and "orchidshield_proposals_blocked" in m


def test_no_secret_leakage(api_client):
    c, _ = api_client
    status = c.get("/api/admin/status", headers=hdr("admin")).text
    blobs = status + c.get("/health").text + c.get("/metrics").text
    assert not any(v in blobs for v in TEST_KEYS.values())
    assert '"<set>"' in status


def test_dev_open_refused_in_prod(api_client, monkeypatch):
    c, _ = api_client
    monkeypatch.setenv("ORCHIDSHIELD_DEV_OPEN", "1")
    monkeypatch.setenv("ORCHIDSHIELD_ENV", "prod")
    assert c.get("/api/scenarios").status_code == 401


def test_api_rate_limit(api_client):
    c, mod = api_client
    mod._rate = mod._Bucket(3)
    codes = [c.get("/api/whoami", headers=hdr("operator")).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and 429 in codes

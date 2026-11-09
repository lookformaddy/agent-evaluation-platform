"""API tests (run in the venv where FastAPI is installed)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import ageval.service as service


@pytest.fixture
def client(tmp_path):
    service.registry = service.EvalRegistry(tmp_path)
    service.runner = service.EvalRunner(service.registry)
    return TestClient(service.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "support-triage" in body["suites"]


def test_list_suites(client):
    resp = client.get("/suites")
    assert resp.status_code == 200
    assert resp.json()["support-triage"]["version"] == "1.0.0"
    assert len(resp.json()["travel-booking"]["tasks"]) == 3


def test_run_eval_first_run_fails_gates(client):
    resp = client.post("/runs", json={"suite_id": "support-triage",
                                      "agent_id": "scripted-good"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["aggregate"]["task_success"] == 1.0
    assert body["gates"]["verdict"] == "FAIL"  # no baseline yet
    assert len(body["scores"]) == 4


def test_run_eval_unknown_suite_404(client):
    resp = client.post("/runs", json={"suite_id": "nope",
                                      "agent_id": "scripted-good"})
    assert resp.status_code == 404


def test_run_eval_unknown_agent_404(client):
    resp = client.post("/runs", json={"suite_id": "support-triage",
                                      "agent_id": "nope"})
    assert resp.status_code == 404


def test_list_and_get_run(client):
    created = client.post("/runs", json={"suite_id": "travel-booking",
                                         "agent_id": "scripted-good"})
    run_id = created.json()["run_id"]
    listed = client.get("/runs")
    assert listed.status_code == 200
    assert any(r["run_id"] == run_id for r in listed.json()["runs"])
    fetched = client.get(f"/runs/{run_id}")
    assert fetched.status_code == 200
    assert fetched.json()["suite_id"] == "travel-booking"


def test_get_missing_run_404(client):
    assert client.get("/runs/no-such-run").status_code == 404


def test_report_markdown_and_json(client):
    created = client.post("/runs", json={"suite_id": "support-triage",
                                         "agent_id": "scripted-good"})
    run_id = created.json()["run_id"]
    md = client.get(f"/runs/{run_id}/report")
    assert md.status_code == 200
    assert "Gate Verdict: FAIL" in md.text
    js = client.get(f"/runs/{run_id}/report", params={"format": "json"})
    assert js.status_code == 200
    assert js.json()["run_id"] == run_id
    bad = client.get(f"/runs/{run_id}/report", params={"format": "yaml"})
    assert bad.status_code == 400


def test_gates_endpoint(client):
    created = client.post("/runs", json={"suite_id": "support-triage",
                                         "agent_id": "scripted-good"})
    run_id = created.json()["run_id"]
    gates = client.get(f"/runs/{run_id}/gates")
    assert gates.status_code == 200
    assert gates.json()["verdict"] == "FAIL"
    assert client.get("/runs/no-such-run/gates").status_code == 404

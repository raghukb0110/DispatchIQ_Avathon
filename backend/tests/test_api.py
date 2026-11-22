import pytest
from fastapi.testclient import TestClient

import app.main as main


@pytest.fixture
def client(monkeypatch, small_scorer):
    monkeypatch.setattr(main, "get_scorer", lambda: small_scorer)
    return TestClient(main.app)


def test_health_and_presets(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert "scarce" in client.get("/api/presets").json()


def test_generate_and_compare(client):
    sc = client.post("/api/scenario/generate", json={"preset": "balanced", "seed": 1}).json()
    assert len(sc["jobs"]) == 60

    body = client.post("/api/compare", json={"scenario": sc}).json()
    assert set(body["results"]) == {"hungarian", "ml"}
    assert 0 <= body["summary"]["agreement"] <= 1
    for r in body["results"].values():
        assert len(r["assignments"]) + len(r["unassigned"]) == 60


def test_allocate_single_algorithm(client):
    sc = client.post("/api/scenario/generate", json={"n_techs": 5, "n_jobs": 20}).json()
    r = client.post("/api/allocate", json={"scenario": sc, "algorithm": "hungarian"})
    assert r.status_code == 200
    assert r.json()["algorithm"] == "hungarian"


def test_invalid_requests(client):
    assert client.post("/api/scenario/generate", json={"preset": "nope"}).status_code == 404
    sc = client.post("/api/scenario/generate", json={}).json()
    assert client.post("/api/allocate", json={"scenario": sc, "algorithm": "magic"}).status_code == 422
    assert client.post("/api/allocate", json={"scenario": sc, "algorithm": "cost_greedy"}).status_code == 422

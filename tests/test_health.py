from fastapi.testclient import TestClient


def test_health_when_loaded(client_loaded: TestClient):
    r = client_loaded.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "Test Model"


def test_health_when_unloaded(client_unloaded: TestClient):
    r = client_unloaded.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_model_info_when_loaded(client_loaded: TestClient):
    r = client_loaded.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Test Model"
    assert "features" in body and len(body["features"]) > 0


def test_model_info_when_unloaded(client_unloaded: TestClient):
    r = client_unloaded.get("/model-info")
    assert r.status_code == 503
    assert "detail" in r.json()

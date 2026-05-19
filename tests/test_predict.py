from fastapi.testclient import TestClient

from api.exceptions import PredictionError
from api.main import create_app
from api.service import ModelService


def test_predict_happy_path(client_loaded: TestClient, sample_payload):
    r = client_loaded.post("/predict", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["churn"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}
    assert 0.0 <= body["threshold"] <= 1.0


def test_predict_invalid_payload(client_loaded: TestClient, sample_payload):
    sample_payload["age"] = -5
    r = client_loaded.post("/predict", json=sample_payload)
    assert r.status_code == 422


def test_predict_missing_required_field(client_loaded: TestClient,
                                         sample_payload):
    sample_payload.pop("gender")
    r = client_loaded.post("/predict", json=sample_payload)
    assert r.status_code == 422


def test_predict_unknown_field_rejected(client_loaded: TestClient,
                                        sample_payload):
    sample_payload["evil"] = 1
    r = client_loaded.post("/predict", json=sample_payload)
    assert r.status_code == 422


class _ExplodingService(ModelService):
    def __init__(self):
        super().__init__(model_path="ignored", name="X", version="0")
        self._model = object()

    @property
    def threshold(self) -> float:
        return 0.5

    def predict(self, features):
        raise PredictionError("forced failure")


def test_predict_handles_prediction_error(settings, sample_payload):
    app = create_app(settings=settings, service=_ExplodingService())
    client = TestClient(app)
    r = client.post("/predict", json=sample_payload)
    assert r.status_code == 500
    assert r.json() == {"detail": "forced failure"}


def test_predict_when_service_not_loaded(client_unloaded: TestClient,
                                          sample_payload):
    r = client_unloaded.post("/predict", json=sample_payload)
    assert r.status_code == 503

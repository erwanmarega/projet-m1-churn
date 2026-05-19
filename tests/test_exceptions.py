import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.exceptions import (
    ModelNotLoadedError,
    PredictionError,
    register_exception_handlers,
)


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise-not-loaded")
    def _r1():
        raise ModelNotLoadedError("not loaded")

    @app.get("/raise-prediction")
    def _r2():
        raise PredictionError("boom")

    return app


def test_model_not_loaded_returns_503(app: FastAPI):
    client = TestClient(app)
    r = client.get("/raise-not-loaded")
    assert r.status_code == 503
    assert r.json() == {"detail": "not loaded"}


def test_prediction_error_returns_500(app: FastAPI):
    client = TestClient(app)
    r = client.get("/raise-prediction")
    assert r.status_code == 500
    assert r.json() == {"detail": "boom"}

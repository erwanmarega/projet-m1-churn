from pathlib import Path

from fastapi.testclient import TestClient

from api.config import Settings
from api.main import app as module_app
from api.main import create_app
from api.service import ModelService


def test_create_app_with_defaults_uses_env_settings():
    app = create_app()
    assert app.title == "Churn Prediction API"
    assert hasattr(app.state, "model_service")


def test_module_level_app_exists():
    assert module_app is not None
    assert module_app.title == "Churn Prediction API"


def test_lifespan_loads_model_when_available(settings: Settings,
                                              loaded_service: ModelService):
    svc = ModelService(model_path=settings.model_path,
                       name=settings.model_name,
                       version=settings.model_version)
    assert not svc.is_loaded
    app = create_app(settings=settings, service=svc)
    with TestClient(app):
        assert svc.is_loaded


def test_lifespan_swallows_load_error(settings: Settings, tmp_path: Path):
    bad_settings = Settings(
        model_path=tmp_path / "absent.pkl",
        model_name=settings.model_name,
        model_version=settings.model_version,
    )
    svc = ModelService(model_path=bad_settings.model_path,
                       name=bad_settings.model_name,
                       version=bad_settings.model_version)
    app = create_app(settings=bad_settings, service=svc)
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "degraded"

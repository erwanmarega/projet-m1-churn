from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.config import Settings
from api.main import create_app
from api.service import ModelService

ROOT = Path(__file__).resolve().parent.parent
FINAL_MODEL = ROOT / "models" / "final_model.pkl"


SAMPLE_PAYLOAD: dict[str, Any] = {
    "gender": "Male",
    "age": 42,
    "customer_segment": "SME",
    "tenure_months": 12,
    "signup_channel": "Web",
    "contract_type": "Monthly",
    "monthly_logins": 20,
    "weekly_active_days": 4,
    "avg_session_time": 12.5,
    "features_used": 5,
    "usage_growth_rate": 0.10,
    "last_login_days_ago": 5,
    "monthly_fee": 40.0,
    "total_revenue": 480.0,
    "payment_method": "PayPal",
    "payment_failures": 0,
    "discount_applied": "No",
    "price_increase_last_3m": "No",
    "support_tickets": 1,
    "avg_resolution_time": 12.0,
    "complaint_type": "Service",
    "csat_score": 4.0,
    "escalations": 0,
    "email_open_rate": 0.6,
    "marketing_click_rate": 0.3,
    "nps_score": 30,
    "survey_response": "Satisfied",
    "referral_count": 1,
}


@pytest.fixture
def sample_payload() -> dict[str, Any]:
    return dict(SAMPLE_PAYLOAD)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        model_path=FINAL_MODEL,
        model_name="Test Model",
        model_version="0.0.1-test",
        api_title="Test API",
        api_description="desc",
    )


@pytest.fixture
def loaded_service(settings: Settings) -> ModelService:
    if not FINAL_MODEL.exists():
        pytest.skip("Le modèle final n'a pas été entraîné (run `uv run python train.py`)")
    svc = ModelService(model_path=settings.model_path,
                       name=settings.model_name,
                       version=settings.model_version)
    svc.load()
    return svc


@pytest.fixture
def unloaded_service(tmp_path: Path) -> ModelService:
    return ModelService(model_path=tmp_path / "missing.pkl",
                        name="Test Model",
                        version="0.0.1-test")


@pytest.fixture
def client_loaded(settings: Settings, loaded_service: ModelService) -> TestClient:
    app = create_app(settings=settings, service=loaded_service)
    return TestClient(app)


@pytest.fixture
def client_unloaded(settings: Settings,
                     unloaded_service: ModelService) -> TestClient:
    app = create_app(settings=settings, service=unloaded_service)
    return TestClient(app)

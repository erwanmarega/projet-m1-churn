import pytest
from pydantic import ValidationError

from api.schemas import (
    CustomerFeatures,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    risk_level_for,
)


@pytest.mark.parametrize("proba,expected", [
    (0.0, "low"),
    (0.29, "low"),
    (0.3, "medium"),
    (0.59, "medium"),
    (0.6, "high"),
    (0.99, "high"),
])
def test_risk_level_thresholds(proba, expected):
    assert risk_level_for(proba) == expected


def test_customer_features_valid(sample_payload):
    obj = CustomerFeatures(**sample_payload)
    assert obj.age == sample_payload["age"]


def test_customer_features_age_out_of_range(sample_payload):
    sample_payload["age"] = 999
    with pytest.raises(ValidationError):
        CustomerFeatures(**sample_payload)


def test_customer_features_invalid_contract(sample_payload):
    sample_payload["contract_type"] = "Weekly"
    with pytest.raises(ValidationError):
        CustomerFeatures(**sample_payload)


def test_customer_features_negative_payment_failures(sample_payload):
    sample_payload["payment_failures"] = -1
    with pytest.raises(ValidationError):
        CustomerFeatures(**sample_payload)


def test_customer_features_extra_field_forbidden(sample_payload):
    sample_payload["unknown_field"] = "x"
    with pytest.raises(ValidationError):
        CustomerFeatures(**sample_payload)


def test_customer_features_optional_complaint_can_be_none(sample_payload):
    sample_payload["complaint_type"] = None
    sample_payload["csat_score"] = None
    sample_payload["survey_response"] = None
    obj = CustomerFeatures(**sample_payload)
    assert obj.complaint_type is None
    assert obj.csat_score is None


def test_prediction_response():
    r = PredictionResponse(churn=1, probability=0.8, threshold=0.5,
                           risk_level="high")
    assert r.risk_level == "high"


def test_health_response():
    r = HealthResponse(status="ok", model_loaded=True, model_name="n",
                       model_version="v")
    assert r.status == "ok"


def test_model_info_response():
    r = ModelInfoResponse(name="n", version="v", threshold=0.3,
                          features=["a", "b"])
    assert r.features == ["a", "b"]


def test_error_response():
    assert ErrorResponse(detail="boom").detail == "boom"

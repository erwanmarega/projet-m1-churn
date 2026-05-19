"""Schémas Pydantic : entrées/sorties de l'API."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Male", "Female", "Other"]
    age: int = Field(ge=0, le=120)
    customer_segment: str
    tenure_months: int = Field(ge=0)
    signup_channel: str
    contract_type: Literal["Monthly", "Yearly", "Two-Year"]
    monthly_logins: int = Field(ge=0)
    weekly_active_days: int = Field(ge=0, le=7)
    avg_session_time: float = Field(ge=0)
    features_used: int = Field(ge=0)
    usage_growth_rate: float
    last_login_days_ago: int = Field(ge=0)
    monthly_fee: float = Field(ge=0)
    total_revenue: float = Field(ge=0)
    payment_method: str
    payment_failures: int = Field(ge=0)
    discount_applied: Literal["Yes", "No"]
    price_increase_last_3m: Literal["Yes", "No"]
    support_tickets: int = Field(ge=0)
    avg_resolution_time: float = Field(ge=0)
    complaint_type: Optional[str] = None
    csat_score: Optional[float] = Field(default=None, ge=0, le=5)
    escalations: int = Field(ge=0)
    email_open_rate: float = Field(ge=0, le=1)
    marketing_click_rate: float = Field(ge=0, le=1)
    nps_score: int = Field(ge=-100, le=100)
    survey_response: Optional[str] = None
    referral_count: int = Field(ge=0)


class PredictionResponse(BaseModel):
    churn: int = Field(description="0 = pas de churn, 1 = churn prédit")
    probability: float = Field(ge=0.0, le=1.0,
                                description="Probabilité de churn (classe 1)")
    threshold: float = Field(ge=0.0, le=1.0,
                              description="Seuil de décision appliqué")
    risk_level: Literal["low", "medium", "high"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str
    model_version: str


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    threshold: float
    features: list[str]


class ErrorResponse(BaseModel):
    detail: str


def risk_level_for(probability: float) -> Literal["low", "medium", "high"]:
    if probability >= 0.6:
        return "high"
    if probability >= 0.3:
        return "medium"
    return "low"

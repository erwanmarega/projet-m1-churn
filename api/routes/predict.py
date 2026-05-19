from fastapi import APIRouter, Depends

from api.dependencies import get_model_service
from api.schemas import (
    CustomerFeatures,
    PredictionResponse,
    risk_level_for,
)
from api.service import ModelService

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: CustomerFeatures,
    service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    pred, proba = service.predict(payload.model_dump())
    return PredictionResponse(
        churn=pred,
        probability=proba,
        threshold=service.threshold,
        risk_level=risk_level_for(proba),
    )

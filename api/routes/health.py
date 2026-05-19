from fastapi import APIRouter, Depends

from api.dependencies import get_model_service
from api.exceptions import ModelNotLoadedError
from api.schemas import HealthResponse, ModelInfoResponse
from api.service import ModelService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(service: ModelService = Depends(get_model_service)) -> HealthResponse:
    return HealthResponse(
        status="ok" if service.is_loaded else "degraded",
        model_loaded=service.is_loaded,
        model_name=service.name,
        model_version=service.version,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(
    service: ModelService = Depends(get_model_service),
) -> ModelInfoResponse:
    if not service.is_loaded:
        raise ModelNotLoadedError("Modèle non chargé.")
    return ModelInfoResponse(
        name=service.name,
        version=service.version,
        threshold=service.threshold,
        features=service.feature_names(),
    )

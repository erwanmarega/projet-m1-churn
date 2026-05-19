from fastapi import Request
from api.service import ModelService

def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service

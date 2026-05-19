from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import Settings, get_settings
from api.exceptions import ModelNotLoadedError, register_exception_handlers
from api.routes import health as health_routes
from api.routes import predict as predict_routes
from api.service import ModelService


def create_app(
    settings: Settings | None = None,
    service: ModelService | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    service = service or ModelService(
        model_path=settings.model_path,
        name=settings.model_name,
        version=settings.model_version,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            service.load()
        except ModelNotLoadedError:
            pass
        app.state.model_service = service
        yield

    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.model_version,
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allowing all origins for development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.state.model_service = service
    register_exception_handlers(app)
    app.include_router(health_routes.router)
    app.include_router(predict_routes.router)
    return app


app = create_app()

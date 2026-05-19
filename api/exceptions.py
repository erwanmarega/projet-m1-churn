from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class ModelNotLoadedError(RuntimeError):
    """Le modèle n'a pas chargé au démarrage"""

class PredictionError(RuntimeError):
    """Échec lors de l'inférence."""

def _model_not_loaded_handler(request: Request,
                              exc: ModelNotLoadedError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def _prediction_error_handler(request: Request,
                               exc: PredictionError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ModelNotLoadedError, _model_not_loaded_handler)
    app.add_exception_handler(PredictionError, _prediction_error_handler)

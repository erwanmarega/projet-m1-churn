from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from api.exceptions import ModelNotLoadedError, PredictionError
from src.features import engineer_features


class ModelService:
    def __init__(self, model_path: Path, name: str, version: str) -> None:
        self._model_path = Path(model_path)
        self._name = name
        self._version = version
        self._model: Any | None = None

    def load(self) -> None:
        if not self._model_path.exists():
            raise ModelNotLoadedError(
                f"Modèle introuvable : {self._model_path}")
        try:
            with self._model_path.open("rb") as f:
                self._model = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError) as exc:
            raise ModelNotLoadedError(
                f"Impossible de désérialiser le modèle : {exc}") from exc

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def threshold(self) -> float:
        if self._model is None:
            raise ModelNotLoadedError("Modèle non chargé.")
        return float(getattr(self._model, "threshold", 0.5))

    def feature_names(self) -> list[str]:
        if self._model is None:
            raise ModelNotLoadedError("Modèle non chargé.")
        pipeline = getattr(self._model, "pipeline", self._model)
        prep = pipeline.named_steps["prep"]
        return [str(c) for c in prep.feature_names_in_]

    def predict(self, features: dict) -> tuple[int, float]:
        if self._model is None:
            raise ModelNotLoadedError("Modèle non chargé.")
        try:
            df = pd.DataFrame([features])
            df = engineer_features(df)
            proba = float(self._model.predict_proba(df)[0, 1])
            pred = int(proba >= self.threshold)
            return pred, proba
        except Exception as exc:
            raise PredictionError(f"Échec de l'inférence : {exc}") from exc

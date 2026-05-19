from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    model_path: Path = _ROOT / "models" / "final_model.pkl"
    model_name: str = "Gradient Boosting (HistGradientBoosting)"
    model_version: str = "1.0.0"
    api_title: str = "Churn Prediction API"
    api_description: str = (
        "Service d'inférence pour la prédiction de churn client. "
        "Modèle candidat : Gradient Boosting."
    )

    model_config = SettingsConfigDict(
        env_prefix="CHURN_API_",
        env_file=None,
        extra="ignore",
        protected_namespaces=("settings_",),
    )


def get_settings() -> Settings:
    return Settings()

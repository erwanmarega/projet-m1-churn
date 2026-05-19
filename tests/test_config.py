from pathlib import Path

from api.config import Settings, get_settings


def test_defaults(monkeypatch):
    for key in list(__import__("os").environ.keys()):
        if key.startswith("CHURN_API_"):
            monkeypatch.delenv(key, raising=False)
    s = Settings()
    assert s.api_title == "Churn Prediction API"
    assert s.model_version == "1.0.0"
    assert s.model_path.name == "final_model.pkl"


def test_env_override(monkeypatch):
    monkeypatch.setenv("CHURN_API_MODEL_NAME", "Custom")
    monkeypatch.setenv("CHURN_API_MODEL_VERSION", "9.9.9")
    monkeypatch.setenv("CHURN_API_MODEL_PATH", "/tmp/foo.pkl")
    s = Settings()
    assert s.model_name == "Custom"
    assert s.model_version == "9.9.9"
    assert s.model_path == Path("/tmp/foo.pkl")


def test_get_settings_returns_settings_instance():
    assert isinstance(get_settings(), Settings)

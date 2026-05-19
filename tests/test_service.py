import pickle
from pathlib import Path

import pytest

from api.exceptions import ModelNotLoadedError, PredictionError
from api.service import ModelService


def test_load_missing_file_raises(tmp_path: Path):
    svc = ModelService(model_path=tmp_path / "no.pkl",
                       name="n", version="v")
    assert not svc.is_loaded
    with pytest.raises(ModelNotLoadedError):
        svc.load()


def test_load_corrupt_file_raises(tmp_path: Path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a valid pickle stream")
    svc = ModelService(model_path=bad, name="n", version="v")
    with pytest.raises(ModelNotLoadedError):
        svc.load()


def test_load_empty_file_raises(tmp_path: Path):
    empty = tmp_path / "empty.pkl"
    empty.write_bytes(b"")
    svc = ModelService(model_path=empty, name="n", version="v")
    with pytest.raises(ModelNotLoadedError):
        svc.load()


def test_load_attribute_error(tmp_path: Path, monkeypatch):
    bad = tmp_path / "attr.pkl"
    bad.write_bytes(pickle.dumps({"k": "v"}))

    def fake_load(_):
        raise AttributeError("Can't get attribute 'Foo' on <module>")

    monkeypatch.setattr("api.service.pickle.load", fake_load)
    svc = ModelService(model_path=bad, name="n", version="v")
    with pytest.raises(ModelNotLoadedError):
        svc.load()


def test_is_loaded_after_load(loaded_service: ModelService):
    assert loaded_service.is_loaded is True


def test_properties(loaded_service: ModelService):
    assert loaded_service.name == "Test Model"
    assert loaded_service.version == "0.0.1-test"
    assert 0.0 < loaded_service.threshold < 1.0


def test_threshold_when_not_loaded(unloaded_service: ModelService):
    with pytest.raises(ModelNotLoadedError):
        _ = unloaded_service.threshold


def test_feature_names_when_loaded(loaded_service: ModelService):
    names = loaded_service.feature_names()
    assert isinstance(names, list) and len(names) > 0
    assert all(isinstance(n, str) for n in names)


def test_feature_names_when_not_loaded(unloaded_service: ModelService):
    with pytest.raises(ModelNotLoadedError):
        unloaded_service.feature_names()


def test_predict_when_not_loaded(unloaded_service: ModelService, sample_payload):
    with pytest.raises(ModelNotLoadedError):
        unloaded_service.predict(sample_payload)


def test_predict_returns_class_and_proba(loaded_service: ModelService,
                                          sample_payload):
    pred, proba = loaded_service.predict(sample_payload)
    assert pred in (0, 1)
    assert 0.0 <= proba <= 1.0


def test_predict_invalid_input_wraps_exception(loaded_service: ModelService):
    with pytest.raises(PredictionError):
        loaded_service.predict({"not": "real"})

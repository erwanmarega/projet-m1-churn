from unittest.mock import MagicMock

from api.dependencies import get_model_service


def test_get_model_service_reads_from_app_state():
    sentinel = object()
    request = MagicMock()
    request.app.state.model_service = sentinel
    assert get_model_service(request) is sentinel

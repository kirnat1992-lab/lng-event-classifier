"""API behavior tests: input validation, model availability, and inference."""
import pytest
from fastapi.testclient import TestClient

from lng_event_classifier.api import create_app
from lng_event_classifier.labels import LABELS
from lng_event_classifier.train import train_and_evaluate


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("api_model")
    train_and_evaluate(
        "data/synthetic_events.csv", root / "artifacts", root / "reports"
    )
    return root / "artifacts" / "event_classifier.joblib"


def test_health_reports_missing_model(tmp_path):
    client = TestClient(create_app(tmp_path / "not-trained.joblib"))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "model_missing", "model_ready": False}


def test_predict_missing_model_returns_503(tmp_path):
    client = TestClient(create_app(tmp_path / "not-trained.joblib"))
    response = client.post("/predict", json={"text": "A vessel departed the terminal."})
    assert response.status_code == 503
    assert "Train" in response.json()["detail"] or "run:" in response.json()["detail"]


def test_health_reports_ready_after_training(trained_model):
    client = TestClient(create_app(trained_model))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_ready": True}


def test_predict_returns_valid_label_and_uncalibrated_score(trained_model):
    client = TestClient(create_app(trained_model))
    response = client.post(
        "/predict",
        json={"text": "The carrier began unloading LNG at an import terminal."},
    )
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"label", "model_probability", "note"}
    assert result["label"] in LABELS
    assert 0 <= result["model_probability"] <= 1
    assert "Uncalibrated" in result["note"]


@pytest.mark.parametrize("payload", [
    {"text": ""},
    {"text": "  \t  "},
    {"text": "x" * 2001},
    {},
    {"text": 123},
])
def test_predict_rejects_invalid_input_without_loading_model(tmp_path, payload):
    client = TestClient(create_app(tmp_path / "absent.joblib"))
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

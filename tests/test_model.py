import json
from lng_event_classifier.labels import LABELS
from lng_event_classifier.train import train_and_evaluate
from lng_event_classifier.predict import predict_text


def test_end_to_end_train_predict(tmp_path):
    metrics = train_and_evaluate("data/synthetic_events.csv", tmp_path / "model", tmp_path / "reports")
    assert metrics["train_examples"] == 80
    assert metrics["test_examples"] == 40
    assert set(metrics["labels"]) == set(LABELS)
    assert 0 <= metrics["macro_f1"] <= 1
    assert (tmp_path / "reports" / "confusion_matrix.png").exists()
    assert (tmp_path / "reports" / "test_predictions.csv").exists()
    assert json.loads((tmp_path / "reports" / "metrics.json").read_text())["macro_f1"] == metrics["macro_f1"]
    prediction = predict_text("A ship began loading LNG at an export terminal", tmp_path / "model" / "event_classifier.joblib")
    assert prediction["label"] in LABELS
    assert 0 <= prediction["model_probability"] <= 1
    assert "Uncalibrated" in prediction["note"]


def test_reproducible_training(tmp_path):
    one = train_and_evaluate("data/synthetic_events.csv", tmp_path / "m1", tmp_path / "r1")
    two = train_and_evaluate("data/synthetic_events.csv", tmp_path / "m2", tmp_path / "r2")
    assert one == two

import pandas as pd
import pytest
from lng_event_classifier.data import load_events
from lng_event_classifier.labels import LABELS


def test_sample_has_valid_splits_and_all_labels():
    df = load_events("data/synthetic_events.csv")
    assert len(df) == 120
    assert set(df["label"]) == set(LABELS)
    assert (df["provenance"] == "SYNTHETIC").all()
    assert df["source_url"].eq("").all()
    assert len(df.loc[df.split == "test"]) == 40


def test_duplicate_text_rejected(tmp_path):
    df = pd.read_csv("data/synthetic_events.csv", keep_default_na=False)
    df.loc[1, "text"] = df.loc[0, "text"]
    path = tmp_path / "duplicated.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Repeated/identical"):
        load_events(path)


def test_public_record_without_source_rejected(tmp_path):
    df = pd.read_csv("data/synthetic_events.csv", keep_default_na=False)
    df.loc[0, "provenance"] = "PUBLIC_VERIFIED"
    path = tmp_path / "uncited.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="source_url"):
        load_events(path)

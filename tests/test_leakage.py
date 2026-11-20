"""Regression tests for train/test leakage guards using synthetic examples."""

import pandas as pd
import pytest

from lng_event_classifier.data import load_events


def _example_indices(df: pd.DataFrame) -> tuple[int, int]:
    """Pick one example from each predeclared split."""
    train_index = int(df.index[df["split"] == "train"][0])
    test_index = int(df.index[df["split"] == "test"][0])
    return train_index, test_index


def test_same_source_group_cannot_span_train_and_test(tmp_path):
    """Two distinct reports from one source group must stay in one split."""
    df = pd.read_csv("data/synthetic_events.csv", keep_default_na=False)
    train_index, test_index = _example_indices(df)

    df["source_group"] = [f"synthetic-source-{i}" for i in range(len(df))]
    df.loc[test_index, "source_group"] = df.loc[train_index, "source_group"]

    path = tmp_path / "leaked_source_groups.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Source groups appear in both train and test"):
        load_events(path)


def test_repeated_text_with_case_and_spacing_variations_rejected(tmp_path):
    """Formatting changes must not conceal duplicates across data splits."""
    df = pd.read_csv("data/synthetic_events.csv", keep_default_na=False)
    train_index, test_index = _example_indices(df)

    original = df.loc[train_index, "text"]
    df.loc[test_index, "text"] = "   " + " \t ".join(original.upper().split()) + "  "

    path = tmp_path / "normalized_duplicate.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Repeated/identical event text"):
        load_events(path)

"""Data loading and honest split validation."""
from pathlib import Path
import pandas as pd
from .labels import LABELS

REQUIRED_COLUMNS = {"event_id", "text", "label", "split", "provenance", "source_url"}


def load_events(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if df["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs found")
    if df["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Blank event text found")
    normalized = df["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    if normalized.duplicated().any():
        raise ValueError("Repeated/identical event text found; check leakage")
    if not set(df["label"]).issubset(set(LABELS)):
        raise ValueError("Unknown event label(s): " + ", ".join(sorted(set(df["label"]) - set(LABELS))))
    if set(df["split"]) != {"train", "test"}:
        raise ValueError("Both predeclared 'train' and 'test' splits are required")
    if not df["provenance"].isin({"SYNTHETIC", "PUBLIC_VERIFIED"}).all():
        raise ValueError("Provenance must be SYNTHETIC or PUBLIC_VERIFIED")
    public_rows = df["provenance"] == "PUBLIC_VERIFIED"
    if (df.loc[public_rows, "source_url"].astype(str).str.strip() == "").any():
        raise ValueError("Verified public records must carry a source_url")
    for label in LABELS:
        if not ((df["label"] == label) & (df["split"] == "train")).any():
            raise ValueError(f"Missing training examples for: {label}")
        if not ((df["label"] == label) & (df["split"] == "test")).any():
            raise ValueError(f"Missing test examples for: {label}")
    if "source_group" in df.columns:
        overlap = set(df.loc[df["split"] == "train", "source_group"]) & set(df.loc[df["split"] == "test", "source_group"])
        if overlap - {""}:
            raise ValueError("Source groups appear in both train and test; possible leakage")
    return df

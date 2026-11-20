"""Predict a label for one short event report using a locally trained model."""
import argparse
import json
from pathlib import Path
import joblib


def predict_text(text: str, model_path: str | Path) -> dict:
    if not text.strip():
        raise ValueError("The --text argument must not be blank")
    # Only load models created by you; joblib/pickle is unsafe for untrusted files.
    model = joblib.load(model_path)
    label = model.predict([text])[0]
    confidence = float(model.predict_proba([text]).max())
    return {"label": str(label), "model_probability": round(confidence, 4), "note": "Uncalibrated model score; not a verification or evidence confidence rating."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="One short LNG-related event description")
    parser.add_argument("--model", default="artifacts/event_classifier.joblib")
    args = parser.parse_args()
    print(json.dumps(predict_text(args.text, args.model), indent=2))


if __name__ == "__main__":
    main()

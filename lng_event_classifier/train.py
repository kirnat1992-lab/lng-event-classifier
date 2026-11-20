"""CLI to train a simple baseline and write reproducible evaluation artifacts."""
import argparse
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from .data import load_events
from .labels import LABELS
from .model import build_pipeline


def train_and_evaluate(data_path: str | Path, model_dir: str | Path, report_dir: str | Path) -> dict:
    df = load_events(data_path)
    train = df.loc[df["split"] == "train"]
    test = df.loc[df["split"] == "test"]
    pipe = build_pipeline()
    pipe.fit(train["text"].tolist(), train["label"].tolist())
    predicted = pipe.predict(test["text"].tolist())
    probabilities = pipe.predict_proba(test["text"].tolist()).max(axis=1)
    label_list = list(LABELS)
    report = classification_report(test["label"], predicted, labels=label_list, output_dict=True, zero_division=0)
    metrics = {
        "disclaimer": "Small, handcrafted synthetic teaching dataset; not a real-world performance estimate.",
        "dataset": str(Path(data_path).name),
        "split": "predeclared (no random splitting)",
        "train_examples": int(len(train)),
        "test_examples": int(len(test)),
        "labels": label_list,
        "accuracy": round(float(report["accuracy"]), 4),
        "macro_precision": round(float(report["macro avg"]["precision"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "per_label": {name: {key: round(float(report[name][key]), 4) if key != "support" else int(report[name][key]) for key in ("precision", "recall", "f1-score", "support")} for name in label_list},
    }
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_dir / "event_classifier.joblib")
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    predictions = test[["event_id", "text", "label"]].copy()
    predictions["predicted_label"] = predicted
    predictions["predicted_probability"] = [round(float(p), 4) for p in probabilities]
    predictions["correct"] = predictions["label"] == predictions["predicted_label"]
    predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    cm = confusion_matrix(test["label"], predicted, labels=label_list)
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list).plot(ax=ax, xticks_rotation=55, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Held-out test: synthetic LNG event reports")
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic_events.csv")
    parser.add_argument("--model-dir", default="artifacts")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()
    result = train_and_evaluate(args.data, args.model_dir, args.report_dir)
    print(f"Synthetic demo: {result['train_examples']} train / {result['test_examples']} test; macro-F1={result['macro_f1']:.3f}, accuracy={result['accuracy']:.3f}")
    print("These values do not measure performance on real shipping reports.")


if __name__ == "__main__":
    main()

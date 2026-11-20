"""Auditable, intentionally simple NLP baseline."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
    ])

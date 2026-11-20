# FastAPI inference demo

This small HTTP service wraps the existing **synthetic-data NLP classifier**.
It does **not** identify verified vessel movements and should not be deployed
as a public, authoritative shipping intelligence service.

## Install and train

From the project root, with Python 3.10+:

```bash
python -m pip install -r requirements-dev.txt
python -m lng_event_classifier.train
uvicorn lng_event_classifier.api:app --reload
```

The server runs locally at http://127.0.0.1:8000. Open
http://127.0.0.1:8000/docs for Swagger's interactive documentation.
Never load externally supplied or untrusted `joblib` model files.

## Try the service

Check readiness:

```bash
curl http://127.0.0.1:8000/health
```

Example response after training:

```json
{"status":"ready","model_ready":true}
```

Send one prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"The carrier began unloading LNG into shore tanks."}'
```

Response shape (label and score depend on the locally trained model):

```json
{
  "label": "discharge",
  "model_probability": 0.1952,
  "note": "Uncalibrated model score; not a verification or evidence confidence rating."
}
```

The probability above illustrates the response format; it is not a verified
sample output from this API version.

## Error behavior

- `422`: missing, empty, whitespace-only, non-string, or over-2,000-character text
- `503`: the model file is missing; run the training command first
- `/health`: reports whether the local model file exists; it does not prove
  that model predictions are accurate or trustworthy.

The app caches the model after the first successful prediction. If you retrain
while the API is running, restart the server to load the new artifact.

## Tests

```bash
python -m pytest -q
```

API tests cover model readiness, missing-model errors, valid responses and
invalid input. These tests do not establish out-of-sample performance on real
shipping reports.

## Security and deployment limits

Run this locally as a teaching demo. The prototype has no authentication,
rate limiting, monitoring or protection against hostile deployments. Do not
publish proprietary or confidential material, expose an unrestricted public
endpoint, or interpret uncalibrated model probabilities as evidence confidence.

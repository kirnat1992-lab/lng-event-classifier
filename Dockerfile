FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY lng_event_classifier ./lng_event_classifier
COPY data ./data
CMD ["python", "-m", "lng_event_classifier.train"]

from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import random

# ===== METRICS UNTUK EARLY WARNING =====
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total number of inference requests"
)

FAILURE_COUNT = Counter(
    "inference_failures_total",
    "Total number of failed predictions"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency of model predictions"
)

CONFIDENCE_SCORE = Gauge(
    "prediction_confidence",
    "Confidence score of prediction"
)

ANOMALY_SCORE = Gauge(
    "early_warning_score",
    "Early warning anomaly score"
)

PREDICTION_TOTAL = Counter(
    "model_prediction_total",
    "Total number of predictions"
)

PREDICTION_FAILURE = Counter(
    "model_prediction_failure",
    "Total failed predictions"
)

PREDICTION_SUCCESS = Counter(
    "model_prediction_success",
    "Total successful predictions"
)

MODEL_LATENCY = Histogram(
    "model_latency_seconds",
    "Inference latency"
)

def simulate_inference():
    REQUEST_COUNT.inc()

    start = time.time()
    time.sleep(random.uniform(0.1, 0.5))  # simulasi latency
    latency = time.time() - start
    PREDICTION_LATENCY.observe(latency)

    confidence = random.uniform(0.6, 0.99)
    CONFIDENCE_SCORE.set(confidence)

    anomaly_score = random.uniform(0, 1)
    ANOMALY_SCORE.set(anomaly_score)

    if anomaly_score > 0.8:
        FAILURE_COUNT.inc()

if __name__ == "__main__":
    start_http_server(8000)
    print("Prometheus Exporter running on port 8000")

    while True:
        simulate_inference()
        time.sleep(2)

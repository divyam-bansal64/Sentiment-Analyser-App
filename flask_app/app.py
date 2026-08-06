import os
import time
import warnings
import pandas as pd
from flask import Flask, render_template, request
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

import sys
from pathlib import Path

# Add project root and flask_app directory to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

FLASK_DIR = Path(__file__).resolve().parent
if str(FLASK_DIR) not in sys.path:
    sys.path.insert(0, str(FLASK_DIR))

from preprocessing_utility import preprocess_text
from load_model import load_registered_model, load_vectorizer

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)

# Prometheus metrics setup
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)
MODEL_INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds", "Latency of model vectorization and prediction in seconds", registry=registry
)
INPUT_TEXT_LENGTH_WORDS = Histogram(
    "input_text_length_words", "Distribution of submitted review word counts", registry=registry,
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)
EMPTY_INPUT_COUNT = Counter(
    "empty_input_request_count", "Count of requests with empty or whitespace-only inputs", registry=registry
)

# ------------------------------------------------------------------------------------------
# Model and vectorizer setup (congruent with capstone_src pipeline)
# ------------------------------------------------------------------------------------------
print("Initializing model and vectorizer for Flask application...")
vectorizer = load_vectorizer()
model = load_registered_model()


# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    raw_text = request.form.get("text", "").strip()

    # Track empty or whitespace-only input metric (Metric #5)
    if not raw_text:
        EMPTY_INPUT_COUNT.inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        return render_template("index.html", result="Invalid Input: Please enter review text.")

    # Track input review word count distribution (Metric #3)
    word_count = len(raw_text.split())
    INPUT_TEXT_LENGTH_WORDS.observe(word_count)

    # Clean text using standardized preprocessing pipeline
    cleaned_text = preprocess_text(raw_text)

    # Measure pure model inference latency (vectorization + prediction) (Metric #1)
    inference_start = time.time()
    features = vectorizer.transform([cleaned_text])

    # Predict sentiment directly on sparse matrix (prevents dense array OOM memory overhead)
    try:
        result = model.predict(features)
    except Exception:
        # Fallback if model wrapper requires dense input
        result = model.predict(features.toarray())

    MODEL_INFERENCE_LATENCY.observe(time.time() - inference_start)

    prediction_raw = result[0]
    
    # Map binary output (1 -> Positive, 0 -> Negative) if numeric
    if str(prediction_raw) in ["1", "1.0", "positive"]:
        prediction_label = "Positive"
    else:
        prediction_label = "Negative"

    # Increment prediction count metric
    PREDICTION_COUNT.labels(prediction=prediction_label).inc()

    # Measure total request latency
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction_label)


@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request


MODEL_DIR = "/opt/ml/model"

model = joblib.load(
    os.path.join(MODEL_DIR, "credit_risk_lgbm_v2.pkl")
)

with open(
    os.path.join(MODEL_DIR, "feature_config.json")
) as f:
    feature_config = json.load(f)

with open(
    os.path.join(MODEL_DIR, "threshold_config.json")
) as f:
    threshold_config = json.load(f)


FEATURES = feature_config["feature_columns"]
CATEGORICAL = feature_config["categorical_columns"]

app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    return "", 200


@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        payload = request.get_json(force=True)

        if isinstance(payload, dict):
            payload = [payload]

        df = pd.DataFrame(payload)

        missing = [
            col
            for col in FEATURES
            if col not in df.columns
        ]

        if missing:
            return jsonify(
                {
                    "error": "missing_features",
                    "missing_features": missing,
                }
            ), 400

        X = df[FEATURES].copy()

        for col in CATEGORICAL:
            X[col] = X[col].astype("category")

        probabilities = model.predict_proba(X)[:, 1]

        threshold_f1 = threshold_config["threshold_f1"]
        threshold_high_recall = threshold_config[
            "threshold_high_recall"
        ]

        results = []

        for probability in probabilities:
            results.append(
                {
                    "default_probability": float(probability),
                    "prediction_f1": int(
                        probability >= threshold_f1
                    ),
                    "prediction_high_recall": int(
                        probability >= threshold_high_recall
                    ),
                }
            )

        return jsonify(results), 200

    except Exception as exc:
        return jsonify(
            {
                "error": str(exc),
            }
        ), 500
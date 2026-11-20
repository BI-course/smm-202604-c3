
import os

import joblib
import numpy as np
import pandas as pd

from flask import Blueprint, jsonify, request


# ---------------------------------------------------------
# 1. LOAD THE SAVED E-COMMERCE CLUSTERING MODEL
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILE = (
    "20261116_161207_kmeans_ecommerce_customer_segmentation.joblib"
)

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

clustering_artifact = joblib.load(MODEL_PATH)

clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
cluster_map = clustering_artifact["cluster_map"]
segment_names = clustering_artifact["segment_names"]
aov_imputer = clustering_artifact["aov_imputer"]
imputer_predictors = clustering_artifact["imputer_predictors"]
feature_names = clustering_artifact["feature_names"]


# ---------------------------------------------------------
# 2. CREATE A FLASK BLUEPRINT
# ---------------------------------------------------------

clustering_bp = Blueprint("ecommerce_clustering", __name__)


# ---------------------------------------------------------
# 3. E-COMMERCE CUSTOMER CLUSTERING ENDPOINT
# ---------------------------------------------------------

@clustering_bp.route(
    "/api/v1/models/ecommerce-customer-segmenter/predictions",
    methods=["POST"]
)
def predict_ecommerce_customer_segment():

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Request body must be a valid JSON object."
        }), 400

    required_fields = [
        "recency_days",
        "frequency_per_year",
        "avg_order_value_kes",
        "num_categories_purchased",
        "total_lifetime_spend_kes"
    ]

    missing_fields = [
        field for field in required_fields
        if field not in payload
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields.",
            "missing_fields": missing_fields
        }), 400

    # Validate the incoming customer information.
    try:
        for field in required_fields:

            if field == "avg_order_value_kes" and payload[field] is None:
                continue

            value = float(payload[field])

            if not np.isfinite(value):
                raise ValueError(
                    f"{field} must be a finite number."
                )

            if value < 0:
                raise ValueError(
                    f"{field} cannot be negative."
                )

    except (TypeError, ValueError) as error:
        return jsonify({
            "error": str(error)
        }), 400

    try:
        input_row = pd.DataFrame([payload])

        # Recover a missing average order value using
        # the saved regression imputer.
        if pd.isna(input_row.loc[0, "avg_order_value_kes"]):

            input_row["aov_recovered"] = aov_imputer.predict(
                input_row[imputer_predictors].astype(float)
            )

            aov_source = "imputed"

        else:
            input_row["aov_recovered"] = input_row[
                "avg_order_value_kes"
            ].astype(float)

            aov_source = "provided"

        # Select the exact four features used during training.
        model_input = input_row[feature_names].astype(float)

        # Apply the saved scaler.
        scaled_input = clustering_scaler.transform(model_input)

        # Predict the raw K-Means cluster.
        raw_cluster = int(
            clustering_kmeans.predict(scaled_input)[0]
        )

        # Convert the raw cluster into the meaningful segment.
        segment_id = int(cluster_map[raw_cluster])
        segment_name = segment_names[segment_id]

        return jsonify({
            "raw_cluster": raw_cluster,
            "segment_id": segment_id,
            "segment": segment_name,
            "aov_source": aov_source,
            "model": "K-Means E-commerce Customer Segmentation",
            "number_of_clusters": 5
        }), 200

    except Exception as error:
        return jsonify({
            "error": "Unable to generate customer segment.",
            "details": str(error)
        }), 500

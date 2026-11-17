import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)

CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",  # VS Code Live Server extension
    "null",  # allows opening a frontend .html file directly (file://) with no web server
]}})


MODEL_DIR = "./model"

regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "regression.joblib")
)

classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "corporate_distress_pipeline.joblib")
)

clustering_artifact = joblib.load(
    os.path.join(MODEL_DIR, "20261117_144405_kmeans_ecommerce_customer_segmentation.joblib")
)
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
cluster_map = clustering_artifact["cluster_map"]
segment_names = clustering_artifact["segment_names"]
aov_imputer = clustering_artifact["aov_imputer"]
imputer_predictors = clustering_artifact["imputer_predictors"]
feature_names = clustering_artifact["feature_names"]

arm_artifact = joblib.load(
    os.path.join(MODEL_DIR, "apriori_grocery_basket_rules.joblib")
)
recommendation_lookup = arm_artifact["recommendation_lookup"]

print("All the four artifacts have been loaded. The API is ready.")


# 1. REGRESSION -- predict daily retail sales revenue (KES)

@app.route("/api/v1/models/retail-daily-sales-regressor/predictions", methods=["POST"])
def predict_daily_sales():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "A valid JSON object is required."}), 400

    required_features = [
        "store_size_sqm",
        "footfall_count",
        "promotions_active",
        "avg_basket_size_kes",
        "store_age_years",
        "online_order_count",
        "store_format",
        "location_type",
        "delivery_available",
        "store_size_category"
    ]

    missing_features = [
        feature for feature in required_features
        if feature not in payload
    ]

    if missing_features:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing_features
        }), 400

    input_row = pd.DataFrame(
        [[payload[feature] for feature in required_features]],
        columns=required_features
    )

    numeric_features = [
        "store_size_sqm",
        "footfall_count",
        "promotions_active",
        "avg_basket_size_kes",
        "store_age_years",
        "online_order_count"
    ]

    input_row[numeric_features] = input_row[numeric_features].astype(float)

    predicted_log_sales = regression_pipeline.predict(input_row)[0]

    predicted_sales_kes = float(np.expm1(predicted_log_sales))

    return jsonify({
        "predicted_daily_sales_revenue_kes": round(predicted_sales_kes, 2),
        "model": "retail_daily_sales_regression"
    })

# 2. CLASSIFICATION -- predict corporate financial distress risk

@app.route("/api/v1/models/corporate-financial-distress-classifier/predictions", methods=["POST"])
def predict_credit_risk():
    payload = request.get_json()

    if not isinstance(payload, dict):
        return jsonify({"error": "A valid JSON object is required."}), 400

    required_features = [
        "industry_sector",
        "listing_status",
        "credit_rating_band",
        "audit_going_concern_flag",
        "years_in_operation",
        "num_employees",
        "debt_to_equity_ratio",
        "current_ratio",
        "interest_coverage_ratio",
        "revenue_growth_pct",
        "net_profit_margin_pct",
        "working_capital_to_assets_ratio",
        "macroeconomic_interest_rate_pct"
    ]

    missing_features = [
        feature for feature in required_features
        if feature not in payload
    ]

    if missing_features:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing_features
        }), 400

    input_row = pd.DataFrame(
        [[payload[feature] for feature in required_features]],
        columns=required_features
    )

    predicted_class = classification_pipeline.predict(input_row)[0]

    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]

    class_labels = classification_pipeline.classes_

    return jsonify({
        "predicted_distress_risk_category": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, predicted_probabilities)
        },
        "model": "corporate_financial_distress_classifier"
    })


# 3. CLUSTERING -- predict e-commerce customer segment

@app.route("/api/v1/models/ecommerce-customer-segmenter/predictions", methods=["POST"])
def predict_customer_segment():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Request body must be a valid JSON object."
        }), 400

    required_features = [
        "recency_days",
        "frequency_per_year",
        "num_categories_purchased",
        "total_lifetime_spend_kes"
    ]

    missing_features = [
        feature for feature in required_features
        if feature not in payload
    ]

    if missing_features:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing_features
        }), 400

    try:
        input_row = pd.DataFrame([payload])

        # Validate numeric inputs.
        for feature in required_features:
            value = float(input_row.loc[0, feature])

            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"{feature} must be a finite, non-negative number."
                )

        # Recover missing average order value using the saved imputer.
        if payload.get("avg_order_value_kes") is None:
            imputer_input = input_row[
                imputer_predictors
            ].astype(float)

            input_row["aov_recovered"] = aov_imputer.predict(
                imputer_input
            )
            aov_source = "imputed"

        else:
            aov_value = float(payload["avg_order_value_kes"])

            if not np.isfinite(aov_value) or aov_value < 0:
                raise ValueError(
                    "avg_order_value_kes must be a finite, non-negative number."
                )

            input_row["aov_recovered"] = aov_value
            aov_source = "provided"

        # Select the exact features used during training.
        model_input = input_row[feature_names].astype(float)

        scaled_input = clustering_scaler.transform(model_input)

        raw_cluster = int(
            clustering_kmeans.predict(scaled_input)[0]
        )

        segment_id = int(cluster_map[raw_cluster])
        segment_name = segment_names[segment_id]

        return jsonify({
            "raw_cluster": raw_cluster,
            "segment_id": segment_id,
            "segment": segment_name,
            "aov_source": aov_source,
            "model": "ecommerce_customer_clustering",
            "number_of_clusters": 5
        }), 200

    except (TypeError, ValueError) as error:
        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:
        return jsonify({
            "error": "Unable to generate customer segment.",
            "details": str(error)
        }), 500


# 4. ASSOCIATION RULE MINING -- grocery basket recommendations

@app.route("/api/v1/models/grocery-recommender/predictions", methods=["POST"])
def recommend_products():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Request body must be a valid JSON object."
        }), 400

    cart = payload.get("cart")

    if not isinstance(cart, list) or not all(
        isinstance(item, str) and item.strip() for item in cart
    ):
        return jsonify({
            "error": "'cart' must be a list of grocery item names."
        }), 400

    try:
        top_n = int(payload.get("top_n", 3))

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    cart = set(item.strip() for item in cart)

    candidates = {}

    for rule in recommendation_lookup:
        antecedents = set(rule["antecedents"])
        consequents = set(rule["consequents"])

        if antecedents.issubset(cart):
            for item in consequents - cart:
                confidence = rule["confidence"]
                lift = rule["lift"]

                if (
                    item not in candidates
                    or confidence > candidates[item]["confidence"]
                ):
                    candidates[item] = {
                        "confidence": confidence,
                        "lift": lift
                    }

    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            item[1]["confidence"],
            item[1]["lift"]
        ),
        reverse=True
    )[:top_n]

    recommendations = [
        {
            "product_name": item,
            "confidence": round(float(metrics["confidence"]), 3),
            "lift": round(float(metrics["lift"]), 2)
        }
        for item, metrics in ranked
    ]

    return jsonify({
        "cart": sorted(cart),
        "recommendations": recommendations,
        "model": "Apriori grocery association rules"
    })


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Development server. debug=True gives auto-reload and an interactive
    # debugger on errors -- convenient in class, but NEVER use debug=True in
    # a deployed/public environment (it can expose a remote code execution
    # console to anyone who can reach the server).
    app.run(debug=False, host="0.0.0.0", port=5000)

    # Production-style alternative (no debugger, no auto-reload):
    # app.run(debug=False, host="0.0.0.0", port=5000)

    # HTTPS/TLS alternative for local development (requires a self-signed
    # certificate; see app_server_reverse_proxy_server_setup.md):
    # app.run(debug=False, ssl_context=("cert.pem", "key.pem"))

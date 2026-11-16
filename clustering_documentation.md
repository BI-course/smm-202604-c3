
# Member 3 – E-commerce Customer Clustering

## 1. Overview

This contribution implements the K-Means clustering prediction endpoint for the e-commerce customer behaviour dataset.

The purpose of the endpoint is to assign a new customer to one of five existing customer segments using the previously trained and saved K-Means model.

The endpoint uses the saved model artifacts and does not retrain the model when receiving prediction requests.

## 2. Dataset

Dataset: `ecommerce_customer_behavior.csv`

Number of customer records: 750

The dataset contains customer purchasing behaviour information, including:

- Recency in days
- Purchase frequency per year
- Average order value in KES
- Total lifetime spending in KES
- Number of product categories purchased

## 3. Model Details

Algorithm: K-Means Clustering

Number of clusters: 5

Random state: 53

Number of iterations: 4

Final inertia: 290.6

The model uses StandardScaler for feature scaling.

The four features used for clustering are:

1. `recency_days`
2. `frequency_per_year`
3. `aov_recovered`
4. `num_categories_purchased`

Missing average order values are recovered using the saved regression imputation model.

## 4. Customer Segment Mapping

The raw K-Means labels are mapped to meaningful customer segments based on average lifetime spending.

| Raw K-Means label | Segment ID | Customer segment |
|---:|---:|---|
| 1 | 1 | 1 - Champions |
| 4 | 2 | 2 - Loyal Core |
| 0 | 3 | 3 - Lapsing High-Value |
| 3 | 4 | 4 - New & Promising |
| 2 | 5 | 5 - Dormant |

The mapping is loaded from the saved model artifact.

## 5. Saved Model Artifact

The trained model is saved at:

`model/20261116_161207_kmeans_ecommerce_customer_segmentation.joblib`

The artifact contains:

- Trained K-Means model
- StandardScaler
- Raw cluster-to-segment mapping
- Segment names
- Average order value imputation model
- Imputation predictor information
- Model metadata

The endpoint loads the saved artifact rather than retraining the model.

## 6. Endpoint Implementation

Python file: `clustering_endpoint.py`

Flask endpoint:

`POST /api/v1/models/ecommerce-customer-segmenter/predictions`

The endpoint is implemented as a Flask Blueprint named `clustering_bp` so that it can be registered by Member 5 during API integration.

### Required JSON input

```json
{
  "recency_days": 20,
  "frequency_per_year": 8,
  "avg_order_value_kes": 5000,
  "num_categories_purchased": 4,
  "total_lifetime_spend_kes": 40000
}
```

The `avg_order_value_kes` field can be null. When null, the saved regression imputer estimates the missing value.

### Prediction process

1. Receive customer information as JSON.
2. Validate the required fields and numeric values.
3. Recover missing average order value when necessary.
4. Select the four clustering features.
5. Apply the saved StandardScaler.
6. Predict the raw K-Means cluster.
7. Map the raw cluster to the corresponding customer segment.
8. Return the prediction as JSON.

## 7. Endpoint Testing Results

The endpoint was tested locally using Flask's test client.

### Test 1: Customer with provided average order value

Input:
- Recency: 20 days
- Frequency: 8 purchases per year
- Average order value: KES 5,000
- Categories purchased: 4
- Total lifetime spending: KES 40,000

Result:

HTTP Status: 200

```json
{
  "aov_source": "provided",
  "model": "K-Means E-commerce Customer Segmentation",
  "number_of_clusters": 5,
  "raw_cluster": 4,
  "segment": "2 - Loyal Core",
  "segment_id": 2
}
```

### Test 2: Customer with missing average order value

The same customer information was submitted with `avg_order_value_kes` set to null.

Result:

HTTP Status: 200

The endpoint successfully recovered the missing average order value using the saved imputation model.

Returned segment: `2 - Loyal Core`

AOV source: `imputed`

### Test 3: Missing required field

A request was submitted without `total_lifetime_spend_kes`.

Result:

HTTP Status: 400

```json
{
  "error": "Missing required fields.",
  "missing_fields": [
    "total_lifetime_spend_kes"
  ]
}
```

The endpoint correctly rejected the incomplete request.

## 8. Conclusion

The e-commerce customer clustering endpoint successfully loads the saved K-Means model, processes new customer information, handles missing average order values, assigns customers to the five meaningful segments, and validates incoming requests.

The endpoint is implemented separately as a Flask Blueprint for integration into the combined API by Member 5.

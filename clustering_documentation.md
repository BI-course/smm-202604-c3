# Member 3 – K-Means Clustering Documentation

## 1. Overview
The K-Means model assigns new mall customers to existing customer clusters based on their age, annual income and spending score.

## 2. Saved Model
The saved model is located at:

model/kmeans_mall_customer_segmentation.joblib

The artifact contains:
- A StandardScaler
- A trained K-Means model

The model uses five clusters.

## 3. Required Input Features
The model requires three numerical features:
1. Age
2. Annual Income (k$)
3. Spending Score (1-100)

The saved scaler standardizes the input before the clustering prediction.

## 4. API Endpoint
HTTP Method: POST

Endpoint:
/api/v1/models/mall-customer-segmenter/predictions

## 5. Sample Input

{
  "Age": 27,
  "Annual Income (k$)": 45,
  "Spending Score (1-100)": 72
}

## 6. Testing and Results
The endpoint was tested locally using cURL.

The API returned:

{
  "assigned_cluster": 4,
  "model": "K-Means (k=5, standardized features)"
}

## 7. Interpretation
The sample customer was assigned to Cluster 4.

Cluster 4 is an identifier for one of the five groups. Further cluster profiling is needed to determine the specific characteristics of this group.

## 8. Conclusion
The saved K-Means model and scaler loaded successfully, and the prediction endpoint returned a cluster assignment for the sample customer.

#!/bin/bash
# Usage:
#   ./test_endpoints.sh                                   # local Flask  (http://127.0.0.1:5000)
#   ./test_endpoints.sh https://127.0.0.1                 # Docker + Nginx (self-signed cert accepted)
#   ./test_endpoints.sh https://smm-202604-c3.onrender.com   # Render
#
# Sends the same real-value requests to whichever address you give it.

BASE="${1:-http://127.0.0.1:5000}"
BASE="${BASE%/}"

OPTS="-sS --max-time 120"   # 120s covers a Render cold start
case "$BASE" in
  https://127.0.0.1*|https://localhost*) OPTS="$OPTS -k" ;;  # self-signed cert
esac

call() {
  local title="$1" path="$2" body="$3"
  echo "=============================================================="
  echo "$title"
  echo "POST $BASE/api/v1/models/$path/predictions"
  echo "--------------------------------------------------------------"
  curl $OPTS -w "\nHTTP %{http_code}\n" -X POST \
    "$BASE/api/v1/models/$path/predictions" \
    -H "Content-Type: application/json" \
    -d "$body"
  echo
}

echo "Testing: $BASE"
echo

# 1. REGRESSION -- large store
call "1a. Regression: Supermarket in a Mall" "retail-daily-sales-regressor" '{
  "store_size_sqm": 450, "footfall_count": 1200, "promotions_active": 3,
  "avg_basket_size_kes": 1800, "store_age_years": 6, "online_order_count": 85,
  "store_format": "Supermarket", "location_type": "Mall",
  "delivery_available": "Yes", "store_size_category": "Medium"}'

# 1b. REGRESSION -- small store (should predict much lower sales)
call "1b. Regression: small Kiosk on a Street" "retail-daily-sales-regressor" '{
  "store_size_sqm": 20, "footfall_count": 150, "promotions_active": 0,
  "avg_basket_size_kes": 300, "store_age_years": 2, "online_order_count": 0,
  "store_format": "Kiosk", "location_type": "Street",
  "delivery_available": "No", "store_size_category": "Small"}'

# 2a. CLASSIFICATION -- risky company
call "2a. Classification: risky company" "corporate-financial-distress-classifier" '{
  "industry_sector": "Manufacturing", "listing_status": "Public",
  "credit_rating_band": "Speculative", "audit_going_concern_flag": "No",
  "years_in_operation": 15, "num_employees": 500,
  "debt_to_equity_ratio": 1.8, "current_ratio": 1.1,
  "interest_coverage_ratio": 2.5, "revenue_growth_pct": -3.2,
  "net_profit_margin_pct": 2.1, "working_capital_to_assets_ratio": 0.08,
  "macroeconomic_interest_rate_pct": 12.5}'

# 2b. CLASSIFICATION -- strong company
call "2b. Classification: strong company" "corporate-financial-distress-classifier" '{
  "industry_sector": "Technology", "listing_status": "Public",
  "credit_rating_band": "Investment Grade", "audit_going_concern_flag": "No",
  "years_in_operation": 25, "num_employees": 1200,
  "debt_to_equity_ratio": 0.4, "current_ratio": 2.5,
  "interest_coverage_ratio": 9.0, "revenue_growth_pct": 12.0,
  "net_profit_margin_pct": 14.0, "working_capital_to_assets_ratio": 0.3,
  "macroeconomic_interest_rate_pct": 8.0}'

# 3. CLUSTERING (avg_order_value_kes null -> model imputes it)
call "3. Clustering: customer segment" "ecommerce-customer-segmenter" '{
  "recency_days": 30, "frequency_per_year": 8, "num_categories_purchased": 4,
  "total_lifetime_spend_kes": 85000, "avg_order_value_kes": null}'

# 4. RECOMMENDER
call "4a. Recommender: Milk + Bread" "grocery-recommender" '{
  "cart": ["Milk 1L", "White Bread"], "top_n": 3}'

call "4b. Recommender: Onions + Rice" "grocery-recommender" '{
  "cart": ["Onions 1kg", "Rice 2kg"], "top_n": 3}'

# 5. ERROR HANDLING -- empty body should give HTTP 400, not 500
call "5. Error handling: empty body to clustering (expect HTTP 400)" "ecommerce-customer-segmenter" '{}'

echo "Done."

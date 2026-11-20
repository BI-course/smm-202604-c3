import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the saved Apriori artifact dictionary
ARTIFACT_PATH = "apriori_recommendation_rules.joblib"
artifact = joblib.load(ARTIFACT_PATH)

rules_df = artifact["cleaned_rules"]
sku_lookup = artifact["sku_lookup"]

# Reverse lookup (lowercase product_name -> sku) so inputs can be SKUs or product names
name_to_sku = {
    row["product_name"].strip().lower(): sku
    for sku, row in sku_lookup.iterrows()
}


def normalize_basket_to_skus(basket):
    """Converts a list of SKUs or product names into a set of valid SKU codes."""
    sku_set = set()
    for item in basket:
        clean_item = str(item).strip()
        upper_item = clean_item.upper()
        lower_item = clean_item.lower()

        if upper_item in sku_lookup.index:
            sku_set.add(upper_item)
        elif lower_item in name_to_sku:
            sku_set.add(name_to_sku[lower_item])
    return sku_set


def enrich_sku_details(sku):
    """Fetches product metadata (name, category, KES price) for a given SKU."""
    if sku in sku_lookup.index:
        row = sku_lookup.loc[sku]
        return {
            "sku": sku,
            "product_name": row["product_name"],
            "category": row["category"],
            "unit_price_kes": float(row["unit_price_kes"]),
        }
    return {"sku": sku, "product_name": "Unknown", "category": "Unknown", "unit_price_kes": 0.0}


def get_recommendations(basket, top_n=5):
    basket_skus = normalize_basket_to_skus(basket)
    if not basket_skus:
        return []

    # Filter rules where all antecedent SKUs are inside the user's basket
    matching_rules = rules_df[
        rules_df["antecedents"].apply(lambda ant: ant.issubset(basket_skus))
    ]

    if matching_rules.empty:
        return []

    # Rank strongest rules by lift, then confidence
    matching_rules = matching_rules.sort_values(
        by=["lift", "confidence"], ascending=False
    )

    recommendations = []
    seen_skus = set()

    for _, row in matching_rules.iterrows():
        for rec_sku in row["consequents"]:
            if rec_sku not in basket_skus and rec_sku not in seen_skus:
                seen_skus.add(rec_sku)
                details = enrich_sku_details(rec_sku)
                recommendations.append({
                    **details,
                    "based_on_skus": sorted(list(row["antecedents"])),
                    "confidence": round(float(row["confidence"]), 4),
                    "lift": round(float(row["lift"]), 4),
                    "support": round(float(row["support"]), 4),
                })
                if len(recommendations) >= top_n:
                    return recommendations

    return recommendations


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True)

    if not data or "basket" not in data:
        return jsonify({
            "error": "Invalid request. Provide a JSON body with a 'basket' list."
        }), 400

    basket = data["basket"]
    if not isinstance(basket, list) or len(basket) == 0:
        return jsonify({
            "error": "'basket' must be a non-empty list of SKUs or product names."
        }), 400

    top_n = data.get("top_n", 5)

    try:
        recs = get_recommendations(basket, top_n=int(top_n))
        return jsonify({
            "input_basket": basket,
            "matched_skus": sorted(list(normalize_basket_to_skus(basket))),
            "recommendations": recs,
            "count": len(recs),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)

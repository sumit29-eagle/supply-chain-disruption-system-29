
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

DELIVERY_FEATURES = [
    "shipping_mode",
    "market",
    "order_region",
    "customer_segment",
    "department_name",
    "category_name",
    "order_item_quantity",
    "order_item_product_price",
    "order_item_discount_rate",
    "order_item_profit_ratio",
    "sales",
]

DEMAND_FEATURES = [
    "store",
    "item",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "day_of_week",
    "month",
    "day_of_month",
]


def load_model(filename):
    path = MODEL_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)


def require_fields(payload, fields):
    missing = [
        field for field in fields
        if field not in payload or payload[field] in (None, "")
    ]

    if missing:
        raise ValueError(
            "Missing required fields: " + ", ".join(missing)
        )


def predict_delivery(payload):
    require_fields(payload, DELIVERY_FEATURES)

    numeric_fields = [
        "order_item_quantity",
        "order_item_product_price",
        "order_item_discount_rate",
        "order_item_profit_ratio",
        "sales",
    ]

    row = dict(payload)

    try:
        for field in numeric_fields:
            row[field] = float(row[field])
    except (TypeError, ValueError):
        raise ValueError("Delivery numeric fields must be valid numbers.")

    if any(row[field] < 0 for field in [
        "order_item_quantity",
        "order_item_product_price",
        "sales",
    ]):
        raise ValueError("Quantity, price and sales cannot be negative.")

    model = load_model("delivery_model.joblib")
    prediction = model.predict(pd.DataFrame([row])[DELIVERY_FEATURES])[0]

    result = {
        "model": "Delivery Delay Classifier",
        "target": "late_delivery_risk",
        "predicted_class": prediction.item()
        if hasattr(prediction, "item") else prediction,
        "interpretation": (
            "Predicted target class from the trained dataset. "
            "This is not a delivery date or a calibrated probability."
        ),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            pd.DataFrame([row])[DELIVERY_FEATURES]
        )[0]
        result["class_probabilities"] = {
            str(label): round(float(probability), 4)
            for label, probability in zip(
                model.classes_, probabilities
            )
        }

    return result


def predict_demand(payload):
    require_fields(payload, DEMAND_FEATURES)

    row = dict(payload)

    try:
        for field in DEMAND_FEATURES:
            row[field] = float(row[field])
    except (TypeError, ValueError):
        raise ValueError("All demand features must be valid numbers.")

    if any(row[field] < 0 for field in [
        "store", "item", "lag_1", "lag_7",
        "lag_14", "rolling_mean_7",
    ]):
        raise ValueError("Store, item and sales-history values cannot be negative.")

    if not 0 <= row["day_of_week"] <= 6:
        raise ValueError("day_of_week must be between 0 and 6.")
    if not 1 <= row["month"] <= 12:
        raise ValueError("month must be between 1 and 12.")
    if not 1 <= row["day_of_month"] <= 31:
        raise ValueError("day_of_month must be between 1 and 31.")

    model = load_model("demand_model.joblib")
    prediction = model.predict(
        pd.DataFrame([row])[DEMAND_FEATURES]
    )[0]

    return {
        "model": "Daily Sales Forecast",
        "predicted_sales": round(float(prediction), 3),
        "unit": "sales units, according to the training dataset",
        "interpretation": (
            "Forecast based on supplied store, item and sales history. "
            "Accuracy depends on the input data and training distribution."
        ),
    }


from ml_predict import predict_delivery, predict_demand

delivery_input = {
    "shipping_mode": "Standard Class",
    "market": "US",
    "order_region": "West",
    "customer_segment": "Consumer",
    "department_name": "Technology",
    "category_name": "Electronics",
    "order_item_quantity": 2,
    "order_item_product_price": 100.0,
    "order_item_discount_rate": 0.05,
    "order_item_profit_ratio": 0.10,
    "sales": 200.0,
}

demand_input = {
    "store": 1,
    "item": 1,
    "lag_1": 20,
    "lag_7": 18,
    "lag_14": 22,
    "rolling_mean_7": 19.5,
    "day_of_week": 2,
    "month": 10,
    "day_of_month": 21,
}

for name, function, payload in [
    ("DELIVERY", predict_delivery, delivery_input),
    ("DEMAND", predict_demand, demand_input),
]:
    print(f"\n--- {name} MODEL ---")
    try:
        print(function(payload))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")


from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "DataCo.csv" / "DataCoSupplyChainDataset.csv"


def get_inventory_analysis(top_n=10):
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(
        DATA_FILE,
        encoding="latin1",
        usecols=["Product Name", "Order Item Quantity", "Order Status"]
    )

    # Exclude cancelled and suspected-fraud orders from this demand signal.
    excluded = {"CANCELED", "SUSPECTED_FRAUD"}
    df = df[~df["Order Status"].isin(excluded)].copy()

    df["Order Item Quantity"] = pd.to_numeric(
        df["Order Item Quantity"], errors="coerce"
    )
    df = df.dropna(subset=["Product Name", "Order Item Quantity"])

    result = (
        df.groupby("Product Name")["Order Item Quantity"]
        .agg(order_lines="count", total_quantity="sum", avg_quantity="mean")
        .sort_values("total_quantity", ascending=False)
        .head(top_n)
        .reset_index()
    )

    return {
        "signal_type": "historical_product_demand",
        "note": (
            "This is historical order-demand analysis, not a stockout prediction. "
            "Current stock and reorder-level data are not available in this dataset."
        ),
        "products": result.round(2).to_dict(orient="records"),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(get_inventory_analysis(), indent=2))

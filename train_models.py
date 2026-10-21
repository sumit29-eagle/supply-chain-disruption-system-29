
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

# Limits training time and memory use on ordinary laptops.
DELIVERY_MAX_ROWS = 100_000
DEMAND_MAX_ROWS = 250_000
DEMAND_TRAIN_MAX_ROWS = 60_000
RANDOM_STATE = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def clean_name(name):
    return (
        str(name).strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def read_csv_safe(path, nrows=None):
    # DataCo commonly uses a non-UTF-8-compatible encoding.
    last_error = None
    for encoding in ("latin1", "utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
                nrows=nrows,
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc

    raise ValueError(
        f"Could not read CSV: {path}\nReason: {last_error}"
    )


def find_csv(required_columns):
    """Find a CSV in data/ whose normalized header has all required columns."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data folder not found: {DATA_DIR}\n"
            "Create a data folder beside train_models.py."
        )

    for path in DATA_DIR.rglob("*.csv"):
        try:
            header = read_csv_safe(path, nrows=0)
            columns = {clean_name(c) for c in header.columns}
            if required_columns.issubset(columns):
                print(f"Found dataset: {path.relative_to(ROOT)}")
                return path
        except Exception as exc:
            print(f"Skipping unreadable CSV {path.name}: {exc}")

    raise FileNotFoundError(
        f"No CSV in {DATA_DIR} contains these columns: "
        f"{sorted(required_columns)}"
    )


def normalized_frame(df):
    df = df.copy()
    df.columns = [clean_name(c) for c in df.columns]
    return df


def make_pipeline(X, estimator):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [
        c for c in X.columns if c not in numeric_cols
    ]

    transformers = []

    if numeric_cols:
        transformers.append((
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_cols,
        ))

    if categorical_cols:
        categorical_pipe = Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])
        transformers.append((
            "categorical",
            categorical_pipe,
            categorical_cols,
        ))

    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline([
        ("preprocess", preprocess),
        ("model", estimator),
    ])


def train_delivery():
    print("\n[1/2] Preparing delivery-delay model...")

    path = find_csv({"late_delivery_risk"})
    df = normalized_frame(
        read_csv_safe(path, nrows=DELIVERY_MAX_ROWS)
    )

    target = "late_delivery_risk"

    # Use information available before delivery, not the actual outcome.
    candidates = [
        "shipping_mode",
        "days_for_shipment_scheduled",
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
        "order_year",
        "order_month",
        "order_day_of_week",
    ]

    features = [c for c in candidates if c in df.columns]
    if len(features) < 3:
        raise ValueError(
            "Not enough expected delivery features. "
            f"Columns found: {df.columns.tolist()}"
        )

    df = df[features + [target]].copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])

    # Keep valid binary labels only.
    df = df[df[target].isin([0, 1])]
    if len(df) < 100:
        raise ValueError("Too few valid delivery labels to train.")

    X = df[features]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_pipeline(
        X_train,
        RandomForestClassifier(
            n_estimators=50,
            max_depth=14,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    )

    print(f"Training rows: {len(X_train):,}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "task": "delivery_delay_classification",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(
            precision_score(
                y_test, predictions,
                average="weighted", zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_test, predictions,
                average="weighted", zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_test, predictions,
                average="weighted", zero_division=0,
            )
        ),
        "features": features,
    }

    joblib.dump(model, MODEL_DIR / "delivery_model.joblib")
    (MODEL_DIR / "delivery_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("Delivery model saved.")
    print(json.dumps(metrics, indent=2))


def train_demand():
    print("\n[2/2] Preparing demand-forecast model...")

    path = find_csv({"date", "store", "item", "sales"})
    df = normalized_frame(
        read_csv_safe(path, nrows=DEMAND_MAX_ROWS)
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "store", "item", "sales"])
    df = df.sort_values(["store", "item", "date"]).copy()

    group = df.groupby(["store", "item"], sort=False)["sales"]

    # Only use previous observations to predict current-day sales.
    df["lag_1"] = group.shift(1)
    df["lag_7"] = group.shift(7)
    df["lag_14"] = group.shift(14)

    group = df.groupby(["store", "item"], sort=False)["sales"]
    df["rolling_mean_7"] = group.transform(
        lambda s: s.shift(1).rolling(7, min_periods=7).mean()
    )

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day

    features = [
        "store", "item", "lag_1", "lag_7", "lag_14",
        "rolling_mean_7", "day_of_week", "month", "day_of_month",
    ]

    df = df.dropna(subset=features + ["sales"])
    dates = sorted(df["date"].unique())

    if len(dates) < 10 or len(df) < 100:
        raise ValueError(
            "Not enough historical demand data after creating lag features."
        )

    split_date = dates[int(len(dates) * 0.80)]
    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()

    if train.empty or test.empty:
        raise ValueError("Could not create chronological train/test split.")

    # Limit model fitting time; retain the later test period for evaluation.
    if len(train) > DEMAND_TRAIN_MAX_ROWS:
        train = train.sample(
            n=DEMAND_TRAIN_MAX_ROWS,
            random_state=RANDOM_STATE,
        )

    X_train = train[features]
    y_train = train["sales"]
    X_test = test[features]
    y_test = test["sales"]

    model = make_pipeline(
        X_train,
        RandomForestRegressor(
            n_estimators=40,
            max_depth=14,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "task": "daily_sales_regression",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "train_end_date": str(train["date"].max().date()),
        "test_start_date": str(test["date"].min().date()),
        "features": features,
    }

    joblib.dump(model, MODEL_DIR / "demand_model.joblib")
    (MODEL_DIR / "demand_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("Demand model saved.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    print("Supply Chain ML training started.")
    print(f"Project: {ROOT}")
    print(f"Data folder: {DATA_DIR}")
    print(f"Models folder: {MODEL_DIR}")

    failures = []

    for name, function in [
        ("Delivery", train_delivery),
        ("Demand", train_demand),
    ]:
        try:
            function()
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"\n{name} training FAILED: {exc}")

    print("\n========== TRAINING SUMMARY ==========")

    for name, filename in [
        ("Delivery", "delivery_model.joblib"),
        ("Demand", "demand_model.joblib"),
    ]:
        status = "SAVED" if (MODEL_DIR / filename).exists() else "NOT SAVED"
        print(f"{name}: {status}")

    if failures:
        print("\nFix the following issues:")
        for name, error in failures:
            print(f"- {name}: {error}")
    else:
        print("\nBoth models trained and saved successfully.")

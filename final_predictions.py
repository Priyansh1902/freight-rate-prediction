
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

BASE = Path(__file__).resolve().parent

TRAIN_FILE = BASE / "train-test.csv"
VALIDATION_FILE = BASE / "validation.csv"
TEMPLATE_FILE = BASE / "validation-predictions-template.csv"
DECEMBER_FILE = BASE / "december-chart-inputs.csv"

VALIDATION_OUTPUT = BASE / "validation_predictions.csv"
DECEMBER_OUTPUT = BASE / "december_predictions.csv"

CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]

NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "month",
    "day_of_week",
    "day",
    "day_of_year",
    "latitude_difference",
    "longitude_difference",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def prepare_features(data):
    """Build the same feature columns for training and prediction."""
    X = data.copy()

    # Some prediction inputs legitimately omit certain model features.
    # CatBoost can handle missing numerical values.
    required_raw = [
        "pickup", "delivery", "equipment", "pickup_lat", "pickup_lon",
        "delivery_lat", "delivery_lon", "distance", "weight",
        "date", "market_index", "quote_signal",
    ]

    for column in required_raw:
        if column not in X.columns:
            X[column] = np.nan

    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].fillna("Missing").astype(str)

    X["date"] = pd.to_datetime(X["date"], errors="coerce")
    X["month"] = X["date"].dt.month
    X["day_of_week"] = X["date"].dt.dayofweek
    X["day"] = X["date"].dt.day
    X["day_of_year"] = X["date"].dt.dayofyear

    X["latitude_difference"] = X["delivery_lat"] - X["pickup_lat"]
    X["longitude_difference"] = X["delivery_lon"] - X["pickup_lon"]

    # Ensure numerical features really are numeric.
    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    return X[FEATURES]


def main():
    for file_path in [
        TRAIN_FILE, VALIDATION_FILE, TEMPLATE_FILE, DECEMBER_FILE
    ]:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)
    template = pd.read_csv(TEMPLATE_FILE)
    december = pd.read_csv(DECEMBER_FILE)

    if "posted_rate" not in train.columns:
        raise ValueError("Training data is missing posted_rate.")
    if train["posted_rate"].isna().any():
        train = train.dropna(subset=["posted_rate"]).copy()

    X_train = prepare_features(train)
    y_train = pd.to_numeric(train["posted_rate"], errors="raise")

    model = CatBoostRegressor(
        iterations=273,
        depth=8,
        learning_rate=0.05,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
    )

    # Predict validation rows, then restore the exact template ID order.
    if "load_id" not in validation.columns or "load_id" not in template.columns:
        raise ValueError("Validation and template must both contain load_id.")

    if validation["load_id"].duplicated().any():
        raise ValueError("Duplicate load_id values found in validation data.")
    if template["load_id"].duplicated().any():
        raise ValueError("Duplicate load_id values found in prediction template.")

    if set(validation["load_id"]) != set(template["load_id"]):
        raise ValueError("Validation IDs do not match template IDs exactly.")

    validation_predictions = model.predict(prepare_features(validation))
    prediction_lookup = dict(
        zip(validation["load_id"], np.maximum(validation_predictions, 1.0))
    )

    output = template[["load_id"]].copy()
    output["predicted_rate"] = output["load_id"].map(prediction_lookup)

    if output["predicted_rate"].isna().any():
        raise ValueError("Some validation IDs did not receive predictions.")

    output.to_csv(VALIDATION_OUTPUT, index=False)

    # December data lacks coordinates and market signals. Those features
    # remain missing rather than being fabricated.
    december_predictions = model.predict(prepare_features(december))

    december_output = december.copy()
    december_output["predicted_rate"] = np.maximum(december_predictions, 1.0)

    required_december_columns = [
        "pickup", "delivery", "distance", "equipment",
        "weight", "date", "predicted_rate",
    ]
    december_output = december_output[required_december_columns]
    december_output.to_csv(DECEMBER_OUTPUT, index=False)

    print("\nPrediction files created:")
    print(f"  Validation: {VALIDATION_OUTPUT}")
    print(f"  December:   {DECEMBER_OUTPUT}")
    print(f"\nValidation rows: {len(output):,}")
    print(f"December rows:   {len(december_output):,}")
    print(f"December prediction range: "
          f"${december_output['predicted_rate'].min():,.2f}–"
          f"${december_output['predicted_rate'].max():,.2f}")


if __name__ == "__main__":
    main()


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

# Use the folder containing this script, regardless of terminal location.
DATA_DIR = Path(__file__).resolve().parent

train = pd.read_csv(DATA_DIR / "train-test.csv")
train["date"] = pd.to_datetime(train["date"], errors="coerce")

# Hold out September for a chronological validation.
fit_data = train[train["date"] < "2025-09-01"].copy()
valid_data = train[
    (train["date"] >= "2025-09-01") &
    (train["date"] < "2025-10-01")
].copy()

if fit_data.empty or valid_data.empty:
    raise ValueError("Training or September validation data is empty.")

TARGET = "posted_rate"

def make_features(df):
    X = df.drop(columns=[TARGET, "load_id"], errors="ignore").copy()

    dates = pd.to_datetime(X.pop("date"), errors="coerce")
    X["month"] = dates.dt.month
    X["day_of_week"] = dates.dt.dayofweek
    X["day_of_month"] = dates.dt.day
    X["day_of_year"] = dates.dt.dayofyear

    # Approximate geographic differences can help distinguish routes.
    X["lat_difference"] = (
        X["delivery_lat"] - X["pickup_lat"]
    )
    X["lon_difference"] = (
        X["delivery_lon"] - X["pickup_lon"]
    )

    categorical = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    for col in categorical:
        X[col] = X[col].fillna("Missing").astype(str)

    return X, categorical

X_train, cat_features = make_features(fit_data)
X_valid, _ = make_features(valid_data)

# Ensure both datasets have the same feature columns and order.
X_valid = X_valid.reindex(columns=X_train.columns)

y_train = fit_data[TARGET]
y_valid = valid_data[TARGET]

print(f"Training rows: {len(X_train):,}")
print(f"September validation rows: {len(X_valid):,}")
print(f"Features: {len(X_train.columns)}")
print("Training date range:",
      fit_data["date"].min().date(), "to",
      fit_data["date"].max().date())
print("Validation date range:",
      valid_data["date"].min().date(), "to",
      valid_data["date"].max().date())

model = CatBoostRegressor(
    iterations=1500,
    depth=8,
    learning_rate=0.05,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=42,
    verbose=200,
    allow_writing_files=False
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=100,
    verbose=200
)

predictions = model.predict(X_valid)
predictions = np.maximum(predictions, 1.0)

mae = mean_absolute_error(y_valid, predictions)
rmse = np.sqrt(mean_squared_error(y_valid, predictions))
wape = np.abs(y_valid.to_numpy() - predictions).sum() / np.abs(
    y_valid.to_numpy()
).sum()
r2 = r2_score(y_valid, predictions)

print("\n--- September validation results ---")
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"WAPE: {wape:.2%}")
print(f"R²:    {r2:.4f}")

# Save feature importance for interpretation.
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(DATA_DIR / "feature_importance.csv", index=False)
print("\nTop 15 features:")
print(importance.head(15).to_string(index=False))

print("\nSaved feature importance to feature_importance.csv")

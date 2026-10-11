
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

DATA_DIR = Path(__file__).resolve().parent
train = pd.read_csv(DATA_DIR / "train-test.csv")
train["date"] = pd.to_datetime(train["date"], errors="coerce")

fit_data = train[train["date"] < "2025-10-01"].copy()
valid_data = train[
    (train["date"] >= "2025-10-01") &
    (train["date"] < "2025-11-01")
].copy()

def make_features(df):
    X = df.drop(columns=["posted_rate", "load_id"], errors="ignore").copy()
    dates = pd.to_datetime(X.pop("date"), errors="coerce")

    X["month"] = dates.dt.month
    X["day_of_week"] = dates.dt.dayofweek
    X["day_of_month"] = dates.dt.day
    X["day_of_year"] = dates.dt.dayofyear
    X["lat_difference"] = X["delivery_lat"] - X["pickup_lat"]
    X["lon_difference"] = X["delivery_lon"] - X["pickup_lon"]

    cats = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    for col in cats:
        X[col] = X[col].fillna("Missing").astype(str)

    return X, cats

X_train, cat_features = make_features(fit_data)
X_valid, _ = make_features(valid_data)
X_valid = X_valid.reindex(columns=X_train.columns)

y_train = fit_data["posted_rate"]
y_valid = valid_data["posted_rate"]

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
    early_stopping_rounds=100
)

pred = np.maximum(model.predict(X_valid), 1.0)

mae = mean_absolute_error(y_valid, pred)
rmse = np.sqrt(mean_squared_error(y_valid, pred))
wape = np.abs(y_valid.to_numpy() - pred).sum() / np.abs(y_valid.to_numpy()).sum()
r2 = r2_score(y_valid, pred)

print("\n--- October validation results ---")
print(f"Training rows: {len(X_train):,}")
print(f"October rows:  {len(X_valid):,}")
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"WAPE: {wape:.2%}")
print(f"R²:    {r2:.4f}")

errors = valid_data[["load_id", "pickup", "delivery", "distance", "posted_rate"]].copy()
errors["predicted_rate"] = pred
errors["absolute_error"] = np.abs(errors["posted_rate"] - pred)
errors.sort_values("absolute_error", ascending=False).head(100).to_csv(
    DATA_DIR / "october_largest_errors.csv", index=False
)
print("Saved largest October errors to october_largest_errors.csv")

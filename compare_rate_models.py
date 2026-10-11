
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).resolve().parent

df = pd.read_csv(DATA_DIR / "train-test.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Train through September; evaluate on October.
train = df[df["date"] < "2025-10-01"].copy()
valid = df[
    (df["date"] >= "2025-10-01") &
    (df["date"] < "2025-11-01")
].copy()

def make_features(data):
    X = data.drop(columns=["posted_rate", "load_id"], errors="ignore").copy()
    dates = pd.to_datetime(X.pop("date"), errors="coerce")

    X["month"] = dates.dt.month
    X["day_of_week"] = dates.dt.dayofweek
    X["day_of_month"] = dates.dt.day
    X["day_of_year"] = dates.dt.dayofyear
    X["lat_difference"] = X["delivery_lat"] - X["pickup_lat"]
    X["lon_difference"] = X["delivery_lon"] - X["pickup_lon"]

    cat_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    for col in cat_features:
        X[col] = X[col].fillna("Missing").astype(str)

    return X, cat_features

X_train, cat_features = make_features(train)
X_valid, _ = make_features(valid)
X_valid = X_valid.reindex(columns=X_train.columns)

y_train = train["posted_rate"].to_numpy()
y_valid = valid["posted_rate"].to_numpy()
distance_valid = valid["distance"].to_numpy()

if np.any(train["distance"] <= 0) or np.any(valid["distance"] <= 0):
    raise ValueError("Non-positive distance found; inspect before training.")

def fit_and_evaluate(name, train_target, valid_target, multiply_by_distance=False):
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
        train_target,
        cat_features=cat_features,
        eval_set=(X_valid, valid_target),
        early_stopping_rounds=100
    )

    pred = model.predict(X_valid)

    if multiply_by_distance:
        pred = pred * distance_valid

    pred = np.maximum(pred, 1.0)

    mae = mean_absolute_error(y_valid, pred)
    rmse = np.sqrt(mean_squared_error(y_valid, pred))
    wape = np.abs(y_valid - pred).sum() / np.abs(y_valid).sum()
    r2 = r2_score(y_valid, pred)

    print(f"\n--- {name} ---")
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"WAPE: {wape:.2%}")
    print(f"R²:    {r2:.4f}")

    return pred, {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "R2": r2
    }

print(f"Training rows: {len(train):,}")
print(f"October rows: {len(valid):,}")

# Model A: predict the total rate directly.
direct_pred, direct_metrics = fit_and_evaluate(
    "Direct-rate model",
    y_train,
    y_valid
)

# Model B: predict rate per mile, then convert to total rate.
rpm_train = y_train / train["distance"].to_numpy()
rpm_valid = y_valid / distance_valid

rpm_pred, rpm_metrics = fit_and_evaluate(
    "Rate-per-mile model",
    rpm_train,
    rpm_valid,
    multiply_by_distance=True
)

comparison = pd.DataFrame([direct_metrics, rpm_metrics])
comparison.to_csv(DATA_DIR / "model_comparison.csv", index=False)

errors = valid[[
    "load_id", "pickup", "delivery", "distance",
    "equipment", "posted_rate"
]].copy()
errors["direct_prediction"] = direct_pred
errors["rpm_prediction"] = rpm_pred
errors["direct_absolute_error"] = np.abs(y_valid - direct_pred)
errors["rpm_absolute_error"] = np.abs(y_valid - rpm_pred)
errors.sort_values("direct_absolute_error", ascending=False).to_csv(
    DATA_DIR / "october_model_predictions.csv", index=False
)

print("\n--- Side-by-side comparison ---")
print(comparison.to_string(index=False, formatters={
    "MAE": "${:,.2f}".format,
    "RMSE": "${:,.2f}".format,
    "WAPE": "{:.2%}".format,
    "R2": "{:.4f}".format
}))
print("\nSaved model_comparison.csv and october_model_predictions.csv")

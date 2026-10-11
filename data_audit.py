
from pathlib import Path

import pandas as pd

DATA_DIR = Path(".")

train = pd.read_csv(DATA_DIR / "train-test.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
template = pd.read_csv(DATA_DIR / "validation-predictions-template.csv")
december = pd.read_csv(DATA_DIR / "december-chart-inputs.csv")

print("\n--- DATASET SHAPES ---")
print("Train:", train.shape)
print("Validation:", validation.shape)
print("Template:", template.shape)
print("December:", december.shape)

print("\n--- TRAINING COLUMNS ---")
print(train.columns.tolist())

print("\n--- DATA TYPES ---")
print(train.dtypes)

print("\n--- MISSING VALUES ---")
print(train.isnull().sum())
print("\nValidation missing values:")
print(validation.isnull().sum())

print("\n--- TARGET SUMMARY ---")
print(train["posted_rate"].describe())

print("\n--- DATE RANGES ---")
print("Train:", train["date"].min(), "to", train["date"].max())
print(
    "Validation:",
    validation["date"].min(),
    "to",
    validation["date"].max(),
)

print("\n--- ID VALIDATION ---")
print("Training IDs unique:", train["load_id"].is_unique)
print("Validation IDs unique:", validation["load_id"].is_unique)
print("Template IDs match validation:",
      template["load_id"].equals(validation["load_id"]))

print("\n--- INVALID VALUES ---")
print("Non-positive targets:", (train["posted_rate"] <= 0).sum())
print("Negative weights:", (train["weight"] < 0).sum())
print("Zero distances:", (train["distance"] <= 0).sum())

print("\n--- CATEGORICAL COVERAGE ---")
for col in ["pickup", "delivery", "equipment"]:
    train_values = set(train[col].dropna().unique())
    valid_values = set(validation[col].dropna().unique())
    print(
        f"{col}: unseen validation categories =",
        sorted(valid_values - train_values),
    )

print("\n--- AUDIT COMPLETE ---")

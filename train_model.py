
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data.csv")

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])
df = df[np.isfinite(df["price"])]

# Target and input features
excluded = ["price"]
for col in ["date", "street"]:
    if col in df.columns:
        excluded.append(col)

X = df.drop(columns=excluded)
y = df["price"]

numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical = [c for c in X.columns if c not in numeric]

transformers = []
if numeric:
    transformers.append((
        "numbers",
        Pipeline([("fill", SimpleImputer(strategy="median"))]),
        numeric
    ))
if categorical:
    transformers.append((
        "categories",
        Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical
    ))

preprocessor = ColumnTransformer(transformers=transformers)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=12, min_samples_leaf=3, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=120, max_depth=18, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
}

results = {}
trained = {}

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)

    results[name] = {
        "MAE": round(float(mean_absolute_error(y_test, prediction)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, prediction))), 2),
        "R2": round(float(r2_score(y_test, prediction)), 4)
    }
    trained[name] = pipeline
    print(name, results[name])

# Select the model with the lowest test-set MAE
best_name = min(results, key=lambda name: results[name]["MAE"])
best_model = trained[best_name]
joblib.dump(best_model, BASE / "house_price_model.joblib")

# Prepare settings for the prediction form
defaults = {}
ranges = {}
integer_features = []

for col in numeric:
    values = pd.to_numeric(X[col], errors="coerce")
    median = values.median()
    if pd.isna(median):
        median = 0
    low, high = values.min(), values.max()
    defaults[col] = float(median)
    ranges[col] = {
        "min": float(low) if pd.notna(low) else float(median - 1),
        "max": float(high) if pd.notna(high) else float(median + 1)
    }
    non_null = values.dropna()
    if len(non_null) and np.all(np.isclose(non_null, np.round(non_null))):
        integer_features.append(col)

options = {}
for col in categorical:
    choices = sorted(X[col].dropna().astype(str).unique().tolist())
    options[col] = choices[:500] or ["Unknown"]

# Feature importance where supported by the selected model
importance = []
estimator = best_model.named_steps["model"]
if hasattr(estimator, "feature_importances_"):
    try:
        names = best_model.named_steps["preprocessor"].get_feature_names_out()
        importance = [
            {"feature": str(n), "importance": float(v)}
            for n, v in zip(names, estimator.feature_importances_)
        ]
        importance.sort(key=lambda item: item["importance"], reverse=True)
        importance = importance[:20]
    except Exception:
        importance = []

config = {
    "feature_columns": X.columns.tolist(),
    "numeric_features": numeric,
    "categorical_features": categorical,
    "numeric_defaults": defaults,
    "numeric_ranges": ranges,
    "integer_features": integer_features,
    "category_options": options,
    "best_model": best_name,
    "metrics": results,
    "feature_importance": importance
}

with open(BASE / "model_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print("\nBest model:", best_name)
print("Model saved successfully!")
print("Now run: python -m streamlit run app.py")

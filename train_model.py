

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# 1. Reproducible synthetic dataset
rng = np.random.default_rng(42)
n = 1500

roles = [
    "Data Scientist",
    "Machine Learning Engineer",
    "Data Analyst",
    "AI Engineer",
    "Software Engineer",
]
locations = ["London", "Bristol", "Manchester", "Leeds"]
education_levels = ["Bachelors", "Masters", "PhD"]

df = pd.DataFrame({
    "job_role": rng.choice(roles, n),
    "location": rng.choice(locations, n),
    "education": rng.choice(education_levels, n),
    "experience_years": rng.integers(0, 16, n),
    "skills_count": rng.integers(2, 13, n),
})

role_salary = {
    "Data Scientist": 40000,
    "Machine Learning Engineer": 48000,
    "Data Analyst": 32000,
    "AI Engineer": 50000,
    "Software Engineer": 42000,
}
location_bonus = {
    "London": 8000,
    "Bristol": 4000,
    "Manchester": 2000,
    "Leeds": 0,
}
education_bonus = {
    "Bachelors": 0,
    "Masters": 4000,
    "PhD": 7000,
}

df["salary_gbp"] = (
    df["job_role"].map(role_salary)
    + df["location"].map(location_bonus)
    + df["education"].map(education_bonus)
    + df["experience_years"] * 1800
    + df["skills_count"] * 500
    + rng.normal(0, 5000, n)
).clip(lower=22000).round(0)


# 2. Features and target
X = df.drop(columns=["salary_gbp"])
y = df["salary_gbp"]

categorical = ["job_role", "location", "education"]

preprocessor = ColumnTransformer([
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical,
    ),
], remainder="passthrough")


# 3. Separate training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Establish a simple baseline
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)


# 5. Build the Random Forest pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=3,
            n_jobs=-1,
        ),
    ),
])


# 6. Cross-validation using training data only
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
print(f"5-fold CV MAE: £{-cv_scores.mean():,.2f}")
print(f"CV MAE standard deviation: £{cv_scores.std():,.2f}")


# 7. Fit once on all training data, then test
model.fit(X_train, y_train)
predictions = model.predict(X_test)


# 8. Report comparable test metrics
for name, actual, predicted in [
    ("Baseline", y_test, baseline_predictions),
    ("Random Forest", y_test, predictions),
]:
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print(f"MAE:  £{mae:,.2f}")
    print(f"RMSE: £{rmse:,.2f}")
    print(f"R²:   {r2:.3f}")


# 9. Save outputs beside this script
project_dir = Path(__file__).resolve().parent
model_dir = project_dir / "models"
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_dir / "salary_model.joblib")
df.to_csv(project_dir / "synthetic_salary_data.csv", index=False)

print("\nModel and synthetic dataset saved.")
print("WARNING: Results use synthetic data, not real UK salary statistics.")

X = df.drop(columns=["salary_gbp"])
y = df["salary_gbp"]

categorical = ["job_role", "location", "education"]
numeric = ["experience_years", "skills_count"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numeric),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        min_samples_leaf=3,
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("MAE:", round(mean_absolute_error(
    y_test, predictions
), 2))
print("R2:", round(r2_score(y_test, predictions), 3))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/salary_model.joblib")
df.to_csv("synthetic_salary_data.csv", index=False)

print("Model saved successfully.")

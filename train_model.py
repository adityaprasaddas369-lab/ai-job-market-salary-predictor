
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

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

import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import joblib

os.makedirs("artifacts", exist_ok=True)
os.makedirs("output", exist_ok=True)

df = pd.read_csv("data/aadhaar_features_ready_for_ML.csv")
pop = pd.read_csv("data/india-districts-census-2011.csv")

df.columns = df.columns.str.lower()
pop.columns = pop.columns.str.lower()

df = df.rename(columns={
    "state": "state_name",
    "district": "district_name",
    "total_enrolment": "enrollment_count",
    "total_bio_updates": "update_count",
    "date": "year_month"
})

pop = pop.rename(columns={
    "state name": "state_name",
    "district name": "district_name",
    "population": "population"
})

pop = pop[["state_name", "district_name", "population"]]

df["year_month"] = df["year_month"].astype(str)

df = df.merge(
    pop,
    on=["state_name", "district_name"],
    how="left"
)

df["population"] = df["population"].fillna(df["population"].median())

df = df.sort_values(["state_name", "district_name", "year_month"])

df["enrollments_per_10k"] = (df["enrollment_count"] / df["population"]) * 10000
df["updates_per_10k"] = (df["update_count"] / df["population"]) * 10000
df["update_to_enrollment_ratio"] = df["update_count"] / (df["enrollment_count"] + 1)

df["enrollment_mom_change"] = df.groupby(
    ["state_name", "district_name"]
)["enrollment_count"].pct_change().fillna(0)

df["update_mom_change"] = df.groupby(
    ["state_name", "district_name"]
)["update_count"].pct_change().fillna(0)

features = [
    "enrollment_count",
    "update_count",
    "enrollments_per_10k",
    "updates_per_10k",
    "update_to_enrollment_ratio",
    "enrollment_mom_change",
    "update_mom_change"
]

X = df[features].replace([np.inf, -np.inf], 0).fillna(0)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

df["anomaly_flag"] = (model.fit_predict(X_scaled) == -1).astype(int)

def reason(row):
    if row["update_to_enrollment_ratio"] > 2:
        return "High biometric re-capture demand"
    if row["update_mom_change"] > 1:
        return "Sudden update surge"
    if row["enrollment_mom_change"] < -0.5:
        return "Enrollment drop"
    return "Normal variance"

df["anomaly_reason"] = df.apply(reason, axis=1)

joblib.dump(model, "artifacts/isolation_forest.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")

with open("artifacts/feature_columns.json", "w") as f:
    json.dump(features, f)

df.to_csv("output/aadhaar_anomaly_results.csv", index=False)

df[df["anomaly_flag"] == 1].sort_values(
    "update_to_enrollment_ratio", ascending=False
).head(10).to_csv(
    "output/top_10_anomalous_districts.csv",
    index=False
)

print("Training complete")

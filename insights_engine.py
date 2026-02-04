import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from prophet import Prophet

AGE_5_17_COL = "bio_age_5_17"
AGE_17_PLUS_COL = "bio_age_17_"

def load_and_prepare_data(files):
    usecols = ["date", "state", "district", AGE_5_17_COL, AGE_17_PLUS_COL]
    df_list = []

    for f in files:
        temp = pd.read_csv(f, usecols=usecols)
        df_list.append(temp)

    df = pd.concat(df_list, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "state", "district"])

    df[[AGE_5_17_COL, AGE_17_PLUS_COL]] = df[
        [AGE_5_17_COL, AGE_17_PLUS_COL]
    ].apply(pd.to_numeric, errors="coerce").fillna(0)

    df["total_updates"] = df[AGE_5_17_COL] + df[AGE_17_PLUS_COL]

    return df

def get_national_state_summary(df):
    return (
        df.groupby("state")[[AGE_5_17_COL, AGE_17_PLUS_COL, "total_updates"]]
        .sum()
        .reset_index()
    )

def get_national_time_series(df):
    ts = (
        df.resample("M", on="date")["total_updates"]
        .sum()
        .reset_index()
    )
    ts.columns = ["ds", "y"]
    return ts

def run_national_prophet_forecast(ts_df, periods=12):
    model = Prophet(yearly_seasonality=True)
    model.fit(ts_df)

    future = model.make_future_dataframe(periods=periods, freq="M")
    forecast = model.predict(future)

    return forecast[["ds", "yhat"]]

def detect_national_state_anomalies(state_df):
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.1,
        random_state=42
    )

    state_df = state_df.copy()
    state_df["anomaly"] = iso.fit_predict(
        state_df[["total_updates"]]
    )

    high = (
        state_df[state_df["anomaly"] == -1]
        .sort_values("total_updates", ascending=False)
        .head(5)
    )

    low = (
        state_df[state_df["anomaly"] == -1]
        .sort_values("total_updates", ascending=True)
        .head(5)
    )

    return high, low

def get_state_district_summary(df, state):
    return (
        df[df["state"] == state]
        .groupby("district")[[AGE_5_17_COL, AGE_17_PLUS_COL, "total_updates"]]
        .sum()
        .reset_index()
    )

def detect_state_district_anomalies(district_df):
    iso = IsolationForest(
        n_estimators=150,
        contamination=0.15,
        random_state=42
    )

    district_df = district_df.copy()
    district_df["anomaly"] = iso.fit_predict(
        district_df[["total_updates"]]
    )

    high = (
        district_df[district_df["anomaly"] == -1]
        .sort_values("total_updates", ascending=False)
        .head(5)
    )

    low = (
        district_df[district_df["anomaly"] == -1]
        .sort_values("total_updates", ascending=True)
        .head(5)
    )

    return high, low

def generate_state_recommendations(state, anomalies, df):
    actions = []

    if anomalies["severity"] == "high":
        actions.append(
            f"{state}: Deploy additional Aadhaar biometric kits and extend operating hours"
        )

    if anomalies["trend"] == "declining":
        actions.append(
            f"{state}: Investigate enrollment friction points and local outreach gaps"
        )

    if not actions:
        actions.append(
            f"{state}: Current biometric update demand is stable — continue monitoring"
        )

    return actions

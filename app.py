import streamlit as st
import matplotlib.pyplot as plt
from insights_engine import *

st.set_page_config(layout="wide")
st.title("UIDAI Biometric Update Demand – Decision Support Dashboard")
st.caption("National outlook, structural imbalances, and operational pressure points")

uploaded_files = st.file_uploader(
    "Upload Aadhaar Biometric CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

df = load_and_prepare_data(uploaded_files)

st.subheader("National-Level Forecast (Forward Outlook)")

ts = get_national_time_series(df)
forecast = run_national_prophet_forecast(ts)

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(ts["ds"], ts["y"], label="Historical")
ax1.plot(forecast["ds"], forecast["yhat"], linestyle="--", label="Forecast")
ax1.legend()
ax1.set_ylabel("Biometric Updates")
st.pyplot(fig1)

st.subheader("National-Level State Anomalies")

state_summary = get_national_state_summary(df)
high_states, low_states = detect_national_state_anomalies(state_summary)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### High Demand Anomalies")
    st.dataframe(high_states)

with col2:
    st.markdown("### Low Demand Anomalies")
    st.dataframe(low_states)

st.subheader("State-Level Deep Dive")

selected_state = st.selectbox(
    "Select State",
    sorted(df["state"].unique())
)

district_summary = get_state_district_summary(df, selected_state)
high_districts, low_districts = detect_state_district_anomalies(district_summary)

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.bar(district_summary["district"], district_summary["total_updates"])
ax2.set_xticklabels(district_summary["district"], rotation=90)
ax2.set_ylabel("Total Updates")
st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    st.markdown("### High Demand Districts")
    st.dataframe(high_districts)

with col4:
    st.markdown("### Low Demand Districts")
    st.dataframe(low_districts)

severity = "high" if len(high_districts) > 0 else "normal"
trend = "declining" if low_districts["total_updates"].mean() < high_districts["total_updates"].mean() else "stable"

anomalies = {
    "severity": severity,
    "trend": trend
}

actions = generate_state_recommendations(
    selected_state,
    anomalies,
    df
)

st.subheader("What UIDAI Should Do Next")

for action in actions:
    st.write("•", action)

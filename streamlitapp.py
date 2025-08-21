# streamlit_app.py
import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="ML Prediction Dashboard", layout="wide")
st.title("🚀 ML Prediction Dashboard")

tab1, tab2 = st.tabs(["📈 Revenue Forecast", "🎯 Lead Conversion"])

with tab1:
    date = st.date_input("Select a future date for forecast")
    if st.button("Get Forecast", key="forecast_btn"):
        formatted_date = date.strftime("%d-%m-%Y")
        response = requests.get(f"{BASE_URL}/forecast/{formatted_date}")
        if response.status_code == 200:
            data = response.json()
            st.success(f"Cumulative Forecast up to {formatted_date}: {data['cumulative_revenue_forecast']}")
        else:
            st.error(response.json()["detail"])

with tab2:
    promo_flag = st.selectbox("Promo Flag", [0, 1])
    marketing_spend = st.number_input("Marketing Spend", min_value=0.0, step=1000.0)
    num_reps = st.number_input("Number of Reps Active", min_value=0)
    call_count = st.number_input("Call Count", min_value=0)
    meeting_scheduled = st.selectbox("Meeting Scheduled", [0, 1])
    year = st.number_input("Year", min_value=2024, max_value=2035, value=2028)
    month = st.number_input("Month", min_value=1, max_value=12, value=7)
    day = st.number_input("Day", min_value=1, max_value=31, value=1)
    sales_channel_online = st.selectbox("Sales Channel: Online", [0, 1])

    if st.button("Predict Lead Conversion", key="lead_btn"):
        payload = {
            "Promo_Flag": promo_flag,
            "Marketing_Spend": marketing_spend,
            "Num_Reps_Active": num_reps,
            "Call_Count": call_count,
            "Meeting_Scheduled": meeting_scheduled,
            "year": year,
            "month": month,
            "day": day,
            "Sales_Channel_Online": sales_channel_online
        }
        response = requests.post(f"{BASE_URL}/lead_conversion/", json=payload)
        if response.status_code == 200:
            data = response.json()
            st.write(f"🔮 Probability of Conversion: {data['conversion_probability']:.2f}")
            st.success(f"Likely to Convert: {data['likely_to_convert']}")
        else:
            st.error(response.json()["detail"])

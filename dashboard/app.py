import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Delivery Delay Predictor",
    page_icon="📦",
    layout="wide"
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "lightgbm_delivery_delay.pkl"
model = joblib.load(MODEL_PATH)

# dashboard title and description

st.title("📦 E-Commerce Delivery Delay Prediction")
col1, col2, col3 = st.columns(3)

col1.metric("Model", "LightGBM")
col2.metric("ROC-AUC", "0.739")
col3.metric("Features", "13")

st.markdown("""
This dashboard predicts the probability of an e-commerce order being delayed.

The prediction system uses:
- LightGBM machine learning model
- FastAPI backend API
- Feature-engineered logistics signals
""")

# input fields

purchase_month = st.number_input(
    "Purchase Month",
    min_value=1,
    max_value=12,
    value=10
)

purchase_weekday = st.number_input(
    "Purchase Weekday (0 = Monday, 6 = Sunday)",
    min_value=0,
    max_value=6,
    value=0
)

purchase_hour = st.number_input(
    "Purchase Hour",
    min_value=0,
    max_value=23,
    value=10
)

is_weekend = st.selectbox(
    "Is Weekend?",
    [0, 1]
)

is_month_end = st.selectbox(
    "Is Month End?",
    [0, 1]
)

promised_delivery_days = st.number_input(
    "Promised Delivery Days",
    min_value=1,
    max_value=200,
    value=15
)

seller_late_rate = st.number_input(
    "Seller Late Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.07
)

total_order_value = st.number_input(
    "Total Order Value",
    min_value=0.0,
    max_value=100000.0,
    value=29.99
)

total_freight_value = st.number_input(
    "Total Freight Value",
    min_value=0.0,
    max_value=10000.0,
    value=8.72
)

avg_product_weight_g = st.number_input(
    "Avg Product Weight (g)",
    min_value=0.0,
    max_value=50000.0,
    value=500.0
)

total_package_volume_cm3 = st.number_input(
    "Total Package Volume (cm³)",
    min_value=0.0,
    max_value=1000000.0,
    value=1976.0
)

avg_freight_price_ratio = st.number_input(
    "Avg Freight Price Ratio",
    min_value=0.0,
    max_value=100.0,
    value=0.29
)

n_items = st.number_input(
    "Number of Items",
    min_value=1,
    max_value=100,
    value=1
)

# payload used for local model prediction

payload = {
    "purchase_month": purchase_month,
    "purchase_weekday": purchase_weekday,
    "purchase_hour": purchase_hour,
    "is_weekend": is_weekend,
    "is_month_end": is_month_end,
    "promised_delivery_days": promised_delivery_days,
    "seller_late_rate": seller_late_rate,
    "total_order_value": total_order_value,
    "total_freight_value": total_freight_value,
    "avg_product_weight_g": avg_product_weight_g,
    "total_package_volume_cm3": total_package_volume_cm3,
    "avg_freight_price_ratio": avg_freight_price_ratio,
    "n_items": n_items
}

# prediction button

# prediction button

if st.button("Predict Delay Risk"):

    try:
        # convert inputs into dataframe
        df = pd.DataFrame([payload])

        # prediction probability
        prob = model.predict_proba(df)[0][1]

        # prediction label
        prediction = (
            "Late Delivery"
            if prob >= 0.5
            else "On-Time Delivery"
        )

        # show prediction

        if prediction == "Late Delivery":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

        # probability score
        st.metric(
            "Delay Probability",
            f"{prob:.2%}"
        )

        # risk interpretation
        if prob >= 0.7:
            st.warning("High delay risk detected")

        elif prob >= 0.4:
            st.info("Moderate delay risk")

        else:
            st.success("Low delay risk")

    except Exception as e:
        st.error(f"Prediction Error: {e}")


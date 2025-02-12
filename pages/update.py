import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model with caching
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

model = load_model()

# Streamlit UI
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🚀 Fraud Detection for Credit Card Transactions")
st.write("Enter transaction details below to predict fraud.")

# User inputs for essential features
st.subheader("Transaction Details")
amount_log = st.number_input("Transaction Amount ", min_value=0.0, format="%f")
hour = st.slider("Hour of Transaction", 0, 23, 12)
dayofweek = st.selectbox("Day of Week", list(range(1, 8)), format_func=lambda x: f"Day {x}")
month = st.selectbox("Month", list(range(1, 13)), format_func=lambda x: f"Month {x}")
category = st.selectbox("Transaction Category", list(range(1, 10)), format_func=lambda x: f"Category {x}")

# Prepare user input
user_input = np.array([[amount_log, category, hour, dayofweek, month]])

# Predict and display results
if st.button("🔍 Predict Fraud"):
    prediction = model.predict(user_input)
    probability = model.predict_proba(user_input)[0][1]  # Probability of fraud
    result = "🚨 Fraud Detected! Take necessary action." if prediction[0] == 1 else "✅ No Fraud Detected. Transaction looks safe."
    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(result)
    else:
        st.success(result)
    
    # Show fraud probability
    st.write(f"**Fraud Probability:** {probability:.2%}")

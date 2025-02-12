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

# Layout using columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Details")
    amount_log = st.number_input("Log Amount", min_value=0.0, format="%f")
    category = st.selectbox("Category", list(range(1, 10)), format_func=lambda x: f"Category {x}")
    hour = st.slider("Hour of Transaction", 0, 23, 12)
    unix_time = st.number_input("Unix Time", min_value=0, format="%d")
    
    st.subheader("Location Details")
    lat = st.number_input("Latitude", format="%f")
    long = st.number_input("Longitude", format="%f")
    merch_lat = st.number_input("Merchant Latitude", format="%f")
    merch_long = st.number_input("Merchant Longitude", format="%f")

with col2:
    st.subheader("Cardholder & Merchant Details")
    merchant = st.number_input("Merchant ID", min_value=0, format="%d")
    dayofweek = st.selectbox("Day of Week", list(range(1, 8)), format_func=lambda x: f"Day {x}")
    month = st.selectbox("Month", list(range(1, 13)), format_func=lambda x: f"Month {x}")
    cc_num = st.number_input("Credit Card Number", min_value=0, format="%d")
    city = st.number_input("City ID", min_value=0, format="%d")
    state = st.number_input("State ID", min_value=0, format="%d")

# Advanced options in expander
with st.expander("🔧 Advanced Options"):
    st.write("These fields are optional and for detailed analysis.")
    extra_feature = st.number_input("Extra Feature", min_value=0, format="%d")

# Prepare user input
user_input = np.array([[amount_log, category, hour, unix_time, merchant, lat, 
                        dayofweek, month, long, cc_num, city, state, merch_long, merch_lat]])

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
    
    # Visualize feature importance (Example Placeholder)
    st.subheader("Feature Contribution")
    features = ["Log Amount", "Category", "Hour", "Unix Time", "Merchant ID", "Latitude", "Day of Week", 
                "Month", "Longitude", "Credit Card Number", "City", "State", "Merchant Longitude", "Merchant Latitude"]
    importance = np.random.rand(len(features))  # Placeholder; replace with actual model feature importance
    df_importance = pd.DataFrame({"Feature": features, "Importance": importance})
    df_importance = df_importance.sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots()
    ax.barh(df_importance["Feature"], df_importance["Importance"], color='skyblue')
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

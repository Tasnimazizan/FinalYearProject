import streamlit as st
import joblib
import numpy as np

# Streamlit UI
st.title("Fraud Detection for Credit Card")
st.write("Enter transaction details to predict fraud.")

# Load the trained model
model = joblib.load("xgb_model.pkl")

# User input fields
def get_user_input():
    amount_log = st.number_input("Log Amount", min_value=0.0, format="%f")
    category = st.number_input("Category", min_value=0, format="%d")
    hour = st.number_input("Hour of Transaction (24-hour format)", min_value=0, max_value=23, format="%d")
    unix_time = st.number_input("Unix Time", min_value=0, format="%d")
    lat = st.number_input("Latitude", format="%f")
    long = st.number_input("Longitude", format="%f")
    dayofweek = st.number_input("Day of Week (Sunday-1, Monday-2, Tuesday-3 and so on)", min_value=1, max_value=7, format="%d")
    month = st.number_input("Month", min_value=1, max_value=12, format="%d")
    cc_num = st.number_input("Credit Card Number", min_value=0, format="%d")
    city = st.number_input("City", min_value=0, format="%d")
    state = st.number_input("State", min_value=0, format="%d")
    merch_long = st.number_input("Merchant Longitude", format="%f")
    merch_lat = st.number_input("Merchant Latitude", format="%f")
    
    return np.array([[amount_log, category, hour, unix_time, lat, dayofweek, 
                      month, long, cc_num, city, state, merch_long, merch_lat]])

# Get user input
user_input = get_user_input()

if st.button("Predict"):
    prediction = model.predict(user_input)
    result = "Fraud" if prediction[0] == 1 else "Non-Fraud"
    st.write("### Prediction:", result)

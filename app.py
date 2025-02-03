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
    amount_log = st.number_input("Log Amount (Transaction Amount in Log Scale)", min_value=0.0, format="%f")
    
    # Use select box for category (this could be a list of categories based on your dataset)
    category = st.selectbox("Category", ["Electronics", "Clothing", "Grocery", "Entertainment", "Other"])
    
    # Hour of transaction (24-hour format)
    hour = st.slider("Hour of Transaction", min_value=0, max_value=23)
    
    # Merchant ID (the user might not need to enter it directly; you could auto-fill this from merchant data)
    merchant = st.text_input("Merchant Name or ID", "")
    
    # Unix time (let the user input a readable date instead if preferred, or auto-fill this)
    unix_time = st.number_input("Unix Time", min_value=0, format="%d")
    
    # Geographical information
    lat = st.number_input("Latitude", format="%f", help="Enter latitude for the transaction location")
    long = st.number_input("Longitude", format="%f", help="Enter longitude for the transaction location")
    
    # Day of the week as dropdown
    dayofweek = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    
    # Month of the transaction
    month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August", 
                                  "September", "October", "November", "December"])
    
    # Credit card number (it's a sensitive input, so ensure you're handling it securely)
    cc_num = st.text_input("Credit Card Number (XXXX-XXXX-XXXX-XXXX)", type="password")
    
    # Use selectbox for city and state (make sure to populate them with relevant values from your dataset)
    city = st.selectbox("City", ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Other"])
    state = st.selectbox("State", ["California", "Texas", "Florida", "New York", "Other"])

    # Merchant's geographical location (latitude and longitude)
    merch_lat = st.number_input("Merchant Latitude", format="%f")
    merch_long = st.number_input("Merchant Longitude", format="%f")
    
    return np.array([[amount_log, category, hour, merchant, unix_time, lat, dayofweek, 
                      month, long, cc_num, city, state, merch_long, merch_lat]])

# Get user input
user_input = get_user_input()

# Show the collected inputs
st.write("### You entered:")
st.write(f"Transaction Amount: {user_input[0][0]}")
st.write(f"Category: {user_input[0][1]}")
st.write(f"Hour: {user_input[0][2]}")
st.write(f"Merchant: {user_input[0][3]}")
st.write(f"Unix Time: {user_input[0][4]}")
st.write(f"Latitude: {user_input[0][5]}")
st.write(f"Longitude: {user_input[0][6]}")
st.write(f"Day of Week: {user_input[0][7]}")
st.write(f"Month: {user_input[0][8]}")
st.write(f"Credit Card Number: {user_input[0][9]}")
st.write(f"City: {user_input[0][10]}")
st.write(f"State: {user_input[0][11]}")
st.write(f"Merchant Latitude: {user_input[0][12]}")
st.write(f"Merchant Longitude: {user_input[0][13]}")

# Prediction button
if st.button("Predict"):
    prediction = model.predict(user_input)
    result = "Fraud" if prediction[0] == 1 else "Non-Fraud"
    st.write("### Prediction:", result)


import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
model = joblib.load('xgb_model.joblib')  # Replace with your actual model file

# Define encoders for categorical columns
label_encoders = {
    'category': LabelEncoder().fit(['Entertainment', 'Services', 'Restaurant', 'Electronics', 
                                    'Children', 'Fashion', 'Food', 'Products', 
                                    'Subscription', 'Gaming']),
    'merchant': LabelEncoder().fit(['Amazon', 'Walmart', 'BestBuy', 'Ebay', 'Starbucks']),  # Add real merchants
    'city': LabelEncoder().fit(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami']),  # Example cities
    'state': LabelEncoder().fit(['NY', 'CA', 'TX', 'FL', 'IL']),  # Example states
}

# Define the scaler (Use the same scaler from training)
scaler = StandardScaler()

# Prediction function
def predict_fraud(amount_log, category, hour, merchant, unix_time, lat, long, dayofweek, month, 
                  cc_num, city, state, merch_long, merch_lat):
    
    input_data = pd.DataFrame({
        'amount_log': [amount_log],
        'category': [category],
        'hour': [hour],
        'merchant': [merchant],
        'unix_time': [unix_time],
        'lat': [lat],
        'long': [long],
        'dayofweek': [dayofweek],
        'month': [month],
        'cc_num': [cc_num],
        'city': [city],
        'state': [state],
        'merch_long': [merch_long],
        'merch_lat': [merch_lat]
    })

    # Encode categorical features
    for col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Standardize numerical columns
    numerical_cols = ['amount_log', 'hour', 'unix_time', 'lat', 'long', 'dayofweek', 'month', 
                      'cc_num', 'merch_long', 'merch_lat']
    input_data[numerical_cols] = scaler.fit_transform(input_data[numerical_cols])  # Use fitted scaler

    # Make prediction
    prediction = model.predict(input_data)
    
    return "Fraud" if prediction[0] == 1 else "Not Fraud"

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Credit Card Fraud Detection App</h1>", unsafe_allow_html=True)

# Input fields
col1, col2, col3 = st.columns(3)
amount_log = col1.number_input("Log Amount", min_value=0.0, format="%.2f")
category = col2.selectbox("Category", label_encoders['category'].classes_)
hour = col3.number_input("Hour", min_value=0, max_value=23, value=12)

col4, col5, col6 = st.columns(3)
merchant = col4.selectbox("Merchant", label_encoders['merchant'].classes_)
unix_time = col5.number_input("Unix Time", min_value=0)
lat = col6.number_input("Latitude", format="%.6f")

col7, col8, col9 = st.columns(3)
long = col7.number_input("Longitude", format="%.6f")
dayofweek = col8.number_input("Day of Week", min_value=0, max_value=6)
month = col9.number_input("Month", min_value=1, max_value=12)

col10, col11, col12 = st.columns(3)
cc_num = col10.number_input("Credit Card Number", min_value=1000000000000000, max_value=9999999999999999, step=1)
city = col11.selectbox("City", label_encoders['city'].classes_)
state = col12.selectbox("State", label_encoders['state'].classes_)

col13, col14 = st.columns(2)
merch_long = col13.number_input("Merchant Longitude", format="%.6f")
merch_lat = col14.number_input("Merchant Latitude", format="%.6f")

if st.button("Predict"):
    prediction = predict_fraud(amount_log, category, hour, merchant, unix_time, lat, long, 
                               dayofweek, month, cc_num, city, state, merch_long, merch_lat)
    
    if prediction == "Fraud":
        st.markdown(f"<p style='color: red; font-size: 24px; font-weight: bold;'>Prediction: {prediction}</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color: green; font-size: 24px; font-weight: bold;'>Prediction: {prediction}</p>", unsafe_allow_html=True)

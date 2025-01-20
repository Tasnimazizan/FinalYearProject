import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
model = joblib.load('random_forest_model.pkl')  # Replace with your actual model file

# Define encoders for categorical columns using the actual values
label_encoders = {
    'Day of Week': LabelEncoder().fit(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
    'Type of Card': LabelEncoder().fit(['Visa', 'MasterCard']),
    'Entry Mode': LabelEncoder().fit(['Tap', 'PIN', 'CVC']),
    'Type of Transaction': LabelEncoder().fit(['POS', 'Online', 'ATM']),
    'Merchant Group': LabelEncoder().fit(['Entertainment', 'Services', 'Restaurant', 'Electronics', 'Children', 
                                          'Fashion', 'Food', 'Products', 'Subscription', 'Gaming']),
    'Country of Transaction': LabelEncoder().fit(['United Kingdom', 'USA', 'India', 'Russia', 'China']),
    'Shipping Address': LabelEncoder().fit(['United Kingdom', 'USA', 'India', 'Russia', 'China']),
    'Country of Residence': LabelEncoder().fit(['United Kingdom', 'USA', 'India', 'Russia', 'China']),
    'Gender': LabelEncoder().fit(['M', 'F'])
}

# Define the scaler for numerical columns (use the scaler from training if available)
scaler = StandardScaler()

# Define the function to make predictions
def predict_fraud(day_of_week, time, type_of_card, entry_mode, amount, type_of_transaction, merchant_group, 
                  country_of_transaction, shipping_address, country_of_residence, gender, age):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Day of Week': [day_of_week],
        'Time': [time],
        'Type of Card': [type_of_card],
        'Entry Mode': [entry_mode],
        'Amount': [amount],
        'Type of Transaction': [type_of_transaction],
        'Merchant Group': [merchant_group],
        'Country of Transaction': [country_of_transaction],
        'Shipping Address': [shipping_address],
        'Country of Residence': [country_of_residence],
        'Gender': [gender],
        'Age': [age],
    })

    # Encode categorical columns
    for col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Standardize numerical features
    numerical_cols = ['Time', 'Amount', 'Age']
    input_data[numerical_cols] = scaler.fit_transform(input_data[numerical_cols])  # Use the fitted scaler from training

    # Make the prediction
    prediction = model.predict(input_data)

    # Convert the numeric prediction to a meaningful label
    return "Fraud" if prediction[0] == 1 else "Not Fraud"

# Custom CSS for background, fonts, and boxes
st.markdown("""
    <style>
    /* Background color for the app */
    .main {
        background-color: #f0f2f6;
        font-family: 'Helvetica', sans-serif;
    }

    /* Title and headers */
    h1, h2, h3, h4, h5, h6 {
        color: #3c3c3c;
        font-family: 'Arial', sans-serif;
    }

    /* Customizing the input headers (labels) */
    .stSelectbox label, .stNumberInput label {
        font-size: 16px;
        color: #333333;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    /* Input boxes */
    .stSelectbox, .stNumberInput {
        background-color: #e6eaf2;
        border-radius: 10px;
        color: #3c3c3c;
    }

    /* Adjust buttons */
    button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
    }

    /* Custom text for the prediction output */
    .output-text {
        font-size: 24px;
        font-weight: bold;
    }

    /* Red text for fraud prediction */
    .fraud {
        color: red;
    }

    /* Green text for not fraud prediction */
    .not-fraud {
        color: green;
    }

    /* Custom styles for plus/minus buttons */
    .stNumberInput button {
        background-color: #d0d3da !important;  /* Light color for the plus/minus buttons */
    }

    </style>
    """, unsafe_allow_html=True)

# Streamlit app layout
st.markdown("<h1 style='text-align: center; font-family: Arial, sans-serif; color: #4CAF50;'>Credit Card Fraud Detection App</h1>", unsafe_allow_html=True)

# First row: Day of Week, Time, Type of Card
col1, col2, col3 = st.columns(3)
day_of_week = col1.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
time = col2.number_input("Time", min_value=0, max_value=24, value=12)  # Example for time in HHMM format
type_of_card = col3.selectbox("Type of Card", ['Visa', 'MasterCard'])

# Second row: Entry Mode, Amount, Type of Transaction
col4, col5, col6 = st.columns(3)
entry_mode = col4.selectbox("Entry Mode", ['Tap', 'PIN', 'CVC'])
amount = col5.number_input("Amount", min_value=0.0, format="%.2f")
type_of_transaction = col6.selectbox("Type of Transaction", ['POS', 'Online', 'ATM'])

# Third row: Merchant Group, Country of Transaction
col7, col8 = st.columns(2)
merchant_group = col7.selectbox("Merchant Group", ['Entertainment', 'Services', 'Restaurant', 'Electronics', 
                                                   'Children', 'Fashion', 'Food', 'Products', 
                                                   'Subscription', 'Gaming'])
country_of_transaction = col8.selectbox("Country of Transaction", ['United Kingdom', 'USA', 'India', 'Russia', 'China'])

# Fourth row: Shipping Address, Country of Residence, Gender
col9, col10, col11 = st.columns(3)
shipping_address = col9.selectbox("Shipping Address", ['United Kingdom', 'USA', 'India', 'Russia', 'China'])
country_of_residence = col10.selectbox("Country of Residence", ['United Kingdom', 'USA', 'India', 'Russia', 'China'])
gender = col11.selectbox("Gender", ['M', 'F'])

# Fifth row: Age
col12 = st.columns(1)
age = col12[0].number_input("Age", min_value=0)

if st.button("Predict"):
    prediction = predict_fraud(day_of_week, time, type_of_card, entry_mode, amount, type_of_transaction, 
                               merchant_group, country_of_transaction, shipping_address, 
                               country_of_residence, gender, age)
    
    # Conditional formatting for prediction output
    if prediction == "Fraud":
        st.markdown(f"<p class='output-text fraud'>Prediction: {prediction}</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='output-text not-fraud'>Prediction: {prediction}</p>", unsafe_allow_html=True)

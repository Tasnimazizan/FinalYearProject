import random
import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

# Load and preprocess the dataset
data = pd.read_csv("fraudTrain.csv")
data = data.sample(frac=0.1, random_state=42)  # Use a smaller subset for performance

# Data preprocessing
data['is_fraud'] = data['is_fraud'].astype(int)
X = data[['amt', 'gender', 'category', 'city_pop', 'age', 'hour', 'state', 'zip', 'trans_date_trans_time']]
y = data['is_fraud']

# One-hot encoding for categorical variables
X = pd.get_dummies(X, columns=['gender', 'category', 'state'], drop_first=True)

# Train an XGBoost model
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# Save the model for deployment
joblib.dump(model, 'fraud_model.pkl')

# Load the model
model = joblib.load('fraud_model.pkl')

# Streamlit App
st.title("Credit Card Fraud Detection App")

# Input fields
amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
gender = st.selectbox("Gender", ["M", "F"])
category = st.selectbox("Transaction Category", data['category'].unique())
city_population = st.number_input("City Population", min_value=0, step=1)
age = st.number_input("Age", min_value=0, max_value=120, step=1)
hour = st.number_input("Transaction Hour", min_value=0, max_value=23, step=1)
state = st.selectbox("State", data['state'].unique())
zip_code = st.number_input("ZIP Code", min_value=0, step=1)
transaction_time = st.text_input("Transaction Date and Time (YYYY-MM-DD HH:MM:SS)")

# Predict button
if st.button("Predict"):
    # Create input data for prediction
    input_data = pd.DataFrame({
        'amt': [amount],
        'city_pop': [city_population],
        'age': [age],
        'hour': [hour],
        'zip': [zip_code],
        'gender_M': [1 if gender == "M" else 0],
        **{f'category_{cat}': [1 if category == cat else 0] for cat in data['category'].unique()},
        **{f'state_{st}': [1 if state == st else 0] for st in data['state'].unique()}
    })

    # Predict
    prediction = model.predict(input_data)[0]
    
    # Display result
    if prediction == 1:
        st.write("Prediction: Fraud")
    else:
        st.write("Prediction: Not Fraud")


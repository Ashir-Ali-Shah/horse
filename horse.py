import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('rf_model.joblib')
scaler = joblib.load('scaler-2.joblib')

# Function to predict the outcome based on input features
def predict_outcome(input_features):
    input_array = np.array(input_features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    return prediction[0]

st.title("Horse Racing Outcome Predictor")

# Input fields for the features
starts = st.number_input('Starts', min_value=0, value=150)
first = st.number_input('First', min_value=0, value=30)
second = st.number_input('Second', min_value=0, value=25)
third = st.number_input('Third', min_value=0, value=20)
purse = st.number_input('Purse', min_value=0, value=200000)
win_percentage = st.number_input('Win %', min_value=0.0, value=20.0)
top3_percentage = st.number_input('Top 3 %', min_value=0.0, value=0.40)

# Predict button
if st.button('Predict'):
    input_features = [starts, first, second, third, purse, win_percentage, top3_percentage]
    prediction = predict_outcome(input_features)
    st.write(f"The predicted number of wins (First) is: {prediction}")

# Run this Streamlit app using the command: streamlit run app.py

import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("house_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit app title
st.title("House Value Prediction App")

# Input form for user to enter the number of rooms
st.header("Enter the Number of Rooms")
total_rooms = st.number_input("Total Rooms", min_value=1, step=1, value=1)

# Predict button
if st.button("Predict"):
    # Scale the input
    scaled_input = scaler.transform(np.array([[total_rooms]]))
    
    # Make prediction
    predicted_value = model.predict(scaled_input)
    
    # Display the result
    st.subheader("Predicted Median House Value:")
    st.write(f"${predicted_value[0]:,.2f}")
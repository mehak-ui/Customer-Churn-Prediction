import streamlit as st
import numpy as np
import pickle
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Customer Churn Prediction")
st.divider()
st.write("Please enter the values and the hit the predict button for getting a prediction")
st.divider()
age = st.number_input("Enter Age", min_value=12, max_value=83, value=30)
gender = st.selectbox("Select Gender",("Male","Female"))
tenure = st.number_input("Enter Tenure", min_value=0, max_value=122, value=10)
monthly_charges = st.number_input("Enter Monthly Charges", min_value=30.0, max_value=120.0, value=75.0)
st.divider()
predictbutton = st.button("Predict")
if predictbutton:
  gender_selection = 1 if gender == "Female" else 0
  X = np.array([age,gender_selection,tenure,monthly_charges])
  X_array = scaler.transform([X])
  prediction = model.predict(X_array)[0]
  predicted = "Yes" if prediction == 1 else "No"
  st.write(f"Predicted: {predicted}")
else:
  st.write("Please hit the predict button")

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Healthcare Patient Records", layout="centered")

st.title("Feature Engineering on Healthcare Patient Records")
st.write("Upload patient details to predict medical condition category.")

model = joblib.load("healthcare_model.pkl")
scaler = joblib.load("scaler.pkl")

gender_map = {'Male': 1, 'Female': 0}
blood_types = ['A', 'B', 'AB', 'O']
admission_types = ['Emergency', 'Elective', 'Urgent']
medications = ['Drug A', 'Drug B', 'Drug C', 'Drug D']
test_results = ['Normal', 'Abnormal', 'Inconclusive']

age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ['Male', 'Female'])
blood_type = st.selectbox("Blood Type", blood_types)
billing_amount = st.number_input("Billing Amount", min_value=0.0, value=1000.0)
admission_type = st.selectbox("Admission Type", admission_types)
medication = st.selectbox("Medication", medications)
test_result = st.selectbox("Test Result", test_results)
year = st.number_input("Admission Year", min_value=2000, max_value=2030, value=2023)
month = st.number_input("Admission Month", min_value=1, max_value=12, value=5)
day = st.number_input("Admission Day", min_value=1, max_value=31, value=15)

input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender_map[gender]],
    'Blood Type': [blood_types.index(blood_type)],
    'Billing Amount': [billing_amount],
    'Admission Type': [admission_types.index(admission_type)],
    'Medication': [medications.index(medication)],
    'Test Results': [test_results.index(test_result)],
    'Admission_Year': [year],
    'Admission_Month': [month],
    'Admission_Day': [day]
})

scaled_data = scaler.transform(input_data)
prediction = model.predict(scaled_data)

if st.button("Predict Medical Condition"):
    st.success(f"Predicted Medical Condition (encoded): {prediction[0]}")

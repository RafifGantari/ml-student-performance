import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')
lencoding = joblib.load('model/label_encoder.pkl')

st.title("Student Performance Prediction")
st.write("Enter the student details to predict their performance.")

hoursStudied = st.number_input("Hours Studied", min_value=0, max_value=100, value=10)
previousScores = st.number_input("Previous Scores", min_value=0, max_value=100, value=75)
extracurricularActivities = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)
sleepHours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
sampleQuestionPapersPracticed = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=100, value=5)

if st.button("Predict Performance"):
    encoded_cat = lencoding.transform([extracurricularActivities])[0]
    input_data = pd.DataFrame({
    'Hours Studied': [hoursStudied],
    'Previous Scores': [previousScores],
    'Extracurricular Activities': [encoded_cat],
    'Sleep Hours': [sleepHours],
    'Sample Question Papers Practiced': [sampleQuestionPapersPracticed]
    })

    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)

    st.success(f"Predicted Student Performance: {prediction[0]:.2f}")
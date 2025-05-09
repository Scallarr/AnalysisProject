
import streamlit as st
import pickle
import numpy as np

# Load model and all necessary components from one file
bundle = pickle.load(open("model.pkl", "rb"))
model = bundle["model"]
scaler = bundle["scaler"]
encoder_gender = bundle["encoder_gender"]
encoder_target = bundle["encoder_target"]

st.title("🧠 Obesity Category Prediction")
st.write("Enter your health information to predict obesity category")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)

# Auto calculate BMI
bmi = weight / ((height / 100) ** 2)
st.write(f"📏 Auto-calculated BMI: **{bmi:.2f}**")

activity = st.slider("Physical Activity Level (1 = Low, 5 = High)", min_value=1, max_value=5, value=3)

if st.button("Predict"):
    gender_encoded = encoder_gender.transform([gender])[0]
    features = np.array([[age, gender_encoded, height, weight, bmi, activity]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    category = encoder_target.inverse_transform([prediction])[0]

    st.success(f"Predicted Obesity Category: **{category}**")

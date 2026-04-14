import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("breast_cancer.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("Breast Cancer Prediction App")
st.write("Enter values for the 10 selected features:")

# Define input fields for user
features = []
feature_names = [
    'radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave_points_mean',
    'radius_worst', 'perimeter_worst', 'area_worst', 'concavity_worst', 'concave_points_worst'
]

for feature in feature_names:
    value = st.number_input(f"Enter {feature}", min_value=0.0, format="%.4f")
    features.append(value)

# Predict when the button is clicked
if st.button("Predict"):
    # Convert input into numpy array & scale
    features_array = np.array([features])
    features_scaled = scaler.transform(features_array)  # Standardize the inputs
    
    prediction = model.predict(features_scaled)[0]
    
    # Map prediction to labels
    result = "M (Malignant)" if prediction == 1 else "B (Benign)"
    
    st.write(f"**Prediction:** {result}")

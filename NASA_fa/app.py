import streamlit as st
import pandas as pd
import joblib
import os

# Title
st.title("NASA Engine RUL Prediction")

# Load the cleaned training data to get feature names
data = pd.read_csv("data/train_FD001_cleaned.csv")

# Safely drop unwanted columns if present
columns_to_drop = [col for col in ["unit_number", "time_in_cycles", "RUL"] if col in data.columns]
feature_columns = data.drop(columns=columns_to_drop).columns.tolist()

# Load the trained model
model_path = "models/rul_model.pkl"
if not os.path.exists(model_path):
    st.error("Trained model not found! Please train the model first.")
    st.stop()

model = joblib.load(model_path)

# Sidebar for input
st.sidebar.header("Enter Sensor and Operational Settings")

input_data = {}
for feature in feature_columns:
    input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

# Predict
if st.sidebar.button("Predict RUL"):
    input_df = pd.DataFrame([input_data])

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Remaining Useful Life (RUL): {int(prediction)} cycles")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Show feature importance if available
if os.path.exists("assets/feature_importance.png"):
    st.image("assets/feature_importance.png", caption="Feature Importance", use_container_width=True)

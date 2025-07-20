
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("income_classifier.pkl")

st.set_page_config(page_title="Income Classification", layout="centered")
st.title("Employee Income Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 47, age - 18)

# Format input for prediction
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

st.write("### Input Preview")
st.write(input_df)

# Predict button
if st.button("Predict Income Class"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Income Class: **{prediction[0]}**")

# Batch prediction
st.markdown("---")
st.markdown("###  Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV file with correct structure", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)

        # Ensure required columns
        expected_columns = ['age', 'education', 'occupation', 'hours-per-week']
        for col in expected_columns:
            if col not in batch_data.columns:
                st.error(f"Missing required column: `{col}`")
                st.stop()

        # Compute or validate 'experience'
        if 'experience' not in batch_data.columns:
            batch_data['experience'] = (batch_data['age'] - 18).clip(lower=0)

        st.write("Uploaded Data Preview:")
        st.write(batch_data.head())

        # Predict
        predictions = model.predict(batch_data)
        batch_data['Predicted Income'] = predictions

        st.write("Predictions:")
        st.write(batch_data)

        # Download results
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "predicted_incomes.csv", "text/csv")

    except Exception as e:
        st.error(f" Error processing file: {e}")

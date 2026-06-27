import streamlit as st
import requests

# Our API URL 
API_URL = "http://127.0.0.1:8000/predict"

st.title("Diabetes Risk Prediction")
st.write("Enter patient details:")

# Taking Input with realistic ranges

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
st.caption("Number of times the patient has been pregnant (0–20)")

glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0)
st.caption("Plasma glucose concentration in blood (0–300 mg/dL)")

bp = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0)
st.caption("Diastolic blood pressure (0–200 mm Hg)")

skin = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0)
st.caption("Triceps skin fold thickness (0–100 mm)")

insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0)
st.caption("2-hour serum insulin level (0–900 mu U/ml)")

bmi = st.number_input("BMI", min_value=10.0, max_value=70.0)
st.caption("Body Mass Index (weight divided by height square) (10–70)")

dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0)
st.caption("Genetic likelihood of diabetes (0–3)")

age = st.number_input("Age", min_value=1, max_value=120, step=1)
st.caption("Age of the patient in years (1–120)")

# Checking for Range Validation 
valid = True

if glucose == 0:
    st.warning("Glucose value should not be 0 in real cases.")
    valid = False

if bmi < 12:
    st.warning("BMI seems unrealistically low.")
    valid = False

if age < 5:
    st.warning("Dataset is mostly for adults.")
    valid = False

# Actual Prediction 

if st.button("Predict"):
    if not valid:
        st.error("Please correct the highlighted inputs before predicting.")
    else:
        response = requests.post(
            API_URL,
            json={
                "pregnancies": pregnancies,
                "glucose": glucose,
                "blood_pressure": bp,
                "skin_thickness": skin,
                "insulin": insulin,
                "bmi": bmi,
                "diabetes_pedigree": dpf,
                "age": age
            }
        )

        if response.status_code == 200:
            result = response.json()

            if result["prediction"] == 1:
                st.error(f"High Risk  ({result['risk_probability']:.2f})")
            else:
                st.success(f"Low Risk  ({result['risk_probability']:.2f})")
        else:
            st.error("Error connecting to API")
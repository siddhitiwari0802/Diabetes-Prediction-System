from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Loading the Model using Joblib

pipeline = joblib.load("diabetes_model.pkl")
model = pipeline["model"]
scaler = pipeline["scaler"]


class InputData(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: float

# Accepting the get Request (Home Page) 
@app.get("/")
def home():
    return {"message": "Diabetes Prediction API Running"}


#Accepting the Post request and sending the prediction in the response with risk probability 
@app.post("/predict")
def predict(data: InputData):
    input_data = np.array([[
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree,
        data.age
    ]])

    scaled = scaler.transform(input_data)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    return {
        "prediction": int(prediction),
        "risk_probability": float(probability)
    }
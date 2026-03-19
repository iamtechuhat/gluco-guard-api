from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="GlucoGuard Diabetes API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("diabetes_model (1).pkl")

class Vitals(BaseModel):
    glucose: float
    systolic: float
    diastolic: float
    bmi: float
    age: float
    pregnancies: float
    insulin: float
    skin_thickness: float
    diabetes_pedigree: float

@app.get("/")
def root():
    return {"message": "GlucoGuard API is running"}

@app.post("/predict")
def predict(vitals: Vitals):
    glucose_bmi = vitals.glucose * vitals.bmi
    age_glucose = vitals.age * vitals.glucose
    bp_ratio = vitals.systolic / (vitals.diastolic + 1)
    insulin_glucose = vitals.insulin / (vitals.glucose + 1)
    bmi_age = vitals.bmi * vitals.age

    features = np.array([[
        vitals.glucose,
        vitals.systolic,
        vitals.bmi,
        vitals.age,
        vitals.pregnancies,
        vitals.insulin,
        vitals.skin_thickness,
        vitals.diabetes_pedigree,
        glucose_bmi,
        age_glucose,
        bp_ratio,
        insulin_glucose,
        bmi_age
    ]])

    prediction = model.predict(features)
    probability = model.predict_proba(features)
    risk = "High Risk" if prediction[0] == 1 else "Low Risk"
    confidence = round(float(max(probability[0])) * 100, 1)

    return {
        "risk": risk,
        "confidence": f"{confidence}%",
        "prediction": int(prediction[0])
    }

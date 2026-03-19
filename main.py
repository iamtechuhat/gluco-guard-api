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
    age: int
    pregnancies: float = 0
    insulin: float = 0
    skin_thickness: float = 0
    diabetes_pedigree: float = 0.0

@app.get("/")
def root():
    return {"message": "GlucoGuard API is running!", "accuracy": "75.32%"}

@app.post("/predict")
def predict_diabetes(vitals: Vitals):
    glucose_bmi     = vitals.glucose * vitals.bmi
    age_glucose     = vitals.age * vitals.glucose
    bp_ratio        = vitals.systolic / (vitals.bmi + 0.001)
    insulin_glucose = vitals.insulin / (vitals.glucose + 0.001)
    bmi_age         = vitals.bmi * vitals.age

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

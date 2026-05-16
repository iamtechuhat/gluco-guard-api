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

model = joblib.load("diabetes_model .pkl")


class Vitals(BaseModel):
    glucose: float
    systolic: float
    diastolic: float
    bmi: float
    age: float
    family_history_score: float


@app.get("/")
def root():
    return {"message": "GlucoGuard API is running"}


@app.post("/predict")
def predict(vitals: Vitals):
    features = np.array([[
        vitals.glucose,
        vitals.systolic,
        vitals.diastolic,
        vitals.bmi,
        vitals.age,
        vitals.family_history_score,
    ]], dtype=float)

    prediction = model.predict(features)
    prediction_value = int(prediction[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features)
        confidence = round(float(max(probability[0])) * 100, 1)

    risk = "High Risk" if prediction_value == 1 else "Low Risk"

    return {
        "risk": risk,
        "prediction": prediction_value,
        "confidence": f"{confidence}%" if confidence is not None else ""
    }

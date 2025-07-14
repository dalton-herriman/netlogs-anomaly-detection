from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Load model once at startup
MODEL_PATH = "models/xgb_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Anomaly Detection API")

# Example input format
class LogEntry(BaseModel):
    duration: float
    protocol: int
    src_port: int
    dst_port: int
    packet_count: int
    byte_count: int

@app.get("/")
def root():
    return {"message": "Anomaly Detection API is running."}

@app.post("/predict")
def predict(log: LogEntry):
    # Convert input to dataframe
    df = pd.DataFrame([log.dict()])

    # Preprocess: scale inputs (temporary scaler fit — replace w/ saved if needed)
    scaler = joblib.load("models/scaler.joblib")
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

    # Predict
    prediction = model.predict(df_scaled)[0]
    score = model.predict_proba(df_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "anomaly_score": round(float(score), 4)
    }

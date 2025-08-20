from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import pandas as pd
import os

# --------------- CONFIG -----------------
BUCKET = "telco-churn-prediction-bucket"
MODEL_KEY = "telco-trained-models/LogisticRegression_selected.pkl"
ENCODER_KEY = "telco-model-artifacts/label_encoders.pkl"
SCALER_KEY = "telco-model-artifacts/tenure_scaler.pkl"

SELECTED_FEATURES = [
    "tenure",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "Contract"
]

app = FastAPI()

s3 = boto3.client("s3")
model = None
encoders = None
tenure_scaler = None

# --------------- Pydantic Schema -----------------
class CustomerFeatures(BaseModel):
    tenure: float
    InternetService: str
    OnlineSecurity: str
    TechSupport: str
    Contract: str

@app.get("/")
def health_check():
    return {"message": "API is healthy"}


# --------------- S3 Downloads -----------------
def download_from_s3(s3_key, local_path):
    s3.download_file(BUCKET, s3_key, local_path)

@app.on_event("startup")
def load_artifacts():
    global model, encoders, tenure_scaler

    # Model
    model_path = "/tmp/model.pkl"
    download_from_s3(MODEL_KEY, model_path)
    model = joblib.load(model_path)

    # Label encoders
    encoder_path = "/tmp/label_encoders.pkl"
    download_from_s3(ENCODER_KEY, encoder_path)
    encoders = joblib.load(encoder_path)

    # Tenure scaler
    scaler_path = "/tmp/tenure_scaler.pkl"
    download_from_s3(SCALER_KEY, scaler_path)
    tenure_scaler = joblib.load(scaler_path)

    print("✅ Model, encoders, and scaler loaded.")

# --------------- Prediction Route -----------------
@app.post("/predict")
def predict(input: CustomerFeatures):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input.dict()])

        # Encode categorical features
        for col in df.columns:
            if col != "tenure" and col in encoders:
                df[col] = encoders[col].transform(df[col])

        # Scale tenure
        df["tenure"] = tenure_scaler.transform(df[["tenure"]])

        # Ensure column order matches model training
        df = df[SELECTED_FEATURES]

        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "churn": bool(prediction),
            "probability": round(probability, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

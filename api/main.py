from fastapi import FastAPI
import joblib
import pandas as pd

# loading trained model
model = joblib.load("models/lightgbm_delivery_delay.pkl")

# creating FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Delivery Delay Prediction API is running"
    }

@app.post("/predict")
def predict(data: dict):

    # convert incoming json into dataframe
    df = pd.DataFrame([data])

    # probability of late delivery
    prob = model.predict_proba(df)[0][1]

    # convert probability into label
    prediction = (
        "Late Delivery"
        if prob >= 0.5
        else "On-Time Delivery"
    )

    return {
        "delay_probability": round(float(prob), 4),
        "prediction": prediction
    }
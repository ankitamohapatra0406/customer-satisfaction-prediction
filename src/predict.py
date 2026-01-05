import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

model = joblib.load(MODEL_PATH)

FEATURES = [
    "Seat comfort",
    "Inflight wifi service",
    "Online boarding",
    "Baggage handling",
    "Cleanliness",
    "Checkin service",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Departure Delay in Minutes"
]

def predict_customer(data):
    row = {f: float(data.get(f, 0)) for f in FEATURES}
    df = pd.DataFrame([row])
    pred = model.predict(df)[0]
    return "Satisfied 😊" if pred == 1 else "Not Satisfied 😞"

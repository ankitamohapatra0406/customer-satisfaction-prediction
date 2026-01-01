import joblib
import pandas as pd

model = joblib.load("models/model.pkl")
FEATURES = model.feature_names_in_

def scale(v):
    # convert 1–10 rating into 0–4 scale
    return round((v - 1) * 4 / 9, 2)

def prepare_input(data):
    row = {}
    for f in FEATURES:
        if f in data:
            row[f] = scale(float(data[f])) if "service" in f.lower() or "comfort" in f.lower() else float(data[f])
        else:
            row[f] = 0
    return pd.DataFrame([row])

def predict_customer(data_dict):
    df = prepare_input(data_dict)
    pred = model.predict(df)[0]
    return "Satisfied 😄" if pred == 1 else "Not Satisfied 😞"

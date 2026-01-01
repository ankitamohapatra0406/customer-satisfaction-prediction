import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

def predict_customer(data_dict):
    df = pd.DataFrame([data_dict])
    prediction = model.predict(df)[0]
    return "Satisfied" if prediction == 1 else "Not Satisfied"

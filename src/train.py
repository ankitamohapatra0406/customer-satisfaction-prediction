import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

df = pd.read_csv("../data/processed/processed_data.csv")


service_cols = [
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

X = df[service_cols]
y = df["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    max_depth=12,
    min_samples_leaf=5
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("\nMODEL ACCURACY:", acc)
print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_test, pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("\nModel saved to models/model.pkl")

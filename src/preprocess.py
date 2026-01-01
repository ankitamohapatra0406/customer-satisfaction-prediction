import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/train.csv")

le = LabelEncoder()

df['satisfaction'] = le.fit_transform(df['satisfaction'])

for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

df.to_csv("data/processed/processed_data.csv", index=False)

print("Processed dataset saved successfully.")

from flask import Flask, request, jsonify
from src.predict import predict_customer

app = Flask(__name__)

@app.route("/")
def home():
    return "Customer Satisfaction Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    result = predict_customer(data)
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)

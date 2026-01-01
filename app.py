from flask import Flask, render_template, request, jsonify
from src.predict import predict_customer

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        data = dict(request.form)
        for k in data:
            data[k] = float(data[k])
        result = predict_customer(data)
    return render_template("index.html", result=result)

@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.json
    result = predict_customer(data)
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)

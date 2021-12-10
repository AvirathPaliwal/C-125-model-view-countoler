from flask import Flask , jsonify , request
from cls import  get_prediction

app = Flask(__name__)

@app.route("/")
def home():
    return "welcome to the home page"

@app.route("/predict-digit", methods = ["POST"])
def predict_digit():
    img = request.files.get("digit")
    prediction = get_prediction(img)
    return jsonify( {
        "prediction" : prediction
    }),200


if __name__ == "__main__":
    app.run(debug = True)
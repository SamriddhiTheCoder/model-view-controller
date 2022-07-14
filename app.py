import jsonify
from flask import Flask
import requests
from classifier import get_prediction

app = Flask(__name__)
@app.route("/predict-digit", methods=["POST"])

def predict_data():
  image = requests.files.get("alphabet")
  prediction = get_prediction(image)
  return jsonify({
    "prediction": prediction
  }), 200

if __name__ == "__main__":
  app.run(debug=True)
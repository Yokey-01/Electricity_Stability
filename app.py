# Import necessary libraries
import pickle
import numpy as np
from flask import Flask, render_template, request
import joblib

# Create a Flask application instance
app = Flask(__name__)

# Prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 13)
    loaded_model = joblib.load("electricity_model_new.pkl")
    result = loaded_model.predict(to_predict)
    return result[0]

# Define the route for the form submission
@app.route('/result', methods=['POST'])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction = 'Unstable'
        else:
            prediction = 'Stable'
        return render_template("result.html", prediction=prediction)

# Define the main route
@app.route('/')
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

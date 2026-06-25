from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("weather_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    humidity = float(request.form["humidity"])
    wind_speed = float(request.form["wind_speed"])
    wind_bearing = float(request.form["wind_bearing"])
    visibility = float(request.form["visibility"])
    pressure = float(request.form["pressure"])
    month = int(request.form["month"])
    day = int(request.form["day"])
    year = float(request.form["year"])

    features = np.array([[
    humidity,
    wind_speed,
    wind_bearing,
    visibility,
    pressure,
    year,
    month,
    day
]])
    prediction = model.predict(features)

    return render_template(
    "index.html",
    prediction_text=f"🌡️ Predicted Temperature: {prediction[0]:.2f} °C"
)


if __name__ == "__main__":
    app.run(debug=True)
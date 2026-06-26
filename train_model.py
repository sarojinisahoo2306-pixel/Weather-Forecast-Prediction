import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("weatherHistory.csv")

# Keep only required columns
data = data[[
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)",
    "Formatted Date",
    "Temperature (C)"
]]

# Convert date
data["Formatted Date"] = pd.to_datetime(data["Formatted Date"], utc=True)

data["Year"] = data["Formatted Date"].dt.year
data["Month"] = data["Formatted Date"].dt.month
data["Day"] = data["Formatted Date"].dt.day

# Features
X = data[[
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)",
    "Year",
    "Month",
    "Day"
]]

# Target
y = data["Temperature (C)"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "weather_model.pkl")

print("Model saved successfully!")
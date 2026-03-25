import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("data.csv")

# Basic stats
print("Data Preview:\n", data.head())

# Detect anomalies (simple threshold)
mean = data["consumption"].mean()
std = data["consumption"].std()

threshold = mean + 2 * std
anomalies = data[data["consumption"] > threshold]

print("\nAnomalies:\n", anomalies)

# Prepare ML model
X = data[["day"]]
y = data["consumption"]

model = LinearRegression()
model.fit(X, y)

# Predict next 5 days
future_days = pd.DataFrame({"day": range(11, 16)})
predictions = model.predict(future_days)

print("\nPredictions for next 5 days:")
for day, pred in zip(future_days["day"], predictions):
    print(f"Day {day}: {pred:.2f}")

# Plot
plt.scatter(data["day"], data["consumption"], label="Actual")
plt.plot(future_days["day"], predictions, color="red", label="Predicted")
plt.xlabel("Day")
plt.ylabel("Energy Consumption")
plt.legend()
plt.title("Energy Consumption Analysis")
plt.show()
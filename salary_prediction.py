# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("Salary_Data.csv")

# Show dataset
print(data.head())

# Split data into input and output
X = data[['YearsExperience']]
y = data['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print predictions
print("\nPredicted salaries:", y_pred)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

# ---------------- GRAPH ----------------

# Plot training data
plt.scatter(X, y, color='blue', label='Actual Data')

# Plot regression line
plt.plot(X, model.predict(X), color='red', label='Regression Line')

plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()

plt.show()
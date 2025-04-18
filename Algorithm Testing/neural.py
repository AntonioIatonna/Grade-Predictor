import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
data = pd.read_csv('Combined_CSV.csv')

# Define feature columns and target column
feature_cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L7B', 'L8', 'A1', 'A2', 'A3', 'A4', 'A5', 'M']
target_col = 'F'

# Separate features (X) and target (y)
X = data[feature_cols]
y = data[target_col]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(len(feature_cols),)),  # Hidden layer 1
    keras.layers.Dense(32, activation='relu'),  # Hidden layer 2
    keras.layers.Dense(16, activation='relu'),  # Hidden layer 3
    keras.layers.Dense(1)  # Output layer (Regression: no activation)
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_data=(X_test_scaled, y_test), verbose=1)

# Predict final grades
y_pred = model.predict(X_test_scaled).flatten()

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print results
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Calculate accuracy metrics
test_size = len(y_test)
total_difference = np.sum(np.abs(y_pred - y_test))
mse_total = np.sum((y_pred - y_test) ** 2)

diff_less_than_20 = np.sum(np.abs(y_pred - y_test) <= 20)
diff_less_than_10 = np.sum(np.abs(y_pred - y_test) <= 10)
diff_less_than_5 = np.sum(np.abs(y_pred - y_test) <= 5)
diff_less_than_1 = np.sum(np.abs(y_pred - y_test) <= 1)

print("RESULTS: out of", test_size, "testing data")
print("Predictions within 1%:", diff_less_than_1)
print("Predictions within 5%:", diff_less_than_5)
print("Predictions within 10%:", diff_less_than_10)
print("Predictions within 20%:", diff_less_than_20)
print("Average difference between predicted and real:", total_difference / test_size)
print("MSE:", mse_total / test_size)
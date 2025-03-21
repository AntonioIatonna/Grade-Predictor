import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load your dataset
data = pd.read_csv('Combined_CSV.csv')

# Define feature columns and target column
feature_cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L7B', 'L8', 'A1', 'A2', 'A3', 'A4', 'A5', 'M']
target_col = 'F'

# Separate features (X) and target (y)
X = data[feature_cols]
y = data[target_col]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Standardize the features (Ridge Regression performs better with scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge Regression model
ridge = Ridge(alpha=2.0)  # Alpha is the regularization strength
ridge.fit(X_train_scaled, y_train)

# Predict final grades
y_pred = ridge.predict(X_test_scaled)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print results
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

test_size = X_test.shape[0]

i=0
total_difference=0
mse=0
diff_less_than_20=0
diff_less_than_10=0
diff_less_than_5=0
diff_less_than_1=0


for x in y_pred:
    diff = abs(y_pred[i] - y_test.iloc[i])
    total_difference = total_difference + diff
    diff_squared = diff * diff
    mse = mse + diff_squared

    if diff<=1:
        diff_less_than_1 = diff_less_than_1 + 1

    if diff<=5:
        diff_less_than_5 = diff_less_than_5 + 1

    if diff<=10:
        diff_less_than_10 = diff_less_than_10 + 1
    
    if diff<=20:
        diff_less_than_20 = diff_less_than_20 + 1

    i = i + 1

print("RESULTS: out of ",test_size, " testing data")
print("Predictions within 1%: ", diff_less_than_1 )
print("Predictions within 5%: ", diff_less_than_5 )
print("Predictions within 10%: ", diff_less_than_10 )
print("Predictions within 20%: ", diff_less_than_20 )
print("Average difference between predicted and real: ", total_difference/test_size)
print("MSE: ", mse/test_size)
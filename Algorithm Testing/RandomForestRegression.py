from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

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

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

test_size = y_pred.size
i = 0

# Initialize variables
total_difference = 0
mse = 0
diff_less_than_1 = 0
diff_less_than_5 = 0
diff_less_than_10 = 0
diff_less_than_20 = 0

# Evaluate predictions
for i, pred in enumerate(y_pred):
    diff = abs(pred - y_test.iloc[i])
    total_difference += diff
    diff_squared = diff * diff
    mse += diff_squared

    if diff <= 1:
        diff_less_than_1 += 1
    if diff <= 5:
        diff_less_than_5 += 1
    if diff <= 10:
        diff_less_than_10 += 1
    if diff <= 20:
        diff_less_than_20 += 1

# Calculate metrics
mse /= test_size

# Print results
print("RESULTS: out of ", test_size, " testing data")
print("Predictions within 1%: ", diff_less_than_1)
print("Predictions within 5%: ", diff_less_than_5)
print("Predictions within 10%: ", diff_less_than_10)
print("Predictions within 20%: ", diff_less_than_20)
print("Average difference between predicted and real: ", total_difference / test_size)
print("MSE: ", mse)
print("R-squared:", r2_score(y_test, y_pred))

import pandas
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR

df = pandas.read_csv("CombiDataset.csv")

# # Remove all rows with 0 in the target column from the dataset
# df = df[df['F'] != 0]

# # print the dataset in its entirety showing all rows
# pandas.set_option('display.max_rows', None)
# print(df)

feature_cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L7B', 'L8', 'A1', 'A2', 'A3', 'A4', 'A5', 'M']
target_col = 'F'

X = df[feature_cols]
y = df[target_col]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train the models
enet1 = ElasticNet(alpha=1, l1_ratio=0.1, max_iter=10000).fit(X, y)
enet2 = ElasticNet(alpha=1, l1_ratio=0.9, max_iter=10000).fit(X, y)
enet3 = ElasticNet(alpha=1, l1_ratio=0.5, max_iter=10000).fit(X, y)
enet4 = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000).fit(X, y)
ksvm1 = SVR(kernel='rbf', C=10, gamma=0.1).fit(X, y)
kvsm2 = SVR(kernel='rbf', C=10, gamma=0.001).fit(X, y)

# Predict the values
# We will use the same data for prediction as the training data for simplicity
sum1 = enet1.predict(X)
sum2 = enet2.predict(X)
sum3 = enet3.predict(X)
sum4 = enet4.predict(X)
ksvm1 = ksvm1.predict(X)
ksvm2 = kvsm2.predict(X)

# Average the predictions of the 4 ElasticNet models
averaged_predictions = np.mean(np.column_stack([sum1, sum2, sum3, sum4]), axis=1)
averaged_series = pandas.Series(averaged_predictions, name="Averaged Predictions")

for index, (pred, actual) in enumerate(zip(averaged_series, y)):
    print(f"Row {index+2}: Averaged Prediction = {pred:.4f}, Actual = {actual}, difference = {pred - actual}")

average_diff = np.mean(np.abs(averaged_series - y))
print(f"Average difference between predicted and real: {average_diff}")

# Average the predictions of the 2 SVR models
averaged_predictions = np.mean(np.column_stack([ksvm1, ksvm2]), axis=1)
averaged_series = pandas.Series(averaged_predictions, name="Averaged Predictions")

for index, (pred, actual) in enumerate(zip(averaged_series, y)):
    print(f"Row {index+2}: Averaged Prediction = {pred:.4f}, Actual = {actual}, difference = {pred - actual}")

average_diff = np.mean(np.abs(averaged_series - y))
print(f"Average difference between predicted and real: {average_diff}")

# Average the predictions of the 4 ElasticNet models and the 2 SVR models each with equal weight
averaged_predictions = np.mean(np.column_stack([sum1, sum2, sum3, sum4, ksvm1, ksvm2]), axis=1)
averaged_series = pandas.Series(averaged_predictions, name="Averaged Predictions")

for index, (pred, actual) in enumerate(zip(averaged_series, y)):
    print(f"Row {index+2}: Averaged Prediction = {pred:.4f}, Actual = {actual}, difference = {pred - actual}")

average_diff = np.mean(np.abs(averaged_series - y))
print(f"Average difference between predicted and real: {average_diff}")

# Average the predictions with the 4 ElasticNet models weighted 0.5 combined and the 2 SVR models weighted 0.5 combined
averaged_predictions1 = np.mean(np.column_stack([sum1, sum2, sum3, sum4]), axis=1)
averaged_predictions2 = np.mean(np.column_stack([ksvm1, ksvm2]), axis=1)
averaged_predictions = (averaged_predictions1 + averaged_predictions2) / 2
averaged_series = pandas.Series(averaged_predictions, name="Averaged Predictions")

for index, (pred, actual) in enumerate(zip(averaged_series, y)):
    print(f"Row {index+2}: Averaged Prediction = {pred:.4f}, Actual = {actual}, difference = {pred - actual}")

average_diff = np.mean(np.abs(averaged_series - y))
print(f"Average difference between predicted and real: {average_diff}")
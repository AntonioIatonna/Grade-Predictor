import pandas
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR


df = pandas.read_csv("CombiDataset.csv")

feature_cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L7B', 'L8', 'A1', 'A2', 'A3', 'A4', 'A5', 'M']
target_col = 'F'

X = df[feature_cols]
y = df[target_col]

scaler = StandardScaler()
X = scaler.fit_transform(X)


enet1 = ElasticNet(alpha=1, l1_ratio=0.1, max_iter=10000, random_state=42).fit(X, y)
enet2 = ElasticNet(alpha=1, l1_ratio=0.9, max_iter=10000, random_state=42).fit(X, y)
enet3 = ElasticNet(alpha=1, l1_ratio=0.5, max_iter=10000, random_state=42).fit(X, y)
enet4 = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000, random_state=42).fit(X, y)

sum1 = enet1.predict(X)
sum2 = enet2.predict(X)
sum3 = enet3.predict(X)
sum4 = enet4.predict(X)

averaged_predictions = np.mean(np.column_stack([sum1, sum2, sum3, sum4]), axis=1)
averaged_series = pandas.Series(averaged_predictions, name="Averaged Predictions")

for index, (pred, actual) in enumerate(zip(averaged_series, y)):
    print(f"Row {index+2}: Averaged Prediction = {pred:.4f}, Actual = {actual}, difference = {pred - actual}")

average_diff = np.mean(np.abs(averaged_series - y))
print(f"Average difference between predicted and real: {average_diff}")
import pandas
from sklearn.exceptions import ConvergenceWarning
import refinedTestHelper
import warnings

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR


def createModelandTest(grades):
    df = pandas.read_csv("CombiDataset.csv")

    feature_cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L7B', 'L8', 'A1', 'A2', 'A3', 'A4', 'A5', 'M']
    target_col = 'F'

    X = df[feature_cols]
    y = df[target_col]

    scaler = StandardScaler()
    feature_columns_scaled = scaler.fit_transform(X)

    svr = SVR(kernel='rbf', C=10, gamma=0.001).fit(feature_columns_scaled, y)
    predicted = svr.predict(grades)

    return predicted[0]  # Extract single predicted value
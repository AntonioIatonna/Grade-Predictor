import pandas
from sklearn import linear_model

def createModelandTest(grades):
    df = pandas.read_csv("CombiDataset.csv")

    feature_cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L7B', 'L8', 'A1', 'A2', 'A3', 'A4', 'A5', 'M']
    target_col = 'F'

    X = df[feature_cols]
    y = df[target_col]

    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    predicted = regr.predict(grades)

    return predicted[0]  # Extract single predicted value


    
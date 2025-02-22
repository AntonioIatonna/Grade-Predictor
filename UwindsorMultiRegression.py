import pandas
from sklearn import linear_model

def createModelandTest(grades):
    df = pandas.read_csv("uwindsorMarksPercentage.csv")

    # Define the exact features used in training
    feature_columns = ['Lab 1','Lab 2','Lab 3','Lab 4','Lab 5','Lab 6','Lab 7','Lab 8','Lab 9','Assignment 1','Assignment 2','Assignment 3','Assignment 4','Assignment 5','Midterm']

    X = df[feature_columns]
    y = df['W-AVG']

    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    predicted = regr.predict(grades)

    return predicted[0]  # Extract single predicted value


    
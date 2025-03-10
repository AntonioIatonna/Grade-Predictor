import numpy as np
import pandas

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


df = pandas.read_csv("CombiDataset.csv")

feature_cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L7B', 'L8', 'A1', 'A2', 'A3', 'A4', 'A5', 'M']
target_col = 'F'

X = df[feature_cols]
y = df[target_col]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

# Standardize the features to be used when needed
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# We will now test and comapre the various different models

# ---- Lasso ----
print("\nTesting Lasso Regression")
alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
max_iters = [1000, 5000, 10000]
for alpha in alphas:
    for max_iter in max_iters:
        lasso = Lasso(alpha=alpha, max_iter=max_iter).fit(X_train_scaled, y_train)
        print("Training set score: {:.8f}".format(lasso.score(X_train_scaled, y_train)))
        print("Test set score: {:.8f}".format(lasso.score(X_test_scaled, y_test)))
        print("Number of features used:", np.sum(lasso.coef_ != 0), " || Alpha:", alpha, " || Max Iterations:", max_iter)

# ---- Multiregression ----
print("\nTesting Linear Regression")
lr = LinearRegression().fit(X_train_scaled, y_train)
print("Training set score: {:.2f}".format(lr.score(X_train_scaled, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test_scaled, y_test)))

# ---- Random Forest ----
print("\nTesting Random Forest")
rf = RandomForestRegressor(n_estimators=100, max_depth=60, random_state=42).fit(X_train_scaled, y_train)
print("Training set score: {:.2f}".format(rf.score(X_train_scaled, y_train)))
print("Test set score: {:.2f}".format(rf.score(X_test_scaled, y_test)))

# ---- Ridge Regression ----
print("\nTesting Ridge Regression")
alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
for alpha in alphas:
    ridge = Ridge(alpha=alpha).fit(X_train, y_train)
    print("Training set score: {:.8f}".format(ridge.score(X_train, y_train)))
    print("Test set score: {:.8f}".format(ridge.score(X_test, y_test)))
    print("Alpha:", alpha)

# ---- Kernelized SVM ----
print("\nTesting Kernelized SVM")
regC = [1.0, 10, 100, 1000]
gamma = [0.001, 0.01, 0.1, 1]
for c in regC:
    for g in gamma:
        svr = SVR(kernel='rbf', C=c, gamma=g).fit(X_train_scaled, y_train)
        print("Training set score: {:.2f}".format(svr.score(X_train_scaled, y_train)))
        print("Test set score: {:.2f}".format(svr.score(X_test_scaled, y_test)))
        print("C:", c, " || Gamma:", g)
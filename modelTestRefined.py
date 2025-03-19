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


df = pandas.read_csv("CombiDataset.csv")

feature_cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L7B', 'L8', 'A1', 'A2', 'A3', 'A4', 'A5', 'M']
target_col = 'F'

X = df[feature_cols]
y = df[target_col]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Standardize the features to be used when needed
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# We will now test and comapre the various different models

# ---- Kernelized SVM ----
print("\nTesting Kernelized SVM")
params = [
    {'c':10, 'Gamma':0.1 },
    {'c':10, 'Gamma':0.001 },
]
for p in params:
    svr = SVR(kernel='rbf', C=p['c'], gamma=p['Gamma']).fit(X_train_scaled, y_train)
    pred = svr.predict(X_test_scaled)
    refinedTestHelper.outPutTestignData(y_test,pred)
   

# ---- ElasticNet Regression ---- 
print("\nTesting ElasticNet Regression")
params = [
    {'Alpha': 1.0000, 'l1_ratio': 0.10},
    {'Alpha': 1.0000, 'l1_ratio': 0.90},
    {'Alpha': 1.0000, 'l1_ratio': 0.50},
    {'Alpha': 0.1000, 'l1_ratio': 0.10}
]
for p in params:
        enet = ElasticNet(alpha=p['Alpha'], l1_ratio=p['l1_ratio'], max_iter=10000, random_state=42).fit(X_train_scaled, y_train)
        pred = enet.predict(X_test_scaled)
        refinedTestHelper.outPutTestignData(y_test,pred)

# ---- Gradient Boosting Regressor ----
print("\nTesting Gradient Boosting Regressor")
gbr = GradientBoostingRegressor(learning_rate=0.01, n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
pred = gbr.predict(X_test_scaled)
refinedTestHelper.outPutTestignData(y_test,pred)

# ---- Neural Network (MLPRegressor) ----
print("\nTesting Neural Network (MLPRegressor)")
params = [
    {'Hidden Layers': (50,), 'Max Iter': 1000},
    {'Hidden Layers': (100,), 'Max Iter': 1000}
]

for p in params:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        mlp = MLPRegressor(hidden_layer_sizes=p['Hidden Layers'],
                           max_iter=p['Max Iter'],
                           learning_rate_init=0.001,
                           alpha=0.01,
                           batch_size=32,
                           random_state=42,
                           ).fit(X_train_scaled, y_train)
        pred = mlp.predict(X_test_scaled)
        refinedTestHelper.outPutTestignData(y_test,pred)
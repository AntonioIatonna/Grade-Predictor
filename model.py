import pandas as pd  # Use 'pd' for consistency
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

def createModelandTest(grades):
    df = pd.read_csv("CombiDataset.csv")

    feature_cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L7B', 'L8', 'A1', 'A2', 'A3', 'A4', 'A5', 'M']
    target_col = 'F'

    X = df[feature_cols]
    y = df[target_col]

    scaler = StandardScaler()
    feature_columns_scaled = scaler.fit_transform(X)

    enet = ElasticNet(alpha=1, l1_ratio=0.1, max_iter=10000, random_state=42).fit(feature_columns_scaled, y)

    # Convert grades to DataFrame with correct column names
    grades_df = pd.DataFrame(grades, columns=feature_cols)

    scaled_input = scaler.transform(grades_df)
    predicted = enet.predict(scaled_input)

    return predicted[0]  # Extract single predicted value

def createModelandTestMidterm(grades):
    df = pd.read_csv("CombiDataset.csv")

    feature_cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'A1', 'A2', 'A3']
    target_col = 'M'

    X = df[feature_cols]
    y = df[target_col]

    scaler = StandardScaler()
    feature_columns_scaled = scaler.fit_transform(X)

    enet = ElasticNet(alpha=1, l1_ratio=0.1, max_iter=10000, random_state=42).fit(feature_columns_scaled, y)

    # Convert grades to DataFrame with correct column names
    grades_df = pd.DataFrame(grades, columns=feature_cols)

    scaled_input = scaler.transform(grades_df)
    predicted = enet.predict(scaled_input)

    return predicted[0]  # Extract single predicted value

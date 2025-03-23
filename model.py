import pandas as pd  # Use 'pd' for consistency
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet
import numpy as np

def createModelandTest(grades):
    df = pd.read_csv("CombiDataset.csv")

    labs = grades.get('labs', [])
    assignments = grades.get('assignments', [])
    midterms = grades.get('midterms', [])
    combined_grades = labs + assignments + midterms

    print(combined_grades)

    feature_cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L7B', 'L8', 'A1', 'A2', 'A3', 'A4', 'A5', 'M']
    target_col = 'F'

    X = df[feature_cols]
    y = df[target_col]

    scaler = MinMaxScaler()
    feature_columns_scaled = scaler.fit_transform(X)

    enet1 = ElasticNet(alpha=1, l1_ratio=0.1, max_iter=10000, random_state=42).fit(feature_columns_scaled, y)
    enet2 = ElasticNet(alpha=1, l1_ratio=0.9, max_iter=10000, random_state=42).fit(feature_columns_scaled, y)
    enet3 = ElasticNet(alpha=1, l1_ratio=0.5, max_iter=10000, random_state=42).fit(feature_columns_scaled, y)
    enet4 = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000, random_state=42).fit(feature_columns_scaled, y)

    # Convert grades to DataFrame with correct column names
    grades_df = pd.DataFrame([combined_grades], columns=feature_cols)
    print(grades_df)


    scaled_input = scaler.transform(grades_df)

    print(scaled_input)

    predicted1 = enet1.predict(scaled_input)
    predicted2 = enet2.predict(scaled_input)
    predicted3 = enet3.predict(scaled_input)
    predicted4 = enet4.predict(scaled_input)

    averaged_prediction = np.mean(
    [predicted1, predicted2, predicted3, predicted4], axis=0
)
    return averaged_prediction[0]  # Extract single predicted value

def createModelandTestMidterm(grades, numLabs, numAssignments):
    df = pd.read_csv("CombiDataset.csv")

    if numLabs > 5:
        raise ValueError("Number of labs cannot exceed 5.")
    if numAssignments > 3:
        raise ValueError("Number of assignments cannot exceed 3.")

    # Define lab and assignment columns
    lab_cols = [f"L{i}" for i in range(1, 10)] + ["L7B"]  # Add L7B explicitly
    assignment_cols = [f"A{i}" for i in range(1, 6)]

    # Select the requested number of columns
    selected_labs = lab_cols[:numLabs]
    selected_assignments = assignment_cols[:numAssignments]

    # Combine the selected columns
    selected_cols = selected_labs + selected_assignments

    # Check if columns exist in the DataFrame and exclude any "M" column
    valid_cols = [col for col in selected_cols if col in df.columns and col != "M"]

    # Create the feature array
    feature_array = df[valid_cols].to_numpy()
    y = df['M']

    scaler = MinMaxScaler()
    feature_columns_scaled = scaler.fit_transform(feature_array)

    enet1 = ElasticNet(alpha=1, l1_ratio=0.1, max_iter=10000, random_state=42).fit(feature_columns_scaled, y)
    enet2 = ElasticNet(alpha=1, l1_ratio=0.9, max_iter=10000, random_state=42).fit(feature_columns_scaled, y)
    enet3 = ElasticNet(alpha=1, l1_ratio=0.5, max_iter=10000, random_state=42).fit(feature_columns_scaled, y)
    enet4 = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000, random_state=42).fit(feature_columns_scaled, y)

    proper_grades = grades['labs'] + grades['assignments']
    proper_grades_2d = [proper_grades]

    print(proper_grades_2d)

    grades_df = pd.DataFrame(proper_grades_2d,valid_cols)

    scaled_input = scaler.transform(grades_df)
       # Convert grades to DataFrame with correct column names
   

    predicted1 = enet1.predict(scaled_input)
    predicted2 = enet2.predict(scaled_input)
    predicted3 = enet3.predict(scaled_input)
    predicted4 = enet4.predict(scaled_input)

    averaged_prediction = np.mean(
    [predicted1, predicted2, predicted3, predicted4], axis=0
)
    return averaged_prediction[0]
    
    



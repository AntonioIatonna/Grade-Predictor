import pandas as pd  # Use 'pd' for consistency
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet
import numpy as np

def createModelandTest(grades,numLabs, numAssignments,numMidterms):
    df = pd.read_csv("FinalDataset.csv")

    if numLabs > 9:
        raise ValueError("Number of labs cannot exceed 9.")
    if numAssignments > 5:
        raise ValueError("Number of assignments cannot exceed 5.")
    if numMidterms > 1:
        raise ValueError("Number of assignments cannot exceed 1.")

 # Define lab and assignment columns
    lab_cols = [f"L{i}" for i in range(1, 10)] 
    assignment_cols = [f"A{i}" for i in range(1, 6)]
    midterm_cols = [f"M"]

    # Select the requested number of columns
    selected_labs = lab_cols[:numLabs]
    selected_assignments = assignment_cols[:numAssignments]
    selected_midterm= midterm_cols[:numMidterms]

    # Combine the selected columns
    selected_cols = selected_labs + selected_assignments + selected_midterm

    # Check if columns exist in the DataFrame and exclude any "M" column
    valid_cols = [col for col in selected_cols if col in df.columns]

    target_col = 'F'

    X = df[valid_cols]
    y = df[target_col]

    scaler = MinMaxScaler()
    feature_columns_scaled = scaler.fit_transform(X)

    enet1 = ElasticNet(alpha=1, l1_ratio=0.1, max_iter=10000, random_state=42).fit(feature_columns_scaled, y)
    enet2 = ElasticNet(alpha=1, l1_ratio=0.9, max_iter=10000, random_state=42).fit(feature_columns_scaled, y)
    enet3 = ElasticNet(alpha=1, l1_ratio=0.5, max_iter=10000, random_state=42).fit(feature_columns_scaled, y)
    enet4 = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000, random_state=42).fit(feature_columns_scaled, y)

    # Convert grades to DataFrame with correct column names
    proper_grades = grades['labs'] + grades['assignments'] + grades['midterms']
    proper_grades_2d = [proper_grades]
    grades_df = pd.DataFrame(proper_grades_2d, columns=valid_cols)
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
    lab_cols = [f"L{i}" for i in range(1, 10)] 
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
    
    



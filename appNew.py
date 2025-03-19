import streamlit as st
import model as model

st.title('Grade Predictor')
st.write("Hello, welcome to the Grade Predictor App. Please enter the required details and we will predict your grade.")

# Inputs for the number of labs, assignments, and midterms
num_labs = st.number_input("How many labs are in your course?", min_value=1, step=1)
num_assignments = st.number_input("How many assignments are in your course?", min_value=1, step=1)
num_midterms = st.number_input("How many midterms are in your course?", min_value=1, step=1)

if st.button("Submit"):
    # Store the user's input in the session state to persist the data
    st.session_state["num_labs"] = num_labs
    st.session_state["num_assignments"] = num_assignments
    st.session_state["num_midterms"] = num_midterms

# Check if the number of inputs have been defined
if "num_labs" in st.session_state and "num_assignments" in st.session_state and "num_midterms" in st.session_state:
    st.header("Enter your grades")

    lab_grades = []
    for i in range(1, st.session_state["num_labs"] + 1):
        lab_grades.append(st.number_input(f"Lab {i}", min_value=0, max_value=100, step=1, key=f"lab_{i}"))

    assignment_grades = []
    for i in range(1, st.session_state["num_assignments"] + 1):
        assignment_grades.append(st.number_input(f"Assignment {i}", min_value=0, max_value=100, step=1, key=f"assignment_{i}"))

    # Switch for selecting prediction type
    predict_midterm = st.toggle("Predict Midterm", value=False)

    midterm_grades = []
    if not predict_midterm:
        for i in range(1, st.session_state["num_midterms"] + 1):
            midterm_grades.append(st.number_input(f"Midterm {i}", min_value=0, max_value=100, step=1, key=f"midterm_{i}"))

    # Collecting all inputs
    grades = {
        "labs": lab_grades,
        "assignments": assignment_grades,
        "midterms": midterm_grades if not predict_midterm else []
    }

    # Prediction function
    def predict_grade(grades, predict_midterm):
        # Flatten the dictionary values into a single list
        
        if predict_midterm:
            input_features = grades["labs"] + grades["assignments"]
        else:
            input_features = grades["labs"] + grades["assignments"] + grades["midterms"]
        input_array = [input_features]  # Convert to 2D array

        if predict_midterm:
            return model.createModelandTestMidterm(input_array)
        else:
            return model.createModelandTest(input_array)

    # Button to submit grades and predict
    if st.button("Predict Grade", key="predict_button"):
        predicted_grade = predict_grade(grades, predict_midterm)
        st.success(f"Your predicted grade is: {predicted_grade}")

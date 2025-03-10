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

    midterm_grades = []
    for i in range(1, st.session_state["num_midterms"] + 1):
        midterm_grades.append(st.number_input(f"Midterm {i}", min_value=0, max_value=100, step=1, key=f"midterm_{i}"))

    # Collecting all inputs
    grades = {
        "labs": lab_grades,
        "assignments": assignment_grades,
        "midterms": midterm_grades
    }

    # Placeholder function for predict_grade, replace with your own model's prediction function
    def predict_grade(grades):
        # Flatten the dictionary values into a single list
        input_features = grades["labs"] + grades["assignments"] + grades["midterms"]

        # Ensure it's a 2D array
        input_array = [input_features]  # Wrapping in a list makes it a 2D array with one sample

        result = model.createModelandTest(input_array)
        return result  # Extract the single predicted value

    # Button to submit grades and predict
    if st.button("Predict Grade", key="predict_button"):
        # Assuming predict_grade is a function that returns the prediction
        predicted_grade = predict_grade(grades)  # You'll need to define the predict_grade function
        st.success(f"Your predicted grade is: {predicted_grade}")

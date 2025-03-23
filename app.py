import streamlit as st
import model as model

st.title('Grade Predictor')
st.write("Hello, welcome to the Grade Predictor App. Please enter the required details and we will predict your grade.")

# Initialize session state
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False
if "choice" not in st.session_state:
    st.session_state["choice"] = None
if "predicted_grade" not in st.session_state:
    st.session_state["predicted_grade"] = None

# Only show the radio buttons if the form hasn't been submitted
if not st.session_state["submitted"]:
    st.session_state["choice"] = st.radio("", ["Predict Midterm", "Predict Final"], index=1)

# If user wants to predict midterm grade
if st.session_state["choice"] == "Predict Midterm":
    # Only show the initial inputs if the form hasn't been submitted
    if not st.session_state["submitted"]:
        num_labs = st.number_input("How many labs have been graded so far?", min_value=0, step=1)
        num_assignments = st.number_input("How many have been graded so far?", min_value=0, step=1)
        num_midterms = st.number_input("How many midterms have been graded so far?", min_value=0, step=1)

        # When user submits the form
        if st.button("Submit"):
            # Store the user's input in session state
            st.session_state["num_labs"] = num_labs
            st.session_state["num_assignments"] = num_assignments
            st.session_state["num_midterms"] = num_midterms
            st.session_state["submitted"] = True  # Mark as submitted to hide inputs
            st.rerun()  # Refresh the app immediately

    # After submission, show grade input fields
    if st.session_state["submitted"]:
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

        # Placeholder function for predict_grade ***** ALEX STARTING CHANGE HERE ********
        def predict_grade(grades):
            return model.createModelandTestMidterm(grades,st.session_state["num_labs"],st.session_state["num_assignments"])

        # Button to submit grades and predict
        if st.button("Predict Grade", key="predict_button"):
            st.session_state["predicted_grade"] = predict_grade(grades)
            st.rerun()  # Refresh to show the prediction

        # **** ALEX ENDING CHANGE HERE ********

# If user wants to predict final grade
elif st.session_state["choice"] == "Predict Final":
    # Only show the initial inputs if the form hasn't been submitted
    if not st.session_state["submitted"]:
        num_labs = st.number_input("How many labs are in your course?", min_value=1, step=1)
        num_assignments = st.number_input("How many assignments are in your course?", min_value=1, step=1)
        num_midterms = st.number_input("How many midterms are in your course?", min_value=1, step=1)

        # When user submits the form
        if st.button("Submit"):
            # Store the user's input in session state
            st.session_state["num_labs"] = num_labs
            st.session_state["num_assignments"] = num_assignments
            st.session_state["num_midterms"] = num_midterms
            st.session_state["submitted"] = True  # Mark as submitted to hide inputs
            st.rerun()  # Refresh the app immediately

    # After submission, show grade input fields
    if st.session_state["submitted"]:
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

        # Placeholder function for predict_grade
        def predict_grade(grades):
            return model.createModelandTest(grades)

        # Button to submit grades and predict
        if st.button("Predict Grade", key="predict_button"):
            st.session_state["predicted_grade"] = predict_grade(grades)
            st.rerun()  # Refresh to show the prediction

# Show predicted grade and reset button
if st.session_state["predicted_grade"] is not None:
    st.success(f"Your predicted grade is: {st.session_state['predicted_grade']}")

    # Reset button
    if st.button("Reset"):
        # Clear session state and restart the app
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

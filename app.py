import streamlit as st

st.title('Grade Predictor')
st.write("Hello, welcome to the Grade Predictor App. Please enter the required details and we will predict your grade.")

# Inputs for user grades
st.header("Enter your grades")
lab_grades = []
for i in range(1, 10):
    lab_grades.append(st.number_input(f"Lab {i}", min_value=0, max_value=100, step=1))

assignment_grades = []
for i in range(1, 6):
    assignment_grades.append(st.number_input(f"Assignment {i}", min_value=0, max_value=100, step=1))

midterm_grade = st.number_input("Midterm", min_value=0, max_value=100, step=1)

# Collecting all inputs
grades = {
    "labs": lab_grades,
    "assignments": assignment_grades,
    "midterm": midterm_grade
}

# Button to submit grades and predict
if st.button("Predict Grade"):
    # Assuming predict_grade is a function that returns the prediction
    predicted_grade = predict_grade(grades)  # You'll need to define the predict_grade function
    st.success(f"Your predicted grade is: {predicted_grade}")

# Placeholder function for predict_grade, replace with your own model's prediction function
def predict_grade(grades):
    # For demonstration, let's assume a simple average
    total_grades = grades["labs"] + grades["assignments"] + [grades["midterm"]]
    return sum(total_grades) / len(total_grades)

if __name__ == '__main__':
    st.run()
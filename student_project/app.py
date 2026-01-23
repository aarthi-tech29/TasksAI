# ==============================
# IMPORT REQUIRED LIBRARIES
# ==============================

import streamlit as st          # Streamlit → used to create web app UI
import pandas as pd             # Pandas → used to create and handle data tables
import numpy as np              # NumPy → used for numerical arrays
from sklearn.ensemble import RandomForestClassifier   # ML model
from sklearn.preprocessing import LabelEncoder        # Convert text labels to numbers


# ==============================
# CREATE SAMPLE DATASET
# ==============================
# Each row represents one student

data = {
    "study_hours": [2, 4, 6, 8, 3],            # Hours studied per day
    "attendance": [60, 75, 85, 90, 70],        # Attendance percentage
    "previous_score": [45, 65, 78, 88, 55],    # Previous exam marks
    "assignments_completed": [3, 5, 7, 9, 4],  # Number of assignments done
    "performance": ["Low", "Medium", "High", "High", "Medium"]  # final result (what we predict)
    # Target column (what we want to predict)
}

# Convert dictionary to DataFrame (table)
df = pd.DataFrame(data)


# ==============================
# SPLIT INPUT (X) AND OUTPUT (y)
# ==============================

X = df.drop("performance", axis=1)
# X contains only input features

y = df["performance"]
# y contains output labels (Low / Medium / High)

# Inputs → Model → Output
# X → inputs (study hours, attendance, etc.)
# y → output (Low / Medium / High)


# ==============================
# CONVERT TEXT LABELS TO NUMBERS
# ==============================

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Example:
# Low → 1
# Medium → 2
# High → 0
# (Exact numbers don’t matter, ML understands numbers)


# ==============================
# TRAIN MACHINE LEARNING MODEL
# ==============================

model = RandomForestClassifier()
model.fit(X, y_encoded)

# Random Forest
# A collection of decision trees
# Each tree makes a decision
# Final result = majority vote

# fit() → model learns patterns from data

# The model finds patterns like:
# More study hours → higher performance
# Low attendance → lower performance


# ==============================
# STREAMLIT USER INTERFACE
# ==============================

st.title("Student Performance Prediction System")
st.write("Enter student details below to predict performance")

# Shows heading text on the webpage
# st.write() shows normal text

# Input fields shown on the webpage
study_hours = st.number_input(
    "Study Hours per Day", 
    min_value=0.0, 
    max_value=24.0
)

attendance = st.slider(
    "Attendance Percentage", 
    min_value=0, 
    max_value=100
)

previous_score = st.number_input(
    "Previous Exam Score", 
    min_value=0, 
    max_value=100
)

assignments = st.number_input(
    "Assignments Completed", 
    min_value=0, 
    max_value=20
)


# ==============================
# PREDICTION BUTTON LOGIC
# ==============================

if st.button("Predict Performance"):
    
    # Combine user inputs into one array
    input_data = np.array([
        [study_hours, attendance, previous_score, assignments]
    ])
    
    # Preparing Input for Model

    # Predict using trained model
    prediction = model.predict(input_data)
    # Model uses learned patterns
    # Predicts a number (0 / 1 / 2)
    
    # Convert number back to text label
    result = le.inverse_transform(prediction)
    # 2-> Medium
    
    # Display result on screen
    st.success(f"Predicted Performance: {result[0]}")


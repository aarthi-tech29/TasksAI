# Input → Age, BP, Glucose, BMI, Cholesterol
# Output → Disease type
# 0 → Healthy
# 1 → Diabetes
# 2 → Heart Disease
# 3 → Liver Disease
# This is a Multi-Class Classification problem (more than 2 classes)

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Create / Load Medical Dataset
data = {
    "Age": [25, 45, 50, 35, 60, 55, 40, 30],
    "BloodPressure": [120, 140, 150, 130, 160, 155, 135, 125],
    "Glucose": [90, 180, 200, 100, 210, 195, 110, 95],
    "BMI": [22, 30, 32, 25, 35, 34, 26, 23],
    "Cholesterol": [180, 220, 240, 200, 260, 250, 210, 190],
    "Disease": [0, 1, 1, 0, 2, 2, 0, 0]
}

df = pd.DataFrame(data)
print(df)

# Step 3: Feature Correlation Analysis
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()
# Shows relationships like:
# High glucose → diabetes
# High BP + cholesterol → heart disease

# Step 4: Split Features & Target
X = df.drop("Disease", axis=1)
y = df["Disease"]
# X → input features
# y → target labels

# Step 5: Normalize the Data
#  Medical values have different scales (BP, glucose, BMI)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# Step 7: Train Multi-Class Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Random Forest:
# Handles multi-class well
# Works with medical tabular data
# Reduces overfitting

# Step 8: Predictions & Probability Prediction
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
# predict() → disease class
# predict_proba() → probability of each disease

# Step 9: Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred)) # Out of all predictions, how many were correct? Accuracy = (Correct Predictions) / (Total Predictions)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred)) # How many samples were predicted correctly and incorrectly for each disease class

# 1 → predicted correctly for this class
# 0 → predicted incorrectly for this class
# 1 patient correctly diagnosed
# 0 patients misdiagnosed
# OR
# 1 patient misdiagnosed
# 0 patients correctly diagnosed

print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# precision-> When model predicts a disease, how often is it correct?
# recall-> Of all actual disease cases, how many did the model detect?
# F1-score-> Balance between precision and recall
# Support-> Number of actual occurrences of each disease in test data

# Step 10: Predict Disease for New Patient

new_patient_df = pd.DataFrame(
    [[50, 145, 190, 31, 235]],
    columns=X.columns
)

new_patient_scaled = scaler.transform(new_patient_df)


prediction = model.predict(new_patient_scaled)
probability = model.predict_proba(new_patient_scaled)

print("Predicted Disease:", prediction[0])
print("Prediction Probabilities:", probability)

# Disease 0 → 21%
# Disease 1 → 64% (highest)
# Disease 2 → 15%
# So model confidently chose Disease 1




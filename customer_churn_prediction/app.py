# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
# For example, using a CSV file named 'customer_churn.csv'
data = pd.read_csv('customer_churn.csv')

# Display first 5 rows
print("Dataset Head:")
print(data.head())

# 3. Exploratory Data Analysis (EDA)
print("\nDataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

# Visualize churn distribution
sns.countplot(x='Churn', data=data)
plt.title("Churn Distribution")
plt.show()

# 4. Data Preprocessing
# Encode categorical variables
label_encoders = {}
categorical_cols = data.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separate features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5. Feature Importance Analysis
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importance")
plt.show()

# 6. Model Training and Evaluation
# Using Random Forest as example
y_pred = rf_model.predict(X_test)

print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Prediction on New Data
# Example: Predict churn for a new customer
new_customer = pd.DataFrame({
    'CustomerID': [9],  # new ID
    'Gender': ['Female'],
    'Age': [29],
    'Tenure': [3],
    'Balance': [50000],
    'NumOfProducts': [1],
    'HasCrCard': [1],
    'IsActiveMember': [1],
    'EstimatedSalary': [52000]
})

# Preprocess new data
for col in categorical_cols:
    if col in new_customer.columns:
        new_customer[col] = label_encoders[col].transform(new_customer[col])

new_customer_scaled = scaler.transform(new_customer)
churn_prediction = rf_model.predict(new_customer_scaled)
print("\nChurn Prediction for new customer:", "Yes" if churn_prediction[0]==1 else "No")




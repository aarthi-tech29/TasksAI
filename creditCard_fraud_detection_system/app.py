# =====================================================
# Credit Card Fraud Detection System 
# =====================================================

# -----------------------------
# STEP 1: Import Required Libraries
# -----------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Pandas → for data handling
# Scikit-learn → for ML models
# Imbalanced-learn (SMOTE) → for handling imbalanced data
# Matplotlib & Seaborn → for visualization

# -----------------------------
# STEP 2: Load Dataset
# -----------------------------
data = pd.read_csv("creditcard.csv")
print("First 5 rows of dataset:")
print(data.head())
print("\nDataset info:")
print(data.info())

# Check first 5 rows and data info to see missing values and data types.

# -----------------------------
# STEP 3: Understand Class Distribution
# -----------------------------
print("\nClass distribution:")
print(data['Class'].value_counts())

# Visualize class imbalance
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.show()

# Check for imbalance (fraud cases are rare)
# Class = 0 → Normal
# Class = 1 → Fraud
# Fraud cases are usually <1% → imbalanced dataset.

# -----------------------------
# STEP 4: Preprocess Data
# -----------------------------
# Scale 'Amount' column
data['NormalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

# Drop unnecessary columns
data = data.drop(['Time', 'Amount'], axis=1)

# Split features and target
X = data.drop('Class', axis=1)
y = data['Class']

# -----------------------------
# STEP 5: Handle Imbalanced Data (SMOTE)
# -----------------------------
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print("\nOriginal dataset shape:", y.value_counts())
print("Resampled dataset shape:", pd.Series(y_res).value_counts())

# SMOTE generates synthetic examples for the minority class

# -----------------------------
# STEP 6: Split Data into Train & Test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# -----------------------------
# STEP 7: Train Random Forest Classifier
# -----------------------------
model = RandomForestClassifier(
    n_estimators=20,     # MUCH faster
    n_jobs=-1,           # use all CPU cores
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# STEP 8: Make Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# STEP 9: Evaluate Model
# -----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# [[TN   FP]
#  [FN   TP]]
# TN	Normal correctly predicted as Normal	56,736
# FP	Normal wrongly predicted as Fraud	14
# FN	Fraud wrongly predicted as Normal	0
# TP	Fraud correctly detected as Fraud	56,976

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Precision 1.00 → Every predicted fraud was actually fraud
# Recall 1.00 → Every real fraud was detected
# F1-score 1.00 → Perfect balance
# Total test samples = 113,726
# Overall performance = perfect
# Model performs equally well on both classes
# No bias toward majority or minority class

# -----------------------------
# STEP 10: Save Model for Deployment
# -----------------------------
joblib.dump(model, "credit_card_fraud_model.pkl")
print("\nModel saved as 'credit_card_fraud_model.pkl'")

# -----------------------------
# STEP 11: Predict New Transactions (Optional)
# -----------------------------
sample_pred = model.predict(X_test.iloc[:5])
print("\nSample Predictions for first 5 test samples:", sample_pred)
# 1 → Fraud
# 0 → Normal


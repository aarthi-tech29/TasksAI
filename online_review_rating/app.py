# =====================================================
# Online Review Rating Prediction System
# =====================================================

# -----------------------------
# STEP 1: Import Libraries
# -----------------------------
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# STEP 2: Create a Sample Dataset
# -----------------------------
# For demonstration, 50 reviews with ratings 1-5
data = pd.DataFrame({
    "Review": [
        "Excellent product, very happy!", "Terrible service, very disappointed.",
        "Average quality, nothing special.", "Loved it, highly recommend!", 
        "Poor packaging, broke on arrival.", "Fantastic experience, will buy again.",
        "Not worth the price.", "Pretty satisfied with the purchase.",
        "Worst product I have ever bought.", "Good quality but delivery was slow.",
        "Amazing! Works perfectly.", "Mediocre, expected better.", 
        "Excellent quality, five stars.", "Bad quality, one star.", 
        "Not bad, but could be better.", "Absolutely love it!", 
        "Terrible experience, do not buy.", "Met expectations, okay product.", 
        "Highly recommend this item!", "Awful, arrived damaged.",
        "Great product for the price.", "Disappointing, low quality.", 
        "Very happy with my purchase.", "Not satisfied at all.", 
        "Excellent and fast delivery.", "Poor material, cheap product.", 
        "Good but not excellent.", "Amazing service and product!", 
        "Broke after first use.", "Decent for the price.", 
        "Five stars, perfect!", "One star, very bad.", 
        "Works as expected.", "Terrible, won’t buy again.", 
        "Loved it, very good quality.", "Not worth the money.", 
        "Great value, excellent!", "Bad experience, never again.", 
        "Average, nothing special.", "Absolutely fantastic!", 
        "Poor quality, disappointed.", "Satisfied with the product.", 
        "Excellent item, very happy!", "Worst purchase ever.", 
        "Good quality, will buy again.", "Terrible product, broken.", 
        "Amazing quality, highly recommend.", "Not satisfied, expected better.", 
        "Fantastic product, works perfectly.", "One star, awful!"
    ],
    "Rating": [
        5,1,3,5,2,5,2,4,1,4,5,3,5,1,3,5,1,3,5,1,
        4,2,5,1,5,2,3,5,2,3,5,1,4,1,5,2,5,1,3,5,
        2,4,5,1,4,1,5,2,5,1
    ]
})

# -----------------------------
# STEP 3: TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(data['Review'])
y = data['Rating']

# -----------------------------
# STEP 4: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# STEP 5: Train Naive Bayes Classifier
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# STEP 6: Make Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# STEP 7: Evaluate Model
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print("\nAccuracy:", round(accuracy, 2))
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# -----------------------------
# STEP 8: Predict New Reviews
# -----------------------------
new_reviews = [
    "Excellent product, very happy with my purchase!",
    "Terrible service, I am very disappointed.",
    "Average quality, nothing special."
]

new_vectors = vectorizer.transform(new_reviews)
predicted_ratings = model.predict(new_vectors)

for review, rating in zip(new_reviews, predicted_ratings):
    print(f"\nReview: {review}\nPredicted Rating: {rating}")

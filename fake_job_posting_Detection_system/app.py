# =======================Streamlit Fake Job Posting Detection System=======================
# =====================================
# IMPORT REQUIRED LIBRARIES
# =====================================

import streamlit as st              # Web app UI
import pandas as pd                 # Data handling
import numpy as np                  # Numerical operations
import nltk                         # NLP library
import re                           # Text cleaning
from nltk.corpus import stopwords   # Remove common words
from sklearn.feature_extraction.text import TfidfVectorizer # feature extraction
from sklearn.linear_model import LogisticRegression # classification


# =====================================
# DOWNLOAD NLTK DATA (only first time)
# =====================================

nltk.download('stopwords')


# =====================================
# CREATE SAMPLE DATASET
# =====================================
# 1 = Fake Job
# 0 = Real Job

data = {
    "job_description": [
        "Earn money from home no experience needed click now",
        "Urgent hiring work from home daily payment",
        "Software engineer required with Python experience",
        "Hiring data analyst with SQL and Power BI skills",
        "No interview no skills easy money",
        "Looking for experienced web developer full time job"
    ],
    "label": [1, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)


# =====================================
# TEXT PREPROCESSING FUNCTION
# =====================================

def clean_text(text):
    text = text.lower()                          # Convert to lowercase
    text = re.sub(r'[^a-z ]', '', text)          # Remove symbols & numbers
    words = text.split()                         # Split into words
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)


# Apply cleaning to dataset
df["cleaned_text"] = df["job_description"].apply(clean_text)


# =====================================
# TF-IDF FEATURE EXTRACTION
# =====================================

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned_text"])
y = df["label"]

# TF-IDF converts text into numbers based on:
# Word importance
# Frequency
# Term Frequency – Inverse Document Frequency
# Term Frequency (TF): How often a word appears in a document
# Inverse Document Frequency (IDF): How unique/rare a word is across all documents
# TF-IDF = TF * IDF-Words that appear many times in one document, BUT appear in very few documents overall.This helps the model focus on meaningful words.

# =====================================
# TRAIN MACHINE LEARNING MODEL
# =====================================

model = LogisticRegression()
model.fit(X, y)

# Fake jobs repeat words like
# “easy money”, “no experience”, “work from home”

# Real jobs use words like
# “skills”, “experience”, “qualification”


# =====================================
# STREAMLIT USER INTERFACE
# =====================================

st.title("Fake Job Posting Detection System")
st.write("Paste a job description below to check if it is fake or real.")

user_input = st.text_area("Job Description")


# =====================================
# PREDICTION LOGIC
# =====================================

if st.button("Check Job"):
    if user_input.strip() == "":
        st.warning("Please enter a job description.")
    else:
        cleaned_input = clean_text(user_input)
        vector_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vector_input)

        if prediction[0] == 1:
            st.error("Fake Job Posting Detected")
        else:
            st.success("This Job Posting Looks Genuine")

# =====================================

#===========================Console Fake Job Posting Detection System===========================
# =====================================
# IMPORT REQUIRED LIBRARIES
# =====================================

import pandas as pd # Data handling
import re  # Text cleaning
import nltk  # NLP library
from nltk.corpus import stopwords # Remove common words
from sklearn.feature_extraction.text import TfidfVectorizer # feature extraction
from sklearn.linear_model import LogisticRegression # classification

# =====================================
# DOWNLOAD NLTK DATA (only first time)
# =====================================

nltk.download('stopwords')

# =====================================
# CREATE SAMPLE DATASET
# =====================================
# 1 = Fake Job
# 0 = Real Job
data = {
    "job_description": [
        "Earn money from home no experience needed click now",
        "Urgent hiring work from home daily payment",
        "Software engineer required with Python experience",
        "Hiring data analyst with SQL and Power BI skills",
        "No interview no skills easy money",
        "Looking for experienced web developer full time job"
    ],
    "label": [1, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

# =====================================
# TEXT PREPROCESSING FUNCTION
# =====================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)  # Remove symbols & numbers
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

df["cleaned_text"] = df["job_description"].apply(clean_text)

# =====================================
# TF-IDF FEATURE EXTRACTION
# =====================================

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned_text"])
y = df["label"]

# TF-IDF converts text into numbers based on:
# Word importance
# Frequency
# Term Frequency – Inverse Document Frequency
# Term Frequency (TF): How often a word appears in a document
# Inverse Document Frequency (IDF): How unique/rare a word is across all documents
# TF-IDF = TF * IDF-Words that appear many times in one document, BUT appear in very few documents overall.This helps the model focus on meaningful words.


# =====================================
# TRAIN MACHINE LEARNING MODEL
# =====================================

model = LogisticRegression()
model.fit(X, y)

# Fake jobs repeat words like
# “easy money”, “no experience”, “work from home”

# Real jobs use words like
# “skills”, “experience”, “qualification”

# =====================================
# CONSOLE INTERFACE
# =====================================

while True:
    user_input = input("\nEnter a job description (or 'exit' to quit):\n> ")
    if user_input.lower() == "exit":
        print("Exiting program.")
        break
    elif user_input.strip() == "":
        print("Please enter a valid job description.")
        continue

    cleaned_input = clean_text(user_input)
    vector_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vector_input)

    if prediction[0] == 1:
        print("Fake Job Posting Detected")
    else:
        print("This Job Posting Looks Genuine")


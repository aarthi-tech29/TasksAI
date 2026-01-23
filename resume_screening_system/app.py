# ========================================================================================
import os
import PyPDF2
import nltk
import string
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import pandas as pd

nltk.download('stopwords')

# ---------- FUNCTIONS ----------

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# ---------- JOB DESCRIPTION LINES ----------
job_description_lines = [
    "We are looking for a software engineer with experience in Python, Django, and machine learning.",
    "Familiarity with REST APIs, databases, and version control systems like Git is required.",
    "The candidate should have strong problem-solving and communication skills."
]

# Combine all lines into a single string and clean it
job_desc_text = " ".join(job_description_lines)
job_desc = clean_text(job_desc_text)

# ---------- LOAD RESUMES ----------
resumes = []
resume_names = []

for file in os.listdir("resumes"):
    if file.endswith(".pdf"):
        path = os.path.join("resumes", file)
        text = extract_text_from_pdf(path)
        resumes.append(clean_text(text))
        resume_names.append(file)

# ---------- EMBEDDINGS & SIMILARITY ----------
model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight & fast

# Create embeddings
job_emb = model.encode(job_desc, convert_to_tensor=True)
resume_embs = model.encode(resumes, convert_to_tensor=True)

# Compute cosine similarity
scores = util.cos_sim(job_emb, resume_embs)[0]  # tensor

# Convert to percentages
scores_percent = (scores.cpu().numpy() * 100).round(2)

# ---------- RESULT ----------
result = pd.DataFrame({
    "Resume": resume_names,
    "Match Percentage": scores_percent
}).sort_values(by="Match Percentage", ascending=False)

print(result)

# ========================================================================================
# STEP 1: Import Required Libraries
import os
import string
import nltk
import PyPDF2
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# STEP 2: Download Stopwords
nltk.download('stopwords')

# STEP 3: Write Job Description
job_description = """
Looking for a Python developer with experience in Django,
REST APIs, machine learning, NLP, and database management.
"""
# STEP 4: Function to Read Resume PDFs
# PDF → Text conversion needed before AI can read it
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# STEP 5: Clean Text Using NLP
# Lowercase
# Remove punctuation
# Remove stopwords
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# STEP 6: Read & Process All Resumes
resume_texts = []
resume_names = []

for file in os.listdir("resumes"):
    if file.endswith(".pdf"):
        file_path = os.path.join("resumes", file)
        text = extract_text_from_pdf(file_path)
        cleaned_text = clean_text(text)
        resume_texts.append(cleaned_text)
        resume_names.append(file)

# STEP 7: Prepare Data for AI Comparison
clean_job_description = clean_text(job_description)

documents = [clean_job_description] + resume_texts

# Job description must also be cleaned
# First item = job description
# Remaining = resumes

# STEP 8: Convert Text → Numbers (TF-IDF)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# AI understands numbers, not words

# STEP 9: Calculate Similarity (Cosine Similarity)
similarity_scores = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:]
)[0]

# Output = match score between job & each resume

# STEP 10: Show Result in Table Format
results = pd.DataFrame({
    "Resume Name": resume_names,
    "Match Percentage": similarity_scores * 100
})

results = results.sort_values(by="Match Percentage", ascending=False)

print("\nResume Screening Results:\n")
print(results)





from fpdf import FPDF
import os

# Make sure the resumes folder exists
if not os.path.exists("resumes"):
    os.makedirs("resumes")

# Sample resume texts
resume_texts = {
    "resume1.pdf": """John Doe
Software Engineer
Skills: Python, Java, SQL, Machine Learning
Experience: 3 years at TechCorp
Education: B.Tech in Computer Science""",
    
    "resume2.pdf": """Jane Smith
Data Analyst
Skills: Excel, Python, Tableau, SQL
Experience: 2 years at DataSolutions
Education: B.Sc in Statistics"""
}

# Generate PDFs
for filename, text in resume_texts.items():
    pdf_path = os.path.join("resumes", filename)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output(pdf_path)

print("resume1.pdf and resume2.pdf created in the resumes folder!")

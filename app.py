import os
from flask import Flask, render_template, request, redirect, url_for, flash
import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import shutil

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def preprocess_text(text):
    doc = nlp(text)
    processed_text = " ".join(token.text.lower() for token in doc if not token.is_punct and not token.is_space)
    return processed_text

def compute_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return similarity[0][0] * 100

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Clear the uploads directory before each use
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'])

        # Check if a job description and multiple resumes are uploaded
        if 'job_description' not in request.files or 'resumes' not in request.files:
            flash('Please upload both a job description and resumes')
            return redirect(request.url)

        job_description_file = request.files['job_description']
        resumes_files = request.files.getlist('resumes')

        # Save job description
        job_description_path = os.path.join(app.config['UPLOAD_FOLDER'], job_description_file.filename)
        job_description_file.save(job_description_path)

        # Process job description
        job_description_text = extract_text_from_pdf(job_description_path)
        processed_job_description = preprocess_text(job_description_text)

        # Process and compare each resume
        similarity_results = []
        for resume_file in resumes_files:
            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(resume_path)
            
            # Extract and preprocess resume text
            resume_text = extract_text_from_pdf(resume_path)
            processed_resume_text = preprocess_text(resume_text)

            # Compute cosine similarity
            similarity = compute_cosine_similarity(processed_resume_text, processed_job_description)
            similarity_results.append((resume_file.filename, similarity))

        # Sort resumes by similarity in decreasing order
        sorted_results = sorted(similarity_results, key=lambda x: x[1], reverse=True)

        return render_template('results.html', results=sorted_results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


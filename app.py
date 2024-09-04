import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import shutil
from dotenv import load_dotenv
load_dotenv()
# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Use PostgreSQL instead of SQLite or MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('STRING')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# User model for the database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Create the database tables
with app.app_context():
    db.create_all()

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Preprocess text for similarity comparison
def preprocess_text(text):
    doc = nlp(text)
    processed_text = " ".join(token.text.lower() for token in doc if not token.is_punct and not token.is_space)
    return processed_text

# Compute cosine similarity between two texts
def compute_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return similarity[0][0] * 100

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if username already exists
        user_exists = User.query.filter_by(username=username).first()
        if user_exists:
            flash('Username already exists')
            return redirect(url_for('signup'))

        # Hash the password and save the user in the database
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Signup successful! Please login.')
        return redirect(url_for('login'))

    return render_template('signup.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Fetch the user from the database
        user = User.query.filter_by(username=username).first()

        # Check if user exists and if the password is correct
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!')
            return render_template('index.html')
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))

    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.')
    return redirect(url_for('home'))

# Upload files and perform resume matching
@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    if 'user_id' not in session:
        flash('You need to login first.')
        return redirect(url_for('login'))

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

    return render_template('upload.html')

# Main function to run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)

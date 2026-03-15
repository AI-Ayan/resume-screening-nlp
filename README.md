# AI Resume Screening System

This project is an AI-powered resume screening tool built using Python, NLP, and Streamlit.  
It helps recruiters quickly rank resumes based on how well they match a job description.

## Features
- Upload multiple resumes (PDF or TXT)
- Enter a job description
- Uses TF-IDF vectorization and cosine similarity
- Ranks resumes based on match score
- Displays results in a table and chart

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas
- PyPDF2

## How It Works
1. The recruiter enters a job description.
2. Multiple resumes are uploaded.
3. The system converts text to TF-IDF vectors.
4. Cosine similarity measures how closely each resume matches the job description.
5. Resumes are ranked based on similarity score.

## How to Run the Project

Clone the repository:

git clone https://github.com/AI-Ayan/resume-screening-nlp.git

Go to the project folder:

cd resume-screening-nlp

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py

Then open in browser:

http://localhost:8501

## Project Structure

resume-screening-nlp
│
├── app.py
├── requirements.txt
├── README.md
└── resumes
    ├── resume1.txt
    ├── resume2.txt
    └── resume3.txt
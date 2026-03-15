# AI Resume Screening System

This project is an AI-powered resume screening tool built using Python, NLP, and Streamlit.  
It helps recruiters quickly rank resumes based on how well they match a job description.

## Live Demo
https://resume-screening-nlp-2u6hzwik9dhwxydr2bz5j4.streamlit.app/

## Features

- Upload multiple resumes (PDF or TXT)
- Enter a job description
- TF-IDF based resume ranking
- Cosine similarity scoring
- Top candidate detection
- Resume preview
- Keyword matching
- Missing skill detection
- Match score visualization
- Resume ranking table

## Technologies Used

Python  
Streamlit  
Scikit-learn  
Pandas  
PyPDF2  

## How It Works

1. The recruiter enters a job description.
2. Multiple resumes are uploaded.
3. Text is converted into TF-IDF vectors.
4. Cosine similarity calculates how closely resumes match the job description.
5. The system ranks candidates and shows match scores.

## How to Run the Project

Clone the repository:

git clone https://github.com/AI-Ayan/resume-screening-nlp.git

Go to the project folder:

cd resume-screening-nlp

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

Open in browser:

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
import streamlit as st
import pandas as pd
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.title("AI Resume Screening System")

job_description = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF or TXT)", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)


def extract_text(file):

    if file.type == "application/pdf":

        pdf_reader = PyPDF2.PdfReader(file)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        return text

    else:
        return file.read().decode("utf-8")


if st.button("Screen Resumes"):

    resumes = []
    resume_names = []

    for file in uploaded_files:

        text = extract_text(file)

        resumes.append(text)
        resume_names.append(file.name)

    documents = resumes + [job_description]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    scores = similarity.flatten()

    results = pd.DataFrame({
        "Resume": resume_names,
        "Match Score": scores
    })

    results = results.sort_values(by="Match Score", ascending=False)

    st.subheader("Resume Ranking")

    st.dataframe(results)

    st.subheader("Match Score Chart")

    st.bar_chart(results.set_index("Resume"))
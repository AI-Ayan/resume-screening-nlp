import streamlit as st
import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.title("AI Resume Screening System")

job_description = st.text_area("Enter Job Description")


if st.button("Screen Resumes"):

    resume_folder = "resumes"

    resumes = []
    resume_names = []

    for file in os.listdir(resume_folder):
        with open(os.path.join(resume_folder, file), "r") as f:
            resumes.append(f.read())
            resume_names.append(file)

    documents = resumes + [job_description]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    scores = similarity.flatten()

    results = pd.DataFrame({
        "Resume": resume_names,
        "Score": scores
    })

    results = results.sort_values(by="Score", ascending=False)

    st.subheader("Resume Ranking")

    st.dataframe(results)
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

    if not job_description.strip():
        st.error("Please enter a job description.")
        st.stop()

    if not uploaded_files:
        st.error("Please upload at least one resume.")
        st.stop()

    st.success("Resumes successfully analyzed! Here are the results:")

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
        "Match Score": scores * 100
    })

    results = results.sort_values(by="Match Score", ascending=False)
    results["Match Score"] = results["Match Score"].round(2)

    top_candidates = results.head(3)

    st.subheader("🏆 Top Candidates")

    for i, row in top_candidates.iterrows():
        st.write(f"{row['Resume']} — Match Score: {row['Match Score']}%")

    st.subheader("Resume Ranking")
    st.dataframe(results)

    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="resume_screening_results.csv",
        mime="text/csv"
    )

    st.subheader("Match Score Chart")
    st.bar_chart(results.set_index("Resume"))
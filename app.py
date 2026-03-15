import streamlit as st
import pandas as pd
import PyPDF2
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
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
            page_text = page.extract_text()
            if page_text:
                text += page_text

        return text

    else:
        return file.read().decode("utf-8")


def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    keywords = [word for word in words if word not in ENGLISH_STOP_WORDS]

    return set(keywords)


if st.button("Screen Resumes"):

    if not job_description.strip():
        st.error("Please enter a job description.")
        st.stop()

    if not uploaded_files:
        st.error("Please upload at least one resume.")
        st.stop()

    resumes = []
    resume_names = []
    resume_texts = []

    for file in uploaded_files:

        text = extract_text(file)

        resumes.append(text)
        resume_texts.append(text)
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

    st.success("Resumes successfully analyzed!")

    st.subheader("🏆 Top Candidates")

    top_candidates = results.head(3)

    for i, row in top_candidates.iterrows():
        st.write(f"{row['Resume']} — Match Score: {row['Match Score']}%")

    st.subheader("Resume Ranking")
    st.dataframe(results)

    st.subheader("Match Score Chart")
    st.bar_chart(results.set_index("Resume"))

    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="resume_ranking.csv",
        mime="text/csv",
    )

    st.subheader("Resume Analysis")

    jd_words = extract_keywords(job_description)

    for i, resume_text in enumerate(resume_texts):

        resume_words = extract_keywords(resume_text)

        matched = resume_words.intersection(jd_words)
        missing = jd_words - resume_words

        st.markdown(f"### {resume_names[i]}")

        st.write("Matched Keywords:", list(matched)[:10])
        st.write("Missing Skills:", list(missing)[:10])

        st.subheader("Resume Preview")
        st.write(resume_text[:500])
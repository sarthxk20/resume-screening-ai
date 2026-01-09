import os
import streamlit as st
import pandas as pd

from src.extractor import extract_text
from src.preprocessing import clean_text
from src.matcher import (
    tfidf_score,
    bert_score,
    skill_gap,
    explain,
    recommend,
    detect_experience
)

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(page_title="Resume Screening System", layout="wide")

st.title("Resume Screening System")
st.markdown(
    """
An AI-driven Applicant Tracking System (ATS) simulation that analyzes resumes
against job descriptions using NLP techniques and semantic similarity models.
"""
)

# ==================================================
# HELPERS
# ==================================================
def skill_chips(skills, empty_msg="No major skill gaps detected."):
    if not skills:
        st.write(empty_msg)
        return

    html = ""
    for skill in sorted(skills):
        html += f"""
        <span style="
            display:inline-block;
            padding:6px 12px;
            margin:4px;
            border-radius:16px;
            border:1px solid #ccc;
            font-size:14px;
        ">
            {skill}
        </span>
        """
    st.markdown(html, unsafe_allow_html=True)


def resume_completeness(resume_text):
    sections = {
        "Skills": "skill",
        "Experience": "experience",
        "Education": "education"
    }
    found = [s for s, kw in sections.items() if kw in resume_text.lower()]
    return found, len(found), len(sections)


def ats_readability(score, skill_ratio, section_ratio):
    if score >= 70 and skill_ratio >= 0.6 and section_ratio >= 0.7:
        return "Good"
    elif score >= 40:
        return "Average"
    return "Poor"

# ==================================================
# ENGINE SELECTION
# ==================================================
engine_choice = st.radio(
    "Matching Engine",
    ["BERT (Semantic Similarity)", "TF-IDF (Keyword Matching)"]
)
engine = "bert" if "BERT" in engine_choice else "tfidf"

# ==================================================
# TABS
# ==================================================
tab_analysis, tab_compare = st.tabs([
    "Resume Analysis",
    "Model Comparison"
])

# ==================================================
# TAB 1 — RESUME ANALYSIS
# ==================================================
with tab_analysis:
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
    jd_text = st.text_area("Paste Job Description", height=200)

    if st.button("Analyze Resume"):
        if not (resume_file and jd_text.strip()):
            st.warning("Please upload a resume and paste a job description.")
            st.stop()

        temp_path = f"temp_{resume_file.name}"
        with open(temp_path, "wb") as f:
            f.write(resume_file.getbuffer())

        resume_text = clean_text(extract_text(temp_path))
        job_desc = clean_text(jd_text)
        os.remove(temp_path)

        # ---------------- SCORING ----------------
        overall_score = (
            bert_score(resume_text, job_desc)
            if engine == "bert"
            else tfidf_score(resume_text, job_desc)
        )

        resume_skills, job_skills, missing_skills = skill_gap(
            resume_text, job_desc
        )
        keywords, _, missing_keywords = explain(
            resume_text, job_desc
        )

        decision, confidence = recommend(overall_score)
        experience = detect_experience(resume_text)

        # ---------------- METRICS ----------------
        skill_score = (
            f"{len(resume_skills & job_skills)} / {len(job_skills)}"
            if job_skills else "0 / 0"
        )
        keyword_score = (
            f"{len(keywords) - len(missing_keywords)} / {len(keywords)}"
            if keywords else "0 / 0"
        )

        found_sections, found_count, total_sections = resume_completeness(
            resume_text
        )
        skill_ratio = (
            len(resume_skills & job_skills) / len(job_skills)
            if job_skills else 0
        )
        ats_score = ats_readability(
            overall_score, skill_ratio, found_count / total_sections
        )

        # ---------------- DISPLAY ----------------
        st.success(f"Overall Match Score: {overall_score}%")
        st.progress(min(overall_score / 100, 1.0))

        st.write(f"Decision: {decision}")
        st.write(f"Confidence: {confidence}")
        st.write(f"Experience Alignment: {experience}")

        st.subheader("Section-wise Scoring")
        c1, c2, c3 = st.columns(3)
        c1.metric("Skills Match", skill_score)
        c2.metric("Keyword Match", keyword_score)
        c3.metric("ATS Readability", ats_score)

        st.subheader("Resume Completeness")
        st.write(
            f"Sections found: {', '.join(found_sections)} "
            f"({found_count}/{total_sections})"
        )

        st.subheader("Missing Skills")
        skill_chips(missing_skills)

        # ---------------- WHY NOT SELECTED ----------------
        if overall_score < 70:
            st.subheader("Why Not Selected?")
            reasons = []

            if missing_skills:
                reasons.append(
                    f"The resume lacks key skills required for the role, such as "
                    f"{', '.join(list(missing_skills)[:5])}."
                )

            if missing_keywords:
                reasons.append(
                    "Several important job-related terms are missing or underrepresented, "
                    f"including {', '.join(list(missing_keywords)[:5])}."
                )

            if not reasons:
                reasons.append(
                    "While the resume broadly aligns with the role, the match score "
                    "suggests insufficient depth or emphasis in critical areas."
                )

            for r in reasons:
                st.write("- " + r)

# ==================================================
# TAB 2 — MODEL COMPARISON
# ==================================================
with tab_compare:
    st.subheader("TF-IDF vs BERT Comparison")

    resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx"], key="mc_resume")
    jd_text = st.text_area("Paste Job Description", height=200, key="mc_jd")

    if st.button("Run Comparison"):
        if not (resume_file and jd_text.strip()):
            st.warning("Upload a resume and paste a job description.")
            st.stop()

        temp_path = f"temp_{resume_file.name}"
        with open(temp_path, "wb") as f:
            f.write(resume_file.getbuffer())

        resume_text = clean_text(extract_text(temp_path))
        job_desc = clean_text(jd_text)
        os.remove(temp_path)

        tfidf_result = tfidf_score(resume_text, job_desc)
        bert_result = bert_score(resume_text, job_desc)

        df = pd.DataFrame({
            "Model": ["TF-IDF (Keyword)", "BERT (Semantic)"],
            "Match Score (%)": [tfidf_result, bert_result]
        })

        st.table(df)

        if bert_result > tfidf_result:
            st.info("BERT captures semantic similarity more effectively for this resume.")
        elif tfidf_result > bert_result:
            st.info("TF-IDF highlights stronger keyword overlap for this resume.")
        else:
            st.info("Both models produce identical scores.")

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown(
    """
<div style="text-align:center; font-size:14px;">
    Built by <b>Sarthak Shandilya</b><br>
    <a href="https://github.com/sarthxk20" target="_blank">GitHub</a> |
    <a href="https://www.linkedin.com/in/sarthxk20" target="_blank">LinkedIn</a>
</div>
""",
    unsafe_allow_html=True
)

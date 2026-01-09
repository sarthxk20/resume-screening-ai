import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


from src.extractor import extract_text
from src.preprocessing import clean_text
from src.matcher import (
    tfidf_score,
    bert_score,
    skill_gap,
    explain,
    recommend,
    decision_trace,
    detect_experience
)

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Resume Screening AI",
    layout="wide"
)

st.title("Resume Screening AI — ATS Simulation Platform")
st.markdown(
    """
An AI-powered Applicant Tracking System (ATS) designed to perform
resume analysis, semantic matching, and explainable hiring decisions.
"""
)

# ==================================================
# UI HELPERS
# ==================================================
def skill_chips(
    skills,
    empty_message="No major skill gaps identified based on the job description."
):
    """
    Display skills as clean visual chips.
    Shows a clear message when no gaps are detected.
    """
    if not skills:
        st.write(empty_message)
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
    """
    Check whether key resume sections are present.
    """
    sections = {
        "Skills": "skill",
        "Experience": "experience",
        "Education": "education"
    }
    found = [name for name, kw in sections.items() if kw in resume_text.lower()]
    return found, len(found), len(sections)


def ats_readability(score, skill_ratio, section_ratio):
    """
    Heuristic indicator of ATS friendliness.
    """
    if score >= 70 and skill_ratio >= 0.6 and section_ratio >= 0.7:
        return "Good"
    elif score >= 40:
        return "Average"
    return "Poor"

# ==================================================
# MATCHING ENGINE SELECTION
# ==================================================
engine_choice = st.radio(
    "Matching Engine",
    ["BERT (Semantic)", "TF-IDF (Keyword)"],
    key="engine_selector"
)
engine = "bert" if "BERT" in engine_choice else "tfidf"

# ==================================================
# APPLICATION TABS
# ==================================================
tab_analysis, tab_compare, tab_eval, tab_dashboard = st.tabs([
    "Resume Analysis",
    "Model Comparison",
])

# ==================================================
# TAB 1 — RESUME ANALYSIS
# ==================================================
with tab_analysis:
    resume_file = st.file_uploader(
        "Upload Resume",
        type=["pdf", "docx"],
        key="ra_resume"
    )
    jd_text = st.text_area(
        "Paste Job Description",
        height=200,
        key="ra_jd"
    )

    if st.button("Analyze Resume", key="ra_btn"):
        if not (resume_file and jd_text.strip()):
            st.warning("Please upload a resume and paste a job description.")
            st.stop()

        # --- Extract & preprocess text ---
        temp_path = f"temp_{resume_file.name}"
        with open(temp_path, "wb") as f:
            f.write(resume_file.getbuffer())

        resume_text = clean_text(extract_text(temp_path))
        job_desc = clean_text(jd_text)
        os.remove(temp_path)

        # --- Matching score ---
        overall_score = (
            bert_score(resume_text, job_desc)
            if engine == "bert"
            else tfidf_score(resume_text, job_desc)
        )

        # --- Skill & keyword analysis ---
        resume_skills, job_skills, missing_skills = skill_gap(resume_text, job_desc)
        keywords, _, missing_keywords = explain(resume_text, job_desc)

        decision, confidence = recommend(overall_score)
        experience = detect_experience(resume_text)

        # --- Section-wise scores ---
        skill_score = (
            f"{len(resume_skills & job_skills)} / {len(job_skills)}"
            if job_skills else "0 / 0"
        )
        keyword_score = (
            f"{len(keywords) - len(missing_keywords)} / {len(keywords)}"
            if keywords else "0 / 0"
        )

        found_sections, found_count, total_sections = resume_completeness(resume_text)
        skill_ratio = (
            len(resume_skills & job_skills) / len(job_skills)
            if job_skills else 0
        )
        ats_score = ats_readability(
            overall_score,
            skill_ratio,
            found_count / total_sections
        )

        # --- Results ---
        st.success(f"Overall Match Score: {overall_score}%")
        st.progress(min(overall_score / 100, 1.0))

        st.write(f"Decision: {decision}")
        st.write(f"Confidence: {confidence}")
        st.write(f"Experience Alignment: {experience}")

        st.subheader("Section-wise Scoring")
        col1, col2, col3 = st.columns(3)
        col1.metric("Skills Match", skill_score)
        col2.metric("Keyword Match", keyword_score)
        col3.metric("ATS Readability", ats_score)

        st.subheader("Resume Completeness")
        st.write(
            f"Sections found: {', '.join(found_sections)} "
            f"({found_count}/{total_sections})"
        )

        st.subheader("Missing Skills")
        skill_chips(missing_skills)

        # --- Explainability ---
        if overall_score < 70:
            st.subheader("Why Not Selected?")

            if missing_skills:
                st.write(
                    "The resume lacks several key skills required for this role, "
                    f"including {', '.join(list(missing_skills)[:5])}."
                )

            if missing_keywords:
                st.write(
                    "Important job-specific terms such as "
                    f"{', '.join(list(missing_keywords)[:5])} "
                    "are not clearly reflected in the resume."
                )

            if not missing_skills and not missing_keywords:
                st.write(
                    "The resume broadly aligns with the role, but the overall score "
                    "suggests gaps in clarity, emphasis, or depth."
                )

# ==================================================
# TAB 2 — MODEL COMPARISON
# ==================================================
with tab_compare:
    st.subheader("TF-IDF vs BERT — Model Comparison")

    resume_file = st.file_uploader(
        "Upload Resume",
        type=["pdf", "docx"],
        key="mc_resume"
    )
    jd_text = st.text_area(
        "Paste Job Description",
        height=200,
        key="mc_jd"
    )

    if st.button("Run Model Comparison", key="mc_btn"):
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

        st.table(pd.DataFrame({
            "Model": ["TF-IDF (Keyword)", "BERT (Semantic)"],
            "Match Score (%)": [tfidf_result, bert_result]
        }))

        if bert_result > tfidf_result:
            st.info("BERT captures semantic similarity more effectively for this case.")
        elif tfidf_result > bert_result:
            st.info("TF-IDF highlights stronger keyword overlap for this case.")
        else:
            st.info("Both models produce identical scores.")


# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown(
    """
<div style="text-align:center; font-size:14px;">
    Built by <b>Sarthak Shandilya</b> · v1.0 · Open Source<br>
    <a href="https://github.com/sarthxk20" target="_blank">GitHub</a> |
    <a href="https://www.linkedin.com/in/sarthxk20" target="_blank">LinkedIn</a>
</div>
""",
    unsafe_allow_html=True

)


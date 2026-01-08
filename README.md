# Resume Screening AI — ATS Simulation Platform

An AI-powered Applicant Tracking System (ATS) simulation designed to analyze resumes against job descriptions using Natural Language Processing (NLP) and semantic similarity models, with a strong focus on explainability and realistic hiring workflows.

---

## Overview

Modern Applicant Tracking Systems often act as black boxes, filtering resumes without providing clear reasoning.  
This project simulates how real ATS systems work by combining semantic embeddings, keyword-based matching, and interpretable decision logic.

The goal is not automation of hiring decisions, but **AI-assisted decision support** that is transparent, auditable, and recruiter-friendly.

---

## Key Features

### Resume Analysis
- Upload resumes in PDF or DOCX format
- Parse and clean unstructured resume text
- Compute overall resume–job match score
- Detect experience level (Junior / Mid / Senior)
- Identify missing skills and job-specific keywords
- Provide human-readable explanations for rejection decisions

### Semantic & Keyword Matching
- BERT-based semantic similarity for contextual understanding
- TF-IDF keyword matching for lexical overlap
- Side-by-side comparison of both approaches

### Explainable Decision Support
- Clear recommendation output (Select / Review / Reject)
- “Why Not Selected?” explanations
- ATS-style readability assessment

### Model Evaluation & Benchmarking
- Offline evaluation using human-labeled match scores
- Mean Absolute Error (MAE) comparison between models
- Visual dashboard for model performance analysis

---

## Modeling Approach

| Component | Description |
|---------|-------------|
| BERT | Captures semantic similarity between resumes and job descriptions |
| TF-IDF | Measures keyword-level relevance and overlap |
| Skill Gap Analysis | Identifies missing job-required skills |
| Explainability Layer | Converts model outputs into recruiter-readable insights |

This hybrid approach reflects real-world ATS design, where interpretability is as important as accuracy.

---

## Application Structure

The application consists of four main views:

1. Resume Analysis — End-to-end ATS-style evaluation  
2. Model Comparison — TF-IDF vs BERT performance comparison  
3. Evaluation Benchmark — Human judgment vs model outputs  
4. Evaluation Dashboard — Visual model performance insights  

---

## Tech Stack

- Python
- Streamlit
- spaCy
- scikit-learn
- Sentence-BERT / Transformers
- pandas
- matplotlib

---

## How to Run Locally

```bash
git clone https://github.com/sarthxk20/resume-screening-ai.git
cd resume-screening-ai

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
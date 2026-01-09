Resume Screening System (ATS Simulation)

An AI-driven Applicant Tracking System (ATS) simulation that analyzes resumes
against job descriptions using Natural Language Processing (NLP) and semantic
similarity models. The system demonstrates how modern hiring platforms score,
compare, and explain resume relevance.

------------------------------------------------------------

Project Overview

This project simulates the core logic of an ATS used in recruitment pipelines.
Users can upload a resume, provide a job description, and receive an objective,
explainable assessment of how well the resume matches the role.

The application is built as an interactive Streamlit web app and is designed to
be stable, deployable, and production-oriented.

------------------------------------------------------------

Key Features

1. Resume Analysis
- Upload resumes in PDF or DOCX format
- Automatic text extraction and normalization
- Overall resume–job match score

2. NLP and AI Models
- TF-IDF (keyword-based matching) using scikit-learn
- BERT-based semantic similarity using sentence-transformers

3. Skill Gap Detection
- Identifies skills present in the job description
- Detects missing or underrepresented skills in the resume
- Displays recruiter-friendly skill insights

4. Explainable Hiring Decisions
- Provides a clear "Why Not Selected?" section
- Highlights missing skills and job-specific terminology
- Includes decision confidence and experience alignment

5. Model Comparison
- Side-by-side comparison of TF-IDF vs BERT match scores
- Demonstrates differences between lexical and semantic NLP approaches

------------------------------------------------------------

Technology Stack

- Python
- Streamlit
- scikit-learn
- Sentence Transformers (BERT)
- Pandas
- PyPDF2
- python-docx

------------------------------------------------------------

Project Structure

resume-screening-ai/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── skills.txt
│
└── src/
    ├── __init__.py
    ├── extractor.py
    ├── preprocessing.py
    ├── matcher.py
    └── evaluation.py

------------------------------------------------------------

How the System Works

1. Resume text is extracted from PDF or DOCX files
2. Text is cleaned and normalized using lightweight NLP preprocessing
3. Resume and job description are vectorized using:
   - TF-IDF for keyword overlap
   - BERT embeddings for semantic similarity
4. Similarity scores are computed
5. Skill and keyword gaps are identified
6. Explainable hiring insights are generated

------------------------------------------------------------

Design Decisions

- Lightweight preprocessing is used instead of heavy NLP pipelines to improve
  deployment stability while retaining semantic intelligence through BERT.
- Model evaluation is kept offline since meaningful evaluation requires labeled
  resume–job pairs and human relevance scores.

------------------------------------------------------------

Use Cases

- Demonstrating how ATS systems evaluate resumes
- Comparing keyword-based and semantic NLP approaches
- Portfolio project for Data Science or Machine Learning roles
- Educational reference for NLP-based matching systems

------------------------------------------------------------

Author

Sarthak Shandilya

GitHub: https://github.com/sarthxk20
LinkedIn: https://www.linkedin.com/in/sarthxk20

------------------------------------------------------------

License

This project is open-source and intended for educational and demonstration
purposes.

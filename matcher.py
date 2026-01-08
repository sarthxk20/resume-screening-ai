import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Load BERT once
# -----------------------------
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# TF-IDF Score
# -----------------------------
def tfidf_score(resume, jd):
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform([resume, jd])
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)

# -----------------------------
# BERT Score
# -----------------------------
def bert_score(resume, jd):
    emb = bert_model.encode([resume, jd], convert_to_tensor=True)
    return round(util.cos_sim(emb[0], emb[1]).item() * 100, 2)

# -----------------------------
# Skills
# -----------------------------
def load_skills():
    with open("data/skills.txt") as f:
        return [s.strip().lower() for s in f.readlines()]

def extract_skills(text, skills):
    return {s for s in skills if s in text}

def skill_gap(resume, jd):
    skills = load_skills()
    r = extract_skills(resume, skills)
    j = extract_skills(jd, skills)
    return r, j, j - r

# -----------------------------
# Skill Suggestions
# -----------------------------
def skill_suggestions(missing):
    return {s: f"Consider adding a project using {s}" for s in missing}

# -----------------------------
# Experience Detection
# -----------------------------
def detect_experience(text):
    years = re.findall(r"(\d+)\+?\s+years", text)
    if years:
        y = max(map(int, years))
        if y >= 5:
            return "Senior"
        elif y >= 2:
            return "Mid-Level"
    return "Junior / Entry"

# -----------------------------
# Explainability
# -----------------------------
def explain(resume, jd, top_n=8):
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform([resume, jd])
    features = vec.get_feature_names_out()
    jd_vec = tfidf[1].toarray()[0]
    top_idx = jd_vec.argsort()[-top_n:][::-1]
    keywords = [features[i] for i in top_idx]
    resume_words = set(resume.split())
    return keywords, [k for k in keywords if k in resume_words], [k for k in keywords if k not in resume_words]

# -----------------------------
# Recommendation Engine
# -----------------------------
def recommend(score):
    if score >= 70:
        return "Shortlist", "High"
    elif score >= 40:
        return "Review", "Medium"
    return "Reject", "Low"

# -----------------------------
# Rank Resumes + CSV
# -----------------------------
def rank_resumes(resumes, jd, mode="bert"):
    rows = []
    for name, text in resumes.items():
        score = bert_score(text, jd) if mode == "bert" else tfidf_score(text, jd)
        rows.append({"Resume": name, "Score": score})
    df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    return df

# --------------------------------------------------
# Decision Trace (Explain WHY a decision was made)
# --------------------------------------------------
def decision_trace(score, missing_skills, missing_keywords):
    reasons = []

    if score < 40:
        reasons.append("Low overall semantic similarity with the job description")

    if missing_skills:
        reasons.append(
            "Missing critical skills: " + ", ".join(list(missing_skills)[:3])
        )

    if missing_keywords:
        reasons.append(
            "Missing important job-related terms: " + ", ".join(missing_keywords[:3])
        )

    if not reasons:
        reasons.append("Strong alignment across skills and experience")

    return reasons
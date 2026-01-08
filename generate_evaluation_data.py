import os
import sys
import random
import pandas as pd

# --------------------------------------------------
# Fix PYTHONPATH
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.extractor import extract_text
from src.preprocessing import clean_text

# --------------------------------------------------
# Configuration
# --------------------------------------------------
RESUME_FOLDER = "data/resumes"
OUTPUT_PATH = "data/evaluation_data.csv"

BASE_JOB_DESCRIPTIONS = [
    "data analyst python sql excel visualization",
    "data scientist python machine learning statistics",
    "business analyst excel reporting dashboards",
    "software engineer python backend api",
    "junior data analyst excel sql reporting",
    "senior data scientist deep learning python",
]

AUGMENT_PHRASES = [
    "hands on experience",
    "strong understanding",
    "real world projects",
    "industry experience",
    "end to end pipelines",
]

# --------------------------------------------------
# Heuristic Human Scoring
# --------------------------------------------------
def heuristic_human_score(resume_text, jd_text):
    resume_words = set(resume_text.split())
    jd_words = set(jd_text.split())

    overlap = len(resume_words & jd_words)
    total = len(jd_words)

    if total == 0:
        return 0.0

    score = overlap / total
    score += random.uniform(-0.05, 0.05)  # human noise
    return round(min(max(score, 0), 1), 2)

# --------------------------------------------------
# JD Augmentation
# --------------------------------------------------
def augment_jd(jd):
    augments = random.sample(AUGMENT_PHRASES, k=2)
    return jd + " " + " ".join(augments)

# --------------------------------------------------
# Generate Evaluation Dataset
# --------------------------------------------------
def generate():
    rows = []

    resumes = []
    for file in os.listdir(RESUME_FOLDER):
        if file.endswith((".pdf", ".docx")):
            path = os.path.join(RESUME_FOLDER, file)
            raw = extract_text(path)
            resumes.append(clean_text(raw)[:1200])

    for resume in resumes:
        for jd in BASE_JOB_DESCRIPTIONS:
            for _ in range(3):  # augmentation factor
                jd_aug = augment_jd(jd)
                rows.append({
                    "resume_text": resume,
                    "job_description": clean_text(jd_aug),
                    "human_score": heuristic_human_score(resume, jd_aug)
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Evaluation dataset generated: {OUTPUT_PATH}")
    print(f"📊 Total samples: {len(df)}")

if __name__ == "__main__":
    generate()
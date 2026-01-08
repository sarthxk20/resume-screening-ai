import pandas as pd
from src.matcher import tfidf_score, bert_score


def evaluate_models(csv_path="data/evaluation_data.csv"):
    """
    Compare TF-IDF and BERT scores against human relevance scores.
    Returns a DataFrame with absolute errors.
    """
    df = pd.read_csv(csv_path)

    results = []

    for _, row in df.iterrows():
        resume = row["resume_text"]
        jd = row["job_description"]
        human = row["human_score"]

        tfidf = tfidf_score(resume, jd) / 100
        bert = bert_score(resume, jd) / 100

        results.append({
            "Human Score": round(human, 2),
            "TF-IDF Score": round(tfidf, 2),
            "BERT Score": round(bert, 2),
            "TF-IDF Error": round(abs(human - tfidf), 2),
            "BERT Error": round(abs(human - bert), 2),
        })

    return pd.DataFrame(results)

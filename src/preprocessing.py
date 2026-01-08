import re
import nltk
import spacy
from nltk.corpus import stopwords

# Load spaCy model once
import spacy
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Clean and preprocess resume text.
    """
    # Lowercase
    text = text.lower()

    # Remove email addresses & URLs
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove special characters & digits
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Lemmatization + stopword removal
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in stop_words and len(token.text) > 2
    ]


    return " ".join(tokens)

import re


def clean_text(text: str) -> str:
    """
    Lightweight NLP text preprocessing.

    - Lowercases text
    - Removes non-alphanumeric characters
    - Normalizes whitespace

    This approach avoids heavy NLP dependencies while remaining
    effective for TF-IDF and semantic embedding models.
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

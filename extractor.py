import pdfplumber
from docx import Document
import os


def extract_text(file_path: str) -> str:
    """
    Extract text from a resume file (PDF or DOCX).

    Args:
        file_path (str): Path to the resume file

    Returns:
        str: Extracted text
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        return _extract_from_pdf(file_path)

    elif file_extension == ".docx":
        return _extract_from_docx(file_path)

    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")


def _extract_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text.strip()


def _extract_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()
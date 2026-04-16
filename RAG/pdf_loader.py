# =========================
# pdf_loader.py
# =========================
import fitz
import re


import fitz

def extract_text_with_metadata(pdf_path):
    doc = fitz.open(pdf_path)
    data = []

    for i, page in enumerate(doc):
        text = page.get_text()
        data.append({
            "text": text,
            "page": i + 1
        })

    return data

def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text

import os
import docx
import fitz  # pymupdf
import re
from typing import List, Dict

import pandas as pd


def extract_articles_from_docx(path: str) -> List[Dict[str, str]]:
    doc = docx.Document(path)
    articles = []
    current_title = None
    current_text_lines = []

    # паттерны-заголовки, которые нужно игнорировать
    skip_patterns = [
        r"^Глава\s+\d+",
        r"^Раздел\s+\d+",
        r"^Подраздел\s+\d+",
        r"^ОСОБЕННАЯ ЧАСТЬ",
        r"^ОБЩАЯ ЧАСТЬ",
        r"^РАЗДЕЛ\s+\d+",
        r"^ПРИЛОЖЕНИЕ",
        r"^СОДЕРЖАНИЕ"
    ]

    for para in doc.paragraphs:
        line = para.text.strip()
        if not line or len(line) < 3:
            continue

        # Начинается новая статья
        if re.match(r"^Статья\s+\d+[.\d]*", line):
            if current_title and current_text_lines:
                articles.append({
                    "title": current_title,
                    "text": " ".join(current_text_lines).strip()
                })
            current_title = line
            current_text_lines = []
        elif any(re.match(p, line, flags=re.IGNORECASE) for p in skip_patterns):
            continue  # пропускаем заголовки разделов, глав и пр.
        elif current_title:
            current_text_lines.append(line)

    # сохранить последнюю статью
    if current_title and current_text_lines:
        articles.append({
            "title": current_title,
            "text": " ".join(current_text_lines).strip()
        })

    return articles




def extract_articles_from_pdf(path: str) -> List[Dict[str, str]]:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()

    lines = text.split("\n")
    articles = []
    current_title = ""
    current_text = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r"^Статья\s+\d+.*", line):
            if current_title:
                articles.append({"title": current_title, "text": current_text.strip()})
            current_title = line
            current_text = ""
        else:
            current_text += line + " "

    if current_title:
        articles.append({"title": current_title, "text": current_text.strip()})

    return articles


def extract_articles_from_exel(path: str) -> List[Dict[str, str]]:
    df = pd.read_excel(path)
    articles = []
    for _, row in df.iterrows():
        if pd.isna(row["name"]) or pd.isna(row["chunks"]):
            continue
        articles.append({
            "title": str(row["name"]).strip(),
            "text": str(row["chunks"]).strip(),
            "egov_link": str(row["eGov_link"]).strip(),
            "egov_link_kaz": str(row["eGov_kaz_link"]).strip(),
            "source": "dataset"
        })

    return articles;

def load_articles(path: str) -> List[Dict[str, str]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        return extract_articles_from_docx(path)
    elif ext == ".pdf":
        return extract_articles_from_pdf(path)
    elif ext == ".xlsx":
        return extract_articles_from_exel(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

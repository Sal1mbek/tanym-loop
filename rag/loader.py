import os
import docx
import fitz  # pymupdf
import re
from typing import List, Dict
import pandas as pd


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Разбивает текст на чанки с перекрытием.
    Для неструктурированных документов.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks if chunks else [text]  # если текст короткий, вернуть как есть


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

    full_text = []
    all_doc_lines = []

    for para in doc.paragraphs:
        line = para.text.strip()
        all_doc_lines.append(line)
        if not line or len(line) < 3:
            continue

        # Начинается новая статья
        if re.match(r"^Статья\s+\d+[.\d]*", line):
            if current_title and current_text_lines:
                articles.append({
                    "title": current_title,
                    "text": " ".join(current_text_lines).strip(),
                    "egov_link": "",
                    "egov_link_kaz": "",
                    "source": "user_upload"
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
            "text": " ".join(current_text_lines).strip(),
            "egov_link": "",
            "egov_link_kaz": "",
            "source": "user_upload"
        })

    if not articles:
        full_doc_text = " ".join(filter(None, all_doc_lines))
        chunks = chunk_text(full_doc_text, chunk_size=500, overlap=50)

        filename = os.path.basename(path)
        for i, chunk in enumerate(chunks, 1):
            articles.append({
                "title": f"{filename} — Часть {i}",
                "text": chunk,
                "egov_link": "",
                "egov_link_kaz": "",
                "source": "user_upload"
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
                articles.append({
                    "title": current_title,
                    "text": current_text.strip(),
                    "egov_link": "",
                    "egov_link_kaz": "",
                    "source": "user_upload"
                })
            current_title = line
            current_text = ""
        else:
            current_text += line + " "

    if current_title:
        articles.append({
            "title": current_title,
            "text": current_text.strip(),
            "egov_link": "",
            "egov_link_kaz": "",
            "source": "user_upload"
        })

    if not articles:
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        filename = os.path.basename(path)

        for i, chunk in enumerate(chunks, 1):
            articles.append({
                "title": f"{filename} — Часть {i}",
                "text": chunk,
                "egov_link": "",
                "egov_link_kaz": "",
                "source": "user_upload"
            })

    return articles


def extract_articles_from_txt(path: str) -> List[Dict[str, str]]:
    """
    Обрабатывает TXT файлы — просто разбивает на чанки.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=500, overlap=50)
    filename = os.path.basename(path)

    articles = []
    for i, chunk in enumerate(chunks, 1):
        articles.append({
            "title": f"{filename} — Часть {i}",
            "text": chunk,
            "egov_link": "",
            "egov_link_kaz": "",
            "source": "user_upload"
        })

    return articles


def extract_articles_from_excel(path: str) -> List[Dict[str, str]]:
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
    elif ext in [".xlsx", ".xls"]:
        return extract_articles_from_excel(path)
    elif ext == ".txt":
        return extract_articles_from_txt(path)
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

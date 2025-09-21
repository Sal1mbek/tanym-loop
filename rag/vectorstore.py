import psycopg2
from typing import List, Dict
from dotenv import load_dotenv
import os

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "raglaw"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "ss123"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", 5432),
}


class VectorStore:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.cursor = self.conn.cursor()
        self._init_table()

    def _init_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                title TEXT,
                content TEXT,
                egov_link TEXT,
                egov_link_kaz TEXT,
                embedding VECTOR(384),
                source TEXT DEFAULT 'dataset' 
            );
        """)
        self.conn.commit()

    def insert_articles(self, articles: List[Dict]):
        for article in articles:
            # Проверка на дубликат по title
            self.cursor.execute("SELECT 1 FROM documents WHERE title = %s AND content = %s", (article["title"], article["text"]))
            if self.cursor.fetchone():
                continue  # уже есть — пропустить

            self.cursor.execute(
                "INSERT INTO documents (title, content, egov_link, egov_link_kaz, embedding, source) VALUES (%s, %s, %s, %s, %s, %s)", (
                article["title"],
                article["text"],
                article["egov_link"],
                article["egov_link_kaz"],
                article["embedding"].tolist(),
                article.get("source", "dataset")
                )
            )
        self.conn.commit()

    def search_similar(self, query_embedding: List[float], top_k=5) -> List[Dict]:
        if hasattr(query_embedding, "tolist"):  # если это numpy
            query_embedding = query_embedding.tolist()

        self.cursor.execute("""
            SELECT title, content, egov_link, egov_link_kaz
            FROM documents 
            ORDER BY embedding <-> %s::vector
            LIMIT %s
            """, (query_embedding, top_k)
        )

        results = self.cursor.fetchall()
        return [
            {
                "title": title,
                "text": content,
                "egov_link": egov_link,
                "egov_link_kaz": egov_kaz_link
            }
            for title, content, egov_link, egov_kaz_link in results
        ]
    def close(self):
        self.cursor.close()
        self.conn.close()

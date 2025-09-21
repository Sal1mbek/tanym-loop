import psycopg2
from typing import Optional
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


class FeedbackStore:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.cursor = self.conn.cursor()
        self._init_table()

    def _init_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                question TEXT,
                answer TEXT,
                correct_answer TEXT,
                comment TEXT,
                rating INT,
                source TEXT DEFAULT 'user',
                embedding VECTOR(384)
            );
        """)
        self.conn.commit()

    def insert_feedback(
        self,
        question: str,
        answer: str,
        comment: str,
        rating: int,
        embedding: Optional[list] = None,
        correct_answer: Optional[str] = None,
        source: str = "user"
    ):
        self.cursor.execute("""
            INSERT INTO feedback (question, answer, correct_answer, comment, rating, source, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (question, answer, correct_answer, comment, rating, source, embedding.tolist() if embedding is not None else None))
        self.conn.commit()

    def search_similar(self, query_embedding: list, top_k=1) -> list[dict]:
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        self.cursor.execute("""
            SELECT question, answer, correct_answer, comment, rating, source, (embedding <-> %s::vector) AS distance
            FROM feedback
            order by embedding <-> %s::vector
            limit %s
        """, (query_embedding, query_embedding, top_k))

        rows = self.cursor.fetchall()
        return [
            {
                "question": q,
                "answer": answer,
                "correct_answer": good,
                "comment": comment,
                "rating": rating,
                "source": source,
                "distance": dist
            }
            for q, answer, good, comment, rating, source, dist in rows
        ]

    def close(self):
        self.cursor.close()
        self.conn.close()

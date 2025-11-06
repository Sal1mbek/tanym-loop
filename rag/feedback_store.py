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
                embedding VECTOR(384),
                owner_id INT
            );
        """)
        try:
            self.cursor.execute("""
                        CREATE INDEX IF NOT EXISTS feedback_embedding_idx
                        ON feedback USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
        except Exception as e:
            # если pgvector не поддерживает ivfflat здесь, пропускаем, но логируем
            print(f"[FEEDBACK] Could not create ivfflat index for feedback.embedding: {e}")
        self.conn.commit()

    def insert_feedback(
        self,
        question: str,
        answer: str,
        comment: str,
        rating: int,
        embedding: Optional[list] = None,
        correct_answer: Optional[str] = None,
        source: str = "user",
        owner_id: Optional[int] = None
    ):
        emb_val = None
        try:
            if embedding is not None:
                emb_val = embedding.tolist() if hasattr(embedding, "tolist") else embedding
        except Exception:
            emb_val = embedding  # fallback

        try:
            self.cursor.execute("""
                        INSERT INTO feedback (question, answer, correct_answer, comment, rating, source, embedding, owner_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (question, answer, correct_answer, comment, rating, source, emb_val, owner_id))

            # Попытка получить id — но не падать, если его нет
            fid = None
            try:
                row = self.cursor.fetchone()
                if row:
                    fid = row[0]
            except Exception as e:
                # Не фатальная ошибка — залогируем и продолжим (fid останется None)
                print(f"[FEEDBACK][WARN] fetchone() failed after INSERT RETURNING: {e}")

            self.conn.commit()
            return fid

        except Exception as e:
            # Откат и логирование — возвращаем исключение выше
            try:
                self.conn.rollback()
            except Exception:
                pass
            print(f"[FEEDBACK][ERROR] insert_feedback failed: {e}")
            raise

    def search_similar(self, query_embedding: list, top_k=1, owner_id: Optional[int] = None) -> list[dict]:
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        if owner_id is None:
            sql = """
                        SELECT question, answer, correct_answer, comment, rating, source, owner_id, (embedding <-> %s::vector) AS distance
                        FROM feedback
                        ORDER BY embedding <-> %s::vector
                        LIMIT %s
                    """
            params = (query_embedding, query_embedding, top_k)
        else:
            sql = """
                        SELECT question, answer, correct_answer, comment, rating, source, owner_id, (embedding <-> %s::vector) AS distance
                        FROM feedback
                        WHERE owner_id = %s
                        ORDER BY embedding <-> %s::vector
                        LIMIT %s
                    """
            params = (query_embedding, owner_id, query_embedding, top_k)

        self.cursor.execute(sql, params)

        rows = self.cursor.fetchall()
        return [
            {
                "question": q,
                "answer": answer,
                "correct_answer": good,
                "comment": comment,
                "rating": rating,
                "source": source,
                "owner_id": owner_id_row,
                "distance": dist
            }
            for q, answer, good, comment, rating, source, owner_id_row, dist in rows
        ]

    def close(self):
        try:
            self.cursor.close()
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass

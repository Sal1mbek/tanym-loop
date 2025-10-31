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
        try:
            # Шаг 1: Убедиться, что расширение pgvector установлено
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Шаг 2: Создать таблицу документов
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    egov_link TEXT,
                    egov_link_kaz TEXT,
                    embedding VECTOR(384),
                    source TEXT DEFAULT 'dataset',
                    filename TEXT
                );
            """)

            self.cursor.execute("""
                            CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                            ON documents 
                            USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = 100);
                        """)

            self.conn.commit()
            print("✅ Таблица 'documents' и расширение 'vector' инициализированы.")
        except psycopg2.Error as e:
            # Если что-то пошло не так, откатываем и выводим ошибку
            self.conn.rollback()
            print("❌ ОШИБКА ИНИЦИАЛИЗАЦИИ БД: Возможно, pgvector не установлен или проблема с правами.")
            print(f"Детали ошибки: {e}")

    def insert_articles(self, articles: List[Dict]):
        inserted_count = 0
        skipped_count = 0
        error_count = 0

        for article in articles:
            try:
                # 1. Проверка на дубликат по title и content
                self.cursor.execute("SELECT 1 FROM documents WHERE title = %s AND content = %s LIMIT 1",
                                    (article["title"], article["text"]))
                if self.cursor.fetchone():
                    skipped_count += 1
                    continue  # уже есть — пропустить

                # 2. Вставка статьи
                # Ошибка, скорее всего, возникает здесь, если embedding.tolist()
                # содержит данные, несовместимые с типом VECTOR(384), или если
                # длина данных превышает 384 (что маловероятно для стандартных эмбеддингов).
                self.cursor.execute(
                    "INSERT INTO documents (title, content, egov_link, egov_link_kaz, embedding, source, filename) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        article["title"],
                        article["text"],
                        article["egov_link"],
                        article["egov_link_kaz"],
                        article["embedding"].tolist(),
                        article.get("source", "dataset"),
                        article.get("filename")
                    )
                )
                inserted_count += 1

            except psycopg2.Error as e:
                # 3. ЕСЛИ ПРОИЗОШЛА ОШИБКА, ОТКАТЫВАЕМ ТРАНЗАКЦИЮ!
                # Это очищает флаг 'aborted' и позволяет продолжить работу.
                self.conn.rollback()
                error_count += 1
                print(f"❌ SQL Error on insert (Title: {article['title']}): {e}")
                # Мы не вызываем commit, потому что этот цикл должен работать в режиме
                # 'сохраняем все, что смогли'.
                # ВНИМАНИЕ: Если вы хотите, чтобы все статьи вставлялись одним батчем
                # (т.е. либо все, либо ничего), то этот try...except нужно ставить
                # вокруг всего цикла, но тогда вам придется откатываться полностью.
                # Для 'ingest' лучше такой режим:
                continue  # Переходим к следующей статье
        try:
            self.conn.commit()
            if inserted_count > 0 or skipped_count > 0:
                print(f"✅ Вставка завершена: {inserted_count} новых, {skipped_count} пропущено, {error_count} ошибок")
        except psycopg2.Error as e:
            print(f"❌ Ошибка при финальном commit: {e}")
            self.conn.rollback()

        return {"inserted": inserted_count, "skipped": skipped_count, "errors": error_count}

    def search_similar(self, query_embedding: List[float], top_k=5, min_distance=0.0) -> List[Dict]:
        if hasattr(query_embedding, "tolist"):  # если это numpy
            query_embedding = query_embedding.tolist()

        self.cursor.execute("""
                    SELECT 
                        title, 
                        content, 
                        egov_link, 
                        egov_link_kaz,
                        source,
                        (embedding <-> %s::vector) AS distance
                    FROM documents 
                    WHERE (embedding <-> %s::vector) >= %s
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, min_distance, query_embedding, top_k))

        results = self.cursor.fetchall()

        return [
            {
                "title": title,
                "text": content,
                "egov_link": egov_link if egov_link else "",
                "egov_link_kaz": egov_kaz_link if egov_kaz_link else "",
                "source": source,
                "distance": float(distance),  # ВАЖНО: метрика для фильтрации
                "similarity": 1 - float(distance)  # человеко-читаемый формат (0-1)
            }
            for title, content, egov_link, egov_kaz_link, source, distance in results
        ]

    def get_stats(self) -> Dict:
        """
        Возвращает статистику по базе документов.

        Полезно для мониторинга и отладки.
        """
        self.cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT source) as unique_sources
            FROM documents
        """)
        total, unique_sources = self.cursor.fetchone()

        self.cursor.execute("SELECT COUNT(DISTINCT filename) FROM documents")
        unique_files = self.cursor.fetchone()[0]

        self.cursor.execute("""
            SELECT source, COUNT(*) as count
            FROM documents
            GROUP BY source
            ORDER BY count DESC
        """)
        by_source = {row[0]: row[1] for row in self.cursor.fetchall()}

        return {
            "total_documents": total,
            "unique_sources": unique_sources,
            "unique_files": unique_files,
            "by_source": by_source
        }

    def close(self):
        """Закрывает соединение с БД"""
        self.cursor.close()
        self.conn.close()

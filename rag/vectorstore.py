import psycopg2
from typing import List, Dict, Optional
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
        # self.cursor = self.conn.cursor()
        self._init_table()

    def _init_table(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        title TEXT,
                        content TEXT,
                        egov_link TEXT,
                        egov_link_kaz TEXT,
                        embedding VECTOR(384),
                        source TEXT DEFAULT 'dataset',
                        filename TEXT,
                        owner_id INT
                    );
                """)
                cur.execute("""
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

    def insert_articles(self, articles: List[Dict],  owner_id: Optional[int] = None):
        inserted_count = 0
        skipped_count = 0
        error_count = 0

        for article in articles:
            try:
                # 1. Проверка на дубликат по title и content
                with self.conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM documents WHERE title = %s AND content = %s LIMIT 1",
                                (article["title"], article["text"]))
                    if cur.fetchone():
                        skipped_count += 1
                        continue

                    owner_for_insert = owner_id if owner_id is not None else article.get("owner_id")

                    cur.execute(
                        "INSERT INTO documents (title, content, egov_link, egov_link_kaz, embedding, source, filename, owner_id) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            article["title"],
                            article["text"],
                            article.get("egov_link"),
                            article.get("egov_link_kaz"),
                            article["embedding"].tolist() if hasattr(article["embedding"], "tolist") else article["embedding"],
                            article.get("source", "dataset"),
                            article.get("filename"),
                            owner_for_insert
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

    def search_similar(self, query_embedding: List[float], top_k=3, owner_id: Optional[int] = None) -> List[Dict]:
        if hasattr(query_embedding, "tolist"):  # если это numpy
            query_embedding = query_embedding.tolist()

        with self.conn.cursor() as cur:
            if owner_id is None:
                cur.execute("""
                                SELECT id, title, content, egov_link, egov_link_kaz, source, (embedding <-> %s::vector) AS distance
                                FROM documents
                                ORDER BY embedding <-> %s::vector
                                LIMIT %s
                            """, (query_embedding, query_embedding, top_k))
            else:
                cur.execute("""
                                SELECT id, title, content, egov_link, egov_link_kaz, source, (embedding <-> %s::vector) AS distance
                                FROM documents
                                WHERE owner_id = %s
                                ORDER BY embedding <-> %s::vector
                                LIMIT %s
                            """, (query_embedding, owner_id, query_embedding, top_k))
            rows = cur.fetchall()

        results = []

        for doc_id, title, content, egov_link, egov_link_kaz, source, distance in rows:
            distance = float(distance)

            raw_cosine = 1.0 - distance
            # Нормируем в 0..1
            similarity = max(0.0, min(1.0, (raw_cosine + 1.0) / 2.0))

            results.append({
                "id": doc_id,
                "title": title,
                "text": content,
                "egov_link": egov_link or "",
                "egov_link_kaz": egov_link_kaz or "",
                "source": source,
                "distance": distance,
                "raw_cosine": raw_cosine,
                "similarity": similarity
            })

        return results

    def list_documents(self, owner_id: Optional[int] = None) -> List[Dict]:
        """
        Возвращает список проиндексированных файлов (filename) с количеством чанков.
        """
        with self.conn.cursor() as cur:
            if owner_id is None:
                cur.execute("""
                                SELECT COALESCE(filename, '') AS filename, COUNT(*) as chunks, MIN(source) as source
                                FROM documents
                                GROUP BY filename
                                ORDER BY chunks DESC;
                            """)
            else:
                cur.execute("""
                                SELECT COALESCE(filename, '') AS filename, COUNT(*) as chunks, MIN(source) as source
                                FROM documents
                                WHERE owner_id = %s
                                GROUP BY filename
                                ORDER BY chunks DESC;
                            """, (owner_id,))
            rows = cur.fetchall()

        docs = []
        for filename, chunks, source in rows:
            docs.append({
                "filename": filename,
                "chunks": int(chunks),
                "source": source
            })
        return docs

    def delete_documents_by_filename(self, filename: str, owner_id: Optional[int] = None) -> bool:
        """
        Удаляет все записи, у которых filename == filename.
        Возвращает True если что-то удалено.
        """
        try:
            with self.conn.cursor() as cur:
                if owner_id is None:
                    cur.execute("DELETE FROM documents WHERE filename = %s;", (filename,))
                else:
                    cur.execute("DELETE FROM documents WHERE filename = %s AND owner_id = %s;", (filename, owner_id))
                deleted = cur.rowcount
            self.conn.commit()
            return deleted > 0
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Ошибка при удалении {filename}: {e}")
            raise

    def get_stats(self, owner_id: Optional[int] = None) -> Dict:
        """
        Возвращает статистику по базе документов.

        Полезно для мониторинга и отладки.
        """
        with self.conn.cursor() as cur:
            if owner_id is None:
                cur.execute("""
                                SELECT COUNT(*) as total, COUNT(DISTINCT source) as unique_sources
                                FROM documents
                            """)
            else:
                cur.execute("""
                                SELECT COUNT(*) as total, COUNT(DISTINCT source) as unique_sources
                                FROM documents WHERE owner_id = %s
                            """, (owner_id,))
            row = cur.fetchone()
            total = int(row[0]) if row else 0
            unique_sources = int(row[1]) if row and len(row) > 1 else 0

            if owner_id is None:
                cur.execute("SELECT COUNT(DISTINCT filename) FROM documents")
            else:
                cur.execute("SELECT COUNT(DISTINCT filename) FROM documents WHERE owner_id = %s", (owner_id,))
            unique_files = int(cur.fetchone()[0] or 0)

            if owner_id is None:
                cur.execute("""
                                SELECT source, COUNT(*) as count
                                FROM documents
                                GROUP BY source
                                ORDER BY count DESC
                            """)
            else:
                cur.execute("""
                                SELECT source, COUNT(*) as count
                                FROM documents
                                WHERE owner_id = %s
                                GROUP BY source
                                ORDER BY count DESC
                            """, (owner_id,))
            by_source = {r[0]: r[1] for r in cur.fetchall()}

        return {
            "total_documents": total,
            "unique_sources": unique_sources,
            "unique_files": unique_files,
            "by_source": by_source
        }

    def get_all_documents(self, owner_id: Optional[int] = None) -> List[Dict]:
        """
        Возвращает все записи documents как список кортежей
        (title, content, egov_link, egov_link_kaz, source, filename).
        """
        with self.conn.cursor() as cur:
            if owner_id is None:
                cur.execute("SELECT title, content, egov_link, egov_link_kaz, source, filename FROM documents;")
            else:
                cur.execute(
                    "SELECT title, content, egov_link, egov_link_kaz, source, filename FROM documents WHERE owner_id = %s;",
                    (owner_id,))
            rows = cur.fetchall()
        return rows

    def close(self):
        """Закрывает соединение с БД"""
        try:
            self.conn.close()
        except Exception:
            pass

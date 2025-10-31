import os
import json
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.feedback_store import FeedbackStore
from rag.generator import Generator
import rag.loader as loader
import pickle

APP_PORT = int(os.environ.get("PORT", "8000"))
DATA_DIR = os.path.abspath("./data/uploads")
STATIC_DIR = os.path.abspath("./static")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ---- Инициализация ядра ----
print("=" * 60)
print("🚀 Запуск Tanym Loop...")
print("=" * 60)

try:
    with open("embedded.pkl", "rb") as f:
        embedded = pickle.load(f)
    print(f"✅ Загружено {len(embedded)} статей из кэша (embedded.pkl)")
except FileNotFoundError:
    print("⚠️  embedded.pkl не найден. База знаний будет пустой.")
    embedded = []

store = VectorStore()
feedback_store = FeedbackStore()
embedder = Embedder()
gen = Generator()

if embedded:
    print(f"📚 Индексация {len(embedded)} статей в PostgreSQL...")
    stats = store.insert_articles(embedded)
    print(f"   Вставлено: {stats['inserted']}, Пропущено: {stats['skipped']}")

# Статистика БД
db_stats = store.get_stats()
print(f"📊 Статистика БД: {db_stats['total_documents']} документов")
for src, cnt in db_stats['by_source'].items():
    print(f"   - {src}: {cnt}")

print("=" * 60)

app = FastAPI(title="Tanym Loop API", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def root():
    index = os.path.join(STATIC_DIR, "tanym_loop_mvp.html")
    if os.path.exists(index):
        return FileResponse(index)
    return HTMLResponse(
        "<h3>Помести tanym_loop_mvp.html в ./static и открой <a href='/'>/</a></h3>"
    )


@app.get("/health")
def health():
    """
    Healthcheck эндпоинт с детальной информацией о системе.
    """
    stats = store.get_stats()
    return {
        "status": "ok",
        "db_documents": stats['total_documents'],
        "embedder_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "embedding_dim": embedder.get_embedding_dimension()
    }


# ====== Q&A (КРИТИЧЕСКИ УЛУЧШЕН) ======
@app.post("/ask")
async def ask(question: str = Form(...), show_articles: bool = Form(True)):
    """
    Возвращает ответ + источники с метриками релевантности.

    КЛЮЧЕВЫЕ УЛУЧШЕНИЯ:
    1. Фильтрация нерелевантных результатов (distance > 0.7)
    2. Метрики релевантности для каждого источника
    3. Детальная информация об источниках
    4. Честный ответ "Нет данных", если не найдено
    """
    if not question.strip():
        return {"answer": "❌ Введите вопрос.", "sources_md": ""}

    # 1. Эмбеддинг запроса
    query_emb = embedder.embed_query(question)

    # 2. Поиск похожих (top_k=N для лучшего покрытия)
    results = store.search_similar(query_emb, top_k=3)

    # 3. КРИТИЧНО: Фильтруем нерелевантные результаты
    # Distance > 0.7 означает слабую связь (для cosine distance)
    # Настраиваемый параметр в зависимости от ваших данных
    RELEVANCE_THRESHOLD = 1.2
    relevant_results = [r for r in results if r['distance'] < RELEVANCE_THRESHOLD]

    if not relevant_results:
        return {
            "answer": "❌ К сожалению, не найдено релевантных документов по вашему запросу.\n\n💡 Попробуйте:\n- Переформулировать вопрос\n- Использовать другие ключевые слова\n- Загрузить дополнительные документы",
            "sources_md": "**Нет подходящих источников**\n\nПопробуйте уточнить запрос или загрузите соответствующие документы.",
            "metadata": {
                "found_results": len(results),
                "relevant_results": 0,
                "used_results": 0,
                "avg_similarity": 0
            }
        }

    # 4. Берем топ-2 для генерации (баланс качества и скорости)
    top_results = relevant_results[:2]
    context_chunks = [f"{r['title']}\n{r['text']}" for r in top_results]

    # 5. Генерация ответа
    answer = gen.generate_answer(question, context_chunks)

    # 6. Формируем источники с детальной информацией
    sources_lines = []

    if show_articles:
        sources_lines.append("## 📚 Использованные источники\n")

        for idx, r in enumerate(top_results, 1):
            # Показываем релевантность
            similarity_percent = int(r['similarity'] * 100)

            sources_lines.append(f"### Источник {idx} (релевантность: {similarity_percent}%)")
            sources_lines.append(f"**{r['title']}**")

            # Превью текста (первые 300 символов)
            preview = r['text'][:300] + "..." if len(r['text']) > 300 else r['text']
            sources_lines.append(f"\n{preview}\n")

            # Метаданные
            sources_lines.append(f"_Источник данных: {r['source']}_")

            # Ссылки (если есть)
            if r.get("egov_link"):
                sources_lines.append(f"🔗 [Ссылка на eGov]({r['egov_link']})")
            if r.get("egov_link_kaz"):
                sources_lines.append(f"🔗 [Қазақша сілтеме]({r['egov_link_kaz']})")

            sources_lines.append("")  # Разделитель

    # 7. Всегда показываем ссылки (даже если show_articles=False)
    else:
        links_found = False
        sources_lines.append("## 📎 Полезные ссылки\n")

        for r in top_results:
            if r.get("egov_link"):
                sources_lines.append(f"- [{r['title']}]({r['egov_link']})")
                links_found = True
            if r.get("egov_link_kaz"):
                sources_lines.append(f"  [Қазақша сілтеме]({r['egov_link_kaz']})")
                links_found = True

        if not links_found:
            sources_lines.append("_Нет доступных ссылок для этих источников_")

    sources_md = "\n".join(sources_lines)

    return {
        "answer": answer,
        "sources_md": sources_md,
        "metadata": {
            "found_results": len(results),
            "relevant_results": len(relevant_results),
            "used_results": len(top_results),
            "avg_similarity": round(sum(r['similarity'] for r in top_results) / len(top_results), 2)
        }
    }


# ====== FEEDBACK ======
@app.post("/feedback")
async def feedback(
        rating: int = Form(...),
        comment: str = Form(""),
        correct_answer: str = Form(""),
        question: Optional[str] = Form(None),
        answer: Optional[str] = Form(None),
):
    """
    Сохраняет отзыв пользователя для самообучения системы.
    """
    if not question or not answer:
        return {"ok": False, "msg": "❌ Сначала задайте вопрос и получите ответ."}

    query_emb = embedder.embed_query(question)
    feedback_store.insert_feedback(
        question=question,
        answer=answer,
        comment=comment or "",
        rating=int(rating),
        embedding=query_emb,
        correct_answer=correct_answer.strip() if correct_answer.strip() else None,
        source="user",
    )
    return {"ok": True, "msg": "✅ Спасибо за ваш отзыв! Он поможет улучшить систему."}


# ====== INGEST (ПОЛНЫЙ ПАЙПЛАЙН) ======
@app.post("/ingest")
async def ingest(
        files: List[UploadFile] = File(...),
        source_tag: str = Form("user"),
):
    """
    Полный пайплайп загрузки документов:
    Сохранение → Парсинг → Чанкинг → Эмбеддинг → БД → Кэш

    Возвращает детальную статистику по каждому файлу.
    """
    processing_results = []
    total_chunks = 0

    for uf in files:
        name = uf.filename
        dest = os.path.join(DATA_DIR, name)

        try:
            # 1. Сохранение файла
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            content = await uf.read()
            with open(dest, "wb") as f:
                f.write(content)

            # 2. Парсинг документа
            articles = loader.load_articles(dest)

            if not articles:
                processing_results.append({
                    "file": name,
                    "status": "warning",
                    "message": "Файл обработан, но не найдено текста для индексации."
                })
                continue

            # 3. Добавляем source_tag
            for art in articles:
                art["source"] = source_tag
                art["filename"] = name

            # 4. Эмбеддинг
            print(f"📄 {name}: Эмбеддинг {len(articles)} чанков...")
            embedded_articles = embedder.embed_articles(articles)

            # 5. Вставка в БД
            stats = store.insert_articles(embedded_articles)

            # 6. Обновление кэша (опционально)
            try:
                with open("embedded.pkl", "rb") as f:
                    cached = pickle.load(f)
            except:
                cached = []

            cached.extend(embedded_articles)

            with open("embedded.pkl", "wb") as f:
                pickle.dump(cached, f)

            total_chunks += len(articles)

            processing_results.append({
                "file": name,
                "status": "ok",
                "chunks": len(articles),
                "inserted": stats['inserted'],
                "skipped": stats['skipped']
            })

        except Exception as e:
            print(f"[INGEST ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()

            processing_results.append({
                "file": name,
                "status": "error",
                "message": f"Ошибка обработки: {str(e)}"
            })

    return {
        "ok": True,
        "results": processing_results,
        "total_chunks": total_chunks,
        "summary": f"Обработано {len(files)} файлов, {total_chunks} чанков"
    }


# ====== СТАТИСТИКА ======
@app.get("/stats")
def get_stats():
    """
    Возвращает детальную статистику по базе знаний.

    Полезно для мониторинга и отладки.
    """
    db_stats = store.get_stats()

    return {
        "database": db_stats,
        "embedder": {
            "model": "paraphrase-multilingual-MiniLM-L12-v2",
            "dimension": embedder.get_embedding_dimension()
        },
        "system": {
            "uploads_dir": DATA_DIR,
            "cache_file": "embedded.pkl"
        }
    }


if __name__ == "__main__":
    import uvicorn

    print(f"\n🌐 Сервер запущен на http://localhost:{APP_PORT}")
    print(f"📱 UI доступен по http://localhost:{APP_PORT}/")
    print(f"📊 Статистика: http://localhost:{APP_PORT}/stats")
    print(f"💚 Health check: http://localhost:{APP_PORT}/health\n")
    uvicorn.run("server:app", host="localhost", port=APP_PORT, reload=True)
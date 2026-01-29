# File: ./README.md

```
## 🚀 Цель проекта

Обеспечить надежную и адаптивную систему поиска и генерации ответов по нормативно-правовой документации с учётом пользовательской обратной связи и соблюдением этических стандартов ИИ.

## 🧠 Технологии

- 🦙 **Meta LLaMA 3** (GGUF формат)
- 🛡 **LLaMA Guard 2** — безопасность и фильтрация
- 🔍 **RAG pipeline** (Retriever + LLM)
- ⚙️ **FastAPI** — API-интерфейс
- 📄 **FAISS** — векторный поиск
- 📚 LangChain (опционально)
```

## 🔐 Безопасность

Используется LLaMA Guard 2 для фильтрации токсичного и небезопасного контента до и после генерации.

## 📦 Установка
Установите PostgreSQL и создайте БД raglaw и включите расширение pgvector:

```
CREATE EXTENSION IF NOT EXISTS vector;
```
Установить Ollama для запуска локальных языковых моделей:
🔗 https://ollama.com/download
Скачать нужную модель через Ollama, например Mistral/Llama3/llama3:8b-instruct-q4_0/llama3:8b-instruct-q5_1:
```
ollama pull mistral
ollama run mistral
```
Клонируйте проект и установите зависимости:
```
git clone https://github.com/truemasterskz/tanym_loop.git
cd tanym_loop
python -m venv venv
pip install -r requirements.txt
```

# File: ./app_gradio.py

```python
import gradio as gr
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.feedback_store import FeedbackStore
from rag.generator import Generator
import pickle

# Загрузка эмбеддингов
with open("embedded.pkl", "rb") as f:
    embedded = pickle.load(f)

store = VectorStore()
feedback_store = FeedbackStore()
store.insert_articles(embedded)
gen = Generator()
embedder = Embedder()

# Глобальные переменные для хранения последнего Q&A
last_question = ""

last_answer = ""

def ask(question, show_articles):
    global last_question, last_answer

    query_emb = embedder.embed_query(question)

    results = store.search_similar(query_emb, top_k=2)
    context_chunks = [f"{r['title']}\n{r['text']}" for r in results]


    answer = gen.generate_answer(question, context_chunks)

    last_question = question
    last_answer = answer

    links_block = "\n\n📎 **Полезные ссылки:**\n"
    for r in results:
        links_block += f"- [{r['title']}]({r['egov_link']})\n"
        if r['egov_link_kaz']:
            links_block += f"  [Қазақша сілтеме]({r['egov_link_kaz']})\n"

    if show_articles:
        similar_articles = "\n\n".join(context_chunks) + links_block
        return answer, similar_articles
    else:
        return answer, links_block

def save_feedback(rating, comment, correct_answer):
    if not last_question or not last_answer:
        return "❌ Вопрос и ответ не найдены. Сначала задайте вопрос."

    query_emb = embedder.embed_query(last_question)

    if correct_answer and correct_answer.strip():
        if not validate_feedback_with_llm(last_question, correct_answer):
            return "⚠️ Предложенный правильный ответ не связан с вопросом. Попробуйте уточнить."

    feedback_store.insert_feedback(
        question=last_question,
        answer=last_answer,
        comment=comment,
        rating=rating,
        embedding=query_emb,
        correct_answer=correct_answer if correct_answer.strip() else None,
        source="user"
    )

    return "✅ Спасибо за ваш отзыв!"

def validate_feedback_with_llm(question, user_answer, threshold=0.75):
    import requests
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # 1. Проверка на сходство по эмбеддингам
    embed_q = embedder.embed_query(question)
    embed_a = embedder.embed_query(user_answer)

    similarity = cosine_similarity(
        np.array(embed_q).reshape(1, -1),
        np.array(embed_a).reshape(1, -1)
    )[0][0]

    if similarity >= threshold:
        return True  # Ответ достаточно близок по смыслу

    # 2. Проверка через LLM, если сходство низкое
    prompt = f"""
Ты эксперт по проверке юридических ответов.
Нужно определить, относится ли предложенный "правильный ответ" к заданному вопросу.
Связь — это когда правильный ответ действительно отвечает на суть вопроса, даже если формулировка другая.

Вопрос: {question}
Предложенный правильный ответ: {user_answer}

Ответь только "yes" если ответ действительно по теме и полезен для пользователя, иначе "no".
"""

    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )

    if resp.status_code == 200:
        return "yes" in resp.json()["response"].strip().lower()

    return False

# Интерфейс
with gr.Blocks() as demo:
    gr.Markdown("## 💼 RAG-помощник по юридическим документам")

    with gr.Row():
        question = gr.Textbox(label="Введите ваш юридический вопрос", placeholder="Например: Нарушение договора аренды...")
        show_articles = gr.Checkbox(label="Показать похожие статьи", value=True)

    with gr.Row():
        output = gr.Textbox(label="Ответ LLM", lines=5)
        articles_output = gr.Markdown(label="Похожие статьи и полезные ссылки")

    btn = gr.Button("Задать вопрос")
    btn.click(fn=ask, inputs=[question, show_articles], outputs=[output, articles_output])

    gr.Markdown("### ✍️ Оставьте обратную связь")

    with gr.Row():
        rating = gr.Slider(minimum=1, maximum=5, step=1, label="Оценка ответа (1 — плохо, 5 — отлично)")
        comment = gr.Textbox(label="Комментарий", placeholder="Что было полезно или неполно...")
        correct_answer_box = gr.Textbox(label="Правильный ответ", placeholder="Введите корректный ответ, если наш ответ не устроил")

    feedback_btn = gr.Button("Отправить отзыв")
    feedback_output = gr.Textbox(label="", lines=1, max_lines=1)

    feedback_btn.click(fn=save_feedback, inputs=[rating, comment, correct_answer_box], outputs=[feedback_output])

demo.launch()

```

# File: ./server.py

```python
import os
import json
from typing import List, Optional
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile
import jwt, bcrypt, datetime
import smtplib
from email.mime.text import MIMEText
import secrets

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.feedback_store import FeedbackStore
from rag.generator import Generator
import rag.loader as loader
from voicing.api import router as voicing_router

import pickle

APP_PORT = int(os.environ.get("PORT", "8000"))
DATA_DIR = os.path.abspath("./data/uploads")
STATIC_DIR = os.path.abspath("./static")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

JWT_SECRET = os.getenv("JWT_SECRET", "change_me")
JWT_ALGO = "HS256"
JWT_EXP_MINUTES = 24*60


EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 465))
EMAIL_USE_SSL = os.getenv("EMAIL_USE_SSL", 'false').lower() in ('true', '1', 't')
EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER")
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD")
DEFAULT_FROM_EMAIL = os.getenv("DEFAULT_FROM_EMAIL") or EMAIL_HOST_USER

VERIFICATION_TOKEN_EXP_HOURS = 24


def generate_verification_token():
    """Генерирует криптостойкий токен для подтверждения."""
    return secrets.token_urlsafe(32)


def send_verification_email(to_email: str, token: str, user_name: Optional[str] = "Пользователь"):
    """Отправляет письмо с ссылкой для подтверждения."""

    # 🚨 Замените этот базовый URL на актуальный, где работает ваш фронтенд
    FRONTEND_APP_URL = os.getenv("FRONTEND_APP_URL", "http://localhost:8000")

    verification_link = f"{FRONTEND_APP_URL}/verify?token={token}&email={to_email}"

    subject = "Tanym Loop: Подтверждение вашего Email"
    body = f"""
Здравствуйте, {user_name}!

Спасибо за регистрацию в Tanym Loop.
Пожалуйста, подтвердите ваш адрес электронной почты, перейдя по ссылке ниже:

{verification_link}

Если вы открыли это письмо на другом устройстве (например, на телефоне), а регистрировались на компьютере — скопируйте ссылку и вставьте её в адресную строку браузера на устройстве, где вы регистрировались (или откройте письмо на том же устройстве и нажмите на ссылку).

Эта ссылка действительна в течение {VERIFICATION_TOKEN_EXP_HOURS} часов.

Если вы не регистрировались на нашем сервисе, просто проигнорируйте это письмо.

С уважением,
Команда Tanym Loop
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = DEFAULT_FROM_EMAIL
    msg["To"] = to_email

    try:
        if EMAIL_USE_SSL:
            server = smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT)
        else:
            server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
            server.starttls()  # Используем STARTTLS, если не SSL

        server.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
        server.sendmail(DEFAULT_FROM_EMAIL, to_email, msg.as_string())
        server.quit()
        print(f"✅ Отправлено письмо подтверждения на {to_email}")
        return True
    except Exception as e:
        print(f"❌ Ошибка отправки email на {to_email}: {e}")
        # В случае ошибки, лучше позволить регистрации пройти, но вернуть ошибку отправки
        return False

def create_access_token(user_id):
    payload = {"sub": str(user_id), "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=JWT_EXP_MINUTES)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth schema")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        user_id = int(payload["sub"])
        # можно вернуть user dict, но для простоты — id
        return {"id": user_id}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

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
gen = Generator(embedder=embedder, feedback_store=feedback_store)


def atomic_write_pickle(obj, path):
    dirn = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirn, prefix=".tmp_emb_")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except:
                pass


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

app = FastAPI(title="Tanym Loop API", version="0.3")
app.include_router(voicing_router)

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
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return HTMLResponse(
        "<h3>Помести index.html в ./static и открой <a href='/'>/</a></h3>"
    )


@app.post("/register")
def register(email: str = Form(...), password: str = Form(...), name: str = Form(None)):
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    # Подтверждение
    verification_token = generate_verification_token()
    token_created_at = datetime.datetime.utcnow()

    try:
        with store.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email, password_hash, name, verification_token, token_created_at) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (email, pw_hash, name, verification_token, token_created_at)
            )
            uid = cur.fetchone()[0]
        store.conn.commit()
    except Exception as e:
        store.conn.rollback()
        if "unique constraint" in str(e).lower():
            raise HTTPException(status_code=400, detail="Этот email уже зарегистрирован.")
        raise HTTPException(status_code=400, detail=f"Registration failed: {e}")

    email_sent = send_verification_email(email, verification_token, name)

    return {
        "ok": True,
        "user_id": uid,
        "message": "Регистрация успешна! Проверьте почту для подтверждения аккаунта.",
        "email_sent": email_sent
    }


@app.post("/login")
def login(email: str = Form(...), password: str = Form(...)):
    try:
        with store.conn.cursor() as cur:
            cur.execute("SELECT id, password_hash, is_verified FROM users WHERE email=%s", (email,))
            row = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not row:
        raise HTTPException(status_code=401, detail="Неверные учетные данные")
    uid, pw_hash, is_verified = row

    if not is_verified:
        raise HTTPException(status_code=403, detail="Аккаунт не подтвержден. Проверьте ваш email.")

    if not bcrypt.checkpw(password.encode(), pw_hash.encode()):
        raise HTTPException(status_code=401, detail="Неверные учетные данные")

    token = create_access_token(uid)
    return {"ok": True, "token": token, "user_id": uid}


@app.get("/verify", response_class=HTMLResponse)
def verify_email(token: str, email: str):
    try:
        with store.conn.cursor() as cur:
            # 1. Ищем пользователя по токену и email
            cur.execute("SELECT id, token_created_at FROM users WHERE email=%s AND verification_token=%s",
                        (email, token))
            row = cur.fetchone()

            if not row:
                return HTMLResponse(
                    "<h2>❌ Ошибка: Неверный токен или email.</h2><p>Возможно, вы уже подтвердили свой аккаунт или ссылка устарела.</p>")

            uid, token_created_at = row

            # 2. Проверяем срок действия токена (24 часа)
            expiration_time = token_created_at + datetime.timedelta(hours=VERIFICATION_TOKEN_EXP_HOURS)
            if datetime.datetime.utcnow() > expiration_time:
                return HTMLResponse(
                    "<h2>❌ Ошибка: Срок действия ссылки истек.</h2><p>Пожалуйста, войдите в систему, чтобы запросить новую ссылку.</p>")

            # 3. Обновляем статус: is_verified = TRUE и очищаем токен
            cur.execute(
                "UPDATE users SET is_verified=TRUE, verification_token=NULL, token_created_at=NULL WHERE id=%s",
                (uid,)
            )
        store.conn.commit()

        # 4. Перенаправляем на главную страницу с сообщением об успехе
        return HTMLResponse(f"""
            <script>
                localStorage.setItem('verification_success', 'true');
                window.location.href = '/?verified=success';
            </script>
            <h2>✅ Аккаунт подтвержден!</h2>
            <p>Вы будете перенаправлены на главную страницу...</p>
        """)

    except Exception as e:
        store.conn.rollback()
        print(f"Verification error: {e}")
        return HTMLResponse(
            "<h2>❌ Ошибка подтверждения аккаунта.</h2><p>Попробуйте снова или свяжитесь со службой поддержки.</p>")



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


# ====== Documents management ======
@app.get("/documents")
def list_documents(current_user: dict = Depends(get_current_user)):
    try:
        docs = store.list_documents(owner_id=current_user['id'])
        # Убедимся, что поля есть и корректны
        safe_docs = []
        for d in docs:
            safe_docs.append({
                "filename": d.get("filename") or "",
                "chunks": int(d.get("chunks") or 0),
                "source": d.get("source") or ""
            })
        return {"ok": True, "documents": safe_docs}
    except Exception as e:
        import traceback; traceback.print_exc()
        # Возвращаем human-friendly сообщение и лог в stdout
        return {"ok": False, "error": str(e)}


@app.post("/documents/delete")
def delete_document(filename: str = Form(...), current_user: dict = Depends(get_current_user)):
    """
    Удаляет все записи, у которых filename == filename.
    Также пытается удалить сам файл и синхронизировать embedded.pkl.
    """
    try:
        # 1. Удаляем из базы все чанки по этому файлу
        deleted = store.delete_documents_by_filename(filename, owner_id=current_user['id'])
        print(f"[DB] Удалено {deleted} записей для файла {filename}")

        # 2. Удаляем сам физический файл, если он есть
        file_path = os.path.join(DATA_DIR, filename)
        file_removed = False
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                file_removed = True
                print(f"[FS] Удалён файл {file_path}")
            except Exception as e:
                print(f"[FS] Не удалось удалить файл {file_path}: {e}")

        # 3. Обновляем (или удаляем) embedded.pkl
        cache_path = "embedded.pkl"
        cache_removed = False
        cache_updated = False
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                # фильтруем: оставляем всё, кроме этого файла
                new_cached = [c for c in cached if not (c.get("filename") == filename and c.get("owner_id") == current_user['id'])]
                if len(new_cached) != len(cached):
                    with open(cache_path, "wb") as f:
                        pickle.dump(new_cached, f)
                    cache_updated = True
                    print(f"[CACHE] Обновлён embedded.pkl — удалены записи для {filename}")
                else:
                    print(f"[CACHE] В embedded.pkl не найдено записей для {filename}")
            except Exception as e:
                # fallback — если файл битый, удалим его
                print(f"[CACHE] Ошибка при обновлении embedded.pkl: {e}, удаляю кэш.")
                try:
                    os.remove(cache_path)
                    cache_removed = True
                except:
                    pass
        else:
            print("[CACHE] embedded.pkl отсутствует — пропускаем обновление.")

        # 4. Возвращаем детальный ответ
        return {
            "ok": True,
            "filename": filename,
            "deleted_from_db": deleted,
            "file_removed": file_removed,
            "cache_updated": cache_updated,
            "cache_removed": cache_removed
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/download")
def download_document(filename: str):
    """
    Отдаёт файл из DATA_DIR по имени (если он там есть).
    """
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

# Кэш
@app.post("/rebuild_cache")
def rebuild_cache(current_user: dict = Depends(get_current_user)):
    """
    Пересобирает embedded.pkl из текущих записей в таблице documents.
    Используется для гарантии синхронизации кэша с БД.
    """
    try:
        rows = store.get_all_documents(owner_id=current_user['id'])
        cached = []
        for title, content, egov_link, egov_link_kaz, source, filename in rows:
            cached.append({
                "title": title,
                "text": content,
                "egov_link": egov_link,
                "egov_link_kaz": egov_link_kaz,
                "source": source,
                "filename": filename,
                "owner_id": current_user['id'],
            })
        with open("embedded.pkl", "wb") as f:
            pickle.dump(cached, f)
        return {"ok": True, "count": len(cached)}
    except Exception as e:
        import traceback;
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ====== Q&A (КРИТИЧЕСКИ УЛУЧШЕН) ======
@app.post("/ask")
async def ask(question: str = Form(...), show_articles: bool = Form(True), current_user: dict = Depends(get_current_user)):
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
    results = store.search_similar(query_emb, top_k=3, owner_id=current_user['id'])

    # 3. КРИТИЧНО: Фильтруем нерелевантные результаты
    # Настраиваемый порог для similarity (0..1)
    # Рекомендация: начать с 0.45..0.6 и отладить с /debug_sim

    RELEVANCE_SIM_THRESHOLD = 0.45
    relevant_results = [r for r in results if r.get('similarity', 0.0) >= RELEVANCE_SIM_THRESHOLD]

    print(f"[ASK] found={len(results)} relevant(sim>={RELEVANCE_SIM_THRESHOLD})={len(relevant_results)}")
    print("[ASK] raw rows (id, distance, raw_cosine, similarity):",
          [(r['id'], r['distance'], r.get('raw_cosine'), r.get('similarity')) for r in results])

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
    answer = gen.generate_answer(question, context_chunks, user_id=current_user['id'])

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


@app.post("/debug_sim")
def debug_sim(query: str = Form(...), top_k: int = Form(10)):
    """
    Возвращает raw distances и similarity для top_k ближайших — помогает подобрать порог.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    query_emb = embedder.embed_query(query)
    results = store.search_similar(query_emb, top_k=top_k)

    # Возвращаем только диагностическую таблицу
    diag = [
        {
            "id": r["id"],
            "title": r["title"],
            "distance": r["distance"],
            "raw_cosine": 1.0 - float(r["distance"]),
            "similarity": max(0.0, min(1.0, (1.0 - float(r["distance"]) + 1.0) / 2.0))
        }
        for r in results
    ]
    return {"ok": True, "debug": diag}


def validate_feedback_with_llm(question: str, user_answer: str, threshold: float = 0.75) -> tuple[bool, str]:
    """
    Двухуровневая валидация:
      1) быстрый check — косинусное сходство эмбеддингов
      2) если не уверены — LLM (локальная модель) даёт yes/no + причину

    Возвращает (is_valid, reason)
    """
    try:
        embed_q = embedder.embed_query(question)
        embed_a = embedder.embed_query(user_answer)

        similarity = cosine_similarity(
            np.array(embed_q).reshape(1, -1),
            np.array(embed_a).reshape(1, -1)
        )[0][0]

        # Логируем
        print(f"[VALIDATION] Embedding similarity={similarity:.3f} (threshold={threshold})")

        if similarity >= threshold:
            return True, f"Прошло по эмбеддингам (сходство: {int(similarity*100)}%)."

        # LLM fallback
        prompt = f"""
Ты эксперт по проверке юридических ответов.
Определи, относится ли предложенный "правильный ответ" к заданному вопросу.
Если да — ответы: "yes | краткая причина"
Если нет — "no | краткая причина".

Вопрос: {question}
Предложенный правильный ответ: {user_answer}
"""
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3:8b-instruct-q4_0", "prompt": prompt, "stream": False},
                timeout=60
            )
            if resp.status_code == 200:
                llm_text = resp.json().get("response", "").strip().lower()
                # Ожидаем "yes|причина" или "no|причина"
                if "|" in llm_text:
                    verdict, reason = [p.strip() for p in llm_text.split("|", 1)]
                else:
                    verdict = llm_text.split()[0] if llm_text else "no"
                    reason = llm_text

                is_valid = verdict.startswith("yes")
                return is_valid, f"LLM: {reason}"
        except Exception as e:
            print(f"[VALIDATION] LLM request failed: {e}")
            # fallback: принимать при посредственном сходстве
            if similarity >= 0.5:
                return True, "LLM недоступен, принят мягкий порог эмбеддингов."

    except Exception as e:
        print(f"[VALIDATION] Error: {e}")

    return False, f"Не прошёл валидацию (сходство эмбеддингов: {int(similarity*100) if 'similarity' in locals() else 0}%)."


# ====== FEEDBACK ======
@app.post("/feedback")
async def feedback(
        rating: int = Form(...),
        comment: str = Form(""),
        correct_answer: str = Form(""),
        question: Optional[str] = Form(None),
        answer: Optional[str] = Form(None),
        current_user: dict = Depends(get_current_user)
):
    """
    Сохраняет фидбек; если указан correct_answer — валидируем его и сохраняем ТОЛЬКО если валидно.
    Возвращаем подробный результат: correct_answer_saved, validation_reason.
    """
    if not question or not answer:
        return {"ok": False, "msg": "❌ Сначала задайте вопрос и получите ответ."}

    print("[FEEDBACK] New feedback incoming:", {"question": question[:80], "rating": rating})

    query_emb = embedder.embed_query(question)
    correct_answer_validated = None
    validation_reason = None

    # 1) Валидация correct_answer (если указан)
    if correct_answer and correct_answer.strip():
        try:
            is_valid, reason = validate_feedback_with_llm(question, correct_answer.strip())
            validation_reason = reason
            if is_valid:
                correct_answer_validated = correct_answer.strip()
                print("[FEEDBACK] correct_answer validated:", validation_reason)
            else:
                print("[FEEDBACK] correct_answer NOT validated:", validation_reason)
        except Exception as e:
            # на случай непредвиденной ошибки в валидаторе
            validation_reason = f"Ошибка валидации: {e}"
            print("[FEEDBACK] Validation function error:", e)

    # 2) Дедупликация: не записываем точные дубликаты
    try:
        # приведём None -> '' для сравнения
        cmp_correct = correct_answer_validated if correct_answer_validated is not None else ""
        check_sql = """
            SELECT 1 FROM feedback
            WHERE question = %s AND answer = %s AND COALESCE(correct_answer, '') = %s AND rating = %s AND owner_id = %s
            LIMIT 1
        """
        feedback_store.cursor.execute(check_sql, (question, answer, cmp_correct, int(rating), current_user['id']))
        if feedback_store.cursor.fetchone():
            print("[FEEDBACK] Duplicate detected — skipping insert")
            return {
                "ok": True,
                "msg": "⚠️ Похожий отзыв уже сохранён — дубликат не добавлен.",
                "correct_answer_saved": bool(correct_answer_validated),
                "validation_reason": validation_reason or ""
            }
    except Exception as e:
        # если проверка дубликата упала — логируем, но не блокируем процесс вставки
        print("[FEEDBACK] Deduplication check failed, continuing with insert. Error:", e)

    # 3) Вставляем фидбек (correct_answer только если валидирован)
    try:
        fid = feedback_store.insert_feedback(
            question=question,
            answer=answer,
            comment=comment or "",
            rating=int(rating),
            embedding=query_emb,
            correct_answer=correct_answer_validated,
            source="user",
            owner_id=current_user['id'],
        )
        if fid:
            print("[FEEDBACK] Saved to DB, id:", fid,
                  {"question": question[:80], "rating": rating, "correct_saved": bool(correct_answer_validated)})
        else:
            # вставка, возможно, прошла, но id не был возвращен — логируем это как предупреждение
            print("[FEEDBACK] Saved to DB, but no id returned (fid is None).")
    except Exception as e:
        print("[FEEDBACK] Insert failed:", e)
        return {"ok": False, "msg": f"❌ Ошибка сохранения отзыва: {e}"}

    # 4) Ответ клиенту с деталями валидации
    friendly_reason = ""
    if validation_reason:
        if "llm" in validation_reason.lower() or "недоступ" in validation_reason.lower():
            friendly_reason = "Проверка выполнена автоматически."
        else:
            friendly_reason = validation_reason

    msg = "✅ Спасибо за ваш отзыв!"
    if friendly_reason:
        msg += f" / Валидация: {friendly_reason}"

    return {
        "ok": True,
        "msg": msg,
        "correct_answer_saved": bool(correct_answer_validated),
        "validation_reason": friendly_reason
    }



# ====== INGEST (ПОЛНЫЙ ПАЙПЛАЙН) ======
@app.post("/ingest")
async def ingest(
        files: List[UploadFile] = File(...),
        source_tag: str = Form("user"),
        current_user: dict = Depends(get_current_user),
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
            stats = store.insert_articles(embedded_articles, owner_id=current_user['id'])

            # 6. Обновление кэша (опционально)
            try:
                with open("embedded.pkl", "rb") as f:
                    cached = pickle.load(f)
            except:
                cached = []

            for a in embedded_articles:
                # ensure owner_id included
                a_copy = dict(a)
                a_copy["owner_id"] = current_user['id']
                cached.append(a_copy)

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
def get_stats(current_user: dict = Depends(get_current_user)):
    """
    Возвращает детальную статистику по базе знаний.

    Полезно для мониторинга и отладки.
    """
    db_stats = store.get_stats(owner_id=current_user['id'])

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
```

# File: ./reqs.txt

```
aiofiles==23.2.1
annotated-types==0.7.0
anyio==4.9.0
av==15.1.0
bcrypt==5.0.0
certifi==2025.7.14
charset-normalizer==3.4.2
click==8.1.8
colorama==0.4.6
coloredlogs==15.0.1
contourpy==1.3.0
ctranslate2==4.6.2
cycler==0.12.1
dotenv==0.9.9
et_xmlfile==2.0.0
exceptiongroup==1.3.0
fastapi==0.116.1
faster-whisper==1.2.1
ffmpy==0.6.1
filelock==3.18.0
flatbuffers==25.12.19
fonttools==4.59.0
fsspec==2025.7.0
gradio==4.44.1
gradio_client==1.3.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==0.33.5
humanfriendly==10.0
idna==3.10
importlib_resources==6.5.2
Jinja2==3.1.6
joblib==1.5.1
kiwisolver==1.4.7
lxml==6.0.0
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.9.4
mdurl==0.1.2
mpmath==1.3.0
networkx==3.2.1
numpy>=2.0.2
onnxruntime==1.19.2
openpyxl==3.1.5
orjson==3.11.1
packaging==25.0
pandas==2.3.1
pillow==10.4.0
protobuf==6.33.2
psycopg2-binary>=2.9.10
PyAudio==0.2.14
pydantic==2.11.7
pydantic_core==2.33.2
pydub==0.25.1
Pygments==2.19.2
PyJWT==2.10.1
PyMuPDF==1.26.3
pyparsing==3.2.3
python-dateutil==2.9.0.post0
python-docx==1.2.0
python-dotenv==1.1.1
python-multipart==0.0.20
pyttsx3==2.99
pytz==2025.2
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.4
rich==14.1.0
ruff==0.12.7
safetensors==0.5.3
scikit-learn==1.6.1
scipy==1.13.1
semantic-version==2.10.0
sentence-transformers==5.0.0
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
starlette==0.47.2
sympy==1.14.0
threadpoolctl==3.6.0
tokenizers==0.21.2
tomlkit==0.12.0
torch>=2.0.0
tqdm==4.67.1
transformers>=4.40.0
typer==0.16.0
typing-inspection==0.4.1
typing_extensions==4.14.1
tzdata==2025.2
urllib3==2.5.0
uvicorn==0.35.0
websockets==12.0
zipp==3.23.0
```

# File: ./main.py

```python
from rag.loader import load_articles
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.generator import Generator
from rag.feedback_store import FeedbackStore
import os
import pickle

EMBED_CACHE = "embedded.pkl"
EXCEL_FILE = "data/data_for_rag.xlsx"

print("=" * 60)
print("🚀 Инициализация Tanym Loop")
print("=" * 60)


def load_all_articles():
    """Загружает статьи из Excel файла"""
    return load_articles(EXCEL_FILE)


# ============================================
# Шаг 1: Загрузка или создание эмбеддингов
# ============================================
if os.path.exists(EMBED_CACHE):
    print("\n📦 Загрузка эмбеддингов из кэша...")
    with open(EMBED_CACHE, "rb") as f:
        embedded = pickle.load(f)
    print(f"✅ Загружено {len(embedded)} статей из кэша (embedded.pkl)")
else:
    print("\n📄 Загрузка статей из Excel...")
    articles = load_all_articles()
    print(f"✅ Загружено {len(articles)} статей из {EXCEL_FILE}")

    print("\n🔄 Создание эмбеддингов...")
    print("   (Это займёт некоторое время при первом запуске)")
    embedder = Embedder()
    embedded = embedder.embed_articles(articles)

    print("\n💾 Сохранение эмбеддингов в кэш...")
    with open(EMBED_CACHE, "wb") as f:
        pickle.dump(embedded, f)
    print(f"✅ Эмбеддинги сохранены в {EMBED_CACHE}")

# ============================================
# Шаг 2: Загрузка в векторную БД
# ============================================
print("\n📊 Подключение к PostgreSQL...")
store = VectorStore()
feedback = FeedbackStore()

print("\n🔄 Индексация документов в БД...")
stats = store.insert_articles(embedded)
print(f"✅ Индексация завершена:")
print(f"   - Вставлено: {stats['inserted']}")
print(f"   - Пропущено (дубликаты): {stats['skipped']}")
print(f"   - Ошибки: {stats['errors']}")

# Статистика БД
db_stats = store.get_stats()
print(f"\n📈 Статистика базы данных:")
print(f"   - Всего документов: {db_stats['total_documents']}")
print(f"   - Уникальных источников: {db_stats['unique_sources']}")
print(f"   По источникам:")
for src, cnt in db_stats['by_source'].items():
    print(f"      • {src}: {cnt}")

print("\n" + "=" * 60)
print("🎯 Система готова к работе!")
print("=" * 60)

# ============================================
# Шаг 3: Интерактивный поиск и ответы
# ============================================
embedder_instance = Embedder()
gen = Generator()

# Порог релевантности (как в server.py)
RELEVANCE_THRESHOLD = 0.7

while True:
    print("\n" + "-" * 60)
    query = input("❓ Введите ваш вопрос (или 'exit' для выхода): ").strip()

    if query.lower() in ['exit', 'quit', 'выход']:
        print("\n👋 До свидания!")
        break

    if not query:
        print("⚠️  Пожалуйста, введите вопрос")
        continue

    # Эмбеддинг запроса
    query_emb = embedder_instance.embed_query(query)

    # Поиск похожих (top_k=5 для фильтрации)
    results = store.search_similar(query_emb, top_k=5)

    # Фильтрация по релевантности
    relevant_results = [r for r in results if r['distance'] < RELEVANCE_THRESHOLD]

    print(f"\n🔍 Результаты поиска:")
    print(f"   Найдено: {len(results)} документов")
    print(f"   Релевантных (distance < {RELEVANCE_THRESHOLD}): {len(relevant_results)}")

    if not relevant_results:
        print("\n❌ К сожалению, не найдено релевантных документов.")
        print("💡 Попробуйте:")
        print("   - Переформулировать вопрос")
        print("   - Использовать другие ключевые слова")
        continue

    # Используем топ-2 релевантных
    top_results = relevant_results[:2]

    print(f"   Использовано для ответа: {len(top_results)}")
    print(f"   Средняя релевантность: {int(sum(r['similarity'] for r in top_results) / len(top_results) * 100)}%")

    # Показываем источники
    print(f"\n📚 Использованные источники:")
    for idx, r in enumerate(top_results, 1):
        similarity_pct = int(r['similarity'] * 100)
        distance = r['distance']

        # Определяем уровень релевантности
        if similarity_pct >= 80:
            badge = "🟢 ОТЛИЧНО"
        elif similarity_pct >= 60:
            badge = "🟡 ХОРОШО"
        else:
            badge = "🟠 СРЕДНЕ"

        print(f"\n   {idx}. {badge} (релевантность: {similarity_pct}%, distance: {distance:.3f})")
        print(f"      📝 {r['title']}")

        # Превью текста
        preview = r['text'][:150] + "..." if len(r['text']) > 150 else r['text']
        print(f"      💬 {preview}")

        # Метаданные
        print(f"      📁 Источник: {r['source']}")

        # Ссылки
        if r.get('egov_link'):
            print(f"      🔗 Ссылка: {r['egov_link']}")
        if r.get('egov_link_kaz'):
            print(f"      🔗 Қазақша: {r['egov_link_kaz']}")

    # Генерация ответа
    print(f"\n🤖 Генерация ответа...")
    context_chunks = [f"{r['title']}\n{r['text']}" for r in top_results]
    answer = gen.generate_answer(query, context_chunks)

    print(f"\n" + "=" * 60)
    print("📋 ОТВЕТ:")
    print("=" * 60)
    print(answer)
    print("=" * 60)

# Закрываем соединение
store.close()
print("\n✅ Соединение с БД закрыто")
```

# File: ./test/test_models.py

```python
import sys
import os
import time
import openpyxl
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.generator import Generator
from rag.embedder import Embedder
from rag.vectorstore import VectorStore


questions = [
    "Что грозит за езду без техосмотра автомобиля?",
    "Как оформить доверенность у нотариуса?",
    "Какие документы нужны для регистрации брака?"
]

models = [
    "mistral",
    "llama3",
    "llama3:8b-instruct-q4_0",
    "llama3:8b-instruct-q5_1"
]

with open("embedded.pkl", "rb") as f:
    embedded = pickle.load(f)

store = VectorStore()
store.insert_articles(embedded)
embedder = Embedder()

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Сравнение моделей для RAG"

ws.append(["Модель", "Вопрос", "Ответ", "Время", "Оценка точности (1-5)", "Полезность (1-5)"])

for model in models:
    gen = Generator(model_name=model)

    for q in questions:
        query_emb = embedder.embed_query(q)
        results = store.search_similar(query_emb, top_k=2)
        context_chunks = [f"{r['title']}\n{r['text']}" for r in results]

        start = time.time()
        try:
            answer = gen.generate_answer(q, context_chunks)
        except Exception as e:
            answer = f"(Error: {str(e)})"
        duration = time.time() - start

        ws.append([model, q, answer, round(duration, 2), "", ""])

output_dir = os.path.join(os.path.dirname(__file__), "test")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "model_comparison_with_context.xlsx")
wb.save(output_path)

print(f"✅ Results saved in {output_path}")
```

# File: ./voicing/__init__.py

```python
# voice/__init__.py
"""
Voice module initialization
"""
from .api import router

__all__ = ['router']
```

# File: ./voicing/api.py

```python
# voice/api.py
"""
Голосовой модуль для Tanym Loop.
Поддерживает Speech-to-Text через Faster Whisper (офлайн).
"""

import os
import tempfile
import wave
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from faster_whisper import WhisperModel

router = APIRouter(prefix="/voice", tags=["voice"])

# Инициализация модели Whisper
# Используем smaller model для быстрой работы, можно заменить на "large-v3"
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

print(f"🎤 Инициализация Whisper модели: {WHISPER_MODEL_SIZE}")

try:
    whisper_model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE
    )
    print("✅ Whisper модель загружена")
except Exception as e:
    print(f"⚠️  Whisper модель не загружена: {e}")
    whisper_model = None


def transcribe_audio(audio_path: str) -> tuple[str, str, float]:
    """
    Распознаёт аудио файл через Faster Whisper.

    Returns:
        (text, language, confidence)
    """
    if not whisper_model:
        raise HTTPException(
            status_code=503,
            detail="Whisper model not initialized. Install faster-whisper."
        )

    try:
        segments, info = whisper_model.transcribe(
            audio_path,
            beam_size=5,
            language="ru",  # Можно сделать auto-detect
            vad_filter=True  # Фильтрация пауз
        )

        text = " ".join([segment.text for segment in segments])

        return text.strip(), info.language, info.language_probability

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")


@router.post("/stt")
async def speech_to_text(
        audio: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)")
):
    """
    Speech-to-Text endpoint.
    Принимает аудио файл и возвращает распознанный текст.

    **Поддерживаемые форматы:** WAV, MP3, M4A, FLAC
    """

    if not whisper_model:
        raise HTTPException(
            status_code=503,
            detail="Speech recognition unavailable. Install faster-whisper: pip install faster-whisper"
        )

    # Проверка расширения файла
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    file_ext = os.path.splitext(audio.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file_ext}. Allowed: {allowed_extensions}"
        )

    # Сохраняем временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Распознаём
        text, lang, confidence = transcribe_audio(tmp_path)

        if not text:
            return {
                "ok": False,
                "text": "",
                "message": "Не удалось распознать речь. Попробуйте говорить громче."
            }

        return {
            "ok": True,
            "text": text,
            "language": lang,
            "confidence": round(confidence, 2),
            "message": "Распознавание успешно"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Удаляем временный файл
        try:
            os.unlink(tmp_path)
        except:
            pass


@router.get("/health")
def voice_health():
    """
    Проверка доступности голосового модуля.
    """
    return {
        "whisper_available": whisper_model is not None,
        "model": WHISPER_MODEL_SIZE if whisper_model else None,
        "device": WHISPER_DEVICE if whisper_model else None
    }
```

# File: ./static/app2.js

```
// /static/app2.js
const API_BASE = ""; // Базовый URL API (оставь "" если бек на том же origin)

// --- Auth helpers & UI ---
const TOKEN_KEY = "tanym_token";
const USER_EMAIL_KEY = "tanym_user_email";
const USER_ID_KEY = "tanym_user_id";

const systemMessageEl = document.getElementById("systemMessage");

function showSystemMessage(message, type = 'info') {
    if (!systemMessageEl) return;
    systemMessageEl.innerHTML = message;
    systemMessageEl.style.display = "block";

    // Простая стилизация
    systemMessageEl.style.backgroundColor = type === 'success' ? '#e6ffed' : type === 'error' ? '#fff0f6' : '#fffbe6';
    systemMessageEl.style.borderColor = type === 'success' ? '#b7eb8f' : type === 'error' ? '#ffadd2' : '#ffe58f';
}

function clearSystemMessage() {
    if (systemMessageEl) systemMessageEl.style.display = "none";
}

if (localStorage.getItem('verification_success') === 'true') {
    showSystemMessage("✅ Ваш аккаунт успешно подтвержден! Теперь вы можете войти.", 'success');
    localStorage.removeItem('verification_success');
}

function authFetch(url, opts = {}) {
  opts = opts || {};
  opts.headers = opts.headers || {};
  const token = localStorage.getItem(TOKEN_KEY);

  // Если токена нет — покажем модал и отклоним запрос с понятной ошибкой
  if (!token) {
    // UX: не спамим модал — только откроем (возможно пользователь уже видит его)
    try { openAuthModal("login"); } catch (e) {}
    return Promise.reject(new Error("not_authenticated"));
  }

  opts.headers["Authorization"] = "Bearer " + token;

  return fetch(url, opts).then(async resp => {
    if (resp.status === 401) {
      // Сервер вернул 401 — откроем модал и пробросим понятную ошибку
      try { openAuthModal("login"); } catch (e) {}
      // попытаемся прочитать тело для логирования, но не показываем пользователю JSON
      let body = null;
      try { body = await resp.text(); } catch (e) {}
      const err = new Error("unauthorized");
      err.details = body;
      throw err;
    }
    return resp;
  });
}


function saveAuth(info) {
  if (info.token) localStorage.setItem(TOKEN_KEY, info.token);
  if (info.user_id) localStorage.setItem(USER_ID_KEY, info.user_id);
  if (info.email) localStorage.setItem(USER_EMAIL_KEY, info.email);
}

function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_ID_KEY);
  localStorage.removeItem(USER_EMAIL_KEY);
}

// --- DOM elements (auth modal + header buttons) ---
const modalEl = document.getElementById("authModal");
const loginForm = document.getElementById("loginForm");
const registerForm = document.getElementById("registerForm");

const switchToLoginBtn = document.getElementById("switchToLogin");
const switchToRegisterBtn = document.getElementById("switchToRegister");
const authCloseBtn = document.getElementById("authCloseBtn");

const authEmail = document.getElementById("authEmail");
const authPassword = document.getElementById("authPassword");

const authName = document.getElementById("authName");
const regEmail = document.getElementById("regEmail");
const regPassword = document.getElementById("regPassword");

const authError = document.getElementById("authError");
const regError = document.getElementById("regError");

const authSubmitBtn = document.getElementById("authSubmitBtn");
const authCancelBtn = document.getElementById("authCancelBtn");
const regSubmitBtn = document.getElementById("regSubmitBtn");
const regCancelBtn = document.getElementById("regCancelBtn");

const showLoginBtn = document.getElementById("showLoginBtn");
const showRegisterBtn = document.getElementById("showRegisterBtn");

const userInfo = document.getElementById("userInfo");
const anonArea = document.getElementById("anonArea");
const userEmailSpan = document.getElementById("userEmail");
const logoutBtn = document.getElementById("logoutBtn");

// --- Modal open/close and switch logic ---
function openAuthModal(mode = "login") {
  if (!modalEl) return;
  // Reset errors and fields
  authError && (authError.style.display = "none");
  regError && (regError.style.display = "none");
  clearSystemMessage();

  if (mode === "login") {
    loginForm && (loginForm.style.display = "");
    registerForm && (registerForm.style.display = "none");
    switchToLoginBtn && switchToLoginBtn.classList.add("active");
    switchToRegisterBtn && switchToRegisterBtn.classList.remove("active");
    setTimeout(()=> authEmail?.focus(), 30);
  } else {
    loginForm && (loginForm.style.display = "none");
    registerForm && (registerForm.style.display = "");
    switchToLoginBtn && switchToLoginBtn.classList.remove("active");
    switchToRegisterBtn && switchToRegisterBtn.classList.add("active");
    setTimeout(()=> authName?.focus(), 30);
  }
  modalEl.classList.add("open");
  modalEl.setAttribute("aria-hidden","false");
}

function closeAuthModal() {
  if (!modalEl) return;
  modalEl.classList.remove("open");
  modalEl.setAttribute("aria-hidden","true");
  // hide error panels
  authError && (authError.style.display = "none");
  regError && (regError.style.display = "none");
}

// Close modal on overlay click or Esc
if (modalEl) {
  modalEl.addEventListener("click", (e) => {
    if (e.target === modalEl) closeAuthModal();
  });
}
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && modalEl && modalEl.classList.contains("open")) closeAuthModal();
});

// wire header buttons
showLoginBtn?.addEventListener("click", () => openAuthModal("login"));
showRegisterBtn?.addEventListener("click", () => openAuthModal("register"));
authCloseBtn?.addEventListener("click", closeAuthModal);

// switch tabs inside modal
switchToLoginBtn?.addEventListener("click", () => openAuthModal("login"));
switchToRegisterBtn?.addEventListener("click", () => openAuthModal("register"));

// cancel buttons
authCancelBtn?.addEventListener("click", closeAuthModal);
regCancelBtn?.addEventListener("click", closeAuthModal);

// expose for other scripts (old code may call openAuthModal)
window.openAuthModal = openAuthModal;
window.closeAuthModal = closeAuthModal;

// --- Auth action handlers ---
// Note: We provide the same endpoints as before (/login, /register) and reuse saveAuth/updateAuthUI
async function performLogin(email, pass) {
  authError && (authError.style.display = "none");
  if (!email || !pass) {
    if (authError) {
      authError.textContent = "Заполните email и пароль";
      authError.style.display = "block";
    }
    return;
  }
  authSubmitBtn && (authSubmitBtn.disabled = true);
  const orig = authSubmitBtn ? authSubmitBtn.innerHTML : null;
  if (authSubmitBtn) authSubmitBtn.innerHTML = "Подождите...";

  try {
    const form = new FormData();
    form.append("email", email);
    form.append("password", pass);
    const resp = await fetch(`${API_BASE}/login`, { method: "POST", body: form });
    const data = await resp.json().catch(()=>null);
    if (!resp.ok) {
        if (resp.status === 403) {
            closeAuthModal();
            showSystemMessage(data?.detail || "Аккаунт не подтвержден. Проверьте ваш email.", 'error');
            return;
        }
        throw new Error(data?.detail || data?.error || JSON.stringify(data) || `HTTP ${resp.status}`);
    }
    saveAuth({ token: data.token, user_id: data.user_id, email });
    closeAuthModal();
    updateAuthUI();
    clearSystemMessage();
    await updateStats().catch(()=>{});
    await loadUploadedFiles().catch(()=>{});
  } catch (err) {
    console.error("Login error:", err);
    if (authError) {
      authError.textContent = err.message || String(err);
      authError.style.display = "block";
    }
  } finally {
    if (authSubmitBtn) {
      authSubmitBtn.disabled = false;
      authSubmitBtn.innerHTML = orig || "Войти";
    }
  }
}

async function performRegister(name, email, pass) {
  regError && (regError.style.display = "none");
  if (!name || !email || !pass) {
    if (regError) {
      regError.textContent = "Заполните все поля";
      regError.style.display = "block";
    }
    return;
  }

  regSubmitBtn && (regSubmitBtn.disabled = true);
  const orig = regSubmitBtn ? regSubmitBtn.innerHTML : null;
  if (regSubmitBtn) regSubmitBtn.innerHTML = "Подождите...";

  try {
    const form = new FormData();
    form.append("name", name);
    form.append("email", email);
    form.append("password", pass);
    const resp = await fetch(`${API_BASE}/register`, { method: "POST", body: form });
    const data = await resp.json().catch(()=>null);
    if (!resp.ok) throw new Error(data?.detail || data?.error || JSON.stringify(data) || `HTTP ${resp.status}`);
    closeAuthModal();
    showSystemMessage(data.message || "Регистрация успешна. Проверьте почту для подтверждения аккаунта.", 'info');
  } catch (err) {
    console.error("Register error:", err);
    if (regError) {
      regError.textContent = err.message || String(err);
      regError.style.display = "block";
    }
  } finally {
    if (regSubmitBtn) {
      regSubmitBtn.disabled = false;
      regSubmitBtn.innerHTML = orig || "Зарегистрироваться";
    }
  }
}

// bind modal submit buttons
authSubmitBtn?.addEventListener("click", () => {
  const email = authEmail?.value?.trim() || "";
  const pass = authPassword?.value || "";
  performLogin(email, pass);
});
regSubmitBtn?.addEventListener("click", () => {
  const name = authName?.value?.trim() || "";
  const email = regEmail?.value?.trim() || "";
  const pass = regPassword?.value || "";
  performRegister(name, email, pass);
});

// logout
logoutBtn?.addEventListener("click", () => {
  clearAuth();
  updateAuthUI();
  loadUploadedFiles().catch(()=>{});
  updateStats().catch(()=>{});
});

// update auth UI in header
function updateAuthUI() {
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) {
    anonArea && (anonArea.style.display = "none");
    userInfo && (userInfo.style.display = "flex");
    userEmailSpan && (userEmailSpan.textContent = localStorage.getItem(USER_EMAIL_KEY) || "User");
  } else {
    anonArea && (anonArea.style.display = "flex");
    userInfo && (userInfo.style.display = "none");
  }
}

// --- main variables from original script ---
let lastQuestion = "";
let lastAnswer = "";
let lastFeedbackKey = null; // локально — предотвращение дублей в одной сессии

// --- 1. Tab switching ---
document.querySelectorAll('.tab-btn').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

// --- 2. Rating display update ---
document.getElementById('rating')?.addEventListener('input', e=>{
  document.getElementById('ratingVal').textContent = e.target.value;
});

// --- 3. Q&A Logic (Ask Button) ---
const askBtn = document.getElementById('askBtn');
const clearBtn = document.getElementById('clearBtn');

askBtn?.addEventListener('click', async ()=>{
  const q = document.getElementById('q').value.trim();
  const show = document.getElementById('showSimilar').checked;
  const ansEl = document.getElementById('ans');
  const srcEl = document.getElementById('sources');
  const metaEl = document.getElementById('ansMetadata');

  if(!q){
    ansEl.value = "❌ Введите вопрос.";
    return;
  }

  try{
    askBtn.disabled = true;
    askBtn.innerHTML = 'Думаю... <span class="spinner"></span>';

    const fd = new FormData();
    fd.append("question", q);
    fd.append("show_articles", String(show));

    // use authFetch to include token if present
    const resp = await authFetch(`${API_BASE}/ask`, { method:"POST", body: fd });
    if(!resp.ok) {
      const txt = await resp.text().catch(()=>null);
      throw new Error(`HTTP ${resp.status} ${txt || ''}`);
    }
    const data = await resp.json();

    ansEl.value = data.answer || "";

    // Metadata
    if(data.metadata){
      const m = data.metadata;
      metaEl.innerHTML = `
        <span class="pill">Найдено: ${m.found_results}</span>
        <span class="pill">Релевантных: ${m.relevant_results}</span>
        <span class="pill">Использовано: ${m.used_results}</span>
        <span class="pill">Ср. релевантность: ${(m.avg_similarity * 100).toFixed(0)}%</span>
      `;
    } else {
      metaEl.innerHTML = "";
    }

    // Sources
    srcEl.innerHTML = "";
    if(data.sources_md){
      const lines = data.sources_md.split('\n');
      let html = '';

      for(let line of lines){
        if(line.startsWith('### ')){
          const match = line.match(/релевантность:\s*(\d+)%/);
          if(match){
            const pct = parseInt(match[1]);
            let badge = 'medium';
            if(pct >= 80) badge = 'high';
            else if(pct < 60) badge = 'low';
            html += `<h3>${line.replace('### ', '')}<span class="similarity-badge ${badge}">${pct}%</span></h3>`;
          } else {
            html += `<h3>${line.replace('### ', '')}</h3>`;
          }
        } else if(line.startsWith('## ')){
          html += `<h2>${line.replace('## ', '')}</h2>`;
        } else if(line.startsWith('**') && line.endsWith('**')){
          html += `<strong>${line.replace(/\*\*/g, '')}</strong><br>`;
        } else if(line.startsWith('_') && line.endsWith('_')){
          html += `<em style="color:var(--muted)">${line.replace(/_/g, '')}</em><br>`;
        } else if(line.startsWith('🔗 [')){
          const linkMatch = line.match(/\[([^\]]+)\]\(([^)]+)\)/);
          if(linkMatch){
            html += `<a href="${linkMatch[2]}" target="_blank" style="color:var(--accent)">${line}</a><br>`;
          }
        } else if(line.trim()){
          html += `${line}<br>`;
        }
      }

      srcEl.innerHTML = html || data.sources_md;
    } else {
      srcEl.innerHTML = '<div class="hint">Источники не найдены</div>';
    }

    lastQuestion = q;
    lastAnswer = data.answer || "";

  }catch(e){
    console.error(e);
    // Разбиение по типам ошибок: наша authFetch бросает not_authenticated/unauthorized
    const msgEl = document.getElementById('ans');
    if (!msgEl) return;

    if (e && (e.message === "not_authenticated" || e.message === "unauthorized")) {
      msgEl.value = "🔐 Войдите в платформу, чтобы задать вопрос.";
      // Дополнительно подсказка в источниках
      const srcEl = document.getElementById('sources');
      if (srcEl) srcEl.innerHTML = '<div class="hint">✳️ Для работы с приватной базой знаний требуется вход. Нажмите «Войти».<\/div>';
    } else {
      msgEl.value = "⚠️ Ошибка при запросе к API: " + (e.message || e);
    }
  }finally{
    askBtn.disabled = false;
    askBtn.textContent = "Задать вопрос";
  }
});

// ====== ГОЛОСОВОЙ ВВОД (Speech-to-Text) ======

const micBtn = document.getElementById("micBtn");
const voiceStatus = document.getElementById("voiceStatus");
let mediaRecorder = null;
let audioChunks = [];

function showVoiceStatus(message, isError = false) {
  if (!voiceStatus) return;
  voiceStatus.textContent = message;
  voiceStatus.style.display = "block";
  voiceStatus.style.color = isError ? "var(--red)" : "var(--accent)";

  if (!isError) {
    setTimeout(() => {
      voiceStatus.style.display = "none";
    }, 5000);
  }
}

micBtn?.addEventListener("click", async () => {
  // Если уже идет запись - останавливаем
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    micBtn.textContent = "⏳";
    micBtn.disabled = true;
    micBtn.title = "Обработка записи...";
    showVoiceStatus("⏳ Обработка записи...");
    return;
  }

  // Начинаем новую запись
  try {
    // Запрашиваем доступ к микрофону
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });

    // Создаем MediaRecorder
    const options = { mimeType: 'audio/webm' };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
      options.mimeType = 'audio/ogg'; // Fallback для Safari
    }

    mediaRecorder = new MediaRecorder(stream, options);
    audioChunks = [];

    // Собираем аудио-чанки
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    // Когда запись остановлена - отправляем на сервер
    mediaRecorder.onstop = async () => {
      // Останавливаем микрофон
      stream.getTracks().forEach(track => track.stop());

      // Создаем blob из чанков
      const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });

      showVoiceStatus("🔄 Распознавание речи...");

      // Отправляем на сервер
      try {
        const formData = new FormData();
        const fileExtension = mediaRecorder.mimeType.includes('webm') ? 'webm' : 'ogg';
        formData.append('audio', audioBlob, `recording.${fileExtension}`);

        const resp = await authFetch(`${API_BASE}/voice/stt`, {
          method: 'POST',
          body: formData
        });

        if (!resp.ok) {
          const errorText = await resp.text().catch(() => '');
          throw new Error(errorText || `HTTP ${resp.status}`);
        }

        const data = await resp.json();

        if (data.ok && data.text) {
          // Вставляем распознанный текст в поле вопроса
          const qField = document.getElementById("q");
          qField.value = data.text;
          qField.focus();

          // Показываем уведомление с точностью
          const confPct = Math.round((data.confidence || 0.8) * 100);
          const previewText = data.text.length > 50 ? data.text.substring(0, 50) + '...' : data.text;
          showVoiceStatus(`✅ Распознано (${confPct}% уверенность): "${previewText}"`);

        } else {
          showVoiceStatus(data.message || "❌ Не удалось распознать речь. Попробуйте говорить четче.", true);
        }

      } catch (e) {
        console.error("STT Error:", e);

        if (e && (e.message === "not_authenticated" || e.message === "unauthorized")) {
          showVoiceStatus("🔐 Войдите в систему для использования голосового ввода.", true);
        } else {
          showVoiceStatus("❌ Ошибка распознавания: " + (e.message || "Неизвестная ошибка"), true);
        }
      } finally {
        micBtn.disabled = false;
        micBtn.textContent = "🎙️";
        micBtn.title = "Голосовой ввод: нажмите для записи";
        micBtn.classList.remove("recording");
      }
    };

    // Начинаем запись
    mediaRecorder.start();
    micBtn.textContent = "⏹️"; // Иконка остановки
    micBtn.title = "Идёт запись... Нажмите чтобы остановить";
    micBtn.classList.add("recording");
    showVoiceStatus("🔴 Запись... Говорите ваш вопрос (макс. 30 сек)");

    // Автоостановка через 30 секунд (защита от зависания)
    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        showVoiceStatus("⏱️ Достигнут лимит времени записи (30 сек)");
        mediaRecorder.stop();
      }
    }, 30000);

  } catch (e) {
    console.error("Microphone access error:", e);

    if (e.name === "NotAllowedError") {
      showVoiceStatus("❌ Доступ к микрофону запрещён. Разрешите доступ в настройках браузера.", true);
    } else if (e.name === "NotFoundError") {
      showVoiceStatus("❌ Микрофон не найден. Подключите микрофон и обновите страницу.", true);
    } else {
      showVoiceStatus("❌ Ошибка доступа к микрофону: " + e.message, true);
    }

    micBtn.textContent = "🎙️";
    micBtn.classList.remove("recording");
  }
});

// --- 4. Q&A Logic (Clear Button) ---
clearBtn?.addEventListener('click', ()=>{
  document.getElementById('q').value = "";
  document.getElementById('ans').value = "";
  document.getElementById('sources').innerHTML = '<div class="hint">Здесь будут показаны источники</div>';
  document.getElementById('ansMetadata').innerHTML = "";
  lastQuestion = "";
  lastAnswer = "";
});

// --- 5. Feedback Logic ---
const sendFbBtn = document.getElementById('sendFb');

sendFbBtn?.addEventListener('click', async ()=> {
  const rating = document.getElementById('rating').value;
  const comment = document.getElementById('comment').value;
  const corr = document.getElementById('corr').value.trim();
  const fbOut = document.getElementById('fbOut');

  if(!lastQuestion || !lastAnswer){
    fbOut.textContent = "❌ Сначала задайте вопрос";
    setTimeout(()=>fbOut.textContent="", 4000);
    return;
  }

  // локальная защита от одинаковых отправок подряд
  const feedbackKey = `${lastQuestion}|||${lastAnswer}|||${corr}|||${rating}`;
  if (feedbackKey === lastFeedbackKey) {
    fbOut.textContent = "⚠️ Похоже, вы уже отправляли этот отзыв (сессия). Подождите результат.";
    fbOut.style.color = "var(--yellow)";
    setTimeout(()=>fbOut.textContent="", 4000);
    return;
  }

  // UI: блокировка + сообщение о процессе
  sendFbBtn.disabled = true;
  const origBtnText = sendFbBtn.textContent;
  sendFbBtn.innerHTML = 'Отправка... <span class="spinner"></span>';
  fbOut.style.color = "var(--muted)";
  fbOut.textContent = "Проверка правильного ответа...";

  try {
    const fd = new FormData();
    fd.append("rating", rating);
    fd.append("comment", comment);
    fd.append("correct_answer", corr);
    fd.append("question", lastQuestion);
    fd.append("answer", lastAnswer);

    const resp = await authFetch(`${API_BASE}/feedback`, { method:"POST", body: fd });
    if(!resp.ok){
      const txt = await resp.text().catch(()=>null);
      throw new Error(`HTTP ${resp.status} ${txt || ''}`);
    }
    const data = await resp.json();

    // Показываем содержимое ответа от сервера — с маскированием технических причин
    let userMsg = data.msg || "OK";
    const rawReason = data.validation_reason || "";
    let friendlyReason = "";
    if (rawReason) {
      if (/llm/i.test(rawReason) || /недоступн/i.test(rawReason) || /embedd/i.test(rawReason)) {
        friendlyReason = " (Проверка выполнена автоматически.)";
      } else {
        friendlyReason = ` (${rawReason})`;
      }
    }

    fbOut.innerHTML = userMsg + (friendlyReason ? ` <span style="color:var(--muted)">${friendlyReason}</span>` : "");
    fbOut.style.color = data.ok ? "var(--green)" : "var(--red)";

    if (data.correct_answer_saved !== undefined) {
      if (data.correct_answer_saved) {
        fbOut.innerHTML += ' <strong style="color:var(--green)">✅ Правильный ответ сохранён</strong>';
      } else if (data.validation_reason) {
        fbOut.innerHTML += ' <span style="color:var(--yellow)">⚠️ ' + (data.validation_reason && !/llm/i.test(data.validation_reason) ? data.validation_reason : 'Правильный ответ не сохранился') + '</span>';
      }
    }

    lastFeedbackKey = feedbackKey;

    if(data.ok){
      document.getElementById('rating').value = 5;
      document.getElementById('ratingVal').textContent = 5;
      document.getElementById('comment').value = "";
      document.getElementById('corr').value = "";
    }

  } catch(e) {
    console.error(e);
    fbOut.textContent = "⚠️ Ошибка отправки: " + (e.message || e);
    fbOut.style.color = "var(--red)";
  } finally {
    sendFbBtn.disabled = false;
    sendFbBtn.textContent = origBtnText || "Отправить отзыв";
    setTimeout(()=>fbOut.textContent="", 8000);
  }
});

// --- 6. Document Upload / Indexing Logic ---
const drop = document.getElementById('drop');
const fileInput = document.getElementById('fileInput');
const indexBtn = document.getElementById('indexBtn');
const fileListEl = document.getElementById('fileList');
const idxOut = document.getElementById('idxOut');
let uploadedFiles = [];

function showNames(){
  const names = uploadedFiles.map(f=>f.name).join(", ");
  fileListEl.textContent = uploadedFiles.length
    ? `📚 Готово к загрузке: ${names}`
    : "";
}

// Drag & Drop event handlers
['dragenter','dragover'].forEach(ev=>{
  drop?.addEventListener(ev, e=>{ e.preventDefault(); drop.classList.add('drag');});
});

['dragleave','drop'].forEach(ev=>{
  drop?.addEventListener(ev, e=>{ e.preventDefault(); drop.classList.remove('drag');});
});

drop?.addEventListener('drop', e=>{
  uploadedFiles = uploadedFiles.concat([...e.dataTransfer.files]);
  showNames();
});

drop?.addEventListener('click', ()=> fileInput?.click());

fileInput?.addEventListener('change', e=>{
  uploadedFiles = uploadedFiles.concat([...e.target.files]);
  showNames();
  fileInput.value = null;
});

indexBtn?.addEventListener('click', async ()=>{
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) {
    openAuthModal("login");
    return;
  }

  const sourceTag = document.getElementById('sourceTag').value;

  if(!uploadedFiles.length){
    idxOut.innerHTML = '<div class="msg warn">⚠️ Выберите файлы</div>';
    return;
  }

  indexBtn.disabled = true;
  indexBtn.innerHTML = 'Индексация... <span class="spinner"></span>';
  idxOut.innerHTML = "";

  try{
    const fd = new FormData();
    uploadedFiles.forEach(f=>fd.append("files", f));
    fd.append("source_tag", sourceTag);

    const resp = await authFetch(`${API_BASE}/ingest`, { method:"POST", body: fd });
    if(!resp.ok) {
      const txt = await resp.text().catch(()=>null);
      throw new Error(`HTTP ${resp.status} ${txt || ''}`);
    }
    const data = await resp.json();

    if(data.ok && data.results){
      data.results.forEach(res => {
        const el = document.createElement('div');
        if (res.status === 'ok') {
          el.className = 'msg ok';
          el.textContent = `✅ ${res.file}: ${res.chunks} чанков (вставлено: ${res.inserted}, пропущено: ${res.skipped})`;
        } else if (res.status === 'warning') {
          el.className = 'msg warn';
          el.textContent = `⚠️ ${res.file}: ${res.message}`;
        } else {
          el.className = 'msg err';
          el.textContent = `❌ ${res.file}: ${res.message}`;
        }
        idxOut.appendChild(el);
      });
      await updateStats();
      await loadUploadedFiles();

      uploadedFiles = [];
      showNames();
    } else {
      idxOut.innerHTML = `<div class="msg err">❌ Ошибка обработки: ${JSON.stringify(data)}</div>`;
    }

  } catch(e) {
    idxOut.innerHTML = `<div class="msg err">❌ ${e.message}</div>`;
  } finally {
    indexBtn.disabled = false;
    indexBtn.textContent = "Индексировать";
  }
});

// --- 7. Stats Functions ---
const statsContent = document.getElementById('statsContent');
const refreshStatsBtn = document.getElementById('refreshStatsBtn');

function renderStats(data) {
    const db_stats = data.database || {};

    if (!db_stats || !db_stats.total_documents) {
        statsContent.innerHTML = `<div class="hint">База данных пуста или недоступна.</div>`;
        return;
    }

    let html = `
        <p><strong>Количество проиндексированных разделов:</strong> ${db_stats.total_documents}</p>
        <p><strong>Количество уникальных файлов:</strong> ${db_stats.unique_files}</p>
        <h4>По источникам:</h4>
        <ul style="margin-top: 5px; list-style-type: none; padding-left: 10px;">
    `;

    const sources = Object.entries(db_stats.by_source || {});
    sources.sort(([, countA], [, countB]) => countB - countA);

    sources.forEach(([source, count]) => {
        html += `<li>• ${source}: <strong>${count}</strong></li>`;
    });

    html += `</ul>
        <div class="hint" style="margin-top: 10px;">
            Эмбеддер: ${data.embedder?.model || '—'} (${data.embedder?.dimension || '—'} dim)
        </div>
    `;
    statsContent.innerHTML = html;
}

async function updateStats() {
    if (!statsContent) return;
    statsContent.innerHTML = 'Загрузка... <span class="spinner"></span>';
    try {
        refreshStatsBtn.disabled = true;

        const resp = await authFetch(`${API_BASE}/stats`);
        if (!resp.ok) {
          const txt = await resp.text().catch(()=>null);
          throw new Error(`HTTP ${resp.status} ${txt || ''}`);
        }

        const data = await resp.json();
        renderStats(data);

    } catch(e) {
        if (e && (e.message === "not_authenticated" || e.message === "unauthorized")) {
            statsContent.innerHTML = `<div class="hint">🔐 Войдите в платформу, чтобы увидеть статистику базы знаний.</div>`;
        } else {
            statsContent.innerHTML = `<div class="msg err">❌ Ошибка загрузки статистики: ${e.message}</div>`;
        }
    } finally {
        refreshStatsBtn.disabled = false;
    }
}

// --- 8. Rebuild Cache Button ---
const rebuildCacheBtn = document.getElementById('rebuildCacheBtn');

rebuildCacheBtn?.addEventListener('click', async () => {
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) { openAuthModal("login"); return; }

  if(!confirm("Пересобрать embedded.pkl из текущей БД? Это обновит кэш.")) return;
  rebuildCacheBtn.disabled = true;
  rebuildCacheBtn.innerHTML = 'Пересборка... <span class="spinner"></span>';
  try {
    const resp = await authFetch(`${API_BASE}/rebuild_cache`, { method: 'POST' });
    if(!resp.ok) {
      const txt = await resp.text().catch(()=>null);
      throw new Error(`HTTP ${resp.status} ${txt || ''}`);
    }
    const data = await resp.json();
    alert(`Кэш пересобран: ${data.count} записей`);
    await updateStats();
    await loadUploadedFiles();
  } catch(e) {
    alert('Ошибка: ' + e.message);
  } finally {
    rebuildCacheBtn.disabled = false;
    rebuildCacheBtn.textContent = 'Пересобрать кэш';
  }
});

refreshStatsBtn?.addEventListener('click', updateStats);

// --- 9. Load Uploaded Files List ---
const uploadedFilesList = document.getElementById('uploadedFilesList');

async function loadUploadedFiles() {
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) {
    uploadedFilesList && (uploadedFilesList.innerHTML = '<div class="hint">Войдите чтобы увидеть свои файлы.</div>');
    return;
  }

  uploadedFilesList.innerHTML = 'Загрузка... <span class="spinner"></span>';
  try {
    const resp = await authFetch(`${API_BASE}/documents`);
    if (!resp.ok) {
      const txt = await resp.text().catch(()=>null);
      throw new Error(`HTTP ${resp.status} ${txt || ''}`);
    }
    const data = await resp.json();
    if (!data.ok) throw new Error("Ошибка сервера");

    const docs = data.documents || [];
    if (docs.length === 0) {
      uploadedFilesList.innerHTML = '<div class="hint">Файлы отсутствуют.</div>';
      return;
    }

    uploadedFilesList.innerHTML = '';
    docs.forEach(doc => {
      try {
        const row = document.createElement('div');
        row.className = 'doc-row';
        const left = document.createElement('div');

        const safeFilename = doc.filename && doc.filename.trim() ? doc.filename.trim() : '(нет имени файла)';
        const safeSource = doc.source || '';

        left.innerHTML = `<strong>${safeFilename}</strong><div class="doc-meta">${doc.chunks} чанков • ${safeSource}</div>`;

        const right = document.createElement('div');

        const viewBtn = document.createElement('button');
        viewBtn.className = 'btn';
        viewBtn.textContent = 'Просмотреть';
        viewBtn.addEventListener('click', ()=> {
          window.open(`${API_BASE}/documents/download?filename=${encodeURIComponent(safeFilename)}`, '_blank');
        });

        const delBtn = document.createElement('button');
        delBtn.className = 'btn';
        delBtn.style.marginLeft = '8px';
        delBtn.textContent = 'Удалить';
        delBtn.addEventListener('click', async ()=> {
          if (!confirm(`Удалить все чанки файла "${safeFilename}"?`)) return;
          try {
            const fd = new FormData();
            fd.append('filename', safeFilename);
            const r = await authFetch(`${API_BASE}/documents/delete`, { method: 'POST', body: fd });
            const res = await r.json();
            if (res.ok) {
              await updateStats();
              await loadUploadedFiles();
            } else {
              alert('Ошибка удаления');
            }
          } catch(e) {
            alert('Ошибка: ' + e.message);
          }
        });

        right.appendChild(viewBtn);
        right.appendChild(delBtn);

        row.appendChild(left);
        row.appendChild(right);
        uploadedFilesList.appendChild(row);
      } catch (e) {
          console.error(`Ошибка при обработке файла ${doc.filename}:`, e);
          const errorRow = document.createElement('div');
          errorRow.className = 'doc-row';
          errorRow.innerHTML = `❌ <strong style="color:var(--red)">Не удалось отобразить файл</strong>: ${doc.filename}`;
          uploadedFilesList.appendChild(errorRow);
      }
    });

  } catch(e) {
    uploadedFilesList.innerHTML = `<div class="msg err">❌ ${e.message}</div>`;
  }
}

// --- 10. Initial Load ---
document.addEventListener('DOMContentLoaded', () => {
  updateAuthUI();
  updateStats();
  loadUploadedFiles();
});

```

# File: ./static/styles.css

```
:root{
  --ink:#0f172a; --muted:#64748b; --line:#e2e8f0;
  --bg:#ffffff; --bg-soft:#f7fafc; --accent:#3b82f6;
  --green:#10b981; --yellow:#f59e0b; --red:#ef4444;
}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial,sans-serif;color:var(--ink);background:var(--bg-soft)}
.wrap{max-width:1080px;margin:0 auto;padding:16px}
/* Header */
.header{display:flex;gap:14px;align-items:flex-start;padding:14px 0 12px;border-bottom:1px solid var(--line)}
.logo{width:60px;height:60px;border-radius:9px;display:flex;align-items:center;justify-content:center;
      background:linear-gradient(180deg,#eef2ff,#f1f5f9);border:1px solid var(--line);font-weight:800}
.h-title{font-size:22px;font-weight:800;line-height:1.1}
.h-sub{color:var(--muted)}
/* Tabs */
.tabs{display:flex;gap:8px;margin:12px 0}
.tab-btn{appearance:none;border:1px solid var(--line);background:var(--bg);color:var(--ink);
         padding:8px 12px;border-radius:10px;cursor:pointer;transition:.2s}
.tab-btn:hover{background:#f8fafc}
.tab-btn.active{background:#fff;box-shadow:0 1px 0 rgba(0,0,0,.03);border-color:#cbd5e1}
.panel{display:none}
.panel.active{display:block}
/* Panels */
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px;margin-top:10px}
.hero{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:16px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:880px){ .grid{grid-template-columns:1fr} }
h1,h2,h3{margin:0 0 8px}
label{display:block;font-weight:600;margin:6px 0 4px}
textarea,input[type="text"],input[type="number"]{
  width:100%; padding:10px 12px; border:1px solid var(--line); border-radius:10px; background:#fff;
  outline:none; box-shadow:none; font-family:inherit;
}
textarea{min-height:120px;resize:vertical}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.btn{border:1px solid var(--line);background:#fff;border-radius:10px;padding:10px 14px;cursor:pointer;transition:.2s}
.btn:hover:not(:disabled){background:#f8fafc}
.btn.primary{background:#fff;border-color:#cbd5e1;box-shadow:0 1px 0 rgba(0,0,0,.03)}
.btn:disabled{opacity:.6;cursor:not-allowed}
.hint{color:var(--muted);font-size:12px}
/* Drag & Drop */
.drop{
  border:2px dashed #cbd5e1;border-radius:14px;background:#fff;
  padding:24px;text-align:center;color:var(--muted);transition:.15s ease all;cursor:pointer;
}
.drop:hover{border-color:var(--accent);background:#f8fbff}
.drop.drag{border-color:var(--accent);color:var(--ink);background:#f0f9ff}
.pill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);
      border-radius:999px;padding:4px 10px;background:#fff;font-size:12px}
/* Messages */
.msg{
  padding:10px 14px; border-radius:10px; margin-top:8px;
  font-size:13px; border:1px solid;
}
.msg.ok{background:#f0fdf4;border-color:#bbf7d0;color:#166534}
.msg.warn{background:#fffbeb;border-color:#fde68a;color:#713f12}
.msg.err{background:#fef2f2;border-color:#fecaca;color:#991b1b}
.spinner{
  display:inline-block;width:12px;height:12px;border:2px solid rgba(0,0,0,.1);
  border-top-color:var(--ink);border-radius:50%;
  animation:spin .6s linear infinite;margin-left:6px;vertical-align:middle;
}
@keyframes spin{ to{transform:rotate(360deg)} }
/* Sources styling */
.sources{
  background:#f8fafc;border:1px solid var(--line);border-radius:10px;padding:12px;
  max-height:400px;overflow-y:auto;font-size:13px;line-height:1.6;
}
.sources h2, .sources h3{font-size:14px;margin:8px 0 4px;color:var(--ink)}
.sources code{background:#fff;padding:2px 6px;border-radius:4px;font-size:12px}
.similarity-badge{
  display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600;
  background:#e0f2fe;color:#0369a1;margin-left:6px;
}
.similarity-badge.high{background:#dcfce7;color:#166534}
.similarity-badge.medium{background:#fef3c7;color:#92400e}
.similarity-badge.low{background:#fee2e2;color:#991b1b}
/* Footer */
.footer{margin-top:16px;padding:12px 0;border-top:1px solid var(--line);display:flex;gap:10px;flex-wrap:wrap;justify-content:space-between;color:var(--muted);font-size:13px}
.links a{color:var(--ink);text-decoration:none;border-bottom:1px dashed var(--line)}
.links a:hover{border-bottom-style:solid}
/* Rating display */
.rating-display{font-size:18px;font-weight:700;color:var(--accent);margin-left:8px}
.doc-row{display:flex;justify-content:space-between;align-items:center;padding:8px;border-bottom:1px dashed var(--line)}
.doc-meta{font-size:13px;color:var(--muted)}

/* ===== Auth modal (compact sign in/up) ===== */
.modal { display: none; position: fixed; inset: 0; align-items: center; justify-content: center;
        background: rgba(0,0,0,0.36); z-index: 1200; pointer-events: none; }
.modal.open { display:flex; pointer-events: auto; }

.auth-card { width: 420px; max-width: calc(100% - 32px); background:#fff; border-radius:12px;
             padding:18px 18px 16px; box-shadow: 0 12px 40px rgba(3,15,30,0.25); position:relative; }

.auth-close { position:absolute; right:10px; top:8px; border: none; background: transparent;
              font-size:24px; line-height:1; cursor:pointer; color:var(--muted); }

.auth-switch { display:flex; gap:6px; margin-bottom:12px; }
.auth-switch-btn { flex:1; padding:8px 10px; border-radius:8px; border:1px solid var(--line);
                   background:var(--bg); cursor:pointer; font-weight:600; }
.auth-switch-btn.active { background:#f3f8ff; border-color:var(--accent); color:var(--accent); }

.auth-form { display:flex; flex-direction:column; gap:10px; }
.auth-form h3 { margin:0 0 2px; font-size:18px; }
.auth-error { color:var(--red); background:#fff6f6; border:1px solid #ffd6d6; padding:8px; border-radius:8px; }

.auth-form label { font-size:13px; font-weight:700; color:var(--muted); }
.auth-form input[type="text"], .auth-form input[type="email"], .auth-form input[type="password"]{
  width:100%; padding:10px 12px; border:1px solid var(--line); border-radius:8px; background:#fff;
}

.auth-form .row { display:flex; gap:8px; align-items:center; }

/* small responsive */
@media (max-width:480px){
  .auth-card{ width:92%; padding:14px; }
}

/* ====== Голосовой ввод ====== */
.textarea-wrap {
  position: relative;
  display: flex;
  align-items: stretch;
  gap: 8px;
}

.textarea-wrap textarea {
  flex: 1;
  min-height: 120px;
  resize: vertical;
}

#micBtn {
  position: relative;
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 8px;
  padding: 12px 20px;
  font-size: 24px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(0,123,255,0.2);
  align-self: flex-start;
  min-width: 60px;
}

#micBtn:hover:not(:disabled) {
  background: #0056b3;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,123,255,0.3);
}

#micBtn:active:not(:disabled) {
  transform: scale(0.95);
}

#micBtn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  background: var(--muted);
}

/* Анимация записи */
@keyframes pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.1); opacity: 0.8; }
}

#micBtn.recording {
  background: #dc3545;
  animation: pulse 1.5s ease-in-out infinite;
}

/* Индикатор обработки */
#micBtn .processing-indicator {
  position: absolute;
  top: -5px;
  right: -5px;
  width: 20px;
  height: 20px;
  background: #ffc107;
  border-radius: 50%;
  animation: blink 0.8s ease-in-out infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

/* Уведомление о распознавании */
.transcription-notice {
  position: fixed;
  top: 20px;
  right: 20px;
  background: white;
  padding: 15px 20px;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  z-index: 1000;
  animation: slideIn 0.3s ease;
}

@keyframes slideIn {
  from {
    transform: translateX(400px);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

.transcription-notice.success {
  border-left: 4px solid #28a745;
}

.transcription-notice.error {
  border-left: 4px solid #dc3545;
}

```

# File: ./static/index.html

```
<!doctype html>
<html lang="ru">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Tanym Loop — MVP UI</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="wrap">
        <header class="header">
            <div class="logo">TL</div>
            <div>
                <div class="h-title">Tanym Loop — локальная RAG-платформа</div>
                <div class="h-sub">Умный поиск и ответы по вашим документам • данные остаются внутри организации</div>
            </div>

            <!-- Auth area -->
            <div id="authArea" style="margin-left:auto;display:flex;align-items:center;gap:8px">
                <div id="userInfo" style="display:none;align-items:center;gap:8px">
                    <span id="userEmail" class="hint"></span>
                    <button id="logoutBtn" class="btn small">Выйти</button>
                </div>

                <div id="anonArea" style="display:flex;align-items:center;gap:6px">
                    <button id="showLoginBtn" class="btn small">Вход</button>
                    <button id="showRegisterBtn" class="btn small">Регистрация</button>
                </div>
            </div>
        </header>

        <nav class="tabs">
            <button class="tab-btn active" data-tab="welcome">🏠 Приветствие</button>
            <button class="tab-btn" data-tab="qa">💬 Поиск / Ответ</button>
            <button class="tab-btn" data-tab="docs">📂 Документы</button>
        </nav>

        <div id="systemMessage" class="system-message" style="display:none;padding:15px;background:#fffbe6;border:1px solid #ffe58f;border-radius:4px;margin-bottom:15px"></div>

        <!-- Auth modal: compact sign-in / sign-up -->
        <div id="authModal" class="modal" aria-hidden="true" role="dialog" aria-modal="true" aria-labelledby="authModalTitle">
          <div class="modal-card auth-card" role="document">
            <button type="button" class="auth-close" id="authCloseBtn" aria-label="Закрыть">&times;</button>

            <div class="auth-switch">
              <button id="switchToLogin" class="auth-switch-btn active" type="button">Вход</button>
              <button id="switchToRegister" class="auth-switch-btn" type="button">Регистрация</button>
            </div>

            <div id="authFormsWrap">
              <!-- LOGIN -->
              <form id="loginForm" class="auth-form" data-mode="login" onsubmit="return false;">
                <h3 id="authModalTitle">Вход</h3>
                <div id="authError" class="auth-error" style="display:none"></div>

                <label for="authEmail">Email</label>
                <input id="authEmail" type="email" autocomplete="username" placeholder="you@example.com" required>

                <label for="authPassword">Пароль</label>
                <input id="authPassword" type="password" autocomplete="current-password" placeholder="Пароль" required>

                <div class="row" style="justify-content:flex-end;margin-top:12px">
                  <button id="authCancelBtn" type="button" class="btn">Отмена</button>
                  <button id="authSubmitBtn" type="button" class="btn primary">Войти</button>
                </div>
              </form>

              <!-- REGISTER -->
              <form id="registerForm" class="auth-form" data-mode="register" style="display:none" onsubmit="return false;">
                <h3>Регистрация</h3>
                <div id="regError" class="auth-error" style="display:none"></div>

                <label for="authName">Имя</label>
                <input id="authName" type="text" autocomplete="name" placeholder="Иван Иванов" required>

                <label for="regEmail">Email</label>
                <input id="regEmail" type="email" autocomplete="email" placeholder="you@example.com" required>

                <label for="regPassword">Пароль</label>
                <input id="regPassword" type="password" autocomplete="new-password" placeholder="Пароль" required>

                <div class="row" style="justify-content:flex-end;margin-top:12px">
                  <button id="regCancelBtn" type="button" class="btn">Отмена</button>
                  <button id="regSubmitBtn" type="button" class="btn primary">Зарегистрироваться</button>
                </div>
              </form>
            </div>

            <div style="margin-top:12px;font-size:13px;color:var(--muted);text-align:center">
              <small>Данные хранятся локально в организации</small>
            </div>
          </div>
        </div>

        <section id="welcome" class="panel active">
            <div class="grid">
                <div class="hero">
                    <h1>Добро пожаловать в Tanym Loop</h1>
                    <p>Самообучающаяся RAG-система: точные ответы на основе ваших документов.</p>
                    <ul>
                        <li>Локально и безопасно — данные не покидают периметр</li>
                        <li>Казахстанский контекст — законы, госуслуги, внутренние регламенты</li>
                        <li>Самообучение — качество ответов растёт с обратной связью</li>
                        <li>🎙️ Голосовой ввод — говорите вопросы вместо набора текста</li>
                    </ul>
                    <div class="row" style="margin-top:6px">
                        <span class="pill">🔒 Конфиденциально</span>
                        <span class="pill">🧠 Self-learning</span>
                        <span class="pill">📚 RAG</span>
                        <span class="pill">🎙️ Voice</span>
                    </div>
                </div>
                <div class="card">
                    <h3>Как начать</h3>
                    <ol style="margin:6px 0 0">
                        <li>Откройте вкладку «Документы» и загрузите файлы</li>
                        <li>Перейдите в «Поиск / Ответ» и задайте вопрос (текстом или голосом 🎙️)</li>
                        <li>Оставляйте обратную связь для улучшения системы</li>
                    </ol>
                    <div style="margin-top:12px;padding:10px;background:#f0f9ff;border-radius:8px;font-size:13px">
                        <strong>💡 Совет:</strong> Чем больше документов загружено, тем точнее ответы!
                    </div>
                </div>
            </div>
        </section>

        <section id="qa" class="panel">
            <div class="card">
                <h2>Вопрос / Ответ</h2>
                <div class="grid" style="margin-top:8px">
                    <div>
                        <label for="q">Введите вопрос или используйте голосовой ввод 🎙️</label>
                        <div class="textarea-wrap">
                            <textarea id="q" placeholder="Например: Как оформить доверенность у нотариуса?"></textarea>
                            <button id="micBtn" type="button" title="Голосовой ввод: нажмите для записи">🎙️</button>
                        </div>
                        <div id="voiceStatus" class="hint" style="margin-top:6px;display:none;color:var(--accent)"></div>
                        <div class="row" style="margin-top:8px">
                            <label class="row" style="gap:6px;margin:0;cursor:pointer">
                                <input id="showSimilar" type="checkbox" checked />
                                <span>Показывать источники</span>
                            </label>
                            <button id="askBtn" class="btn primary">Задать вопрос</button>
                            <button id="clearBtn" class="btn">Очистить</button>
                        </div>
                    </div>
                    <div>
                        <label for="ans">Ответ ассистента</label>
                        <textarea id="ans" readonly placeholder="💡 Здесь появится ответ ассистента."></textarea>
                        <div id="ansMetadata" class="hint" style="margin-top:6px"></div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>📚 Использованные источники</h3>
                <div id="sources" class="sources">
                    <div class="hint">Здесь будут показаны источники и метрики релевантности.</div>
                </div>
            </div>

            <div class="card">
                <h3>✍️ Обратная связь</h3>
                <div class="grid" style="margin-top:6px">
                    <div>
                        <label for="rating">Оценка ответа: <span id="ratingVal" class="rating-display">5</span></label>
                        <input id="rating" type="range" min="1" max="5" value="5" />
                    </div>
                    <div>
                        <label for="corr">Правильный ответ (опционально)</label>
                        <input id="corr" type="text" placeholder="Введите корректный вариант" />
                    </div>
                </div>
                <label for="comment" style="margin-top:6px">Комментарий</label>
                <input id="comment" type="text" placeholder="Что было полезно, чего не хватило…" />
                <div class="row" style="margin-top:8px">
                    <button id="sendFb" class="btn">Отправить отзыв</button>
                    <span id="fbOut" class="hint"></span>
                </div>
            </div>
        </section>

        <section id="docs" class="panel">
            <div class="card">
                <h2>База знаний организации</h2>
                <div class="grid" style="margin-top:8px">
                    <div>
                        <div id="drop" class="drop" tabindex="0">
                            <div style="font-weight:600">📁 Перетащите файлы сюда</div>
                            <div class="hint">Поддержка: PDF, DOCX, TXT, XLSX</div>
                            <div class="hint">Или нажмите для выбора</div>
                            <input id="fileInput" type="file" multiple style="display:none"
                                   accept=".pdf,.docx,.xlsx,.xls,.txt" />
                        </div>
                        <div class="row" style="margin-top:8px">
                            <select id="sourceTag" class="btn">
                                <option value="user">👤 user (пользователь)</option>
                                <option value="dataset">📊 dataset (база)</option>
                                <option value="other">📂 other (другое)</option>
                            </select>
                            <button id="indexBtn" class="btn primary">Индексировать</button>
                        </div>
                    </div>
                    <div>
                        <div class="card" style="border:none;padding:0">
                            <h3>💡 Рекомендации</h3>
                            <ul style="margin:6px 0 0;font-size:13px">
                                <li>Избегайте сканов без OCR</li>
                                <li>Структурированные документы (со "Статья X") обрабатываются лучше</li>
                                <li>Оптимальный размер файла: до 10 МБ</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <div id="fileList" class="hint" style="margin-top:10px"></div>
                <div id="idxOut" style="margin-top:10px"></div>

                <div id="dbStats" class="card" style="margin-top: 15px;">
                    <h3>📊 Статистика базы знаний</h3>
                    <div id="statsContent" style="min-height: 50px;">
                        <div class="hint">Нажмите "Обновить статистику" или загрузите файл.</div>
                    </div>
                    <button id="refreshStatsBtn" class="btn" style="margin-top: 10px;">Обновить статистику</button>
                    <button id="rebuildCacheBtn" class="btn" style="margin-left:8px;">Пересобрать кэш</button>
                </div>
                <div class="card" style="margin-top:12px;">
                    <h3>Список загруженных файлов</h3>
                    <div id="uploadedFilesList" style="min-height:80px;">
                        <div class="hint">Файлы появятся здесь после загрузки / индексации.</div>
                    </div>
                </div>
            </div>
        </section>

        <footer class="footer">
            <div>© 2025 True Masters • Tanym Loop v0.3 🎙️</div>
            <div class="links">
                <a href="https://truemasters.kz" target="_blank" rel="noopener">Tanym Loop</a> ·
                <a href="mailto:truemasters@inbox.ru">Связаться</a> ·
                <a href="/stats" target="_blank">Статистика</a> ·
                <a href="/voicing/health" target="_blank">Voice Status</a>
            </div>
        </footer>
    </div>


    <script src="/static/app2.js"></script>
</body>
</html>
```

# File: ./static/tanym_loop_mvp.html

```
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Tanym Loop — MVP UI</title>
<style>
  :root{
    --ink:#0f172a; --muted:#64748b; --line:#e2e8f0;
    --bg:#ffffff; --bg-soft:#f7fafc; --accent:#3b82f6;
    --green:#10b981; --yellow:#f59e0b; --red:#ef4444;
  }
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial,sans-serif;color:var(--ink);background:var(--bg-soft)}
  .wrap{max-width:1080px;margin:0 auto;padding:16px}
  /* Header */
  .header{display:flex;gap:14px;align-items:flex-start;padding:14px 0 12px;border-bottom:1px solid var(--line)}
  .logo{width:60px;height:60px;border-radius:9px;display:flex;align-items:center;justify-content:center;
        background:linear-gradient(180deg,#eef2ff,#f1f5f9);border:1px solid var(--line);font-weight:800}
  .h-title{font-size:22px;font-weight:800;line-height:1.1}
  .h-sub{color:var(--muted)}
  /* Tabs */
  .tabs{display:flex;gap:8px;margin:12px 0}
  .tab-btn{appearance:none;border:1px solid var(--line);background:var(--bg);color:var(--ink);
           padding:8px 12px;border-radius:10px;cursor:pointer;transition:.2s}
  .tab-btn:hover{background:#f8fafc}
  .tab-btn.active{background:#fff;box-shadow:0 1px 0 rgba(0,0,0,.03);border-color:#cbd5e1}
  .panel{display:none}
  .panel.active{display:block}
  /* Panels */
  .card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px;margin-top:10px}
  .hero{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:16px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  @media (max-width:880px){ .grid{grid-template-columns:1fr} }
  h1,h2,h3{margin:0 0 8px}
  label{display:block;font-weight:600;margin:6px 0 4px}
  textarea,input[type="text"],input[type="number"]{
    width:100%; padding:10px 12px; border:1px solid var(--line); border-radius:10px; background:#fff;
    outline:none; box-shadow:none; font-family:inherit;
  }
  textarea{min-height:120px;resize:vertical}
  .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .btn{border:1px solid var(--line);background:#fff;border-radius:10px;padding:10px 14px;cursor:pointer;transition:.2s}
  .btn:hover:not(:disabled){background:#f8fafc}
  .btn.primary{background:#fff;border-color:#cbd5e1;box-shadow:0 1px 0 rgba(0,0,0,.03)}
  .btn:disabled{opacity:.6;cursor:not-allowed}
  .hint{color:var(--muted);font-size:12px}
  /* Drag & Drop */
  .drop{
    border:2px dashed #cbd5e1;border-radius:14px;background:#fff;
    padding:24px;text-align:center;color:var(--muted);transition:.15s ease all;cursor:pointer;
  }
  .drop:hover{border-color:var(--accent);background:#f8fbff}
  .drop.drag{border-color:var(--accent);color:var(--ink);background:#f0f9ff}
  .pill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);
        border-radius:999px;padding:4px 10px;background:#fff;font-size:12px}
  /* Messages */
  .msg{
    padding:10px 14px; border-radius:10px; margin-top:8px;
    font-size:13px; border:1px solid;
  }
  .msg.ok{background:#f0fdf4;border-color:#bbf7d0;color:#166534}
  .msg.warn{background:#fffbeb;border-color:#fde68a;color:#713f12}
  .msg.err{background:#fef2f2;border-color:#fecaca;color:#991b1b}
  .spinner{
    display:inline-block;width:12px;height:12px;border:2px solid rgba(0,0,0,.1);
    border-top-color:var(--ink);border-radius:50%;
    animation:spin .6s linear infinite;margin-left:6px;vertical-align:middle;
  }
  @keyframes spin{ to{transform:rotate(360deg)} }
  /* Sources styling */
  .sources{
    background:#f8fafc;border:1px solid var(--line);border-radius:10px;padding:12px;
    max-height:400px;overflow-y:auto;font-size:13px;line-height:1.6;
  }
  .sources h2, .sources h3{font-size:14px;margin:8px 0 4px;color:var(--ink)}
  .sources code{background:#fff;padding:2px 6px;border-radius:4px;font-size:12px}
  .similarity-badge{
    display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600;
    background:#e0f2fe;color:#0369a1;margin-left:6px;
  }
  .similarity-badge.high{background:#dcfce7;color:#166534}
  .similarity-badge.medium{background:#fef3c7;color:#92400e}
  .similarity-badge.low{background:#fee2e2;color:#991b1b}
  /* Footer */
  .footer{margin-top:16px;padding:12px 0;border-top:1px solid var(--line);display:flex;gap:10px;flex-wrap:wrap;justify-content:space-between;color:var(--muted);font-size:13px}
  .links a{color:var(--ink);text-decoration:none;border-bottom:1px dashed var(--line)}
  .links a:hover{border-bottom-style:solid}
  /* Rating display */
  .rating-display{font-size:18px;font-weight:700;color:var(--accent);margin-left:8px}
  .doc-row{display:flex;justify-content:space-between;align-items:center;padding:8px;border-bottom:1px dashed var(--line)}
  .doc-meta{font-size:13px;color:var(--muted)}
</style>
</head>
<body>
  <div class="wrap">
    <!-- Header -->
    <header class="header">
      <div class="logo">TL</div>
      <div>
        <div class="h-title">Tanym Loop — локальная RAG-платформа</div>
        <div class="h-sub">Умный поиск и ответы по вашим документам • данные остаются внутри организации</div>
      </div>
    </header>

    <!-- Tabs -->
    <nav class="tabs">
      <button class="tab-btn active" data-tab="welcome">🏠 Приветствие</button>
      <button class="tab-btn" data-tab="qa">💬 Поиск / Ответ</button>
      <button class="tab-btn" data-tab="docs">📂 Документы</button>
    </nav>

    <!-- Welcome -->
    <section id="welcome" class="panel active">
      <div class="grid">
        <div class="hero">
          <h1>Добро пожаловать в Tanym Loop</h1>
          <p>Самообучающаяся RAG-система: точные ответы на основе ваших документов.</p>
          <ul>
            <li>Локально и безопасно — данные не покидают периметр</li>
            <li>Казахстанский контекст — законы, госуслуги, внутренние регламенты</li>
            <li>Самообучение — качество ответов растёт с обратной связью</li>
          </ul>
          <div class="row" style="margin-top:6px">
            <span class="pill">🔒 Конфиденциально</span>
            <span class="pill">🧠 Self-learning</span>
            <span class="pill">📚 RAG</span>
          </div>
        </div>
        <div class="card">
          <h3>Как начать</h3>
          <ol style="margin:6px 0 0">
            <li>Откройте вкладку «Документы» и загрузите файлы</li>
            <li>Перейдите в «Поиск / Ответ» и задайте вопрос</li>
            <li>Оставляйте обратную связь для улучшения системы</li>
          </ol>
          <div style="margin-top:12px;padding:10px;background:#f0f9ff;border-radius:8px;font-size:13px">
            <strong>💡 Совет:</strong> Чем больше документов загружено, тем точнее ответы!
          </div>
        </div>
      </div>
    </section>

    <!-- Q&A -->
    <section id="qa" class="panel">
      <div class="card">
        <h2>Вопрос / Ответ</h2>
        <div class="grid" style="margin-top:8px">
          <div>
            <label for="q">Введите вопрос</label>
            <textarea id="q" placeholder="Например: Как оформить доверенность у нотариуса?"></textarea>
            <div class="row" style="margin-top:8px">
              <label class="row" style="gap:6px;margin:0;cursor:pointer">
                <input id="showSimilar" type="checkbox" checked />
                <span>Показывать источники</span>
              </label>
              <button id="askBtn" class="btn primary">Задать вопрос</button>
              <button id="clearBtn" class="btn">Очистить</button>
            </div>
          </div>
          <div>
            <label for="ans">Ответ ассистента</label>
            <textarea id="ans" readonly placeholder="💡 Здесь появится ответ ассистента."></textarea>
            <div id="ansMetadata" class="hint" style="margin-top:6px"></div>
          </div>
        </div>
      </div>

      <div class="card">
        <h3>📚 Использованные источники</h3>
        <div id="sources" class="sources">
          <div class="hint">Здесь будут показаны источники и метрики релевантности.</div>
        </div>
      </div>

      <div class="card">
        <h3>✍️ Обратная связь</h3>
        <div class="grid" style="margin-top:6px">
          <div>
            <label for="rating">Оценка ответа: <span id="ratingVal" class="rating-display">5</span></label>
            <input id="rating" type="range" min="1" max="5" value="5" />
          </div>
          <div>
            <label for="corr">Правильный ответ (опционально)</label>
            <input id="corr" type="text" placeholder="Введите корректный вариант" />
          </div>
        </div>
        <label for="comment" style="margin-top:6px">Комментарий</label>
        <input id="comment" type="text" placeholder="Что было полезно, чего не хватило…" />
        <div class="row" style="margin-top:8px">
          <button id="sendFb" class="btn">Отправить отзыв</button>
          <span id="fbOut" class="hint"></span>
        </div>
      </div>
    </section>

    <!-- Documents -->
    <section id="docs" class="panel">
      <div class="card">
        <h2>База знаний организации</h2>
        <div class="grid" style="margin-top:8px">
          <div>
            <div id="drop" class="drop" tabindex="0">
              <div style="font-weight:600">📁 Перетащите файлы сюда</div>
              <div class="hint">Поддержка: PDF, DOCX, TXT, XLSX</div>
              <div class="hint">Или нажмите для выбора</div>
              <input id="fileInput" type="file" multiple style="display:none"
                     accept=".pdf,.docx,.xlsx,.xls,.txt" />
            </div>
            <div class="row" style="margin-top:8px">
              <select id="sourceTag" class="btn">
                <option value="user">👤 user (пользователь)</option>
                <option value="dataset">📊 dataset (база)</option>
                <option value="other">📂 other (другое)</option>
              </select>
              <button id="indexBtn" class="btn primary">Индексировать</button>
            </div>
          </div>
          <div>
            <div class="card" style="border:none;padding:0">
              <h3>💡 Рекомендации</h3>
              <ul style="margin:6px 0 0;font-size:13px">
                <li>Избегайте сканов без OCR</li>
                <li>Структурированные документы (со "Статья X") обрабатываются лучше</li>
                <li>Оптимальный размер файла: до 10 МБ</li>
              </ul>
            </div>
          </div>
        </div>

        <div id="fileList" class="hint" style="margin-top:10px"></div>
        <div id="idxOut" style="margin-top:10px"></div>

        <div id="dbStats" class="card" style="margin-top: 15px;">
          <h3>📊 Статистика базы знаний</h3>
          <div id="statsContent" style="min-height: 50px;">
              <div class="hint">Нажмите "Обновить статистику" или загрузите файл.</div>
          </div>
          <button id="refreshStatsBtn" class="btn" style="margin-top: 10px;">Обновить статистику</button>
          <button id="rebuildCacheBtn" class="btn" style="margin-left:8px;">Пересобрать кэш</button>
        </div>
        <div class="card" style="margin-top:12px;">
          <h3>Список загруженных файлов</h3>
          <div id="uploadedFilesList" style="min-height:80px;">
            <div class="hint">Файлы появятся здесь после загрузки / индексации.</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Footer -->
    <footer class="footer">
      <div>© 2025 True Masters • Tanym Loop v0.2</div>
      <div class="links">
        <a href="https://truemasters.kz" target="_blank" rel="noopener">Tanym Loop</a> ·
        <a href="mailto:truemasters@inbox.ru">Связаться</a> ·
        <a href="/stats" target="_blank">Статистика</a>
      </div>
    </footer>
  </div>

<script>
  const API_BASE = "";
  let lastQuestion = "";
  let lastAnswer = "";

  document.querySelectorAll('.tab-btn').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });

  document.getElementById('rating')?.addEventListener('input', e=>{
    document.getElementById('ratingVal').textContent = e.target.value;
  });

  const askBtn = document.getElementById('askBtn');
  const clearBtn = document.getElementById('clearBtn');

  askBtn?.addEventListener('click', async ()=>{
    const q = document.getElementById('q').value.trim();
    const show = document.getElementById('showSimilar').checked;
    const ansEl = document.getElementById('ans');
    const srcEl = document.getElementById('sources');
    const metaEl = document.getElementById('ansMetadata');

    if(!q){
      ansEl.value = "❌ Введите вопрос.";
      return;
    }

    try{
      askBtn.disabled = true;
      askBtn.innerHTML = 'Думаю... <span class="spinner"></span>';

      const fd = new FormData();
      fd.append("question", q);
      fd.append("show_articles", String(show));

      const resp = await fetch(`${API_BASE}/ask`, { method:"POST", body: fd });
      if(!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      ansEl.value = data.answer || "";

      if(data.metadata){
        const m = data.metadata;
        metaEl.innerHTML = `
          <span class="pill">Найдено: ${m.found_results}</span>
          <span class="pill">Релевантных: ${m.relevant_results}</span>
          <span class="pill">Использовано: ${m.used_results}</span>
          <span class="pill">Ср. релевантность: ${(m.avg_similarity * 100).toFixed(0)}%</span>
        `;
      }

      srcEl.innerHTML = "";
      if(data.sources_md){
        const lines = data.sources_md.split('\n');
        let html = '';

        for(let line of lines){
          if(line.startsWith('### ')){
            const match = line.match(/релевантность:\s*(\d+)%/);
            if(match){
              const pct = parseInt(match[1]);
              let badge = 'medium';
              if(pct >= 80) badge = 'high';
              else if(pct < 60) badge = 'low';
              html += `<h3>${line.replace('### ', '')}<span class="similarity-badge ${badge}">${pct}%</span></h3>`;
            } else {
              html += `<h3>${line.replace('### ', '')}</h3>`;
            }
          } else if(line.startsWith('## ')){
            html += `<h2>${line.replace('## ', '')}</h2>`;
          } else if(line.startsWith('**') && line.endsWith('**')){
            html += `<strong>${line.replace(/\*\*/g, '')}</strong><br>`;
          } else if(line.startsWith('_') && line.endsWith('_')){
            html += `<em style="color:var(--muted)">${line.replace(/_/g, '')}</em><br>`;
          } else if(line.startsWith('🔗 [')){
            const linkMatch = line.match(/\[([^\]]+)\]\(([^)]+)\)/);
            if(linkMatch){
              html += `<a href="${linkMatch[2]}" target="_blank" style="color:var(--accent)">${line}</a><br>`;
            }
          } else if(line.trim()){
            html += `${line}<br>`;
          }
        }

        srcEl.innerHTML = html || data.sources_md;
      } else {
        srcEl.innerHTML = '<div class="hint">Источники не найдены</div>';
      }

      lastQuestion = q;
      lastAnswer = data.answer || "";

    }catch(e){
      console.error(e);
      ansEl.value = "⚠️ Ошибка при запросе к API: " + e.message;
    }finally{
      askBtn.disabled = false;
      askBtn.textContent = "Задать вопрос";
    }
  });

  clearBtn?.addEventListener('click', ()=>{
    document.getElementById('q').value = "";
    document.getElementById('ans').value = "";
    document.getElementById('sources').innerHTML = '<div class="hint">Здесь будут показаны источники</div>';
    document.getElementById('ansMetadata').innerHTML = "";
    lastQuestion = "";
    lastAnswer = "";
  });

// ======= Обработчик отправки фидбека (заменить текущий sendFb handler) =======
const sendFbBtn = document.getElementById('sendFb');
let lastFeedbackKey = null; // локально — предотвращение дублей в одной сессии

sendFbBtn?.addEventListener('click', async ()=> {
  const rating = document.getElementById('rating').value;
  const comment = document.getElementById('comment').value;
  const corr = document.getElementById('corr').value.trim();
  const fbOut = document.getElementById('fbOut');

  if(!lastQuestion || !lastAnswer){
    fbOut.textContent = "❌ Сначала задайте вопрос";
    setTimeout(()=>fbOut.textContent="", 4000);
    return;
  }

  // локальная защита от одинаковых отправок подряд
  const feedbackKey = `${lastQuestion}|||${lastAnswer}|||${corr}|||${rating}`;
  if (feedbackKey === lastFeedbackKey) {
    fbOut.textContent = "⚠️ Похоже, вы уже отправляли этот отзыв (сессия). Подождите результат.";
    fbOut.style.color = "var(--yellow)";
    setTimeout(()=>fbOut.textContent="", 4000);
    return;
  }

  // UI: блокировка + сообщение о процессе
  sendFbBtn.disabled = true;
  const origBtnText = sendFbBtn.textContent;     // <-- запомним оригинальный текст (исправлено)
  sendFbBtn.innerHTML = 'Отправка... <span class="spinner"></span>';
  fbOut.style.color = "var(--muted)";
  fbOut.textContent = "Проверка правильного ответа...";

  try {
    const fd = new FormData();
    fd.append("rating", rating);
    fd.append("comment", comment);
    fd.append("correct_answer", corr);
    fd.append("question", lastQuestion);
    fd.append("answer", lastAnswer);

    const resp = await fetch(`${API_BASE}/feedback`, { method:"POST", body: fd });
    if(!resp.ok){
      const txt = await resp.text().catch(()=>null);
      throw new Error(`HTTP ${resp.status} ${txt || ''}`);
    }
    const data = await resp.json();

    // Показываем содержимое ответа от сервера — с маскированием технических причин
    let userMsg = data.msg || "OK";
    // Если сервер вернул техническую причину (например про LLM), нормализуем её
    const rawReason = data.validation_reason || "";
    let friendlyReason = "";
    if (rawReason) {
      // Маскирование/замена технических сообщений (настройте под ваш бэкэнд)
      if (/llm/i.test(rawReason) || /недоступн/i.test(rawReason) || /embedd/i.test(rawReason)) {
        friendlyReason = " (Проверка выполнена автоматически.)";
      } else {
        friendlyReason = ` (${rawReason})`;
      }
    }

    fbOut.innerHTML = userMsg + (friendlyReason ? ` <span style="color:var(--muted)">${friendlyReason}</span>` : "");
    fbOut.style.color = data.ok ? "var(--green)" : "var(--red)";

    if (data.correct_answer_saved !== undefined) {
      if (data.correct_answer_saved) {
        fbOut.innerHTML += ' <strong style="color:var(--green)">✅ Правильный ответ сохранён</strong>';
      } else if (data.validation_reason) {
        // Если validation_reason техничесная — показываем дружелюбный вариант (см. выше)
        fbOut.innerHTML += ' <span style="color:var(--yellow)">⚠️ ' + (data.validation_reason && !/llm/i.test(data.validation_reason) ? data.validation_reason : 'Правильный ответ не сохранился') + '</span>';
      }
    }

    // Успешно отправлено — запомним ключ чтобы не отправлять дубль
    lastFeedbackKey = feedbackKey;

    // очистка формы (как у вас было)
    if(data.ok){
      document.getElementById('rating').value = 5;
      document.getElementById('ratingVal').textContent = 5;
      document.getElementById('comment').value = "";
      document.getElementById('corr').value = "";
    }

  } catch(e) {
    console.error(e);
    fbOut.textContent = "⚠️ Ошибка отправки: " + (e.message || e);
    fbOut.style.color = "var(--red)";
  } finally {
    // Восстановим кнопку сразу (и гарантируем восстановление текста)
    sendFbBtn.disabled = false;
    sendFbBtn.textContent = origBtnText || "Отправить отзыв";
    // очищаем сообщение через 8 сек
    setTimeout(()=>fbOut.textContent="", 8000);
  }
});


  // Drag & Drop / Indexing (не менял)
  const drop = document.getElementById('drop');
  const fileInput = document.getElementById('fileInput');
  const indexBtn = document.getElementById('indexBtn');
  const fileListEl = document.getElementById('fileList');
  const idxOut = document.getElementById('idxOut');
  let uploadedFiles = [];

  function showNames(){
    const names = uploadedFiles.map(f=>f.name).join(", ");
    fileListEl.textContent = uploadedFiles.length
      ? `📚 Готово к загрузке: ${names}`
      : "";
  }

  ['dragenter','dragover'].forEach(ev=>{
    drop.addEventListener(ev, e=>{ e.preventDefault(); drop.classList.add('drag');});
  });

  ['dragleave','drop'].forEach(ev=>{
    drop.addEventListener(ev, e=>{ e.preventDefault(); drop.classList.remove('drag');});
  });

  drop.addEventListener('drop', e=>{
    uploadedFiles = uploadedFiles.concat([...e.dataTransfer.files]);
    showNames();
  });

  drop.addEventListener('click', ()=> fileInput.click());

  fileInput.addEventListener('change', e=>{
    uploadedFiles = uploadedFiles.concat([...e.target.files]);
    showNames();
    fileInput.value = null;
  });

  indexBtn?.addEventListener('click', async ()=>{
    const sourceTag = document.getElementById('sourceTag').value;

    if(!uploadedFiles.length){
      idxOut.innerHTML = '<div class="msg warn">⚠️ Выберите файлы</div>';
      return;
    }

    indexBtn.disabled = true;
    indexBtn.innerHTML = 'Индексация... <span class="spinner"></span>';
    idxOut.innerHTML = "";

    try{
      const fd = new FormData();
      uploadedFiles.forEach(f=>fd.append("files", f));
      fd.append("source_tag", sourceTag);

      const resp = await fetch(`${API_BASE}/ingest`, { method:"POST", body: fd });
      if(!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const data = await resp.json();

      if(data.ok && data.results){
        data.results.forEach(res => {
          const el = document.createElement('div');
          if (res.status === 'ok') {
            el.className = 'msg ok';
            el.textContent = `✅ ${res.file}: ${res.chunks} чанков (вставлено: ${res.inserted}, пропущено: ${res.skipped})`;
          } else if (res.status === 'warning') {
            el.className = 'msg warn';
            el.textContent = `⚠️ ${res.file}: ${res.message}`;
          } else {
            el.className = 'msg err';
            el.textContent = `❌ ${res.file}: ${res.message}`;
          }
          idxOut.appendChild(el);
        });
        await updateStats();
        await loadUploadedFiles();

        uploadedFiles = [];
        showNames();
      }

    } catch(e) {
      idxOut.innerHTML = `<div class="msg err">❌ ${e.message}</div>`;
    } finally {
      indexBtn.disabled = false;
      indexBtn.textContent = "Индексировать";
    }
  });

  // Stats functions (как раньше)
  const statsContent = document.getElementById('statsContent');
  const refreshStatsBtn = document.getElementById('refreshStatsBtn');

  function renderStats(data) {
      const db_stats = data.database;

      if (!db_stats || !db_stats.total_documents) {
          statsContent.innerHTML = `<div class="hint">База данных пуста или недоступна.</div>`;
          return;
      }

      let html = `
          <p><strong>Количество проиндексированных разделов:</strong> ${db_stats.total_documents}</p>
          <p><strong>Количество уникальных файлов:</strong> ${db_stats.unique_files}</p>
          <h4>По источникам:</h4>
          <ul style="margin-top: 5px; list-style-type: none; padding-left: 10px;">
      `;

      const sources = Object.entries(db_stats.by_source || {});
      sources.sort(([, countA], [, countB]) => countB - countA);

      sources.forEach(([source, count]) => {
          html += `<li>• ${source}: <strong>${count}</strong></li>`;
      });

      html += `</ul>
          <div class="hint" style="margin-top: 10px;">
              Эмбеддер: ${data.embedder.model} (${data.embedder.dimension} dim)
          </div>
      `;
      statsContent.innerHTML = html;
  }

  async function updateStats() {
      statsContent.innerHTML = 'Загрузка... <span class="spinner"></span>';
      try {
          refreshStatsBtn.disabled = true;

          const resp = await fetch(`${API_BASE}/stats`);
          if (!resp.ok) throw new Error("Не удалось загрузить статистику");

          const data = await resp.json();
          renderStats(data);

      } catch(e) {
          statsContent.innerHTML = `<div class="msg err">❌ Ошибка загрузки статистики: ${e.message}</div>`;
      } finally {
          refreshStatsBtn.disabled = false;
      }
  }

  const rebuildCacheBtn = document.getElementById('rebuildCacheBtn');

rebuildCacheBtn?.addEventListener('click', async () => {
  if(!confirm("Пересобрать embedded.pkl из текущей БД? Это обновит кэш.")) return;
  rebuildCacheBtn.disabled = true;
  rebuildCacheBtn.innerHTML = 'Пересборка... <span class="spinner"></span>';
  try {
    const resp = await fetch(`${API_BASE}/rebuild_cache`, { method: 'POST' });
    if(!resp.ok) throw new Error('Ошибка сервера при пересборке кэша');
    const data = await resp.json();
    alert(`Кэш пересобран: ${data.count} записей`);
    await updateStats();
    await loadUploadedFiles();
  } catch(e) {
    alert('Ошибка: ' + e.message);
  } finally {
    rebuildCacheBtn.disabled = false;
    rebuildCacheBtn.textContent = 'Пересобрать кэш';
  }
});


  refreshStatsBtn?.addEventListener('click', updateStats);
  document.addEventListener('DOMContentLoaded', () => {
    updateStats();
    loadUploadedFiles();
  });

  // ====== NEW: Load uploaded files list and deletion ======
  const uploadedFilesList = document.getElementById('uploadedFilesList');

  async function loadUploadedFiles() {
    uploadedFilesList.innerHTML = 'Загрузка... <span class="spinner"></span>';
    try {
      const resp = await fetch(`${API_BASE}/documents`);
      if (!resp.ok) throw new Error("Не удалось загрузить список");
      const data = await resp.json();
      if (!data.ok) throw new Error("Ошибка сервера");

      const docs = data.documents || [];
      if (docs.length === 0) {
        uploadedFilesList.innerHTML = '<div class="hint">Файлы отсутствуют.</div>';
        return;
      }

      uploadedFilesList.innerHTML = '';
      docs.forEach(doc => {
        const row = document.createElement('div');
        row.className = 'doc-row';
        const left = document.createElement('div');
        left.innerHTML = `<strong>${doc.filename && doc.filename.trim() ? doc.filename : '(нет имени файла)'}</strong><div class="doc-meta">${doc.chunks} чанков • ${doc.source || ''}</div>`;

        const right = document.createElement('div');
        const viewBtn = document.createElement('button');
        viewBtn.className = 'btn';
        viewBtn.textContent = 'Просмотреть';
        viewBtn.addEventListener('click', ()=> {
          // открываем скачивание
          window.open(`${API_BASE}/documents/download?filename=${encodeURIComponent(doc.filename)}`, '_blank');
        });

        const delBtn = document.createElement('button');
        delBtn.className = 'btn';
        delBtn.style.marginLeft = '8px';
        delBtn.textContent = 'Удалить';
        delBtn.addEventListener('click', async ()=> {
          if (!confirm(`Удалить все чанки файла "${doc.filename}"?`)) return;
          try {
            const fd = new FormData();
            fd.append('filename', doc.filename);
            const r = await fetch(`${API_BASE}/documents/delete`, { method: 'POST', body: fd });
            const res = await r.json();
            if (res.ok) {
              await updateStats();
              await loadUploadedFiles();
            } else {
              alert('Ошибка удаления');
            }
          } catch(e) {
            alert('Ошибка: ' + e.message);
          }
        });

        right.appendChild(viewBtn);
        right.appendChild(delBtn);

        row.appendChild(left);
        row.appendChild(right);
        uploadedFilesList.appendChild(row);
      });

    } catch(e) {
      uploadedFilesList.innerHTML = `<div class="msg err">❌ ${e.message}</div>`;
    }
  }

</script>

</body>
</html>
```

# File: ./rag/vectorstore.py

```python
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
                    if owner_id is not None:
                        cur.execute(
                            "SELECT 1 FROM documents WHERE title = %s AND content = %s AND owner_id = %s LIMIT 1",
                            (article["title"], article["text"], owner_id)
                        )
                    else:
                        cur.execute(
                            "SELECT 1 FROM documents WHERE title = %s AND content = %s LIMIT 1",
                            (article["title"], article["text"])
                        )

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

```

# File: ./rag/load_feedback_json.py

```python
import json
from feedback_store import FeedbackStore
from embedder import  Embedder

def load_feedback(path="data/feedback_data.json"):
    store = FeedbackStore()
    embedder = Embedder()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        emb = embedder.embed_query(item["question"])

        store.insert_feedback(
            question=item["question"],
            bad_answer=item["answer"],
            comment=item.get("comment", ""),
            rating=item.get("rating", 0),
            correct_answer=item.get("correct_answer", None),
            source="file",
            embedding=emb
        )

    store.close()
    print(f"✅ Загружено {len(data)} записей фидбека")

if __name__ == "__main__":
    load_feedback()

```

# File: ./rag/generator.py

```python
import requests
import json
import re
from typing import List, Optional

from rag.feedback_store import FeedbackStore
from rag.embedder import Embedder

CYRILLIC_RE = re.compile('[\u0400-\u04FF]')


def detect_language(text: str) -> str:
    """
    Простая эвристика: если есть кириллица -> 'ru', иначе 'en'.
    Можно заменить на langdetect/fasttext для лучшей точности.
    """
    if not text:
        return "en"
    if CYRILLIC_RE.search(text):
        return "ru"
    return "en"


class Generator:
    def __init__(self, model_name="llama3:8b-instruct-q4_0", base_url="http://localhost:11434",
                 embedder: Embedder = None, feedback_store: FeedbackStore = None):
        self.model = model_name
        self.base_url = base_url
        self.feedback_store = feedback_store or FeedbackStore()
        self.embedder = embedder or Embedder()

    def load_feedback(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            entry["question"]: entry.get("correct_answer")
            for entry in data
            if entry["rating"] <= 3 and entry.get("correct_answer") and entry["correct_answer"] != "N/A"
        }

    def generate_answer(self, question: str, context_chunks: List[str], user_id: Optional[int] = None) -> str:
        query_emb = self.embedder.embed_query(question)

        if user_id is not None:
            feedback_result = self.feedback_store.search_similar(query_emb, top_k=1, owner_id=user_id)
        else:
            feedback_result = self.feedback_store.search_similar(query_emb, top_k=1, owner_id=None)

        if feedback_result:
            fb = feedback_result[0]

            if fb.get("distance") is not None and fb["distance"] < 0.25:
                if fb.get("correct_answer") and fb["correct_answer"] != "N/A":
                    context_chunks.insert(0, f"(Исправленный ответ из фидбека)\n{fb['correct_answer']}")
                # if fb.get("rating") <= 3 and fb.get("correct_answer") and fb["correct_answer"] != "N/A":
                #     context_chunks.insert(0, f"(Исправленный ответ из фидбека)\n{fb['correct_answer']}")
                #
                # elif fb["rating"] >= 4 and fb.get("answer"):
                #     context_chunks.insert(0, f"(Ранее клиент получил этот ответ, он был оценён {fb['rating']}/5)\n{fb['answer']}")

        context_text = "\n\n".join(context_chunks)

        user_lang = detect_language(question)

        if user_lang == "ru":
            lang_instruction = "Пожалуйста, отвечай только на том языке, на котором задан вопрос. Если вопрос на русском — отвечай по-русски."
        else:
            lang_instruction = "Please answer only in the same language as the user's question."

        system_prefix = (
            "You are a helpful assistant that MUST answer in the same language as the user's question.\n"
            "If the user's question is in Russian, answer only in Russian.\n"
            "If the user's question is in another language, answer in that language.\n"
            "Do not translate the user's question or introduce other languages.\n"
            "Always cite the source documents (by title or short excerpt) used when answering.\n\n"
            "Пожалуйста, используй только данные из контекста и не выдумывай фактов. "
            "Если данных недостаточно — честно скажи, что информации недостаточно."
        )



        prompt = f"""
        {system_prefix}
        {lang_instruction}
        
        Ты юридический помощник. Ответь на вопрос на основе приведённого контекста.
        
        Ограничения:
        1. Используй факты из контекста, если информаций нет то укажи, что приведенных данных недостаточно.
        2. Если в контексте есть ссылки на официальные источники - упоминай их в ответе.
        3. если контекст содержит этапы по решению вопроса то включи их.
        4. Не придумывай законы, услуги или факты, которых нет в контексте.
        

        Контекст:
        {context_text}
        
        Вопрос: {question}
        Ответ:"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False}
            )
        except requests.exceptions.RequestException as e:
            raise Exception(f"LLM request error: {e}")

        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            raise Exception(f"Ollama error: {response.status_code} - {response.text}")

```

# File: ./rag/feedback_store.py

```python
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

```

# File: ./rag/embedder.py

```python
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from tqdm import tqdm
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
            Инициализация эмбеддера.

            Рекомендуемые модели:
            - all-MiniLM-L6-v2 (384 dim) - быстрая, хорошо для общих задач
            - paraphrase-multilingual-MiniLM-L12-v2 (384 dim) - лучше для русского/казахского
            - all-mpnet-base-v2 (768 dim) - точнее, но медленнее
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Embedder инициализирован: {model_name} (размерность: {self.embedding_dim})")

    def embed_articles(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        results = []

        batch_texts = []
        batch_indices = []

        for idx, article in enumerate(articles):
            title = article.get("title", "")
            text = article.get("text", "")

            # Комбинируем title и text для лучшего контекста
            # Формат важен: title помогает при поиске по ключевым словам
            combined = f"{title}\n\n{text}" if title else text
            batch_texts.append(combined)
            batch_indices.append(idx)

            # Батчевое эмбеддинг (быстрее чем по одному)
        print(f"Эмбеддинг {len(batch_texts)} статей...")
        embeddings = self.model.encode(
            batch_texts,
            show_progress_bar=True,
            batch_size=32,  # оптимально для CPU
            normalize_embeddings=True  # ВАЖНО: нормализация для лучшего поиска
        )

        # Формируем результаты
        for idx, embedding in zip(batch_indices, embeddings):
            article = articles[idx]
            results.append({
                "title": article.get("title", ""),
                "text": article.get("text", ""),
                "embedding": embedding,
                "egov_link": article.get("egov_link", ""),
                "egov_link_kaz": article.get("egov_link_kaz", ""),
                "source": article.get("source", "dataset"),
                "filename": article.get("filename", None)
            })

        return results

    def embed_query(self, query: str) -> np.ndarray:
        """
        Эмбеддит поисковый запрос.

        ВАЖНО: Используем normalize_embeddings=True для консистентности с базой
        """
        embedding = self.model.encode(
            query,
            normalize_embeddings=True  # должно совпадать с embed_articles
        )
        return embedding

    def get_embedding_dimension(self) -> int:
        """Возвращает размерность эмбеддингов модели"""
        return self.embedding_dim

```

# File: ./rag/loader.py

```python
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

```


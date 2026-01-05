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
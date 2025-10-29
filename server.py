# server.py
import os
import io
import json
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ==== твои модули (как в gradio-версии) ====
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.feedback_store import FeedbackStore
from rag.generator import Generator
import pickle

APP_PORT = int(os.environ.get("PORT", "8000"))
DATA_DIR = os.path.abspath("./data/uploads")
STATIC_DIR = os.path.abspath("./static")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ---- Инициализация ядра (как у тебя в Gradio) ----
try:
    with open("embedded.pkl", "rb") as f:
        embedded = pickle.load(f)
except FileNotFoundError:
    embedded = []

store = VectorStore()
feedback_store = FeedbackStore()
if embedded:
    store.insert_articles(embedded)

gen = Generator()
embedder = Embedder()

app = FastAPI(title="Tanym Loop API", version="0.1")

# Разрешим фронту дергать API (если будешь открывать HTML как файл)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # при желании сузить: ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Отдаём статические файлы (куда положим HTML)
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
    return {"status": "ok"}

# ====== Q&A ======
@app.post("/ask")
async def ask(question: str = Form(...), show_articles: bool = Form(True)):
    """
    Возвращает ответ + markdown с источниками/похожими статьями.
    """
    if not question.strip():
        return {"answer": "❌ Введите вопрос.", "sources_md": ""}

    query_emb = embedder.embed_query(question)
    results = store.search_similar(query_emb, top_k=2)
    context_chunks = [f"{r['title']}\n{r['text']}" for r in results]
    answer = gen.generate_answer(question, context_chunks)

    # Ссылки
    links_lines = ["\n\n📎 **Полезные ссылки:**"]
    for r in results:
        if r.get("egov_link"):
            links_lines.append(f"- [{r['title']}]({r['egov_link']})")
        if r.get("egov_link_kaz"):
            links_lines.append(f"  [Қазақша сілтеме]({r['egov_link_kaz']})")
    links_block = "\n".join(links_lines)

    sources_md = ""
    if show_articles:
        sources_md = "\n\n".join(context_chunks) + links_block
    else:
        sources_md = links_block

    return {"answer": answer, "sources_md": sources_md}

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
    Сохраняет отзыв в твой FeedbackStore.
    Если question/answer не переданы с фронта, просто откажем.
    """
    if not question or not answer:
        return {"ok": False, "msg": "❌ Сначала задайте вопрос и получите ответ."}

    # (опц) Валидация корректного ответа — можно добавить, как у тебя, позже
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
    return {"ok": True, "msg": "✅ Спасибо! Отзыв сохранён."}

# ====== INGEST ======
@app.post("/ingest")
async def ingest(
    files: List[UploadFile] = File(...),
    source_tag: str = Form("user"),
):
    """
    Принимает файлы, сохраняет в ./data/uploads и возвращает список имён.
    Здесь ты можешь подключить свой пайплайн:
      - распарсить текст
      - сделать эмбеддинги
      - store.insert_articles(new_items)
      - (опц.) пересохранить embedded.pkl
    """
    saved = []
    for uf in files:
        name = uf.filename
        dest = os.path.join(DATA_DIR, name)
        # гарантия, что папка существует
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        content = await uf.read()
        with open(dest, "wb") as f:
            f.write(content)
        saved.append(name)

        # TODO: тут можно дернуть твой loader/parsers,
        #       сделать эмбеддинги и вставить в VectorStore.
        # Пример (псевдо):
        # text = extract_text(dest)  # сам реализуешь
        # emb = embedder.embed_doc(text)
        # store.insert_articles([{
        #    "title": name, "text": text, "embedding": emb,
        #    "egov_link": "", "egov_link_kaz": "", "source": source_tag
        # }])

    return {"ok": True, "saved": saved, "folder": os.path.abspath(DATA_DIR), "source": source_tag}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="localhost", port=APP_PORT, reload=True)

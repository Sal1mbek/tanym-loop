from rag.loader import load_articles
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.generator import Generator
from rag.feedback_store import FeedbackStore
import os
import pickle

EMBED_CACHE = "embedded.pkl"
EXCEL_FILE = "data/data_for_rag.xlsx"

# FILES = [
#     "data/Кодекс_Республики_Казахстан_об_административных_правонарушениях.docx",
#     "data/ЗРК о гос закупках.pdf",
#     "data/ЗРК о гос услугах.pdf"
# ]

def load_all_articles():
    return load_articles(EXCEL_FILE)
    # all_articles = []
    # for path in paths:
    #     articles = load_articles(path)
    #     all_articles.extend(articles)
    # return all_articles

if os.path.exists(EMBED_CACHE):
    with open(EMBED_CACHE, "rb") as f:
        embedded = pickle.load(f)
    print("✅ Эмбеддинги загружены из кэша")
else:
    articles = load_all_articles()
    print(f"Загружено статей: {len(articles)}")

    embedder = Embedder()
    embedded = embedder.embed_articles(articles)

    with open(EMBED_CACHE, "wb") as f:
        pickle.dump(embedded, f)
    print("✅ Эмбеддинги сохранены")

# Загрузка в векторную БД
store = VectorStore()
feedback = FeedbackStore()
store.insert_articles(embedded)
print("✅ Загружено в PostgreSQL (без дубликатов)")

# --- Поиск и генерация ответа ---
query = input("Введите ваш вопрос: ")
query_emb = Embedder().embed_query(query)
results = store.search_similar(query_emb, top_k=2)

print("\n🔍 Похожие статьи:")
for r in results:
    print(f"    \n {r['title']}")
    print(f"   📎 Ссылка: {r['egov_link']}")
    print(f"   📎 Ссылка (каз): {r['egov_link_kaz']}")
    print(f"   📚 Текст: {r['text']}...")

# Генерация ответа
gen = Generator()
context_chunks = [f"{r['title']}\n{r['text']}" for r in results]
answer = gen.generate_answer(query, context_chunks)

print("\n🤖 Ответ LLM:")
print(answer)

store.close()
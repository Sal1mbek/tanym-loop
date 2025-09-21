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
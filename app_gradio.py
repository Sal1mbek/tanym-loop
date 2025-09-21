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

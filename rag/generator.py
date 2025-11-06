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

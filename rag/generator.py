import requests
import json
from rag.feedback_store import FeedbackStore
from rag.embedder import Embedder


class Generator:
    def __init__(self, model_name="llama3:8b-instruct-q4_0", base_url="http://localhost:11434"): #mistral/llama3 испытывал
        self.model = model_name
        self.base_url = base_url
        self.feedback_store = FeedbackStore()

    def load_feedback(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            entry["question"]: entry.get("correct_answer")
            for entry in data
            if entry["rating"] <= 3 and entry.get("correct_answer") and entry["correct_answer"] != "N/A"
        }

    def generate_answer(self, question: str, context_chunks: list[str]) -> str:
        embedder = Embedder()
        query_emb = embedder.embed_query(question)

        feedback_result = self.feedback_store.search_similar(query_emb, top_k=1)

        if feedback_result:
            fb = feedback_result[0]

            if fb["distance"] < 0.25:
                if fb["rating"] <= 3 and fb.get("correct_answer") and fb["correct_answer"] != "N/A":
                    context_chunks.insert(0, f"(Исправленный ответ из фидбека)\n{fb['correct_answer']}")

                elif fb["rating"] >= 4 and fb.get("answer"):
                    context_chunks.insert(0, f"(Ранее клиент получил этот ответ, он был оценён {fb['rating']}/5)\n{fb['answer']}")

        context_text = "\n\n".join(context_chunks)
        prompt = f"""
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

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )

        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            raise Exception(f"Ollama error: {response.status_code} - {response.text}")

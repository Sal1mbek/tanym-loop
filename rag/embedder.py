import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

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

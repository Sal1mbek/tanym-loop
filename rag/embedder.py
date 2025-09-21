from sentence_transformers import SentenceTransformer
from typing import List, Dict
from tqdm import tqdm


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_articles(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        results = []
        for article in tqdm(articles, desc="Эмбеддинг статей"):
            title = article["title"]
            text = article["text"]
            egov_link = article["egov_link"]
            egov_link_kaz = article["egov_link_kaz"]
            # Можно использовать и только text, но иногда title полезен для контекста
            embedding = self.model.encode(f"{title}\n{text}")
            results.append({
                "title": title,
                "text": text,
                "embedding": embedding,
                "egov_link": egov_link,
                "egov_link_kaz": egov_link_kaz,
                "source": article.get("source", "dataset")
            })
        return results

    def embed_query(self, query: str):
        return self.model.encode(query)

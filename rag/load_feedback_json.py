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

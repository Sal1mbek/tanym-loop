from rag.loader import load_articles
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.generator import Generator
from rag.feedback_store import FeedbackStore
import os
import pickle

EMBED_CACHE = "embedded.pkl"
EXCEL_FILE = "data/data_for_rag.xlsx"

print("=" * 60)
print("🚀 Инициализация Tanym Loop")
print("=" * 60)


def load_all_articles():
    """Загружает статьи из Excel файла"""
    return load_articles(EXCEL_FILE)


# ============================================
# Шаг 1: Загрузка или создание эмбеддингов
# ============================================
if os.path.exists(EMBED_CACHE):
    print("\n📦 Загрузка эмбеддингов из кэша...")
    with open(EMBED_CACHE, "rb") as f:
        embedded = pickle.load(f)
    print(f"✅ Загружено {len(embedded)} статей из кэша (embedded.pkl)")
else:
    print("\n📄 Загрузка статей из Excel...")
    articles = load_all_articles()
    print(f"✅ Загружено {len(articles)} статей из {EXCEL_FILE}")

    print("\n🔄 Создание эмбеддингов...")
    print("   (Это займёт некоторое время при первом запуске)")
    embedder = Embedder()
    embedded = embedder.embed_articles(articles)

    print("\n💾 Сохранение эмбеддингов в кэш...")
    with open(EMBED_CACHE, "wb") as f:
        pickle.dump(embedded, f)
    print(f"✅ Эмбеддинги сохранены в {EMBED_CACHE}")

# ============================================
# Шаг 2: Загрузка в векторную БД
# ============================================
print("\n📊 Подключение к PostgreSQL...")
store = VectorStore()
feedback = FeedbackStore()

print("\n🔄 Индексация документов в БД...")
stats = store.insert_articles(embedded)
print(f"✅ Индексация завершена:")
print(f"   - Вставлено: {stats['inserted']}")
print(f"   - Пропущено (дубликаты): {stats['skipped']}")
print(f"   - Ошибки: {stats['errors']}")

# Статистика БД
db_stats = store.get_stats()
print(f"\n📈 Статистика базы данных:")
print(f"   - Всего документов: {db_stats['total_documents']}")
print(f"   - Уникальных источников: {db_stats['unique_sources']}")
print(f"   По источникам:")
for src, cnt in db_stats['by_source'].items():
    print(f"      • {src}: {cnt}")

print("\n" + "=" * 60)
print("🎯 Система готова к работе!")
print("=" * 60)

# ============================================
# Шаг 3: Интерактивный поиск и ответы
# ============================================
embedder_instance = Embedder()
gen = Generator()

# Порог релевантности (как в server.py)
RELEVANCE_THRESHOLD = 0.7

while True:
    print("\n" + "-" * 60)
    query = input("❓ Введите ваш вопрос (или 'exit' для выхода): ").strip()

    if query.lower() in ['exit', 'quit', 'выход']:
        print("\n👋 До свидания!")
        break

    if not query:
        print("⚠️  Пожалуйста, введите вопрос")
        continue

    # Эмбеддинг запроса
    query_emb = embedder_instance.embed_query(query)

    # Поиск похожих (top_k=5 для фильтрации)
    results = store.search_similar(query_emb, top_k=5)

    # Фильтрация по релевантности
    relevant_results = [r for r in results if r['distance'] < RELEVANCE_THRESHOLD]

    print(f"\n🔍 Результаты поиска:")
    print(f"   Найдено: {len(results)} документов")
    print(f"   Релевантных (distance < {RELEVANCE_THRESHOLD}): {len(relevant_results)}")

    if not relevant_results:
        print("\n❌ К сожалению, не найдено релевантных документов.")
        print("💡 Попробуйте:")
        print("   - Переформулировать вопрос")
        print("   - Использовать другие ключевые слова")
        continue

    # Используем топ-2 релевантных
    top_results = relevant_results[:2]

    print(f"   Использовано для ответа: {len(top_results)}")
    print(f"   Средняя релевантность: {int(sum(r['similarity'] for r in top_results) / len(top_results) * 100)}%")

    # Показываем источники
    print(f"\n📚 Использованные источники:")
    for idx, r in enumerate(top_results, 1):
        similarity_pct = int(r['similarity'] * 100)
        distance = r['distance']

        # Определяем уровень релевантности
        if similarity_pct >= 80:
            badge = "🟢 ОТЛИЧНО"
        elif similarity_pct >= 60:
            badge = "🟡 ХОРОШО"
        else:
            badge = "🟠 СРЕДНЕ"

        print(f"\n   {idx}. {badge} (релевантность: {similarity_pct}%, distance: {distance:.3f})")
        print(f"      📝 {r['title']}")

        # Превью текста
        preview = r['text'][:150] + "..." if len(r['text']) > 150 else r['text']
        print(f"      💬 {preview}")

        # Метаданные
        print(f"      📁 Источник: {r['source']}")

        # Ссылки
        if r.get('egov_link'):
            print(f"      🔗 Ссылка: {r['egov_link']}")
        if r.get('egov_link_kaz'):
            print(f"      🔗 Қазақша: {r['egov_link_kaz']}")

    # Генерация ответа
    print(f"\n🤖 Генерация ответа...")
    context_chunks = [f"{r['title']}\n{r['text']}" for r in top_results]
    answer = gen.generate_answer(query, context_chunks)

    print(f"\n" + "=" * 60)
    print("📋 ОТВЕТ:")
    print("=" * 60)
    print(answer)
    print("=" * 60)

# Закрываем соединение
store.close()
print("\n✅ Соединение с БД закрыто")
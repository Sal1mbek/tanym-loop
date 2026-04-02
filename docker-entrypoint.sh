#!/bin/bash
set -e

echo "Starting Tanym Loop Docker container..."

echo "Waiting for PostgreSQL..."
until pg_isready -h postgres -p 5432 -U postgres; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 2
done
echo "PostgreSQL is ready!"

echo "Waiting for Ollama..."
until curl -s http://ollama:11434/api/tags > /dev/null 2>&1; do
  echo "Ollama is unavailable - sleeping"
  sleep 5
done
echo "Ollama is ready!"

echo "Pulling model llama3:8b-instruct-q4_0 via API..."
curl -s http://ollama:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name":"llama3:8b-instruct-q4_0"}' || echo "Warning: pull failed or model exists"

mkdir -p /app/data/uploads /app/logs

echo "Running database initialization..."
python -c "
from rag.vectorstore import VectorStore
from rag.feedback_store import FeedbackStore
print('Database tables initialized')
"

echo "Starting Tanym Loop server..."
exec "$@"
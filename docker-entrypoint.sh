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
until curl -s http://ollama:11434/api/tags > /dev/null; do
  echo "Ollama is unavailable - sleeping"
  sleep 5
done
echo "Ollama is ready!"

echo "Checking for required models..."
if ! ollama list | grep -q "llama3:8b-instruct-q4_0"; then
  echo "Pulling llama3:8b-instruct-q4_0..."
  ollama pull llama3:8b-instruct-q4_0
fi

if ! ollama list | grep -q "llama3:latest"; then
  echo "Pulling llama3:latest..."
  ollama pull llama3
fi

echo "All models ready!"

mkdir -p /app/data/uploads /app/logs

echo "Running database initialization..."
python -c "
from rag.vectorstore import VectorStore
from rag.feedback_store import FeedbackStore
print('Database tables initialized')
"

echo "Starting Tanym Loop server..."
exec "$@"
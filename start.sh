#!/bin/bash

# kích hoạt venv
source .venv/Scripts/activate

echo "Starting FastAPI..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

echo "Starting Celery..."
celery -A app.workers.tasks worker --loglevel=info

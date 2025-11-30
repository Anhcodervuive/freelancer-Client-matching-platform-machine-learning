#!/bin/bash

# kích hoạt venv
source .venv/Scripts/activate

echo "Starting FastAPI..."
uvicorn app.main:app

echo "Starting Celery..."
celery -A app.workers.tasks worker --loglevel=info

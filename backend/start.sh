#!/bin/bash
# Start both Celery worker and Uvicorn in a single container.
# Used for free-tier hosting (e.g. Render) where you can only run one service.

set -e

# Start Celery worker in the background
celery -A backend.worker worker --loglevel=info --concurrency=1 &

# Start FastAPI (foreground – keeps the container alive)
exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}

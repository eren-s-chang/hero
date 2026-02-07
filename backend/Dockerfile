# ---------- build stage ----------
FROM python:3.10-slim AS base

# Prevents Python from writing .pyc files and enables unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps required by MediaPipe / OpenCV-headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/backend/

# Make start script executable
RUN chmod +x /app/backend/start.sh

# Ensure the upload directory exists
RUN mkdir -p /tmp/formperfect_uploads

EXPOSE 8000

# Run both FastAPI + Celery worker via start script
CMD ["/app/backend/start.sh"]

# ---------- build stage ----------
FROM python:3.10-slim AS base

# Prevents Python from writing .pyc files and enables unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps required by MediaPipe / OpenCV-headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
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

# Ensure the upload directory exists
RUN mkdir -p /tmp/formperfect_uploads

EXPOSE 8000

# Default command – overridden per-service in docker-compose
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

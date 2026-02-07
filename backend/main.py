"""
FormPerfect – FastAPI entry-point.

Endpoints
---------
POST /analyze        – Upload a workout video → kicks off async Celery task.
GET  /result/{id}    – Poll for the analysis result.
GET  /health         – Liveness probe.
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.worker import analyze_video  # Celery task
from celery.result import AsyncResult

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FormPerfect API",
    version="0.1.0",
    description="Analyse workout videos with MediaPipe + Gemini.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compress_video(raw_bytes: bytes, ext: str) -> bytes:
    """Re-encode video with FFmpeg to shrink it before sending to Redis.

    Strategy: scale to 480p, CRF 30, fast preset, strip audio.
    Falls back to the original bytes if FFmpeg is unavailable or fails.
    """
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as src:
        src.write(raw_bytes)
        src_path = src.name

    dst_path = src_path + ".compressed.mp4"

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src_path,
                "-vf", "scale=-2:480",      # max 480p height
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "30",
                "-an",                       # strip audio
                "-movflags", "+faststart",
                dst_path,
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        compressed = Path(dst_path).read_bytes()
        ratio = len(compressed) / len(raw_bytes) * 100
        logger.info(
            "Compressed %s: %.1f MB → %.1f MB (%.0f%%)",
            ext,
            len(raw_bytes) / 1e6,
            len(compressed) / 1e6,
            ratio,
        )
        return compressed

    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("FFmpeg compression failed, using original bytes: %s", exc)
        return raw_bytes

    finally:
        Path(src_path).unlink(missing_ok=True)
        Path(dst_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    """Accept a video upload, persist it, and dispatch the analysis task."""

    # --- Validate extension ---------------------------------------------------
    ext = Path(video.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # --- Read file into memory -------------------------------------------------
    try:
        video_bytes = await video.read()
    finally:
        await video.close()

    file_size_mb = len(video_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size_mb:.1f} MB). Max {MAX_FILE_SIZE_MB} MB.",
        )

    # --- Compress before sending through Redis --------------------------------
    video_bytes = _compress_video(video_bytes, ext)

    # --- Dispatch Celery task (video bytes as base64) -------------------------
    video_b64 = base64.b64encode(video_bytes).decode("ascii")

    try:
        task = analyze_video.delay(video_b64, ext)
    except Exception as exc:
        logger.error("Failed to enqueue task (Redis may be full): %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. The processing queue is full. Please try again later.",
        )

    return JSONResponse(
        status_code=202,
        content={"task_id": task.id, "detail": "Video accepted. Processing started."},
    )


@app.get("/result/{task_id}")
async def result(task_id: str):
    """Poll the status / result of an analysis task."""

    task_result = AsyncResult(task_id, app=analyze_video.app)

    if task_result.state == "PENDING":
        return {"status": "Processing", "detail": "Task is queued or in progress."}

    if task_result.state == "STARTED":
        return {"status": "Processing", "detail": "Task is currently running."}

    if task_result.state == "FAILURE":
        return JSONResponse(
            status_code=500,
            content={
                "status": "Failed",
                "detail": str(task_result.info),
            },
        )

    if task_result.state == "SUCCESS":
        return {"status": "Completed", "result": task_result.result}

    # Catch-all for custom states
    return {"status": task_result.state}

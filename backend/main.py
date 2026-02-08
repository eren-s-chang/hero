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
import json
import logging
import os
from pathlib import Path

import redis
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from backend.worker import analyze_video  # Celery task
from celery.result import AsyncResult

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
_redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)

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


@app.get("/landmarks/{task_id}")
async def landmarks(task_id: str):
    """Return the raw MediaPipe landmarks buffered during analysis.

    The frontend can use these to draw a skeleton overlay on the video.
    Each frame has: {time_s: float, landmarks: [{name, x, y, z}, ...]}.
    Coordinates are normalised 0-1 relative to the video frame.
    """
    redis_key = f"landmarks:{task_id}"
    data = _redis.get(redis_key)

    if data is None:
        raise HTTPException(
            status_code=404,
            detail="Landmarks not found. They may have expired or the task hasn't completed yet.",
        )

    return {"task_id": task_id, "frames": json.loads(data)}


@app.get("/rep-frame/{task_id}/{index}")
async def rep_frame(task_id: str, index: int):
    """Return a single frame JPEG by flat index (across all reps and frame types).
    
    New data structure (post-enhancement):
    [
      {"rep_number": 1, "frames": {"start": {...}, "mid": {...}, "end": {...}}},
      {"rep_number": 2, "frames": {...}},
      ...
    ]
    
    This flattens to: rep1_start, rep1_mid, rep1_end, rep2_start, rep2_mid, ...
    """
    redis_key = f"rep_frames:{task_id}"
    data = _redis.get(redis_key)

    if data is None:
        raise HTTPException(
            status_code=404,
            detail="Rep frames not found. They may have expired or the task hasn't completed yet.",
        )

    payload = json.loads(data)
    
    # Support both data formats:
    # Old/current format: [{"t": ..., "b64": "..."}, ...]
    # New format: [{"rep_number": 1, "frames": {"start": {...}, "mid": {...}, "end": {...}}}, ...]
    flat_frames = []
    if payload and isinstance(payload[0], dict):
        if "b64" in payload[0]:
            # Old flat format – each item is already a frame
            flat_frames = payload
        elif "frames" in payload[0]:
            # New nested format – flatten start/mid/end per rep
            for rep_data in payload:
                for frame_type in ["start", "mid", "end"]:
                    if frame_type in rep_data.get("frames", {}):
                        flat_frames.append(rep_data["frames"][frame_type])
    
    if index < 0 or index >= len(flat_frames):
        raise HTTPException(
            status_code=404,
            detail=f"Frame index {index} out of range (0–{len(flat_frames) - 1}). Total frames: {len(flat_frames)}",
        )

    frame_data = flat_frames[index]
    return Response(
        content=base64.b64decode(frame_data["b64"]),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/correction-audio/{task_id}")
async def correction_audio(task_id: str):
    """Return the synthesized correction audio as MP3.

    The audio is stored in Redis as base64-encoded data during analysis.
    Returns {\"audio_url\": \"data:audio/mpeg;base64,...\", \"text\": \"...\"}."""
    redis_key = f"correction_audio:{task_id}"
    data = _redis.get(redis_key)

    if data is None:
        raise HTTPException(
            status_code=404,
            detail="Correction audio not found. It may have expired or the task hasn't completed yet.",
        )

    payload = json.loads(data)
    audio_b64 = payload.get("audio", "")
    correction_text = payload.get("text", "")

    return {
        "audio_url": f"data:audio/mpeg;base64,{audio_b64}",
        "text": correction_text,
    }

"""
FormPerfect – Celery worker.

Pipeline
--------
1. Decode video with OpenCV, process every frame.
2. Run MediaPipe Pose to extract 33×3 landmarks per frame.
3. Flatten landmarks into a compact text representation.
4. Send to Gemini 1.5 Pro with a strict JSON system prompt.
5. Return the parsed JSON to the result backend.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import google.generativeai as genai
from celery import Celery

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Celery setup
# ---------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "backend.worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,          # auto-delete results after 1 hour
)

# ---------------------------------------------------------------------------
# Gemini config
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

SYSTEM_PROMPT = (
    "You are an elite fitness coach. "
    "Analyze this time-series of 3D pose landmarks from a workout video. "
    "Identify the exercise being performed. "
    "Output **strictly valid JSON** with the following keys:\n"
    "  • exercise_detected  (string)\n"
    "  • form_rating_1_to_10  (integer 1-10)\n"
    "  • main_mistakes  (list of strings)\n"
    "  • actionable_correction  (string)\n"
    "Do NOT wrap the JSON in markdown code-fences. Return raw JSON only."
)

# ---------------------------------------------------------------------------
# MediaPipe helpers
# ---------------------------------------------------------------------------
LANDMARK_NAMES = [lm.name for lm in mp.solutions.pose.PoseLandmark]


def _extract_landmarks(video_path: str, every_n: int = 1) -> str:
    """Read *video_path*, run MediaPipe Pose on every *every_n*-th frame,
    and return a compact text representation of all sampled landmarks.

    Format (one line per sampled frame):
        frame=<N> | <landmark_name>:(x,y,z) <landmark_name>:(x,y,z) ...
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,  # 0=lite, 1=full, 2=heavy
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    lines: list[str] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_landmarks:
                    parts: list[str] = []
                    for lm, name in zip(result.pose_landmarks.landmark, LANDMARK_NAMES):
                        parts.append(f"{name}:({lm.x:.4f},{lm.y:.4f},{lm.z:.4f})")
                    lines.append(f"frame={frame_idx} | " + " ".join(parts))

            frame_idx += 1
    finally:
        cap.release()
        pose.close()

    if not lines:
        raise RuntimeError("MediaPipe could not detect any pose in the video.")

    logger.info("Extracted landmarks from %d / %d frames", len(lines), frame_idx)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gemini helper
# ---------------------------------------------------------------------------
def _interpret_with_gemini(landmark_text: str) -> dict[str, Any]:
    """Send the landmark data to Gemini and parse the JSON response."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    response = model.generate_content(
        f"Here are the 3D pose landmarks extracted from a workout video:\n\n{landmark_text}",
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )

    raw = response.text.strip()

    # Strip markdown code fences if the model wraps them anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Gemini returned non-JSON: %s", raw)
        return {
            "exercise_detected": "unknown",
            "form_rating_1_to_10": 0,
            "main_mistakes": ["Analysis could not be parsed."],
            "actionable_correction": raw,
        }


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------
@celery_app.task(name="analyze_video", bind=True, max_retries=2)
def analyze_video(self, video_b64: str, ext: str = ".mp4") -> dict[str, Any]:
    """Full pipeline: decode base64 → temp file → MediaPipe → Gemini → JSON."""
    # Write base64 video to a temp file so OpenCV can read it
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp_path = Path(tmp.name)
    try:
        tmp.write(base64.b64decode(video_b64))
        tmp.close()
        logger.info("Wrote %.1f MB to %s", tmp_path.stat().st_size / 1e6, tmp_path.name)

        # Step A – extract pose landmarks
        logger.info("Step A: extracting landmarks from %s", tmp_path.name)
        landmark_text = _extract_landmarks(str(tmp_path), every_n=5)

        # Step B – interpret with Gemini
        logger.info("Step B: sending %d chars to Gemini", len(landmark_text))
        analysis = _interpret_with_gemini(landmark_text)

        return analysis

    except Exception as exc:
        logger.exception("analyze_video failed")
        raise self.retry(exc=exc, countdown=5)

    finally:
        tmp_path.unlink(missing_ok=True)
        logger.info("Cleaned up %s", tmp_path.name)

"""
FormPerfect – Celery worker.

Pipeline
--------
1. Decode video with OpenCV, sample ~3 frames/sec.
2. Run MediaPipe Pose to extract 33×3 landmarks per frame.
3. Compute joint angles (knee, hip, elbow, shoulder, spine, ankle).
4. Send compact angle time-series to Gemini for coaching analysis.
5. Buffer raw landmarks in Redis for the frontend to overlay on the video.
6. Return the parsed JSON to the result backend.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import redis
from google import genai
from google.genai import types
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
# Redis client (for buffering landmarks for frontend overlay)
# ---------------------------------------------------------------------------
_redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
LANDMARKS_TTL = 3600  # same as result_expires

# ---------------------------------------------------------------------------
# Gemini config
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

SYSTEM_PROMPT = (
   "You are an elite Biomechanical Signal Processing Expert and Strength Coach. "
"You will receive a time-series of computed joint angles (in degrees) "
"sampled from a workout video. Each line is one sampled frame.\n\n"

"PHASE 1: SIGNAL CLEANING & SEGMENTATION\n"
"1. Scan the time-series to identify the 'Active Work Set'. Ignore the first and last 10-15% of frames "
"if they represent setup (walking to position) or teardown (racking weights). Only analyze the periodic signal in the middle.\n"
"2. Identify local minima and maxima in the joint angles to count reps. A valid rep MUST have a distinct Eccentric (lowering) "
"and Concentric (lifting) phase. If the signal is chaotic or non-periodic, set 'analysis_allowed' to false.\n\n"

"PHASE 2: KINEMATIC FINGERPRINTING (The 'What')\n"
"Classify the exercise by analyzing 'Joint Coupling' (how joints move relative to each other):\n"
"   - SQUAT PATTERN: Knee Flexion and Hip Flexion are COUPLED (move in-phase together). High ROM for both.\n"
"   - HINGE/DEADLIFT PATTERN: Hip Flexion is DOMINANT. Knee Flexion is subordinate (<40% of Hip ROM).\n"
"   - LUNGE PATTERN: Asymmetrical Hip/Knee angles (if split data provided) or similar to squat but with stability noise.\n"
"   - PRESS PATTERN: Elbow Extension coupled with Shoulder Flexion/Abduction.\n"
"   - PULL PATTERN: Elbow Flexion coupled with Shoulder Extension/Adduction.\n"
"   - ISOLATION: One joint has High ROM (Primary Mover), adjacent joints are STATIC (Stabilizers).\n"
"Reject analysis if the motion matches a Sport (tennis, running), Cardio, or Dance.\n\n"

"PHASE 3: BIOMECHANICAL SCORING (The 'How')\n"
"If an exercise is detected, compare the user's angles to the 'Bio-Mechanical Ideal' for that SPECIFIC movement.\n"
"Score strictly using this Hierarchy of Errors:\n"
"1. CRITICAL FAULT (Max Score: 3/10): Safety violation. (e.g., Lumbar spine flexion in a Deadlift, Knee valgus in a Squat).\n"
"2. MAJOR FAULT (Max Score: 6/10): Range of Motion failure. (e.g., Squat depth < 90deg knee flexion, half-reps).\n"
"3. MINOR FAULT (Max Score: 8/10): Efficiency leaks. (e.g., Elbow drift in a Curl, looking up/down excessively).\n"
"4. OPTIMAL (Score: 9-10/10): Angles align with the Ideal Model within 5% variance.\n\n"

"PHASE 4: OUTPUT GENERATION\n"
"Analyze the data and output **strictly valid JSON** with:\n"
"  • analysis_allowed  (boolean – true only if a periodic gym exercise is found)\n"
"  • rejection_reason  (string – if false, explain why based on signal noise or sport detection)\n"
"  • exercise_detected  (string)\n"
"  • rep_count  (integer – derived from the count of concentric/eccentric peaks)\n"
"  • form_rating_1_to_10  (integer – global average based on the Hierarchy of Errors above)\n"
"  • main_mistakes  (list of strings – specific deviations from the Optimal Angles)\n"
"  • rep_analyses  (list of objects, one per rep, containing rep_number, timestamp_start, timestamp_end, rating_1_to_10, and mistakes)\n"
"  • problem_joints  (list of strings – the angle names (from the input data columns) that have the WORST form deviations, "
"ordered from most problematic to least. Use the EXACT column names from the input like 'L_knee', 'R_hip', 'spine', etc. "
"Include 1-3 joints maximum – only those with Critical or Major faults.)\n"
"  • actionable_correction  (string – the single most impactful fix for the Critical or Major faults)\n"
"If analysis_allowed is false, set exercise_detected to 'unrecognized', rep_count to 0, "
"form_rating_1_to_10 to 0, main_mistakes to [], rep_analyses to [], and provide a short correction.\n"
"Do NOT wrap the JSON in markdown code-fences. Return raw JSON only."
)

RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis_allowed": {"type": "boolean"},
        "rejection_reason": {"type": "string"},
        "exercise_detected": {"type": "string"},
        "rep_count": {"type": "integer"},
        "form_rating_1_to_10": {"type": "integer"},
        "main_mistakes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "rep_analyses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rep_number": {"type": "integer"},
                    "timestamp_start": {"type": "number"},
                    "timestamp_end": {"type": "number"},
                    "rating_1_to_10": {"type": "integer"},
                    "mistakes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["rep_number", "timestamp_start", "timestamp_end", "rating_1_to_10", "mistakes"],
            },
        },
        "problem_joints": {
            "type": "array",
            "items": {"type": "string"},
        },
        "actionable_correction": {"type": "string"},
    },
    "required": [
        "analysis_allowed",
        "rejection_reason",
        "exercise_detected",
        "rep_count",
        "form_rating_1_to_10",
        "main_mistakes",
        "rep_analyses",
        "problem_joints",
        "actionable_correction",
    ],
}

# ---------------------------------------------------------------------------
# MediaPipe helpers
# ---------------------------------------------------------------------------
PoseLandmark = mp.solutions.pose.PoseLandmark
LANDMARK_NAMES = [lm.name for lm in PoseLandmark]

TARGET_SAMPLES_PER_SEC = 10  # ~10 snapshots per second of video

# Joint-angle definitions: (point_a, vertex, point_b)
# The angle is measured at the *vertex* joint.
ANGLE_DEFS: dict[str, tuple[PoseLandmark, PoseLandmark, PoseLandmark]] = {
    "L_knee":      (PoseLandmark.LEFT_HIP,       PoseLandmark.LEFT_KNEE,      PoseLandmark.LEFT_ANKLE),
    "R_knee":      (PoseLandmark.RIGHT_HIP,      PoseLandmark.RIGHT_KNEE,     PoseLandmark.RIGHT_ANKLE),
    "L_hip":       (PoseLandmark.LEFT_SHOULDER,   PoseLandmark.LEFT_HIP,       PoseLandmark.LEFT_KNEE),
    "R_hip":       (PoseLandmark.RIGHT_SHOULDER,  PoseLandmark.RIGHT_HIP,      PoseLandmark.RIGHT_KNEE),
    "L_elbow":     (PoseLandmark.LEFT_SHOULDER,   PoseLandmark.LEFT_ELBOW,     PoseLandmark.LEFT_WRIST),
    "R_elbow":     (PoseLandmark.RIGHT_SHOULDER,  PoseLandmark.RIGHT_ELBOW,    PoseLandmark.RIGHT_WRIST),
    "L_shoulder":  (PoseLandmark.LEFT_ELBOW,      PoseLandmark.LEFT_SHOULDER,  PoseLandmark.LEFT_HIP),
    "R_shoulder":  (PoseLandmark.RIGHT_ELBOW,     PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    "L_ankle":     (PoseLandmark.LEFT_KNEE,       PoseLandmark.LEFT_ANKLE,     PoseLandmark.LEFT_FOOT_INDEX),
    "R_ankle":     (PoseLandmark.RIGHT_KNEE,      PoseLandmark.RIGHT_ANKLE,    PoseLandmark.RIGHT_FOOT_INDEX),
    "spine":       (PoseLandmark.LEFT_SHOULDER,    PoseLandmark.LEFT_HIP,       PoseLandmark.LEFT_KNEE),
    # ↑ spine forward lean approximated via shoulder-hip-knee on the left side
}

# Map angle names → landmark names involved (used by frontend to highlight in red)
ANGLE_TO_LANDMARKS: dict[str, list[str]] = {
    name: [PoseLandmark(a).name, PoseLandmark(v).name, PoseLandmark(b).name]
    for name, (a, v, b) in ANGLE_DEFS.items()
}


def _angle_between(a, vertex, b) -> float:
    """Return the angle in degrees at *vertex* formed by points *a* and *b*.

    Each point is a MediaPipe landmark with .x, .y, .z attributes.
    """
    ax, ay, az = a.x - vertex.x, a.y - vertex.y, a.z - vertex.z
    bx, by, bz = b.x - vertex.x, b.y - vertex.y, b.z - vertex.z

    dot = ax * bx + ay * by + az * bz
    mag_a = math.sqrt(ax * ax + ay * ay + az * az) or 1e-9
    mag_b = math.sqrt(bx * bx + by * by + bz * bz) or 1e-9

    cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
    return math.degrees(math.acos(cos_angle))


def _compute_angles(landmarks) -> dict[str, float]:
    """Compute all defined joint angles from a single frame's landmarks."""
    lm = landmarks.landmark
    angles: dict[str, float] = {}
    for name, (a_idx, v_idx, b_idx) in ANGLE_DEFS.items():
        angles[name] = round(_angle_between(lm[a_idx], lm[v_idx], lm[b_idx]), 1)
    return angles


def _process_video(
    video_path: str, angle_every_n: int | None = None
) -> tuple[str, list[dict[str, Any]]]:
    """Run MediaPipe on *every* frame for smooth frontend overlay,
    but only compute joint angles on sampled frames for the LLM.

    Returns
    -------
    angle_text : str
        Compact time-series of joint angles (sampled) for the LLM.
    raw_frames : list[dict]
        Per-frame raw landmark data (every frame) for the frontend overlay.
        Each dict: {"time_s": float, "landmarks": [{name, x, y, z}, ...]}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if angle_every_n is None:
        angle_every_n = max(1, round(fps / TARGET_SAMPLES_PER_SEC))
    logger.info(
        "Landmarks: every frame | Angles: every %d-th frame (fps=%.1f, ~%d angle samples/sec)",
        angle_every_n, fps, round(fps / angle_every_n),
    )

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    angle_lines: list[str] = []
    raw_frames: list[dict[str, Any]] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_s = round(frame_idx / fps, 3)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                # ── Raw landmarks → Redis/frontend (EVERY frame) ──
                lm_list = [
                    {"name": name, "x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)}
                    for lm, name in zip(
                        result.pose_landmarks.landmark, LANDMARK_NAMES
                    )
                ]
                raw_frames.append({"time_s": timestamp_s, "landmarks": lm_list})

                # ── Angles → LLM (sampled frames only) ───────────
                if frame_idx % angle_every_n == 0:
                    angles = _compute_angles(result.pose_landmarks)
                    angle_parts = [f"{k}={v}" for k, v in angles.items()]
                    angle_lines.append(
                        f"t={timestamp_s:.3f} " + " ".join(angle_parts)
                    )

            frame_idx += 1
    finally:
        cap.release()
        pose.close()

    if not angle_lines:
        raise RuntimeError("MediaPipe could not detect any pose in the video.")

    logger.info(
        "Processed %d frames → %d landmark frames, %d angle samples",
        frame_idx, len(raw_frames), len(angle_lines),
    )
    return "\n".join(angle_lines), raw_frames


# ---------------------------------------------------------------------------
# Gemini helper
# ---------------------------------------------------------------------------
def _interpret_with_gemini(angle_text: str) -> dict[str, Any]:
    """Send the joint-angle time-series to Gemini and parse the JSON response."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Here are the joint angles (degrees) extracted from a workout video:\n\n"
        f"{angle_text}"
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=8192,
                response_mime_type="application/json",
                response_schema=RESPONSE_JSON_SCHEMA,
            ),
        )
    except Exception as exc:
        logger.warning(
            "Structured output schema not supported by SDK; falling back to JSON mode only: %s",
            exc,
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            },
        )

    if getattr(response, "parsed", None):
        return response.parsed

    raw = response.text.strip()

    # Strip markdown code fences if the model wraps them anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # Best-effort extraction if extra text sneaks in
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        logger.error("Gemini returned non-JSON: %s", raw)
        return {
            "analysis_allowed": False,
            "rejection_reason": "Could not parse analysis output.",
            "exercise_detected": "unknown",
            "rep_count": 0,
            "form_rating_1_to_10": 0,
            "main_mistakes": ["Analysis could not be parsed."],
            "rep_analyses": [],
            "actionable_correction": raw,
        }


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------
def _ensure_mp4(video_path: Path) -> Path:
    """Convert non-MP4 videos (especially WebM) to MP4 for reliable OpenCV decoding."""
    if video_path.suffix.lower() in (".mp4", ".m4v"):
        return video_path

    mp4_path = video_path.with_suffix(".mp4")
    logger.info("Converting %s → %s via ffmpeg", video_path.name, mp4_path.name)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-an", str(mp4_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        logger.error("ffmpeg conversion failed: %s", result.stderr[-500:])
        raise RuntimeError(f"ffmpeg conversion failed (exit {result.returncode})")

    logger.info("Converted to MP4: %.1f MB", mp4_path.stat().st_size / 1e6)
    return mp4_path


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------
@celery_app.task(name="analyze_video", bind=True, max_retries=2)
def analyze_video(self, video_b64: str, ext: str = ".mp4") -> dict[str, Any]:
    """Full pipeline: decode base64 → temp file → MediaPipe → Gemini → JSON."""
    # Write base64 video to a temp file so OpenCV can read it
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp_path = Path(tmp.name)
    mp4_path: Path | None = None
    try:
        tmp.write(base64.b64decode(video_b64))
        tmp.close()
        logger.info("Wrote %.1f MB to %s", tmp_path.stat().st_size / 1e6, tmp_path.name)

        # Convert WebM/other formats to MP4 for reliable OpenCV decoding
        mp4_path = _ensure_mp4(tmp_path)

        # Step A – extract joint angles + raw landmarks
        logger.info("Step A: processing video %s", mp4_path.name)
        angle_text, raw_frames = _process_video(str(mp4_path))

        # Step B – buffer raw landmarks in Redis for frontend overlay
        task_id = self.request.id
        if task_id:
            redis_key = f"landmarks:{task_id}"
            try:
                _redis.set(redis_key, json.dumps(raw_frames), ex=LANDMARKS_TTL)
                logger.info(
                    "Buffered %d frames of landmarks in Redis (%s)",
                    len(raw_frames), redis_key,
                )
            except Exception as exc:
                logger.warning("Failed to buffer landmarks in Redis: %s", exc)

        # Step C – interpret with Gemini
        logger.info("Step C: sending %d chars of angle data to Gemini", len(angle_text))
        analysis = _interpret_with_gemini(angle_text)

        # Step D – resolve problem_joints to actual landmark names for frontend
        problem_joints = analysis.get("problem_joints", [])
        problem_landmarks: list[str] = []
        seen = set()
        for joint_name in problem_joints:
            for lm_name in ANGLE_TO_LANDMARKS.get(joint_name, []):
                if lm_name not in seen:
                    problem_landmarks.append(lm_name)
                    seen.add(lm_name)
        analysis["problem_landmarks"] = problem_landmarks
        logger.info("Problem joints: %s → landmarks: %s", problem_joints, problem_landmarks)

        return analysis

    except Exception as exc:
        logger.exception("analyze_video failed")
        raise self.retry(exc=exc, countdown=5)

    finally:
        tmp_path.unlink(missing_ok=True)
        if mp4_path and mp4_path != tmp_path:
            mp4_path.unlink(missing_ok=True)
        logger.info("Cleaned up temp files")

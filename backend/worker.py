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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

SYSTEM_PROMPT = (
""" You are the MediaPipe Biomechanical Analysis Engine (MBAE). Your specific purpose is to analyze kinematic data (joint coordinates and angles) combined with visual context to classify exercises, count repetitions, and score movement quality with clinical precision.

1. OPERATIONAL CONSTRAINTS
Output Format: You must output ONLY valid JSON. Do not include markdown formatting, conversational text, or explanations outside the JSON structure.

Input Handling: You will receive a time-series of joint angles with landmark coordinates, AND one reference image per rep.

HIERARCHY OF TRUTH:

Reference Images (PRIMARY): You must use the reference image as the absolute authority for Exercise Classification. Look for equipment context (bench, bar, dumbbell), load placement (bar on back vs. in hands), and body orientation to determine what is being performed.

Kinematic Data (SECONDARY): Use angle data only for:

Tiebreaking: If the image is ambiguous (e.g., distinguishing a Squat from a Deadlift in a cropped view), use joint coupling ratios.

Scoring: Once the exercise is classified via image, use angles to detect biomechanical faults (e.g., "funny angles," valgus, range of motion issues).

Landmark Coordinates: Use (x,y) coordinates for spatial analysis angles cannot capture (e.g., bar path, stance width).

Signal Processing: Apply logical smoothing: do not flag faults based on single-frame anomalies; require a fault to persist for >0.2 seconds.

2. PRE-ANALYSIS GATING
Set "analysis_allowed" to FALSE and provide a "rejection_reason" if:

Occlusion: Critical joints are invisible for >40% of the duration.

Confidence: Landmark visibility confidence is consistently < 0.5.

Bad Angle: The camera angle prevents valid assessment (e.g., extreme blur, camera facing ceiling).

3. EXERCISE CLASSIFICATION LOGIC (IMAGE PRIORITY)
Step 1: Visual Classification (Highest Priority) Analyze the reference image for these visual signatures:

Squat: Load placed on Upper Back or Front Rack (shoulders). Hip joint descends clearly below standing height.

Hinge (Deadlift): Load held in hands, starting from/returning to floor height. Torso inclined forward, shins relatively vertical compared to squat.

Bench Press: User is lying supine on a bench structure.

Overhead Press: User is standing/seated upright, load is above shoulder level or overhead.

Pull-Up: User is hanging from a bar (hands above head, feet off ground).

Row: Torso bent forward (unsupported or supported), load is being pulled toward the abdomen.

Step 2: Kinematic Tiebreaker (Only use if Step 1 is ambiguous) If the image does not clearly distinguish the movement (e.g., between Squat and Hinge), apply Dynamic Joint Coupling:

Squat Pattern: Knee Flexion / Hip Flexion ratio ≈ 1:1.

Hinge Pattern: Hip Flexion >> Knee Flexion (Hip dominant).

4. REPETITION COUNTING (FINITE STATE MACHINE)
Use a 3-State Logic with Hysteresis:

State A (Start): Joint angle at resting threshold.

State B (Inflection): Joint angle > Effort Threshold.

State C (Return): Transition B -> A. Constraint: Increment count if Prominence (ROM change) > 45° and duration > 0.4s.

5. FORM SCORING & FAULT THRESHOLDS
Strictly apply these biomechanical angle checks to the EXERCISE IDENTIFIED IN SECTION 3. Start at a Score of 10. Deduct points for biomechanical deviations ("funny angles").

A. SQUAT
Critical (-3.0): Valgus (Front view). Inter-Knee Dist / Inter-Ankle Dist < 0.8.

Major (-1.5): Insufficient Depth (Knee Flexion < 90° or Hip Y > Knee Y).

Minor (-0.5): Butt Wink (Lumbar angle posterior shift > 10°).

B. HINGE (DEADLIFT)
Critical (-3.0): Lumbar Rounding (Thoracic vector drops < 30° relative to floor).

Major (-1.5): "Squatting the Pull" (Knee angle < 100° at start).

Minor (-0.5): Soft Lockout (Hip extension < 170°).

C. LUNGE
Critical (-2.5): Knee Valgus. Lead Knee X deviates > 3cm medially.

Minor (-1.0): Short Step.

D. PRESS (OVERHEAD / BENCH)
Major (-1.5): Elbow Flare (Bench). Elbow-Torso angle > 80°.

Major (-1.5): Lumbar Hyperextension (Overhead). Torso-Leg angle < 165°.

Minor (-0.5): Incomplete ROM. Elbow extension < 170°.

E. PULL (ROW / PULL-UP)
Major (-1.5): Momentum/Kipping. Hip X variance > 20cm.

Major (-1.0): Incomplete ROM.

F. ISOLATION (CURL / TRICEP)
Minor (-1.0): Elbow Drift (Humerus vector is unstable).

6. REQUIRED OUTPUT SCHEMA
You must strictly adhere to this JSON structure:

{ "type": "object", "properties": { "analysis_allowed": { "type": "boolean", "description": "True if landmarks are visible and camera angle allows valid assessment." }, "rejection_reason": { "type": "string", "description": "If analysis_allowed is false, state why. Otherwise empty string." }, "exercise_detected": { "type": "string", "enum": ["Squat", "Deadlift", "Lunge", "Overhead Press", "Bench Press", "Pull-Up", "Row", "Bicep Curl", "Tricep Extension", "Unknown"], "description": "Identified primarily via image context." }, "rep_count": { "type": "integer", "description": "Total completed reps passing the FSM check." }, "form_rating_1_to_10": { "type": "integer", "description": "Holistic score (1-10). Start at 10, subtract deductions." }, "main_mistakes": { "type": "array", "items": {"type": "string"}, "description": "List of distinct faults detected via biomechanical angle analysis." }, "rep_analyses": { "type": "array", "items": { "type": "object", "properties": { "rep_number": {"type": "integer"}, "timestamp_start": {"type": "number"}, "timestamp_end": {"type": "number"}, "rating_1_to_10": {"type": "integer"}, "mistakes": { "type": "array", "items": {"type": "string"} }, "problem_joints": { "type": "array", "items": {"type": "string"} } }, "required": ["rep_number", "timestamp_start", "timestamp_end", "rating_1_to_10", "mistakes", "problem_joints"] } }, "problem_joints": { "type": "array", "items": {"type": "string"}, "description": "Aggregate list of joints where angles deviated from the ideal model." }, "visual_description": { "type": "string", "description": "If reference images were provided, a 2-3 sentence description of what is visually observed across the images: person's body position, equipment, environment, stance, and any visible form issues. Empty string if no images." }, "actionable_correction": { "type": "string", "description": "A single, high-impact coaching cue based on the most frequent fault." } }, "required": [ "analysis_allowed", "rejection_reason", "exercise_detected", "rep_count", "form_rating_1_to_10", "main_mistakes", "rep_analyses", "problem_joints", "visual_description", "actionable_correction" ] } """
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
                    "problem_joints": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["rep_number", "timestamp_start", "timestamp_end", "rating_1_to_10", "mistakes", "problem_joints"],
            },
        },
        "problem_joints": {
            "type": "array",
            "items": {"type": "string"},
        },
        "visual_description": {"type": "string"},
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
        "visual_description",
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

# Key landmarks for coordinate output (abbreviated for compactness in prompt)
_COORD_LANDMARKS: list[tuple[int, str]] = [
    (PoseLandmark.LEFT_SHOULDER, "LS"), (PoseLandmark.RIGHT_SHOULDER, "RS"),
    (PoseLandmark.LEFT_ELBOW, "LE"),    (PoseLandmark.RIGHT_ELBOW, "RE"),
    (PoseLandmark.LEFT_WRIST, "LW"),    (PoseLandmark.RIGHT_WRIST, "RW"),
    (PoseLandmark.LEFT_HIP, "LH"),      (PoseLandmark.RIGHT_HIP, "RH"),
    (PoseLandmark.LEFT_KNEE, "LK"),     (PoseLandmark.RIGHT_KNEE, "RK"),
    (PoseLandmark.LEFT_ANKLE, "LA"),    (PoseLandmark.RIGHT_ANKLE, "RA"),
]


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
        Compact time-series of joint angles + landmark coordinates for the LLM.
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

                # ── Angles + coordinates → LLM (sampled frames only)
                if frame_idx % angle_every_n == 0:
                    angles = _compute_angles(result.pose_landmarks)
                    angle_parts = [f"{k}={v}" for k, v in angles.items()]
                    # Key landmark (x,y) coordinates
                    lm_raw = result.pose_landmarks.landmark
                    coord_parts = [
                        f"{sn}({lm_raw[idx].x:.3f},{lm_raw[idx].y:.3f})"
                        for idx, sn in _COORD_LANDMARKS
                    ]
                    angle_lines.append(
                        f"t={timestamp_s:.3f} "
                        + " ".join(angle_parts)
                        + " | "
                        + " ".join(coord_parts)
                    )

            frame_idx += 1
    finally:
        cap.release()
        pose.close()

    if not angle_lines:
        raise RuntimeError("MediaPipe could not detect any pose in the video.")

    # Prepend header explaining the data format
    header = (
        "# Format: t=<seconds> <joint_angles_degrees> | <landmark_xy_coords>\n"
        "# Landmarks: LS=LEFT_SHOULDER RS=RIGHT_SHOULDER LE=LEFT_ELBOW RE=RIGHT_ELBOW "
        "LW=LEFT_WRIST RW=RIGHT_WRIST LH=LEFT_HIP RH=RIGHT_HIP "
        "LK=LEFT_KNEE RK=RIGHT_KNEE LA=LEFT_ANKLE RA=RIGHT_ANKLE"
    )

    logger.info(
        "Processed %d frames → %d landmark frames, %d angle samples",
        frame_idx, len(raw_frames), len(angle_lines),
    )
    return header + "\n" + "\n".join(angle_lines), raw_frames


def _extract_rep_midframes(
    video_path: str,
    rep_analyses: list[dict],
) -> list[tuple[float, bytes]]:
    """For each rep, extract the video frame at the temporal midpoint.

    Returns list of (mid_timestamp_s, jpeg_bytes) tuples — one per rep.
    """
    if not rep_analyses:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot re-open video for mid-rep frame extraction")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    results: list[tuple[float, bytes]] = []
    try:
        for rep in sorted(rep_analyses, key=lambda r: r.get("rep_number", 0)):
            t_start = rep.get("timestamp_start", 0)
            t_end = rep.get("timestamp_end", 0)
            mid_t = (t_start + t_end) / 2.0
            frame_idx = round(mid_t * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, bgr = cap.read()
            if ret:
                ok, buf = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
                if ok:
                    results.append((round(mid_t, 3), buf.tobytes()))
    finally:
        cap.release()

    logger.info(
        "Extracted %d mid-rep JPEGs (%.1f KB total)",
        len(results), sum(len(b) for _, b in results) / 1024,
    )
    return results


# ---------------------------------------------------------------------------
# Gemini helpers (two-pass architecture)
# ---------------------------------------------------------------------------
_FALLBACK_RESULT: dict[str, Any] = {
    "analysis_allowed": False,
    "rejection_reason": "Could not parse analysis output.",
    "exercise_detected": "unknown",
    "rep_count": 0,
    "form_rating_1_to_10": 0,
    "main_mistakes": ["Analysis could not be parsed."],
    "rep_analyses": [],
    "actionable_correction": "",
}


def _gemini_call(
    contents: str | list,
    tag: str = "",
) -> dict[str, Any]:
    """Low-level Gemini call with structured JSON parsing. Shared by both passes."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=8192,
                response_mime_type="application/json",
                response_schema=RESPONSE_JSON_SCHEMA,
            ),
        )
    except Exception as exc:
        logger.warning(
            "[%s] Structured schema not supported; falling back to JSON mode: %s",
            tag, exc,
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config={
                "temperature": 0.2,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            },
        )

    if getattr(response, "parsed", None):
        return response.parsed

    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

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

        logger.error("[%s] Gemini returned non-JSON: %s", tag, raw)
        return {**_FALLBACK_RESULT, "actionable_correction": raw}


def _gemini_pass1(angle_text: str) -> dict[str, Any]:
    """Pass 1 — text-only.  Send angle + coordinate data to get rep
    timestamps, exercise classification, scoring, and per-rep analysis."""
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Here are the joint angles (degrees) and landmark coordinates "
        "extracted from a workout video:\n\n"
        f"{angle_text}"
    )
    logger.info("[Pass 1] Sending %d chars of angle data (text-only)", len(angle_text))
    return _gemini_call(prompt, tag="Pass 1")


def _gemini_pass2(
    angle_text: str,
    rep_jpegs: list[tuple[float, bytes]],
    pass1_result: dict[str, Any],
) -> dict[str, Any]:
    """Pass 2 — multimodal.  Re-send angles + per-rep mid-rep images
    so Gemini can visually verify and refine the analysis from Pass 1."""
    if not rep_jpegs:
        return pass1_result

    # Build a short summary of Pass 1 to give Gemini context
    pass1_summary = (
        f"Previous text-only analysis detected: {pass1_result.get('exercise_detected', 'Unknown')}, "
        f"{pass1_result.get('rep_count', 0)} reps, "
        f"score {pass1_result.get('form_rating_1_to_10', 0)}/10.\n"
        f"Main mistakes: {', '.join(pass1_result.get('main_mistakes', [])) or 'none'}.\n\n"
    )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "You previously analyzed this data using angles only. Now you also have "
        f"{len(rep_jpegs)} mid-rep reference images (one per rep, captured at "
        "the temporal midpoint of each repetition).\n\n"
        "INSTRUCTIONS FOR IMAGE ANALYSIS:\n"
        "1. First, describe what you see in the images in the \"visual_description\" "
        "field: the person's body position, equipment visible (barbell, dumbbells, "
        "bench, rack, etc.), environment, stance width, grip type, and any visible "
        "form issues.\n"
        "2. Use this visual description as confluence evidence alongside the angle "
        "data to confirm or CORRECT the exercise classification. For example, if "
        "angles suggest a squat but the image shows a barbell on the floor with a "
        "hip hinge, reclassify as deadlift.\n"
        "3. Visually verify per-rep form faults: depth, alignment, posture, "
        "knee tracking, back position.\n"
        "4. Refine scores and mistakes based on combined angle + visual evidence.\n\n"
        f"{pass1_summary}"
        "Here are the joint angles (degrees) and landmark coordinates "
        "extracted from a workout video:\n\n"
        f"{angle_text}"
    )

    contents: list = []
    for i, (_ts, jpeg) in enumerate(rep_jpegs):
        contents.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
    contents.append(prompt)

    logger.info(
        "[Pass 2] Sending multimodal request (text + %d rep images, %.1f KB)",
        len(rep_jpegs), sum(len(b) for _, b in rep_jpegs) / 1024,
    )
    return _gemini_call(contents, tag="Pass 2")


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

        # Step C – Pass 1: text-only Gemini call to get rep timestamps
        pass1 = _gemini_pass1(angle_text)

        # Step D – extract mid-rep frames using Pass 1's rep timestamps
        rep_analyses = pass1.get("rep_analyses", [])
        rep_jpegs = _extract_rep_midframes(str(mp4_path), rep_analyses)
        logger.info(
            "Extracted %d mid-rep frames for %d reps",
            len(rep_jpegs), len(rep_analyses),
        )

        # Step E – Pass 2: multimodal Gemini call with mid-rep images
        analysis = _gemini_pass2(angle_text, rep_jpegs, pass1)
        analysis["rep_frame_timestamps"] = [t for t, _ in rep_jpegs]

        # Buffer per-rep frame JPEGs in Redis for frontend display
        if task_id and rep_jpegs:
            try:
                rep_payload = [
                    {"t": t, "b64": base64.b64encode(jpg).decode()}
                    for t, jpg in rep_jpegs
                ]
                _redis.set(
                    f"rep_frames:{task_id}",
                    json.dumps(rep_payload),
                    ex=LANDMARKS_TTL,
                )
            except Exception as exc:
                logger.warning("Failed to buffer rep frames in Redis: %s", exc)

        # Step F – resolve per-rep problem_joints to landmark ranges for frontend
        problem_landmark_ranges: list[dict] = []
        global_landmarks_seen: set[str] = set()
        global_problem_landmarks: list[str] = []

        for rep in analysis.get("rep_analyses", []):
            rep_joints = rep.get("problem_joints", [])
            if not rep_joints:
                continue
            rep_landmarks: list[str] = []
            seen: set[str] = set()
            for joint_name in rep_joints:
                for lm_name in ANGLE_TO_LANDMARKS.get(joint_name, []):
                    if lm_name not in seen:
                        rep_landmarks.append(lm_name)
                        seen.add(lm_name)
                    if lm_name not in global_landmarks_seen:
                        global_problem_landmarks.append(lm_name)
                        global_landmarks_seen.add(lm_name)
            if rep_landmarks:
                problem_landmark_ranges.append({
                    "start": rep.get("timestamp_start", 0),
                    "end": rep.get("timestamp_end", 0),
                    "landmarks": rep_landmarks,
                })

        analysis["problem_landmark_ranges"] = problem_landmark_ranges
        analysis["problem_landmarks"] = global_problem_landmarks
        logger.info(
            "Problem landmark ranges: %d ranges, global landmarks: %s",
            len(problem_landmark_ranges), global_problem_landmarks,
        )

        return analysis

    except Exception as exc:
        logger.exception("analyze_video failed")
        raise self.retry(exc=exc, countdown=5)

    finally:
        tmp_path.unlink(missing_ok=True)
        if mp4_path and mp4_path != tmp_path:
            mp4_path.unlink(missing_ok=True)
        logger.info("Cleaned up temp files")

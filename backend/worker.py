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
from collections import deque
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
""" You are the Biomechanical Auditor. Your ONLY purpose is to analyze movement quality within pre-defined temporal windows. Do NOT count repetitions; the timestamps provided are absolute.

1. INDEPENDENT UNIT AUDITING (i.i.d.) & FATIGUE HEURISTIC
- Treat every provided rep window as an independent data point. 
- Fatigue Heuristic: Biomechanical integrity (Rep N+1) is statistically likely to be worse than Rep N. Look for 'Micro-Struggles' (jitter, slower ascent) compared to the baseline set by Rep 1.

2. HIERARCHY OF TRUTH (THE "SAGGING" FIX)
- Image Authority: If kinematic math suggests a fault but the visual reference shows a rigid, straight plank, prioritize the image. Dismiss the math as sensor jitter.
- Rigid Cylinder Rule: In a Push-Up, if the Hip Y and Shoulder Y move in a 1:1 ratio, the core is stable.

3. ENERGY LEAK ANALYSIS
A 'Rep' is a transfer of energy through a kinetic chain. You must identify 'Energy Leaks'—movements where tension is lost and force is diverted to non-target joints:
- Segmental Lag: If one body segment (e.g., the hips) moves significantly after another (e.g., the shoulders) on the ascent, this is a 'Moderate' energy leak.
- Structural Collapse: If a joint fails to maintain its position relative to the rigid cylinder (e.g., hips dropping to the floor), this is a 'Severe' energy leak.
- Tension Loss: Identify jitter or 'shaking' as high-frequency energy leaks indicative of neurological fatigue.
- Lateral Integrity Rule: Evaluate the $X$-axis (side-to-side) and $Z$-axis (depth) synchronicity. 
- If the left and right sides of the 'Rigid Cylinder' do not move in a 1:1 ratio, classify it as an Asymmetrical Energy Leak. 
- Use adjectives like 'Lopsided' or 'Unbalanced' to trigger semantic penalties.

4. ADJECTIVE-BASED PENALIZATION (SMART SCORING)
Every rep starts at 10.0. The deduction for a fault is scaled by the 'Connotation' of the adjective you choose:
If you find no issues, you can just give the rep a 10, no need to be a teacher who doesn't give out 10s just for the sake of not giving out 10s.
- 'Negligible' / 'Trace' (0.1x - 0.2x): Slight aesthetic deviation; no functional energy leak.
- 'Noticeable' / 'Minor' (0.3x - 0.5x): Clear energy leak; requires corrective cueing.
- 'Significant' / 'Pronounced' (0.6x - 0.8x): Major structural leak; energy is no longer transferring efficiently.
- 'Severe' / 'Critical' (0.9x - 1.0x): Structural failure; high injury risk and total loss of tension.

5. HEURISTIC-DRIVEN FAULT ANALYSIS
- Push-Up: Analyze 'Synchronicity.' If hips 'lag' or 'drift' out of sync with shoulders, it is a stability failure/energy leak.
- Squat/Deadlift: Analyze 'Pivot Stability.' Penalize based on the 'Rate of Curvature Change' in the spine.

6. REQUIRED OUTPUT SCHEMA
(Ensure your JSON adheres to the provided RESPONSE_JSON_SCHEMA, using 'i_i_d_audit_notes' to specifically describe the location and severity of any Energy Leaks.)
""" 
)

RESPONSE_JSON_SCHEMA = {
  "type": "object",
  "properties": {
    "analysis_allowed": { "type": "boolean" },
    "rejection_reason": { "type": "string" },
    "internal_signal_analysis": {
      "type": "string",
      "description": "Internal monologue reconciling image vs math, mass synchronicity, and application of the fatigue heuristic."
    },
    "exercise_detected": {
      "type": "string",
      "enum": ["Push-Up", "Squat", "Deadlift", "Unknown"]
    },
    "rep_count": { "type": "integer" },
    "form_rating_1_to_10": { "type": "number" },
    "main_mistakes": {
      "type": "array",
      "items": { "type": "string" }
    },
    "rep_analyses": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "rep_number": { "type": "integer" },
          "timestamp_start": { "type": "number" },
          "timestamp_end": { "type": "number" },
          "rating_1_to_10": { "type": "number" },
          "synchronicity_score": { 
            "type": "number",
            "description": "0.0 to 1.0 scale of how well the body moved as a synchronized rigid unit."
          },
          "primary_fault_category": {
            "type": "string",
            "enum": ["Stability", "ROM", "Alignment", "None"]
          },
          "severity": {
            "type": "string",
            "enum": ["Trace", "Moderate", "Severe", "None"]
          },
          "mistakes": {
            "type": "array",
            "items": { "type": "string" }
          },
          "problem_joints": {
            "type": "array",
            "items": { "type": "string" }
          },
          "i_i_d_audit_notes": {
            "type": "string",
            "description": "Unique biomechanical observation for this specific isolated rep regarding energy leaks or structural risk."
          }
        },
        "required": [
          "rep_number",
          "timestamp_start",
          "timestamp_end",
          "rating_1_to_10",
          "synchronicity_score",
          "primary_fault_category",
          "severity",
          "mistakes",
          "problem_joints",
          "i_i_d_audit_notes"
        ]
      }
    },
    "problem_joints": {
      "type": "array",
      "items": { "type": "string" }
    },
    "visual_description": { "type": "string" },
    "actionable_correction": { "type": "string" }
  },
  "required": [
    "analysis_allowed",
    "rejection_reason",
    "internal_signal_analysis",
    "exercise_detected",
    "rep_count",
    "form_rating_1_to_10",
    "main_mistakes",
    "rep_analyses",
    "problem_joints",
    "visual_description",
    "actionable_correction"
  ]
}

# Pass 1 schema — minimal, rep detection only (no classification or scoring)
_PASS1_SCHEMA = {
    "type": "object",
    "properties": {
        "rep_count": {"type": "integer"},
        "rep_analyses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rep_number": {"type": "integer"},
                    "timestamp_start": {"type": "number"},
                    "timestamp_end": {"type": "number"},
                },
                "required": ["rep_number", "timestamp_start", "timestamp_end"],
            },
        },
    },
    "required": ["rep_count", "rep_analyses"],
}

_PASS1_PROMPT = (
    """You are a Pattern Recognition Engine. Your task is to identify exercise 
repetitions from biomechanical time-series data. 

1. IDENTIFY THE ANCHOR SIGNAL:
   Locate the joint or coordinate (e.g., Elbow Flexion or Hip Y) that 
   exhibits the clearest rhythmic trend.

2. REP DETECTION LOGIC (OPEN-LOOP):
   A repetition is triggered the moment the body transitions from a 
   'Stable' state into a clear 'Action' phase.
   - Start: The moment a sustained directional trend begins.
   - Inflection: The moment the movement reaches its maximum depth 
     (velocity hits zero or reverses).
   - Validation: Once the Inflection point is reached, count the rep. 

3. EDGE CASE HANDLING (VIDEO CUTOFFS):
   Do NOT wait for the signal to return to the original 'Stable' baseline 
   to count the rep. If the data shows a clear descent and reaches an 
   inflection point, it is a counted rep, even if the data ends 
   immediately after the turn.

4. COHESION FILTER:
   Ignore fluctuations that only affect a single joint. A 'Smart' rep 
   requires 'Cohesion'—the primary joint movement must be accompanied 
   by a shift in the torso's global position (Shoulder/Hip Y).

OUTPUT ONLY JSON: 
{ 'rep_count': int, 'rep_analyses': [{'rep_number': int, 'timestamp_start': float, 'timestamp_end': float}] }"""
)

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
    """Compute all defined joint angles from a single frame's landmarks using a median filter."""
    lm = landmarks.landmark
    angles: dict[str, float] = {}
    for name, (a_idx, v_idx, b_idx) in ANGLE_DEFS.items():
        current_angle = _angle_between(lm[a_idx], lm[v_idx], lm[b_idx])

        # Maintain a history of recent angles for this joint
        if name not in _angle_history:
            _angle_history[name] = deque(maxlen=MEDIAN_WINDOW_SIZE)
        _angle_history[name].append(current_angle)

        # Compute the median of the recent angles
        angles[name] = round(sorted(_angle_history[name])[len(_angle_history[name]) // 2], 1)

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
    schema: dict | None = None,
) -> dict[str, Any]:
    """Low-level Gemini call with structured JSON parsing. Shared by both passes."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    effective_schema = schema or RESPONSE_JSON_SCHEMA
    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=8192,
                response_mime_type="application/json",
                response_schema=effective_schema,
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
    """Pass 1 — text-only.  Detect rep boundaries ONLY (no classification
    or scoring).  Returns {rep_count, rep_analyses[{rep_number, timestamp_start,
    timestamp_end}]}."""
    prompt = (
        f"{_PASS1_PROMPT}\n\n"
        "Here are the joint angles (degrees) and landmark coordinates "
        "extracted from a workout video:\n\n"
        f"{angle_text}"
    )
    logger.info("[Pass 1] Sending %d chars of angle data (rep detection only)", len(angle_text))
    return _gemini_call(prompt, tag="Pass 1", schema=_PASS1_SCHEMA)


def _gemini_pass2(
    angle_text: str,
    rep_jpegs: list[tuple[float, bytes]],
    pass1_rep_analyses: list[dict],
) -> dict[str, Any]:
    """Pass 2 — PRIMARY analysis.  Classify exercise from images, score
    form from angles, detect faults.  Uses rep timestamps from Pass 1 as
    structural context only."""

    # Build rep-boundary context from Pass 1
    rep_context = ""
    if pass1_rep_analyses:
        rep_lines = []
        for r in pass1_rep_analyses:
            rep_lines.append(
                f"  Rep {r.get('rep_number', '?')}: "
                f"{r.get('timestamp_start', 0):.2f}s – {r.get('timestamp_end', 0):.2f}s"
            )
        rep_context = (
            "Rep boundaries detected from angle analysis:\n"
            + "\n".join(rep_lines) + "\n\n"
        )

    if rep_jpegs:
        # ── Multimodal: images + angles (primary path) ──
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"You have {len(rep_jpegs)} mid-rep reference images (one per rep, "
            "captured at the temporal midpoint of each repetition).\n\n"
            "CRITICAL — EXERCISE CLASSIFICATION:\n"
            "Your #1 job is to look at the images and determine the exercise.\n"
            "Look for: equipment (barbell, dumbbells, bench, rack, cables, "
            "pull-up bar), body orientation (standing, lying, bent over), "
            "load placement (bar on back, bar in hands, overhead), and movement "
            "plane.  The images are the GROUND TRUTH for classification.\n\n"
            "INSTRUCTIONS:\n"
            "1. Describe what you see in the images in the \"visual_description\" "
            "field: body position, equipment, environment, stance, grip, and any "
            "visible form issues.\n"
            "2. Classify the exercise based PRIMARILY on what you SEE.\n"
            "3. Use the rep boundaries below to structure your per-rep analysis.\n"
            "4. Use the angle data to score form quality and detect biomechanical "
            "faults for each rep.\n\n"
            f"{rep_context}"
            "Joint angles (degrees) and landmark coordinates:\n\n"
            f"{angle_text}"
        )

        contents: list = []
        for _ts, jpeg in rep_jpegs:
            contents.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
        contents.append(prompt)

        logger.info(
            "[Pass 2] Multimodal primary analysis: text + %d images (%.1f KB)",
            len(rep_jpegs), sum(len(b) for _, b in rep_jpegs) / 1024,
        )
    else:
        # ── Text-only fallback (no rep images available) ──
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{rep_context}"
            "Here are the joint angles (degrees) and landmark coordinates "
            "extracted from a workout video:\n\n"
            f"{angle_text}"
        )
        contents = prompt
        logger.info("[Pass 2] Text-only fallback (%d chars)", len(prompt))

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

        # Step E – Pass 2: primary analysis (classification + scoring + faults)
        analysis = _gemini_pass2(angle_text, rep_jpegs, rep_analyses)
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

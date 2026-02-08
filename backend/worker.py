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
""" You are the MediaPipe Biomechanical Analysis Engine (MBAE). Your purpose is to classify three movements (Push-Up, Deadlift, Squat), count repetitions, and score movement quality with clinical precision.

DATA FORMAT YOU WILL RECEIVE
────────────────────────────
1. Joint Angles (degrees): L_knee, R_knee, L_hip, R_hip, L_elbow, R_elbow, L_shoulder, R_shoulder, L_ankle, R_ankle, spine
2. Landmark Coordinates (2D x,y and 3D x,y,z normalized): 3 frames per rep (start, mid, end) with LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA
3. Reference Images: Three JPEGs per rep (start, midpoint, end positions)

1. OPERATIONAL CONSTRAINTS
Output Format: You must output ONLY valid JSON. Do not include markdown formatting, conversational text, or explanations outside the JSON structure.

Strict Scope: Classify ONLY Push-Up, Deadlift, or Squat. Return "Unknown" ONLY if the movement pattern or equipment clearly indicates a different exercise.

Leniency Rule: If intent matches one of the three but form is flawed, classify as that movement and list corrections. Never change classification based on form quality.

HIERARCHY OF TRUTH:
- Reference Images (PRIMARY): Determine exercise class via visual context (equipment, body orientation, load placement).
- Angle Time-Series (SECONDARY): Quantify form quality, detect faults, validate classification.
- Landmark Coordinates (TERTIARY): Assess spatial metrics (stance width, bar path, body alignment) that angles cannot capture.

Signal Processing: Apply smoothing; ignore single-frame anomalies (<0.2s). Require faults to persist ≥0.2s or appear in ≥2 adjacent frames.

2. PRE-ANALYSIS GATING
Set "analysis_allowed" = FALSE if:
- Critical joints invisible for >40% of rep duration
- Landmark confidence consistently <0.5
- Camera angle prevents assessment (extreme tilt, blur, obstructions)

3. EXERCISE CLASSIFICATION (IMAGE PRIMARY)
Classify based on visual context first; use angles to break ties only.

Push-Up:
  - Horizontal body near floor, hands planted at shoulder level
  - Feet on floor, elbows flex/extend, torso stable
  - No barbell/dumbbells
  - Angle signature: Elbow flexion drives motion; hip angle stable; spine erect

Squat:
  - Load on back (barbell), front rack (kettlebell/dumbbell), or bodyweight
  - Feet planted shoulder-width, hips descend below standing height
  - Knees and hips flex together, moderate forward lean
  - Angle signature: Knee and hip flex in tandem (~1:1 ratio at deep position); knee angle reaches 80–110° at bottom

Deadlift:
  - Load held in hands (barbell/dumbbells), starts near floor or mid-shin
  - Hips higher than squat at start; torso hinged forward
  - Lifting via hip extension first, knees extend later
  - Angle signature: Hip extension dominates; knee angle stays 100–130° at start; large change in spine angle during pull

4. BODY PROPORTION ESTIMATION
─────────────────────────────
From the images, estimate lifter's relative build:
- If tall (appears >6'): Squat depth expectations are deeper (knee 70–95°)
- If short (<5'8"): Squat depth expectations are shallower (knee 85–105°)
- Apply ±5° variance to depth/lean/angle thresholds based on proportions

Example: "Person appears 6'2" (tall build). Knee angle 92° at depth is acceptable for their proportions."

5. VISUAL-ANGLE CONVERGENCE RULE (CRITICAL)
────────────────────────────────────────────
To flag a fault, you must have BOTH visual AND angle evidence:
✓ VALID: Image shows knees caving inward AND inter-knee/ankle ratio < 0.75
✗ INVALID: Angles show ratio 0.72, but image clearly shows vertical knee tracking

If visual contradicts angles:
  1. Describe what you see in the image
  2. Note the angle reading
  3. Apply the VISUAL assessment (images override angles for safety)
  4. Explain why (likely camera artifact, body proportions, etc.)

Deduct points ONLY when visual + angles BOTH confirm the fault (confidence ≥ 0.75).

6. REP PHASE ANALYSIS
─────────────────────
Assess movement quality in distinct phases:

SQUAT:
  Phase 1 (Descent, 0–40%): Assess knee-hip coupling, torso stability, even weight distribution
  Phase 2 (Bottom, 40–60%): Assess depth, knee valgus, core stability, hip control
  Phase 3 (Ascent, 60–100%): Assess drive symmetry, lockout quality, no asymmetric weight shift

DEADLIFT:
  Phase 1 (Positioning, 0–20%): Hips at correct height, spine angle 25–35°, neutral lumbar
  Phase 2 (Pull, 20–80%): Bar path close to body, proper hip/knee sequencing, lumbar stays neutral
  Phase 3 (Lockout, 80–100%): Complete hip extension (hips fully forward), vertical torso, no shoulder shrug

PUSH-UP:
  Phase 1 (Descent, 0–50%): Elbows track parallel, core stable (no sag), spine neutral
  Phase 2 (Bottom, 50–50%): Full depth (elbows 80–90°), chest near floor, hips level
  Phase 3 (Ascent, 50–100%): Even elbow drive, complete lockout, no shoulder shrug

Score each phase separately (0–3.3/10 each), then average for rep rating.

7. SIMPLIFIED REP QUALITY RUBRIC (5 CORE FAULTS PER EXERCISE)
══════════════════════════════════════════════════════════════
Start each rep at 10/10. Deduct points ONLY when visual + angles both confirm:

A. SQUAT FAULTS
  1. Insufficient Depth (−3.0): Hips clearly above knees in images; Min(L_knee, R_knee) > 95°
  2. Knee Valgus (−3.0): Knees visibly cave inward; inter-knee/ankle ratio < 0.75
  3. Excessive Forward Lean (−2.0): Torso >50° from vertical in bottom position; spine angle > 55°
  4. Loss of Core Stability (−2.0): Hips shift side-to-side visibly; lateral hip drift > 0.1 units
  5. Asymmetry (−1.5): One side visibly weaker in images; |L_knee − R_knee| > 15° during descent

B. DEADLIFT FAULTS
  1. Lumbar Rounding (−3.0): Lower back visibly curved in images; spine angle < 25° during pull
  2. Hips Too Low at Start (−2.5): Resembles squat position; (L_knee + R_knee)/2 < 95° at start
  3. Incomplete Lockout (−2.0): Not fully upright at top; final hip extension < 160° or torso not vertical
  4. Bar Path Drift (−2.0): Bar visibly moves away from body; horizontal distance from mid-hip > 0.08
  5. Asymmetric Pull (−1.5): One side clearly stronger in images; |L_knee − R_knee| > 12° during pull

C. PUSH-UP FAULTS
  1. Sagging Core (−3.0): Hips visibly drop below shoulder-ankle line; hip angle deviation > 25°
  2. Incomplete Depth (−3.0): Elbows don't bend sufficiently; Min(L_elbow, R_elbow) < 75°
  3. Extreme Elbow Flare (−2.0): Elbows way out to sides in images; L_shoulder or R_shoulder > 80°
  4. Head Position (−1.5): Head visibly cranks forward; upper spine angle change > 12° from start to bottom
  5. Incomplete Lockout (−1.5): Elbows not straight at top; final (L_elbow + R_elbow)/2 > 20°

8. CONFIDENCE-BASED SCORING
──────────────────────────
Only apply deduction if confidence ≥ 0.75 (75% certain).
For each fault, output:
  - fault: name of the fault
  - confidence: 0–1 (how certain you are)
  - visual_evidence: what you see in the image ("knees caving inward visually", etc.)
  - angle_evidence: which angles support this ("inter-knee ratio 0.72", etc.)
  - deduction: only apply if confidence ≥ 0.75

If confidence < 0.75, include the fault for user awareness but do NOT deduct points.

9. BASELINE QUALITY ASSESSMENT
──────────────────────────────
After scoring Rep 1:
  - If Rep 1 ≥ 8/10: "Rep 1 shows excellent form. This is the standard for this session."
  - If Rep 1 < 6/10: "Form breaks down from the start. Focus initial position corrections."
  - Use this context to interpret fatigue or technical breakdown in later reps.

10. REQUIRED OUTPUT SCHEMA
───────────────────────────
{ 
  "analysis_allowed": boolean, 
  "rejection_reason": string,
  "exercise_detected": enum["Push-Up", "Squat", "Deadlift", "Unknown"],
  "rep_count": integer,
  "form_rating_1_to_10": integer (1–10 scale; average of all rep ratings),
  "main_mistakes": array[string] (list of distinct fault types detected across all reps),
  "rep_analyses": array[
    {
      "rep_number": integer,
      "timestamp_start": number (seconds),
      "timestamp_end": number (seconds),
      "rating_1_to_10": integer,
      "mistakes": array[object] (faults in THIS rep only, each with confidence, evidence, deduction),
      "problem_joints": array[string] (which joints had angle deviations)
    }
  ],
  "problem_joints": array[string] (aggregate across all reps),
  "visual_description": string (2–3 sentences describing what you see in images),
  "actionable_correction": string (single, highest-impact coaching cue)
}

"""
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
                        "items": {
                            "type": "object",
                            "properties": {
                                "fault": {"type": "string"},
                                "confidence": {"type": "number"},
                                "visual_evidence": {"type": "string"},
                                "angle_evidence": {"type": "string"},
                                "deduction": {"type": "number"},
                            },
                            "required": ["fault", "confidence", "visual_evidence", "angle_evidence", "deduction"],
                        },
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
    "You are a repetition-detection engine. Your ONLY task is to identify "
    "individual exercise repetitions from joint-angle time-series data.\n\n"
    "Use a 3-State Finite State Machine with Hysteresis:\n"
    "- State A (Start): Joint angle at resting threshold.\n"
    "- State B (Inflection): Joint angle crosses effort threshold "
    "(significant ROM change in the primary mover joints — knees, hips, "
    "elbows, or shoulders).\n"
    "- State C (Return): Transition B → A. Count a rep if ROM change > 45° "
    "and duration > 0.4s.\n\n"
    "Apply logical smoothing: ignore single-frame anomalies (<0.2s).\n\n"
    "Output ONLY:\n"
    '- \"rep_count\": total completed reps\n'
    '- \"rep_analyses\": array of {\"rep_number\", \"timestamp_start\", '
    '\"timestamp_end\"} for each rep\n\n'
    "Do NOT classify the exercise. Do NOT score form. ONLY detect rep "
    "boundaries and timestamps."
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

# Set of coordinate landmark names for filtering
_COORD_LANDMARK_NAMES: set[str] = {name for _, name in _COORD_LANDMARKS}


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


def _extract_rep_frames_with_landmarks(
    video_path: str,
    rep_analyses: list[dict],
    raw_frames: list[dict],
) -> list[dict]:
    """Extract 3 frames (start, mid, end) per rep with 2D and 3D landmarks.

    Returns list of dicts:
    [
      {
        "rep_number": int,
        "frames": {
          "start": {"t": float, "jpeg": bytes, "landmarks_2d": dict, "landmarks_3d": dict},
          "mid": {...},
          "end": {...}
        }
      }
    ]
    """
    if not rep_analyses:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot re-open video for rep frame extraction")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    results: list[dict] = []

    try:
        for rep in sorted(rep_analyses, key=lambda r: r.get("rep_number", 0)):
            t_start = rep.get("timestamp_start", 0)
            t_end = rep.get("timestamp_end", 0)
            rep_num = rep.get("rep_number", 0)

            frames_data = {"rep_number": rep_num, "frames": {}}

            # Extract 3 frames: start, mid, end
            for frame_type, t in [("start", t_start), ("mid", (t_start + t_end) / 2), ("end", t_end)]:
                frame_idx = round(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, bgr = cap.read()

                if ret:
                    # Extract JPEG
                    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        # Find closest landmark frame in raw_frames
                        closest_frame = min(raw_frames, key=lambda f: abs(f["time_s"] - t))
                        lm_data = closest_frame.get("landmarks", [])

                        # Build 2D and 3D landmark dicts (only key landmarks)
                        landmarks_2d = {}
                        landmarks_3d = {}
                        for lm in lm_data:
                            name = lm.get("name", "")
                            if name in _COORD_LANDMARK_NAMES:
                                landmarks_2d[name] = [round(lm["x"], 4), round(lm["y"], 4)]
                                landmarks_3d[name] = [round(lm["x"], 4), round(lm["y"], 4), round(lm["z"], 4)]

                        frames_data["frames"][frame_type] = {
                            "t": round(t, 3),
                            "jpeg": buf.tobytes(),
                            "landmarks_2d": landmarks_2d,
                            "landmarks_3d": landmarks_3d,
                        }

            if frames_data["frames"]:
                results.append(frames_data)
    finally:
        cap.release()

    logger.info(
        "Extracted %d reps with 3 frames each (start, mid, end); total %.1f KB JPEG",
        len(results), sum(
            sum(len(f.get("jpeg", b"")) for f in r["frames"].values())
            for r in results
        ) / 1024,
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

    # Temperature tuning by pass
    if tag == "Pass 1":
        temperature = 0.2
        max_tokens = 8192
    elif tag == "Pass 2":
        temperature = 0.3
        max_tokens = 12000
    else:
        temperature = 0.2
        max_tokens = 8192

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
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
                "temperature": temperature,
                "max_output_tokens": max_tokens,
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
    rep_frames_with_landmarks: list[dict],
    pass1_rep_analyses: list[dict],
) -> dict[str, Any]:
    """Pass 2 — PRIMARY analysis.  Classify exercise from images, score
    form from angles, detect faults.  Uses rep timestamps from Pass 1 as
    structural context only."""

    # Build landmark context from rep frames
    landmark_context = ""
    if rep_frames_with_landmarks:
        for rep_data in rep_frames_with_landmarks:
            rep_num = rep_data["rep_number"]
            landmark_context += f"\nRep {rep_num} Landmark Data (2D x,y normalized | 3D x,y,z normalized):\n"
            for frame_type in ["start", "mid", "end"]:
                if frame_type in rep_data["frames"]:
                    frame = rep_data["frames"][frame_type]
                    t = frame["t"]
                    lm2d = frame["landmarks_2d"]
                    lm3d = frame["landmarks_3d"]

                    landmark_context += f"  {frame_type.upper()} ({t}s):\n"
                    for name in sorted(lm2d.keys()):
                        x2, y2 = lm2d[name]
                        x3, y3, z3 = lm3d[name]
                        landmark_context += f"    {name}: 2D({x2:.3f},{y2:.3f}) 3D({x3:.3f},{y3:.3f},{z3:.3f})\n"

                    # Add computed metrics for key landmarks
                    if "LA" in lm2d and "RA" in lm2d:
                        stance_width = abs(lm2d["LA"][0] - lm2d["RA"][0])
                        landmark_context += f"    Stance width: {stance_width:.3f}\n"
                    if "LH" in lm2d and "RH" in lm2d:
                        hip_width = abs(lm2d["LH"][0] - lm2d["RH"][0])
                        landmark_context += f"    Hip width: {hip_width:.3f}\n"

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

    if rep_frames_with_landmarks:
        # ── Multimodal: images + angles + landmarks (primary path) ──
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"You have {len(rep_frames_with_landmarks)} reps with 3 frames each "
            "(start, mid, end positions), each with 2D and 3D landmark coordinates.\n\n"
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
            "4. Use the angle data and landmark coordinates to score form quality and detect biomechanical "
            "faults for each rep.\n"
            "5. For each fault, output confidence, visual_evidence, and angle_evidence to validate your assessment.\n\n"
            f"{rep_context}"
            f"{landmark_context}\n"
            "Joint angles (degrees) time-series:\n\n"
            f"{angle_text}"
        )

        contents: list = []
        for rep_data in rep_frames_with_landmarks:
            for frame_type in ["start", "mid", "end"]:
                if frame_type in rep_data["frames"]:
                    jpeg = rep_data["frames"][frame_type]["jpeg"]
                    contents.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
        contents.append(prompt)

        logger.info(
            "[Pass 2] Multimodal primary analysis: text + %d images (%.1f KB)",
            len(contents) - 1, sum(
                sum(len(f.get("jpeg", b"")) for f in r["frames"].values())
                for r in rep_frames_with_landmarks
            ) / 1024,
        )
    else:
        # ── Text-only fallback (no rep images available) ──
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{rep_context}"
            f"{landmark_context}\n"
            "Here are the joint angles (degrees) time-series "
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

        # Step D – extract 3 frames (start, mid, end) per rep with 2D/3D landmarks using Pass 1's rep timestamps
        rep_analyses = pass1.get("rep_analyses", [])
        rep_frames_with_landmarks = _extract_rep_frames_with_landmarks(str(mp4_path), rep_analyses, raw_frames)
        logger.info(
            "Extracted %d reps with 3 frames each (start, mid, end)",
            len(rep_frames_with_landmarks),
        )

        # Step E – Pass 2: primary analysis (classification + scoring + faults)
        analysis = _gemini_pass2(angle_text, rep_frames_with_landmarks, rep_analyses)
        
        # Collect frame timestamps for frontend
        rep_frame_timestamps: list[float] = []
        for rep_data in rep_frames_with_landmarks:
            for frame_type in ["start", "mid", "end"]:
                if frame_type in rep_data["frames"]:
                    rep_frame_timestamps.append(rep_data["frames"][frame_type]["t"])
        analysis["rep_frame_timestamps"] = rep_frame_timestamps

        # Buffer per-rep frames with landmarks in Redis for frontend display
        # Recalculate per-rep and overall scores from LLM deductions to keep scoring deterministic
        reps = analysis.get("rep_analyses", []) or []
        if isinstance(reps, list) and reps:
            for rep in reps:
                if not isinstance(rep, dict):
                    continue
                mistakes = rep.get("mistakes", []) or []
                total_deduction = 0.0
                for m in mistakes:
                    if not isinstance(m, dict):
                        continue
                    conf = m.get("confidence")
                    # Apply deduction only when confidence is missing or >= 0.75
                    if conf is not None and conf < 0.75:
                        continue
                    try:
                        ded = float(m.get("deduction", 0) or 0)
                    except (TypeError, ValueError):
                        ded = 0.0
                    if ded > 0:
                        total_deduction += ded

                rating = max(1, min(10, int(round(10 - total_deduction))))
                rep["rating_1_to_10"] = rating

            # Overall form rating = average of per-rep ratings
            ratings = [r.get("rating_1_to_10", 0) for r in reps if isinstance(r, dict)]
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                analysis["form_rating_1_to_10"] = max(1, min(10, int(round(avg_rating))))
                analysis["rep_count"] = len(ratings)
        if task_id and rep_frames_with_landmarks:
            try:
                rep_payload = []
                for rep_data in rep_frames_with_landmarks:
                    rep_entry = {"rep_number": rep_data["rep_number"], "frames": {}}
                    for frame_type in ["start", "mid", "end"]:
                        if frame_type in rep_data["frames"]:
                            frame = rep_data["frames"][frame_type]
                            rep_entry["frames"][frame_type] = {
                                "t": frame["t"],
                                "b64": base64.b64encode(frame["jpeg"]).decode(),
                                "landmarks_2d": frame["landmarks_2d"],
                                "landmarks_3d": frame["landmarks_3d"],
                            }
                    rep_payload.append(rep_entry)
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

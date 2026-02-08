# Hero — Exercise Form Analysis

Short description
-----------------
Hero is an AI-powered exercise form analysis system that evaluates bodyweight and barbell exercises (Push-up, Squat, Deadlift, etc.), produces rep-level feedback and scores, and offers short, actionable coaching corrections — including synthesized voice feedback.

Repository layout
-----------------
- `backend/` — FastAPI server, Celery worker, MediaPipe/ffmpeg processing, Gemini LLM orchestration, and ElevenLabs TTS integration.
- `src/` — Frontend (Vite + React + TypeScript) with pages for recording/uploading, analysis progress, and results overlays.
- `public/` — Static assets.
- `package.json` & `bun.lockb` — Frontend build and scripts.

Key features
------------
- Two-pass LLM analysis (compact angle time-series + multimodal pass with key frames)
- Median-filtered angle smoothing and canonical joint-name normalization
- Client-side downscaling to 480p before upload to save bandwidth
- ElevenLabs TTS synthesis of `actionable_correction` stored in Redis and played in the Results UI

Quick local setup
-----------------
Requirements
- Node.js (recommended 18+)
- Python 3.10+ (backend)
- Redis (for Celery broker/result store)
- ffmpeg (in PATH) — required by backend for video processing

Environment
-----------
Copy `.env.example` to `.env` and fill in values. Important variables:

- `GEMINI_API_KEY` — Google Gemini API key
- `GEMINI_MODEL` — (optional) model to use, e.g. `gemini-1.5-pro`
- `REDIS_URL` — Redis connection string
- `ELEVENLABS_API_KEY` — ElevenLabs API key (for TTS)
- `ELEVENLABS_VOICE_ID` — ElevenLabs voice identifier to use for synthesis

Start the frontend (development)

```bash
# from repository root
npm install
npm run dev
```

Start the backend (development)

```bash
# create and activate Python venv
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r backend/requirements.txt

# start FastAPI server
uvicorn backend.main:app --reload

# start Celery worker in a separate terminal
celery -A backend.worker worker --loglevel=info
```

Notes: The worker expects Redis to be running and the `.env` to contain the keys listed above. Video uploads will be processed, analyzed by the two-pass Gemini pipeline, and if a correction is generated it will be synthesized to audio and stored in Redis under `correction_audio:{task_id}`.

How it works — high level
-------------------------
1. Frontend captures or uploads a short exercise video and posts it to the backend `/analyze` endpoint.
2. Backend normalizes frames with MediaPipe, computes joint angles, applies median smoothing, and buffers landmark frames to Redis.
3. Pass 1 (Gemini): text-only analysis on compact angle time-series to detect repetitions and boundaries.
4. Pass 2 (Gemini multimodal): classification, scoring, and an `actionable_correction` text field, optionally with named joints.
5. The backend synthesizes `actionable_correction` via ElevenLabs and stores the audio in Redis; the frontend fetches and auto-plays the audio on the Results page.

Important files
---------------
- `backend/worker.py` — main Celery processing pipeline, Gemini calls, and ElevenLabs TTS helper
- `backend/main.py` — FastAPI endpoints (including `/correction-audio/{task_id}`)
- `src/pages/Demo.tsx` — client-side 480p downscaling and upload flow
- `src/pages/Results.tsx` — polling results, fetching & auto-playing correction audio
- `src/lib/api.ts` — frontend API helpers (includes `fetchCorrectionAudio`)

Testing
-------
- Frontend unit tests use `vitest` (see `vitest.config.ts`) — run with:

```bash
npm run test
```

Production notes
----------------
- Do not commit secrets to the repository. Use environment/secret managers in production.
- Rate-limit / authenticate public endpoints; the LLM and ElevenLabs APIs are billable.

Next steps
----------------------
- Add a visible "Play correction" button in the Results UI (currently audio auto-plays).
- Add fallback handling when TTS fails: surface textual correction and a retry button.
- Add CI steps to lint and run tests, and a minimal Docker Compose for local integration testing (Redis + backend + frontend).

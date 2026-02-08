// ---------------------------------------------------------------------------
// FormPerfect API client
// ---------------------------------------------------------------------------

const API_BASE = import.meta.env.VITE_API_BASE || "";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface AnalyzeResponse {
  task_id: string;
  detail: string;
}

export interface Mistake {
  fault?: string;
  confidence?: number;
  visual_evidence?: string;
  angle_evidence?: string;
  deduction?: number;
}

export interface RepAnalysis {
  rep_number: number;
  timestamp_start: number;
  timestamp_end: number;
  rating_1_to_10: number;
  mistakes: Array<string | Mistake>;
  problem_joints?: string[];
}

export interface ProblemLandmarkRange {
  start: number;
  end: number;
  landmarks: string[];
}

export interface AnalysisResult {
  analysis_allowed?: boolean;
  rejection_reason?: string;
  exercise_detected: string;
  rep_count: number;
  form_rating_1_to_10: number;
  main_mistakes: string[];
  rep_analyses: RepAnalysis[];
  problem_joints?: string[];
  problem_landmarks?: string[];
  problem_landmark_ranges?: ProblemLandmarkRange[];
  rep_frame_timestamps?: number[];
  actionable_correction: string;
}

export interface ResultResponse {
  status: "Processing" | "Completed" | "Failed";
  detail?: string;
  result?: AnalysisResult;
}

export interface Landmark {
  name: string;
  x: number;
  y: number;
  z: number;
}

export interface LandmarkFrame {
  time_s: number;
  landmarks: Landmark[];
}

export interface LandmarksResponse {
  task_id: string;
  frames: LandmarkFrame[];
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

/** Upload a video for analysis. Returns the task_id for polling. */
export async function uploadVideo(file: File): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("video", file);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail ?? `Upload failed (${res.status})`;
    throw new Error(detail);
  }

  return res.json();
}

/** Poll the status / result of a task. */
export async function fetchResult(taskId: string): Promise<ResultResponse> {
  const res = await fetch(`${API_BASE}/result/${taskId}`);

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Failed to fetch result (${res.status})`);
  }

  return res.json();
}

/** Fetch raw landmark frames for skeleton overlay. */
export async function fetchLandmarks(taskId: string): Promise<LandmarksResponse> {
  const res = await fetch(`${API_BASE}/landmarks/${taskId}`);

  if (!res.ok) {
    if (res.status === 404) throw new Error("Landmarks expired or not found.");
    throw new Error(`Failed to fetch landmarks (${res.status})`);
  }

  return res.json();
}

/** Health check. */
export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

/** Fetch a single mid-rep reference frame image as an object URL. */
export async function fetchRepFrame(
  taskId: string,
  index: number,
): Promise<string | null> {
  try {
    const res = await fetch(`${API_BASE}/rep-frame/${taskId}/${index}`);
    if (!res.ok) return null;
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  } catch {
    return null;
  }
}

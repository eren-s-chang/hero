import { useEffect, useRef } from "react";
import type { LandmarkFrame, ProblemLandmarkRange } from "@/lib/api";

// MediaPipe Pose connections for drawing the skeleton
const POSE_CONNECTIONS: [string, string][] = [
  // Torso
  ["LEFT_SHOULDER", "RIGHT_SHOULDER"],
  ["LEFT_SHOULDER", "LEFT_HIP"],
  ["RIGHT_SHOULDER", "RIGHT_HIP"],
  ["LEFT_HIP", "RIGHT_HIP"],
  // Left arm
  ["LEFT_SHOULDER", "LEFT_ELBOW"],
  ["LEFT_ELBOW", "LEFT_WRIST"],
  // Right arm
  ["RIGHT_SHOULDER", "RIGHT_ELBOW"],
  ["RIGHT_ELBOW", "RIGHT_WRIST"],
  // Left leg
  ["LEFT_HIP", "LEFT_KNEE"],
  ["LEFT_KNEE", "LEFT_ANKLE"],
  ["LEFT_ANKLE", "LEFT_FOOT_INDEX"],
  // Right leg
  ["RIGHT_HIP", "RIGHT_KNEE"],
  ["RIGHT_KNEE", "RIGHT_ANKLE"],
  ["RIGHT_ANKLE", "RIGHT_FOOT_INDEX"],
];

// Key joints get larger dots
const KEY_JOINTS = new Set([
  "LEFT_SHOULDER", "RIGHT_SHOULDER",
  "LEFT_ELBOW", "RIGHT_ELBOW",
  "LEFT_WRIST", "RIGHT_WRIST",
  "LEFT_HIP", "RIGHT_HIP",
  "LEFT_KNEE", "RIGHT_KNEE",
  "LEFT_ANKLE", "RIGHT_ANKLE",
]);

interface SkeletonOverlayProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  frames: LandmarkFrame[];
  /** HSL color string, e.g. "48 90% 55%" */
  color?: string;
  /** Time-ranged problem landmark highlights — red only during each range */
  problemRanges?: ProblemLandmarkRange[];
}

/** Binary search for the frame closest to `time`. */
function findClosestFrame(
  frames: LandmarkFrame[],
  time: number
): LandmarkFrame | null {
  if (!frames.length) return null;
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].time_s < time) lo = mid + 1;
    else hi = mid;
  }
  // lo is the first frame >= time; check lo and lo-1
  let best = frames[lo];
  if (lo > 0 && Math.abs(frames[lo - 1].time_s - time) < Math.abs(best.time_s - time)) {
    best = frames[lo - 1];
  }
  return Math.abs(best.time_s - time) < 0.5 ? best : null;
}

/**
 * Draws a MediaPipe pose skeleton on a <canvas> overlaying the <video>.
 * Syncs to `video.currentTime` via requestAnimationFrame.
 */
export default function SkeletonOverlay({
  videoRef,
  frames,
  color = "48 90% 55%",
  problemRanges = [],
}: SkeletonOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    if (!frames.length) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      const w = video.videoWidth || video.clientWidth;
      const h = video.videoHeight || video.clientHeight;

      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }

      ctx.clearRect(0, 0, w, h);

      // The backend analysed the *downscaled* video whose duration may
      // differ from the original video shown here.  Map the video's
      // playback time into the landmark timeline so they stay in sync.
      const lastT = frames[frames.length - 1]?.time_s || 0;
      const vidDur = video.duration || lastT || 1;
      const scale = lastT > 0 ? lastT / vidDur : 1;
      const lookupTime = video.currentTime * scale;

      // Build the set of problem landmarks active at this moment
      // by checking which ranges contain the current (downscaled) time.
      const problemSet = new Set<string>();
      for (const range of problemRanges) {
        if (lookupTime >= range.start && lookupTime <= range.end) {
          for (const lm of range.landmarks) {
            problemSet.add(lm);
          }
        }
      }
      const RED = "0 85% 55%";

      const frame = findClosestFrame(frames, lookupTime);
      if (!frame) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const lmMap = new Map(frame.landmarks.map((lm) => [lm.name, lm]));

      // Helper: is this connection (edge) part of a problem area?
      const isEdgeProblem = (a: string, b: string) =>
        problemSet.has(a) && problemSet.has(b);

      // ── Pass 1: Dark outline for contrast against any background ──
      ctx.save();
      ctx.lineWidth = 8;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (const [a, b] of POSE_CONNECTIONS) {
        const la = lmMap.get(a);
        const lb = lmMap.get(b);
        if (la && lb) {
          ctx.strokeStyle = isEdgeProblem(a, b)
            ? "rgba(80, 0, 0, 0.8)"
            : "rgba(0, 0, 0, 0.7)";
          ctx.beginPath();
          ctx.moveTo(la.x * w, la.y * h);
          ctx.lineTo(lb.x * w, lb.y * h);
          ctx.stroke();
        }
      }
      ctx.restore();

      // ── Pass 2: Bright colored lines with glow ────────────────────
      ctx.save();
      ctx.lineWidth = 4;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (const [a, b] of POSE_CONNECTIONS) {
        const la = lmMap.get(a);
        const lb = lmMap.get(b);
        if (la && lb) {
          const isProblem = isEdgeProblem(a, b);
          const edgeColor = isProblem ? RED : color;
          ctx.shadowColor = `hsl(${edgeColor} / 0.8)`;
          ctx.shadowBlur = isProblem ? 20 : 14;
          ctx.strokeStyle = `hsl(${edgeColor})`;
          ctx.beginPath();
          ctx.moveTo(la.x * w, la.y * h);
          ctx.lineTo(lb.x * w, lb.y * h);
          ctx.stroke();
        }
      }
      ctx.restore();

      // ── Pass 3: Joint dots ────────────────────────────────────────
      ctx.save();
      for (const [name, lm] of lmMap) {
        const isKey = KEY_JOINTS.has(name);
        const isProblem = problemSet.has(name);
        const jointColor = isProblem ? RED : color;
        const radius = isProblem ? 9 : isKey ? 7 : 4;

        ctx.shadowColor = `hsl(${jointColor} / 0.8)`;
        ctx.shadowBlur = isProblem ? 22 : 10;

        // Dark outline ring
        ctx.fillStyle = isProblem ? "rgba(80, 0, 0, 0.7)" : "rgba(0, 0, 0, 0.6)";
        ctx.beginPath();
        ctx.arc(lm.x * w, lm.y * h, radius + 2, 0, Math.PI * 2);
        ctx.fill();

        // Bright fill
        ctx.fillStyle = `hsl(${jointColor})`;
        ctx.beginPath();
        ctx.arc(lm.x * w, lm.y * h, radius, 0, Math.PI * 2);
        ctx.fill();

        // White center highlight on key joints / problem joints
        if (isKey || isProblem) {
          ctx.fillStyle = isProblem
            ? "rgba(255, 255, 255, 0.8)"
            : "rgba(255, 255, 255, 0.6)";
          ctx.beginPath();
          ctx.arc(lm.x * w, lm.y * h, isProblem ? 4 : 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [frames, color, videoRef, problemRanges]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ objectFit: "contain" }}
    />
  );
}

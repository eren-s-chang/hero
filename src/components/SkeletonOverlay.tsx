import { useEffect, useRef } from "react";
import type { LandmarkFrame } from "@/lib/api";

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

      const frame = findClosestFrame(frames, video.currentTime);
      if (!frame) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const lmMap = new Map(frame.landmarks.map((lm) => [lm.name, lm]));

      // ── Pass 1: Dark outline for contrast against any background ──
      ctx.save();
      ctx.strokeStyle = "rgba(0, 0, 0, 0.7)";
      ctx.lineWidth = 8;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (const [a, b] of POSE_CONNECTIONS) {
        const la = lmMap.get(a);
        const lb = lmMap.get(b);
        if (la && lb) {
          ctx.beginPath();
          ctx.moveTo(la.x * w, la.y * h);
          ctx.lineTo(lb.x * w, lb.y * h);
          ctx.stroke();
        }
      }
      ctx.restore();

      // ── Pass 2: Bright colored lines with glow ────────────────────
      ctx.save();
      ctx.shadowColor = `hsl(${color} / 0.8)`;
      ctx.shadowBlur = 14;
      ctx.strokeStyle = `hsl(${color})`;
      ctx.lineWidth = 4;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (const [a, b] of POSE_CONNECTIONS) {
        const la = lmMap.get(a);
        const lb = lmMap.get(b);
        if (la && lb) {
          ctx.beginPath();
          ctx.moveTo(la.x * w, la.y * h);
          ctx.lineTo(lb.x * w, lb.y * h);
          ctx.stroke();
        }
      }
      ctx.restore();

      // ── Pass 3: Joint dots ────────────────────────────────────────
      ctx.save();
      ctx.shadowColor = `hsl(${color} / 0.8)`;
      ctx.shadowBlur = 10;
      for (const [name, lm] of lmMap) {
        const isKey = KEY_JOINTS.has(name);
        const radius = isKey ? 7 : 4;

        // Dark outline ring
        ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
        ctx.beginPath();
        ctx.arc(lm.x * w, lm.y * h, radius + 2, 0, Math.PI * 2);
        ctx.fill();

        // Bright fill
        ctx.fillStyle = `hsl(${color})`;
        ctx.beginPath();
        ctx.arc(lm.x * w, lm.y * h, radius, 0, Math.PI * 2);
        ctx.fill();

        // White center highlight on key joints
        if (isKey) {
          ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
          ctx.beginPath();
          ctx.arc(lm.x * w, lm.y * h, 3, 0, Math.PI * 2);
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
  }, [frames, color, videoRef]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ objectFit: "contain" }}
    />
  );
}

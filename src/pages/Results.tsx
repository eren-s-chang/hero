import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Loader2,
  RotateCcw,
  Zap,
  Target,
  TrendingDown,
  Camera,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";
import { toast } from "sonner";
import {
  fetchResult,
  fetchLandmarks,
  fetchRepFrame,
  type AnalysisResult,
  type RepAnalysis,
  type LandmarkFrame,
  type Mistake,
} from "@/lib/api";
import SkeletonOverlay from "@/components/SkeletonOverlay";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function ratingColor(r: number): string {
  if (r >= 8) return "148 60% 50%"; // green
  if (r >= 5) return "48 90% 55%"; // gold/yellow (primary)
  return "0 75% 50%"; // red (secondary)
}

function ratingLabel(r: number): string {
  if (r >= 9) return "Excellent";
  if (r >= 8) return "Great";
  if (r >= 6) return "Good";
  if (r >= 4) return "Fair";
  return "Needs Work";
}

function RatingRing({ rating, size = 120 }: { rating: number; size?: number }) {
  const pct = (rating / 10) * 100;
  const color = ratingColor(rating);
  const r = (size - 12) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (pct / 100) * circ;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="hsl(0 0% 18%)"
          strokeWidth={8}
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={`hsl(${color})`}
          strokeWidth={8}
          strokeLinecap="round"
          strokeDasharray={circ}
          initial={{ strokeDashoffset: circ }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.2, ease: "easeOut" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span
          className="font-heading text-4xl"
          style={{ color: `hsl(${color})` }}
        >
          {rating}
        </span>
        <span className="text-muted-foreground text-xs font-modern font-semibold">
          / 10
        </span>
      </div>
    </div>
  );
}

function RepCard({
  rep,
  isActive,
  onClick,
}: {
  rep: RepAnalysis;
  isActive: boolean;
  onClick: () => void;
}) {
  const color = ratingColor(rep.rating_1_to_10);
  const clean = rep.mistakes.length === 0;

  return (
    <motion.button
      whileHover={{ scale: 1.04 }}
      whileTap={{ scale: 0.97 }}
      onClick={onClick}
      className={`flex-shrink-0 w-36 border-2 rounded-md p-4 text-left transition-colors ${
        isActive
          ? "border-primary bg-primary/10"
          : "border-border bg-card hover:border-primary/50"
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="font-heading text-lg tracking-wider">REP {rep.rep_number}</span>
        <span
          className="font-modern font-bold text-lg"
          style={{ color: `hsl(${color})` }}
        >
          {rep.rating_1_to_10}
        </span>
      </div>
      {clean ? (
        <span className="flex items-center gap-1 text-xs font-modern text-green-400">
          <CheckCircle2 className="w-3 h-3" /> Clean
        </span>
      ) : (
        <span className="text-xs text-muted-foreground font-modern">
          {rep.mistakes.length} issue{rep.mistakes.length > 1 ? "s" : ""}
        </span>
      )}
    </motion.button>
  );
}

// ---------------------------------------------------------------------------
// Main Results page
// ---------------------------------------------------------------------------

export default function Results() {
  const { taskId } = useParams<{ taskId: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const videoUrl: string | undefined = (location.state as any)?.videoUrl;

  const videoRef = useRef<HTMLVideoElement>(null);

  // Polling
  const [status, setStatus] = useState<"polling" | "completed" | "failed">("polling");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string>("");
  const [elapsed, setElapsed] = useState(0);

  // Landmarks
  const [frames, setFrames] = useState<LandmarkFrame[]>([]);

  // Per-rep mid-rep reference frames
  const [repFrameUrls, setRepFrameUrls] = useState<string[]>([]);

  // Rep selection
  const [activeRep, setActiveRep] = useState<number | null>(null);

  // ---- Polling loop --------------------------------------------------------
  useEffect(() => {
    if (!taskId) return;
    let cancelled = false;
    const start = Date.now();

    const poll = async () => {
      while (!cancelled) {
        try {
          const data = await fetchResult(taskId);
          if (cancelled) break;

          if (data.status === "Completed" && data.result) {
            setResult(data.result);
            setStatus("completed");
            return;
          }

          if (data.status === "Failed") {
            setError(data.detail ?? "Analysis failed.");
            setStatus("failed");
            return;
          }

          // Still processing — wait 2s and retry
          setElapsed(Math.round((Date.now() - start) / 1000));
          await new Promise((r) => setTimeout(r, 2000));
        } catch (err: any) {
          if (cancelled) break;
          setError(err.message ?? "Connection lost.");
          setStatus("failed");
          return;
        }
      }
    };

    poll();
    const timer = setInterval(() => {
      setElapsed(Math.round((Date.now() - start) / 1000));
    }, 1000);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [taskId]);

  // ---- Fetch landmarks once completed ----------------------------------------
  useEffect(() => {
    if (status !== "completed" || !taskId) return;

    fetchLandmarks(taskId)
      .then((data) => setFrames(data.frames))
      .catch(() => {
        console.warn("Could not load landmarks for overlay");
      });
  }, [status, taskId]);

  // ---- Fetch per-rep mid-rep frames once result is available ----------------
  useEffect(() => {
    if (!taskId || !result?.rep_frame_timestamps?.length) return;

    Promise.all(
      result.rep_frame_timestamps.map((_, i) => fetchRepFrame(taskId, i))
    )
      .then((urls) =>
        setRepFrameUrls(urls.filter((u): u is string => u !== null))
      )
      .catch(() => console.warn("Could not load rep frames"));
  }, [taskId, result]);

  // ---- Time-scale factor ---------------------------------------------------
  // The backend analysed the *downscaled* video whose duration may differ
  // from the original video shown in the player.  All rep timestamps and
  // landmark times are on the downscaled timeline, so we scale them.
  const getTimeScale = (): number => {
    const video = videoRef.current;
    if (!video?.duration || !frames.length) return 1;
    const lastLandmark = frames[frames.length - 1]?.time_s;
    return lastLandmark > 0 ? video.duration / lastLandmark : 1;
  };

  // ---- Seek video to rep ---------------------------------------------------
  const seekToRep = (rep: RepAnalysis) => {
    setActiveRep(rep.rep_number);
    const video = videoRef.current;
    if (video) {
      const scale = getTimeScale();
      video.currentTime = rep.timestamp_start * scale;
      video.play().catch(() => {});
    }
  };

  // ---- Figure out which rep is active based on video time ------------------
  useEffect(() => {
    if (!result || result.analysis_allowed === false || !videoRef.current) return;
    const video = videoRef.current;

    const onTimeUpdate = () => {
      // Map video time back to the downscaled timeline for rep matching
      const scale = getTimeScale();
      const t = video.currentTime / scale;
      const current = result.rep_analyses.find(
        (r) => t >= r.timestamp_start && t <= r.timestamp_end
      );
      if (current) setActiveRep(current.rep_number);
    };

    video.addEventListener("timeupdate", onTimeUpdate);
    return () => video.removeEventListener("timeupdate", onTimeUpdate);
  }, [result]);

  // ---- Render: Polling state -----------------------------------------------
  if (status === "polling") {
    return (
      <main className="bg-background text-foreground min-h-screen relative overflow-hidden flex flex-col">
        {/* Particles */}
        <div className="absolute inset-0 pointer-events-none">
          {[...Array(8)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-primary/30 rounded-full"
              initial={{ x: `${Math.random() * 100}%`, y: "100%", opacity: 0 }}
              animate={{ y: "-10%", opacity: [0, 1, 0] }}
              transition={{
                duration: 5 + Math.random() * 3,
                repeat: Infinity,
                delay: i * 0.6,
                ease: "linear",
              }}
            />
          ))}
        </div>

        <Nav onBack={() => navigate("/demo")} />

        <div className="flex-1 flex flex-col items-center justify-center px-6 gap-8">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          >
            <Loader2 className="w-16 h-16 text-primary" />
          </motion.div>

          <div className="text-center">
            <h2 className="font-heading text-4xl md:text-5xl tracking-wider mb-2">
              ANALYZING YOUR <span className="text-primary">FORM</span>
            </h2>
            <p className="text-muted-foreground font-modern text-lg">
              This usually takes 10–20 seconds…
            </p>
            <p className="text-muted-foreground font-modern text-sm mt-2">
              {elapsed}s elapsed
            </p>
          </div>

          {/* Pulsing video thumbnail */}
          {videoUrl && (
            <motion.div
              animate={{ opacity: [0.4, 0.7, 0.4] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-full max-w-md rounded-md overflow-hidden border-2 border-border"
            >
              <video
                src={videoUrl}
                muted
                className="w-full max-h-[240px] object-contain bg-background"
              />
            </motion.div>
          )}
        </div>
      </main>
    );
  }

  // ---- Render: Failed state ------------------------------------------------
  if (status === "failed") {
    return (
      <main className="bg-background text-foreground min-h-screen relative overflow-hidden flex flex-col">
        <Nav onBack={() => navigate("/demo")} />

        <div className="flex-1 flex flex-col items-center justify-center px-6 gap-6">
          <XCircle className="w-16 h-16 text-destructive" />
          <h2 className="font-heading text-4xl tracking-wider">ANALYSIS FAILED</h2>
          <p className="text-muted-foreground font-modern max-w-md text-center">
            {error}
          </p>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => navigate("/demo")}
            className="font-heading bg-secondary text-secondary-foreground px-8 py-3 text-xl tracking-wider rounded-md flex items-center gap-2"
          >
            <RotateCcw className="w-5 h-5" /> TRY AGAIN
          </motion.button>
        </div>
      </main>
    );
  }

  // ---- Render: Results -----------------------------------------------------
  if (!result) return null;

  const analysisAllowed = result.analysis_allowed !== false;
  const rejectionReason = (result.rejection_reason || "").trim();
  const overlayProblemRanges =
    result.problem_landmark_ranges && result.problem_landmark_ranges.length > 0
      ? result.problem_landmark_ranges
      : result.problem_landmarks && result.problem_landmarks.length > 0 && frames.length > 0
        ? [
            {
              start: 0,
              end: frames[frames.length - 1].time_s,
              landmarks: result.problem_landmarks,
            },
          ]
        : undefined;

  const overlayProblemRanges =
    result.problem_landmark_ranges && result.problem_landmark_ranges.length > 0
      ? result.problem_landmark_ranges
      : result.problem_landmarks && result.problem_landmarks.length > 0 && frames.length > 0
        ? [
            {
              start: 0,
              end: frames[frames.length - 1].time_s,
              landmarks: result.problem_landmarks,
            },
          ]
        : undefined;

  if (!analysisAllowed) {
    return (
      <main className="bg-background text-foreground min-h-screen relative overflow-hidden">
        {/* Particles */}
        <div className="absolute inset-0 pointer-events-none">
          {[...Array(6)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-primary/20 rounded-full"
              initial={{ x: `${Math.random() * 100}%`, y: "100%", opacity: 0 }}
              animate={{ y: "-10%", opacity: [0, 1, 0] }}
              transition={{
                duration: 6 + Math.random() * 3,
                repeat: Infinity,
                delay: i * 0.8,
                ease: "linear",
              }}
            />
          ))}
        </div>

        <Nav onBack={() => navigate("/demo")} />

        <div className="relative z-20 px-6 md:px-12 max-w-4xl mx-auto pb-20">
          <motion.div
            initial={{ y: 30, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="flex items-start gap-4 bg-card border-2 border-border rounded-md p-6"
          >
            <AlertTriangle className="w-8 h-8 text-yellow-400 flex-shrink-0" />
            <div>
              <p className="font-heading text-primary text-lg tracking-wider">
                VIDEO NOT ELIGIBLE
              </p>
              <h1 className="font-heading text-3xl md:text-5xl tracking-wider mb-2">
                REP-BASED GYM EXERCISES ONLY
              </h1>
              <p className="text-muted-foreground font-modern">
                {rejectionReason ||
                  "We could not detect a common gym exercise with clear, countable reps. Sports activities and non-rep movements are not supported."}
              </p>
            </div>
          </motion.div>

          {videoUrl && (
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="mt-6"
            >
              <h3 className="font-heading text-2xl tracking-wider mb-4">
                YOUR VIDEO
              </h3>
              <div className="relative rounded-md overflow-hidden border-2 border-border bg-background">
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  playsInline
                  className="w-full max-h-[500px] object-contain"
                />
              </div>
            </motion.div>
          )}

          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mt-10 flex justify-center"
          >
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => navigate("/demo")}
              className="font-heading bg-secondary text-secondary-foreground px-10 py-4 text-2xl tracking-wider rounded-md flex items-center gap-3"
            >
              <RotateCcw className="w-6 h-6" /> UPLOAD A NEW CLIP
            </motion.button>
          </motion.div>
        </div>
      </main>
    );
  }

  return (
    <main className="bg-background text-foreground min-h-screen relative overflow-hidden">
      {/* Particles */}
      <div className="absolute inset-0 pointer-events-none">
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-primary/20 rounded-full"
            initial={{ x: `${Math.random() * 100}%`, y: "100%", opacity: 0 }}
            animate={{ y: "-10%", opacity: [0, 1, 0] }}
            transition={{
              duration: 6 + Math.random() * 3,
              repeat: Infinity,
              delay: i * 0.8,
              ease: "linear",
            }}
          />
        ))}
      </div>

      <Nav onBack={() => navigate("/demo")} />

      <div className="relative z-20 px-6 md:px-12 max-w-6xl mx-auto pb-20">
        {/* ── Header row ─────────────────────────────────────────────── */}
        <motion.div
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="flex flex-col md:flex-row items-start md:items-center gap-6 mb-10"
        >
          <RatingRing rating={result.form_rating_1_to_10} />

          <div className="flex-1">
            <p className="font-heading text-primary text-lg tracking-wider">
              {result.exercise_detected.toUpperCase()}
            </p>
            <h1 className="font-heading text-4xl md:text-6xl tracking-wider">
              {ratingLabel(result.form_rating_1_to_10).toUpperCase()}
            </h1>
            <p className="text-muted-foreground font-modern mt-1">
              {result.rep_count} rep{result.rep_count !== 1 ? "s" : ""} detected
            </p>
          </div>
        </motion.div>

        {/* ── Actionable correction ─────────────────────────────────────────────── */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="bg-secondary/10 border-2 border-secondary rounded-md p-5 mb-8 flex gap-4 items-start"
        >
          <Zap className="w-6 h-6 text-secondary flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-heading text-xl tracking-wider text-secondary mb-1">
              TOP CORRECTION
            </p>
            <p className="font-modern text-foreground/90 leading-relaxed">
              {result.actionable_correction}
            </p>
          </div>
        </motion.div>

        {/* ── Main mistakes ─────────────────────────────────────────── */}
        {result.main_mistakes.length > 0 && (
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.15 }}
            className="mb-8"
          >
            <h3 className="font-heading text-2xl tracking-wider mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-primary" /> COMMON MISTAKES
            </h3>
            <ul className="space-y-2">
              {result.main_mistakes.map((m, i) => (
                <li
                  key={i}
                  className="flex items-start gap-3 bg-card border border-border rounded-md p-3"
                >
                  <AlertTriangle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                  <span className="font-modern">{m}</span>
                </li>
              ))}
            </ul>
          </motion.div>
        )}

        {/* ── Video player with skeleton overlay ────────────────────── */}
        {videoUrl && (
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mb-8"
          >
            <h3 className="font-heading text-2xl tracking-wider mb-4">
              YOUR VIDEO
            </h3>

            {/* Per-rep reference frames */}
            {repFrameUrls.length > 0 && (
              <div className="mb-4 bg-card border border-border rounded-md p-4">
                <p className="font-heading text-sm tracking-wider text-primary flex items-center gap-1.5 mb-2">
                  <Camera className="w-3.5 h-3.5" /> REP REFERENCE FRAMES
                </p>
                <p className="text-xs text-muted-foreground font-modern leading-relaxed mb-3">
                  One frame per rep at the midpoint — sent to the AI for visual
                  verification.
                </p>
                <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
                  {repFrameUrls.map((url, i) => (
                    <div
                      key={i}
                      className="relative flex-shrink-0 rounded overflow-hidden border border-border"
                    >
                      <img
                        src={url}
                        alt={`Rep ${i + 1} frame`}
                        className="w-24 h-auto object-contain"
                      />
                      <div className="absolute bottom-0 inset-x-0 bg-black/70 text-center py-0.5">
                        <span className="text-[10px] font-modern text-primary">
                          {result.rep_frame_timestamps?.[i] != null
                            ? `Rep ${i + 1} · ${result.rep_frame_timestamps[i].toFixed(1)}s`
                            : `Rep ${i + 1}`}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div className="relative rounded-md overflow-hidden border-2 border-border bg-background">
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                playsInline
                className="w-full max-h-[500px] object-contain"
              />
              {frames.length > 0 && (
                <SkeletonOverlay
                  videoRef={videoRef}
                  frames={frames}
                  problemRanges={overlayProblemRanges}
                />
              )}
            </div>

            {/* Rep timeline bar under video */}
            {result.rep_analyses.length > 0 && videoRef.current && (
              <RepTimelineBar
                reps={result.rep_analyses}
                duration={
                  frames.length > 0
                    ? frames[frames.length - 1].time_s
                    : videoRef.current.duration || 1
                }
                activeRep={activeRep}
                onSeek={seekToRep}
              />
            )}
          </motion.div>
        )}

        {/* ── Rep cards ─────────────────────────────────────────────── */}
        {result.rep_analyses.length > 0 && (
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.25 }}
          >
            <h3 className="font-heading text-2xl tracking-wider mb-4 flex items-center gap-2">
              <TrendingDown className="w-5 h-5 text-primary" /> REP BREAKDOWN
            </h3>

            <div className="flex gap-3 overflow-x-auto pb-4 scrollbar-thin">
              {result.rep_analyses.map((rep) => (
                <RepCard
                  key={rep.rep_number}
                  rep={rep}
                  isActive={activeRep === rep.rep_number}
                  onClick={() => seekToRep(rep)}
                />
              ))}
            </div>

            {/* Detail panel for active rep */}
            <AnimatePresence mode="wait">
              {activeRep != null && (
                <ActiveRepDetail
                  rep={result.rep_analyses.find((r) => r.rep_number === activeRep)!}
                />
              )}
            </AnimatePresence>
          </motion.div>
        )}

        {/* ── Upload another ────────────────────────────────────────── */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mt-12 flex justify-center"
        >
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => navigate("/demo")}
            className="font-heading bg-secondary text-secondary-foreground px-10 py-4 text-2xl tracking-wider rounded-md flex items-center gap-3"
          >
            <RotateCcw className="w-6 h-6" /> ANALYZE ANOTHER
          </motion.button>
        </motion.div>
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Nav({ onBack }: { onBack: () => void }) {
  const navigate = useNavigate();
  return (
    <motion.nav
      initial={{ y: -30, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6 }}
      className="relative z-30 flex items-center justify-between px-6 md:px-12 py-8"
    >
      <span
        className="opm-title text-6xl md:text-7xl lg:text-8xl cursor-pointer"
        onClick={() => navigate("/")}
      >
        HERO
      </span>
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.98 }}
        onClick={onBack}
        className="font-heading text-muted-foreground hover:text-foreground flex items-center gap-2 text-xl tracking-wider transition-colors"
      >
        <ArrowLeft className="w-5 h-5" />
        BACK
      </motion.button>
    </motion.nav>
  );
}

function RepTimelineBar({
  reps,
  duration,
  activeRep,
  onSeek,
}: {
  reps: RepAnalysis[];
  duration: number;
  activeRep: number | null;
  onSeek: (rep: RepAnalysis) => void;
}) {
  return (
    <div className="mt-3 relative h-3 rounded-md overflow-hidden bg-border/30 w-full">
      {reps.map((rep) => {
        const widthPct =
          ((rep.timestamp_end - rep.timestamp_start) / duration) * 100;
        const leftPct = (rep.timestamp_start / duration) * 100;
        const color = ratingColor(rep.rating_1_to_10);
        const isActive = activeRep === rep.rep_number;

        return (
          <button
            key={rep.rep_number}
            onClick={() => onSeek(rep)}
            title={`Rep ${rep.rep_number} — ${rep.rating_1_to_10}/10`}
            className="absolute top-0 h-full transition-opacity rounded-sm"
            style={{
              left: `${leftPct}%`,
              width: `${Math.max(widthPct, 1)}%`,
              backgroundColor: `hsl(${color})`,
              opacity: isActive ? 1 : 0.5,
            }}
          />
        );
      })}
    </div>
  );
}

function ActiveRepDetail({ rep }: { rep: RepAnalysis }) {
  const color = ratingColor(rep.rating_1_to_10);
  const clean = rep.mistakes.length === 0;

  return (
    <motion.div
      key={rep.rep_number}
      initial={{ y: 10, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: -10, opacity: 0 }}
      transition={{ duration: 0.25 }}
      className="mt-4 bg-card border border-border rounded-md p-5"
    >
      <div className="flex items-center justify-between mb-3">
        <span className="font-heading text-2xl tracking-wider">
          REP {rep.rep_number}
        </span>
        <span
          className="font-heading text-3xl"
          style={{ color: `hsl(${color})` }}
        >
          {rep.rating_1_to_10}/10
        </span>
      </div>
      <p className="text-muted-foreground font-modern text-sm mb-3">
        {rep.timestamp_start.toFixed(1)}s – {rep.timestamp_end.toFixed(1)}s
      </p>
      {clean ? (
        <p className="flex items-center gap-2 text-green-400 font-modern">
          <CheckCircle2 className="w-4 h-4" /> No issues detected — clean rep!
        </p>
      ) : (
        <ul className="space-y-2">
          {rep.mistakes.map((m, i) => (
            <MistakeItem key={i} mistake={m} />
          ))}
        </ul>
      )}
    </motion.div>
  );
}

function MistakeItem({ mistake }: { mistake: string | Mistake }) {
  if (typeof mistake === "string") {
    return (
      <li className="flex items-start gap-2 font-modern">
        <AlertTriangle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
        <span>{mistake}</span>
      </li>
    );
  }

  const { fault, visual_evidence, angle_evidence, confidence, deduction } = mistake;
  const details: string[] = [];
  if (visual_evidence) details.push(`Visual: ${visual_evidence}`);
  if (angle_evidence) details.push(`Angles: ${angle_evidence}`);

  const meta: string[] = [];
  if (typeof confidence === "number") meta.push(`Confidence ${(confidence * 100).toFixed(0)}%`);
  if (typeof deduction === "number") meta.push(`Deduction -${deduction.toFixed(1)}`);

  return (
    <li className="flex items-start gap-2 font-modern">
      <AlertTriangle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
      <div className="space-y-1">
        <p className="font-semibold text-foreground">
          {fault || "Form issue"}
        </p>
        {details.length > 0 && (
          <p className="text-xs text-muted-foreground leading-relaxed">{details.join(" · ")}</p>
        )}
        {meta.length > 0 && (
          <p className="text-[11px] text-muted-foreground/80">{meta.join(" · ")}</p>
        )}
      </div>
    </li>
  );
}

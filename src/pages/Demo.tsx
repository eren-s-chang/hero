import { motion } from "framer-motion";
import { Upload, ArrowLeft, Video, X } from "lucide-react";
import { useState, useRef } from "react";
import { ArrayBufferTarget, Muxer } from "webm-muxer";
import { useNavigate } from "react-router-dom";

const Demo = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  const [trimStart, setTrimStart] = useState(0);
  const [trimEnd, setTrimEnd] = useState(0);
  const [isTrimming, setIsTrimming] = useState(false);
  const [trimmedFile, setTrimmedFile] = useState<File | null>(null);
  const [trimmedUrl, setTrimmedUrl] = useState<string | null>(null);
  const [activeThumb, setActiveThumb] = useState<"start" | "end" | null>(null);

  const formatTime = (value: number) => {
    const minutes = Math.floor(value / 60);
    const seconds = Math.floor(value % 60)
      .toString()
      .padStart(2, "0");
    const tenths = Math.floor((value % 1) * 10);
    return `${minutes}:${seconds}.${tenths}`;
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const handleFile = (file: File) => {
    if (file.type.startsWith("video/")) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setTrimmedFile(null);
      if (trimmedUrl) URL.revokeObjectURL(trimmedUrl);
      setTrimmedUrl(null);
      setDuration(0);
      setTrimStart(0);
      setTrimEnd(0);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) handleFile(e.target.files[0]);
  };

  const clearFile = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    if (trimmedUrl) URL.revokeObjectURL(trimmedUrl);
    setTrimmedUrl(null);
    setTrimmedFile(null);
    setDuration(0);
    setTrimStart(0);
    setTrimEnd(0);
  };

  const revertToOriginal = () => {
    if (trimmedUrl) URL.revokeObjectURL(trimmedUrl);
    setTrimmedUrl(null);
    setTrimmedFile(null);
    if (duration > 0) {
      setTrimStart(0);
      setTrimEnd(duration);
    }
  };

  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(video.duration)) return;
    const nextDuration = Math.max(0, video.duration);
    setDuration(nextDuration);
    setTrimStart(0);
    setTrimEnd(nextDuration);
  };

  const handleTrim = async () => {
    const video = videoRef.current;
    const sourceFile = trimmedFile ?? selectedFile;
    if (!video || !sourceFile) return;
    if (trimEnd <= trimStart) return;
    if (!(video as any).captureStream) return;
    if (!(window as any).VideoEncoder || !(window as any).AudioEncoder) return;
    if (!(window as any).MediaStreamTrackProcessor) return;

    setIsTrimming(true);
    const waitForSeek = (targetTime: number) =>
      new Promise<void>((resolve) => {
        const handleSeeked = () => {
          video.removeEventListener("seeked", handleSeeked);
          resolve();
        };

        if (Math.abs(video.currentTime - targetTime) < 0.01) {
          resolve();
          return;
        }

        video.addEventListener("seeked", handleSeeked, { once: true });
        video.currentTime = targetTime;
      });

    const trimDuration = Math.max(0, trimEnd - trimStart);
    const muxerTarget = new ArrayBufferTarget();
    const stream = (video as any).captureStream();
    const [videoTrack] = stream.getVideoTracks();
    const [audioTrack] = stream.getAudioTracks();
    const videoSettings = videoTrack?.getSettings?.() ?? {};
    const audioSettings = audioTrack?.getSettings?.() ?? {};
    const width = video.videoWidth || Number(videoSettings.width) || 1280;
    const height = video.videoHeight || Number(videoSettings.height) || 720;
    const frameRate = Number(videoSettings.frameRate) || 30;
    const sampleRate = Number(audioSettings.sampleRate) || 48000;
    const channelCount = Number(audioSettings.channelCount) || 2;

    const muxer = new Muxer({
      target: muxerTarget,
      video: {
        codec: "V_VP8",
        width,
        height,
        frameRate,
      },
      audio: audioTrack
        ? {
            codec: "A_OPUS",
            sampleRate,
            numberOfChannels: channelCount,
          }
        : undefined,
    });

    let videoTimestampOffset: number | null = null;
    let audioTimestampOffset: number | null = null;
    let hasKeyframe = false;
    let stopRequested = false;

    const handleVideoChunk = (chunk: EncodedVideoChunk, meta?: EncodedVideoChunkMetadata) => {
      if (videoTimestampOffset === null) videoTimestampOffset = chunk.timestamp;
      const data = new Uint8Array(chunk.byteLength);
      chunk.copyTo(data);
      const adjustedChunk = new EncodedVideoChunk({
        type: chunk.type,
        timestamp: chunk.timestamp - (videoTimestampOffset ?? 0),
        duration: chunk.duration ?? undefined,
        data,
      });
      muxer.addVideoChunk(adjustedChunk, meta);
    };

    const handleAudioChunk = (chunk: EncodedAudioChunk, meta?: EncodedAudioChunkMetadata) => {
      if (audioTimestampOffset === null) audioTimestampOffset = chunk.timestamp;
      const data = new Uint8Array(chunk.byteLength);
      chunk.copyTo(data);
      const adjustedChunk = new EncodedAudioChunk({
        type: chunk.type,
        timestamp: chunk.timestamp - (audioTimestampOffset ?? 0),
        duration: chunk.duration ?? undefined,
        data,
      });
      muxer.addAudioChunk(adjustedChunk, meta);
    };

    const videoEncoder = new VideoEncoder({
      output: handleVideoChunk,
      error: () => {},
    });
    const audioEncoder = audioTrack
      ? new AudioEncoder({
          output: handleAudioChunk,
          error: () => {},
        })
      : null;

    videoEncoder.configure({
      codec: "vp8",
      width,
      height,
      bitrate: 2_500_000,
      framerate: frameRate,
    });

    if (audioEncoder) {
      audioEncoder.configure({
        codec: "opus",
        sampleRate,
        numberOfChannels: channelCount,
        bitrate: 128_000,
      });
    }

    const videoProcessor = new (window as any).MediaStreamTrackProcessor({ track: videoTrack });
    const audioProcessor = audioTrack
      ? new (window as any).MediaStreamTrackProcessor({ track: audioTrack })
      : null;
    const videoReader = videoProcessor.readable.getReader();
    const audioReader = audioProcessor?.readable.getReader() ?? null;

    const stopCapture = () => {
      if (stopRequested) return;
      stopRequested = true;
      if (!video.paused) video.pause();
    };

    try {
      const previousMuted = video.muted;
      video.muted = true;
      await waitForSeek(trimStart);
      await video.play();

      const timeGuard = window.setInterval(() => {
        if (video.currentTime >= trimEnd) stopCapture();
      }, 20);

      const consumeVideo = (async () => {
        while (!stopRequested) {
          const result = await videoReader.read();
          if (result.done) break;
          const frame = result.value as VideoFrame;
          try {
            if (stopRequested) break;
            if (!hasKeyframe) {
              videoEncoder.encode(frame, { keyFrame: true });
              hasKeyframe = true;
            } else {
              videoEncoder.encode(frame);
            }
          } finally {
            frame.close();
          }
        }
      })();

      const consumeAudio = audioReader
        ? (async () => {
            while (!stopRequested) {
              const result = await audioReader.read();
              if (result.done) break;
              const audioData = result.value as AudioData;
              try {
                if (stopRequested) break;
                audioEncoder?.encode(audioData);
              } finally {
                audioData.close();
              }
            }
          })()
        : Promise.resolve();

      await Promise.all([consumeVideo, consumeAudio]);
      window.clearInterval(timeGuard);
      await videoReader.cancel();
      if (audioReader) await audioReader.cancel();
      await videoEncoder.flush();
      if (audioEncoder) await audioEncoder.flush();
      muxer.finalize();

      video.muted = previousMuted;
      const blob = new Blob([muxerTarget.buffer], { type: "video/webm" });
      const baseName = sourceFile.name.replace(/\.[^/.]+$/, "");
      const nextFile = new File([blob], `${baseName}-trimmed.webm`, { type: blob.type });
      setTrimmedFile(nextFile);
      if (trimmedUrl) URL.revokeObjectURL(trimmedUrl);
      setTrimmedUrl(URL.createObjectURL(nextFile));
    } catch (error) {
      stopCapture();
    } finally {
      setIsTrimming(false);
    }
  };

  return (
    <main className="bg-background text-foreground min-h-screen relative overflow-hidden">
      {/* Background particles */}
      <div className="absolute inset-0 pointer-events-none">
        {[...Array(8)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-primary/30 rounded-full"
            initial={{ x: `${Math.random() * 100}%`, y: "100%", opacity: 0 }}
            animate={{ y: "-10%", opacity: [0, 1, 0] }}
            transition={{ duration: 5 + Math.random() * 3, repeat: Infinity, delay: i * 0.6, ease: "linear" }}
          />
        ))}
      </div>

      {/* Nav */}
      <motion.nav
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="relative z-30 flex items-center justify-between px-6 md:px-12 py-8"
      >
        <span className="opm-title text-6xl md:text-7xl lg:text-8xl cursor-pointer" onClick={() => navigate("/")}>
          HERO
        </span>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => navigate("/")}
          className="font-heading text-muted-foreground hover:text-foreground flex items-center gap-2 text-xl tracking-wider transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          BACK
        </motion.button>
      </motion.nav>

      {/* Content */}
      <div className="relative z-20 px-6 md:px-12 max-w-4xl mx-auto pt-8 pb-20">
        <motion.div
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mb-4"
        >
          <p className="font-heading text-primary text-lg tracking-wider">DEMO</p>
        </motion.div>

        <motion.h1
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="font-heading text-5xl md:text-7xl tracking-wider mb-4"
        >
          UPLOAD YOUR <span className="text-primary">EXERCISE</span>
        </motion.h1>

        <motion.p
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="text-muted-foreground text-lg mb-12 max-w-xl"
        >
          Record or upload a video of your squat, deadlift, or push-up and our AI will analyze your form.
        </motion.p>

        {/* Upload area */}
        <motion.div
          initial={{ y: 40, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.5 }}
        >
          {!selectedFile ? (
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-4 border-dashed rounded-md p-16 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${
                dragActive
                  ? "border-primary bg-primary/10"
                  : "border-border hover:border-primary/50 hover:bg-card/50"
              }`}
            >
              <motion.div
                animate={dragActive ? { scale: 1.1, y: -5 } : { scale: 1, y: 0 }}
                className="w-20 h-20 flex items-center justify-center bg-primary/10 border-2 border-primary rounded-md mb-6"
              >
                <Upload className="w-10 h-10 text-primary" />
              </motion.div>
              <p className="font-heading text-2xl md:text-3xl tracking-wider mb-2">
                DROP YOUR VIDEO HERE
              </p>
              <p className="text-muted-foreground text-sm">or click to browse • MP4, MOV, AVI</p>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleChange}
                className="hidden"
              />
            </div>
          ) : (
            <div className="border-4 border-primary bg-card rounded-md p-6 relative">
              <button
                onClick={clearFile}
                className="absolute top-4 right-4 z-10 w-10 h-10 flex items-center justify-center bg-background/80 hover:bg-destructive hover:text-destructive-foreground rounded-md transition-colors border border-border"
              >
                <X className="w-5 h-5" />
              </button>

              {(trimmedUrl || previewUrl) && (
                <video
                  src={trimmedUrl ?? previewUrl ?? undefined}
                  controls
                  ref={videoRef}
                  onLoadedMetadata={handleLoadedMetadata}
                  className="w-full rounded-md mb-6 max-h-[400px] object-contain bg-background"
                />
              )}

              {duration > 0 && (
                <div className="mb-6 border border-border rounded-md p-4 bg-background/60">
                  <div className="flex items-center justify-between text-sm text-muted-foreground mb-3">
                    <span>Trim start: {formatTime(trimStart)}</span>
                    <span>Trim end: {formatTime(trimEnd)}</span>
                    <span>Length: {formatTime(Math.max(0, trimEnd - trimStart))}</span>
                  </div>
                  <div className="relative h-8" style={{ ['--thumb-size' as string]: '18px' }}>
                    <div
                      className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full bg-border"
                      style={{ left: "calc(var(--thumb-size) / -2)", right: "calc(var(--thumb-size) / -2)" }}
                    />
                    <div
                      className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full bg-primary"
                      style={{
                        left: `calc(${(trimStart / duration) * 100}% - (var(--thumb-size) / 2))`,
                        right: `calc(${100 - (trimEnd / duration) * 100}% - (var(--thumb-size) / 2))`,
                      }}
                    />
                    <input
                      type="range"
                      min={0}
                      max={duration}
                      step={0.1}
                      value={trimStart}
                      onChange={(e) => {
                        const nextValue = Math.min(Number(e.target.value), trimEnd - 0.1);
                        setTrimStart(Math.max(0, nextValue));
                      }}
                      onMouseDown={() => setActiveThumb("start")}
                      onTouchStart={() => setActiveThumb("start")}
                      className={`trim-range absolute inset-0 bg-transparent appearance-none cursor-pointer accent-primary ${
                        activeThumb === "start" ? "z-20" : "z-10"
                      }`}
                      style={{
                        left: "calc(var(--thumb-size) / -2)",
                        width: "calc(100% + var(--thumb-size))",
                      }}
                    />
                    <input
                      type="range"
                      min={0}
                      max={duration}
                      step={0.1}
                      value={trimEnd}
                      onChange={(e) => {
                        const nextValue = Math.max(Number(e.target.value), trimStart + 0.1);
                        setTrimEnd(Math.min(duration, nextValue));
                      }}
                      onMouseDown={() => setActiveThumb("end")}
                      onTouchStart={() => setActiveThumb("end")}
                      className={`trim-range absolute inset-0 bg-transparent appearance-none cursor-pointer accent-primary ${
                        activeThumb === "end" ? "z-20" : "z-10"
                      }`}
                      style={{
                        left: "calc(var(--thumb-size) / -2)",
                        width: "calc(100% + var(--thumb-size))",
                      }}
                    />
                  </div>
                  <div className="mt-4 flex flex-col gap-3">
                    <div className="grid gap-3 md:grid-cols-2">
                      <motion.button
                        whileHover={{ scale: 1.03 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={handleTrim}
                        disabled={isTrimming || trimEnd - trimStart <= 0.1}
                        className="font-heading bg-primary text-primary-foreground py-3 text-lg uppercase tracking-wider rounded-md hover:bg-primary/90 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
                      >
                        {isTrimming ? "TRIMMING..." : "TRIM VIDEO"}
                      </motion.button>
                      <motion.button
                        whileHover={{ scale: 1.03 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={revertToOriginal}
                        disabled={!trimmedFile}
                        className="font-heading bg-secondary text-secondary-foreground py-3 text-lg uppercase tracking-wider rounded-md hover:bg-secondary/90 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
                      >
                        REVERT TO ORIGINAL
                      </motion.button>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Trimming runs locally with WebCodecs and may take a few seconds.
                    </p>
                  </div>
                </div>
              )}

              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 flex items-center justify-center bg-primary/10 border-2 border-primary rounded-md">
                  <Video className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <p className="font-heading text-xl tracking-wider">{selectedFile.name}</p>
                  <p className="text-muted-foreground text-sm">
                    {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
                  </p>
                </div>
              </div>

              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.98 }}
                className="w-full font-heading bg-secondary text-secondary-foreground py-4 text-2xl uppercase tracking-wider rounded-md hover:bg-secondary/90 transition-colors"
              >
                {trimmedFile ? "ANALYZE TRIMMED VIDEO" : "ANALYZE MY FORM"}
              </motion.button>
            </div>
          )}
        </motion.div>
      </div>

      {/* Bottom accent line */}
      <motion.div
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ duration: 1, delay: 1, ease: "easeOut" }}
        className="relative z-20 h-[2px] bg-gradient-to-r from-transparent via-primary to-transparent mx-6 md:mx-12 mb-8"
        style={{ transformOrigin: "center" }}
      />
    </main>
  );
};

export default Demo;

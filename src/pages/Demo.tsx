import { motion } from "framer-motion";
import { Upload, ArrowLeft, Video, X, Loader2 } from "lucide-react";
import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { uploadVideo } from "@/lib/api";

const Demo = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const [processing, setProcessing] = useState(false);

  const handleFile = (file: File) => {
    if (file.type.startsWith("video/")) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
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
  };

  const downscaleVideoTo480 = (file: File): Promise<File> => {
    return new Promise(async (resolve) => {
      if (!('MediaRecorder' in window)) return resolve(file);

      const url = URL.createObjectURL(file);
      const video = document.createElement('video');
      video.src = url;
      video.muted = true;
      video.playsInline = true;

      await new Promise((res) => {
        const onLoaded = () => {
          video.removeEventListener('loadedmetadata', onLoaded);
          res(true);
        };
        video.addEventListener('loadedmetadata', onLoaded);
      });

      const srcWidth = video.videoWidth || 640;
      const srcHeight = video.videoHeight || 360;
      const targetHeight = 480;
      const targetWidth = Math.round((srcWidth / srcHeight) * targetHeight);

      const canvas = document.createElement('canvas');
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) return resolve(file);

      const stream = (canvas as any).captureStream?.(30);
      if (!stream) return resolve(file);

      const options: any = {};
      let mime = 'video/webm';
      if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) mime = 'video/webm;codecs=vp9';
      else if (MediaRecorder.isTypeSupported('video/webm;codecs=vp8')) mime = 'video/webm;codecs=vp8';
      options.mimeType = mime;

      const recorder = new MediaRecorder(stream, options);
      const chunks: Blob[] = [];
      recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: mime });
        const newName = file.name.replace(/\.[^.]+$/, '') + '-480p.webm';
        const newFile = new File([blob], newName, { type: mime });
        URL.revokeObjectURL(url);
        resolve(newFile);
      };

      let raf: number | null = null;

      const draw = () => {
        try {
          ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
        } catch (e) {
          // drawing may fail if video not ready
        }
        if (!video.paused && !video.ended) raf = requestAnimationFrame(draw);
      };

      recorder.start(1000);
      video.currentTime = 0;
      video.play().catch(() => {
        // ignore play errors
      });
      raf = requestAnimationFrame(draw);

      video.addEventListener('ended', () => {
        if (raf) cancelAnimationFrame(raf);
        recorder.stop();
      });

      // safety: stop after duration if something hangs
      setTimeout(() => {
        if (recorder.state !== 'inactive') {
          try { recorder.stop(); } catch {}
        }
      }, (file.size / (1024 * 1024)) * 2000 + 20000); // heuristic timeout
    });
  };

  const analyzeFile = async () => {
    if (!selectedFile) return;
    setProcessing(true);
    try {
      // Downscale to 480p for the backend (smaller upload, faster processing)
      let fileToUpload: File;
      try {
        fileToUpload = await downscaleVideoTo480(selectedFile);
      } catch {
        // Fall back to original if downscale fails
        fileToUpload = selectedFile;
      }

      const { task_id } = await uploadVideo(fileToUpload);

      // Navigate with the ORIGINAL uncompressed video URL so the
      // Results page draws pose markers on the full-quality video
      navigate(`/results/${task_id}`, {
        state: { videoUrl: previewUrl },
      });
    } catch (err: any) {
      const msg = err?.message || "Upload failed";
      toast.error(msg);
      setProcessing(false);
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
          Upload a rep-based gym exercise (squat, deadlift, push-up, etc.). Sports and non-rep videos are not supported.
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

              {previewUrl && (
                <video
                  src={previewUrl}
                  controls
                  className="w-full rounded-md mb-6 max-h-[400px] object-contain bg-background"
                />
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
                onClick={analyzeFile}
                disabled={processing}
                className="w-full font-heading bg-secondary text-secondary-foreground py-4 text-2xl uppercase tracking-wider rounded-md hover:bg-secondary/90 transition-colors disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-3"
              >
                {processing ? (
                  <><Loader2 className="w-6 h-6 animate-spin" /> UPLOADING…</>
                ) : (
                  "ANALYZE MY FORM"
                )}
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

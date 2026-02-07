import heroImage from "@/assets/hero-exercise.jpg";
import { ArrowRight } from "lucide-react";

const Hero = () => {
  return (
    <section className="relative min-h-screen flex items-end overflow-hidden">
      {/* Background image */}
      <div className="absolute inset-0">
        <img
          src={heroImage}
          alt="Person performing a squat with proper form"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-background via-background/80 to-background/40" />
      </div>

      {/* Nav */}
      <nav className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-6 md:px-12 py-6">
        <span className="font-heading text-2xl tracking-tight">FormGuard</span>
        <span className="text-sm text-muted-foreground tracking-widest uppercase hidden md:block">
          AI Movement Analysis
        </span>
      </nav>

      {/* Hero content */}
      <div className="relative z-10 w-full max-w-6xl mx-auto px-6 md:px-12 pb-20 pt-40">
        <p className="text-primary font-medium tracking-widest uppercase text-sm mb-4 animate-fade-up">
          Exercise Safety
        </p>
        <h1 className="font-heading text-5xl sm:text-7xl md:text-8xl lg:text-9xl leading-[0.9] mb-6 animate-fade-up" style={{ animationDelay: "0.1s" }}>
          MOVE.<br />TRAIN.<br />SAFELY.
        </h1>
        <p className="text-muted-foreground text-lg md:text-xl max-w-md mb-10 animate-fade-up" style={{ animationDelay: "0.2s" }}>
          AI-powered form analysis that detects injury risk and helps you train with confidence.
        </p>
        <div className="flex items-center gap-4 animate-fade-up" style={{ animationDelay: "0.3s" }}>
          <button className="bg-primary text-primary-foreground px-8 py-4 rounded-lg font-medium text-sm tracking-wide uppercase hover:opacity-90 transition-opacity flex items-center gap-3">
            Upload Video
            <ArrowRight className="w-4 h-4" />
          </button>
          <button className="border border-border text-foreground px-8 py-4 rounded-lg font-medium text-sm tracking-wide uppercase hover:border-primary/40 transition-colors">
            Try Demo
          </button>
        </div>
      </div>
    </section>
  );
};

export default Hero;

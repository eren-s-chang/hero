import heroImage from "@/assets/hero-exercise.jpg";
import { ArrowRight } from "lucide-react";

const Hero = () => {
  return (
    <section className="relative min-h-screen flex flex-col overflow-hidden bg-background">
      {/* Nav */}
      <nav className="relative z-30 flex items-center justify-between px-6 md:px-12 py-6">
        <span className="font-heading text-2xl tracking-tight">FormGuard</span>
        <span className="text-sm text-muted-foreground tracking-widest uppercase hidden md:block">
          AI Movement Analysis
        </span>
      </nav>

      {/* Main hero area */}
      <div className="relative flex-1 flex items-center justify-center">
        {/* Large background text — sits BEHIND the image */}
        <div className="absolute inset-0 z-0 flex items-center justify-center pointer-events-none select-none">
          <h1 className="font-heading text-[12vw] md:text-[10vw] lg:text-[9vw] leading-[0.95] tracking-tight text-foreground/10 text-center whitespace-nowrap">
            MOVE. TRAIN.<br />SAFELY.
          </h1>
        </div>

        {/* Hero image — sits ON TOP of the text */}
        <div className="relative z-10 w-[90%] md:w-[70%] lg:w-[60%] aspect-[16/10] rounded-2xl overflow-hidden shadow-2xl shadow-black/40 mx-auto">
          <img
            src={heroImage}
            alt="Person performing a squat with proper form"
            className="w-full h-full object-cover"
          />
          {/* Subtle bottom gradient on the image */}
          <div className="absolute inset-0 bg-gradient-to-t from-background/60 via-transparent to-transparent" />
        </div>

        {/* Foreground headline that overlaps over the image bottom — sits ON TOP */}
        <div className="absolute bottom-0 left-0 right-0 z-20 px-6 md:px-12 pb-12">
          <div className="max-w-6xl mx-auto">
            <h2 className="font-heading text-5xl sm:text-7xl md:text-8xl lg:text-9xl leading-[0.85] mb-0 animate-fade-up mix-blend-difference">
              MOVE. TRAIN.<br />SAFELY.
            </h2>
          </div>
        </div>
      </div>

      {/* Bottom bar with subtitle and CTAs */}
      <div className="relative z-20 px-6 md:px-12 pb-10 pt-6">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row md:items-end justify-between gap-8">
          <div>
            <p className="text-primary font-medium tracking-widest uppercase text-sm mb-3 animate-fade-up">
              Exercise Safety
            </p>
            <p className="text-muted-foreground text-lg md:text-xl max-w-md animate-fade-up" style={{ animationDelay: "0.1s" }}>
              AI-powered form analysis that detects injury risk and helps you train with confidence.
            </p>
          </div>
          <div className="flex items-center gap-4 animate-fade-up" style={{ animationDelay: "0.2s" }}>
            <button className="bg-primary text-primary-foreground px-8 py-4 rounded-lg font-medium text-sm tracking-wide uppercase hover:opacity-90 transition-opacity flex items-center gap-3">
              Upload Video
              <ArrowRight className="w-4 h-4" />
            </button>
            <button className="border border-border text-foreground px-8 py-4 rounded-lg font-medium text-sm tracking-wide uppercase hover:border-primary/40 transition-colors">
              Try Demo
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;

import { motion } from "framer-motion";
import heroImage from "@/assets/hero-exercise.jpg";
import { ArrowRight } from "lucide-react";

const Hero = () => {
  return (
    <section className="relative min-h-screen flex flex-col overflow-hidden bg-background">
      {/* Full-bleed hero image */}
      <div className="absolute inset-0 z-0">
        <img
          src={heroImage}
          alt="Saitama in serious mode"
          className="w-full h-full object-cover object-right"
        />
        {/* Gradient overlay to blend with background */}
        <div className="absolute inset-0 hero-gradient" />
        <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-background/60" />
      </div>

      {/* Nav */}
      <motion.nav 
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="relative z-30 flex items-center justify-between px-6 md:px-12 py-8"
      >
        <span className="font-heading text-3xl md:text-4xl tracking-widest text-foreground">
          FORMGUARD
        </span>
        <span className="text-xs text-muted-foreground tracking-[0.3em] uppercase hidden md:block font-medium">
          AI Movement Analysis
        </span>
      </motion.nav>

      {/* Main hero content - left aligned */}
      <div className="relative z-20 flex-1 flex items-center px-6 md:px-12">
        <div className="max-w-2xl">
          {/* Main headline */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            <motion.h1 
              initial={{ y: 40, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ duration: 0.7, delay: 0.4 }}
              className="font-heading text-6xl sm:text-7xl md:text-8xl lg:text-9xl leading-[0.9] tracking-wider text-foreground mb-6"
            >
              <span className="block">ONE</span>
              <span className="block text-primary text-glow">PERFECT</span>
              <span className="block">FORM.</span>
            </motion.h1>
          </motion.div>

          {/* Subtitle */}
          <motion.p 
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.8 }}
            className="text-muted-foreground text-base md:text-lg max-w-md mb-10 font-medium leading-relaxed"
          >
            AI-powered biomechanical analysis. Detect flaws. Prevent injuries. 
            Master your movement with surgical precision.
          </motion.p>

          {/* CTA */}
          <motion.div 
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 1 }}
          >
            <motion.button 
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="bg-primary text-primary-foreground px-8 py-4 font-heading text-xl tracking-widest uppercase flex items-center gap-4 hover:bg-primary/90 transition-colors"
            >
              Start Analysis
              <ArrowRight className="w-5 h-5" />
            </motion.button>
          </motion.div>
        </div>
      </div>

      {/* Bottom accent line */}
      <motion.div 
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ duration: 1, delay: 1.2, ease: "easeOut" }}
        className="relative z-20 h-[2px] bg-gradient-to-r from-primary via-primary/50 to-transparent mx-6 md:mx-12 mb-8 origin-left"
      />
    </section>
  );
};

export default Hero;

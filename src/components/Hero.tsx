import { motion } from "framer-motion";
import heroImage from "@/assets/hero-exercise.jpg";
import { ArrowRight } from "lucide-react";

const Hero = () => {
  const letterVariants = {
    hidden: { y: 100, opacity: 0, rotateX: -90 },
    visible: (i: number) => ({
      y: 0,
      opacity: 1,
      rotateX: 0,
      transition: {
        delay: 0.5 + i * 0.08,
        duration: 0.6,
        ease: [0.22, 1, 0.36, 1] as const
      }
    })
  };

  const glowPulse = {
    initial: { textShadow: "0 0 20px hsl(0 70% 50% / 0.3)" },
    animate: {
      textShadow: [
        "0 0 20px hsl(0 70% 50% / 0.3)",
        "0 0 40px hsl(0 70% 50% / 0.6)",
        "0 0 20px hsl(0 70% 50% / 0.3)"
      ],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: "easeInOut" as const
      }
    }
  };

  const titleText = "HERO";

  return (
    <section className="relative min-h-[90vh] flex flex-col overflow-hidden bg-background">
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

      {/* Animated background particles */}
      <div className="absolute inset-0 z-10 overflow-hidden pointer-events-none">
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-primary/40 rounded-full"
            initial={{ 
              x: Math.random() * 100 + "%", 
              y: "100%",
              opacity: 0 
            }}
            animate={{ 
              y: "-10%",
              opacity: [0, 1, 0]
            }}
            transition={{
              duration: 4 + Math.random() * 2,
              repeat: Infinity,
              delay: i * 0.8,
              ease: "linear"
            }}
          />
        ))}
      </div>

      {/* Nav */}
      <motion.nav 
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="relative z-30 flex items-center justify-between px-6 md:px-12 py-8"
      >
        <span className="font-heading text-3xl md:text-4xl tracking-widest text-foreground">
          HERO
        </span>
        <motion.span 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="text-xs text-muted-foreground tracking-[0.3em] uppercase hidden md:block font-medium"
        >
          Movement Analysis
        </motion.span>
      </motion.nav>

      {/* Main hero content - left aligned */}
      <div className="relative z-20 flex-1 flex items-center px-6 md:px-12">
        <div className="max-w-3xl">
          {/* Main headline with One Punch Man style */}
          <div className="mb-8 perspective-1000">
            <div className="flex flex-wrap">
              {titleText.split("").map((letter, i) => (
                <motion.span
                  key={i}
                  custom={i}
                  variants={letterVariants}
                  initial="hidden"
                  animate="visible"
                  className="opm-title text-8xl sm:text-9xl md:text-[12rem] lg:text-[14rem] leading-none"
                >
                  {letter}
                </motion.span>
              ))}
            </div>
          </div>

          {/* Subtitle with staggered animation */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 1.2 }}
          >
            <motion.h2 
              variants={glowPulse}
              initial="initial"
              animate="animate"
              className="font-heading text-4xl sm:text-5xl md:text-6xl tracking-wider text-primary mb-8"
            >
              ONE PERFECT FORM.
            </motion.h2>
          </motion.div>

          {/* CTA */}
          <motion.div 
            initial={{ y: 30, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 1.5 }}
          >
            <motion.button 
              whileHover={{ scale: 1.02, boxShadow: "0 0 30px hsl(48 80% 55% / 0.4)" }}
              whileTap={{ scale: 0.98 }}
              className="bg-primary text-primary-foreground px-8 py-4 font-heading text-xl tracking-widest uppercase flex items-center gap-4 hover:bg-primary/90 transition-colors"
            >
              Start Analysis
              <motion.span
                animate={{ x: [0, 5, 0] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              >
                <ArrowRight className="w-5 h-5" />
              </motion.span>
            </motion.button>
          </motion.div>
        </div>
      </div>

      {/* Bottom accent line */}
      <motion.div 
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ duration: 1, delay: 1.8, ease: "easeOut" }}
        className="relative z-20 h-[2px] bg-gradient-to-r from-primary via-primary/50 to-transparent mx-6 md:mx-12 mb-8 origin-left"
      />

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2 }}
        className="absolute bottom-12 left-1/2 -translate-x-1/2 z-20 flex flex-col items-center gap-2"
      >
        <span className="text-xs text-muted-foreground tracking-widest uppercase">Scroll</span>
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="w-[1px] h-8 bg-gradient-to-b from-primary to-transparent"
        />
      </motion.div>
    </section>
  );
};

export default Hero;

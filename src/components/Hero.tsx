import { motion } from "framer-motion";
import heroImage from "@/assets/hero-exercise.jpg";
import { ArrowRight, Zap } from "lucide-react";

const Hero = () => {
  // Floating animation for decorative elements
  const floatAnimation = {
    y: [0, -10, 0],
    transition: {
      duration: 3,
      repeat: Infinity,
      ease: "easeInOut" as const
    }
  };

  // Pulse glow animation
  const pulseGlow = {
    opacity: [0.3, 0.6, 0.3],
    scale: [1, 1.05, 1],
    transition: {
      duration: 2,
      repeat: Infinity,
      ease: "easeInOut" as const
    }
  };

  return (
    <section className="relative min-h-screen flex flex-col overflow-hidden bg-background">
      {/* Animated background glow */}
      <motion.div 
        animate={pulseGlow}
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full bg-primary/10 blur-3xl pointer-events-none"
      />
      <motion.div 
        animate={{
          ...pulseGlow,
          transition: { ...pulseGlow.transition, delay: 1 }
        }}
        className="absolute top-1/3 right-1/4 w-[400px] h-[400px] rounded-full bg-secondary/10 blur-3xl pointer-events-none"
      />
      
      {/* Speed lines */}
      <div className="absolute inset-0 speed-lines pointer-events-none" />
      
      {/* Nav */}
      <motion.nav 
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="relative z-30 flex items-center justify-between px-6 md:px-12 py-6"
      >
        <motion.span 
          whileHover={{ scale: 1.05 }}
          className="font-heading text-3xl tracking-wide text-foreground flex items-center gap-2 cursor-pointer"
        >
          <motion.div animate={floatAnimation}>
            <Zap className="w-8 h-8 text-primary fill-primary" />
          </motion.div>
          FormGuard
        </motion.span>
        <motion.span 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="text-sm text-muted-foreground tracking-widest uppercase hidden md:block font-bold"
        >
          AI Movement Analysis
        </motion.span>
      </motion.nav>

      {/* Main hero area */}
      <div className="relative flex-1 flex items-center justify-center py-8">
        {/* Large background text — sits BEHIND the image */}
        <motion.div 
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 1, delay: 0.3, ease: "easeOut" }}
          className="absolute inset-0 z-0 flex items-center justify-center pointer-events-none select-none"
        >
          <motion.h1 
            animate={{ 
              textShadow: [
                "2px 2px 0px hsl(345 50% 35%)",
                "4px 4px 0px hsl(345 50% 35%)",
                "2px 2px 0px hsl(345 50% 35%)"
              ]
            }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            className="font-heading text-[14vw] md:text-[12vw] lg:text-[10vw] leading-[0.9] tracking-wider text-primary/15 text-center whitespace-nowrap"
          >
            PERFECT<br />FORM
          </motion.h1>
        </motion.div>

        {/* Hero image — sits ON TOP of the text */}
        <motion.div 
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ 
            duration: 0.8, 
            delay: 0.5, 
            type: "spring",
            stiffness: 100
          }}
          className="relative z-10 w-[95%] md:w-[75%] lg:w-[60%] aspect-[16/9] rounded-2xl overflow-hidden mx-auto"
          style={{
            boxShadow: "0 0 60px hsl(50 35% 45% / 0.15), 0 0 100px hsl(345 50% 35% / 0.1)"
          }}
        >
          {/* Animated border glow */}
          <motion.div 
            animate={{
              opacity: [0.5, 1, 0.5],
            }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            className="absolute inset-0 rounded-2xl border border-primary/40 pointer-events-none z-20"
          />
          <img
            src={heroImage}
            alt="Athletic silhouette in dramatic lighting"
            className="w-full h-full object-cover"
          />
          {/* Vignette overlay */}
          <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-background/50 pointer-events-none" />
        </motion.div>

        {/* Foreground headline that overlaps over the image bottom — sits ON TOP */}
        <div className="absolute bottom-0 left-0 right-0 z-20 px-6 md:px-12 pb-8">
          <div className="max-w-6xl mx-auto">
            <motion.h2 
              initial={{ y: 80, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ duration: 0.7, delay: 0.8, type: "spring", stiffness: 80 }}
              className="font-heading text-4xl sm:text-6xl md:text-7xl lg:text-8xl leading-[0.9] mb-0 text-foreground"
            >
              <motion.span
                initial={{ opacity: 0, x: -30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 1, duration: 0.5 }}
                className="block"
              >
                PRECISION
              </motion.span>
              <motion.span
                initial={{ opacity: 0, x: -30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 1.2, duration: 0.5 }}
                className="block text-primary"
              >
                IN EVERY
              </motion.span>
              <motion.span
                initial={{ opacity: 0, x: -30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 1.4, duration: 0.5 }}
                className="block"
              >
                MOVEMENT.
              </motion.span>
            </motion.h2>
          </div>
        </div>
      </div>

      {/* Bottom bar with CTA */}
      <motion.div 
        initial={{ y: 50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, delay: 1.6 }}
        className="relative z-20 px-6 md:px-12 pb-10 pt-6"
      >
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row md:items-end justify-between gap-8">
          <motion.p 
            initial={{ x: -30, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 1.8 }}
            className="text-muted-foreground text-lg md:text-xl max-w-md font-semibold"
          >
            AI-powered biomechanical analysis that identifies risk and optimizes your training form.
          </motion.p>
          <motion.div 
            initial={{ x: 30, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 2 }}
            className="flex items-center gap-4"
          >
            <motion.button 
              whileHover={{ 
                scale: 1.05,
                boxShadow: "0 0 30px hsl(345 50% 35% / 0.6), 6px 6px 0px hsl(0 0% 0%)"
              }}
              whileTap={{ scale: 0.95 }}
              className="bg-secondary text-secondary-foreground px-8 py-4 rounded-xl font-heading text-xl tracking-wide uppercase flex items-center gap-3 border border-secondary-foreground/20"
              style={{
                boxShadow: "0 0 20px hsl(345 50% 35% / 0.4), 4px 4px 0px hsl(0 0% 0%)"
              }}
            >
              Try Demo
              <motion.div
                animate={{ x: [0, 5, 0] }}
                transition={{ duration: 1, repeat: Infinity, ease: "easeInOut" }}
              >
                <ArrowRight className="w-5 h-5" />
              </motion.div>
            </motion.button>
          </motion.div>
        </div>
      </motion.div>
    </section>
  );
};

export default Hero;

import { motion } from "framer-motion";
import heroImage from "@/assets/hero-exercise.jpg";
import { ArrowRight, Zap } from "lucide-react";

const Hero = () => {
  return (
    <section className="relative min-h-screen flex flex-col overflow-hidden bg-background speed-lines">
      {/* Animated burst background */}
      <div className="absolute inset-0 action-burst pointer-events-none" />
      
      {/* Nav */}
      <motion.nav 
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="relative z-30 flex items-center justify-between px-6 md:px-12 py-6"
      >
        <span className="font-heading text-3xl tracking-wide text-foreground flex items-center gap-2">
          <Zap className="w-8 h-8 text-secondary fill-secondary" />
          FormGuard
        </span>
        <span className="text-sm text-muted-foreground tracking-widest uppercase hidden md:block font-bold">
          AI Movement Analysis
        </span>
      </motion.nav>

      {/* Main hero area */}
      <div className="relative flex-1 flex items-center justify-center py-8">
        {/* Large background text — sits BEHIND the image */}
        <motion.div 
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
          className="absolute inset-0 z-0 flex items-center justify-center pointer-events-none select-none"
        >
          <h1 className="font-heading text-[14vw] md:text-[12vw] lg:text-[10vw] leading-[0.9] tracking-wider text-primary/20 text-center whitespace-nowrap comic-shadow">
            ONE PUNCH<br />FORM!
          </h1>
        </motion.div>

        {/* Hero image — sits ON TOP of the text */}
        <motion.div 
          initial={{ scale: 0.5, opacity: 0, rotate: -5 }}
          animate={{ scale: 1, opacity: 1, rotate: 0 }}
          transition={{ 
            duration: 0.6, 
            delay: 0.4, 
            type: "spring",
            stiffness: 200
          }}
          className="relative z-10 w-[90%] md:w-[70%] lg:w-[55%] aspect-[16/9] rounded-3xl overflow-hidden shadow-2xl mx-auto border-4 border-foreground"
          style={{
            boxShadow: "8px 8px 0px hsl(0 85% 55%), 16px 16px 0px hsl(0 0% 0% / 0.3)"
          }}
        >
          <img
            src={heroImage}
            alt="Anime-style person performing a powerful squat"
            className="w-full h-full object-cover"
          />
        </motion.div>

        {/* Foreground headline that overlaps over the image bottom — sits ON TOP */}
        <div className="absolute bottom-0 left-0 right-0 z-20 px-6 md:px-12 pb-8">
          <div className="max-w-6xl mx-auto">
            <motion.h2 
              initial={{ y: 100, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.6, type: "spring", stiffness: 100 }}
              className="font-heading text-5xl sm:text-7xl md:text-8xl lg:text-9xl leading-[0.85] mb-0 text-foreground comic-shadow"
            >
              TRAIN LIKE<br />A HERO!
            </motion.h2>
          </div>
        </div>
      </div>

      {/* Bottom bar with CTA */}
      <motion.div 
        initial={{ y: 50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.8 }}
        className="relative z-20 px-6 md:px-12 pb-10 pt-6"
      >
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row md:items-end justify-between gap-8">
          <motion.p 
            initial={{ x: -30, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 1 }}
            className="text-muted-foreground text-lg md:text-xl max-w-md font-semibold"
          >
            AI-powered form analysis that detects injury risk and helps you train with the power of a true hero.
          </motion.p>
          <motion.div 
            initial={{ x: 30, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 1.1 }}
            className="flex items-center gap-4"
          >
            <motion.button 
              whileHover={{ scale: 1.05, rotate: -2 }}
              whileTap={{ scale: 0.95 }}
              className="bg-secondary text-secondary-foreground px-8 py-4 rounded-xl font-heading text-xl tracking-wide uppercase flex items-center gap-3 border-4 border-foreground"
              style={{
                boxShadow: "4px 4px 0px hsl(0 0% 0%)"
              }}
            >
              Try Demo
              <ArrowRight className="w-5 h-5" />
            </motion.button>
          </motion.div>
        </div>
      </motion.div>
    </section>
  );
};

export default Hero;

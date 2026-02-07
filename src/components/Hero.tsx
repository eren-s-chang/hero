import { motion } from "framer-motion";
import heroImage from "@/assets/hero-exercise.jpg";
import { ArrowRight } from "lucide-react";

const Hero = () => {
  const wordVariants = {
    hidden: { y: 60, opacity: 0 },
    visible: (i: number) => ({
      y: 0,
      opacity: 1,
      transition: {
        delay: 0.6 + i * 0.15,
        duration: 0.6,
        ease: [0.22, 1, 0.36, 1] as const
      }
    })
  };

  const words = ["ONE", "PERFECT", "FORM."];

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
        <span className="opm-title text-4xl md:text-5xl">
          HERO
        </span>
      </motion.nav>

      {/* Main hero content - left aligned */}
      <div className="relative z-20 flex-1 flex items-center px-6 md:px-12">
        <div className="max-w-2xl">
          {/* Main headline */}
          <div className="mb-10 overflow-hidden">
            {words.map((word, i) => (
              <motion.div
                key={i}
                custom={i}
                variants={wordVariants}
                initial="hidden"
                animate="visible"
                className="overflow-hidden"
              >
                <span className={`font-heading text-6xl sm:text-7xl md:text-8xl lg:text-9xl leading-[0.95] tracking-wider block ${
                  i === 1 ? "text-primary text-glow" : "text-foreground"
                }`}>
                  {word}
                </span>
              </motion.div>
            ))}
          </div>

          {/* CTA */}
          <motion.div 
            initial={{ y: 30, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 1.2 }}
          >
            <motion.button 
              whileHover={{ scale: 1.02, boxShadow: "0 0 30px hsl(48 80% 55% / 0.4)" }}
              whileTap={{ scale: 0.98 }}
              className="bg-primary text-primary-foreground px-10 py-5 font-heading text-2xl tracking-widest uppercase flex items-center gap-4 hover:bg-primary/90 transition-colors"
            >
              Try Demo
              <motion.span
                animate={{ x: [0, 5, 0] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              >
                <ArrowRight className="w-6 h-6" />
              </motion.span>
            </motion.button>
          </motion.div>
        </div>
      </div>

      {/* Bottom accent line */}
      <motion.div 
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ duration: 1, delay: 1.5, ease: "easeOut" }}
        className="relative z-20 h-[2px] bg-gradient-to-r from-transparent via-primary to-transparent mx-6 md:mx-12 mb-8"
        style={{ transformOrigin: "center" }}
      />
    </section>
  );
};

export default Hero;
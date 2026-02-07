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

      {/* Nav - bigger HERO title */}
      <motion.nav 
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="relative z-30 flex items-center justify-between px-6 md:px-12 py-8"
      >
        <span className="opm-title text-8xl md:text-9xl lg:text-[10rem]">
          HERO
        </span>
      </motion.nav>

      {/* Spacer to push content down */}
      <div className="flex-1" />

      {/* Main hero content - bottom left aligned */}
      <div className="relative z-20 px-6 md:px-12 pb-16">
        <div className="max-w-2xl">
          {/* Main headline - horizontal */}
          <motion.div 
            initial={{ y: 40, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="mb-8"
          >
            <h1 className="font-heading text-3xl sm:text-4xl md:text-5xl lg:text-6xl tracking-wider uppercase">
              <span className="text-foreground">Your </span>
              <span className="text-primary">AI </span>
              <span className="text-foreground">Physical Trainer.</span>
            </h1>
          </motion.div>

          {/* CTA */}
          <motion.div 
            initial={{ y: 30, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.9 }}
          >
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
              className="font-heading bg-secondary text-secondary-foreground px-8 py-4 text-2xl sm:text-3xl uppercase flex items-center gap-3 hover:bg-secondary/90 transition-colors rounded-md tracking-wider"
            >
              TRY DEMO
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
        transition={{ duration: 1, delay: 1.5, ease: "easeOut" }}
        className="relative z-20 h-[2px] bg-gradient-to-r from-transparent via-primary to-transparent mx-6 md:mx-12 mb-8"
        style={{ transformOrigin: "center" }}
      />
    </section>
  );
};

export default Hero;
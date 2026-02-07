import { motion } from "framer-motion";
import { Upload, ScanSearch, ShieldCheck, Zap } from "lucide-react";

const HowItWorks = () => {
  const steps = [
    {
      icon: Upload,
      title: "Upload Video",
      description: "Record or upload a video of your exercise — squat, deadlift, or push-up.",
    },
    {
      icon: ScanSearch,
      title: "Analyze Form",
      description: "Our AI analyzes every frame to detect alignment, depth, and movement patterns.",
    },
    {
      icon: ShieldCheck,
      title: "Get Feedback",
      description: "Receive clear, actionable corrections to reduce injury risk and boost performance.",
    },
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2
      }
    }
  };

  const cardVariants = {
    hidden: { y: 50, opacity: 0, rotate: -3 },
    visible: { 
      y: 0, 
      opacity: 1, 
      rotate: 0,
      transition: {
        type: "spring" as const,
        stiffness: 200,
        damping: 20
      }
    }
  };

  return (
    <section className="py-24 px-6 bg-primary/10 speed-lines relative overflow-hidden">
      {/* Decorative elements */}
      <motion.div 
        animate={{ rotate: 360 }}
        transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
        className="absolute top-10 right-10 text-primary/20"
      >
        <Zap className="w-32 h-32" />
      </motion.div>
      
      <div className="max-w-6xl mx-auto relative z-10">
        <motion.div
          initial={{ x: -50, opacity: 0 }}
          whileInView={{ x: 0, opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <p className="text-secondary font-bold tracking-widest uppercase text-sm mb-3">How It Works</p>
          <h2 className="font-heading text-5xl md:text-6xl mb-16 comic-shadow">
            Three Steps to<br />ULTIMATE POWER!
          </h2>
        </motion.div>
        
        <motion.div 
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid md:grid-cols-3 gap-8"
        >
          {steps.map((step, i) => (
            <motion.div
              key={step.title}
              variants={cardVariants}
              whileHover={{ scale: 1.05, rotate: 2 }}
              className="bg-card rounded-2xl p-8 border-4 border-foreground cursor-pointer"
              style={{
                boxShadow: "6px 6px 0px hsl(0 0% 0%)"
              }}
            >
              <div className="flex items-center justify-between mb-6">
                <motion.div 
                  whileHover={{ rotate: 15, scale: 1.2 }}
                  className="w-14 h-14 rounded-xl bg-secondary flex items-center justify-center text-secondary-foreground border-2 border-foreground"
                >
                  <step.icon className="w-7 h-7" />
                </motion.div>
                <span className="text-primary font-heading text-4xl">0{i + 1}</span>
              </div>
              <h3 className="font-heading text-2xl mb-3 tracking-wide">{step.title}</h3>
              <p className="text-muted-foreground leading-relaxed font-semibold">{step.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default HowItWorks;

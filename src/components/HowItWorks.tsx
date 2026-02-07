import { motion } from "framer-motion";
import { Upload, ScanSearch, ShieldCheck } from "lucide-react";

const HowItWorks = () => {
  const steps = [
    {
      icon: Upload,
      title: "UPLOAD",
      description: "Record or upload a video of your exercise — squat, deadlift, or push-up.",
    },
    {
      icon: ScanSearch,
      title: "ANALYZE",
      description: "Our AI analyzes every frame to detect alignment, depth, and movement patterns.",
    },
    {
      icon: ShieldCheck,
      title: "OPTIMIZE",
      description: "Receive precise corrections to eliminate injury risk and maximize performance.",
    },
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15
      }
    }
  };

  const cardVariants = {
    hidden: { y: 30, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: {
        duration: 0.5,
        ease: "easeOut" as const
      }
    }
  };

  return (
    <section className="py-24 px-6 md:px-12 bg-background relative">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mb-16"
        >
          <p className="font-modern text-primary text-sm tracking-[0.3em] mb-4 font-semibold">PROCESS</p>
          <h2 className="font-modern text-5xl md:text-6xl text-foreground tracking-wider font-bold">
            HOW IT WORKS
          </h2>
        </motion.div>
        
        <motion.div 
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid md:grid-cols-3 gap-6"
        >
          {steps.map((step, i) => (
            <motion.div
              key={step.title}
              variants={cardVariants}
              whileHover={{ y: -8, scale: 1.02 }}
              className="bg-card p-8 border-4 border-black hover:border-primary transition-colors relative"
            >
              <div className="flex items-center justify-between mb-8">
                <div className="w-14 h-14 flex items-center justify-center text-primary bg-primary/10 border-2 border-primary">
                  <step.icon className="w-7 h-7" strokeWidth={2} />
                </div>
                <span className="font-modern text-muted-foreground/40 text-5xl font-bold">0{i + 1}</span>
              </div>
              <h3 className="font-modern text-2xl text-foreground mb-4 tracking-wider font-bold">{step.title}</h3>
              <p className="text-muted-foreground text-sm leading-relaxed font-modern font-medium">{step.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default HowItWorks;

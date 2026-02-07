import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";

const exercises = [
  {
    name: "SQUAT",
    description: "Knee tracking, depth, back angle, and weight distribution analysis.",
    tag: "Lower Body",
  },
  {
    name: "DEADLIFT",
    description: "Spinal alignment, hip hinge mechanics, and lockout form detection.",
    tag: "Posterior Chain",
  },
  {
    name: "PUSH-UP",
    description: "Elbow flare, core sag, and range of motion evaluation.",
    tag: "Upper Body",
  },
];

const Exercises = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { x: -20, opacity: 0 },
    visible: { 
      x: 0, 
      opacity: 1,
      transition: {
        duration: 0.4,
        ease: "easeOut" as const
      }
    }
  };

  return (
    <section className="py-24 px-6 md:px-12 bg-muted/30 relative">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mb-16"
        >
          <p className="text-primary font-medium tracking-[0.3em] uppercase text-xs mb-4">Exercises</p>
          <h2 className="font-heading text-5xl md:text-6xl tracking-wider">
            SUPPORTED MOVEMENTS
          </h2>
        </motion.div>
        
        <motion.div 
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="space-y-4"
        >
          {exercises.map((ex) => (
            <motion.div
              key={ex.name}
              variants={itemVariants}
              whileHover={{ x: 8 }}
              className="flex flex-col md:flex-row md:items-center justify-between bg-card p-6 md:p-8 border border-border hover:border-primary/40 transition-all cursor-pointer group"
            >
              <div className="flex items-center gap-6 mb-4 md:mb-0">
                <h3 className="font-heading text-3xl md:text-4xl tracking-wider">{ex.name}</h3>
                <span className="text-[10px] tracking-[0.2em] uppercase text-primary bg-primary/10 px-3 py-1.5 font-medium">
                  {ex.tag}
                </span>
              </div>
              <div className="flex items-center gap-6">
                <p className="text-muted-foreground text-sm max-w-xs">{ex.description}</p>
                <ArrowRight className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors shrink-0" />
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default Exercises;
